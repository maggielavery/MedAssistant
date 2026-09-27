"""API and web views."""
from __future__ import annotations

import logging

from django.db import transaction
from django.db.models import Max
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from rest_framework import status as http_status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import llm, retrieval, summarizer, workflow
from .models import (
    ActionEvent,
    ChatSession,
    Flag,
    Message,
    Patient,
    PatientStatus,
)
from .serializers import (
    ActionEventSerializer,
    FlagSerializer,
    MessageSerializer,
    PatientSerializer,
    PatientStatusSerializer,
    SessionListSerializer,
)

def greeting(patient: Patient) -> str:
    """Opening message; the patient's name comes from profile_data."""
    first_name = patient.display_name.split()[0] if patient.display_name else "there"
    return (
        f"Hi {first_name}, I'm Maggie, your health check-in assistant. How "
        "are you feeling today? You can tell me about any symptoms, ask a "
        "question, request an appointment, or say \"check-in\" to go through "
        "your routine review."
    )

logger = logging.getLogger(__name__)


# ===========================================================================
# Patient-facing API
# ===========================================================================
@api_view(["GET"])
def session_list(request, patient_id: int):
    patient = get_object_or_404(Patient, pk=patient_id)
    sessions = patient.sessions.all()
    return Response(SessionListSerializer(sessions, many=True).data)


@api_view(["POST"])
def session_new(request, patient_id: int):
    """Start a new writable session; end + summarize any previous writable one."""
    patient = get_object_or_404(Patient, pk=patient_id)
    text = greeting(patient)
    try:
        # Embed first so an unavailable model changes nothing.
        greeting_embedding = retrieval.embed_message(text)
    except llm.LLMUnavailable as exc:
        return Response(
            {"detail": f"The assistant is unavailable right now. ({exc})"},
            status=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    with transaction.atomic():
        previous = list(patient.sessions.filter(ended_at__isnull=True))
        for prev in previous:
            prev.ended_at = timezone.now()
            prev.save(update_fields=["ended_at", "updated_at"])
        session = ChatSession.objects.create(
            patient=patient, workflow_type="open_conversation"
        )
        # Assistant greeting for the new session.
        Message.objects.create(
            session=session,
            role="assistant",
            content=text,
            message_embedding=greeting_embedding,
            sequence_number=1,
        )

    # Summarize the sessions just ended; on failure summarize_idle_sessions
    # retries later.
    for prev in previous:
        try:
            summarizer.summarize_session(prev, reason="new_chat")
        except llm.LLMUnavailable:
            logger.warning("Summary for session %s deferred: model unavailable", prev.pk)
    return Response(
        SessionListSerializer(session).data, status=http_status.HTTP_201_CREATED
    )


@api_view(["GET"])
def session_messages(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id)
    messages = session.messages.all()
    return Response(
        {
            "session": SessionListSerializer(session).data,
            "messages": MessageSerializer(messages, many=True).data,
        }
    )


@api_view(["POST"])
def session_chat(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id)
    if not session.is_writable:
        return Response(
            {"detail": "This session has ended and is read-only."},
            status=http_status.HTTP_409_CONFLICT,
        )
    text = (request.data.get("content") or "").strip()
    if not text:
        return Response(
            {"detail": "content is required."},
            status=http_status.HTTP_400_BAD_REQUEST,
        )

    try:
        # [TEMP DEBUG] return_trace=True surfaces the per-node flow + final prompt.
        assistant_message, debug = workflow.run_turn(session, text, return_trace=True)
    except llm.LLMUnavailable as exc:
        # The turn was rolled back, so the patient can simply resend.
        return Response(
            {"detail": "The assistant is unavailable right now, so your message "
                       f"was not saved. Please try again. ({exc})"},
            status=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    session.refresh_from_db()

    new_flags = Flag.objects.filter(pk__in=session.flag_ids or [])
    new_actions = ActionEvent.objects.filter(pk__in=session.action_ids or [])
    return Response(
        {
            "assistant_message": MessageSerializer(assistant_message).data,
            "chat_state": session.chat_state,
            "flags": FlagSerializer(new_flags, many=True).data,
            "actions": ActionEventSerializer(new_actions, many=True).data,
            "debug": debug,  # [TEMP DEBUG] remove with the trace pane
        }
    )


@api_view(["POST"])
def session_end(request, session_id: int):
    session = get_object_or_404(ChatSession, pk=session_id)
    if session.ended_at is None:
        session.ended_at = timezone.now()
        session.save(update_fields=["ended_at", "updated_at"])
        try:
            summarizer.summarize_session(session, reason="manual")
        except llm.LLMUnavailable:
            # The session still ends; summarize_idle_sessions retries later.
            logger.warning("Summary for session %s deferred: model unavailable", session.pk)
        session.refresh_from_db()
    return Response(SessionListSerializer(session).data)


# ===========================================================================
# Admin / ops API
# ===========================================================================
@api_view(["GET"])
def admin_patients(request):
    return Response(PatientSerializer(Patient.objects.all(), many=True).data)


@api_view(["GET"])
def admin_dashboard(request, patient_id: int):
    patient = get_object_or_404(Patient, pk=patient_id)
    profile = patient.profile_data or {}

    unresolved_flags = patient.flags.exclude(status__in=["RESOLVED", "DISMISSED"])
    recent_sessions = patient.sessions.all()[:10]
    current_status = (
        patient.statuses.order_by("-reported_at", "-created_at")[:15]
    )

    return Response(
        {
            "patient": PatientSerializer(patient).data,
            "profile": profile,
            "conditions": profile.get("conditions", []),
            "medications": profile.get("medications", []),
            "last_visit": profile.get("last_visit", {}),
            "upcoming_appointment": profile.get("upcoming_appointment", {}),
            "current_status_timeline": PatientStatusSerializer(
                current_status, many=True
            ).data,
            "recent_conversations": SessionListSerializer(
                recent_sessions, many=True
            ).data,
            "unresolved_concerns": FlagSerializer(unresolved_flags, many=True).data,
            "alerts": {
                "flags": FlagSerializer(patient.flags.all(), many=True).data,
                "appointments": ActionEventSerializer(
                    patient.actions.filter(
                        action_type__in=["appointment_request", "appointment_change"]
                    ),
                    many=True,
                ).data,
                "doctor_messages": ActionEventSerializer(
                    patient.actions.filter(action_type="contact_doctor"), many=True
                ).data,
            },
        }
    )


EMBEDDING_FIELDS = {"message_embedding", "symptom_embedding"}
EMBEDDING_PREVIEW = 3


def _table_rows(queryset, *, preview: bool) -> dict:
    """Every concrete column of a model, as stored (FKs as ``<name>_id``).

    With ``preview`` the embedding vectors are shortened for display; the
    database always keeps the full vectors.
    """
    model = queryset.model
    fields = model._meta.concrete_fields
    rows = []
    for obj in queryset:
        row = {}
        for field in fields:
            value = getattr(obj, field.attname)
            if preview and field.name in EMBEDDING_FIELDS and isinstance(value, list):
                value = {"preview": value[:EMBEDDING_PREVIEW], "dimensions": len(value)}
            elif hasattr(value, "isoformat"):
                value = value.isoformat()
            row[field.attname] = value
        rows.append(row)
    return {
        "db_table": model._meta.db_table,
        "columns": [f.attname for f in fields],
        "rows": rows,
    }


@api_view(["GET"])
def admin_database_tables(request):
    """Patient-scoped rows from each table, with every stored column."""
    patient_id = request.query_params.get("patient_id")
    preview = request.query_params.get("preview") in ("1", "true")
    patients = Patient.objects.all()
    if patient_id:
        patients = patients.filter(pk=patient_id)
    patient = patients.first()
    if patient is None:
        return Response({"detail": "No patient found."}, status=404)

    tables = {
        "Patient": Patient.objects.filter(pk=patient.pk),
        "ChatSession": patient.sessions.order_by("started_at"),
        "Message": Message.objects.filter(session__patient=patient).order_by(
            "session", "sequence_number"
        ),
        "PatientStatus": patient.statuses.order_by("created_at"),
        "Flag": patient.flags.order_by("created_at"),
        "ActionEvent": patient.actions.order_by("created_at"),
    }
    return Response(
        {name: _table_rows(qs, preview=preview) for name, qs in tables.items()}
    )


@api_view(["GET"])
def admin_alert_detail(request, kind: str, alert_id: int):
    """Detail for a flag / appointment / doctor-message, with source links."""
    if kind == "flag":
        flag = get_object_or_404(Flag, pk=alert_id)
        data = FlagSerializer(flag).data
        source_id = flag.evidence.get("source_message_id")
        if source_id:
            msg = Message.objects.filter(pk=source_id).first()
            data["source_message"] = MessageSerializer(msg).data if msg else None
        return Response(data)
    if kind in ("appointment", "action", "doctor_message"):
        action = get_object_or_404(ActionEvent, pk=alert_id)
        return Response(ActionEventSerializer(action).data)
    return Response({"detail": "Unknown alert kind."}, status=400)


@api_view(["PATCH"])
def admin_flag_update(request, flag_id: int):
    flag = get_object_or_404(Flag, pk=flag_id)
    new_status = request.data.get("status")
    valid = {c[0] for c in Flag.STATUS_CHOICES}
    if new_status not in valid:
        return Response(
            {"detail": f"status must be one of {sorted(valid)}."}, status=400
        )
    flag.status = new_status
    flag.save(update_fields=["status", "updated_at"])
    return Response(FlagSerializer(flag).data)


@api_view(["PATCH"])
def admin_action_update(request, action_id: int):
    action = get_object_or_404(ActionEvent, pk=action_id)
    result = dict(action.result or {})
    # Allow updating mock bookkeeping fields only (never claim real completion).
    for key in ("status", "note"):
        if key in request.data:
            result[key] = request.data[key]
    action.result = result
    action.save(update_fields=["result"])
    return Response(ActionEventSerializer(action).data)


# ===========================================================================
# Web pages
# ===========================================================================
def chat_page(request, patient_id: int):
    patient = get_object_or_404(Patient, pk=patient_id)
    return render(request, "assistant_app/chat.html", {"patient": patient})


def ops_dashboard_root(request):
    patient = Patient.objects.first()
    return render(
        request,
        "assistant_app/dashboard.html",
        {"patient": patient, "patients": Patient.objects.all()},
    )


def ops_dashboard_patient(request, patient_id: int):
    patient = get_object_or_404(Patient, pk=patient_id)
    return render(
        request,
        "assistant_app/dashboard.html",
        {"patient": patient, "patients": Patient.objects.all()},
    )
