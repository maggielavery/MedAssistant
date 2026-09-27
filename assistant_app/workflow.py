"""LangGraph orchestration of the assistant workflow.

Stages (each an explicit, independently testable node):

    Patient message
        -> persist raw Message
        -> load profile + bounded current-session context
        -> intent agent
        -> symptom agent
        -> longitudinal retrieval
        -> flag & action agent
        -> persist statuses, flags, actions
        -> responder agent
        -> persist assistant Message

The graph state is a plain dict so each stage's inputs/outputs stay legible and
each agent can be evaluated in isolation.
"""
from __future__ import annotations

from typing import Any, TypedDict

from django.conf import settings
from django.db import transaction
from django.db.models import Max
from django.utils import timezone
from langgraph.graph import END, StateGraph

from . import agents, retrieval
from .models import (
    ActionEvent,
    ChatSession,
    Flag,
    Message,
    PatientStatus,
)

# Follow-up questions asked about one pending symptom before it is persisted
# with whatever is known, so an unanswerable detail cannot loop forever.
MAX_SYMPTOM_FOLLOW_UPS = 2

ACTION_INTENTS = {
    "appointment_request": "appointment_request",
    "appointment_change": "appointment_change",
    "contact_doctor": "contact_doctor",
}


class TurnState(TypedDict, total=False):
    session_id: int
    patient_message_id: int
    message_text: str
    profile: dict
    chat_state: dict
    recent_messages: list[dict]
    intent: str
    observations: list[dict]
    follow_up: Any
    timeline: list[dict]
    flags: list[dict]
    new_status_ids: list[int]
    new_flag_ids: list[int]
    linked_flag_ids: list[int]
    persisted_flags: list[dict]
    recorded_actions: list[dict]
    action_updates: list[dict]
    updated_actions: list[dict]
    closed_actions: list[dict]
    open_requests: list[dict]
    checkup_just_completed: bool
    checkup_just_started: bool
    reply_text: str
    assistant_message_id: int
    trace: list[dict]
    responder_prompt: dict


# ---------------------------------------------------------------------------
# [DEBUG] Trace helper -- records each node's inputs/outputs so the chat
# UI can display the flow. 
# ---------------------------------------------------------------------------
def _trace(state: TurnState, node: str, inputs: dict, outputs: dict) -> None:
    state.setdefault("trace", []).append(
        {
            "node": node,
            "inputs": _safe(inputs),
            "outputs": _safe(outputs),
        }
    )


def _safe(obj):
    """Best-effort JSON-serializable snapshot for the trace."""
    import json

    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return {k: str(v) for k, v in (obj or {}).items()}


# ---------------------------------------------------------------------------
# Stage nodes
# ---------------------------------------------------------------------------
def _load_context(state: TurnState) -> TurnState:
    """Load the patient profile and bounded session context for this turn.

    Objective: give every later stage the patient-specific configuration and
    the session's working memory without loading the whole conversation.

    Consumes: ``session_id`` (and ``message_text`` for the trace). Reads the
    ChatSession, its Patient, and the last ``RECENT_MESSAGE_WINDOW`` (12)
    messages, which include the patient message just persisted by
    ``run_turn``.

    Returns/updates:
      - ``profile``: ``Patient.profile_data`` (conditions, medications,
        watch-fors, guidance).
      - ``chat_state``: a copy of ``ChatSession.chat_state`` (active workflow,
        checkup progress, pending symptoms, open request ids). Later nodes
        mutate this copy; ``_persist_node`` writes it back.
      - ``recent_messages``: ``[{role, content}]`` oldest first.
      - ``trace``: appends a ``load_context`` entry (debug).
    """
    session = ChatSession.objects.select_related("patient").get(pk=state["session_id"])
    profile = session.patient.profile_data or {}
    window = getattr(settings, "RECENT_MESSAGE_WINDOW", 12)
    recent = list(
        session.messages.order_by("-sequence_number")[:window]
    )
    recent.reverse()
    state["profile"] = profile
    state["chat_state"] = dict(session.chat_state or {})
    state["recent_messages"] = [
        {"role": m.role, "content": m.content} for m in recent
    ]
    _trace(
        state,
        "load_context",
        {"session_id": state["session_id"], "message_text": state["message_text"]},
        {
            "active_workflow": state["chat_state"].get("active_workflow", "open_conversation"),
            "recent_message_count": len(state["recent_messages"]),
            "chat_state": state["chat_state"],
        },
    )
    return state


def _intent_node(state: TurnState) -> TurnState:
    """Classify the message's primary intent and start a checkup if asked.

    Objective: decide what the patient is asking for (routine_checkup,
    appointment_request, appointment_change, contact_doctor, general_chat) so
    the workflow and the action step know how to handle the turn.

    Consumes: ``message_text``, ``chat_state`` (active workflow), ``profile``
    (watch-fors, when starting a checkup), ``patient_message_id``.

    Returns/updates:
      - ``intent``: the label from ``agents.classify_intent``; also saved on
        the patient ``Message.intent`` row.
      - ``chat_state``: when the intent is ``routine_checkup`` and no checkup
        is running, sets ``active_workflow`` and a fresh ``checkup`` (first
        watch-for as the current item). A running checkup keeps its progress.
      - ``checkup_just_started``: True only when a new checkup began (set only
        for a routine_checkup intent).
      - ``trace``: appends an ``intent_agent`` entry (debug).
    """
    result = agents.classify_intent(
        state["message_text"], state["chat_state"], state["profile"]
    )
    state["intent"] = result["intent"]
    # Intent is an LLM label, not a calibrated confidence score.
    Message.objects.filter(pk=state["patient_message_id"]).update(
        intent=result["intent"]
    )
    # Maintain active workflow. Starting a checkup switches workflow; an action
    # intent during a checkup is handled without discarding checkup progress.
    chat_state = state["chat_state"]
    if result["intent"] == "routine_checkup":
        state["checkup_just_started"] = _start_or_resume_checkup(chat_state, state["profile"])
    _trace(
        state,
        "intent_agent",
        {"message_text": state["message_text"],
         "active_workflow": chat_state.get("active_workflow", "open_conversation")},
        {"intent": result["intent"]},
    )
    return state


def _symptom_node(state: TurnState) -> TurnState:
    """Extract observations and advance pending follow-ups and the checkup.

    Objective: find every symptom or clinical observation the patient states,
    merge answers into pending follow-up questions, and move the checkup to
    its next item when the current one was addressed.

    Consumes: ``message_text``, ``profile`` (watch-fors), ``chat_state``
    (pending symptoms, checkup, open request ids), ``intent`` (an action
    request mid-follow-up does not count as an unanswered follow-up),
    ``patient_message_id``, ``session_id``. Reads the patient message's
    timestamp (to resolve relative onsets) and the previous assistant message
    (context for short answers like "6").

    Returns/updates:
      - ``observations``: normalized observations from
        ``agents.extract_symptoms``, then merged in place by
        ``_update_pending_symptoms`` (prior fields, ``message_ids``,
        ``status_id``, recomputed ``missing_fields``, ``follow_ups_asked``).
        A pending symptom closed after too many unanswered follow-ups is
        appended here so it gets persisted.
      - ``follow_up``: the agent's suggested follow-up question, or None.
      - ``chat_state``: updated ``pending_symptoms``; ``checkup`` advanced by
        ``_advance_checkup`` (and ``active_workflow`` reset when complete).
      - ``checkup_just_completed``: True only on the turn the checkup ends.
      - ``trace``: appends a ``symptom_agent`` entry (debug).
    """
    message = Message.objects.get(pk=state["patient_message_id"])
    previous_reply = (
        Message.objects.filter(
            session_id=state["session_id"],
            role="assistant",
            sequence_number__lt=message.sequence_number,
        )
        .order_by("-sequence_number")
        .first()
    )
    result = agents.extract_symptoms(
        state["message_text"],
        state["profile"],
        state["chat_state"],
        previous_assistant_message=previous_reply.content if previous_reply else "",
        reference_time=timezone.localtime(message.timestamp),
    )
    state["observations"] = result["observations"]
    state["follow_up"] = result.get("follow_up")
    _update_pending_symptoms(
        state["chat_state"],
        result["observations"],
        state["follow_up"],
        source_message_id=state["patient_message_id"],
        # An action request mid-follow-up is an interruption, not a non-answer.
        count_unanswered=state["intent"] not in ACTION_INTENTS,
    )
    was_completed = (state["chat_state"].get("checkup") or {}).get("completed", True)
    _advance_checkup(state["chat_state"], state["profile"], result["observations"])
    state["checkup_just_completed"] = bool(
        not was_completed and (state["chat_state"].get("checkup") or {}).get("completed")
    )
    _trace(
        state,
        "symptom_agent",
        {"message_text": state["message_text"],
         "previous_assistant_message": previous_reply.content if previous_reply else "",
         "message_time": timezone.localtime(message.timestamp).isoformat(),
         "watch_for_ids": [w["id"] for w in state["profile"].get("watch_fors", [])]},
        {"observations": result["observations"],
         "follow_up": result.get("follow_up"),
         "pending_symptoms": state["chat_state"].get("pending_symptoms", [])},
    )
    return state


def _retrieval_node(state: TurnState) -> TurnState:
    """Retrieve the patient's relevant symptom history as a timeline.

    Objective: give the responder source-linked history for every topic in
    play, so it can reference earlier reports and never treat an old report as
    the current status.

    Consumes: ``observations``, ``chat_state`` (``pending_symptoms``; the
    checkup's current item when a routine checkup is active), ``profile``
    (watch-for label and guidance for the checkup item), ``session_id``,
    ``patient_message_id``.

    Builds one query per topic (each observation or pending symptom, using
    its own first source message as context; plus the checkup item about to
    be asked, using its guidance as context) and calls
    ``retrieval.build_timeline``. With no topic, nothing is searched.

    Returns/updates:
      - ``timeline``: patient-scoped PatientStatus entries in clinical-time
        order, each with source message id and excerpt, times, status,
        watch-for, scores, the matching query, and why it matched; ``[]``
        when there is no topic.
      - ``trace``: appends a ``longitudinal_retrieval`` entry (debug).
    Read-only apart from lazily backfilling missing embeddings.
    """
    session = ChatSession.objects.select_related("patient").get(pk=state["session_id"])
    # One query per topic in play: this turn's observations, pending symptoms,
    # and the checkup item the responder will ask about next. With none (e.g.
    # "I just want to chat") the status table is not searched at all.
    queries: list[dict] = []
    seen: set[str] = set()

    def add(label: str, keywords: list[str], *, watch_for_id: str = "",
            context_message_id=None, context_text: str | None = None) -> None:
        keywords = list(dict.fromkeys([label] + [k for k in keywords if k]))
        if label and label not in seen:
            seen.add(label)
            queries.append({"label": label, "keywords": keywords,
                            "watch_for_id": watch_for_id or "",
                            "context_message_id": context_message_id,
                            "context_text": context_text})

    for item in state.get("observations", []) + state["chat_state"].get("pending_symptoms", []):
        if item.get("symptom"):
            # The symptom's own source message (first report), so a follow-up
            # answer like "since yesterday, mild" still matches its history.
            source_ids = item.get("message_ids") or [state["patient_message_id"]]
            add(item["symptom"], item.get("keywords") or [],
                watch_for_id=item.get("aligns_watch_for_id", ""),
                context_message_id=source_ids[0])
    checkup = state["chat_state"].get("checkup") or {}
    if state["chat_state"].get("active_workflow") == "routine_checkup" and checkup.get("current_watch_for_id"):
        watch_for = next(
            (w for w in state["profile"].get("watch_fors", [])
             if w["id"] == checkup["current_watch_for_id"]),
            None,
        )
        if watch_for:
            add(
                watch_for["label"].lower(),
                [],
                watch_for_id=watch_for["id"],
                context_text=f"{watch_for['label']}. {watch_for.get('guidance', '')}".strip(),
            )

    if queries:
        message = Message.objects.get(pk=state["patient_message_id"])
        state["timeline"] = retrieval.build_timeline(session.patient, message, queries=queries)
    else:
        state["timeline"] = []

    _trace(
        state,
        "longitudinal_retrieval",
        {"source_message_id": state["patient_message_id"], "queries": queries,
         "skipped_no_topic": not queries},
        {"timeline": state["timeline"]},
    )
    return state


def _flag_action_node(state: TurnState) -> TurnState:
    """Decide which observations to flag and which open requests gain details.

    Objective: propose care-team flags for concerning observations, and
    detect details this message adds to a request that is still open. Nothing
    is written here; ``_persist_node`` saves the results.

    Consumes: ``observations``, ``profile`` (watch-fors and guidance),
    ``timeline``, ``patient_message_id`` (its stored embedding, for flag
    confidence), ``chat_state.open_action_ids``, ``session_id``,
    ``message_text``, ``intent``.

    Returns/updates:
      - ``flags``: proposed flags from ``agents.evaluate_flags``
        (``symptom``, ``description``, ``watch_for_id``, embedding-based
        ``confidence`` and its two similarity components).
      - ``action_updates``: ``[{action_id, detail}]`` from
        ``agents.match_action_updates``, considering only open requests.
      - ``trace``: appends a ``flag_and_action_agent`` entry (debug).
    """
    message = Message.objects.only("message_embedding").get(pk=state["patient_message_id"])
    state["flags"] = agents.evaluate_flags(
        state.get("observations", []),
        state["profile"],
        state.get("timeline", []),
        message.message_embedding,
    )
    # Only requests still open for details (chat_state.open_action_ids) can
    # gain details from later chat (a preferred date, symptom specifics for
    # the doctor, ...). A request closes once a reply adds nothing to it.
    open_ids = state["chat_state"].get("open_action_ids", [])
    open_actions = [
        {
            "id": a.pk,
            "action_type": a.action_type,
            "request_text": a.request_text,
            "follow_up_details": a.follow_up_details,
        }
        for a in ActionEvent.objects.filter(session_id=state["session_id"], pk__in=open_ids)
    ]
    state["action_updates"] = agents.match_action_updates(
        state["message_text"], state.get("intent", ""), open_actions
    )
    _trace(
        state,
        "flag_and_action_agent",
        {"observations": state.get("observations", []),
         "intent": state.get("intent"),
         "open_action_ids": [a["id"] for a in open_actions]},
        {"flags": state["flags"], "action_updates": state["action_updates"]},
    )
    return state


def _persist_node(state: TurnState) -> TurnState:
    """Persist this turn's statuses, flags, and requests in one transaction.

    Objective: turn the agents' proposals into traceable records and keep the
    session's working memory and aggregates consistent.

    Consumes: ``observations``, ``flags``, ``action_updates``, ``intent``,
    ``message_text``, ``chat_state`` (``pending_symptoms``,
    ``open_action_ids``, ``active_workflow``), ``session_id``,
    ``patient_message_id``.

    Writes to the database:
      - PatientStatus: creates one row per complete observation (or a flagged
        incomplete one, early); completes the early row when a follow-up
        supplies its details.
      - Flag: creates a flag for each proposed one whose status was saved and
        is not absent/historical. If this session already has an active flag
        for the same concern (same watch-for, else label), no new flag is
        created; a new report is instead appended to that flag's
        ``evidence.repeat_reports`` and linked from the status. A report in a
        new session always gets its own flag.
      - ActionEvent: appends ``follow_up_details`` to updated open requests;
        creates a new request for an action intent unless it only updated an
        open request of the same type.
      - Message (the patient message): ``action``, ``patient_status_ids``,
        ``flag_ids``.
      - ChatSession: aggregated ``patient_status_ids``, ``flag_ids``,
        ``action_ids``; ``chat_state``; ``workflow_type``.

    Returns/updates:
      - ``chat_state``: ``status_id`` on pending symptoms saved early;
        ``open_action_ids`` (a request stays open only while this turn added
        to it; new requests open).
      - ``new_status_ids``, ``new_flag_ids``: ids created or completed.
      - ``linked_flag_ids``: this session's active flags that this turn's new
        report was linked to as a repeat.
      - ``persisted_flags``: flag specs the care team can see because of this
        turn (created, or linked as a repeat with ``repeat_of_flag_id``).
      - ``recorded_actions``: ``[{id, action_type}]`` new requests.
      - ``updated_actions``: ``[{id, action_type, detail}]`` details added.
      - ``closed_actions``: ``[{id, action_type}]`` requests closed this turn.
      - ``open_requests``: ``[{id, action_type}]`` requests still open.
      - ``trace``: appends a ``persist`` entry (debug).
    """
    with transaction.atomic():
        session = ChatSession.objects.select_related("patient").get(
            pk=state["session_id"]
        )
        patient = session.patient
        message = Message.objects.get(pk=state["patient_message_id"])

        new_status_ids: list[int] = []
        status_by_symptom: dict[str, PatientStatus] = {}
        flagged_symptoms = {f["symptom"] for f in state.get("flags", [])}
        pending_by_symptom = {
            p["symptom"]: p for p in state["chat_state"].get("pending_symptoms", [])
        }
        for obs in state.get("observations", []):
            if not obs.get("symptom"):
                continue
            existing_id = obs.get("status_id")
            # Incomplete observations normally live only in chat_state until the
            # focused follow-up supplies their missing context. A flagged one is
            # persisted now so the care team sees it without waiting.
            if (
                obs.get("missing_fields")
                and obs["symptom"] not in flagged_symptoms
                and not existing_id
            ):
                continue
            # Only messages that contributed to this observation are linked.
            message_ids = list(dict.fromkeys(obs.get("message_ids") or [message.pk]))
            status = (
                PatientStatus.objects.filter(pk=existing_id, patient=patient).first()
                if existing_id
                else None
            )
            if status is not None:
                # Follow-up details for the same report complete the row that
                # was persisted early; this is not a newer, contradicting report.
                status.message_ids = list(dict.fromkeys(status.message_ids + message_ids))
                status.symptom_status = obs["status"]
                status.onset = obs["onset"]
                status.clinical_time = _onset_date(obs["onset"]) or status.clinical_time
                status.severity = obs["severity"]
                status.description = obs["description"]
                status.missing_fields = obs.get("missing_fields", [])
                status.watch_for_id = obs.get("aligns_watch_for_id") or status.watch_for_id
                status.save(update_fields=[
                    "message_ids", "symptom_status", "onset", "clinical_time", "severity",
                    "description", "missing_fields", "watch_for_id", "updated_at",
                ])
            else:
                status = PatientStatus.objects.create(
                    patient=patient,
                    session=session,
                    source_message_id=message_ids[0],
                    date=timezone.now().date(),
                    reported_at=timezone.now(),
                    clinical_time=_onset_date(obs["onset"]),
                    symptoms=[obs["symptom"]],
                    watch_for_id=obs.get("aligns_watch_for_id") or "",
                    symptom_embedding=retrieval.embed_symptom(obs["symptom"]),
                    message_ids=message_ids,
                    symptom_status=obs["status"],
                    onset=obs["onset"],
                    severity=obs["severity"],
                    description=obs["description"],
                    missing_fields=obs.get("missing_fields", []),
                )
                if obs["symptom"] in pending_by_symptom:
                    pending_by_symptom[obs["symptom"]]["status_id"] = status.pk
            new_status_ids.append(status.pk)
            status_by_symptom[obs["symptom"]] = status

        # Flags: skip absent/historical and de-duplicate active repeats.
        new_flag_ids: list[int] = []
        linked_flag_ids: list[int] = []
        persisted_flags: list[dict] = []
        for flag_spec in state.get("flags", []):
            status = status_by_symptom.get(flag_spec["symptom"])
            if status is None or status.symptom_status in ("absent", "historical"):
                continue
            existing = _active_flag_for(patient, status, session)
            if existing is not None:
                # Same unresolved concern within this session: no second flag,
                # but a new report is linked to the active one so the Alerts
                # pane shows it again. Follow-ups completing an already-linked
                # report don't repeat. A report in a new session always gets
                # its own flag.
                if existing.pk not in (status.flags_raised or []):
                    evidence = dict(existing.evidence or {})
                    evidence["repeat_reports"] = (evidence.get("repeat_reports") or []) + [{
                        "message_id": message.pk,
                        "status_id": status.pk,
                        "session_id": session.pk,
                        "quote": message.content[:280],
                        "reported_at": timezone.now().isoformat(),
                    }]
                    existing.evidence = evidence
                    existing.save(update_fields=["evidence", "updated_at"])
                    status.flags_raised = list(dict.fromkeys((status.flags_raised or []) + [existing.pk]))
                    status.save(update_fields=["flags_raised"])
                    linked_flag_ids.append(existing.pk)
                    persisted_flags.append({**flag_spec, "repeat_of_flag_id": existing.pk})
                continue
            flag = Flag.objects.create(
                patient=patient,
                patient_status=status,
                description=flag_spec["description"],
                evidence={
                    "source_message_id": message.pk,
                    "message_ids": status.message_ids,
                    "quote": message.content[:280],
                    "watch_for_id": flag_spec.get("watch_for_id", ""),
                    "symptom_similarity": flag_spec.get("symptom_similarity"),
                    "message_similarity": flag_spec.get("message_similarity"),
                    "missing_fields": next(
                        (o.get("missing_fields", []) for o in state.get("observations", [])
                         if o.get("symptom") == flag_spec["symptom"]),
                        [],
                    ),
                    "reported_at": timezone.now().isoformat(),
                },
                confidence=flag_spec["confidence"],
                status="NEW",
            )
            status.flags_raised = list(dict.fromkeys((status.flags_raised or []) + [flag.pk]))
            status.save(update_fields=["flags_raised"])
            new_flag_ids.append(flag.pk)
            persisted_flags.append(flag_spec)

        # Details for requests already sent this session. The original
        # request_text is never rewritten; additions go to follow_up_details.
        updated_actions: list[dict] = []
        for update in state.get("action_updates", []):
            action = ActionEvent.objects.filter(
                pk=update["action_id"], session=session
            ).first()
            if action is None:
                continue
            action.follow_up_details = list(action.follow_up_details or []) + [{
                "message_id": message.pk,
                "detail": update["detail"],
                "quote": message.content[:280],
                "added_at": timezone.now().isoformat(),
            }]
            action.save(update_fields=["follow_up_details", "updated_at"])
            if message.action_id is None:
                message.action = action
            updated_actions.append({
                "id": action.pk,
                "action_type": action.action_type,
                "detail": update["detail"],
            })

        # Mocked operational actions from intent. A repeat of a request type
        # that this message just updated is a detail, not a second request.
        recorded_actions: list[dict] = []
        action_type = ACTION_INTENTS.get(state["intent"])
        if action_type and action_type not in {a["action_type"] for a in updated_actions}:
            action = ActionEvent.objects.create(
                patient=patient,
                session=session,
                action_type=action_type,
                request_text=state["message_text"],
                result={
                    "status": "sent_to_care_team",
                    "booked": False,
                    "delivered": False,
                    "note": "Prototype: routed to the care-team dashboard only; "
                    "not booked or delivered to the doctor.",
                },
            )
            message.action = action
            recorded_actions.append(
                {"id": action.pk, "action_type": action.action_type}
            )
        message.save(update_fields=["action"])

        # A request stays open for details only while the patient keeps adding
        # to it; a reply that adds nothing ("that's everything", or moving on
        # to something else) closes it for good.
        previous_open = state["chat_state"].get("open_action_ids", [])
        updated_ids = {a["id"] for a in updated_actions}
        still_open = [i for i in previous_open if i in updated_ids]
        new_open = [a["id"] for a in recorded_actions]
        state["chat_state"]["open_action_ids"] = still_open + new_open
        types = dict(
            ActionEvent.objects.filter(pk__in=previous_open + new_open)
            .values_list("pk", "action_type")
        )
        closed_actions = [
            {"id": i, "action_type": types[i]}
            for i in previous_open if i not in updated_ids and i in types
        ]
        open_requests = [
            {"id": i, "action_type": types[i]} for i in still_open + new_open if i in types
        ]

        # Update message + session aggregates.
        message.patient_status_ids = new_status_ids
        message.flag_ids = list(dict.fromkeys(new_flag_ids + linked_flag_ids))
        message.save(update_fields=["patient_status_ids", "flag_ids"])

        session.patient_status_ids = list(
            dict.fromkeys((session.patient_status_ids or []) + new_status_ids)
        )
        session.flag_ids = list(
            dict.fromkeys((session.flag_ids or []) + new_flag_ids + linked_flag_ids)
        )
        session.action_ids = list(
            dict.fromkeys(
                (session.action_ids or []) + [a["id"] for a in recorded_actions]
            )
        )
        session.chat_state = state["chat_state"]
        session.workflow_type = state["chat_state"].get(
            "active_workflow", session.workflow_type
        )
        session.save(
            update_fields=[
                "patient_status_ids",
                "flag_ids",
                "action_ids",
                "chat_state",
                "workflow_type",
                "updated_at",
            ]
        )

        state["new_status_ids"] = new_status_ids
        state["new_flag_ids"] = new_flag_ids
        state["linked_flag_ids"] = linked_flag_ids
        # Flags the care team can see because of this turn: newly created, or
        # an active flag this new report was linked to as a repeat.
        state["persisted_flags"] = persisted_flags
        state["recorded_actions"] = recorded_actions
        state["updated_actions"] = updated_actions
        state["closed_actions"] = closed_actions
        state["open_requests"] = open_requests
    _trace(
        state,
        "persist",
        {"observations": state.get("observations", []),
         "flags": state.get("flags", []),
         "intent": state.get("intent")},
        {"new_status_ids": state["new_status_ids"],
         "new_flag_ids": state["new_flag_ids"],
         "linked_flag_ids": state["linked_flag_ids"],
         "recorded_actions": state["recorded_actions"],
         "updated_actions": state["updated_actions"],
         "closed_actions": state["closed_actions"],
         "open_action_ids": state["chat_state"].get("open_action_ids", [])},
    )
    return state


def _responder_node(state: TurnState) -> TurnState:
    """Compose the assistant's reply from bounded, structured context.

    Objective: respond naturally while following the turn's facts: confirm
    requests sent this turn, acknowledge added details, ask one pending
    question, continue or conclude the checkup, and never claim anything that
    did not happen.

    Consumes: ``profile``, ``chat_state`` (active workflow, checkup, pending
    symptoms), ``recent_messages``, ``message_text``, ``observations``,
    ``persisted_flags``, ``recorded_actions``, ``updated_actions``,
    ``closed_actions``, ``open_requests``, ``checkup_just_completed``,
    ``checkup_just_started``, ``timeline``.

    Returns/updates:
      - ``reply_text``: the reply from ``agents.compose_reply``.
      - ``responder_prompt``: the exact ``{system, user}`` prompt sent
        (debug).
      - ``trace``: appends a ``responder_agent`` entry (debug).
    """
    chat_state = state["chat_state"]
    checkup = chat_state.get("checkup") or {}
    context = {
        "profile": state["profile"],
        "active_workflow": chat_state.get("active_workflow", "open_conversation"),
        "checkup": checkup,
        "pending_symptoms": chat_state.get("pending_symptoms", []),
        "recent_messages": state.get("recent_messages", []),
        "message_text": state["message_text"],
        "new_statuses": [
            {"symptom": o.get("symptom", ""), "status": o["status"]}
            for o in state.get("observations", [])
        ],
        "new_flags": state.get("persisted_flags", []),
        "recorded_actions": state.get("recorded_actions", []),
        "updated_actions": state.get("updated_actions", []),
        "closed_actions": state.get("closed_actions", []),
        "open_requests": state.get("open_requests", []),
        "checkup_just_completed": state.get("checkup_just_completed", False),
        "checkup_just_started": state.get("checkup_just_started", False),
        "timeline": state.get("timeline", []),
    }
    # [TEMP DEBUG] Capture the exact final prompt sent to the LLM.
    prompt = agents.build_responder_prompt(context)
    state["responder_prompt"] = prompt
    state["reply_text"] = agents.compose_reply(context)
    _trace(
        state,
        "responder_agent",
        {"final_prompt_system": prompt["system"],
         "final_prompt_user": prompt["user"]},
        {"reply_text": state["reply_text"]},
    )
    return state


def _persist_reply_node(state: TurnState) -> TurnState:
    """Save the assistant's reply as the next message in the session.

    Objective: record the reply (with its embedding) so it appears in the
    chat and becomes context for later turns.

    Consumes: ``session_id``, ``reply_text``.

    Writes an assistant Message with the next ``sequence_number``.

    Returns/updates:
      - ``assistant_message_id``: the new Message id, which ``run_turn``
        returns to the caller.
    """
    session = ChatSession.objects.get(pk=state["session_id"])
    seq = _next_sequence(session)
    reply = Message.objects.create(
        session=session,
        role="assistant",
        content=state["reply_text"],
        message_embedding=retrieval.embed_message(state["reply_text"]),
        sequence_number=seq,
    )
    state["assistant_message_id"] = reply.pk
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def _build_graph():
    graph = StateGraph(TurnState)
    graph.add_node("load_context", _load_context)
    graph.add_node("intent", _intent_node)
    graph.add_node("symptom", _symptom_node)
    graph.add_node("retrieval", _retrieval_node)
    graph.add_node("flag_action", _flag_action_node)
    graph.add_node("persist", _persist_node)
    graph.add_node("responder", _responder_node)
    graph.add_node("persist_reply", _persist_reply_node)

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "intent")
    graph.add_edge("intent", "symptom")
    graph.add_edge("symptom", "retrieval")
    graph.add_edge("retrieval", "flag_action")
    graph.add_edge("flag_action", "persist")
    graph.add_edge("persist", "responder")
    graph.add_edge("responder", "persist_reply")
    graph.add_edge("persist_reply", END)
    return graph.compile()


_COMPILED_GRAPH = None


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()
    return _COMPILED_GRAPH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_turn(session: ChatSession, patient_text: str, return_trace: bool = False):
    """Persist the patient message, run the graph, return the assistant Message.

    When ``return_trace`` is True, returns ``(Message, debug)`` where ``debug``
    holds the per-node trace and final responder prompt. [TEMP DEBUG]
    """
    if not session.is_writable:
        raise ValueError("Cannot write to an ended/read-only session.")

    # One transaction per turn: if any agent raises (e.g. llm.LLMUnavailable),
    # nothing from the turn is kept, including the patient message.
    with transaction.atomic():
        seq = _next_sequence(session)
        patient_message = Message.objects.create(
            session=session,
            role="patient",
            content=patient_text,
            message_embedding=retrieval.embed_message(patient_text),
            sequence_number=seq,
        )

        final = _get_graph().invoke(
            {
                "session_id": session.pk,
                "patient_message_id": patient_message.pk,
                "message_text": patient_text,
                "trace": [],
            }
        )
    message = Message.objects.get(pk=final["assistant_message_id"])
    if return_trace:
        debug = {
            "trace": final.get("trace", []),
            "final_prompt": final.get("responder_prompt", {}),
        }
        return message, debug
    return message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_sequence(session: ChatSession) -> int:
    current = session.messages.aggregate(m=Max("sequence_number"))["m"]
    return (current or 0) + 1


def _onset_date(onset: str):
    """Clinical date implied by an approximate ISO onset, if there is one."""
    parsed = agents.parse_onset(onset)
    return parsed.date() if parsed else None


def _active_flag_for(patient, status: PatientStatus, session: ChatSession) -> Flag | None:
    """This session's active (NEW/OPEN/ACKNOWLEDGED) flag for the same concern.

    Only flags raised in the same chat session count, so a report in a new
    session always gets its own flag, whatever happened to earlier ones.
    Same concern means the same watch-for, else the same symptom label, so
    "chest pressure" and "chest pain" (both chest_pain) are one concern.
    Compared in Python because SQLite lacks JSON ``contains`` lookups.
    """
    active = Flag.objects.filter(
        patient=patient,
        patient_status__session=session,
        status__in=["NEW", "OPEN", "ACKNOWLEDGED"],
    ).select_related("patient_status").order_by("created_at")
    for flag in active:
        prior = flag.patient_status
        if status.watch_for_id and prior.watch_for_id == status.watch_for_id:
            return flag
        if status.label in (prior.symptoms or []):
            return flag
    return None


def _start_or_resume_checkup(chat_state: dict, profile: dict) -> bool:
    """Start a checkup unless one is running. Returns True when newly started."""
    watch_fors = profile.get("watch_fors", [])
    if not watch_fors:
        return False
    if chat_state.get("active_workflow") == "routine_checkup" and chat_state.get(
        "checkup"
    ):
        return False  # already running; keep progress
    chat_state["active_workflow"] = "routine_checkup"
    chat_state["checkup"] = {
        "watch_for_ids_completed": [],
        "current_watch_for_id": watch_fors[0]["id"],
        "current_watch_for_label": watch_fors[0]["label"],
        "completed": False,
    }
    return True


def _advance_checkup(chat_state: dict, profile: dict, observations: list[dict]) -> None:
    checkup = chat_state.get("checkup")
    if not checkup or checkup.get("completed"):
        return
    watch_fors = profile.get("watch_fors", [])
    order = [w["id"] for w in watch_fors]
    label_by_id = {w["id"]: w["label"] for w in watch_fors}
    current = checkup.get("current_watch_for_id")

    # If the patient addressed the current watch-for (present OR explicit absent),
    # record it as completed and move on.
    addressed = {
        o["aligns_watch_for_id"]
        for o in observations
        if o.get("aligns_watch_for_id") and o["status"] in ("present", "absent",
                                                            "resolved", "uncertain")
    }
    completed = set(checkup.get("watch_for_ids_completed", []))
    if current in addressed:
        completed.add(current)

    # Advance to the next not-yet-completed watch-for.
    next_id = None
    for wid in order:
        if wid not in completed:
            next_id = wid
            break

    checkup["watch_for_ids_completed"] = [w for w in order if w in completed]
    if next_id is None:
        checkup["completed"] = True
        checkup["current_watch_for_id"] = None
        checkup["current_watch_for_label"] = None
        chat_state["active_workflow"] = "open_conversation"
    else:
        checkup["current_watch_for_id"] = next_id
        checkup["current_watch_for_label"] = label_by_id.get(next_id, next_id)
    chat_state["checkup"] = checkup


def _update_pending_symptoms(
    chat_state: dict,
    observations: list[dict],
    follow_up: Any,
    *,
    source_message_id: int,
    count_unanswered: bool = True,
) -> None:
    pending = {p["symptom"]: p for p in chat_state.get("pending_symptoms", [])}
    previously_pending = set(pending)
    addressed = {obs.get("symptom") for obs in observations}
    for obs in observations:
        symptom = obs.get("symptom")
        if not symptom:
            continue
        prior = pending.get(symptom)
        if prior:
            # The extractor may return only the newly supplied values. Merge
            # them with the temporary record before deciding whether it is ready
            # to persist, so the final row retains the original description.
            for field in ("label", "status", "onset", "severity", "description", "aligns_watch_for_id"):
                if not obs.get(field):
                    obs[field] = prior.get(field, "")
            if not obs.get("keywords"):
                obs["keywords"] = prior.get("keywords", [])
            if prior.get("status_id"):
                obs["status_id"] = prior["status_id"]
            obs["message_ids"] = list(dict.fromkeys(
                prior.get("message_ids", []) + obs.get("message_ids", []) + [source_message_id]
            ))
            prior_description = prior.get("description", "")
            if (
                prior_description
                and obs.get("description")
                and obs["description"] != prior_description
            ):
                obs["description"] = (
                    f"{prior_description} Follow-up: {obs['description']}"
                )
            obs["missing_fields"] = agents.missing_detail_fields(
                obs["status"], onset=obs.get("onset"), severity=obs.get("severity")
            )
            follow_ups = prior.get("follow_ups_asked", 0) + 1
            if obs["missing_fields"] and follow_ups >= MAX_SYMPTOM_FOLLOW_UPS:
                # Stop asking; persist what is known and record the gap.
                obs["description"] = (
                    f"{obs.get('description', '')} Not provided after follow-up: "
                    f"{', '.join(obs['missing_fields'])}."
                ).strip()
                obs["missing_fields"] = []
            obs["follow_ups_asked"] = follow_ups
            if not obs["missing_fields"]:
                pending.pop(symptom, None)
                continue
        if obs["missing_fields"] and obs["status"] in ("present", "uncertain"):
            obs["message_ids"] = list(dict.fromkeys(obs.get("message_ids", []) + [source_message_id]))
            pending[symptom] = {
                "symptom": symptom,
                "keywords": obs.get("keywords", [symptom]),
                "label": obs.get("label", symptom),
                "status": obs["status"],
                "onset": obs.get("onset", ""),
                "severity": obs.get("severity", ""),
                "description": obs.get("description", ""),
                "aligns_watch_for_id": obs.get("aligns_watch_for_id", ""),
                "message_ids": obs["message_ids"],
                "missing_fields": obs["missing_fields"],
                "status_id": obs.get("status_id"),
                "follow_ups_asked": obs.get("follow_ups_asked", 0),
                "question": follow_up.get("question")
                if isinstance(follow_up, dict)
                and follow_up.get("symptom") == symptom
                else "",
            }
        elif not obs.get("message_ids"):
            obs["message_ids"] = [source_message_id]

    # A reply that doesn't address a pending question ("no idea", a change of
    # subject) still uses up a follow-up, so an unanswerable detail can't loop.
    for symptom in previously_pending - addressed:
        record = pending.get(symptom)
        if record is None or not count_unanswered:
            continue
        record["follow_ups_asked"] = record.get("follow_ups_asked", 0) + 1
        if record["follow_ups_asked"] >= MAX_SYMPTOM_FOLLOW_UPS:
            pending.pop(symptom)
            observations.append({
                **{k: record.get(k) for k in (
                    "symptom", "keywords", "label", "status", "onset", "severity",
                    "aligns_watch_for_id", "message_ids", "status_id", "follow_ups_asked",
                )},
                "onset": record.get("onset") or "",
                "severity": record.get("severity") or "",
                "description": (
                    f"{record.get('description', '')} Not provided after follow-up: "
                    f"{', '.join(record.get('missing_fields', []))}."
                ).strip(),
                "is_patient_report": True,
                "missing_fields": [],
            })
    chat_state["pending_symptoms"] = list(pending.values())
