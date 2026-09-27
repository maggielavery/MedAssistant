"""Seed the single demo patient, Jordan Ellis.

All patient-specific data lives here (or is derived from ``profile_data``), never
in agent code. Dates are seeded relative to the day the command runs so phrases
like "12 days ago" and "in 3 weeks" stay accurate in a fresh install.

Watch-fors are structured entries; the ``id`` supports workflow tracking and the
``guidance`` is interpreted by the LLM. There is deliberately NO weight-specific
threshold rule in code -- the "> 3 lbs in 2 days" instruction is guidance text.
"""
from __future__ import annotations

from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from assistant_app import retrieval
from assistant_app.llm import LLMUnavailable
from assistant_app.models import (
    ChatSession,
    Flag,
    Message,
    Patient,
    PatientStatus,
)


class Command(BaseCommand):
    help = "Seed the demo patient (Jordan Ellis) and a small longitudinal history."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing patients before seeding.",
        )

    def handle(self, *args, **options):
        if options["reset"]:
            Patient.objects.all().delete()
            self.stdout.write(self.style.WARNING("Cleared existing patients."))

        if Patient.objects.exists():
            self.stdout.write(
                self.style.WARNING(
                    "A patient already exists; skipping. Use --reset to reseed."
                )
            )
            return

        today = timezone.now().date()
        last_visit = today - timedelta(days=12)
        upcoming = today + timedelta(days=21)

        profile = {
            "demographics": {"name": "Jordan Ellis", "age": 71, "sex": "male"},
            "conditions": [
                "Congestive heart failure with reduced ejection fraction",
                "Moderate COPD",
            ],
            "medications": [
                {"name": "Furosemide", "dose": "40 mg", "schedule": "daily"},
                {"name": "Lisinopril", "dose": "10 mg", "schedule": "daily"},
                {"name": "Metoprolol succinate", "dose": "50 mg", "schedule": "daily"},
                {"name": "Albuterol inhaler", "dose": "", "schedule": "as needed"},
                {"name": "Tiotropium inhaler", "dose": "", "schedule": "daily"},
            ],
            "baseline": {"weight_lbs": 182},
            "last_visit": {
                "date": last_visit.isoformat(),
                "type": "Routine CHF follow-up",
                "notes": (
                    "Weight stable, mild bilateral ankle edema, scattered wheezes, "
                    "no acute distress. Advised to continue daily weight checks and "
                    "report a gain of more than 3 lbs in 2 days."
                ),
            },
            "upcoming_appointment": {
                "date": upcoming.isoformat(),
                "type": "Cardiology follow-up",
            },
            "care_guidance": (
                "Daily weight checks. Maintain fluid and sodium guidance. Use "
                "inhalers as prescribed. Escalate rapid weight gain, worsening "
                "breathlessness, or chest pain."
            ),
            "watch_fors": [
                {
                    "id": "rapid_weight_gain",
                    "label": "Rapid weight gain",
                    "guidance": "Report weight gain over 3 lbs in 2 days.",
                },
                {
                    "id": "increased_swelling",
                    "label": "Increased swelling",
                    "guidance": "Report new or worsening swelling in the ankles or legs.",
                },
                {
                    "id": "worsening_shortness_of_breath",
                    "label": "Worsening shortness of breath",
                    "guidance": (
                        "Report worsening breathlessness, especially at rest or "
                        "when lying flat (orthopnea)."
                    ),
                },
                {
                    "id": "increased_wheeze",
                    "label": "Increased wheeze",
                    "guidance": "Report new or increased wheezing.",
                },
                {
                    "id": "productive_cough",
                    "label": "Productive cough",
                    "guidance": "Report a new productive cough or change in sputum.",
                },
                {
                    "id": "chest_pain",
                    "label": "Chest pain",
                    "guidance": "Report any chest pain or pressure promptly.",
                },
                {
                    "id": "confusion",
                    "label": "Confusion",
                    "guidance": "Report new confusion or mental fog.",
                },
                {
                    "id": "dizziness",
                    "label": "Dizziness",
                    "guidance": "Report new dizziness or lightheadedness.",
                },
            ],
        }

        # History messages and statuses are embedded, which needs the model;
        # seed all-or-nothing so a failure leaves no half-seeded patient.
        try:
            with transaction.atomic():
                patient = Patient.objects.create(profile_data=profile)
                self._seed_history(patient, today)
        except LLMUnavailable as exc:
            raise CommandError(
                f"Seeding needs the embedding model ({exc}). Set OPENAI_API_KEY."
            ) from exc
        self.stdout.write(self.style.SUCCESS(f"Created patient #{patient.pk}: Jordan Ellis"))
        self.stdout.write(self.style.SUCCESS("Seeded longitudinal history."))
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Open /chat/patients/{patient.pk}/ and /ops/dashboard/."
            )
        )

    def _seed_history(self, patient: Patient, today) -> None:
        """A prior, read-only session showing a resolved-over-time observation.

        Ankle swelling was reported ~60 days ago and later reported resolved
        ~14 days ago. This demonstrates that the newest clinical status
        (resolved) should win over an older 'present' mention during retrieval.
        """
        now = timezone.now()
        past_session = ChatSession.objects.create(
            patient=patient,
            title="Earlier check-in",
            summary=(
                "Patient reported mild ankle swelling that later resolved. No "
                "chest pain. Continued daily weights."
            ),
            ended_at=now - timedelta(days=14),
            workflow_type="open_conversation",
            chat_state={"last_summarized_sequence": 3},
        )

        m1 = Message.objects.create(
            session=past_session,
            role="patient",
            content="My ankles have been a bit swollen the last few days.",
            message_embedding=retrieval.embed_message(
                "My ankles have been a bit swollen the last few days."
            ),
            sequence_number=1,
        )
        m2 = Message.objects.create(
            session=past_session,
            role="assistant",
            content="Thanks for letting me know. I'll note the ankle swelling.",
            message_embedding=retrieval.embed_message(
                "Thanks for letting me know. I'll note the ankle swelling."
            ),
            sequence_number=2,
        )
        # The patient's own report is the evidence for the resolved status.
        resolved_text = "The swelling in my ankles has gone down completely."
        m3 = Message.objects.create(
            session=past_session,
            role="patient",
            content=resolved_text,
            message_embedding=retrieval.embed_message(resolved_text),
            sequence_number=3,
        )
        Message.objects.filter(pk__in=[m1.pk, m2.pk]).update(
            timestamp=now - timedelta(days=60)
        )
        Message.objects.filter(pk=m3.pk).update(timestamp=now - timedelta(days=14))

        s_present = PatientStatus.objects.create(
            patient=patient,
            session=past_session,
            source_message=m1,
            date=(today - timedelta(days=60)),
            reported_at=now - timedelta(days=60),
            clinical_time=(today - timedelta(days=61)),
            symptoms=["swelling"],
            watch_for_id="increased_swelling",
            symptom_embedding=retrieval.embed_symptom("swelling"),
            message_ids=[m1.pk],
            symptom_status="present",
            onset=f"{(today - timedelta(days=61)).isoformat()}T12:00",
            severity="mild",
            description="Mild bilateral ankle swelling reported ~2 months ago.",
        )
        s_resolved = PatientStatus.objects.create(
            patient=patient,
            session=past_session,
            source_message=m3,
            date=(today - timedelta(days=14)),
            reported_at=now - timedelta(days=14),
            clinical_time=(today - timedelta(days=15)),
            symptoms=["swelling"],
            watch_for_id="increased_swelling",
            symptom_embedding=retrieval.embed_symptom("swelling"),
            message_ids=[m3.pk],
            symptom_status="resolved",
            description="Ankle swelling reported resolved ~2 weeks ago.",
        )
