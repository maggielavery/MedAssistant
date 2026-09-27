"""Database schema for the Maggie Health Assistant.

The schema follows the take-home spec. Only provenance, longitudinal-retrieval,
and session-title fields are added on top of the original models. ``profile_data``
is the single source of patient-specific configuration (conditions, medications,
watch-fors, guidance) so replacing the patient never requires touching agent code.
"""
from __future__ import annotations

from django.db import models
from django.utils import timezone


class Patient(models.Model):
    """A patient. All patient-specific config lives in ``profile_data``."""

    profile_data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:  # pragma: no cover - display helper
        demographics = self.profile_data.get("demographics", {})
        return demographics.get("name", f"Patient {self.pk}")

    @property
    def display_name(self) -> str:
        return self.profile_data.get("demographics", {}).get(
            "name", f"Patient {self.pk}"
        )

    @property
    def watch_fors(self) -> list[dict]:
        return self.profile_data.get("watch_fors", [])


class ChatSession(models.Model):
    """A single chat conversation. Only the latest un-ended session is writable."""

    patient = models.ForeignKey(
        Patient, related_name="sessions", on_delete=models.CASCADE
    )
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    workflow_type = models.CharField(max_length=64, default="open_conversation")
    # Compact working memory: checkup progress, pending symptoms, summary checkpoint.
    chat_state = models.JSONField(default=dict)
    # Human-readable sidebar title (new).
    title = models.CharField(max_length=200, blank=True, default="")
    # Readable whole-session summary text (NOT JSON).
    summary = models.TextField(blank=True, default="")
    action_ids = models.JSONField(default=list)
    patient_status_ids = models.JSONField(default=list)
    flag_ids = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:  # pragma: no cover
        return self.title or f"Session {self.pk}"

    @property
    def is_writable(self) -> bool:
        return self.ended_at is None

    @property
    def last_message_at(self):
        last = self.messages.order_by("-sequence_number").first()
        return last.timestamp if last else self.started_at


class ActionEvent(models.Model):
    """A mocked operational action (appointment or doctor-message request)."""

    ACTION_TYPES = (
        ("appointment_request", "appointment_request"),
        ("appointment_change", "appointment_change"),
        ("contact_doctor", "contact_doctor"),
    )

    patient = models.ForeignKey(
        Patient, related_name="actions", on_delete=models.CASCADE
    )
    session = models.ForeignKey(
        ChatSession, related_name="actions", on_delete=models.CASCADE
    )
    action_type = models.CharField(max_length=64, choices=ACTION_TYPES)
    # The patient's original wording; never rewritten.
    request_text = models.TextField()
    # Mock result, e.g. {"status": "sent_to_care_team", "booked": false,
    # "delivered": false}.
    result = models.JSONField(default=dict)
    # Details the patient added later in the same chat:
    # [{"message_id", "detail", "quote", "added_at"}].
    follow_up_details = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]


class Message(models.Model):
    """A single message in a session. Patient messages are primary evidence."""

    ROLE_CHOICES = (
        ("patient", "patient"),
        ("assistant", "assistant"),
        ("system", "system"),
    )

    session = models.ForeignKey(
        ChatSession, related_name="messages", on_delete=models.CASCADE
    )
    action = models.ForeignKey(
        ActionEvent,
        related_name="messages",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    intent = models.CharField(max_length=64, blank=True, default="")
    # Stored for every persisted message; used as source evidence in retrieval.
    message_embedding = models.JSONField(default=list)
    flag_ids = models.JSONField(default=list)
    patient_status_ids = models.JSONField(default=list)
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    sequence_number = models.PositiveIntegerField()

    class Meta:
        unique_together = ("session", "sequence_number")
        ordering = ["sequence_number"]

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.role}] {self.content[:40]}"


class PatientStatus(models.Model):
    """One normalized symptom / clinical observation. The atomic retrieval unit."""

    SYMPTOM_STATUS_CHOICES = (
        ("present", "present"),
        ("absent", "absent"),
        ("historical", "historical"),
        ("uncertain", "uncertain"),
        ("resolved", "resolved"),
    )

    patient = models.ForeignKey(
        Patient, related_name="statuses", on_delete=models.CASCADE
    )
    session = models.ForeignKey(
        ChatSession, related_name="statuses", on_delete=models.CASCADE
    )
    # Primary evidence message (new).
    source_message = models.ForeignKey(
        Message,
        related_name="derived_statuses",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    date = models.DateField(default=timezone.now)
    # When the observation was reported (new).
    reported_at = models.DateTimeField(null=True, blank=True)
    # When the described symptom/change actually applied, if known (new).
    clinical_time = models.DateField(null=True, blank=True)
    # One concise symptom label (multi-word when clinically useful).
    symptoms = models.JSONField(default=list)
    # Profile watch-for this observation aligns with ("" if none). Gives
    # retrieval an exact way to group a symptom's history across wordings.
    watch_for_id = models.CharField(max_length=64, blank=True, default="")
    # Embedding of the concise symptom label, used for semantic retrieval.
    symptom_embedding = models.JSONField(default=list)
    # All patient Message IDs that supplied details for this observation.
    message_ids = models.JSONField(default=list)
    symptom_status = models.CharField(
        max_length=16, choices=SYMPTOM_STATUS_CHOICES, default="present"
    )
    # Approximate ISO datetime (YYYY-MM-DDTHH:MM) resolved from the report.
    onset = models.CharField(max_length=128, blank=True, default="")
    severity = models.CharField(max_length=128, blank=True, default="")
    description = models.TextField(blank=True, default="")
    # Details still unknown when a flagged observation is persisted early;
    # emptied when the follow-up completes the same row.
    missing_fields = models.JSONField(default=list)
    actions_requested = models.JSONField(default=list)
    actions_taken = models.JSONField(default=list)
    flags_raised = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover
        label = self.symptoms[0] if self.symptoms else "symptom"
        return f"{label} ({self.symptom_status})"

    @property
    def label(self) -> str:
        if self.symptoms:
            return str(self.symptoms[0])
        return "symptom"


class Flag(models.Model):
    """A care-team attention flag raised from an observation."""

    STATUS_CHOICES = (
        ("NEW", "NEW"),
        ("OPEN", "OPEN"),
        ("ACKNOWLEDGED", "ACKNOWLEDGED"),
        ("RESOLVED", "RESOLVED"),
        ("DISMISSED", "DISMISSED"),
    )

    patient = models.ForeignKey(
        Patient, related_name="flags", on_delete=models.CASCADE
    )
    patient_status = models.ForeignKey(
        PatientStatus, related_name="flags", on_delete=models.CASCADE
    )
    description = models.TextField()
    # Source message IDs, quotations, relevant profile guidance, timing.
    evidence = models.JSONField(default=dict)
    confidence = models.FloatField(default=0.5)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="NEW")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover
        return f"Flag {self.pk}: {self.description[:40]}"
