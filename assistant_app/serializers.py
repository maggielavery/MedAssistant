"""DRF serializers for API responses."""
from __future__ import annotations

from rest_framework import serializers

from .models import (
    ActionEvent,
    ChatSession,
    Flag,
    Message,
    Patient,
    PatientStatus,
)


class SessionListSerializer(serializers.ModelSerializer):
    writable = serializers.BooleanField(source="is_writable", read_only=True)

    class Meta:
        model = ChatSession
        fields = [
            "id",
            "title",
            "summary",
            "workflow_type",
            "started_at",
            "ended_at",
            "writable",
        ]


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = [
            "id",
            "role",
            "content",
            "intent",
            "message_embedding",
            "flag_ids",
            "patient_status_ids",
            "action",
            "sequence_number",
            "timestamp",
        ]


class PatientStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientStatus
        fields = [
            "id",
            "symptoms",
            "watch_for_id",
            "symptom_embedding",
            "message_ids",
            "symptom_status",
            "onset",
            "severity",
            "description",
            "missing_fields",
            "date",
            "reported_at",
            "clinical_time",
            "source_message",
            "session",
            "flags_raised",
        ]


class FlagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Flag
        fields = [
            "id",
            "patient",
            "patient_status",
            "description",
            "evidence",
            "confidence",
            "status",
            "created_at",
            "updated_at",
        ]


class ActionEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ActionEvent
        fields = [
            "id",
            "patient",
            "session",
            "action_type",
            "request_text",
            "result",
            "follow_up_details",
            "created_at",
            "updated_at",
        ]


class PatientSerializer(serializers.ModelSerializer):
    name = serializers.CharField(source="display_name", read_only=True)

    class Meta:
        model = Patient
        fields = ["id", "name", "profile_data", "created_at", "updated_at"]
