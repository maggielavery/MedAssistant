"""Summarize sessions that have been idle for more than the threshold.

Run this on a schedule during development, e.g. every 30-60 seconds:

    # simple loop (documented option):
    watch -n 30 "python manage.py summarize_idle_sessions"

    # or via cron:
    * * * * * cd /path/to/project && python manage.py summarize_idle_sessions

Inactivity is measured from the latest MESSAGE timestamp (not the session row's
``updated_at``, which a summary write would bump). An idle summary does not end
the session; if the patient resumes, the next idle period re-summarizes.
"""
from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Max
from django.utils import timezone

from assistant_app import summarizer
from assistant_app.llm import LLMUnavailable
from assistant_app.models import ChatSession


class Command(BaseCommand):
    help = "Summarize sessions idle for more than SESSION_IDLE_SECONDS."

    def add_arguments(self, parser):
        parser.add_argument(
            "--seconds",
            type=int,
            default=getattr(settings, "SESSION_IDLE_SECONDS", 60),
            help="Idle threshold in seconds (default from settings).",
        )

    def handle(self, *args, **options):
        threshold = options["seconds"]
        cutoff = timezone.now() - timedelta(seconds=threshold)

        # Consider every session (ended or active) and judge idleness from the
        # latest message timestamp, not updated_at.
        sessions = ChatSession.objects.annotate(
            last_msg=Max("messages__timestamp")
        ).filter(last_msg__isnull=False, last_msg__lt=cutoff)

        summarized = 0
        for session in sessions:
            try:
                wrote = summarizer.summarize_session(session, reason="idle")
            except LLMUnavailable as exc:
                self.stderr.write(f"Skipped session #{session.pk}: {exc}")
                continue
            if wrote:
                summarized += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Summarized session #{session.pk}: {session.title!r}"
                    )
                )
        self.stdout.write(
            self.style.SUCCESS(
                f"Checked {sessions.count()} idle session(s); summarized {summarized}."
            )
        )
