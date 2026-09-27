"""Whole-session summarization + title generation.

Runs only when the patient explicitly ends a chat, or when the session has been
idle for more than the configured threshold. Key properties:

    * Idempotent: skips when no new messages arrived since the last summary.
    * Whole-session: summarizes original messages (batched if large), never
      re-compresses a previous summary.
    * Safe: re-checks that no message arrived during summarization before it
      commits. Inactivity is judged from the latest message timestamp, not the
      session row's ``updated_at`` (which the summary write itself bumps).
    * No offline fallback: if the model is unavailable, ``llm.LLMUnavailable``
      propagates and nothing is written, so a later run retries.
"""
from __future__ import annotations

from django.db.models import Max

from . import llm
from .models import ChatSession, Message

_BATCH_SIZE = 40


def latest_sequence(session: ChatSession) -> int:
    return session.messages.aggregate(m=Max("sequence_number"))["m"] or 0


def needs_summary(session: ChatSession) -> bool:
    last_summarized = session.chat_state.get("last_summarized_sequence", 0)
    return latest_sequence(session) > last_summarized


def summarize_session(session: ChatSession, *, reason: str = "manual") -> bool:
    """Summarize the session. Returns True when a summary was written."""
    if not needs_summary(session):
        return False

    seq_at_start = latest_sequence(session)
    messages = list(session.messages.order_by("sequence_number"))
    if not messages:
        return False

    summary_text = _summarize_messages(session, messages)
    title = _make_title(session, messages, summary_text)

    # Re-check that nothing arrived while we were summarizing.
    session.refresh_from_db()
    if latest_sequence(session) != seq_at_start:
        return False  # new message arrived; skip committing a stale summary

    session.summary = summary_text
    if title:
        session.title = title
    chat_state = dict(session.chat_state or {})
    chat_state["last_summarized_sequence"] = seq_at_start
    chat_state["last_summary_reason"] = reason
    session.chat_state = chat_state
    session.save(update_fields=["summary", "title", "chat_state", "updated_at"])
    return True


def _summarize_messages(session: ChatSession, messages: list[Message]) -> str:
    """Summarize the whole session, batching original messages if large."""
    if len(messages) <= _BATCH_SIZE:
        return _summary_call(_render_messages(messages))

    partials = []
    for i in range(0, len(messages), _BATCH_SIZE):
        batch = messages[i : i + _BATCH_SIZE]
        partials.append(_summary_call(_render_messages(batch), batch=True))
    combined = "\n\n".join(f"Batch {i+1}: {p}" for i, p in enumerate(partials))
    system = (
        "Combine these ordered batch summaries of one patient chat session into a "
        "single readable clinical summary. Keep source message references. Capture "
        "concerns, meaningful positives and negatives, changes over time, recorded "
        "actions, and unresolved questions."
    )
    return llm.complete_text(system, combined)


def _summary_call(rendered: str, *, batch: bool = False) -> str:
    scope = "batch of messages" if batch else "chat session"
    system = (
        f"Summarize this {scope} between a patient and a health assistant into a "
        "readable clinical summary (plain prose, not JSON). Capture concerns, "
        "meaningful positives and negatives, changes over time, recorded actions, "
        "unresolved questions, and cite source message #sequence numbers."
    )
    return llm.complete_text(system, rendered)


def _make_title(
    session: ChatSession, messages: list[Message], summary_text: str
) -> str:
    system = (
        "Generate a short (max 8 words) sidebar title for this patient chat "
        "session. Return only the title text, no quotes."
    )
    first_patient = next((m for m in messages if m.role == "patient"), None)
    seed = first_patient.content if first_patient else summary_text
    return llm.complete_text(system, seed).strip().strip('"')[:200]


def _render_messages(messages: list[Message]) -> str:
    return "\n".join(
        f"#{m.sequence_number} {m.role}: {m.content}" for m in messages
    )
