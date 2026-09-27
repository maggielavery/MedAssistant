"""Patient-scoped semantic retrieval and embedding-based confidence scores.

Retrieval gives each status one reproducible confidence score per query
(one query per symptom in play, plus the checkup item about to be asked),
matches on that score or an exact watch-for, then expands each matched
symptom to its full history (see ``search_statuses``):

    0.7 * cosine(query keyword embedding, status symptom embedding)
    + 0.3 * cosine(new-message embedding, status source-message embedding)

For a checkup-item query the 0.3 term uses the item's "label. guidance" text
in place of the new message.

Flag confidence uses the same weighting against the matched watch-for:

    0.7 * cosine(extracted-keyword embedding, watch-for label embedding)
    + 0.3 * cosine(new-message embedding, watch-for label + guidance embedding)

Status symptom embeddings and message embeddings are persisted as JSON vectors.
Existing rows are lazily backfilled when they are first considered for search.
"""
from __future__ import annotations

from functools import lru_cache

from . import llm
from .models import Message, Patient, PatientStatus

MATCH_THRESHOLD = 0.6
SYMPTOM_WEIGHT = 0.7
MESSAGE_WEIGHT = 0.3
# Most recent rows of one symptom's history added by history expansion.
HISTORY_LIMIT = 8


def weighted_score(symptom_similarity: float, message_similarity: float) -> float:
    """Combine the two similarity components used by retrieval and flags."""
    return SYMPTOM_WEIGHT * symptom_similarity + MESSAGE_WEIGHT * message_similarity


def embed_message(text: str) -> list[float]:
    """Embed one persisted chat message."""
    return llm.embed(text)


def embed_symptom(symptom: str) -> list[float]:
    """Embed the concise status symptom label."""
    return llm.embed(symptom)


def _ensure_message_embedding(message: Message | None) -> list[float]:
    if not message:
        return []
    if message.message_embedding:
        return list(message.message_embedding)
    embedding = embed_message(message.content)
    Message.objects.filter(pk=message.pk).update(message_embedding=embedding)
    message.message_embedding = embedding
    return embedding


def _ensure_symptom_embedding(status: PatientStatus) -> list[float]:
    if status.symptom_embedding:
        return list(status.symptom_embedding)
    embedding = embed_symptom(status.label)
    PatientStatus.objects.filter(pk=status.pk).update(symptom_embedding=embedding)
    status.symptom_embedding = embedding
    return embedding


def symptom_key(status: PatientStatus) -> tuple[str, str]:
    """Group a status with the rest of its symptom's history.

    Statuses aligned to a profile watch-for group by that id, so different
    wordings ("swelling", "puffy ankles", "leg edema") share one history;
    others group by their exact normalized label.
    """
    if status.watch_for_id:
        return ("watch_for", status.watch_for_id)
    return ("label", status.label.strip().lower())


def search_statuses(
    patient: Patient,
    source_message: Message,
    *,
    queries: list[dict],
    top_k: int = 6,
    use_watch_for: bool = True,
    expand_history: bool = True,
) -> list[dict]:
    """Return patient-scoped statuses relevant to the queries.

    Each query is ``{"label", "keywords", "watch_for_id"?,
    "context_message_id"?, "context_text"?}`` and is scored on its own, so one
    symptom's words never dilute another's:

        0.7 * cosine(query keywords, status symptom label)
        + 0.3 * cosine(query context, status source message)

    The context is the query's own source message (where that symptom was
    first reported, so a short follow-up answer still matches its history),
    else the new patient message; a checkup-item query uses its
    ``context_text``. A status keeps its best score across queries.

    Matching, in order:
      1. ``semantic``: score > MATCH_THRESHOLD.
      2. ``watch_for``: same profile watch-for as the query (exact), whatever
         the score (``use_watch_for``).
    The ``top_k`` strongest matches are kept. Then ``history`` expansion adds
    the rest of each matched symptom's history (same watch-for, else same
    label; at most HISTORY_LIMIT most recent), so a later "resolved" report
    is never lost just because it was worded differently
    (``expand_history``). Scores rank results but never exclude history.
    Results are returned in clinical-time order.
    """
    prepared = []
    for query in queries:
        keyword_text = " ".join(k for k in query.get("keywords", []) if k)
        if not keyword_text:
            continue
        context_text = query.get("context_text")
        context_message = (
            Message.objects.filter(pk=query["context_message_id"]).first()
            if query.get("context_message_id")
            else None
        )
        if context_text:
            context_embedding = list(_embed_profile_text(context_text))
        else:
            context_embedding = _ensure_message_embedding(context_message or source_message)
        prepared.append((query, embed_symptom(keyword_text), context_embedding))
    if not prepared:
        return []

    scored: dict[int, dict] = {}
    statuses = list(
        PatientStatus.objects.filter(patient=patient).select_related("source_message")
    )
    for status in statuses:
        symptom_embedding = _ensure_symptom_embedding(status)
        source_embedding = _ensure_message_embedding(status.source_message)
        if not symptom_embedding or not source_embedding:
            continue
        for query, keyword_embedding, context_embedding in prepared:
            symptom_similarity = llm.cosine_similarity(keyword_embedding, symptom_embedding)
            message_similarity = llm.cosine_similarity(context_embedding, source_embedding)
            confidence = weighted_score(symptom_similarity, message_similarity)
            if confidence > MATCH_THRESHOLD:
                match = "semantic"
            elif (
                use_watch_for
                and query.get("watch_for_id")
                and status.watch_for_id == query["watch_for_id"]
            ):
                match = "watch_for"
            else:
                match = None
            current = scored.get(status.pk)
            rank = (match is not None, confidence)
            if current is None or rank > (current["match"] is not None, current["confidence"]):
                scored[status.pk] = {
                    "status": status,
                    "confidence": confidence,
                    "symptom_similarity": symptom_similarity,
                    "message_similarity": message_similarity,
                    "query": query.get("label", ""),
                    "match": match,
                }

    matched = sorted(
        (hit for hit in scored.values() if hit["match"]),
        key=lambda hit: hit["confidence"],
        reverse=True,
    )[:top_k]

    if expand_history and matched:
        chosen = {hit["status"].pk for hit in matched}
        keys = {symptom_key(hit["status"]) for hit in matched}
        for key in keys:
            family = sorted(
                (s for s in statuses if symptom_key(s) == key and s.pk in scored),
                key=_clinical_order,
                reverse=True,
            )[:HISTORY_LIMIT]
            for status in family:
                if status.pk not in chosen:
                    chosen.add(status.pk)
                    matched.append({**scored[status.pk], "match": "history"})

    matched.sort(key=lambda hit: _clinical_order(hit["status"]))
    return matched


def _clinical_order(status: PatientStatus):
    return (
        status.clinical_time or status.date,
        status.reported_at or status.created_at,
    )


def message_excerpt(message: Message | None, length: int = 240) -> str:
    if not message:
        return ""
    text = message.content.strip()
    return text if len(text) <= length else text[: length - 1] + "…"


@lru_cache(maxsize=256)
def _embed_profile_text(text: str) -> tuple[float, ...]:
    """Watch-for text is static profile config, so embed it once per process."""
    return tuple(llm.embed(text))


def score_watch_for_match(
    keywords: list[str],
    message_embedding: list[float],
    watch_for: dict,
) -> dict:
    """Score an observation against a watch-for with the retrieval weighting.

    The score is computed from embeddings only; no LLM output feeds into it.
    """
    keyword_text = " ".join(keyword for keyword in keywords if keyword)
    label = watch_for.get("label", "")
    guidance = watch_for.get("guidance", "")
    symptom_similarity = llm.cosine_similarity(
        embed_symptom(keyword_text) if keyword_text else [],
        list(_embed_profile_text(label)) if label else [],
    )
    message_similarity = llm.cosine_similarity(
        list(message_embedding or []),
        list(_embed_profile_text(f"{label}. {guidance}".strip())),
    )
    return {
        "confidence": round(weighted_score(symptom_similarity, message_similarity), 3),
        "symptom_similarity": round(symptom_similarity, 3),
        "message_similarity": round(message_similarity, 3),
    }


def build_timeline(
    patient: Patient,
    source_message: Message,
    *,
    queries: list[dict],
) -> list[dict]:
    """Build a source-linked timeline from semantic retrieval hits."""
    return [
        {**_timeline_entry(
            hit["status"],
            hit["confidence"],
            hit["symptom_similarity"],
            hit["message_similarity"],
        ), "matched_query": hit["query"], "match": hit["match"]}
        for hit in search_statuses(patient, source_message, queries=queries)
    ]


def _timeline_entry(
    status: PatientStatus,
    confidence: float,
    symptom_similarity: float,
    message_similarity: float,
) -> dict:
    reported = status.reported_at or status.created_at
    return {
        "status_id": status.pk,
        "symptom": status.label,
        "watch_for_id": status.watch_for_id,
        "symptom_status": status.symptom_status,
        "onset": status.onset,
        "severity": status.severity,
        "description": status.description,
        "reported_time": reported.isoformat() if reported else None,
        "clinical_time": status.clinical_time.isoformat() if status.clinical_time else None,
        "source_message_id": status.source_message_id,
        "message_ids": status.message_ids,
        "source_excerpt": message_excerpt(status.source_message),
        "symptom_similarity": round(symptom_similarity, 3),
        "message_similarity": round(message_similarity, 3),
        "confidence": round(confidence, 3),
    }
