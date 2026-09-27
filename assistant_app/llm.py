"""Thin OpenAI wrapper used by every agent.

One place that knows about the OpenAI Responses + Embeddings APIs. There are
no offline fallbacks: a missing key, a failed request, or an unparseable
response raises ``LLMUnavailable`` so a turn fails visibly instead of
continuing on heuristic output.
"""
from __future__ import annotations

import json
import logging
import math
import re
from functools import lru_cache

from django.conf import settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The model or embeddings could not produce a usable result."""


def is_live() -> bool:
    """True when a real OpenAI key is configured."""
    return bool(getattr(settings, "OPENAI_API_KEY", "").strip())


@lru_cache(maxsize=1)
def _client():
    """Lazily build a cached OpenAI client."""
    if not is_live():
        raise LLMUnavailable("OPENAI_API_KEY is not configured.")
    from openai import OpenAI

    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _respond(system: str, user: str, temperature: float) -> str:
    try:
        resp = _client().responses.create(
            model=settings.OPENAI_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
    except LLMUnavailable:
        raise
    except Exception as exc:
        logger.warning("Model request failed: %s", exc)
        raise LLMUnavailable(f"Model request failed: {exc}") from exc
    return _extract_text(resp)


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------
def complete_json(system: str, user: str, *, temperature: float = 0.0) -> dict:
    """Return a JSON object from the model; raise if none can be parsed."""
    return _parse_json(_respond(system, user, temperature))


def complete_text(system: str, user: str, *, temperature: float = 0.0) -> str:
    """Return non-empty free-form text from the model."""
    text = _respond(system, user, temperature).strip()
    if not text:
        raise LLMUnavailable("Model returned an empty response.")
    return text


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def embed(text: str) -> list[float]:
    """Embed ``text`` with the configured embedding model."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        resp = _client().embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL, input=text
        )
    except LLMUnavailable:
        raise
    except Exception as exc:
        logger.warning("Embedding request failed: %s", exc)
        raise LLMUnavailable(f"Embedding request failed: {exc}") from exc
    return list(resp.data[0].embedding)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _extract_text(resp) -> str:
    """Pull text out of a Responses API result across SDK versions."""
    text = getattr(resp, "output_text", None)
    if text:
        return text
    chunks = []
    for item in getattr(resp, "output", None) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)
    return "".join(chunks)


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip().rstrip("`").strip()
    for candidate in (text, (re.search(r"\{.*\}", text, re.DOTALL) or [None])[0]):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMUnavailable("Model response was not a JSON object.")
