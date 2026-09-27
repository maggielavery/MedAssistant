"""Global runtime configuration for the Maggie Health Assistant.

The OpenAI credentials live here as module-level globals so a new operator can
run the prototype by pasting their own key below (or, preferably, exporting it
as an environment variable).

To run locally without touching your shell, replace the empty string on the
``OPENAI_API_KEY`` line with your key. Do NOT commit a real key.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# OpenAI configuration -- edit the fallbacks below or set env vars.
# ---------------------------------------------------------------------------
# Paste your key between the quotes to run without setting an env var:
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Chat / reasoning model used by the agents (Responses API).
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")

# Embedding model used for longitudinal retrieval.
OPENAI_EMBEDDING_MODEL: str = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)


def has_openai() -> bool:
    """Return True when a usable OpenAI key is configured.

    The key is required: the agents and embeddings have no offline fallbacks.
    """
    return bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())
