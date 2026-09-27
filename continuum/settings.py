"""Django settings for the Maggie Health Assistant prototype."""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load a local .env if present (optional; config.py also reads env vars directly).
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

# Make the project-root config module importable (config.py holds OpenAI keys).
sys.path.insert(0, str(BASE_DIR))
import config  # noqa: E402  (import after sys.path tweak, intentional)

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-insecure-secret-key")
DEBUG = os.getenv("DJANGO_DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",")

# The prototype has no logins, sessions, or Django admin, so the auth, admin,
# contenttypes, sessions, and messages apps (and their tables) are not
# installed. The database holds only the assistant_app tables plus
# django_migrations.
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "rest_framework",
    "assistant_app",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "continuum.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "continuum.wsgi.application"
ASGI_APPLICATION = "continuum.asgi.application"

# SQLite is sufficient for local dev; the models stay PostgreSQL-compatible.
if os.getenv("DATABASE_URL"):
    import dj_database_url  # type: ignore

    DATABASES = {
        "default": dj_database_url.parse(
            os.getenv("DATABASE_URL"), conn_max_age=600
        ),
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Open, unauthenticated JSON API (no django.contrib.auth installed).
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
}

# ---------------------------------------------------------------------------
# OpenAI + retrieval / summarization tunables (sourced from config.py).
# ---------------------------------------------------------------------------
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_MODEL = config.OPENAI_MODEL
OPENAI_EMBEDDING_MODEL = config.OPENAI_EMBEDDING_MODEL

# Idle threshold for the summarizer: sessions idle > 60s get summarized.
SESSION_IDLE_SECONDS = int(os.getenv("SESSION_IDLE_SECONDS", "60"))
# Longitudinal retrieval controls.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
# Bounded context sizes for prompts.
RECENT_MESSAGE_WINDOW = int(os.getenv("RECENT_MESSAGE_WINDOW", "12"))
