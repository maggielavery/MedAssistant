"""ASGI config for the continuum project."""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "continuum.settings")

application = get_asgi_application()
