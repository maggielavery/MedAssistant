"""WSGI config for the continuum project."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "continuum.settings")

application = get_wsgi_application()
