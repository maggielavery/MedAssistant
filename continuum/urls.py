"""Root URL configuration."""
from django.urls import include, path

urlpatterns = [
    path("api/", include("assistant_app.api_urls")),
    path("", include("assistant_app.web_urls")),
]
