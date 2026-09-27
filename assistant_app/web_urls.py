"""Web page URL routes."""
from django.urls import path

from . import views

urlpatterns = [
    path("chat/patients/<int:patient_id>/", views.chat_page, name="chat_page"),
    path("ops/dashboard/", views.ops_dashboard_root, name="ops_dashboard"),
    path(
        "ops/dashboard/patients/<int:patient_id>/",
        views.ops_dashboard_patient,
        name="ops_dashboard_patient",
    ),
]
