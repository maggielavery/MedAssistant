"""API URL routes."""
from django.urls import path

from . import views

urlpatterns = [
    # Patient-facing
    path("patients/<int:patient_id>/sessions/", views.session_list),
    path("patients/<int:patient_id>/sessions/new/", views.session_new),
    path("sessions/<int:session_id>/messages/", views.session_messages),
    path("sessions/<int:session_id>/chat/", views.session_chat),
    path("sessions/<int:session_id>/end/", views.session_end),
    # Admin / ops
    path("admin/patients/", views.admin_patients),
    path("admin/patients/<int:patient_id>/dashboard/", views.admin_dashboard),
    path("admin/database-tables/", views.admin_database_tables),
    path("admin/alerts/<str:kind>/<int:alert_id>/", views.admin_alert_detail),
    path("admin/flags/<int:flag_id>/", views.admin_flag_update),
    path("admin/actions/<int:action_id>/", views.admin_action_update),
]
