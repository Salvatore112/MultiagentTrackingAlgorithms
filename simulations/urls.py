from django.urls import path
from . import views

urlpatterns: list = [
    path("", views.setup_view, name="setup"),
    path("results/", views.results_view, name="results"),
]
