from django.urls import path
from . import views

urlpatterns = [
    path("", views.setup_view, name="setup"),
    path("run/", views.run_simulation_view, name="run_simulation"),
    path("run_algorithm/", views.run_algorithm_view, name="run_algorithm"),
    path("results/", views.simulation_results_view, name="simulation_results"),
]
