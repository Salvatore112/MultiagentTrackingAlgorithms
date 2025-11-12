from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.http import HttpRequest, HttpResponse


def redirect_to_setup(request: HttpRequest) -> HttpResponse:
    return redirect("/setup/")


urlpatterns: list = [
    path("admin/", admin.site.urls),
    path("", redirect_to_setup),
    path("setup/", include("simulations.urls")),
    path("simulation/", include("simulations.urls")),
]
