from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('profile/', views.profile_view, name='profile'),
    path('algorithms/upload/', views.upload_algorithm, name='upload_algorithm'),
    path('algorithms/<uuid:algorithm_id>/delete/', views.delete_algorithm, name='delete_algorithm'),
    path('algorithms/<uuid:algorithm_id>/rename/', views.rename_algorithm, name='rename_algorithm'),
]
