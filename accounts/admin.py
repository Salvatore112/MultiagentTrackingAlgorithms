from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, CustomAlgorithm


class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'is_staff', 'is_active')
    search_fields = ('username', 'email')


admin.site.register(User, CustomUserAdmin)


class CustomAlgorithmAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'created_at', 'is_active')
    list_filter = ('user', 'is_active', 'created_at')
    search_fields = ('name', 'user__username')
    readonly_fields = ('id', 'created_at', 'updated_at', 'module_name')


admin.site.register(CustomAlgorithm, CustomAlgorithmAdmin)
