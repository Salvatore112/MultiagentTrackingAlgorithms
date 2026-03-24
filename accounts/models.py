import os
import uuid
import importlib.util
import sys

from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import FileExtensionValidator


class User(AbstractUser):
    email = models.EmailField(unique=True, blank=True, null=True)
    
    def __str__(self):
        return self.username


class CustomAlgorithm(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='algorithms')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    file = models.FileField(
        upload_to='algorithms/',
        validators=[FileExtensionValidator(allowed_extensions=['py'])]
    )
    module_name = models.CharField(max_length=200, blank=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        unique_together = ['user', 'name']
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} (by {self.user.username})"
    
    def filename(self):
        return os.path.basename(self.file.name)
    
    def save(self, *args, **kwargs):
        self.module_name = f"custom_algo_{self.user.id}_{self.id}_{self.name}"
        import re
        self.module_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.module_name)
        
        super().save(*args, **kwargs)
        
        self.load_module()
    
    def load_module(self):
        try:
            file_path = self.file.path
            module_name = self.module_name
            
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            return module
        except Exception as e:
            print(f"Error loading algorithm {self.name}: {e}")
            return None
    
    def get_algorithm_class(self):
        module = self.load_module()
        if module:
            import inspect
            from algorithms.tracking_algorithm import TrackingAlgorithm
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, TrackingAlgorithm) and obj != TrackingAlgorithm:
                    return obj
        return None
    
    def delete(self, *args, **kwargs):
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]
        
        if self.file:
            self.file.delete(save=False)
        super().delete(*args, **kwargs)
