import re

from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import User, CustomAlgorithm


class RegisterForm(UserCreationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm Password'})
    )
    
    class Meta:
        model = User
        fields = ('username',)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'


class CustomAlgorithmForm(forms.ModelForm):
    class Meta:
        model = CustomAlgorithm
        fields = ('name', 'description', 'file')
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Algorithm name'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Algorithm description (optional)'}),
            'file': forms.FileInput(attrs={'class': 'form-control', 'accept': '.py'}),
        }
    
    def clean_name(self):
        name = self.cleaned_data.get('name')
        if name:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                raise forms.ValidationError('Algorithm name must be a valid Python identifier (start with letter/underscore, contain only letters, numbers, underscores).')
        return name
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            try:
                content = file.read().decode('utf-8')
                if 'TrackingAlgorithm' not in content:
                    raise forms.ValidationError('Algorithm must import and inherit from TrackingAlgorithm')
                file.seek(0)
            except Exception as e:
                raise forms.ValidationError(f'Invalid Python file: {e}')
        return file


class RenameAlgorithmForm(forms.Form):
    new_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'New algorithm name'})
    )
    
    def clean_new_name(self):
        name = self.cleaned_data.get('new_name')
        if name:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                raise forms.ValidationError('Algorithm name must be a valid Python identifier (start with letter/underscore, contain only letters, numbers, underscores).')
        return name
