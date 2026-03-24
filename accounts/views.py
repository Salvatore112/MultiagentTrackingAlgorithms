from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages


from .forms import RegisterForm, LoginForm, CustomAlgorithmForm, RenameAlgorithmForm
from .models import CustomAlgorithm


def register_view(request):
    if request.user.is_authenticated:
        return redirect('simulations:setup')
    
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('simulations:setup')
    else:
        form = RegisterForm()
    
    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('simulations:setup')
    
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {username}!')
                next_url = request.GET.get('next', 'simulations:setup')
                return redirect(next_url)
        messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    
    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('simulations:setup')


@login_required
def profile_view(request):
    algorithms = request.user.algorithms.filter(is_active=True)
    return render(request, 'accounts/profile.html', {
        'algorithms': algorithms,
        'user': request.user,
    })


@login_required
def upload_algorithm(request):
    if request.method == 'POST':
        form = CustomAlgorithmForm(request.POST, request.FILES)
        if form.is_valid():
            algorithm = form.save(commit=False)
            algorithm.user = request.user
            algorithm.save()
            
            algorithm_class = algorithm.get_algorithm_class()
            if algorithm_class:
                messages.success(request, f'Algorithm "{algorithm.name}" uploaded and validated successfully!')
            else:
                messages.warning(request, f'Algorithm "{algorithm.name}" uploaded but could not find a TrackingAlgorithm subclass.')
            
            return redirect('accounts:profile')
    else:
        form = CustomAlgorithmForm()
    
    return render(request, 'accounts/upload_algorithm.html', {'form': form})


@login_required
def delete_algorithm(request, algorithm_id):
    algorithm = get_object_or_404(CustomAlgorithm, id=algorithm_id, user=request.user)
    algorithm_name = algorithm.name
    algorithm.delete()
    messages.success(request, f'Algorithm "{algorithm_name}" deleted successfully!')
    return redirect('accounts:profile')


@login_required
def rename_algorithm(request, algorithm_id):
    algorithm = get_object_or_404(CustomAlgorithm, id=algorithm_id, user=request.user)
    
    if request.method == 'POST':
        form = RenameAlgorithmForm(request.POST)
        if form.is_valid():
            new_name = form.cleaned_data['new_name']
            algorithm.name = new_name
            algorithm.save()
            messages.success(request, f'Algorithm renamed to "{new_name}"!')
            return redirect('accounts:profile')
    else:
        form = RenameAlgorithmForm(initial={'new_name': algorithm.name})
    
    return render(request, 'accounts/rename_algorithm.html', {
        'form': form,
        'algorithm': algorithm,
    })
