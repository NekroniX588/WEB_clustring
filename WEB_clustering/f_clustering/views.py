from .models import Projects
from .forms import AuthUserForm, RegisterUserForm, ProjectsForm

from django.shortcuts import render
from django.urls import reverse, reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView, LogoutView
# Create your views here.

def main(request):
	context = {
		'name': 'Pidr'
	}
	template = 'main.html'
	return render(request, template, context)

class SpeechLoginView(LoginView):
	template_name = 'login.html'
	form_class = AuthUserForm
	success_url = reverse_lazy('main')
	def get_success_url(self):
		return self.success_url

class SpeechRegisterView(CreateView):
	model = User
	template_name = 'register.html'
	form_class = RegisterUserForm
	success_url = reverse_lazy('main')
	def form_valid(self, form):
		form_valid = super().form_valid(form)
		username = form.cleaned_data["username"]
		password = form.cleaned_data["password"]
		auth_user = authenticate(username=username, password=password)
		login(self.request, auth_user)
		return form_valid
		
class SpeechLogout(LogoutView):
	next_page = reverse_lazy('main')

class SpeechProjectsView(ListView):
	model = Projects
	template_name = 'projects.html'
	def get_context_data(self, **kwargs):
		kwargs['data'] = Projects.objects.filter(author = self.request.user).order_by('-date')
		return super().get_context_data(**kwargs)

class SpeechProjectsCreate(CreateView): # новый
	model = Projects
	form_class = ProjectsForm
	template_name = 'projects_add.html'
	success_url = reverse_lazy('projects')
	def form_valid(self, form):
		self.object = form.save(commit=False)
		self.object.author = self.request.user
		self.object.save()
		return super().form_valid(form)