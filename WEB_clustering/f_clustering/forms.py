from .models import Projects

from django import forms
from django.forms import Textarea
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User

class AuthUserForm(AuthenticationForm, forms.ModelForm):
	class Meta:
		model = User
		fields = ('username', 'password')
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		for field in self.fields:
			self.fields[field].widget.attrs['class'] = 'form-control'

class RegisterUserForm(forms.ModelForm):
	class Meta:
		model = User
		# print(model.field)
		fields = ('username', 'password', 'email')

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		for field in self.fields:
			self.fields[field].widget.attrs['class'] = 'form-control'

	def save(self, commit=True):
		user = super().save(commit=False)
		user.set_password(self.cleaned_data["password"])
		if commit:
			user.save()
		return user

class ProjectsForm(forms.ModelForm):
 
	class Meta:
		model = Projects
		fields = ['name', 'attach']

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		for field in self.fields:
			self.fields[field].widget.attrs['class'] = 'form-control'