import os 

from .models import Projects
from .forms import AuthUserForm, RegisterUserForm, ProjectsForm

from core.reader import Reader
from core.const import Const
from core.fastclustering import Fast_Clusters
from core.clustering import Clusters
from core.i_merge import IMerger
from core.Subclusterring import Subclusters

from django.shortcuts import render, redirect, HttpResponse
from django.urls import reverse, reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView, LogoutView
from django.core.files import File
# Create your views here.


reader = Reader()

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

		f = open('./settings.yaml', 'r')
		self.object.settings = File(f, name=os.path.basename(f.name))

		self.object.stage = 1

		self.object.save()
		if reader.read('./df/'+self.object.attach.url) is not None:
			self.object.status = True
		else:
			self.object.status = False

		return super().form_valid(form)

def table(request, pk):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df'+data.attach.url)
	geeks_object = df.to_html()
	return HttpResponse(geeks_object)

def statistic(request, pk):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)
	text, df = reader.statistic(df)
	data.comments += text
	data.save()
	reader.write(df, './df/'+data.attach.url)
	return redirect(reverse('start_project', args=(pk,)))

def project_start(request,pk):
	data = Projects.objects.get(pk=pk)

	df = reader.read('./df/'+data.attach.url)

	context = {
		'data': data,
	}
	template = 'project_start.html'
	return render(request, template, context)

def const_start(request,pk):
	if request.method =='GET':
		data = Projects.objects.get(pk=pk)
		data.stage = 2
		data.save()
		const = Const('./settings/'+data.settings.url)

		context = {
			'data': data,
			'settings': const.config,
		}
		template = 'const_start.html'

		return render(request, template, context)
	elif request.method =='POST':
		data = Projects.objects.get(pk=pk)
		const = Const('./settings/'+data.settings.url)

		for domen in const.config:
			for key in request.POST.keys():
				if key == 'min_dif_0':
					const.config['isolated_cluster']['min_dif'][0] = float(request.POST[key])
				elif key == 'min_dif_1':
					const.config['isolated_cluster']['min_dif'][1] = float(request.POST[key])
				elif key == 'min_diff_0':
					const.config['conturs']['min_diff'][0] = float(request.POST[key])
				elif key == 'min_diff_1':
					const.config['conturs']['min_diff'][1] = float(request.POST[key])
				if key in const.config[domen]:
					if key in ['min_points', 'contour_points', 'num_of_lenghts', 'divider', 'max_div_num', 'round_const',\
					'down_steps', 'up_steps', 'max_depth']:
						const.config[domen][key] = int(round(float(request.POST[key]),0))
					else:
						const.config[domen][key] = float(request.POST[key])

		const.save_consts('./settings/'+data.settings.url)
		return redirect(reverse('const_start', args=(pk, )))

def calculate_a(request, pk, type_optimization):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)

	const = Const('./settings/'+data.settings.url)
	const.norm(df)
	const.calculate_a(df, type_optimization)
	const.save_consts('./settings/'+data.settings.url)

	return redirect(reverse('const_start', args=(pk, )))

def const_reload(request, pk):
	data = Projects.objects.get(pk=pk)
	const = Const('./settings/'+data.settings.url)
	f = open('./settings.yaml', 'r')
	data.settings = File(f, name=os.path.basename(f.name))
	data.save()
	return redirect(reverse('const_start', args=(pk, )))

def clustering_start(request,pk):
	if request.method =='GET':
		data = Projects.objects.get(pk=pk)
		data.stage = 3
		data.save()

		status_del_cluster = None
		status_del_subcluster = None
		df = reader.read('./df/'+data.attach.url)
		if 'cluster_id' in df.columns:
			status_cluster = False
			status_merge = True
			status_del_cluster = True
		else:
			status_cluster = True
			status_merge = False
		if 'subcluster_id' in df.columns:
			status_subcluster = False
			status_del_subcluster = True
		else:
			status_subcluster = True

		const = Const('./settings/'+data.settings.url)

		context = {
			'data': data,
			'settings': const.config,
			'status_cluster': status_cluster,
			'status_merge': status_merge,
			'status_subcluster': status_subcluster,
			'status_del_cluster': status_del_cluster,
			'status_del_subcluster': status_del_subcluster,
		}
		template = 'clustering_start.html'

		return render(request, template, context)
	elif request.method =='POST':
		data = Projects.objects.get(pk=pk)
		const = Const('./settings/'+data.settings.url)

		for domen in const.config:
			for key in request.POST.keys():
				if key == 'min_dif_0':
					const.config['isolated_cluster']['min_dif'][0] = float(request.POST[key])
				elif key == 'min_dif_1':
					const.config['isolated_cluster']['min_dif'][1] = float(request.POST[key])
				elif key == 'min_diff_0':
					const.config['conturs']['min_diff'][0] = float(request.POST[key])
				elif key == 'min_diff_1':
					const.config['conturs']['min_diff'][1] = float(request.POST[key])
				if key in const.config[domen]:
					if key in ['min_points', 'contour_points', 'num_of_lenghts', 'divider', 'max_div_num', 'round_const',\
					'down_steps', 'up_steps', 'max_depth']:
						const.config[domen][key] = int(round(float(request.POST[key]),0))
					else:
						const.config[domen][key] = float(request.POST[key])

		const.save_consts('./settings/'+data.settings.url)
		return redirect(reverse('clustering_start', args=(pk, )))

def compute_clustering(request, pk, type_c):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)
	const = Const('./settings/'+data.settings.url)
	if type_c == 1:
		cluster = Clusters(const.config) 
		df = cluster.get_isolated_clusters(df)
		data.comments += 'finded '+str(len(set(df['cluster_id'])))+' clusters\n'
	elif type_c == 2:
		fastcluster = Fast_Clusters(const.config) 
		df = fastcluster.get_isolated_clusters(df)
		data.comments += 'finded '+str(len(set(df['cluster_id'])))+' clusters\n'
	elif type_c == 3:
		Merger = IMerger(const.config)
		Merger.mergeClusters(df)
		data.comments += 'finded '+str(len(set(df['cluster_id'])))+' clusters\n'
	elif type_c == 4:
		sub = Subclusters(const.config)
		df = sub.subclustering(df, type_of_closed=2)
		data.comments += 'finded '+str(len(set(df['subcluster_id'])))+' subclusters\n'
	data.save()
	reader.write(df, './df/'+data.attach.url)
	return redirect(reverse('clustering_start', args=(pk, )))

def del_clustering(request, pk, type_c):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)
	if type_c==1:
		del df['cluster_id']
	elif type_c==2:
		del df['subcluster_id']

	reader.write(df, './df/'+data.attach.url)
	return redirect(reverse('clustering_start', args=(pk, )))
	
def delete_project(request, pk):
	data = Projects.objects.get(pk=pk)
	data.delete()
	return redirect(reverse('projects'))

