import os 

from .models import Projects
from .forms import AuthUserForm, RegisterUserForm, ProjectsForm

from core.reader import Reader
from core.const import Const
from core.fastclustering import Fast_Clusters
from core.clustering import Clusters
from core.i_merge import IMerger
from core.subclusterring import Subclusters
from core.predictor import Classsifier


from django.shortcuts import render, redirect, HttpResponse
from django.urls import reverse, reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView, LogoutView
from django.core.files import File
from django.http import HttpResponseRedirect
# Create your views here.


reader = Reader()

def main(request):
	context = {
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

def split_data(request, pk):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df'+data.attach.url)
	df_train, df_test = reader.split_data(df, train_size = int(request.POST['split_size']))
	name = data.name
	data.delete()

	data_train = Projects()
	data_train.author = request.user
	data_train.status = True
	data_train.stage = 1
	f = open('./settings.yaml', 'r')
	data_train.settings = File(f, name=os.path.basename(f.name))
	d = reader.write(df_train, './df/'+'.'.join(data.attach.url.split('.')[:-1])+'_train.'+data.attach.url.split('.')[-1])
	ff = open('./df/'+'.'.join(data.attach.url.split('.')[:-1])+'_train.'+data.attach.url.split('.')[-1], 'rb')
	data_train.attach = File(ff, name=os.path.basename(ff.name))
	data_train.name = name+'_train'
	data_train.save()

	data_test = Projects()
	data_test.author = request.user
	data_test.status = True
	data_test.stage = 1
	f = open('./settings.yaml', 'r')
	data_test.settings = File(f, name=os.path.basename(f.name))
	d = reader.write(df_test, './df/'+'.'.join(data.attach.url.split('.')[:-1])+'_test.'+data.attach.url.split('.')[-1])
	ff = open('./df/'+'.'.join(data.attach.url.split('.')[:-1])+'_test.'+data.attach.url.split('.')[-1], 'rb')
	data_test.attach = File(ff, name=os.path.basename(ff.name))
	data_test.name = name+'_test'
	data_test.save()
	return redirect(reverse('projects'))

def statistic(request, pk):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)
	text, df = reader.statistic(df, num_of_intervals = int(request.POST['num_interval']))
	data.comments += text
	data.save()
	reader.write(df, './df/'+data.attach.url)
	return redirect(reverse('start_project', args=(pk,)))

def f_statistic(request, pk):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)
	const = Const('./settings/'+data.settings.url)
	const.add_Fcolumn(df)
	text = const.statistic(df, num_of_intervals = int(request.POST['num_interval']))
	data.comments += text
	data.save()
	return HttpResponseRedirect(request.META.get('HTTP_REFERER'))

def project_start(request,pk):
	data = Projects.objects.get(pk=pk)

	df = reader.read('./df/'+data.attach.url)

	context = {
		'data': data,
		'rows': df.shape[0],
	}
	template = 'project_start.html'
	return render(request, template, context)

def const_start(request,pk):
	if request.method =='GET':
		
		data = Projects.objects.get(pk=pk)
		df = reader.read('./df/'+data.attach.url)
		if data.stage<=2:
			data.stage = 2
			data.save()
		const = Const('./settings/'+data.settings.url)

		need_columns = []
		for col in df.columns:
			if 'X' in col:
				if col in const.nameignore:
					need_columns.append([col, True])
				else:
					need_columns.append([col, False])

		context = {
			'columns': need_columns,
			'data': data,
			'settings': const.config,
		}
		template = 'const_start.html'

		return render(request, template, context)

	elif request.method =='POST':
		data = Projects.objects.get(pk=pk)
		const = Const('./settings/'+data.settings.url)

		const.config['ignore_coord'] = request.POST.getlist('need_coords')
		for domen in const.config:
			if domen == 'ignore_coord':
				continue
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
	text = const.calculate_a(df, type_optimization)
	data.comments += text
	data.save()

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
		const = Const('./settings/'+data.settings.url)
		
		if data.stage<3:
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

		context = {
			'data': data,
			'settings': const.config,
			'status_cluster': status_cluster,
			'status_merge': status_merge,
			'status_subcluster': status_subcluster,
			'status_del_cluster': status_del_cluster,
			'status_del_subcluster': status_del_subcluster,
			'rows': df.shape[0],
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
		if 'F' not in df.columns:
			const.add_Fcolumn(df)
		df = sub.subclustering(df, type_of_closed=2)
		data.comments += 'finded '+str(len(set(df['subcluster_id'])))+' subclusters\n'
	data.save()
	reader.write(df, './df/'+data.attach.url)
	return redirect(reverse('clustering_start', args=(pk, )))

def classification_start(request, pk):
	if request.method =='GET':
		data = Projects.objects.get(pk=pk)
		if data.stage<4:
			data.stage = 4
			data.save()

		all_projects = Projects.objects.filter(author=request.user)

		df = reader.read('./df/'+data.attach.url)
		const = Const('./settings/'+data.settings.url)
		need_columns = []
		for col in df.columns:
			if 'X' in col:
				if col in const.nameignore:
					need_columns.append([col, True])
				else:
					need_columns.append([col, False])

		other_project = []
		for projct in all_projects:
			if projct.pk != pk:
				other_project.append(projct)
		context = {
			'columns': need_columns,
			'data': data,
			'other_projects': other_project,
		}

		template = 'classification_start.html'

		return render(request, template, context)

	elif request.method =='POST':
		data = Projects.objects.get(pk=pk)
		const = Const('./settings/'+data.settings.url)

		const.config['ignore_coord'] = request.POST.getlist('need_coords')

		const.save_consts('./settings/'+data.settings.url)
		return redirect(reverse('classification_start', args=(pk, )))

def classification(request, pk):
	if request.method =='POST':
		pk_test = int(request.POST.get('project', None))

		data = Projects.objects.get(pk=pk)
		const = Const('./settings/'+data.settings.url)
		df_train = reader.read('./df/'+data.attach.url)

		data_test = Projects.objects.get(pk=pk_test)
		df_test = reader.read('./df/'+data_test.attach.url)

		classifier = Classsifier(const.config)

		df_test, text = classifier.predict(df_train, df_test)
		data.comments += text
		data.save()
		geeks_object = df_test.to_html()

		return HttpResponse(geeks_object)


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

