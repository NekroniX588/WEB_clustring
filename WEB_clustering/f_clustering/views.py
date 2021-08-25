import os 
import io
import yaml
import pandas as pd
import pickle
import zipfile

from .models import Projects
from .forms import AuthUserForm, RegisterUserForm, ProjectsForm
from .models import Projects

from core.reader import Reader
from core.const import Const
from core.fastclustering import Fast_Clusters
from core.clustering import Clusters
from core.i_merge import IMerger
from core.subclusterring import Subclusters
from core.predictor import Classsifier
from core.utils import GetProfile


from django.shortcuts import render, redirect, HttpResponse
from django.urls import reverse, reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView, LogoutView
from django.core.files import File
from django.http import HttpResponseRedirect, HttpResponse, Http404
from django.conf import settings
from django.contrib import messages
from django.core.files.base import ContentFile
# Create your views here.

LOGGING = False

reader = Reader()

def main(request):
	context = {
	}
	template = 'main_page.html'
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

		df = reader.read('./df/'+self.object.attach.url)
		if df is not None:
			self.object.status = True
			# if 'cluster_id' in df.columns:
			# 	del df['cluster_id']
			# if 'subcluster_id' in df.columns:
			# 	del df['subcluster_id']
			# reader.write(df, './df/'+self.object.attach.url)
		else:
			messages.error(self.request, 'Ошибка формата данных!!!')

		return super().form_valid(form)

def table(request, pk):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df'+data.attach.url)
	geeks_object = df.to_html()
	return HttpResponse(geeks_object)

def clear_log(request, pk):
	data = Projects.objects.get(pk=pk)
	data.comments = ""
	data.save()
	return HttpResponseRedirect(request.META.get('HTTP_REFERER'))

def download_data(request, pk):
	data = Projects.objects.get(pk=pk)
	file_path = './df'+data.attach.url
	if os.path.exists(file_path):
		with open(file_path, 'rb') as fh:
			response = HttpResponse(fh.read(), content_type="application/liquid")
			response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
			return response
	raise Http404

def download_settings(request, pk):
	data = Projects.objects.get(pk=pk)
	file_path = './settings/'+data.settings.url
	if request.user.is_superuser:
		if os.path.exists(file_path):
			with open(file_path, 'rb') as fh:
				response = HttpResponse(fh.read(), content_type="application/liquid")
				response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
				return response
	else:
		Drop_category = ['conturs', 'knots', 'isolated_cluster', 'subcluster']
		Drop_keys = [['consts', 'round_const'], ['consts', 'power_koef'], ['consts', 'percent_Y'], \
					 ['consts', 'threshold'], ['consts', 'cluster_importancy'], ['consts', 'w'], \
					 ['consts', 'Y_step'], ['consts', 'down_steps'], ['consts', 'up_steps'],\
					 ['consts', 'max_depth'], ['consts', 'percent_for_norms'], ['consts', 'percent_of_zeros'],
					 ['consts', 'U']]
		const = Const('./settings/'+data.settings.url)
		for domen in Drop_category:
			const.config.pop(domen)
		for domen, key in Drop_keys:
			const.config[domen].pop(key)

		byte = io.StringIO()
		yaml.dump(const.config, byte)

		response = HttpResponse(byte.getvalue(), content_type="application/liquid")
		response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
		return response

	raise Http404

def download_clusters(request, pk):
	data = Projects.objects.get(pk=pk)

	df = reader.read('./df'+data.attach.url)

	byte = io.BytesIO()
	zip = zipfile.ZipFile(byte, "a")

	for cluster in sorted(df['cluster_id'].unique()):
		
		data_byte = io.BytesIO()
		df[df['cluster_id']==cluster].to_csv(data_byte, index=False)

		zip.writestr('Cluster'+str(cluster)+'.csv', data_byte.getvalue())
	zip.close()

	response = HttpResponse(byte.getvalue(), content_type="application/zip")
	response['Content-Disposition'] = 'inline; filename=Clusters.zip'
	return response
	raise Http404

def download_subclusters(request, pk):
	data = Projects.objects.get(pk=pk)

	df = reader.read('./df'+data.attach.url)

	byte = io.BytesIO()
	zip = zipfile.ZipFile(byte, "a")

	for cluster in sorted(df['subcluster_id'].unique()):
		
		data_byte = io.BytesIO()
		df[df['subcluster_id']==cluster].to_csv(data_byte, index=False)

		zip.writestr('Subcluster'+str(cluster)+'.csv', data_byte.getvalue())
	zip.close()

	response = HttpResponse(byte.getvalue(), content_type="application/zip")
	response['Content-Disposition'] = 'inline; filename=Subclusters.zip'
	return response
	raise Http404

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
	text = const.f_statistic(df, num_of_intervals = int(request.POST['num_interval']))
	data.comments += text
	data.save()
	return HttpResponseRedirect(request.META.get('HTTP_REFERER'))

def distance_statistic(request, pk):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)
	const = Const('./settings/'+data.settings.url)
	text = const.distance_statistic(df, num_of_intervals = int(request.POST['num_interval']))
	data.comments += text
	data.save()
	return HttpResponseRedirect(request.META.get('HTTP_REFERER'))

def get_profile(request, pk):
	if request.method =='GET':
		data = Projects.objects.get(pk=pk)
		template = 'profile_select.html'
		context = {
			'data': data,
		}
		return render(request, template, context)

	elif request.method =='POST':
		p1_id = int(request.POST.get('point1', None))
		p2_id = int(request.POST.get('point2', None))

		if p1_id == p2_id:
			messages.error(request, 'id 2-х точек должны быть различны')
			return redirect(reverse('get_profile', args=(pk, )))

		data = Projects.objects.get(pk=pk)
		df = reader.read('./df/'+data.attach.url)

		if p1_id not in df['id']:
			messages.error(request, 'точки с id {} нет в данных'.format(p1_id))
			return redirect(reverse('get_profile', args=(pk, )))
		if p2_id not in df['id']:
			messages.error(request, 'точки с id {} нет в данных'.format(p2_id))
			return redirect(reverse('get_profile', args=(pk, )))

		const = Const('./settings/'+data.settings.url)
		if 'F' not in df.columns:
			const.add_Fcolumn(df)
		getProfile = GetProfile(const.config)
		result, array = getProfile.get_profile(df, p1_id, p2_id)

		data = Projects.objects.get(pk=pk)
		template = 'profile_result.html'
		context = {
			'result': result,
			'array': array,
		}
		return render(request, template, context)

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
			'const_status': const.status,
			'columns': need_columns,
			'data': data,
			'settings': const.config,
		}
		if request.user.is_superuser:
			template = 'const_start.html'
		else:
			template = 'const_start_base.html'

		return render(request, template, context)

	elif request.method =='POST':
		data = Projects.objects.get(pk=pk)
		const = Const('./settings/'+data.settings.url)

		calulate_with_started_a = None

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
				elif key == 'started_a':
					if request.POST[key] != '':
						calulate_with_started_a = float(request.POST[key])
				if key in const.config[domen]:
					if key in ['min_points', 'contour_points', 'num_of_lenghts', 'divider', 'max_div_num', 'round_const',\
					'down_steps', 'up_steps', 'max_depth', 'w', 'U', 'max_key_points', 'Steps', 'Max_persent']:
						const.config[domen][key] = int(round(float(request.POST[key]),0))
					else:
						const.config[domen][key] = float(request.POST[key])
		const.save_consts('./settings/'+data.settings.url)
		if calulate_with_started_a is not None:
			df = reader.read('./df/'+data.attach.url)
			const.norm(df)
			reader.write(df, './df/'+data.attach.url)
			text = const.calculate_a(df, 3, calulate_with_started_a)
			data.comments += text
			data.save()
			const.save_consts('./settings/'+data.settings.url)

		return redirect(reverse('const_start', args=(pk, )))

def calculate_norms(request, pk):

	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)
	const = Const('./settings/'+data.settings.url)
	const.norm(df)
	const.save_consts('./settings/'+data.settings.url)
	reader.write(df, './df/'+data.attach.url)

	return redirect(reverse('const_start', args=(pk, )))

def calculate_pca_norms(request, pk):
	if request.method =='GET':

		data = Projects.objects.get(pk=pk)
		df = reader.read('./df/'+data.attach.url)

		need_columns = []
		for col in df.columns:
			if 'X' in col:
				need_columns.append(col)
		context = {
			'columns': need_columns,
			'data': data,
			}

		template = 'pca.html'

		return render(request, template, context)
	elif request.method =='POST':
		data = Projects.objects.get(pk=pk)
		coords = request.POST.getlist('need_coords')
		if len(coords) < 2:
			messages.error(request, 'Необходимо минимум 2 координаты для PCA')
			return redirect(reverse('calculate_pca_norms', args=(pk, )))

		df = reader.read('./df/'+data.attach.url)
		const = Const('./settings/'+data.settings.url)
		df, pca = const.pca_norm(df, coords)

		content = pickle.dumps(pca)
		fid = ContentFile(content)
		data.pca.save("pca.pkl", fid)
		fid.close()
		const.save_consts('./settings/'+data.settings.url)
		reader.write(df, './df/'+data.attach.url)

	return redirect(reverse('const_start', args=(pk, )))

def calculate_a(request, pk, type_optimization):
	data = Projects.objects.get(pk=pk)
	df = reader.read('./df/'+data.attach.url)

	const = Const('./settings/'+data.settings.url)
	
	reader.write(df, './df/'+data.attach.url)
	text = const.calculate_a(df, type_optimization, logging_save=LOGGING)
	data.comments += text
	data.save()

	const.save_consts('./settings/'+data.settings.url)

	return redirect(reverse('const_start', args=(pk, )))

def const_reload(request, pk):
	data = Projects.objects.get(pk=pk)
	const = Const('./settings/'+data.settings.url)
	const_clear = Const('./settings.yaml')
	for major_key in const_clear.config:
		for minor_key in const_clear.config[major_key]:
			const.config[major_key][minor_key] = const_clear.config[major_key][minor_key]
	if 'a' in const.config['consts']:
		const.config['consts'].pop('a')
	const.save_consts('./settings/'+data.settings.url)
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
		if request.user.is_superuser:
			template = 'clustering_start.html'
		else:
			template = 'clustering_start_base.html'

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
					'down_steps', 'up_steps', 'max_depth', 'w', 'U', 'max_key_points', 'Steps', 'Max_persent']:
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
		data.comments += 'Нашлось '+str(len(set(df['cluster_id'])))+' кластера\n'
	elif type_c == 2:
		fastcluster = Fast_Clusters(const.config) 
		df = fastcluster.get_isolated_clusters(df, logging_save=LOGGING)
		data.comments += 'Нашлось '+str(len(set(df['cluster_id'])))+' кластера\n'
	elif type_c == 3:
		Merger = IMerger(const.config)
		df = Merger.mergeClusters(df, logging_save=LOGGING)
		data.comments += 'Мерджинг проделан\n'
		data.comments += 'Нашлось '+str(len(set(df['cluster_id'])))+' кластера\n'
	elif type_c == 4:
		sub = Subclusters(const.config)
		if 'F' not in df.columns:
			const.add_Fcolumn(df)
		df = sub.subclustering(df, type_of_closed=2, logging_save=LOGGING)
		data.comments += 'Нашлось '+str(len(set(df['subcluster_id'])))+' сабкластеров\n'
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
		pk_test = request.POST.get('project', None)
		if pk_test is None:
			data = Projects.objects.get(pk=pk)
			data.comments += 'Нет данных для классификации \n'
			data.save()
			return redirect(reverse('classification_start', args=(pk, )))

		pk_test = int(pk_test)

		data = Projects.objects.get(pk=pk)
		const = Const('./settings/'+data.settings.url)
		df_train = reader.read('./df/'+data.attach.url)

		data_test = Projects.objects.get(pk=pk_test)
		df_test = reader.read('./df/'+data_test.attach.url)

		classifier = Classsifier(const.config)

		print(data.pca)
		if data.pca:
			df_test, text = classifier.predict(df_train, df_test, './pcas/'+data.pca.url)
		else:
			df_test, text = classifier.predict(df_train, df_test, None)
		
		
		data.comments += text
		data.save()

		with io.BytesIO() as b:
			# Use the StringIO object as the filehandle.
			writer = pd.ExcelWriter(b, engine='xlsxwriter')
			df_test.to_excel(writer, sheet_name='Sheet1')
			writer.save()
			# Set up the Http response.
			filename = 'result.xlsx'
			response = HttpResponse(
				b.getvalue(),
				content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
			)
			response['Content-Disposition'] = 'attachment; filename=%s' % filename
			return response

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
	if os.path.exists('./df/'+data.attach.url):
		os.remove('./df/'+data.attach.url)
	if os.path.exists('./settings/'+data.settings.url):
		os.remove('./settings/'+data.settings.url)
	data.delete()
	return redirect(reverse('projects'))

