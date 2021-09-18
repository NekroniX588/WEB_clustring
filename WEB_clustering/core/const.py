import os
import datetime
import math
import yaml
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from core.utils import get_F, get_F_example

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


np.set_printoptions(suppress=True)

def write_log(str):
	with open('const_log.log', 'a', encoding='utf-8') as f:
		f.write(str+'\n')

class Const(object):
	def __init__(self, path=None):
		self.status = False
		if path is None:
			self.zz = yaml.load(open("./settings.yaml", 'r'))
		else:
			self.config = yaml.load(open(path, 'r'))
			if 'a' in self.config['consts']:
				self.status = True
		self.__norms = {}
		if 'ignore_coord' in self.config:
			self.nameignore = ['F', 'cluster_id', 'subcluster_id'] + self.config['ignore_coord']
		else:
			self.nameignore = ['F', 'cluster_id', 'subcluster_id']

	def __normalize(self, X, percent):
		#add dict of means
		norms = []
		for i in range(X.shape[1]):
			x = X[:, i]
			d = pairwise_distances(x[:, np.newaxis]).ravel()
			d.sort()
			lenght = max(int(d.shape[0] * percent / 100), 1)
			start = 0
			finish = lenght
			time_d = d[start:lenght]
			print('Persent of zeros start:',len(time_d[time_d==0])/len(time_d),'for X', i)
			while finish<=len(d)-1 and len(time_d[time_d==0])/len(time_d) > self.config['consts']['percent_of_zeros']/100:
				start += 1
				finish += 1
				time_d = d[start:finish]
			print('Persent of zeros finish:',len(time_d[time_d==0])/len(time_d),'for X', i)
			norm = d[start:finish].mean()
			x = x / norm
			norms.append(norm)
			X[:, i] = x
		return X, norms

	def norm(self, df):
		'''
		Inplace method for normilaze coords
		args:
		df -- data frame ['id', 'X1', 'X2', ..., 'Xn']
		'''
		self.config['norms'] = {}
		if 'ignore_coord' in self.config:
			need_names = [n for n in df.columns if n not in self.nameignore + ['id']] + self.config['ignore_coord']
		else:
			need_names = [n for n in df.columns if n not in self.nameignore + ['id']]
		df_for_norm = df[need_names]
		X, norms = self.__normalize(df_for_norm.values, self.config['consts']['percent_for_norms'])
		# print(X[:10])
		
		for i, col in enumerate(df_for_norm.columns[:]):
			self.config['norms'][col] = float(np.round(norms[i], self.config['consts']['round_const']))
		print(self.config)
		df[need_names] = X
		# print(df)

	def pca_norm(self, df, coords):
		self.config['norms'] = {}
		if 'ignore_coord' in self.config:
			need_names = [n for n in df.columns if n not in self.nameignore + ['id']] + self.config['ignore_coord']
		else:
			need_names = [n for n in df.columns if n not in self.nameignore + ['id']]
		df_for_norm = df[need_names]
		X, norms = self.__normalize(df_for_norm.values, self.config['consts']['percent_for_norms'])
		# print(X[:10])
		
		need_for_pca = []
		pca_names = []
		for i, col in enumerate(df_for_norm.columns[:]):
			if col in coords:
				self.config['norms'][col+'_original'] = float(np.round(norms[i], self.config['consts']['round_const']))
				need_for_pca.append(i)
				pca_names.append(col)
			else:
				self.config['norms'][col] = float(np.round(norms[i], self.config['consts']['round_const']))
		for i in need_for_pca:
			X[:,i] = X[:,i] * norms[i]

		pca = PCA(n_components=len(need_for_pca))
		pca.fit(X[:,need_for_pca])

		Y = pca.transform(X[:,need_for_pca])

		Y, after_pca_norms = self.__normalize(Y, self.config['consts']['percent_for_norms'])

		for i in range(len(pca_names)):
			self.config['norms'][pca_names[i]] = float(np.round(after_pca_norms[i], self.config['consts']['round_const']))

		X[:,need_for_pca] = Y
		df[need_names] = X
		return df, pca

	def get_norms(self):
		if len(self.config['norms'])==0:
			print("Norms is not calculated. Maybe you don't normalize training data")
			return 
		else:
			return self.config['norms']

	def inverse_norm(self, df):
		if 'norms' not in self.config:
			return df
		for key in self.config['norms']:
			df[key] *= self.config['norms'][key]

		return df

	def inverse_pca_norm(self, df, pca_path):
		if 'norms' not in self.config:
			return df

		pca_coord = []
		pca_norm = []
		original_cood = []
		original_norm = []

		ordinaty_norm = []
		ordinaty_coord = []
		for name in self.config['norms']:
			if 'original' not in name and name+'_original' in self.config['norms']:
				pca_coord.append(name)
				pca_norm.append(self.config['norms'][name])
				original_cood.append(name+'_original')
				original_norm.append(self.config['norms'][name+'_original'])
			if 'original' not in name and name+'_original' not in self.config['norms']:
				ordinaty_coord.append(name)
				ordinaty_norm.append(self.config['norms'][name])

		#Переводим PCA координаты в обычные
		X = df[pca_coord].values
		for i in range(len(pca_norm)):
			X[:,i] *= pca_norm[i]

		#Загружаем PCA
		with open(pca_path, "rb") as f:
			pca = pickle.load(f)

		#РеPCA
		X = pca.inverse_transform(X)

		df[pca_coord] = X
		#нормируем на оригиналы
		for i in range(len(ordinaty_coord)):
			df[ordinaty_coord[i]] *= ordinaty_norm[i]

		return df

	def f_statistic(self, df, num_of_intervals=10):
		text = ''
		if 'F' in df.columns:
			text += 'F_max = %.4f, F_min = %.4f \n'%(max(df['F']),min(df['F']))
			F = df['F'].values
			F.sort()
			start = 0
			finish = len(F)/num_of_intervals
			step = len(F)/num_of_intervals
			for i in range(num_of_intervals-1):
				text += 'Интревал %.d:% 5f \n'%(i,F[math.floor(start):math.floor(finish)].mean())
				start = finish
				finish += step
			text += 'Интревал %.d:% 5f \n'%(i+1,F[math.floor(start):].mean())
			text += '*'*20+'\n'
			text += 'Статистика по количеству \n'
			step = F[-1] / num_of_intervals
			interval = 0
			previous = 0
			current = step
			counter = 0
			for i in range(len(F)):
				if F[i] < current:
					counter += 1
				else:
					text += 'Интревал %.d [%5f, %5f] количество = %.d \n'%(interval,previous,current,counter)
					interval += 1
					previous = current
					current += step
					counter = 0
		else:
			text += 'F not calculated\n'
		return text

	def distance_statistic(self, df, num_of_intervals=10):
		text = ''
		need_column = []
		for col in df.columns:
			if 'X' in col:
				need_column.append(col)
		x = df[need_column].values
		D = np.tril(pairwise_distances(x))
		d = []
		for i in range(D.shape[0]):
			for j in range(D.shape[1]):
				if i>j:
					d.append(D[i,j])
		d = np.array(d)
		# d_abs = (d[d==0].shape[0]-df.shape[0])//2
		# d_rel = ((d[d==0].shape[0]-df.shape[0])//2)/((d.shape[0]-df.shape[0])//2)
		d_abs = d[d==0].shape[0]
		d_rel = d[d==0].shape[0]/d.shape[0]
		text += 'data contain absolute: %.d \n'%(d_abs)
		text += 'data contain relation: %.5f \n'%(d_rel)
		d.sort()
		start = 0
		finish = len(d)/num_of_intervals
		step = len(d)/num_of_intervals
		for i in range(num_of_intervals-1):
			print(d[math.floor(start):math.floor(finish)].shape)
			text += 'Interval: %.d:% 5f \n'%(i,d[math.floor(start):math.floor(finish)].mean())
			start = finish
			finish += step
		print(d[math.floor(start):].shape)
		text += 'Interval: %.d:% 5f \n'%(i+1,d[math.floor(start):].mean())
		text += '='*20 + '\n'
		return text

	def save_consts(self, path):
		assert type(path) == str, 'Name should be str'
		with open(path, 'w') as file:
			documents = yaml.dump(self.config, file)

	def add_Fcolumn(self, df, force=False):
		'''
		Inplace method for normilaze coords
		args:
		df -- data frame ['id', 'X1', 'X2', ..., 'Xn']
		'''
		if self.status == False and not force:
			print("Const a is not calculated yet. F can't be calculated") 
			return
		need_names = [n for n in df.columns if n not in self.nameignore] 
		X = df[need_names].values
		df['F'] =  np.array(get_F_example(X, self.config['consts']['a']))[:,-1]


	def __calculate_weights_by_max(self, X_percent_matrix, X, started_a):

		first = True# задаем флаг для первого нахождения a и средней суммы весов
		
		#Создаем список, где 0 (т.е. started) a будет стоять в начале
		var = [i for i in range(-self.config['consts']['down_steps'],self.config['consts']['up_steps']+1)]
		var.pop(var.index(0))
		list_for_k = [0] + var

		for k in tqdm(list_for_k):# задаем цикл диапазона [-down_steps:up_steps]
			
			a =  started_a * (self.config['consts']['power_koef']**k)#Вычисляем текущее значение а
			
			F = get_F_example(X, a)#Вычисляем F для матрицы (!!!ВНИМАНИЕ ФОРМУЛА!!!) [id, X1,..,Xn, F]

			def F_sort(arr):#Вспомогательная функция для сортировки
				return arr[-1]
			F.sort(key=F_sort,reverse = True)#Сортируем F по убыванию
			
			# Выбираем Y% матрицы
			lenght = int(np.round(len(F)*self.config['consts']['percent_Y']/100, 0))
			Y_percent_matrix = np.array(F[:lenght])
			
			#Высчитываем сумму ребер
			summ = 0
			for edge in X_percent_matrix:# Проходим построчно матрицу X% - row = [id вершины 1, id вершины 2, расстояние]
				if edge[0] in Y_percent_matrix[:,0] and edge[1] in Y_percent_matrix[:,0]: #Если обе вершины в матрице, то сумма + 1
					summ += 1
				elif edge[0] in Y_percent_matrix[:,0] or edge[1] in Y_percent_matrix[:,0]: #Если только 1 из вершин в матрице, то сумма + 0.5
					summ += 0.5

			if first:# Если флаг первого вхождения = True, то искомые a и среднюю сумму задаем тут
				first =False #Меняем флаг на False
				max_summ_edge = summ/len(X_percent_matrix)
				max_a = a
			else:# Если флаг первого фхождения = False, то у нас уже есть искомая a и средняя сумма
				if max_summ_edge < summ/len(X_percent_matrix): #Сравниваем среднюю сумму с предыдущим значение, если оно большо, тообновляем max_summ_edge и max a
					max_summ_edge = summ/len(X_percent_matrix)
					max_a = a

		return max_a, F

	def __calculate_weights_by_Y(self, X_percent_matrix, X, started_a):


		list_a = [] #Создаем список для хранения всех а
		list_Y_percent = [] #Создаем список для хранения всех Y% при которых средняя сумма больше порога
		list_summ_edge = [] #Создаем список для хранения всех средних сумм, которые впервые превысили порог
		
		arr_Y_step = [] # задаем массив шагов по Y%
		current = self.config['consts']['Y_step'] #Устанавлиаем первое значение как шаг по Y%

		while current < 100: #Пока текущее значение процента не превышает 100% 
			arr_Y_step.append(current) # Добавляем текущее значение в массив шагов по Y%
			current += self.config['consts']['Y_step'] # Прибавляем к текущему значению шаг 
		if arr_Y_step[-1]<100: #Для случая если шаг 100/Y_step не целое (например Y_step=3%), то добавляем в конец массивf шагов по Y% 100%
			arr_Y_step.append(100)

		#Создаем список, где 0 (т.е. started) a будет стоять в начале
		var = [i for i in range(-self.config['consts']['down_steps'],self.config['consts']['up_steps']+1)]
		var.pop(var.index(0))
		list_for_k = [0] + var

		for k in tqdm(list_for_k): # задаем цикл диапазона [-down_steps:up_steps]

			a =  started_a * (self.config['consts']['power_koef']**k)#Вычисляем текущее значение а

			list_a.append(a)#Добавляем текущее значение а в список для хранения всех а

			
			F = get_F_example(X, a) #Вычисляем F для матрицы (!!!ВНИМАНИЕ ФОРМУЛА!!!) [id, X1,..,Xn, F]
			
			def F_sort(arr):#Вспомогательная функция для сортировки
				return arr[-1]
			F.sort(key=F_sort,reverse = True)#Сортируем F по убыванию

			for step in arr_Y_step:#Итерируемся по матрице шагов Y%
				
				# Выбираем текущий Y% матрицы
				lenght = max(int(np.round(len(F)*step/100, 0)),1)
				Y_percent_matrix = np.array(F[:lenght])

				#Высчитываем сумму ребер
				summ = 0
				for edge in X_percent_matrix:# Проходим построчно матрицу X% - row = [id вершины 1, id вершины 2, расстояние]
					if edge[0] in Y_percent_matrix[:,0] and edge[1] in Y_percent_matrix[:,0]:#Если обе вершины в матрице, то сумма + 1
						summ += 1
					elif edge[0] in Y_percent_matrix[:,0] or edge[1] in Y_percent_matrix[:,0]:#Если только 1 из вершин в матрице, то сумма + 0.5
						summ += 0.5
				if summ/len(X_percent_matrix) > self.config['consts']['threshold']:#Если текущая средняя сумма больше порога
					list_Y_percent.append(step)#Добавляем текущее значение Y% в список для хранения всех всех Y%
					list_summ_edge.append(summ/len(X_percent_matrix))#Добаляем среднюю сумму в список для хранения всех средних сумм
					# print('a=',a, 'step=',step)
					break # Прерываем цикл

		#Задаем начальные значения как 0 элементы списков
		# print(list_Y_percent)
		min_Y_percent = list_Y_percent[0] 
		max_summ_edge = list_summ_edge[0]
		max_a = list_a[0]
		for i in range(1, len(list_Y_percent)):
			if min_Y_percent > list_Y_percent[i]: # Если текущий элемент списка Y% меньше минимального, то обновялем наши значения
				min_Y_percent = list_Y_percent[i]
				max_summ_edge = list_summ_edge[i]
				max_a = list_a[i]

		return max_a

	def __calculate_weights_by_integral_Y(self, X_percent_matrix, X, started_a):

		landscape = {}
		list_a = [] #Создаем список для хранения всех а
		list_summ_edge = [] #Создаем список для хранения всех средних сумм
		
		arr_Y_step = [] # задаем массив шагов по Y%
		current = self.config['consts']['Y_step'] #Устанавлиаем первое значение как шаг по Y%

		while current < 100: #Пока текущее значение процента не превышает 100% 
			arr_Y_step.append(current) # Добавляем текущее значение в массив шагов по Y%
			current += self.config['consts']['Y_step'] # Прибавляем к текущему значению шаг 
		if arr_Y_step[-1]<100: #Для случая если шаг 100/Y_step не целое (например Y_step=3%), то добавляем в конец массивf шагов по Y% 100%
			arr_Y_step.append(100)

		#Создаем список, где 0 (т.е. started) a будет стоять в начале
		var = [i for i in range(-self.config['consts']['down_steps'],self.config['consts']['up_steps']+1)]
		var.pop(var.index(0))
		list_for_k = [0] + var

		for k in tqdm(list_for_k): # задаем цикл диапазона [-down_steps:up_steps]

			a =  started_a * (self.config['consts']['power_koef']**k)#Вычисляем текущее значение а

			list_a.append(a)#Добавляем текущее значение а в список для хранения всех а

			F = get_F_example(X, a) #Вычисляем F для матрицы (!!!ВНИМАНИЕ ФОРМУЛА!!!) [id, X1,..,Xn, F]
			
			def F_sort(arr):#Вспомогательная функция для сортировки
				return arr[-1]
			F.sort(key=F_sort,reverse = True)#Сортируем F по убыванию

			for i, step in enumerate(arr_Y_step):#Итерируемся по матрице шагов Y%
				
				# Выбираем текущий Y% матрицы
				lenght = max(int(np.round(len(F)*step/100, 0)),1)
				Y_percent_matrix = np.array(F[:lenght])
				#Высчитываем сумму ребер
				summ = 0
				for edge in X_percent_matrix:# Проходим построчно матрицу X% - row = [id вершины 1, id вершины 2, расстояние]
					if edge[0] in Y_percent_matrix[:,0] and edge[1] in Y_percent_matrix[:,0]:#Если обе вершины в матрице, то сумма + 1
						summ += 1
					elif edge[0] in Y_percent_matrix[:,0] or edge[1] in Y_percent_matrix[:,0]:#Если только 1 из вершин в матрице, то сумма + 0.5
						summ += 0.5
				if i==0:
					list_summ_edge.append(summ/len(X_percent_matrix))#Добаляем среднюю сумму в список для хранения всех средних сумм
				else:
					list_summ_edge[-1] += summ/len(X_percent_matrix)
			landscape[k] = list_summ_edge[-1]

		#Задаем начальные значения как 0 элементы списков
		max_summ_edge = list_summ_edge[0]
		max_a = list_a[0]
		for i in range(1, len(list_summ_edge)):
			if max_summ_edge < list_summ_edge[i]: # Если текущий элемент списка Y% меньше минимального, то обновялем наши значения
				max_summ_edge = list_summ_edge[i]
				max_a = list_a[i]

		return max_a, landscape

	def __one_step_integral_Y(self, X_percent_matrix, arr_Y_step, X, a):
		
		F = get_F_example(X, a) #Вычисляем F для матрицы (!!!ВНИМАНИЕ ФОРМУЛА!!!) [id, X1,..,Xn, F]
		
		def F_sort(arr):#Вспомогательная функция для сортировки
			return arr[-1]
		F.sort(key=F_sort,reverse = True)#Сортируем F по убыванию

		for i, step in enumerate(arr_Y_step):#Итерируемся по матрице шагов Y%
			
			# Выбираем текущий Y% матрицы
			lenght = max(int(np.round(len(F)*step/100, 0)),1)
			Y_percent_matrix = np.array(F[:lenght])

			#Высчитываем сумму ребер
			summ = 0
			for edge in X_percent_matrix:# Проходим построчно матрицу X% - row = [id вершины 1, id вершины 2, расстояние]
				if edge[0] in Y_percent_matrix[:,0] and edge[1] in Y_percent_matrix[:,0]:#Если обе вершины в матрице, то сумма + 1
					summ += 1
				elif edge[0] in Y_percent_matrix[:,0] or edge[1] in Y_percent_matrix[:,0]:#Если только 1 из вершин в матрице, то сумма + 0.5
					summ += 0.5
			if i==0:
				summ_edge = summ/len(X_percent_matrix)#Добаляем среднюю сумму в список для хранения всех средних сумм
			else:
				summ_edge += summ/len(X_percent_matrix)
				
		return summ_edge

	def __calculate_window(self, X_percent_matrix, arr_Y_step, X, started_a, landscape, start_step, total_steps):
		print(total_steps)
		if start_step not in landscape:
			current_a =  started_a * (self.config['consts']['power_koef']**0)#Вычисляем текущее значение а
			current_summ = self.__one_step_integral_Y(X_percent_matrix, arr_Y_step, X, current_a)
			current_ = current_summ
			landscape[start_step] = current_summ
			total_steps += 1
			if total_steps == self.config['consts']['max_depth']:
				return True, total_steps

		up_ = []
		for i in range(1,self.config['consts']['up_steps']+1):
			up_step = start_step+i
			if up_step not in landscape:
				up_a = started_a * (self.config['consts']['power_koef']**up_step)
				up_summ = self.__one_step_integral_Y(X_percent_matrix, arr_Y_step, X, up_a)
				landscape[up_step] = up_summ
				total_steps += 1
				if total_steps == self.config['consts']['max_depth']:
					return True, total_steps

		down_ = []
		for i in range(1,self.config['consts']['down_steps']+1):
			down_step = start_step-i
			if down_step not in landscape:
				down_a = started_a * (self.config['consts']['power_koef']**down_step)
				down_summ = self.__one_step_integral_Y(X_percent_matrix, arr_Y_step, X, down_a)
				landscape[down_step] = down_summ
				total_steps += 1
				if total_steps == self.config['consts']['max_depth']:
					return True, total_steps

		return False, total_steps

	def __calculate_weights_by_integral_Y_direct(self, X_percent_matrix, X, started_a, logging_save = False):

		def find_max(landscape):
			max_value = -1
			max_key = -1
			for key in landscape:
				if landscape[key] > max_value:
					max_key = key
					max_value = landscape[key]

			return max_key

		def check_plato_max(landscape):
			max_value = -1
			max_key = -1
			for key in landscape:
				if landscape[key] > max_value:
					max_key = key
					max_value = landscape[key]
			values = []
			for key in landscape:
				if landscape[key] == max_value:
					values.append(key)
			return values

		arr_Y_step = [] # задаем массив шагов по Y%
		current = self.config['consts']['Y_step'] #Устанавлиаем первое значение как шаг по Y%

		while current < 100: #Пока текущее значение процента не превышает 100% 
			arr_Y_step.append(current) # Добавляем текущее значение в массив шагов по Y%
			current += self.config['consts']['Y_step'] # Прибавляем к текущему значению шаг 
		if arr_Y_step[-1]<100: #Для случая если шаг 100/Y_step не целое (например Y_step=3%), то добавляем в конец массивf шагов по Y% 100%
			arr_Y_step.append(100)

		total_steps = 0

		landscape = {}

		status, total_steps = self.__calculate_window(X_percent_matrix, arr_Y_step, X, started_a, landscape, 0, total_steps)

		max_now = find_max(landscape)
		values_old = None
		while not status:
			print(landscape)
			values = check_plato_max(landscape)
			if values == values_old:
				break
			values_old = values
			if len(values) > 1:
				status, total_steps = self.__calculate_window(X_percent_matrix, arr_Y_step, X, started_a, landscape, max(values), total_steps)
				status, total_steps = self.__calculate_window(X_percent_matrix, arr_Y_step, X, started_a, landscape, min(values), total_steps)
			else:
				status, total_steps = self.__calculate_window(X_percent_matrix, arr_Y_step, X, started_a, landscape, max_now, total_steps)
				if find_max(landscape) == max_now:
					break
				else:
					max_now = find_max(landscape)

		if logging_save and status:
			write_log("достигли максимального шага")

		max_now = find_max(landscape)
		if logging_save:
			write_log("Ландшафт")
			for key in landscape:
				write_log("Степень {}, значение {}".format(key, landscape[key]))
		if logging_save:
			write_log("Финальная а = {}".format(started_a * (self.config['consts']['power_koef']**max_now)))


		return started_a * (self.config['consts']['power_koef']**max_now), landscape

	def get_profile(self, F, p1, p2, logging_save = False):
		#Функция рассчета профиля F - матрица формата ['id','x1',...,'xn','F'], p1, p2 - точки формата ['id','x1',...,'xn','F']

		# if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/self.cluster_config['divider'] <= self.contur_config['min_diff']:
		if logging_save:
			log_message = 'Профиль для пары точек'
			write_log(log_message)
			write_log(str(p1))
			write_log(str(p2))

		if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])) <= self.config['isolated_cluster']['min_len']:
			return None
		else:
			div_num = 0
			segment_len = np.linalg.norm( np.array(p1[1:-1]) - np.array(p2[1:-1]))
			while (div_num < self.config['isolated_cluster']['max_div_num']-1) and (segment_len > self.config['isolated_cluster']['min_len']):
				div_num+= 1
				num_of_segments = (self.config['isolated_cluster']['divider']+1)*div_num + self.config['isolated_cluster']['divider']
				segment_len = np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/num_of_segments

			if div_num == 0:
				return None

			points = []
			for i in range(num_of_segments):
				x = [-1.]
				for j in range(1, len(p1)-1):
					x.append((p1[j] + p2[j]*(i+1)/(num_of_segments-i))/(1+(i+1)/(num_of_segments-i)))
				points.append(np.array(x))

			Fs = []
			for point in points:
				pp = [p for p in point[1:]] + [get_F_example([f[:-1] for f in F], self.config['consts']['a'], target=point)]
				Fs.append(pp)
			F_all = [p1.tolist()[1:]] + Fs + [p2.tolist()[1:]]
			F_all = np.stack(F_all)

			if logging_save:
				log_message = 'Последовательность проведенных точек'
				write_log(log_message)
				for p in F_all:
					write_log(str(p))

			F_diff = []
			for i in range(1,F_all.shape[0]-1):
				max_l = F_all[:i,-1].max()
				max_r = F_all[i+1:,-1].max()
				max_c = min(max_l, max_r)
				F_diff.append(max_c-F_all[i,-1])
			F_diff = max(F_diff)

			if logging_save:
				log_message = 'Просадка профиля'
				write_log(log_message)
				write_log(str(F_diff))

			return F_diff

	def calculate_dif(self, F, logging_save = False):

		if logging_save:
			write_log('НАЧАЛО ПОДБОРА DIF ' + str(datetime.datetime.now()))

		def eucl(p1,p2):
			return sum((p1 - p2)**2)

		F = sorted(F, key=lambda x: x[-1], reverse=True)
		F = np.stack(F)
		lenght = max(int(np.round(F.shape[0]*(self.config['consts']['U']/100), 0)), 1)
		F = F[:lenght]
		X = pairwise_distances(F[:,1:-1])# U
		max_i = 0
		max_j = 0
		max_v = 0
		for i in range(X.shape[0]):
			for j in range(i+1,X.shape[1]):
				if X[i,j]>max_v:
					max_i = i
					max_j = j
					max_v = X[i,j]

		A = F[max_i]
		B = F[max_j]
		current_points = [A,B]
		dim = F.shape[1]-2
		N = 2 + int(self.config['consts']['w'])*(dim-1)
		for i in range(min(N-2,F.shape[0])):# 2+int(self.config['conturs']['n_points_for_dif'])*(A.shape[0]-2+1)
			max_v = min(eucl(F[0, 1:-1], p[1:-1]) for p in current_points)
			max_p = F[0]
			for j in range(1,len(F)):
				v = min(eucl(F[j, 1:-1], p[1:-1]) for p in current_points)
				if v>max_v:
					max_v = v
					max_p = F[j]
			current_points.append(max_p)
		if logging_save:
			for k,p in enumerate(current_points):
				log_message = 'Ключевая точка ' + str(k) + ':' + str(p)
				write_log(log_message)

		F_dif = []
		done = set()
		for i in range(len(current_points)):
			for j in range(len(current_points)):
				if i==j:
					continue
				name = str(i)+'_'+str(j)
				if name in done:
					continue
				else:
					F_dif.append(self.get_profile(F,current_points[i],current_points[j], logging_save = logging_save))
					done.add(name)
					done.add(name[::-1])
					
		F_dif_good = []
		for item in F_dif:
			if item is not None and item>0:
				F_dif_good.append(item)
		if len(F_dif_good) == 0:
			return None, None
		if len(F_dif_good)==1 and F_dif_good[0]>0:
			F_dif_max = F_dif_good[0]
			F_diff_max = F_dif_max * 0.2
			return F_dif_max, F_diff_max

		if logging_save:
			write_log("Все хорошие просадки:")
			for F_item in F_dif_good:
				write_log(str(F_item))

		F_dif_mean = sum(F_dif_good)/len(F_dif_good)
		F_dif_max = -9999999
		F_dif_item = -9999999
		F_dif_good.sort(reverse=True)

		if logging_save:
			write_log("Процесс поиска")

		for i in range(1,len(F_dif_good)):
			F_dif_mean_i = i*(sum(F_dif_good[:i])/len(F_dif_good[:i])-F_dif_mean)
			if logging_save:
				write_log(str(F_dif_mean_i))
			if F_dif_mean_i>F_dif_max:
				F_dif_max = F_dif_mean_i
				F_dif_item = sum(F_dif_good[:i])/len(F_dif_good[:i])
		if logging_save:
			write_log("Максимум")
			write_log(str(F_dif_max))
		F_dif_max = F_dif_item * self.config['isolated_cluster']['min_dif']
		F_diff_max = F_dif_max * 0.2
		return F_dif_max, F_diff_max

	def __calculate_const(self, X, type = 1, cluster_id=None, subcluster_id=None, logging_save = False):
		#
		matrix = []
		map_index = {k:index for k,index in enumerate(X[:,0])}
		print(X)
		print('X',X.shape)
		distance_matrix = np.tril(pairwise_distances(X[:,1:]))#Вычисляем матрицу расстояний [len(id), len(id)]
		print(distance_matrix.shape)
		#[0,   0,   0]
		#[1.3, 0,   0] - пример
		#[1.8, 5.2, 0]
		#Цикл для составления матрицы соответствия [id вершины 1, id вершины 2, расстояние]
		for i in range(distance_matrix.shape[0]):
			for j in range(distance_matrix.shape[1]):
				if i>j:
					row = [map_index[i], map_index[j], distance_matrix[i,j]]
					if cluster_id is not None and subcluster_id is not None:
						if cluster_id[map_index[i]]==cluster_id[map_index[j]] and subcluster_id[map_index[i]]==subcluster_id[map_index[j]]:
							matrix.append(row)
					elif cluster_id is not None:
						if cluster_id[map_index[i]]==cluster_id[map_index[j]]:
							matrix.append(row)
					elif subcluster_id is not None:
						if subcluster_id[map_index[i]]==subcluster_id[map_index[j]]:
							matrix.append(row)
					else:
						matrix.append(row)
		#[1, 2, 1.3]
		#[1, 3, 1.8] - пример
		#[2, 3, 5.2]
		print(len(matrix))
		matrix.sort(key=lambda x: x[2])
		distance_matrix = np.array(matrix) #Переводим матрицу в np.array
		print('distance_matrix',distance_matrix.shape)
		#Выбираем X% матрицы расстояний
		lenght = max(int(np.round(distance_matrix.shape[0]*(self.config['consts']['percent_X']/100), 0)), 1)
		start = 0
		finish = lenght
		time_d = distance_matrix[start:finish]
		print('Persent of zeros start:',len(time_d[time_d[:,2]==0])/len(time_d))
		while finish<=len(distance_matrix)-1 and len(time_d[time_d[:,2]==0])/len(time_d) > self.config['consts']['percent_of_zeros']/100:
			print('Persent of zeros new:',len(time_d[time_d[:,2]==0])/len(time_d))
			start += 1
			finish += 1
			time_d = distance_matrix[start:finish]
		X_percent_matrix = distance_matrix[start:finish]
		print('X_percent_matrix',X_percent_matrix.shape)
		#Вычисляем начальное значение a
		print(X_percent_matrix)
		mean_distance_matrix = X_percent_matrix[:,2].mean()
		print(mean_distance_matrix)
		started_a = self.config['consts']['const'] * mean_distance_matrix
		if type == 0:
			max_a = started_a
		elif type == 1:#В первом варианте высчитываем расстояния по 1-му варианту в документе
			# max_a = self.__calculate_weights_by_max(X_percent_matrix, X, started_a)
			max_a = started_a
			print('DEPRECATED')
		elif type == 2:#Во втором варианте высчитываем расстояния по 2-му варианту в документе
			# max_a = self.__calculate_weights_by_Y(X_percent_matrix, X, started_a)
			max_a = started_a
			print('DEPRECATED')
		elif type == 3:#В третьем варианте высчитываем расстояния по 3-му варианту в документе
			max_a, landscape = self.__calculate_weights_by_integral_Y(X_percent_matrix, X, started_a)
		elif type == 4:#В третьем варианте высчитываем расстояния по 3-му варианту в документе
			max_a, landscape = self.__calculate_weights_by_integral_Y_direct(X_percent_matrix, X, started_a, logging_save = logging_save)
		#На выходе получаем 2 значения (коэффициент a, и среднее значение весов)
		# if logging_save:
		# 	X = []
		# 	Y = []
		# 	for x,y in sorted(list(landscape.items()), key=lambda x: x[0]):
		# 		X.append(x)
		# 		Y.append(y)

		
		# 	if os.path.isfile('landscape.png'):
		# 		os.remove('landscape.png')   # Opt.: os.system("rm "+strFile)
		# 	plt.plot(X, Y)
		# 	plt.ylabel('Result value')
		# 	plt.xlabel('Step')
		# 	plt.savefig('landscape.png', format='png')
		# 	plt.clf()
		return max_a

	def calculate_a(self, df, type_of_optimization=2, max_a=None, logging_save = False):
		'''
		Method for calcualte_a and change consts with a
		args:
		df - data frame ['id', 'X1', 'X2', ..., 'Xn']
		type_of_optimization - type of optimization (type1 - using max; type2 - using threshold) default = type2
		'''
		text = ''
		if max_a is None:
			if self.status == True:
				text += 'Константы уже подсчитаны. Начните проект заново.\n'
				return text

			cluster_id = None
			subcluster_id = None
			if 'cluster_id' in df.columns:
				cluster_id = {item[0]:item[1] for item in df[['id','cluster_id']].values}
				
			if 'subcluster_id' in df.columns:
				subcluster_id = {item[0]:item[1] for item in df[['id','subcluster_id']].values}

			need_names = [n for n in df.columns if n not in self.nameignore] 
			X = df[need_names].values#приводим их np.array [id, X1, X2]
			max_a = self.__calculate_const(X, type_of_optimization, cluster_id, subcluster_id, logging_save = logging_save)
		else:
			text += 'Calculation consts with started a\n'
		#Изменяем необходимые константы
		self.config['consts']['a'] = float(np.round(max_a, self.config['consts']['round_const']))

		self.config['isolated_cluster']['min_len'] = \
			float(np.round(self.config['isolated_cluster']['min_len'] * max_a, self.config['consts']['round_const']))

		self.config['knots']['stop_const'] = \
			float(np.round(self.config['knots']['stop_const'] * max_a, self.config['consts']['round_const']))

		need_names = [n for n in df.columns if n not in self.nameignore] + ['F']
		self.add_Fcolumn(df, force=True)

		X = df[need_names].values#приводим их np.array [id, X1, X2]

		min_dif, min_diff = self.calculate_dif(X, logging_save = logging_save)
		if min_dif is None or min_dif is None:
			text += 'Вероятно, Ваши данные распределены монолитно и не распадаются на кластеры. Если кластеризации все-таки требуется добиться, — попробуйте вручную уменьшить величину «а» и повторить процесс\n'
			self.config['conturs']['min_diff'] = float(round(1.1, self.config['consts']['round_const']))
			self.config['isolated_cluster']['min_dif'] = float(round(1.1, self.config['consts']['round_const']))
		self.config['conturs']['min_diff'] = float(np.round(min_diff, self.config['consts']['round_const']))
		self.config['isolated_cluster']['min_dif'] = float(np.round(min_dif, self.config['consts']['round_const']))
		self.config['isolated_cluster']['merge_threshold'] = float(np.round(1.5*min_dif, self.config['consts']['round_const']))
		# value = 1./((max_a * self.config['conturs']['min_diff'][0])**2\
		# 	+ max_a * self.config['conturs']['min_diff'][1])
		# self.config['conturs']['min_diff'] = float(np.round(value, self.config['consts']['round_const']))

		# value = 1./((max_a * self.config['isolated_cluster']['min_dif'][0])**2\
		# 	+ max_a * self.config['isolated_cluster']['min_dif'][1])
		# self.config['isolated_cluster']['min_dif'] = float(np.round(value, self.config['consts']['round_const']))
		# self.status = True
		text += 'Константы подсчитаны\n'
		return text