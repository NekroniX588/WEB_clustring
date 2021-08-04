import numpy as np
from tqdm import tqdm
from core.utils import get_F_example
import copy

import datetime

np.set_printoptions(suppress=True)

def write_log(str):
	with open('subclustering.log', 'a', encoding='utf-8') as f:
		f.write(str+'\n')

class Subclusters(object):

	def __init__(self, config):
		self.config = config
		if 'ignore_coord' in self.config:
			self.nameignore = ['cluster_id', 'subcluster_id'] + self.config['ignore_coord']
		else:
			self.nameignore = ['cluster_id', 'subcluster_id']

	def __get_profile(self, F, p1, p2, logging_save = False):
		# print(p1)
		#Функция рассчета профиля F - матрица формата ['id','x1',...,'xn','F'], p1, p2 - точки формата ['id','x1',...,'xn','F']
		# if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/self.config['isolated_cluster']['divider'] <= self.config['conturs']['min_diff']:
		if logging_save:
			log_message = 'Профиль для пары точек'
			write_log(log_message)
			write_log(str(p1))
			write_log(str(p2))

		if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])) <= self.config['isolated_cluster']['min_len']:
			if logging_save:
				log_message = 'Расстояние между точками меньше min_len'
				write_log(log_message)
			return 'close', 0
		else:
			div_num = 0
			segment_len = np.linalg.norm( np.array(p1[1:-1]) - np.array(p2[1:-1]))
			while (div_num < self.config['isolated_cluster']['max_div_num']-1) and (segment_len > self.config['isolated_cluster']['min_len']):
				div_num+= 1
				num_of_segments = (self.config['isolated_cluster']['divider']+1)*div_num + self.config['isolated_cluster']['divider']
				segment_len = np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/num_of_segments

			if div_num == 0:
				if logging_save:
					log_message = 'Не удалось разбить отрезок'
					write_log(log_message)
				return 'close', 0

			points = []
			for i in range(num_of_segments):
				x = [-1.]
				for j in range(1, len(p1)-1):
					x.append((p1[j] + p2[j]*(i+1)/(num_of_segments-i))/(1+(i+1)/(num_of_segments-i)))
				points.append(np.array(x))
			# Fs = []
			# for point in points:
			# 	F_cur = get_F_example([f[:-1] for f in F], self.config['consts']['a'], target=point)
			# 	Fs.append([point[1],  point[2], F_cur])
			# Fs = [[point[1],  point[2], get_F_example([f[:-1] for f in F], self.config['consts']['a'], target=point)] for point in points]#&&&&&&&&&&&???????????
			# Fs = [[p for p in point[1:] + [get_F_example([f[:-1] for f in F], self.config['consts']['a'], target=point)]] \
			# 																						for point in points]
			Fs = []
			for point in points:
				pp = [p for p in point[1:]] + [get_F_example([f[:-1] for f in F], self.config['consts']['a'], target=point)]
				Fs.append(pp)

			if logging_save:
				log_message = 'Последовательность проведенных точек'
				write_log(log_message)
				for p in Fs:
					write_log(str(p))

			Fs = sorted(Fs, key = lambda S: S[-1], reverse = False)
	#         print(Fs)
			

			Fmin = np.min([F[-1] for F in Fs])
			Fstar = p1[-1] if  p1[-1] < p2[-1] else p2[-1]

			if Fstar - Fmin >= self.config['isolated_cluster']['min_dif']:
				if logging_save:
					log_message = 'Просадка выше min_dif, статус different'
					write_log(log_message)
				return 'different', Fstar - Fmin
			else:
				if logging_save:
					log_message = 'Просадка ниже min_dif, статус common'
					write_log(log_message)
				return 'common', Fstar - Fmin

	def __calculate_distance(self, p, matrix):
		"""
		p - [id, X1,...,Xn,F]
		matrix - [[id, X1,...,Xn,F],...,[id, X1,...,Xn,F]]
		========
		return - index of nearest neighbor
		"""
		start = True
		for row_index, row in enumerate(matrix):
			if p[0]==row[0]:
				continue
			distance = np.linalg.norm(np.array(p[1:-1]) - np.array(row[1:-1]))
			if start:
				min_distanse = distance
				min_row_index = row_index
				start = False
			if min_distanse > distance:
	#             print(min_distanse)
				min_distanse = distance
				min_row_index = row_index
		return min_row_index

	def __calculate_distance_for_2_points(self, A,B):
		"""
		A - [id, X1,...,Xn,F]
		B - [id, X1,...,Xn,F]
		=========
		return - distance between 2 points
		"""
		return np.linalg.norm(np.array(A[1:-1]) - np.array(B[1:-1]))
	
	def __merge(self, A, A_id, B, B_id, F_matrix):
		"""
		A - list of points [id, X1,...,Xn,F]
		B - list of points [id, X1,...,Xn,F]
		F_matrix - [[id, X1,...,Xn,F],...,[id, X1,...,Xn,F]]
		=======
		return - bool value, status for merging clusters
		"""
		for i in range(len(A)):
			for j in range(len(B)):
				if A_id[i]==B_id[j]:
					result, _ = self.__get_profile(F_matrix, A[i], B[j])

					if result == 'different':
						return False
		return True

	def __norm_sub_index(self, results):
		now_index = sorted(list(set(results.values())))
		transform = {index:self.max_index_sub+k for k,index in enumerate(now_index)}
		self.max_index_sub += len(now_index)
		for name in results:
			results[name] = transform[results[name]]

	def __generate_layers(self, X):
		"""
		Generate F layers, due to F_step from settings
		"""
		F_layers = []

		len_points = max(int(np.round(X.shape[0]*(self.config['subcluster']['Max_persent']/100), 0)), 1)
		need_points = X[:len_points]

		start = 0
		finish = len(need_points)//self.config['subcluster']['Steps']
		step = len(need_points)//self.config['subcluster']['Steps']

		for i in range(self.config['subcluster']['Steps']-1):
			F_layers.append(need_points[start:finish,-1].mean())
			start = finish
			finish += step
		F_layers.append(need_points[start:,-1].mean())
		return F_layers

	def __find_nearest_point(self, p, X):
		distances = []
		min_len_points = []
		for i in range(len(X)):
			p2 = X[i]
			d = self.__calculate_distance_for_2_points(p,p2)
			distances.append([i,d])
			if d<self.config['isolated_cluster']['min_len']:
				min_len_points.append(i)
		point_id = sorted(distances, key=lambda x: x[1])[0][0]
		return X[point_id], min_len_points

	def __check_point(self, p1, p2, F_layers):
		for k,F in enumerate(F_layers):
			if p1[-1]>=F and p2[-1]<F:
				return True, k 
		return False, None

	def __generate_key_points(self, X):
		"""
		Generate key points
		"""
		key_points = []
		key_points_id = []

		F_layers = self.__generate_layers(X)
		X = X[X[:,-1]>self.config['isolated_cluster']['min_dif']]
		skip_points = []
		for i in range(len(X)-1):
			if i in skip_points:
				continue
			point = X[i]
			nearest, min_len_points = self.__find_nearest_point(point, X[i+1:])
			boolean, layer_id = self.__check_point(point, nearest, F_layers)

			if boolean:
				# print(point)
				# print(min_len_points)
				key_points.append(point)
				key_points_id.append(layer_id)
				# print(i)
				skip_points += [i+1+j for j in min_len_points]
				# print(skip_points)
				# print('======='*30)
		return key_points, key_points_id
	
	def __delete_procedure(self, points, global_id):
		indexes = []
		while len(points)>self.max_key_points:
			distances = []
			for i in range(len(points)):
				for j in range(len(points)):
					if i == j:
						continue
					distances.append([i,j,self.__calculate_distance_for_2_points(points[i],points[j])])
			index = sorted(distances, key=lambda x: x[2])[0][0]
			indexes.append(global_id.pop(index))
			points.pop(index)

		return indexes

	def __delete_closed_points(self, points, points_id, global_id):
		indexes = []
		remove = {}
		for key in points_id:
			if key not in remove:
				remove[key] = 1
			else:
				remove[key] += 1
		for key in remove:
			if remove[key]>self.max_key_points:
				list_delete = []
				list_global_id = []
				for i in range(len(points_id)):
					if points_id[i] == key:
						list_delete.append(points[i])
						list_global_id.append(global_id[i])
				indexes += self.__delete_procedure(list_delete, list_global_id)
		return indexes

	def __subclusterig(self, df, type_of_closed= 0, logging_save = False):

		df = df.sort_values('F', ascending=False)
		X = df.values
		self.max_key_points = self.config['subcluster']['max_key_points']*(X.shape[1]-2)


		closed_points, closed_points_id = self.__generate_key_points(X)
		if logging_save:
			log_message = 'Концевые точки'
			write_log(log_message)
			for p in closed_points:
				write_log(str(p))
			log_message = 'Уровни концевых точек'
			write_log(log_message)
			for p in closed_points_id:
				write_log(str(p))
		df = df.sort_values('F', ascending=True)
		X = df.values
		F_matrix = X

		subcluster_result = {int(x[0]):None for x in X}
		subcluster = 0
		start = True

		for k,p in enumerate(tqdm(X)):
			if k == X.shape[0]-1:
				index = self.__calculate_distance(p,X)
				subcluster_result[int(p[0])] = subcluster_result[int(X[index][0])]
				nearest_p = X[index]
				result, value = self.__get_profile(F_matrix, p, nearest_p, logging_save = logging_save) #calculate profile
			else:
				index = self.__calculate_distance(p,X[k:]) # find nearest neighbor
				result, value = self.__get_profile(F_matrix, p, X[k+index,], logging_save = logging_save) #calculate profile
				nearest_p = X[k+index]

			if result != 'common' and result != 'close': # Если точки разные
				if subcluster_result[int(p[0])] == None:
					if not start:
						subcluster += 1
					subcluster_result[int(p[0])] = subcluster
					
			else: # Если точки близки или профиль прошел проверку
				#Обе точки не имеют подкластера-создаем новый
				
				if subcluster_result[int(p[0])] == None and subcluster_result[int(nearest_p[0])] == None:
					if not start:
						subcluster += 1
					subcluster_result[int(p[0])] = subcluster
					subcluster_result[int(nearest_p[0])] = subcluster
					
				#Одна из точек не имеет подкластера или они не совпадают
				elif subcluster_result[int(p[0])] != subcluster_result[int(nearest_p[0])]:
					A = [p]
					A_id = [-1]
					A_id_merge = [-1]
					B = [nearest_p]
					B_id = [-1]
					B_id_merge = [-1]
					A_global_id = []
					B_global_id = []
					#Находим все точки принадлежащие данным подкластерам
					for index, point in enumerate(closed_points):
						if subcluster_result[int(point[0])] == subcluster_result[int(p[0])] and subcluster_result[int(p[0])]!= None:
							A.append(point)
							A_id.append(closed_points_id[index])
							A_id_merge.append(1)
							A_global_id.append(index)
						if subcluster_result[int(point[0])] == subcluster_result[int(nearest_p[0])] and \
							subcluster_result[int(nearest_p[0])]!= None:
							B.append(point)
							B_id.append(closed_points_id[index])
							B_id_merge.append(1)
							B_global_id.append(index)
					# Проверяем их на слияние
					if not self.__merge(A, A_id_merge, B, B_id_merge, F_matrix):#Если разные, то проверяем куда отнести ближайшую точку
						if subcluster_result[int(p[0])] == None:
							if not start:
								subcluster += 1
							subcluster_result[int(p[0])] = subcluster
						status = True
						start_dist = self.__calculate_distance_for_2_points(A[0],B[0])
						for i in range(1,len(B)):
							if self.__calculate_distance_for_2_points(B[0],B[i])<start_dist:
								status = False
								break
						if status:
							subcluster_result[int(nearest_p[0])] = subcluster_result[int(p[0])]
					else:#Если подкластеры сливаются - то все точки анализируемого подкластера сливаются
						if subcluster_result[int(nearest_p[0])] is None:
							subcluster_result[int(nearest_p[0])] = subcluster_result[int(p[0])]
						elif subcluster_result[int(p[0])] is None:
							subcluster_result[int(p[0])] = subcluster_result[int(nearest_p[0])]
						else:
							deleted_cluster = subcluster_result[int(p[0])]
							for p_name in subcluster_result:
								if subcluster_result[p_name] == deleted_cluster:
									subcluster_result[p_name] = subcluster_result[int(nearest_p[0])]
							remove_ids = self.__delete_closed_points(A[1:]+B[1:], A_id[1:]+B_id[1:], A_global_id+B_global_id)
							closed_points_new = []
							closed_points_id_new = []
							for i in range(len(closed_points)):
								if i not in remove_ids:
									closed_points_new.append(closed_points[i])
									closed_points_id_new.append(closed_points_id[i])
							closed_points = copy.deepcopy(closed_points_new)
							closed_points_id = copy.deepcopy(closed_points_id_new)
				
			start = False
		return subcluster_result


	def subclustering(self, df, type_of_closed = 0, logging_save = False):

		if logging_save:
			write_log('НАЧАЛО САБКЛАСТЕРИЗАЦИИ ' + str(datetime.datetime.now()))

		assert 'F' in df.columns, 'ERROR! F not calulated.'
		df['subcluster_id'] = None
		self.max_index_sub = 0

		if 'cluster_id' in df.columns:
			clusters = list(set(df['cluster_id']))
			write_log('Сабкластеризация для {} кластеров'.format(len(clusters)))
			for cluster in clusters:
				current = df[df['cluster_id']==cluster]
				if current.shape[0]<2:#Исправить если в кластере 1 точка
					# print(df[df['cluster_id']==cluster]['subcluster_id'].shape)
					df.at[df[df['cluster_id']==cluster].index, 'subcluster_id'] = self.max_index_sub
					self.max_index_sub += 1
					continue
				need_names = [n for n in current.columns if n not in self.nameignore] 
				current = current[need_names]
				result = self.__subclusterig(current, type_of_closed, logging_save = logging_save)
				self.__norm_sub_index(result)
				for name in result.keys():
					df.at[df[df['id']==name].index, 'subcluster_id'] = result[name]
		else:
			current = df
			need_names = [n for n in current.columns if n not in self.nameignore] 
			current = current[need_names]
			result = self.__subclusterig(current, type_of_closed, logging_save = logging_save)
			self.__norm_sub_index(result)
			for name in result.keys():
				df.at[df[df['id']==name].index, 'subcluster_id'] = result[name]
				
		df = df.sort_values(by=['id'])

		return df
