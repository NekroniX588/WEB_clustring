# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from core.utils import get_F_example

import logging

logging.basicConfig(level=logging.DEBUG, filename='log.log')
logger = logging.getLogger(__name__)

class Clusters():
	def __init__(self, config):
		self.nameignore = ['F', 'cluster_id', 'subcluster_id']
		self.config = config
		self.contur_config = config['conturs']
		self.cluster_config = config['isolated_cluster']

	def __get_contours(self, F):
		"""
		input - F = ['id', 'X1',...,'Xn', 'F']
		"""
		contours = [[]]
		lenghts = []
		i = -1
		while i < len(F)-1:
			i += 1
			# добавляем точку в контур
			contours[-1].append(F[i])
			# решаем, начинать ли добавление новый контур
			if (len(contours[-1]) >= self.contur_config['contour_points']) and \
				(contours[-1][0][-1] - contours[-1][-1][-1] >= self.contur_config['min_diff']):
				contours.append([])
				i -= self.contur_config['min_points']
		contours_lens = []
		k = 0
		# print('contours', contours)
		for contour in contours:
			k += 1
			# считаем среднее по контуру расстояние
			lens = []
			min_lens = []
			for i in range(len(contour)):
				lens.append([])
				for j in range(len(contour)):
					curlen = np.linalg.norm(np.array(contour[i])[1:-1] - np.array(contour[j])[1:-1])
					if contour[j][0]!= contour[i][0] and curlen!= 0:
						lens[i].append(curlen)
				lens[i] = sorted(lens[i])
				for c in range(self.contur_config['num_of_lenghts']):
					min_lens.append(lens[i][c])
			contours_lens.append(np.mean(min_lens))
		return contours, contours_lens

	def get_profile(self, F, p1, p2):
		#Функция рассчета профиля F - матрица формата ['id','x1',...,'xn','F'], p1, p2 - точки формата ['id','x1',...,'xn','F']
		log_message = 'Расчет профиля для точек: {} и {}\n'.format(p1, p2)

		log_message+= 'Расстояние между точками: {}'.format(np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])))

		# if min(p1[-1],p2[-1])<self.cluster_config['min_dif']*self.cluster_config['min_dif']:
		# 	return 'common'

		# if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/self.cluster_config['divider'] <= self.contur_config['min_diff']:
		if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])) <= self.cluster_config['min_len']:
			log_message+= 'Расстояние между точками/divier меньше минимальной разности, поэтому, точки близкие.\n'
			logging.debug(log_message)
			return 'close', 0
		else:
			div_num = 0
			segment_len = np.linalg.norm( np.array(p1[1:-1]) - np.array(p2[1:-1]))
			while (div_num < self.cluster_config['max_div_num']-1) and (segment_len > self.cluster_config['min_len']):
				div_num+= 1
				num_of_segments = (self.cluster_config['divider']+1)*div_num + self.cluster_config['divider']
				segment_len = np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/num_of_segments

			if div_num == 0:
				log_message+= 'Число разбиений получилось равным 0, поэтому, точки близкие.\n'
				logging.debug(log_message)
				return 'close', 0

			points = []
			for i in range(num_of_segments):
				x = [-1.]
				for j in range(1, len(p1)-1):
					x.append((p1[j] + p2[j]*(i+1)/(num_of_segments-i))/(1+(i+1)/(num_of_segments-i)))
				points.append(np.array(x))

			Fs = [[point[1],  point[2], get_F_example([f[:-1] for f in F], self.config['consts']['a'], target=point)] \
																									for point in points]
			Fs = sorted(Fs, key = lambda S: S[-1], reverse = False)

			log_message+= 'Разбиение:\n'
			for F in Fs:
				log_message+= '    {}\n'.format(F)

			Fmin = np.min([F[-1] for F in Fs])
			Fstar = p1[-1] if  p1[-1] < p2[-1] else p2[-1]

			if Fstar - Fmin >= self.cluster_config['min_dif']:
				log_message+= 'Статус: разные кластеры.\n'
				logging.debug(log_message)
				return 'different', Fstar - Fmin
			else:
				log_message+= 'Статус: общий кластер.\n'
				logging.debug(log_message)
				return 'common', Fstar - Fmin



	def get_isolated_clusters(self, df, step=None):
		"""
		input - DataFrame ['id', 'X1', 'X2', ..., 'Xn', 'F'] 
		"""
		#transform df to ['id', 'X1',...,'Xn'] format
		strted_ids = df['id'].min()

		need_names = [n for n in df.columns if n not in self.nameignore] 
		df_correct = df[need_names].copy()

		X = df_correct.iloc[:].values #get numpy array ['id', 'X1',...,'Xn']
		F = np.array(get_F_example(X, self.config['consts']['a']))[:,-1] #Calculate F

		df['F'] = F #add F to first df
		# print(F)
		df_correct['F'] = F #add F to correct df

		F = df_correct.iloc[:].values # ['id', 'X1',...,'Xn', 'F']
		F = sorted(F, key = lambda S: S[-1], reverse = True)

		contours, contours_lens = self.__get_contours(F)

		clusters = []
		clusters_F_max = []

		# вспомогательная функция для поиска кластера, которому принадлежит точка point
		def find_cluster(point):
			for i in range(len(clusters)):
				for p in clusters[i]:
					if p[0] == point[0]:
						return i
			return None

		cur_contour = 0
		merge_times = 1

		# перебираем точки по убыванию функционала
		c = 0
		for cur_point in tqdm(F):
			c+= 1 # номер итерации

			cur_point = np.array(cur_point)
			if cur_point[0] not in [point[0] for point in contours[cur_contour]]:
				cur_contour+= 1

			log_message = 'Точка {}, из контура номер {} (среднее расстояние: {}):\n'.format(cur_point, cur_contour+1, contours_lens[cur_contour])
			
			# если точка первая, кладем ее в первый кластер
			if cur_point[0] == F[0][0]:
				clusters.append([cur_point])
				clusters_F_max.append(cur_point)
				log_message+= '	Это первая точка, поэтому она образовала первый кластер\n'

			# если точка не первая, ищем, куда ее положить
			else:
				log_message+= '	Ищем в каждом кластере точку, ближайшую к данной\n'
				# ищем в каждом кластере точку, ближайшую к данной
				verticles = []
				min_verticles_lens = []
				nvids = []
				clid = 0
				for cluster in clusters:
					lens = []
					len_points = []
					for point in cluster:
						lens.append(np.linalg.norm(cur_point[1:-1] - point[1:-1]))
						len_points.append(point)

					min_v_len = np.min(lens)
					verticle = [point for point in len_points if np.linalg.norm(cur_point[1:-1] - point[1:-1]) == min_v_len][0]
					nvids.append(clid)
					clid+= 1
				
					min_verticles_lens.append(min_v_len)
					verticles.append(verticle)

				neighbors = [verticles[i] for i in range(len(verticles)) \
							 if min_verticles_lens[i] < contours_lens[cur_contour] * self.cluster_config['constU1']]
				min_lens = [min_verticles_lens[i] for i in range(len(min_verticles_lens))\
							if min_verticles_lens[i] < contours_lens[cur_contour] * self.cluster_config['constU1']]
				nids = [nvids[i] for i in range(len(min_lens)) \
						if min_lens[i] < contours_lens[cur_contour] * self.cluster_config['constU1']]

				if len(neighbors) > 1:
					log_message+= '	Выбрано {} точек. Выбираем две из них.\n'.format(len(neighbors))
					logging.debug(log_message)

					# Выбираем два минимальных расстояния, 
					# остальные кластеры на данном шаге уже не рассматриваем
					neighbors = [x for _,x in sorted(zip(min_lens, neighbors))][:2]
					nids = [find_cluster(neighbor) for neighbor in neighbors]
					
					profile_1, difs1 = self.get_profile(F, neighbors[0], cur_point)
					profile_2, difs2 = self.get_profile(F, neighbors[1], cur_point)

					len1 = np.linalg.norm(cur_point-neighbors[0])
					len2 = np.linalg.norm(cur_point-neighbors[1])
					
					log_message = 'Выбрано две ближайшие точки:\n'
					log_message+= '	1. Точка {} из кластера {}, профиль: {} - {}\n'.format(neighbors[0], nids[0], difs1, profile_1)
					log_message+= '	Точка с Fmax в кластере {}: {}\n'.format(nids[0], clusters_F_max[nids[0]])
					log_message+= '	2. Точка {} из кластера {}, профиль: {} - {}\n'.format(neighbors[1], nids[1], difs2, profile_2)
					log_message+= '	Точка с Fmax в кластере {}: {}\n'.format(nids[1], clusters_F_max[nids[1]])

					if min(clusters_F_max[nids[0]][-1]-cur_point[-1], clusters_F_max[nids[1]][-1]-cur_point[-1]) < self.cluster_config['min_dif']:
						log_message+= '	Слияние кластеров номер {}:'.format(merge_times)
						
						log_message+= '	Кластер {}:\n'.format(nids[0])
						for t in clusters[find_cluster(neighbors[0])]:
							log_message+= '	{}\n'.format(t)
						log_message+= '	Кластер {}:'.format(nids[1])
						for t in clusters[find_cluster(neighbors[1])]:
							log_message+= '	{}\n'.format(t)

						merge_times+= 1

						cluster_index = nids[0 if clusters_F_max[nids[0]][-1] > clusters_F_max[nids[1]][-1] else 1]
						del_index = nids[1 if clusters_F_max[nids[0]][-1] > clusters_F_max[nids[1]][-1] else 0]

						clusters[cluster_index].append(cur_point)

						if clusters_F_max[cluster_index][-1] >= clusters_F_max[del_index][-1]:
							clusters_F_max[cluster_index] = clusters_F_max[cluster_index]
						else:
							clusters_F_max[cluster_index] = clusters_F_max[del_index]

						for cluster_point in clusters[del_index]:
							clusters[cluster_index].append(cluster_point)

						del clusters[del_index]
						del clusters_F_max[del_index]
						
					else:
						log_message+= '	Кластеры не сливаются: {} >= {}\n'.format(min(clusters_F_max[nids[0]][-1]-cur_point[-1], clusters_F_max[nids[1]][-1]-cur_point[-1]), self.cluster_config['min_dif'])
						cluster_index = nids[0] if len1 <= len2 else nids[1]
						log_message+= '	Точка добавлена в кластер №{}\n'.format(cluster_index+1)
						clusters[cluster_index].append(cur_point)

				elif len(neighbors) == 1:
					log_message+= '	Выбрана одна ближайшая точка: {}'.format(neighbors[0])
					log_message+= '	Точка добавлена в кластер №{}:\n'.format(find_cluster(neighbors[0])+1)
					for t in clusters[find_cluster(neighbors[0])]:
						log_message+= '{}\n'.format(t)
					clusters[find_cluster(neighbors[0])].append(cur_point)

				else:
					candidates = []
					log_message+= '	Ищем подходящие точки среди точек с Fmax в каждом кластере\n'
					logging.debug(log_message)
					for i in range(len(clusters)):

						F_max = clusters_F_max[i][-1]
						F_max_point = clusters_F_max[i]
						cur_profile, cur_dif = self.get_profile(F, F_max_point, cur_point)
						if cur_profile == 'close' or cur_profile == 'common':
							curlen = np.linalg.norm(np.array(cur_point[1:-1]) - np.array(F_max_point[1:-1]))
							candidates.append(i)

					log_message = 'Близкие точки найдены в кластерах: {}. Ищем близкую точку в каждом из этих кластеров\n'.format(candidates)
						
					# в каждом кластере-кандидате ищем ближайшую точку
					closest_points = {}
					closest_point = []
					for candidate in candidates:
						closest_points[candidate] = []
						minlen = -1
						for point in clusters[candidate]:
							curlen = np.linalg.norm(np.array(cur_point[1:-1]) - np.array(point[1:-1]))
							if minlen == -1 or curlen < minlen:
								minlen = curlen
								closest_points[candidate] = curlen
								closest_point = point
						log_message+= '	В кластере {} ближайшая точка: {}, расстояние: {}\n'.format(candidate, closest_point, minlen)

					log_message+= '	Близкие точки найдены в {} кластерах. Проверяем профили с каждой из них\n'.format(len(closest_points))

					closest_cluster = -1
					minlen = -1
					for key in closest_points.keys():
						if minlen == -1 or closest_points[key] < minlen:
							closest_cluster = key
							minlen = closest_points[key]

					if closest_cluster!= -1:
						clusters[closest_cluster].append(cur_point)
						log_message+= '	\nТочка добавлена в кластер №{}:\n'.format(closest_cluster+1)
						for t in clusters[closest_cluster]:
							log_message+= '{}\n'.format(t)

					else:
						log_message+= '	задает новый кластер\n'.format(cur_point)
						clusters.append([cur_point])
						clusters_F_max.append(cur_point)
			
			logging.debug(log_message)

		if step == None:
			df['cluster_id'] = 0
			for idx, cluster in enumerate(clusters):
			    for point in cluster:
			        df.at[df[df['id']==int(point[0])].index, 'cluster_id'] = idx
			df = df.sort_values(by=['id'])
		else:
			for idx, cluster in enumerate(clusters):
			    for point in cluster:
			        df.at[df[df['id']==int(point[0])].index, 'cluster_id_'+str(step)] = self.max_index + idx
			df = df.sort_values(by=['id'])
			self.max_index = self.max_index + idx + 1
		return df