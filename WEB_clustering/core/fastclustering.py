# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from core.utils import get_F_example

import logging
import datetime

np.set_printoptions(suppress=True)

class Fast_Clusters():
	def __init__(self, config):
		self.config = config
		self.contur_config = config['conturs']
		self.cluster_config = config['isolated_cluster']
		if 'ignore_coord' in self.config:
			self.nameignore = ['F', 'cluster_id', 'subcluster_id'] + self.config['ignore_coord']
		else:
			self.nameignore = ['F', 'cluster_id', 'subcluster_id']

	def get_profile(self, F, p1, p2, logging_save = False):
		#Функция рассчета профиля F - матрица формата ['id','x1',...,'xn','F'], p1, p2 - точки формата ['id','x1',...,'xn','F']

		# if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/self.cluster_config['divider'] <= self.contur_config['min_diff']:
		if logging_save:
			log_message = 'Профиль для пары точек'
			logging.debug(log_message)
			logging.debug(str(p1))
			logging.debug(str(p2))

		if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])) <= self.cluster_config['min_len']:
			if logging_save:
				log_message = 'Расстояние между точками меньше min_len'
			return 'close', 0
		else:
			div_num = 0
			segment_len = np.linalg.norm( np.array(p1[1:-1]) - np.array(p2[1:-1]))
			while (div_num < self.cluster_config['max_div_num']-1) and (segment_len > self.cluster_config['min_len']):
				div_num+= 1
				num_of_segments = (self.cluster_config['divider']+1)*div_num + self.cluster_config['divider']
				segment_len = np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/num_of_segments

			if div_num == 0:
				if logging_save:
					log_message = 'Не удалось разбить отрезок'
				return 'close', 0

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
				
			Fs = sorted(Fs, key = lambda S: S[-1], reverse = False)

			if logging_save:
				log_message = 'Последовательность проведенных точек'
				logging.debug(log_message)
				for p in Fs:
					logging.debug(str(p))

			Fmin = np.min([F[-1] for F in Fs])
			Fstar = p1[-1] if  p1[-1] < p2[-1] else p2[-1]

			if Fstar - Fmin >= self.cluster_config['min_dif']:
				if logging_save:
					log_message = 'Просадка выше min_dif, статус different'
					logging.debug(log_message)
				return 'different', Fstar - Fmin
			else:
				if logging_save:
					log_message = 'Просадка ниже min_dif, статус common'
					logging.debug(log_message)
				return 'common', Fstar - Fmin


	def multistep_clustering(self, df, num_steps=None):
		self.max_index = 0
		df['cluster_id_'+str(0)] = 0
		self.get_isolated_clusters(df, 0)
		for step in range(1,num_steps):
			self.max_index = 0
			df['cluster_id_'+str(step)] = 0
			for cluster in list(set(df['cluster_id_'+str(step-1)])):
				self.get_isolated_clusters(df[df['cluster_id_'+str(step-1)]==cluster], step, cluster)
		return df

	def get_isolated_clusters(self, df, step=None, cluster_num=None, logging_save = False):
		"""
		input - DataFrame ['id', 'X1', 'X2', ..., 'Xn', 'F'] 
		"""
		#transform df to ['id', 'X1',...,'Xn'] format
		if logging_save:
			logging.basicConfig(level=logging.DEBUG, filename='fastclustering.log')
			log_message = 'НАЧАЛО БЫСТРОЙ КЛАСТЕРИЗАЦИИ ' + str(datetime.datetime.now())

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

		clusters = {}
		current_cluster = 0

		for cur_point in tqdm(F):
			if len(clusters)==0:
				clusters[current_cluster] = [cur_point]
				current_cluster += 1


			status = True
			for cluster in clusters:
				for p in clusters[cluster]:
					d = np.linalg.norm(np.array(p[1:-1]) - np.array(cur_point[1:-1]))
					if status:
						status = False
						min_d = d
						min_p = p
						min_c = cluster
					elif d<min_d:
						min_d = d
						min_p = p
						min_c = cluster
			result, _ = self.get_profile(F, cur_point, min_p, logging_save = logging_save)
			if result == 'different':
				clusters[current_cluster] = [cur_point]
				current_cluster += 1
			else:
				clusters[min_c].append(cur_point)

		if step == None:
			df['cluster_id'] = 0
			for idx, cluster in enumerate(clusters):
				for point in clusters[cluster]:
					df.at[df[df['id']==int(point[0])].index, 'cluster_id'] = cluster
			df = df.sort_values(by=['id'])
		else:
			for idx, cluster in enumerate(clusters):
				for point in clusters[cluster]:
					df.at[df[df['id']==int(point[0])].index, 'cluster_id_'+str(step)] = cluster
			df = df.sort_values(by=['id'])
			self.max_index = self.max_index + idx + 1
		return df