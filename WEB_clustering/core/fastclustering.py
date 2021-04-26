# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from utils import get_F_example

import logging

class Fast_Clusters():
	def __init__(self, config):
		self.nameignore = ['F', 'cluster_id', 'subcluster_id']
		self.config = config
		self.contur_config = config['conturs']
		self.cluster_config = config['isolated_cluster']


	def get_profile(self, F, p1, p2):
		#Функция рассчета профиля F - матрица формата ['id','x1',...,'xn','F'], p1, p2 - точки формата ['id','x1',...,'xn','F']

		# if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/self.cluster_config['divider'] <= self.contur_config['min_diff']:
		if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])) <= self.cluster_config['min_len']:
			return 'close', 0
		else:
			div_num = 0
			segment_len = np.linalg.norm( np.array(p1[1:-1]) - np.array(p2[1:-1]))
			while (div_num < self.cluster_config['max_div_num']-1) and (segment_len > self.cluster_config['min_len']):
				div_num+= 1
				num_of_segments = (self.cluster_config['divider']+1)*div_num + self.cluster_config['divider']
				segment_len = np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/num_of_segments

			if div_num == 0:
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

			Fmin = np.min([F[-1] for F in Fs])
			Fstar = p1[-1] if  p1[-1] < p2[-1] else p2[-1]

			if Fstar - Fmin >= self.cluster_config['min_dif']:
				return 'different', Fstar - Fmin
			else:
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

	def get_isolated_clusters(self, df, step=None, cluster_num=None):
		"""
		input - DataFrame ['id', 'X1', 'X2', ..., 'Xn', 'F'] 
		"""
		#transform df to ['id', 'X1',...,'Xn'] format
		strted_ids = df['id'].min()

		need_names = [n for n in df.columns if n not in self.nameignore] 
		df_correct = df[need_names].copy()

		X = df_correct.iloc[:].values #get numpy array ['id', 'X1',...,'Xn']
		F = np.array(get_F_example(X, self.config['consts']['a']))[:,-1] #Calculate F
		print(step, cluster_num, X.shape)
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
			result, _ = self.get_profile(F, cur_point, min_p)
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