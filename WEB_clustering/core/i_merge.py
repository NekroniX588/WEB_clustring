from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import random
import numpy as np
from core.utils import get_F_example
import pandas as pd

import datetime

def write_log(str):
	with open('merging.log', 'a', encoding='utf-8') as f:
		f.write(str+'\n')

class IMerger():
	def __init__(self, config):
		self.config = config

	def __get_profile(self, F, p1, p2, logging_save = False):

		if logging_save:
			log_message = 'Профиль для пары точек'
			write_log(log_message)
			write_log(str(p1))
			write_log(str(p2))

		if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])) <= self.config['isolated_cluster']['min_len']:
			return min(p1[-1],p2[-1])
		else:
			div_num = 0
			segment_len = np.linalg.norm( np.array(p1[1:-1]) - np.array(p2[1:-1]))
			while (div_num < self.config['isolated_cluster']['max_div_num']-1)\
					 and (segment_len > self.config['isolated_cluster']['min_len']):
				div_num+= 1
				num_of_segments = (self.config['isolated_cluster']['divider']+1)*div_num\
								  + self.config['isolated_cluster']['divider']
				segment_len = np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/num_of_segments

			if div_num == 0:
				return min(p1[-1],p2[-1])

			points = []
			for i in range(num_of_segments):
				x = [-1.]
				for j in range(1, len(p1)-1):
					x.append((p1[j] + p2[j]*(i+1)/(num_of_segments-i))/(1+(i+1)/(num_of_segments-i)))
				points.append(np.array(x))
			# Fs = [[point[1],  point[2], get_F_example([f[:-1] for f in F], self.config['consts']['a'], target=point)] for point in points]
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

			min_F = Fs[0][-1]
			for s in Fs:
				if s[-1]<min_F:
					min_F = s[-1]
			return min_F

	def mergeClusters(self, df, logging_save = False):
		if logging_save:
			write_log('НАЧАЛО МЕРДЖИНГА ' + str(datetime.datetime.now()))

		if 'cluster_id' not in df.columns:
			if logging_save:
				write_log("Не расчитаны кластеры")
			return df
		elif len(set(df['cluster_id'])) <= 1:
			if logging_save:
				write_log("Только 1 кластер")
			return df

		num_clusters = len(set(df['cluster_id']))

		matrix = df.iloc[:, 1:-2].values
		map_index = {index:k for k,index in enumerate(df['id'].values)}
		dist_matrix = np.zeros((len(matrix), len(matrix)))
		for i in range(len(matrix)):
			for j in range(len(matrix)):
				dist_matrix[i,j] = distance.euclidean(matrix[i], matrix[j])

		id_cluster_matrix = df[['id', 'cluster_id']].iloc[:, ].values
		dist_arr = []
		for line1 in id_cluster_matrix:
			a_dist = []
			# набираем расстояния
			for line2 in id_cluster_matrix:
				# если тот же кластер - лесом
				
				if line1[1] == line2[1]:
					continue
				a_dist.append(
					[line1[0], dist_matrix[map_index[int(line1[0])], map_index[int(line2[0])]], line2[0], line2[1]])
			a_dist.sort(key=lambda i: float(i[1]), reverse=False)
			dist_arr.append(a_dist[0])

		if logging_save:
			write_log("Оставляем только межкластерные связи")
			for print_item in dist_arr:
				write_log(str(print_item))

		good_dots = []
		for b in dist_arr:
			for c in dist_arr:
				if b[0] == c[2] and b[1] == c[1]:
					good_dots.append(b)

		data = []
		for g in good_dots:
			dot1 = df[df['id']==g[0]]
			dot2 = df[df['id']==g[2]]
			f_max_clus_1 = df[df['cluster_id']==dot1['cluster_id'].values[0]]['F'].max()
			f_max_clus_2 = df[df['cluster_id']==dot2['cluster_id'].values[0]]['F'].max()
			data.append({'p1':dot1.values[0,:], 'F_max_cluster1':f_max_clus_1, 'p2':dot2.values[0,:], 'F_max_cluster2':f_max_clus_2})
		
		if logging_save:
			write_log("Хорошие точки")
			for print_item in data:
				write_log(str(print_item))
		i = 0
		while i<len(data):
			for j in range(len(data)):
				if data[i]['p1'][0] == data[j]['p2'][0] and j!=i:
					data.pop(j)
					break
			i += 1

		if logging_save:
			write_log("Очищенные хорошие точки")
			for print_item in data:
				write_log(str(print_item))

		while True:
			F = df.iloc[:].values[:,:-1]
			F_matrix = np.full((num_clusters,num_clusters), np.inf)
			for i in range(len(data)):
				if data[i]['F_max_cluster1'] <= data[i]['F_max_cluster2']:
					F_max = data[i]['F_max_cluster1']
					x = int(data[i]['p2'][-1])
					y = int(data[i]['p1'][-1])
				else:
					F_max = data[i]['F_max_cluster2']
					x = int(data[i]['p1'][-1])
					y = int(data[i]['p2'][-1])
				if x == y:
					continue
				delta = F_max - self.__get_profile(F, data[i]['p1'][:-1], data[i]['p2'][:-1], logging_save = logging_save)
				if F_matrix[x, y] == np.inf:
					F_matrix[x, y] = delta
				elif F_matrix[x, y] > delta:
					F_matrix[x, y] = delta

			print(F_matrix)
	
			min_item = F_matrix.argmin()
			row = min_item // num_clusters
			col = min_item % num_clusters
			print(F_matrix[row,col])
			if F_matrix[row,col] > self.config['isolated_cluster']['merge_threshold']:
				break
			F_matrix[row,col]=np.inf
			df.loc[df['cluster_id']==col, 'cluster_id'] = row
			for point_item in data:
				if point_item['p1'][-1] == col:
					point_item['p1'][-1] = row
					point_item['F_max_cluster1'] = df[df['cluster_id']==row]['F'].max()
				if point_item['p2'][-1] == col:
					point_item['p2'][-1] = row
					point_item['F_max_cluster2'] = df[df['cluster_id']==row]['F'].max()

		indexes = sorted(list(set(df['cluster_id'].values)))
		index_map = {index:k for k,index in enumerate(indexes)}
		new_indexes = []
		for inds in df['cluster_id'].values:
			new_indexes.append(index_map[inds])
		df['cluster_id'] = new_indexes
		return df
