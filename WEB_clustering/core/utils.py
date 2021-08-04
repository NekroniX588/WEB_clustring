# -*- coding: utf-8 -*-
import numpy as np

def get_F_example(data, a, target=[]):#add attribute of norm on consts
	if len(target)==0:
		F = []
		for cur_point in data:
			# cur_point = np.append(cur_point, np.sum([( np.linalg.norm(point[1:] - cur_point[1:]) + a)**(-2) for point in data]))
			cur_point = np.append(cur_point, np.sum([a/(np.linalg.norm(point[1:] - cur_point[1:])**2 + a) for point in data])/(data.shape[0]))
			# cur_point = np.append(cur_point, np.sum([(np.linalg.norm(point[1:] - cur_point[1:])**2 + a)**(-1) for point in data]))
			F.append(cur_point)
		return F
	else:
		return np.sum([a/(np.linalg.norm(point[1:] - target[1:])**2 + a) for point in data])/len(data)

def get_F(data, a, target=[]):
	if len(target)==0:
		F = []
		for i, cur_point in enumerate(data):
		    line = [( np.linalg.norm(point[1:] - cur_point[1:])**2 + a)**(-1) for point in data]
		    cur_point = np.append(cur_point, np.sum(line[:i] + line[i+1:])/(data.shape[0]))
		    F.append(cur_point)
		return F
	else:
		return np.sum([(np.linalg.norm(point[1:] - target[1:]) + a)**(-2) for point in data])

class GetProfile():
	def __init__(self, config):
		self.config = config
		if 'ignore_coord' in self.config:
			self.nameignore = ['cluster_id', 'subcluster_id'] + self.config['ignore_coord']
		else:
			self.nameignore = ['cluster_id', 'subcluster_id']

	def get_profile(self, df, p1_id, p2_id):
		need_names = [n for n in df.columns if n not in self.nameignore] 
		df_correct = df[need_names].copy()
		
		F = df_correct.values
		p1 = df_correct.loc[df_correct['id']==p1_id].values[0]
		p2 = df_correct.loc[df_correct['id']==p2_id].values[0]

		if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])) <= self.config['isolated_cluster']['min_len']:
			return 'Точки близки', []
		else:
			div_num = 0
			segment_len = np.linalg.norm( np.array(p1[1:-1]) - np.array(p2[1:-1]))
			while (div_num < self.config['isolated_cluster']['max_div_num']-1) and (segment_len > self.config['isolated_cluster']['min_len']):
				div_num+= 1
				num_of_segments = (self.config['isolated_cluster']['divider']+1)*div_num + self.config['isolated_cluster']['divider']
				segment_len = np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/num_of_segments

			if div_num == 0:
				return 'Точки близки', []

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

			return 'Точки далеки', Fs
