import numpy as np
from tqdm import tqdm
from core.utils import get_F_example

class Subclusters(object):

	def __init__(self, config):
		self.config = config
		self.nameignore = ['cluster_id', 'subcluster_id']
		

	def __get_profile(self, F, p1, p2):
		# print(p1)
		#Функция рассчета профиля F - матрица формата ['id','x1',...,'xn','F'], p1, p2 - точки формата ['id','x1',...,'xn','F']
		# if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/self.config['isolated_cluster']['divider'] <= self.config['conturs']['min_diff']:
		if np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1])) <= self.config['isolated_cluster']['min_len']:
			return 'close', 0
		else:
			div_num = 0
			segment_len = np.linalg.norm( np.array(p1[1:-1]) - np.array(p2[1:-1]))
			while (div_num < self.config['isolated_cluster']['max_div_num']-1) and (segment_len > self.config['isolated_cluster']['min_len']):
				div_num+= 1
				num_of_segments = (self.config['isolated_cluster']['divider']+1)*div_num + self.config['isolated_cluster']['divider']
				segment_len = np.linalg.norm(np.array(p1[1:-1]) - np.array(p2[1:-1]))/num_of_segments

			if div_num == 0:
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
			Fs = [[point[1],  point[2], get_F_example([f[:-1] for f in F], self.config['consts']['a'], target=point)] for point in points]#&&&&&&&&&&&???????????
			Fs = sorted(Fs, key = lambda S: S[-1], reverse = False)
	#         print(Fs)

			Fmin = np.min([F[-1] for F in Fs])
			Fstar = p1[-1] if  p1[-1] < p2[-1] else p2[-1]

			if Fstar - Fmin >= self.config['isolated_cluster']['min_dif']:
				return 'different', Fstar - Fmin
			else:
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

	def __check_levels(self, p, connected):
		"""
		p - [id, X1,...,Xn,F]
		connected - list of points [id, X1,...,Xn,F]
		layers - list
		========
		return - bool value, point is closed or not
		"""
		if len(connected) == 0:
			return True
		curret_layer = None
		for i in range(len(self.layers_F)):
			if p[-1]<self.layers_F[i]:
				curret_layer = self.layers_F[i-1]
				break
		if curret_layer is None:
			curret_layer = self.layers_F[-1]
		for p_c in connected:
			if p_c[-1]>curret_layer:
				return False
		return True

	def __get_layer(self, p):
		curret_layer = None
		for i in range(len(self.layers_F)):
			if p[-1]<self.layers_F[i]:
				curret_layer = self.layers_F[i-1]
				break
		if curret_layer is None:
			curret_layer = self.layers_F[-1]
		return curret_layer


	def __select_points(self, A, point):
		"""
		A - list of points [id, X1,...,Xn,F]
		"""
		dist = {}
		main_layer = self.__get_layer(A[0])
		for i in range(1,len(A)):
			d = self.__calculate_distance_for_2_points(point, A[i])
			point_layer = self.__get_layer(A[i])
			if main_layer>point_layer:
				if point_layer not in dist:
					dist[point_layer] = [d, A[i]]
				else:
					if dist[point_layer][0] < d:
						dist[point_layer] = [d, A[i]]
		new_A = [A[0]]
		for name in dist.keys():
			new_A.append(dist[name][1])
		return new_A

	def __calculate_distance_for_2_points(self, A,B):
		"""
		A - [id, X1,...,Xn,F]
		B - [id, X1,...,Xn,F]
		=========
		return - distance between 2 points
		"""
		return np.linalg.norm(np.array(A[1:-1]) - np.array(B[1:-1]))
	
	def __merge(self, A,B, F_matrix):
		"""
		A - list of points [id, X1,...,Xn,F]
		B - list of points [id, X1,...,Xn,F]
		F_matrix - [[id, X1,...,Xn,F],...,[id, X1,...,Xn,F]]
		=======
		return - bool value, status for merging clusters
		"""
		for p_1 in A:
			for p_2 in B:
				result, _ = self.__get_profile(F_matrix, p_1, p_2)
	#             print(result)
				if result == 'different':
					return False
		return True

	def __generate_layers(self, X):
		"""
		Generate F layers, due to F_step from settings
		"""
		self.layers_F = [self.config['subcluster']['F_step']]
		current = 2*self.config['subcluster']['F_step']
		while current<X[:,-1].max():
			self.layers_F.append(current)
			current += self.config['subcluster']['F_step']

	def __norm_sub_index(self, results):
		now_index = sorted(list(set(results.values())))
		transform = {index:self.max_index_sub+k for k,index in enumerate(now_index)}
		self.max_index_sub += len(now_index)
		for name in results:
			results[name] = transform[results[name]]

	def __subclusterig(self, df, type_of_closed= 0):
		df = df.sort_values('F')
		X = df.values
		F_matrix = X
		F_matrix.shape
		self.__generate_layers(X)

		subcluster_result = {int(x[0]):None for x in X}
		connected_points = {int(x[0]):[] for x in X}
		closed_points = []
		subcluster = 0
		start = True

		for k,p in enumerate(tqdm(X)):
			if k == X.shape[0]-1:
				index = self.__calculate_distance(p,X)
				subcluster_result[int(p[0])] = subcluster_result[int(X[index][0])]
				nearest_p = X[index]
				result, value = self.__get_profile(F_matrix, p, nearest_p) #calculate profile
			else:
				index = self.__calculate_distance(p,X[k:]) # find nearest neighbor
				result, value = self.__get_profile(F_matrix, p, X[k+index,]) #calculate profile
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
					B = [nearest_p]
					#Находим все точки принадлежащие данным подкластерам
					for point in closed_points:
						if subcluster_result[int(point[0])] == subcluster_result[int(p[0])] and subcluster_result[int(p[0])]!= None:
							A.append(point)
						if subcluster_result[int(point[0])] == subcluster_result[int(nearest_p[0])] and \
							subcluster_result[int(nearest_p[0])]!= None:
							B.append(point)
					if type_of_closed == 2:
						A = self.__select_points(A,A[0])
						B = self.__select_points(B,B[0])
					# Проверяем их на слияние
					if not self.__merge(A,B, F_matrix):#Если разные, то проверяем куда отнести ближайшую точку
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
					else:#Если подкластеры сливаются - то всем точкам анализируемого подкластера сливаются
						if subcluster_result[int(nearest_p[0])] is None:
							subcluster_result[int(nearest_p[0])] = subcluster_result[int(p[0])]
						else:
							if subcluster_result[int(p[0])] is None:
								subcluster_result[int(p[0])] = subcluster_result[int(nearest_p[0])]
							else:
								deleted_cluster = subcluster_result[int(p[0])]
								for p_name in subcluster_result:
									if subcluster_result[p_name] == deleted_cluster:
										subcluster_result[p_name] = subcluster_result[int(nearest_p[0])]
				connected_points[int(nearest_p[0])].append(p)
				
			start = False
			if type_of_closed == 0:
				closed_points.append(p)
			elif type_of_closed == 1 or type_of_closed == 2:
				if self.__check_levels(p, connected_points[int(p[0])]):
					closed_points.append(p)
			else:
				closed_points.append(p)
		return subcluster_result


	def subclustering(self, df, type_of_closed = 0):

		assert 'F' in df.columns, 'ERROR! F not calulated.'
		df['subcluster_id'] = None
		self.max_index_sub = 0
		strted_ids = df['id'].min()

		if 'cluster_id' in df.columns:
			clusters = list(set(df['cluster_id']))
			print('Calculate for ', len(clusters), 'clusters')
			for cluster in clusters:
				current = df[df['cluster_id']==cluster]
				if current.shape[0]<2:#Исправить если в кластере 1 точка
					continue
				need_names = [n for n in current.columns if n not in self.nameignore] 
				current = current[need_names]
				result = self.__subclusterig(current, type_of_closed)
				self.__norm_sub_index(result)
				for name in result.keys():
					df.at[df[df['id']==name].index, 'subcluster_id'] = result[name]
		else:
			current = df
			need_names = [n for n in current.columns if n not in self.nameignore] 
			current = current[need_names]
			result = self.__subclusterig(current, type_of_closed)
			self.__norm_sub_index(result)
			for name in result.keys():
				df.at[df[df['id']==name].index, 'subcluster_id'] = result[name]
				
		df = df.sort_values(by=['id'])

		return df
