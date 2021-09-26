import pandas as pd
import math
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
import numpy as np 

class Reader(object):
	def __init__(self):
		self.types = ['int', 'int16', 'int32', 'int64', 'float', 'float16', 'float32', 'float64']
		self.nameignore = ['id', 'F', 'cluster_id', 'subcluster_id']

	def statistic(self, df, num_of_intervals=10):#Add distances
		text = ''
		text += 'Stated size of data ' + str(df.shape[0]) + '\n'
		start = df.shape[0]
		df = df.dropna()
		finish = df.shape[0]
		text += 'Data contained ' + str(start-finish) + ' incorrect rows\n' 
		text += 'Size of data ' + str(finish) + '\n'

		need_names = [n for n in df.columns if n not in self.nameignore] 
		df_correct = df[need_names]

		for col in df_correct.columns:
			x = df_correct[col].values
			d = pairwise_distances(x[:, np.newaxis]).ravel()#Удалить главную диагональ
			d_abs = (d[d==0].shape[0]-df.shape[0])//2
			d_rel = ((d[d==0].shape[0]-df.shape[0])//2)/((d.shape[0]-df.shape[0])//2)
			text += str(col) + ' contain zeros distance absolute:' + str(d_abs) + '\n'
			text += str(col) + ' contain zeros distance relation:' + str(round(d_rel,5))+ '\n'
			d.sort()
			start = 0
			finish = len(d)/num_of_intervals
			step = len(d)/num_of_intervals
			for i in range(num_of_intervals-1):
				text += 'Interval:' + str(d[math.floor(start):math.floor(finish)].mean()) + '\n'
				start = finish
				finish += step
			text += 'Interval:' + str(d[math.floor(start):].mean()) + '\n'
			text += '*'*20 + '\n'
		text += '='*20 + '\n'
		return text, df

	def split_data(self, df, train_size = 10, shuffle=True):
		df_train, df_test = train_test_split(df, train_size=train_size/100, shuffle=shuffle)
		return df_train, df_test

	def read(self, file_path, first_open = False):
		'''
		args:
		file_path -- path to the file

		returns:
		dataframe if format is correct
		None otherwise
		'''

		file_extension = file_path.split('.')[-1]
		if file_extension == 'csv':
			try:
				in_df = pd.read_csv(file_path) 
			except Exception as e:
				return "Проблемы с файлом"
		elif file_extension == 'xlsx' or file_extension == 'xls':
			try:
				in_df = pd.read_excel(file_path) 
			except Exception as e:
				return None, "Проблемы с файлом"
		else:
			return None, "Неверное расширение файла"

		if first_open:
			if self.is_valid(in_df)[0]:
				return in_df, self.is_valid(in_df)[1]
			else:
				return None, self.is_valid(in_df)[1]
		else:
			return in_df

	def is_valid(self, df):
		'''
		Checks if input dataframe's columns are in the right format
		mode -- read or write 
		'''
		if df.shape[0] < 10:
			return False, "Слишком мало данных (меньше 10 строк)"
		if df.shape[0] > 2000:
			return False, "Слишком много данных (более 2000 строк)"

		if "id" not in df.columns:
			return False, "Нет столбца id"

		if df["id"].shape[0] != df['id'].unique().shape[0]:
			return False, "Не все id уникальные"

		coords_nums = [int(col_name[1:]) for col_name in df.columns if col_name not in self.nameignore]
		if len(coords_nums) > 50:
			return False, "Слишком много параметров X (больше 50)"
		#Add information ('F', Cluster_Id, Subcluster)
		is_asc = coords_nums == sorted(coords_nums)
		if is_asc == False:
			return False, "Ошибка данных"
		types = df.dtypes
		for i, col_name in enumerate(df.columns):
			# print(col_name[0])
			if i == 0 and col_name != 'id':
				return False, 'Id должно быть первым столбцом'
			if types[col_name] not in self.types:
				return False, "Не верный формат столбцов (проверьте на присутствие символов)"
		return True, "Данные корректны"

	def write(self, out_df, file_path):
		print(file_path)
		'''writes df to file, returns nothing or None'''
		file_extension = file_path.split('.')[-1]
		print(file_extension)
		if file_extension == 'csv':
			out_df.to_csv(file_path, index=False) 
		elif file_extension == 'xlsx' or file_extension == 'xls':
			out_df.to_excel(file_path, index=False) 
		else:
			print('Unsupported format')
			return



