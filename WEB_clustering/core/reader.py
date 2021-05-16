import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
import numpy as np 

class Reader(object):
	def __init__(self):
		self.types = ['int', 'int16', 'int32', 'int64', 'float', 'float16', 'float32', 'float64']
		self.nameignore = ['id', 'F', 'cluster_id', 'subcluster_id']

	def statistic(self, df,  num_of_intervals=10):#Add distances
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
			d = pairwise_distances(x[:, np.newaxis]).ravel()
			text += str(col) + ' contain absolute:' + str(d[d==0].shape[0]) + '\n'
			text += str(col) + ' contain relation:' + str(round(d[d==0].shape[0]/d.shape[0],5))+ '\n'
			d.sort()
			start = 0
			finish = len(d)//num_of_intervals
			step = len(d)//num_of_intervals
			for i in range(num_of_intervals-1):
				text += 'Interval:' + str(d[start:finish].mean()) + '\n'
				start = finish
				finish += step
			text += 'Interval:' + str(d[start:].mean()) + '\n'

		return text, df

	def split_data(self, df, train_size = 10, shuffle=True):
		df_train, df_test = train_test_split(df, train_size=train_size/100, shuffle=shuffle)
		return df_train, df_test

	def read(self, file_path):
		'''
		args:
		file_path -- path to the file

		returns:
		dataframe if format is correct
		None otherwise
		'''
		file_extension = file_path.split('.')[-1]
		if file_extension == 'csv':
			in_df = pd.read_csv(file_path) 
		elif file_extension == 'xlsx' or file_extension == 'xls':
			in_df = pd.read_excel(file_path) 
		else:
			print('Unsupported format')
			return
		if self.is_valid(in_df):
			return in_df
		else:
			print('Unsupported format of file')
			return

	def is_valid(self, df):
		'''
		Checks if input dataframe's columns are in the right format
		mode -- read or write 
		'''
		coords_nums = [int(col_name[1:]) for col_name in df.columns[1:]]
		#Add information ('F', Cluster_Id, Subcluster)
		is_asc = coords_nums == sorted(coords_nums)
		if is_asc == False:
			print('Wrong column name format')
			return False
		types = df.dtypes
		for i, col_name in enumerate(df.columns):
			# print(col_name[0])
			if i == 0 and col_name != 'id':
				print('Wrong column name format')
				return False
			assert types[col_name] in self.types, "Bad type of column "
		return True

	def write(self, out_df, file_path):
		'''writes df to file, returns nothing or None'''
		file_extension = file_path.split('.')[1]
		if file_extension == 'csv':
			out_df.to_csv(file_path, index=False) 
		elif file_extension == 'xlsx' or file_extension == 'xls':
			out_df.to_excel(file_path, index=False) 
		else:
			print('Unsupported format')
			return



