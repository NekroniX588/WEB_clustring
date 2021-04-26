import numpy as np
import pandas as pd
from functional import get_F, get_F_example

from sklearn.metrics import pairwise_distances
from const import calculate_const

def norm(df, mean):
	for name in mean:
		df[name] /= mean[name]
	return df

def predict(output_name, predict_name, mean={'X1':6, 'X2':8}, a=None):

	dfo = pd.read_csv(output_name, sep=';')#Считываем обучающие данные: формат [id, X1, .., Xn, Cluster_id, i, Subcluster_id]

	dfi = pd.read_csv(predict_name, sep=';')#Считываем данные для предсказания: формат [id, X1, .., Xn]

	names_ouput = list(dfo)[1:]#Выделяем все имена из обучающих данных  
	names_input = list(dfi)[1:]#Выделяем все имена из данных для предсказания

	uncheck = ['Cluster_id', 'i', 'Subcluster_id']# Набор имен, которые толчно должны быть в обучающих данных

	#Находим пересечения имен
	needed_mean = {}
	needed_names = ['id']

	for name in names_input:
		if name in names_ouput:
			needed_names.append(name)
			print(name)
			needed_mean[name] = mean[name]

	#Проверяем наличие общих координат
	if len(needed_names)==1:
		print('Нет общих координат') #Если нет общих координат, то прерываем программу
		exit()

	dfo = dfo[needed_names+uncheck] # Выделяем необходимые имена
	dfi = dfi[needed_names] # Выделяем необходимые имена
	dfi = norm(dfi, needed_mean) #Нормируем данные

	inputs = dfi.iloc[:].values

	if a is None:
		max_summ_edge, a = calculate_const(dfo[needed_names].iloc[:].values, 1)# \
		#сделать выбор для пользователя (считаем или сами задаем)

	# Вычисляем все F для всех клстеров и подкластеров
	for cluster in dfo['Cluster_id'].unique(): #проходим все кластеры
		time_df = dfo[lambda x: x['Cluster_id']==cluster] #Выделяем все строки относящиеся к этому кластеру
		for subcluster in time_df['Subcluster_id'].unique(): #проходим все подкластеры
			name = 'F_'+str(cluster) +'_' + str(subcluster) # Формируем имя столбца
			time_dfs = time_df[lambda x: x['Subcluster_id']==subcluster] #Выделяем все строки относящиеся к этому подкластеру
			F = []
			for line in inputs:
				F.append(get_F_example(time_dfs[needed_names].iloc[:].values, a, line)) #Вычисляем F для каждой строки данных для предсказания
			dfi[name] = F
	dfi.to_excel('Return.xlsx') #Записываем ответ


if __name__=="__main__":
	predict("1x18crst3D.csv", 'input.csv', {'X1':1, 'X2':1, 'X3':1})