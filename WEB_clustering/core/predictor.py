import numpy as np
import pandas as pd
from core.utils import get_F, get_F_example

from sklearn.metrics import pairwise_distances

class Classsifier():
    def __init__(self, config):
        self.config = config
        self.mean = self.config['norms']
        if 'ignore_coord' in self.config:
            self.nameignore = ['F', 'cluster_id', 'subcluster_id'] + self.config['ignore_coord']
        else:
            self.nameignore = ['F', 'cluster_id', 'subcluster_id']
            
    def __norm(self, df):
        for name in self.mean:
            if name in df.columns:
                df[name] /= self.mean[name]
        return df
    
    def predict(self, df_train, df_predict, a=None):
        names_ouput = list(df_train)[1:]#Выделяем все имена из обучающих данных  
        names_input = list(df_predict)[1:]#Выделяем все имена из данных для предсказания
        needed_names = ['id']
        for name in names_input:
            if name in names_ouput and name not in self.nameignore:
                needed_names.append(name)
        if len(needed_names)==1:
            print('Нет общих координат') #Если нет общих координат, то прерываем программу
            return df_predict
        
        df_train = df_train[needed_names+self.nameignore] # Выделяем необходимые имена
        df_predict = df_predict[needed_names] # Выделяем необходимые имена
        df_predict = self.__norm(df_predict) #Нормируем данные
        
        inputs = df_predict.iloc[:].values
        #Работа с ребрами
        for cluster in sorted(df_train['cluster_id'].unique()): #проходим все кластеры
            time_df = df_train[lambda x: x['cluster_id']==cluster] #Выделяем все строки относящиеся к этому кластеру
            for subcluster in sorted(time_df['subcluster_id'].unique()): #проходим все подкластеры
                name = 'F_'+str(cluster) +'_' + str(subcluster) # Формируем имя столбца
                time_dfs = time_df[lambda x: x['subcluster_id']==subcluster] #Выделяем все строки относящиеся к этому подкластеру
                F = []
                for line in inputs:
                    F.append(get_F_example(time_dfs[needed_names].iloc[:].values, self.config['consts']['a'], line)) #Вычисляем F для каждой строки данных для предсказания
                df_predict[name] = F
        return df_predict