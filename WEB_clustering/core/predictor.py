import numpy as np
import pandas as pd
import pickle
from core.utils import get_F, get_F_example

from sklearn.metrics import pairwise_distances

class Classsifier():
    def __init__(self, config):
        self.config = config
        if 'ignore_coord' in self.config:
            self.nameignore = ['F', 'cluster_id', 'subcluster_id'] + self.config['ignore_coord']
        else:
            self.nameignore = ['F', 'cluster_id', 'subcluster_id']
            
    def __norm(self, df):
        for name in self.config['norms']:
            if name in df.columns:
                if name+'_original' in self.config['norms']:
                    df[name] /= self.config['norms'][name+'_original']
                else:
                    df[name] /= self.config['norms'][name]
        return df
    
    def __inverse_transform(self, df_train, pca_path):
        pca_coord = []
        pca_norm = []
        original_cood = []
        original_norm = []
        for name in self.config['norms']:
            if 'original' not in name and name+'_original' in self.config['norms']:
                pca_coord.append(name)
                pca_norm.append(self.config['norms'][name])
                original_cood.append(name+'_original')
                original_norm.append(self.config['norms'][name+'_original'])

        #Переводим PCA координаты в обычные
        X = df_train[pca_coord].values
        for i in range(len(pca_norm)):
            X[:,i] *= pca_norm[i]

        #Загружаем PCA
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

        #РеPCA
        X = pca.inverse_transform(X)

        #нормируем на оригиналы
        for i in range(len(original_norm)):
            X[:,i] /= original_norm[i]

        df_train[pca_coord] = X
        return df_train

    def predict(self, df_train, df_predict, pca_path, check_self=False):
        text = ''
        names_ouput = list(df_train)[1:]#Выделяем все имена из обучающих данных  
        names_input = list(df_predict)[1:]#Выделяем все имена из данных для предсказания
        needed_names = ['id']
        needed_pca_names = []
        for name in names_input:
            if name in names_ouput and name not in self.nameignore:
                needed_names.append(name)
                if name+'_original' in self.config['norms']:
                    needed_pca_names.append(name)

        if len(needed_names)==1:
            text += 'Нет общих координат\n'#Если нет общих координат, то прерываем программу
            return df_predict
        
        if len(needed_pca_names)>0:
            df_train = self.__inverse_transform(df_train, pca_path)

        nameignore = [name for name in self.nameignore if name in df_train.columns]

        df_train = df_train[needed_names+nameignore] # Выделяем необходимые имена
        df_predict = df_predict[needed_names] # Выделяем необходимые имена
        if not check_self:
            df_predict = self.__norm(df_predict) #Нормируем данные

        inputs = df_predict.iloc[:].values
        #Работа с ребрами
        if 'cluster_id' in df_train.columns and 'subcluster_id' in df_train.columns:
            for cluster in sorted(df_train['cluster_id'].unique()): #проходим все кластеры
                time_df = df_train[lambda x: x['cluster_id']==cluster] #Выделяем все строки относящиеся к этому кластеру
                for subcluster in sorted(time_df['subcluster_id'].unique()): #проходим все подкластеры
                    name = 'F_'+str(cluster) +'_' + str(subcluster) # Формируем имя столбца
                    time_dfs = time_df[lambda x: x['subcluster_id']==subcluster] #Выделяем все строки относящиеся к этому подкластеру
                    F = []
                    for line in inputs:
                        F.append(get_F_example(time_dfs[needed_names].iloc[:].values, self.config['consts']['a'], line, self.config['consts']['cluster_importancy'])) #Вычисляем F для каждой строки данных для предсказания
                    df_predict[name] = F
                    # df_predict.loc[:,name]= F
        elif 'cluster_id' in df_train.columns:
            for cluster in sorted(df_train['cluster_id'].unique()): #проходим все кластеры
                time_df = df_train[lambda x: x['cluster_id']==cluster] #Выделяем все строки относящиеся к этому кластеру
                name = 'F_'+str(cluster) +'_NuN' # Формируем имя столбца
                F = []
                for line in inputs:
                    F.append(get_F_example(time_df[needed_names].values, self.config['consts']['a'], line, self.config['consts']['cluster_importancy'])) #Вычисляем F для каждой строки данных для предсказания
                df_predict[name] = F
                # df_predict.loc[:,name]= F
        elif 'subcluster_id' in df_train.columns:
            for subcluster in sorted(df_train['subcluster_id'].unique()): #проходим все сабкластеры
                time_df = df_train[lambda x: x['subcluster_id']==subcluster] #Выделяем все строки относящиеся к этому сабкластеру
                name = 'F_NuN_'+str(subcluster) # Формируем имя столбца
                F = []
                for line in inputs:
                    F.append(get_F_example(time_df[needed_names].values, self.config['consts']['a'], line, self.config['consts']['cluster_importancy'])) #Вычисляем F для каждой строки данных для предсказания
                df_predict[name] = F
                # df_predict.loc[:,name]= F
        else:
            text += 'Неверный формат данных\n'
        return df_predict, text