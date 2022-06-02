# manipulacion de los datos
import numpy as np
import pandas as pd 
# gestion de archivos
import os
# omitir warnings 
import warnings
warnings.filterwarnings('ignore')
# matplotlib y seaborn para realizar gráficos
import matplotlib.pyplot as plt
import seaborn as sns
# sklearn para el modelo
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# para generar el archivo pkl
import joblib

path ='LiverPatient.csv'
dataset =pd.read_csv(path, header =0)
# nombres de columnas 
dataset.columns= ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'AG', 'Status']
# transformar género a categórica 
dataset['Gender'].replace('Female',0 ,inplace=True)
dataset['Gender'].replace('Male',1,inplace=True)
dataset.head()
# quitar valores faltantes 
dataset = dataset.dropna()
data = np.array(dataset['Status']).copy()
data[data == 1] = 0
data[data == 2] = 1
dataset.drop('Status', inplace=True, axis=1)
dataset['Status'] = data.tolist()
dataset.head()
# reducir multicolinealidad 
new_data = dataset.drop('TP', axis=1)
new_data = new_data.drop('Sgot', axis=1)
# generar dataframe etiquetas
labels = new_data['Status']
labels = labels.astype('int64')
mydata_train, mydata_test, labels_train, labels_test = train_test_split(new_data.loc[:,'Age':'AG'], labels, test_size=0.33,random_state=2)
#modelo
gnb = GaussianNB()
#prediccion
y_pred = gnb.fit(mydata_train, labels_train).predict(mydata_test)
import joblib

joblib.dump(gnb, "gnb.pkl")
