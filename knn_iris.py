#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 22:04:07 2023

@author: mcerdeiro
"""
#%% modulos

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#%%######################

        ####            Cargo dataset

#########################
#%%        

arbol=pd.read_csv('/home/Estudiante/Descargas/arboles.csv')
X1=arbol['altura_tot']
Y1=arbol['diametro']

modelo=KNeighborsClassifier(n_neighbors=3)
modelo.fit(X1,Y1)
Y_pre=modelo.predict(X1)
a=metrics.accuracy_score(Y1,Y_pre)
print(metrics.confusion_matrix(Y1, Y_pre))
print(a)






iris = load_iris(as_frame = True)

data = iris.frame
X = iris.data
Y = iris.target

iris.target_names
diccionario = dict(zip( [0,1,2], iris.target_names))
#%%######################

        ####            Un primer modelo

#########################
#%%  #Defino el clasificador 
model = KNeighborsClassifier(n_neighbors = 5) # modelo en abstracto
model.fit(X, Y) # entreno el modelo con los datos X e Y, trata de aprender con datos y variables que saco del dataset
Y_pred = model.predict(X) # me fijo qué clases les asigna el modelo a mis datos, chequea como predice para cada punto sin etiquetas , osea que clase le da
metrics.accuracy_score(Y, Y_pred) # y esto chequea como clasifico, los originales con los que predijo
metrics.confusion_matrix(Y, Y_pred) # me arma la matriz de confusion

#%%  #Para testearlo, datos que el modelo no vio pero vos tenes etiquetados       
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) # 70% para train y 30% para test
#le das tus datos y etiquetas y el test_size te los guarda en este caso el 30%
#vuelvo a definir el modelo usando solo el 70% es lo mismo que lo anterior pero ahora usa menos datos entrena con 70%
#veo que me dice con el 30% que no vio
model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X_train, Y_train) # entreno el modelo con los datos X_train e Y_train, entreno con el 70%
Y_pred = model.predict(X_test) # me fijo qué clases les asigna el modelo a mis datos X_test , chequeo como predice el 30% que no vio
print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
metrics.confusion_matrix(Y_test, Y_pred)
#%%

# ¿QUÉ PASA SI REPETIMOS CON OTRO SPLIT?

#%%

Nrep = 5
valores_n = range(1, 20)

resultados_test = np.zeros((Nrep, len(valores_n)))
resultados_train = np.zeros((Nrep, len(valores_n)))


for i in range(Nrep):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    for k in valores_n:
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test[i, k-1] = acc_test
        resultados_train[i, k-1] = acc_train

#%%

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test = np.mean(resultados_test, axis = 0) 
#%%

plt.plot(valores_n, promedios_train, label = 'Train')
plt.plot(valores_n, promedios_test, label = 'Test')
plt.legend()
plt.title('Exactitud del modelo de knn')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')
#%%

#%%

# ¿QUÉ PASA SI REPETIMOS MÁS VECES?

#%%









        
        
        
        
        
        
        
        
        
        
        
        
        
        