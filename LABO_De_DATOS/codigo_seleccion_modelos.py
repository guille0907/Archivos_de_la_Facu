#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:23:04 2024

@author: mcerdeiro
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree

#%% funciones para medir performance
def matriz_confusion_binaria(y_test, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_test)):
        if y_test[i]:
            if y_pred[i]:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i]:
                fp += 1
            else:
                tn += 1
    
    return tp, tn, fp, fn

def accuracy_score(tp, tn, fp, fn):
    acc = (tp+tn)/(tp+tn+fp+fn)
    return acc

def precision_score(tp, tn, fp, fn):
    prec = tp/(tp+fp)
    return prec

def recall_score(tp, tn, fp, fn):
    rec = tp/(tp+fn)
    return rec

def f1_score(tp, tn, fp, fn):
    prec = precision_score(tp, tn, fp, fn)
    rec = recall_score(tp, tn, fp, fn)
    f1 = 2*prec*rec/(prec+rec)
    return f1

#%% cargamos los datos
df = pd.read_csv('seleccion_modelos.csv')

X = df.drop("Y", axis=1)
y = df.Y
#%% separamos entre dev y eval
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,random_state=1,test_size=0.1)

#%% experimento

alturas = [1,2,3,5,10]
nsplits = 5
kf = KFold(n_splits=nsplits)

resultados = np.zeros((nsplits, len(alturas)))
# una fila por cada fold, una columna por cada modelo

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas):
        
        arbol = tree.DecisionTreeClassifier(max_depth = hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        tp, tn, fp, fn = matriz_confusion_binaria(kf_y_test.values, pred)
        score = accuracy_score(tp, tn, fp, fn)
        
        resultados[i, j] = score
#%% promedio scores sobre los folds
scores_promedio = resultados.mean(axis = 0)


#%% 
for i,e in enumerate(alturas):
    print(f'Score promedio del modelo con hmax = {e}: {scores_promedio[i]:.4f}')

#%% entreno el modelo elegido en el conjunto dev entero
arbol_elegido = tree.DecisionTreeClassifier(max_depth = 1)
arbol_elegido.fit(X_dev, y_dev)
y_pred = arbol_elegido.predict(X_dev)

tp, tn, fp, fn = matriz_confusion_binaria(y_dev.values, y_pred)
score_arbol_elegido_dev = accuracy_score(tp, tn, fp, fn)
print(score_arbol_elegido_dev)

#%% pruebo el modelo elegid y entrenado en el conjunto eval
y_pred_eval = arbol_elegido.predict(X_eval)
tp, tn, fp, fn = matriz_confusion_binaria(y_eval.values, y_pred_eval)
score_arbol_elegido_eval = accuracy_score(tp, tn, fp, fn)
print(score_arbol_elegido_eval)















