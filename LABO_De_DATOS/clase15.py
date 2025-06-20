#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:05:03 2023

@author: mcerdeiro
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import duckdb as dd

#%%######################

        ####            Análisis exploratorio

#########################
#%%        


iris = load_iris(as_frame = True)

data = iris.frame
atributos = iris.data
Y = iris.target

iris.target_names
diccionario = dict(zip( [0,1,2], iris.target_names))
#%%
atri = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#%%
nbins = 35

f, s = plt.subplots(2,2)
plt.suptitle('Histogramas de los 4 atributos', size = 'large')


sns.histplot(data = data, x = 'sepal length (cm)', hue = 'target', bins = nbins, stat = 'probability', ax=s[0,0], palette = 'viridis')

sns.histplot(data = data, x = 'sepal width (cm)', hue = 'target', bins = nbins, stat = 'probability', ax=s[0,1], palette = 'viridis')

sns.histplot(data = data, x = 'petal length (cm)', hue = 'target', bins = nbins, stat = 'probability', ax=s[1,0], palette = 'viridis')

sns.histplot(data = data, x = 'petal width (cm)', hue = 'target', bins = nbins, stat = 'probability', ax=s[1,1], palette = 'viridis')


#%%######################

        ####            Métodos de umbral

#########################
#%%        

def clasificador_iris(fila):
    pet_l = fila['petal length (cm)']
    if pet_l < 2.5:
        clase = 0
    elif pet_l < 4.5:
        clase = 1
    else:
        clase = 2
    return clase
#%%
umbral_0 = 2.5
umbral_1 = 4.5

plt.figure()
plt.title('Histograma de petal length y los umbrales utilizados')
sns.histplot(data = data, x = 'petal length (cm)', hue = 'target', bins = nbins, stat = 'probability',  palette = 'viridis')
plt.axvline(x=umbral_0)
plt.axvline(x=umbral_1)
#%%
data_clasif = data.copy()
data_clasif['clase_asignada'] = atributos.apply(lambda row: clasificador_iris(row), axis=1)
        
#clasificador_iris(data['sepal length (cm)'], data['sepal width (cm)'], data['petal length (cm)'], data['petal width (cm)'])
#%%

clases = set(data['target'])

matriz_confusion = np.zeros((3,3))


for i in range(3):
  for j in range(3):
    filtro = (data_clasif['target']== i) & (data_clasif['clase_asignada'] == j)
    cuenta = len(data_clasif[filtro])
    matriz_confusion[i, j] = cuenta
  
matriz_confusion

#%%

exacti = sum(data_clasif['target']== data_clasif['clase_asignada'])

#%%
def exactitud(clasif):
    data_temp = data_clasif.copy()
    data_temp['clase_asignada'] = atributos.apply(lambda row: clasif(row), axis=1)
    suma = sum(data_temp['target']== data_temp['clase_asignada'])
    total = len(data_clasif)
    return suma/total

    

#%%
posibles_cortes = np.arange(start= 4, stop= 6, step=0.1)
exactitudes = []

for c in posibles_cortes:
    def clasificador_temp(fila):
        pet_l = fila['petal length (cm)']
        if pet_l < 2.5:
            clase = 0
        elif pet_l < c:
            clase = 1
        else:
            clase = 2
        return clase
    exact_c = exactitud(clasificador_temp)
    exactitudes.append(exact_c)

#%%
plt.plot(posibles_cortes, exactitudes)
plt.xlabel('posibles cortes')
plt.ylabel('exactitud')
plt.title('Exactitud en función del corte elegido')
#%%
max(exactitudes)
np.argmax(exactitudes)

exactitudes[8]
corte_selec = posibles_cortes[np.argmax(exactitudes)]

#%%######################

        ####            Árboles de decisión

#########################
#%%

X = atributos
Y = data['target']
#%%        
        
clf_info = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 4)
clf_info = clf_info.fit(X, Y)


plt.figure(figsize= [15,10])
tree.plot_tree(clf_info, feature_names = iris['feature_names'], class_names = iris['target_names'],filled = True, rounded = True, fontsize = 10)
#%%

datonuevo= pd.DataFrame([dict(zip(iris['feature_names'], [6.8,3,4.5, 2.15]))])
clf_info.predict(datonuevo)


#%%
# otra forma de ver el arbol
r = tree.export_text(clf_info, feature_names=iris['feature_names'])
print(r)
#%%

######################################################### EJERCICIOS
#%%
data=pd.read_csv('/users/Guille/Downloads/arboles.csv')


# Crear el histograma con separación por especie
plt.figure(figsize=(8,6))
sns.histplot(data=data, x="altura_tot", hue="nombre_com", palette="viridis", bins=20, edgecolor="black", alpha=0.6)

# Agregar títulos y etiquetas
plt.title("Distribución de Altura por Especie", fontsize=14)
plt.xlabel("Altura (m)", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.legend(title="Especie")

# Mostrar el gráfico



#INCLINACION
plt.figure(figsize=(8,6))
sns.histplot(data=data, x="inclinacio", hue="nombre_com", palette="viridis", bins=20, edgecolor="black", alpha=0.6)

# Agregar títulos y etiquetas
plt.title("Distribución de inclinacion por Especie", fontsize=14)
plt.xlabel("inclinacion (m)", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.legend(title="Especie")



#DIAMETRO
plt.figure(figsize=(8,6))
sns.histplot(data=data, x="diametro", hue="nombre_com", palette="viridis", bins=20, edgecolor="black", alpha=0.6)

# Agregar títulos y etiquetas
plt.title("Distribución de diametro por Especie", fontsize=14)
plt.xlabel("diametro (m)", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.legend(title="Especie")

# Mostrar el gráfico




### SCATTER PLOT


# HUE muestra como se clasifican y en este caso es por especie, palette es el color, S es el tamaño de los puntos 
plt.figure(figsize=(8,6))
sns.scatterplot(x=data["diametro"], y=data["altura_tot"], hue=data["nombre_com"], palette="viridis", s=15, edgecolor="black")

# Agregar etiquetas y título
plt.title("Altura vs Diámetro por Especie", fontsize=14)
plt.xlabel("Diámetro (cm)", fontsize=12)
plt.ylabel("Altura (m)", fontsize=12)
plt.legend(title="Especie")

# Mostrar el gráfico


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1) Carga de datos
# Ajustá la ruta y el método según tu archivo
df = pd.read_csv('arboles.csv')  
# df = pd.read_excel('arboles.xlsx', sheet_name='Hoja1')

# Características y etiqueta
X = df[['altura_tot', 'diametro', 'inclinacio']]
y = df['nombre_com']

# 2) Train/Test split
X_trai, X_tet, y_rain, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3) Bucle de entrenamiento y evaluación
resultados = []
for criterion in ['gini', 'entropy']:
    for max_depth in [2, 4, 6, 8, None]:
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=42
        )
        clf.fit(X_trai, y_rain)
        y_pred = clf.predict(X_tet)
        acc = accuracy_score(y_test, y_pred)
        resultados.append({
            'criterion': criterion,
            'max_depth': max_depth or 'None',
            'accuracy': acc
        })

# 4) Mostrar resultados ordenados
res_df = pd.DataFrame(resultados)
print(res_df.sort_values(['criterion','max_depth']))



titanic=pd.read_csv('/users/Guille/Downloads/titanic_training.csv')
x1=titanic[['Pclass','Sex','Age']]
y1=titanic['Survived']
x1['Sex'] = x1['Sex'].map({'male': 0, 'female': 1})

x1['Age'].fillna(x1['Age'].median(), inplace=True)



X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2)
arbol=DecisionTreeClassifier(criterion='entropy'   ,max_depth=6)
arbol.fit(X_train,y_train)
Y_pred=arbol.predict(X_test)
a=metrics.accuracy_score(y_test,Y_pred)
print(metrics.confusion_matrix(y_test, Y_pred))
print(a)

def matriz_confusion_binaria(y_test, y_pred):
    tp = fp = tn = fn = 0
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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, Y_pred)
prec = precision_score(y_test, Y_pred)
rec = recall_score(y_test, Y_pred)
f1 = f1_score(y_test, Y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")



import pandas as pd

df_sucursales = pd.DataFrame({
    "N_sucursal": [123, 52, 107, 141],
    "Barrio": ["Palermo", "Villa Crespo", "Belgrano", "Palermo"],
    "Cant_Cajeros": [2, 4, 3, 3],
    "Cant_Cajas": [3, 3, 2, 4]
})

df_zonas = pd.DataFrame({
    "Barrio": ["Belgrano", "Caballito", "Flores", "Palermo", "Palermo"],
    "Zona": ["Norte", "Sur", "Sur", "Norte", "Sur"],
    "Ciudad": ["CABA"] * 5,
    "Codigo_Postal": [1411, 1406, 1321, 1418, 1425]
})


c="""
 SELECT s.Barrio,COUNT(*) as TotalSUCU,SUM(s.Cant_Cajeros) as TotalCAJEROS
 FROM df_sucursales as s 
 INNER JOIN df_zonas as z ON s.Barrio=z.Barrio
 GROUP BY s.Barrio
 HAVING TotalCAJEROS > 3
 ORDER BY s.Barrio ASC
"""
res=dd.sql(c).df()
print(res)






















