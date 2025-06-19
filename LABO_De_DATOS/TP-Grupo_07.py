#!/usr/bin/env python
# coding: utf-8

#INTEGRANTES: Guillermo de la Vega, Matias Naddeo y Francisco Anllo
#En este archivo se encuentran las funciones utilizadas para el analisis exploratorio, los graficos necesarios para estudiar los datos y 
# los modelos de clasificacion tanto el de KNN como el de Arboles de Decision.


#%% Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb as dd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn import metrics
import random
from sklearn.tree import DecisionTreeClassifier
#%% Load dataset 

data_df = pd.read_csv("Fashion-MNIST.csv", index_col=0)
img_nbr = 13727

# keep label out
img = np.array(data_df.iloc[img_nbr,:-1]).reshape(28,28)

# Plot image
plt.imshow(img, cmap = "gray")
### FUNCIONES PARA PRECISION;ACCURACY,RECALL,F1 SCORE
def matriz_confusion_binaria (y_test, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_test)):
        real = y_test[i]   # <-- aquí
        pred = y_pred[i]
        if real == 0:
            if real == pred:
                TN +=1
            else:
                FP += 1
        else:
            if real == pred:
                TP +=1
            else:
                FN += 1
    return TP, TN, FP, FN


def precision_score(tp, tn, fp, fn):
    if tp != 0 or fp != 0:
        return tp/(tp+fp)
    else:
        return 0


def recall_score(tp, tn, fp, fn):
    if tp != 0 or fn!= 0:
        return tp/(tp+fn)
    else:
        return 0


def accuracy_score(tp, tn, fp, fn):
    return (tp+tn)/(tp+fn+fp+tn)


def f1_score(tp, tn, fp, fn):
    precision = precision_score(tp,tn,fp,fn)
    recall = recall_score(tp,tn,fp,fn)
    return(2*precision*recall)/(precision+recall)


#%% Punto 1
#region        Achicar dataset, crear Fashion-MNIST_mini
#ESTE CODIGO REDUCE EL Fashion-MNIST.csv (70k elementos) a Fashion-MNIST_mini.csv (10k elementos)

def crear_mini_copia(semilla):
    df = pd.read_csv("Fashion-MNIST.csv")
    df = df.drop(columns=['Unnamed: 0'], errors="ignore") # Borra la columna "Unnamed: 0" (si existe)
    clases=[]
    for i in range(10):# Selecciona 1000 elementos de cada clase
        clases.append(df[df['label'] == i].sample(n=1000, random_state=semilla))

    df_mini = pd.concat(clases).sample(frac=1, random_state=semilla)
    df_mini.to_csv("Fashion-MNIST_mini.csv", index=False) # Guarda el dataframe en Fashion-MNIST_mini.csv
    print("Fashion-MNIST_mini.csv creado correctamente!")

crear_mini_copia(1)
#endregion 

# region       Imagen promedio global

#ESTE CODIGO DADAS CIERTAS CLASES, CALCULA COMO SE VERÁ LA IMÁGEN PROMEDIO.
def graficar_mean_global():
    df = pd.read_csv('Fashion-MNIST_mini.csv') # Usa una muestra reducida de Fashion-MNIST
        
    clases="0,1,2,3,4,5,6,7,8,9" #Solo estas clases serán utilizadas para calcular la imagen promedio. En este caso, todas.
    filtroclases=f'''SELECT * FROM df WHERE label IN ({clases})'''
    df=dd.sql(filtroclases).df()

    columnas=[]
    for i in range(784):
        columnas.append("pixel"+str(i))
    valores_promedio = df[columnas].mean().values

    imagen_promedio = valores_promedio.reshape(28, 28)
    plt.imshow(imagen_promedio, cmap="gray")
    if clases == "0,1,2,3,4,5,6,7,8,9": plt.title(f"Promedio global", fontsize=20)    
    else: plt.title(f"Imagen promedio de las clases {clases} juntas", fontsize=20)
    plt.xticks([0, 7, 14, 21, 27])
    plt.yticks([0, 7, 14, 21, 27])
    plt.show()

graficar_mean_global()

#endregion

# region       Grafica las 10 imágenes promedio

# ESTE CODIGO GRAFICA LA IMAGEN PROMEDIO DE CADA CLASE
def graficar_mean_xclase():
    df = pd.read_csv('Fashion-MNIST_mini.csv')    # Usa una muestra reducida de Fashion-MNIST    
    columnas = []
    for i in range(784):
        columnas.append("pixel" + str(i))

    _, axs = plt.subplots(2, 5, figsize=(12, 6)) # Creamos el grafico para que entren las 10 clases

    for clase in range(10):
        clases = str(clase)
        filtroclases = f'''SELECT * FROM df WHERE label IN ({clases})'''
        df_clase = dd.sql(filtroclases).df()

        valores_promedio = df_clase[columnas].mean().values
        imagen_promedio = valores_promedio.reshape(28, 28)
        if clase<5: fila=0 # Le damos la fila y columna adecuada a cada clase 
        else: fila=1
        col = clase-fila*5
        
        axs[fila, col].axis("off")
        axs[fila, col].imshow(imagen_promedio, cmap="gray")
        axs[fila, col].set_title(f"Clase {clase}", fontsize=18)

    plt.tight_layout()
    plt.show()

graficar_mean_xclase()

#endregion 

# region       Imagenes de las clases promedio con puntos rojos
#GRAFICA EL PROMEDIO DE LAS CLASES QUE QUIERAS, Y ADEMÁS MARCA CON UN PUNTO ROJO EL PÍXEL QUE QUIERAS. SIRVE PARA COMPARAR.
def gaficar_clases_pts_rojos(clases, puntos):

    df = pd.read_csv('Fashion-MNIST_mini.csv') # Usa una muestra reducida de Fashion-MNIST

    columnas = [f"pixel{i}" for i in range(784)]
    x,y=[],[]

    for i in puntos: x.append(i%28), y.append(i//28) # Consigue las coordenadas de cada uno de los puntos.

    _, axs = plt.subplots(1, len(clases), figsize=(3 * len(clases), 3))
    for idx, clase in enumerate(clases): #Hace un gráfico por clase.
        df_clase = dd.sql(f'''SELECT * FROM df WHERE label = {clase}''').df()
        valores_promedio = df_clase[columnas].mean().values
        imagen_promedio = valores_promedio.reshape(28, 28)
        axs[idx].set_xticks([0, 7, 14, 21, 27])
        axs[idx].set_yticks([0, 7, 14, 21, 27])
        axs[idx].imshow(imagen_promedio, cmap="gray") # Grafica la imagen promedio
        axs[idx].scatter(x, y, color="red", s=50) # Marca los puntos
        axs[idx].set_title(f"Clase {clase}", fontsize=20)

    plt.tight_layout()
    plt.show()

gaficar_clases_pts_rojos([0,1], [202,218,574, 518, 630]) # Elegimos que clases y que puntos seran graficados 
#endregion

#region        Método 2

# ESTE SCRIPT ES NUESTRO "MÉTODO 2", SIRVE PARA ENCONTRAR LOS PIXELES QUE MAS DIFERENCIA PROMEDIO TIENEN ENTRE DOS CLASES.
#
# DADAS DOS CLASES  (N Y M), GENERA UN NUEVO ARCHIVO LLAMADO N_M_sum.csv, ESTE ARCHIVO TIENE LAS SIGUIENTES COLUMNAS: 
# pixel,promedioN,promedioM,diferencia
# LA PRIMERA EL NOMBRE DEL PIXEL, POR EJEMPLO "pixel273", LAS DE PROMEDIO INDICAN EL VALOR PROMEDIO QUE TOMA ESE PIXEL PARA CADA UNA
# DE LAS CLASES. LA COLUMNA DIFERENCIA SIMPLEMENTE CONTIENE LA DIFERENCIA ENTRE PROMEDIOS, ESTO NOS FACILITA SU USO. 
# ESTA ORDENADA JUSTAMENTE POR DIFERENCIA, POR LO QUE CON SOLO MIRAR EL HEAD PODEMOS OBTENER PIXELES UTILES.
def diferencia_promedio(clase1, clase2):
    full_data=pd.read_csv("Fashion-MNIST_mini.csv") # Usa una muestra reducida de Fashion-MNIST    
    filtro_x_clase=f'''SELECT * FROM full_data WHERE label IN ({clase1}, {clase2})''' # Filtramos únicamente las clases de interés
    data = dd.sql(filtro_x_clase).df()
    
    mean_value = diferencias_sum(clase1, clase2, data) # Llamamos a la función para conseguir los valores de diferencia promedio por pixel

    mean_value.sort_values(by="diferencia", ascending=False).to_csv(f"{clase1}_{clase2}_sum.csv", index=False) # Guardamos ordenados en csv
    print(f"Resultados guardados en {clase1}_{clase2}_sum.csv")

    print(mean_value.sort_values(by="diferencia", ascending=False).head(5)) # Mostramos los pixeles con mayor diferencia promedio.

    ver_diferencia(clase1, clase2, mean_value) # Esta funcion muestra la imagen, que es el resultado de restar ambas imagenes promedio.

def diferencias_sum(clase1, clase2, data): # Dadas dos clases, y una variable data, con UNICAMENTE registros de las clases de interes
    # calcula el promedio por clase de cada uno de los pixeles, y la diferencia. devuelve el dataframe 
    print("diferencias_sum()...")
    df_value_dif = []
    columnas = columnas = ["pixel", f"promedio{clase1}", f"promedio{clase2}","diferencia"]
    for i in range(784):
        if i==392:print(f"Procesando...")
        pixel = f"pixel{i}"
        promedio1 = data[data["label"] == clase1][pixel].mean()
        promedio2 = data[data["label"] == clase2][pixel].mean()
        df_value_dif.append([pixel,promedio1, promedio2, abs(promedio1 - promedio2)])
    df_value_dif = pd.DataFrame(df_value_dif, columns=columnas)
    return df_value_dif

def ver_diferencia(clase1, clase2, data): # Esta funcion esencialmente, grafica la tabla generada por data, donde el valor del pixel es el 
    # valor de la columna "diferencia", Valores más claros indican una mayor distancia entre los promedios de ambas clases. 
    
    img = np.array(data["diferencia"]).reshape(28,28) # Usamos "diferencia" para asignar el valor a cada pixel
    plt.imshow(img, cmap = "gray",vmin=0, vmax=255) # Usamos vmin=0, vmax=255 para que se vea la diferencia real
    plt.title(f"Diferencia promedio entre las clases {clase1} y {clase2}",  fontsize=18)
    plt.axis("off")
    plt.show()

# Acá elegimos con que clases vamos a querer trabajar.
diferencia_promedio(1, 7)
diferencia_promedio(2,6)

#endregion

#region        Scatter-Plot

# GRAFICA UN SCATTER PLOT DONDE FIGURAN TODOS LOS PIXELES. LA IDEA ES MOSTRAR LA RELACION ENTRE DIFERENCIA PROMEDIO Y EXACTITUD.
# ENTONCES, DADAS DOS CLASES (QUE CLASES PARTICULARMENTE ELEGIMOS ES POCO RELEVANTE, EL GRAFICO SE VA A VER SIMILAR), BASADO EN LA  
# DIFERENCIA PROMEDIO DEL VALOR DE CADA PIXEL POR CLASE (EJE X), CALCULA LA EXACTITUD DE UN MODELO DE CLASIFICACION BINARIA MUY SIMPLE
# BASADO EN UN UMBRAL. DADO UN PIXEL, UBICA EL UMBRAL (QUE DEFINE SI UN ELEMENTO DESCONOCIDO ES DE UNA CLASE U OTRA) A MITAD DE CAMINO
# ENTRE UN PROMEDIO Y EL OTRO. (menor_promedio +umbral/2). DADO ESTE ALGORITMO BASICO, CALCULA LA EXACTITUD. ESTO PARA CADA PIXEL.

def probar_umbral(pixel, umbral, clase1, clase2, data):# Función que devuelve la exactitud de un píxel para un umbral dado. 
    # (Aclaración: la clase 1 DEBE SER la que tiene el promedio menor)
    # data debe contener únicamente elementos de las clases pasadas por parámetro
    res = np.where(data[f"pixel{pixel}"] <= umbral, clase1, clase2) # Si el valor es menor al umbral le asigna clase1, sino clase2 
    exactitud = (res==data["label"].values).mean() # calcula la exactitud, osea que tan preciso fue.
    return exactitud

def test_all_pixels(clase1, clase2):# Esta funcion simplemente llama a probar_umbral() para cada pixel y grafica los resultados.
    # En cada pixel usa como umbral el punto medio entre los promedios de las clases. 
    res=[]
    data = pd.read_csv('Fashion-MNIST_mini.csv') # Usa una muestra reducida de Fashion-MNIST
    data = dd.sql(f'''SELECT * FROM data WHERE label IN ({clase1}, {clase2})''').df() # Filtramos únicamente las clases de interés
    try:

        dif_data = pd.read_csv(f'{clase1}_{clase2}_sum.csv') #Trata de leer {clase1}_{clase2}_sum.csv asi es mas rapido.
    except FileNotFoundError:
        dif_data = diferencias_sum(clase1, clase2, data) # Sino lo calcula. Usando la funcion de diferencias_sum
        
    for i in range(784): # Para cada uno de los 784 pixeles, busca la exactitud del sencillo algoritmo de clasificacion.
        pixel_i = dif_data.loc[dif_data["pixel"] == f"pixel{i}"].iloc[0]
        prom1 = pixel_i[f"promedio{clase1}"]
        prom2 = pixel_i[f"promedio{clase2}"]
        diferencia = pixel_i["diferencia"]
        if prom1 < prom2: # Asigna el umbral segun cual promedio es mayor. #Además llama a probar_umbral con de menor promedio primero.
            umbral = prom1 + diferencia / 2
            exactitud=probar_umbral(i,umbral, clase1, clase2, data) 
        else:
            umbral = prom2 + diferencia / 2
            exactitud=probar_umbral(i,umbral, clase2, clase1, data)

        res.append({"pixel": i, "diferencia": diferencia, "exactitud": exactitud})
    res = pd.DataFrame(res)
    # Graficamos el scatter plot, eje X diferencia promedio, eje Y exactitud del modelo
    plt.figure(figsize=(12, 6))
    plt.scatter(res["diferencia"], res["exactitud"], s=7, c="green")
    plt.ylabel("Exactitud", fontsize=16)
    plt.grid(True, alpha=0.3) # Grilla para que no se vea tan vacío
    plt.xlabel(f"Diferencia de promedios (clases {clase1} y {clase2})", fontsize=16)
    plt.title("Exactitud vs. diferencia de promedios (Por píxel)", fontsize=20)
    plt.show()

test_all_pixels(2,6)

#endregion


#region        Heatmap

def graficar_heatmap(clases):
    data = pd.read_csv("Fashion-MNIST_mini.csv") # Usa una muestra reducida de Fashion-MNIST
    plt.figure(figsize=(15, 5))

    for i in range(len(clases)):
        res = data[data['label'] == clases[i]].drop('label', axis=1).values # conseguimos la varianza
        res = np.sqrt(np.var(res, axis=0)) # Conseguimos la desviación estándar
        
        heatmap = res.reshape(28, 28) # graficamos
        plt.subplot(1, 3, i+1)
        plt.imshow(heatmap, cmap='hot', vmin=0, vmax=110) # ponemos esto para que todos se comparen parejamente
        plt.title(f"Clase {clases[i]}", fontsize=20)
        plt.axis('off')
        plt.colorbar(fraction=0.05, pad=0.05) # Ajustamos el tamaño de laa barra

    plt.tight_layout()
    plt.show()

graficar_heatmap([5,1,8])




#endregion







#%% Punto 2

#################### EJERCICIO 2 KNN

################################################################################################################
#consulta sql que nos permite reducir el dataframe para solo las dos clases que necesitamos
c="""
 SELECT *
 FROM data_df as d
 WHERE d.label = 0 OR d.label = 8
 ORDER BY d.label
"""
res=dd.sql(c).df()

#Modelo con todos los pixeles encontrados en el analisis exploratorio

pixeles=['pixel70','pixel554','pixel526','pixel126','pixel40','pixel68','pixel96'] 

resultados_completo=[]

X1=res[pixeles]
Y1=res['label']
#separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=42)
#armamos el modelo y vemos que tal predice 
k_candidatos=[1,3,5,7,12,17]
for k in k_candidatos:
    modelo=KNeighborsClassifier(n_neighbors=k)
    modelo.fit(x1_train,y1_train)
    Y_pre=modelo.predict(x1_test)
    a=metrics.accuracy_score(y1_test,Y_pre)
    
    y_test_bin = [1 if y == 0 else 0 for y in y1_test] #esto y la linea de abajo hacen que la comparacion para poder sacar las medidas sea binaria ya que sino seria multiclase y las medidas no pueden
    y_pred_bin = [1 if y == 0 else 0 for y in Y_pre] #  lo mismo que la de arriba hace que la comparacion sea binaria para calcular la perfomance

    TP, TN, FP, FN = matriz_confusion_binaria(y_test_bin, y_pred_bin)

    precision= precision_score(TP, TN, FP, FN)
    recall=recall_score(TP, TN, FP, FN)
    f1=f1_score(TP, TN, FP, FN)
    accuracy=accuracy_score(TP, TN, FP, FN)
    resultados_completo.append((k,accuracy,precision,recall,f1))

print("Pixeles usados:", pixeles)

for r in resultados_completo:
    k, acc, prec, rec, f1 = r
    print(f"(K={k}, Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1 Score={f1:.2f})")




#Modelo con los 3 pixeles del que separan los grupos grandes
   
pixeles=['pixel40','pixel68','pixel96'] 

resultados_corte_general=[]

X1=res[pixeles]
Y1=res['label']
#separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=42)
#armamos el modelo y vemos que tal predice 
k_candidatos=[1,3,5,7,12,17]
for k in k_candidatos:
    modelo=KNeighborsClassifier(n_neighbors=k)
    modelo.fit(x1_train,y1_train)
    Y_pre=modelo.predict(x1_test)
    a=metrics.accuracy_score(y1_test,Y_pre)
    y_test_bin = [1 if y == 0 else 0 for y in y1_test] #esto y la linea de abajo hacen que la comparacion para poder sacar las medidas sea binaria ya que sino seria multiclase y las medidas no pueden
    y_pred_bin = [1 if y == 0 else 0 for y in Y_pre] #  lo mismo que la de arriba hace que la comparacion sea binaria para calcular la perfomance

    TP, TN, FP, FN = matriz_confusion_binaria(y_test_bin, y_pred_bin)

    precision= precision_score(TP, TN, FP, FN)
    recall=recall_score(TP, TN, FP, FN)
    f1=f1_score(TP, TN, FP, FN)
    accuracy=accuracy_score(TP, TN, FP, FN)
    resultados_corte_general.append((k,accuracy,precision,recall,f1))

print("Pixeles usados:", pixeles)

for r in resultados_corte_general:
    k, acc, prec, rec, f1 = r
    print(f"(K={k}, Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1 Score={f1:.2f})")





###########################################

#Modelo con los pixeles encontrados a traves del metodo 2

pixeles=['pixel70','pixel554','pixel526','pixel126'] 

resultados_best=[]

X1=res[pixeles]
Y1=res['label']
#separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=42)
#armamos el modelo y vemos que tal predice 
k_candidatos=[1,3,5,7,12,17]
for k in k_candidatos:
    modelo=KNeighborsClassifier(n_neighbors=k)
    modelo.fit(x1_train,y1_train)
    Y_pre=modelo.predict(x1_test)
    a=metrics.accuracy_score(y1_test,Y_pre)
    y_test_bin = [1 if y == 0 else 0 for y in y1_test] #esto y la linea de abajo hacen que la comparacion para poder sacar las medidas sea binaria ya que sino seria multiclase y las medidas no pueden
    y_pred_bin = [1 if y == 0 else 0 for y in Y_pre] #  lo mismo que la de arriba hace que la comparacion sea binaria para calcular la perfomance

    TP, TN, FP, FN = matriz_confusion_binaria(y_test_bin, y_pred_bin)

    precision= precision_score(TP, TN, FP, FN)
    recall=recall_score(TP, TN, FP, FN)
    f1=f1_score(TP, TN, FP, FN)
    accuracy=accuracy_score(TP, TN, FP, FN)
    resultados_best.append((k,accuracy,precision,recall,f1))

print("Pixeles usados:", pixeles)

for r in resultados_best:
    k, acc, prec, rec, f1 = r
    print(f"(K={k}, Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1 Score={f1:.2f})")


#%% Punto 3


###############################################################################################################
#Arboles

#pixeles conseguidos a traves del analisis explotario que consideramos que son adecuados para hacer una buena clasificacion
pixeles_arbol=['pixel40','pixel68','pixel96','pixel454','pixel173','pixel145','pixel201','pixel63','pixel733','pixel38',
               'pixel427','pixel399','pixel340','pixel312','pixel380','pixel381','pixel574','pixel70','pixel554','pixel526',
               'pixel126','pixel579','pixel607']


x1=data_df[pixeles_arbol]
y1=data_df['label']

#aca separamos el conjunto en desarrollo y held-out, los 3 primeros modelos de arbol solo utilizan los datos de desarrollo solo el ultimo usa el held-out.

X_dev,X_heldout,y_dev,y_heldout=train_test_split(x1,y1,test_size=0.2,random_state=42)

#profundidades que vamos a utilizar
alturas=[1,2,3,4,5,6,7,8,9,10]


#Modelo sin K-Folding y con conjunto de desarrollo
resultado_sin_kfolding=[]
for altura in alturas:
        arbol_elegido = tree.DecisionTreeClassifier(max_depth = altura,random_state=42)
        arbol_elegido.fit(X_dev, y_dev)
        y_pred = arbol_elegido.predict(X_dev)
        acc=metrics.accuracy_score(y_dev,y_pred)
        resultado_sin_kfolding.append((acc,altura))
print(resultado_sin_kfolding)



#Modelo con Entropia
#mezclamos los datos al principio de los k-fold y ponemos una semilla para tener siempre la misma particion
kfold=KFold(n_splits=5,shuffle=True,random_state=42)


result_entropia = np.zeros((5, len(alturas)))
result_promedio_entropia=[]

for i,(train_index,test_index) in enumerate(kfold.split(X_dev)):
    
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j,altura in enumerate(alturas):

        arbol = tree.DecisionTreeClassifier(criterion='entropy',max_depth = altura,random_state=42)
        arbol.fit(kf_X_train, kf_y_train)
        y_pred = arbol.predict(kf_X_test)
        a=metrics.accuracy_score(kf_y_test,y_pred)
        result_entropia[i,j]=a
result_promedio_entropia=result_entropia.mean(axis=0) 
     
 
for i,e in enumerate(alturas):
    print(f'Score promedio del modelo con hmax con entropia = {e}: {result_promedio_entropia[i]:.4f}')   



#Modelo con GINI
result_gini = np.zeros((5, len(alturas)))
result_promedio_gini=[]

for i,(train_index,test_index) in enumerate(kfold.split(X_dev)):
    
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j,altura in enumerate(alturas):

        arbol = tree.DecisionTreeClassifier(criterion='gini',max_depth = altura,random_state=42)
        arbol.fit(kf_X_train, kf_y_train)
        y_pred = arbol.predict(kf_X_test)
        a=metrics.accuracy_score(kf_y_test,y_pred)
        result_gini[i,j]=a
result_promedio_gini=result_gini.mean(axis=0) 
     
 
for i,e in enumerate(alturas):
    print(f'Score promedio del modelo con hmax con gini = {e}: {result_promedio_gini[i]:.4f}')   



#Modelo con Heldout utilizando los mejores hiperparametros que consguimos de probar el modelo en el conjunto de desarrollo.
arbol_elegido = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 10,random_state=42)
arbol_elegido.fit(X_dev, y_dev)
y_pred_heldout = arbol_elegido.predict(X_heldout)
acc_heldout=metrics.accuracy_score(y_heldout,y_pred_heldout)
matriz_confusion=metrics.confusion_matrix(y_heldout, y_pred_heldout)
print("Matriz de confusión: ")
print(matriz_confusion)
print("Exactitud: ")
print(acc_heldout)




#Funciones para calcular medidad de performance para las clases despues de usar el modelo del arbol 
def calcularrecall(lista):
    res = []
    for i in range(len(lista)):
        sumatotal = 0
        for j in range(len(lista)):
            sumatotal += lista[i][j]
        recall = lista[i][i]/sumatotal
        res.append(recall)
    return res

print("Recall por clase: ")
print(calcularrecall(matriz_confusion))
   
def calculcarprecision(lista):
    res = []
    for i in range(len(lista)):
        sumatotal = 0
        for j in range(len(lista)):
            sumatotal += lista[j][i]
        precision = lista[i][i]/sumatotal
        res.append(precision)
    return res

print("Precisión por clase: ")
print(calculcarprecision(matriz_confusion))


def calcularF1score(precision, recall):
    res = []
    for i in range(len(precision)):
        n = (2 * precision[i] * recall[i])/(recall[i] + precision[i])
        res.append(n)
    return res
print("F1 score por clase: ")
print(calcularF1score(calculcarprecision(matriz_confusion), calcularrecall(matriz_confusion)))

