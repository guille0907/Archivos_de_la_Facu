#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 22:04:07 2023

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



arbol=pd.read_csv('/users/Guille/Downloads/arboles.csv')
X1=arbol[['altura_tot','diametro','inclinacio']]
Y1=arbol['nombre_com']



x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
modelo=KNeighborsClassifier(n_neighbors=8)
modelo.fit(x1_train,y1_train)
Y_pre=modelo.predict(x1_test)
a=metrics.accuracy_score(y1_test,Y_pre)
print(metrics.confusion_matrix(y1_test, Y_pre))
print(a)

#%%




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

#####################################################################################################################################
#####################################################################################################################################
#COSAS DEL TP 
indices = [39750,16804,63750,16812,39110,16823,16832,16833,
16870,
16871
]

fig, axes = plt.subplots(1, 10, figsize=(12, 3))

for ax, ith in zip(axes, indices):
    # Extraer y reconstruir la imagen
    img = np.array(data_df.iloc[ith, :-1]).reshape(28, 28)
    # Mostrar la imagen en el subplot correspondiente
    ax.imshow(img, cmap="gray")
    ax.axis("off")  # Opcional: quitar ejes para mayor claridad




# Excluimos la columna 'label' antes de sumar
suma_pixeles = data_df.drop("label", axis=1).sum(axis=1)



# Agregamos la suma como nueva columna
data_df["suma_pixeles"] = suma_pixeles

# Agrupamos por clase y calculamos el promedio
promedio_por_clase = data_df.groupby("label")["suma_pixeles"].mean()



454,534



franjanegra = data_df[['pixel482', 'pixel454', 'pixel510', 'pixel534','pixel506','pixel499', 'label']]
countfranjanegra = """SELECT label, count(*) AS cantidadnegros
                      FROM franjanegra
                      WHERE   pixel534 =0  and  pixel454=0
                      GROUP BY label
                      ORDER BY cantidadnegros DESC
"""
contarnegros = dd.sql(countfranjanegra).df()
print(contarnegros)







pixeles=['pixel770','pixel768','pixel769','pixel10','pixel14','pixel18','pixel527','pixel528','pixel182','pixel555','pixel126','pixel154',
        'pixel618','pixel590','pixel562']



resultados=[]
while len(pixeles)>=2:
 X1=res[pixeles]
 Y1=res['label']
#Separamos entre Train y Test
 x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
 modelo=KNeighborsClassifier(n_neighbors=3)
 modelo.fit(x1_train,y1_train)
 Y_pre=modelo.predict(x1_test)
 a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
 resultados.append((a,len(pixeles)))
 pixeles.pop()

#########################################



############################### BASTANTE BIEN
pixeles=['pixel126','pixel526',]

resultados1=[]
for i in range(2,len(pixeles)+1):
 X1=res[pixeles[:i]]
 Y1=res['label']
#Separamos entre Train y Test
 x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
 modelo=KNeighborsClassifier(n_neighbors=3)
 modelo.fit(x1_train,y1_train)
 Y_pre=modelo.predict(x1_test)
 a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
 resultados1.append((a,i))
#print(resultados1)

#############################################3


############## TOMO RANDOM
pixeles=['pixel770','pixel768','pixel769','pixel10','pixel14','pixel18','pixel527','pixel528','pixel182','pixel555','pixel126','pixel154',
        'pixel618','pixel590','pixel562']
random.shuffle(pixeles)
resultados2=[]
while len(pixeles)>=2:
 X1=res[pixeles]
 Y1=res['label']
#Separamos entre Train y Test
 x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
 modelo=KNeighborsClassifier(n_neighbors=3)
 modelo.fit(x1_train,y1_train)
 Y_pre=modelo.predict(x1_test)
 a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
 resultados2.append((a,len(pixeles)))
 pixeles.pop()
 #if len(pixeles)<=4:
   #print(a,pixeles)


######################################

##### EL QUE MEJOR ANDA
pixeles=['pixel182','pixel10','pixel527'] # MAQUINA
509,537,565
resultados4=[]

X1=res[pixeles]
Y1=res['label']
#Separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
modelo=KNeighborsClassifier(n_neighbors=3)
modelo.fit(x1_train,y1_train)
Y_pre=modelo.predict(x1_test)
a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
resultados4.append((a,i))
#print(resultados4)

#######################################




























#Uno que varia
pixeles=['pixel618','pixel528',]

resultados6=[]

X1=res[pixeles]
Y1=res['label']
#Separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
modelo=KNeighborsClassifier(n_neighbors=3)
modelo.fit(x1_train,y1_train)
Y_pre=modelo.predict(x1_test)
a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
resultados6.append((a,i))

####################################


######################################

pixeles=['pixel154','pixel562','pixel528'] # CHILL 

resultados7=[]

X1=res[pixeles]
Y1=res['label']
#Separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
modelo=KNeighborsClassifier(n_neighbors=3)
modelo.fit(x1_train,y1_train)
Y_pre=modelo.predict(x1_test)
a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
resultados7.append((a,i))

############################################

#MUY BUENO DE DOS
pixeles=['pixel565','pixel537','pixel509']  

resultados8=[]

X1=res[pixeles]
Y1=res['label']
#Separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
modelo=KNeighborsClassifier(n_neighbors=3)
modelo.fit(x1_train,y1_train)
Y_pre=modelo.predict(x1_test)
a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
resultados8.append((a,575))
#print(resultados8)

#####################################


####################################

pixeles=['pixel154','pixel562','pixel528','pixel182','pixel10','pixel527']
resultados9=[]

for i in range(len(pixeles)):
  for j in range(i+1,len(pixeles)):
        dupla=[pixeles[i],pixeles[j]]
        X1=res[dupla]
        Y1=res['label']
#Separamos entre Train y Test
        x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
        modelo=KNeighborsClassifier(n_neighbors=3)
        modelo.fit(x1_train,y1_train)
        Y_pre=modelo.predict(x1_test)
        a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
        resultados9.append((a,dupla))







###### D) Variacion de k
#%%
#################################################
# K=1
pixeles_best=['pixel154','pixel562','pixel528','pixel182','pixel527']

resultados_k1=[]

for i in range(len(pixeles_best)):
  for j in range(i+1,len(pixeles_best)):
    for k in range(j+i,len(pixeles_best)):
        trio=[pixeles_best[i],pixeles_best[j],pixeles_best[k]]
        X1=res[dupla]
        Y1=res['label']
#Separamos entre Train y Test
        x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
        modelo=KNeighborsClassifier(n_neighbors=1)
        modelo.fit(x1_train,y1_train)
        Y_pre=modelo.predict(x1_test)
        a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
        resultados_k1.append((a,trio))
#print(resultados_k1)





#%%
############################
#  K=5
resultados_k5=[]

for i in range(len(pixeles_best)):
  for j in range(i+1,len(pixeles_best)):
        dupla=[pixeles_best[i],pixeles_best[j]]
        X1=res[dupla]
        Y1=res['label']
#Separamos entre Train y Test
        x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
        modelo=KNeighborsClassifier(n_neighbors=5)
        modelo.fit(x1_train,y1_train)
        Y_pre=modelo.predict(x1_test)
        a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
        resultados_k5.append((a,dupla))
#print(resultados_k5)



#%%
##########################
#  K=7
pixeles_best=['pixel182','pixel10','pixel527']
resultados_k7=[]

X1=res[pixeles_best]
Y1=res['label']
#Separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
modelo=KNeighborsClassifier(n_neighbors=7)
modelo.fit(x1_train,y1_train)
Y_pre=modelo.predict(x1_test)
a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
resultados_k7.append((a,pixeles_best))
#print(resultados_k7)







#%%
################################################
#PRUEBA FINAL CON LOS K
pixeles_best=['pixel10','pixel182','pixel527']
resultados_ktotal=[]

X1=res[pixeles_best]
Y1=res['label']
#Separamos entre Train y Test
x1_train, x1_test, y1_train, y1_test=train_test_split(X1,Y1,test_size=0.2)
#Armamos el modelo y vemos que tal predice 
k_candidatos=[1,3,5,7,11,17]
for k in k_candidatos:
    modelo=KNeighborsClassifier(n_neighbors=k)
    modelo.fit(x1_train,y1_train)
    Y_pre=modelo.predict(x1_test)
    a=metrics.accuracy_score(y1_test,Y_pre)
 #print(a,len(pixeles))
 #print(metrics.confusion_matrix(y1_test, Y_pre))
    resultados_ktotal.append((k,a))
#print(resultados_ktotal)

####################################3















        
        
        
        
        
        
        
        
        
        
        
        
        
        