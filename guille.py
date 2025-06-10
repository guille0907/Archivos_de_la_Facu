import numpy as np
import scipy
import math
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns
import networkx as nx # Construcción de la red en NetworkX
import funciones_exocet as tf
#%%
# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

#%%
def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    # Have fun!!    
    
    if A.all()!=A.T.all(): 
        As = np.ceil((A+np.transpose(A))/2)#Creamos As simetrica
    else:
        As=A.copy()
    n=As.shape[0] # Tomamos dimension de As
    
    k=np.eye(n) # Creamos k
    for i in range(n): # Llenamos la diagonal de k con la suma de las filas
        v=0
        for j in range(n): # Sumamos el valor de las filas
            v+=As[i,j]
        k[i,i]=v
        
    L = k-As
    
    return L
#%%
def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    # Have fun!!
    As = np.ceil((A+np.transpose(A))/2)#Creamos As simetrica
    n=As.shape[0] # Tomamos dimension de As
    
    dosE = np.sum(As) #Creamos 2E
    k=np.eye(n) # Creamos k
    for i in range(n): # Llenamos la diagonal de k con la suma de las filas
        v=0
        for j in range(n): # Sumamos el valor de las filas
            v+=As[i,j]
        k[i,i]=v
    
    p=np.eye(n) # Creamos p
    for i in range(n):
        for j in range(n):
            p[i,j] = (k[i,i]*k[j,j])/dosE
    
    R = As - p
    
    return R
#%%
def calcula_lambda(L,v):
    # Recibe L y v y retorna el corte asociado
    # Have fun!
    s = np.array(len(v))
    for i in range(len(v)):
        s[i] = np.sign(v)
    
    lambdon = (np.transpose(s)@L@s)/4
    
    return lambdon

def calcula_Q(R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s = np.array(len(v))
    for i in range(len(v)):
        s[i] = np.sign(v)
    
    Q = (np.transpose(s)@R@s)    
    return Q

#%%
def norma_2(v):
    cont = 0
    for i in range(len(v)):
        cont = cont + v[i]*v[i]
        
    return math.sqrt(cont)


def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   n = A.shape[0]
    # 1) Vector inicial aleatorio en R^n
   v = np.random.uniform(-1, 1, size=n)
       
   v = v/norma_2(v) # Lo normalizamos
   v1 = A@v # Aplicamos la matriz una vez
   v1 = v1/norma_2(v1) # normalizamos
   l = (np.transpose(v)@A@v)/(np.transpose(v)@v) # Calculamos el autovalor estimado
   l1 = (np.transpose(v1)@A@v1)/(np.transpose(v1)@v1) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A@v # Calculo nuevo v1
      v1 = v1/norma_2(v1) # Normalizo
      l1 = (np.transpose(v1)@A@v1)/(np.transpose(v1)@v1) # Calculo autovalor
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = (np.transpose(v1)@A@v1)/(np.transpose(v1)@v1) # Calculamos el autovalor
   return v1,l,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1*((np.outer(v1,v1.T))/(v1.T@v1)) # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   # Have fun!
   deflA=A - l1*((np.outer(v1,v1.T))/(v1.T@v1))
   return metpot1(deflA,tol,maxrep)


def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    I=np.eye(A.shape[0])
    U=mu*I
    M=A+U
    M_res=tf.calcula_inversa(M)
    return metpot1(M_res,tol=tol,maxrep=maxrep)

def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   I=np.eye(A.shape[0])
   U=mu*I
   X = A+U # Calculamos la matriz A shifteada en mu
   iX = np.linalg.inv(X)# La invertimos
   defliX = deflaciona(iX,tol,maxrep) # La deflacionamos
   v,l,_ =  metpot1(defliX,tol,maxrep) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_


def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        #v1,l1,_=metpot1(L,tol=1e-8,maxrep=np.inf)
        v,l,_ = metpotI2(L,1e-1,tol=1e-8,maxrep=np.inf) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        """pos = []
        neg = []
        for i in range(len(v)):
            if v[i] < 0:
                pos.append(A[i,:])
                neg.append(0*A[i,:])
            else:
                neg.append(A[i,:])
                pos.append(0*A[i,:])"""
        Ap = A[v>=0,:][:,v>=0] # Asociado al signo positivo
        Am = A[v<0,:][:,v<0]  # Asociado al signo negativo

        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return(nombres_s)
    else:
        v,l,_ = metpot1(R,tol=1e-8,maxrep=np.inf) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return(nombres_s)
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            
            Rp = R[v>=0,:][:,v>=0] # Asociado al signo positivo
            Rm = R[v<0,:][:,v<0]  # Asociado al signo negativo
                               # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp,tol=1e-8,maxrep=np.inf)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm,tol=1e-8,maxrep=np.inf) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                nombres_p = [nombres_s[i] for i in range(min(len(v), len(nombres_s))) if v[i] > 0]
                nombres_m = [nombres_s[i] for i in range(min(len(v), len(nombres_s))) if v[i] < 0]
                comunidadesPos = modularidad_iterativo(A,Rp,nombres_p)
                comunidadesNeg = modularidad_iterativo(A,Rm,nombres_m)

                return(comunidadesPos + comunidadesNeg)
  
                
