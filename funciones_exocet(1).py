import numpy as np
import scipy
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns
import networkx as nx # Construcción de la red en NetworkX

museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()


def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(A):
    # Función que calcula la descomposición LU
    # A matriz de adyaciencia
    # Retorna las matrices L y U
    m=A.shape[0] # Tomamos dimension de A
    n=A.shape[1]
    if m!=n:  # Acá chequeamos que la matriz sea cuadrada
        print('Matriz no cuadrada') 
        return
    L=np.eye(n) # Creamos L 
    U=A.copy()  # Creamos U
    for j in range(n): # Con este nido de ciclos redefinimos L y U
        for i in range(j+1,n):
            L[i,j]=U[i,j]/U[j,j] # Rellenamos L con los valores de U divididos por los de la diagonal 
            U[i,:]=U[i,:]-L[i,j]*U[j,:] # Rellenamos U con los valores de la filas de U menos los valores correspondientes de L mulplicado por las filas anteriores de U
    return L, U

def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    n=A.shape[0] # Tomamos dimension de A
    
    At=np.transpose(A) # Trasponemos A
    
    k=np.eye(n) # Creamos k
    for i in range(n): # Llenamos la diagonal de k con la suma de las filas
        v=0
        for j in range(n): # Sumamos el valor de las filas
            v+=A[i,j]
        k[i,i]=v  
    Kinv = np.eye(n) # Calculamos la inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    for i in range(n): # Calculamos la inversa
        Kinv[i,i] = 1/k[i,i]
    C = At@Kinv # Calcula C multiplicando Kinv y A
    return C


def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    I = np.eye(N) # Creamos la Identidad
    M = (N/alfa)*(I-(1-alfa)*C) #creamos M
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones(A.shape[0]) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.

    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p


def grafico(D,m,alfa): 
    # Funcion para graficar las visualizaciones del mapa de CABA y además los museos principales y los valores del Pagerank
    # D matriz de distancias
    # m cantidad de conexiones entre museos
    # alfa factor de amortiguamiento
    A = construye_adyacencia(D, m) #creamos A

    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
    
    factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles
    fig, ax = plt.subplots(figsize=(5, 5)) # Visualización de la red en el mapa
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
    
    pr = calcula_pagerank(A,alfa)# Calculamos el PageRank
    pr = pr/pr.sum() # Normalizamos para que sume 1
    
    Nprincipales = 3 # Cantidad de principales
    principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los N principales
    labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos
    
    nx.draw_networkx(G,G_layout,node_size = pr*factor_escala, ax=ax,with_labels=False) # Graficamos red
    nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k") # Agregamos los nombres
    return principales,pr

def juntar_Arrays(tops): 
    # Función para obtener la lista de museos relevantes segun m o alfa nos sivre para el grafico del 3)a)a)
    # tops lista de todos los museos principales, incluye repetidos 
    l=[] # Creamos la lista vacia
    for t in tops: # Llenamos la lista l con los elementos sin repetir de los arrays de tops
        for i in t:
            if i not in l:
                l.append(i)
    return l   
############################### APARTADO DE RESULTADOS PARA GRAFICAR

top0,pr0 = grafico(D,3,1/5)
ltops = [top0]
ltops = (juntar_Arrays(ltops))


top1,pr1 =grafico(D,1,1/5)
top2,pr2=grafico(D,5,1/5)
top3,pr3=grafico(D,10,1/5)
ltopsb = [top1,top2,top3]
ltopsb = (juntar_Arrays(ltopsb))



top4,pr4=grafico(D,5,6/7)
top5,pr5=grafico(D,5,4/5)
top6,pr6=grafico(D,5,2/3)
top7,pr7=grafico(D,5,1/2)
top8,pr8=grafico(D,5,1/3)
top9,pr9=grafico(D,5,1/7)
ltopsc = [top4,top5,top6,top7,top8,top9]
ltopsc = (juntar_Arrays(ltopsc))

############################### GRAFICOSS B)

def graficadoraM():
    # Función para graficar la variación de m (cantidad de conexiones) del ejercicio 3)b)b)
    m_values = [1,3, 5, 10]  # Valores de m
    pr_values = [pr1,pr0, pr2, pr3]  # Lista de PageRanks para cada m
    
    plt.figure(figsize=(8, 5)) # Creamos el gráfico
    
    ltopsFinal=[top0,top1,top2,top3] # Unimos los museos principales
    ltopsFinal = (juntar_Arrays(ltopsFinal)) # Eliminamos repetidos
    
    for museo in ltopsFinal: # Agregamos al gráfico los museos con su PageRank correspondiente
        plt.plot(m_values, [pr[museo] for pr in pr_values], marker="o", label=f'Museo {museo}') 
        
    # Nombramos los ejes y título del gráfico
    plt.xlabel('Cantidad de conexiones (m)')
    plt.ylabel('PageRank')
    plt.title('Evolución del PageRank de los museos más centrales')

    plt.legend()
    plt.show() 


def graficadoraA():
    # Función para graficar la variación de alfa (factor de amortiguamiento) del ejercicio 3)b)b)
    a_values = [6/7,4/5,2/3,1/2,1/3,1/5,1/7]  # Valores de alfa
    pr_values = [pr4,pr5,pr6,pr7,pr8,pr2,pr9]  # Lista de PageRanks para cada alfa
    
    plt.figure(figsize=(8, 5)) # Creamos el gráfico
    
    ltopsFinal=[top4,top5,top6,top7,top8,top9] # Unimos los museos principales
    ltopsFinal = (juntar_Arrays(ltopsFinal)) # Eliminamos repetidos
    
    for museo in ltopsFinal: # Agregamos al gráfico los museos con su PageRank correspondiente
        plt.plot(a_values, [pr[museo] for pr in pr_values], marker="o", label=f'Museo {museo}')
        
    # Nombramos los ejes y título del gráfico
    plt.xlabel('Alfa')
    plt.ylabel('PageRank')
    plt.title('Evolución del PageRank de los museos más centrales')

    plt.legend()
    plt.show() 

###################################################

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)

    n=D.shape[0] #Tomamos la dimensión de D
    k=np.eye(n) #Creamos k
    for i in range(n): #Llenamos la diagonal de k con la suma de las filas
        v=0
        for j in range(n): #Sumamos los elementos de las filas
            v+=F[i,j]
        k[i,i]=v 
    Kinv = np.eye(n) # Creamos K inversa
    for i in range(n): #Rellenamos la diagonal de K inversa
        Kinv[i,i] = 1/k[i,i] # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = F.transpose()@Kinv
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    n = C.shape[0] #Tomamos la Dimensión de C
    B = np.eye(n) #Creamos una matriz vacía
    cm = C.copy() #Tomamos una copia de C
    for i in range(1,cantidad_de_visitas): # Sumamos las matrices de transición para cada cantidad de pasos
        B+=cm 
        cm = cm@C
    return B

#punto5
###############



###############
def calcula_norma1_vector(V):
    # Función para calcular la norma 1 de un vector
    # V vector
    sum=0 # Inicializamos el resultado en 0
    for num in V: # Sumamos los módulos de los elementos de V
        sum+=abs(num)
    return sum    

def norma_vectorV(D,w): 
    # Función para calcular el vector (cantidad total de visitas) y la norma del mismo, a partir de la matriz de distancias
    # D matriz de distancias
    # W es el vector que tiene en sus componentes el numero total de visitas a cada museo
    # Calculamos las matrices C (matriz de transición), B (sumatoria de C), L y U (matrices de la descomposición LU)
    C=calcula_matriz_C_continua(D) 
    B=calcula_B(C,3)
    L,U=calculaLU(B)
    
    # Calculamos los vectores y (intermedio del proceso de descomposición LU) y v (resultado del proceso de descpmposición LU)
    y = scipy.linalg.solve_triangular(L,w,lower=True) 
    v = scipy.linalg.solve_triangular(U,y)
    return calcula_norma1_vector(v)

def calcula_norma1_matriz(mat): 
    # Devuelve la norma 1 de una matriz
    # mat matriz
    n = len(mat) # Devulve la cantidad de filas
    lnormas = []
    for j in range(n): # Suma todos los absolutos de una columna y los pone en una lista
        suma = 0
        for i in range(n):
            suma+=abs(mat[i,j])
        lnormas.append(suma)
    return max(lnormas) # Devuelve el numero mas grande de la lista

def calcula_inversa(M):
    # Función para ecalcular la inversa de una matriz
    n = M.shape[0]
    # M matriz
    G = M.copy()
    I=np.eye(n) #matriz Identidad
    # Proceso de descomposición LU
    L,U=calculaLU(M)
    y=scipy.linalg.solve_triangular(L,I,lower=True)
    Inversa=scipy.linalg.solve_triangular(U,y,lower=False)
    return Inversa

def error_estimado(D):
    # Función para calcular el error estimado dado en el ejercicio 6)
    # D matriz de distancias
    C=calcula_matriz_C_continua(D) # Creamos C
    B=calcula_B(C,3) # Creamos B
    numero_condB=calcula_norma1_matriz(B)*calcula_norma1_matriz(calcula_inversa(B)) # Calculamos el número de condicion de B
    error=0.05
    error_estimadoV=numero_condB*error # Calculamos el error estimado
    return error_estimadoV,numero_condB

