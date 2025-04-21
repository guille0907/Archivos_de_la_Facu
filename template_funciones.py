import numpy as np
import scipy
from scipy.linalg import solve_triangular
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
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    if m!=n:
        print('Matriz no cuadrada')
        return
    L=np.eye(n)
    U=Ac
    for j in range(n):
        for i in range(j+1,n):
            L[i,j]=U[i,j]/U[j,j]
            U[i,:]=U[i,:]-L[i,j]*U[j,:]
    return L, U

def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    n=A.shape[0]
    
    At=np.transpose(A)
    
    k=np.eye(n)
    for i in range(n):
        v=0
        for j in range(n):
            v+=A[i,j]
        k[i,i]=v
    Kinv = np.eye(n) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    for i in range(n):
        Kinv[i,i] = 1/k[i,i]
    C = Kinv@At # Calcula C multiplicando Kinv y A
    return C


    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    I = np.eye(N)
    M = (N/alfa)*(I-(1-alfa)*C)
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.empty() # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    b.fill(alfa/N)
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)

    n=D.shape[0]
    k=np.eye(n)
    for i in range(n):
        v=0
        for j in range(n):
            v+=D[i,j]
        k[i,i]=v
    Kinv = np.eye(n) 
    for i in range(n):
        Kinv[i,i] = 1/k[i,i] # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = Kinv@F # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    n = C.shape[0]
    B = np.eye(n)*0
    for i in range(cantidad_de_visitas-1):
        cm = C.copy()# Sumamos las matrices de transición para cada cantidad de pasos
        for j in range(i):
            cm = cm@C
        B+=cm
    return B

#punto5
w = np.loadtxt("visitas.txt")
def ecuacion5(D): 
    C=calcula_matriz_C_continua(D)   
    B=calcula_B(C,3)
    L,U=calculaLU(B)
    y = scipy.linalg.solve_triangular(L,w,lower=True) 
    v = scipy.linalg.solve_triangular(U,y)
    return v

#punto6
def ecuacion(D):
    C=calcula_matriz_C_continua(D)   
    B=calcula_B(C,3)
    numero_condicionB=np.linalg.cond(B,1)
    error=0.05
    error_estimadoV=numero_condicionB*error
    return error_estimadoV