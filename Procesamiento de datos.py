import numpy as np
empleado_01=np.array([[20222333,45,2,20000],
                      [33456234,40,0,25000], 
                      [45432345,41,1,10000]])
Empleado_01=[[20222333,45,2,20000],
             [33456234,40,0,25000], 
             [45432345,41,1,10000],
             [43967304,37,0,12000],
             [42236276,36,0,18000]]
#1)

def superan_salario(M,umbral):
    res=[]
    for empleado in M:
        if empleado[3]>=umbral:
            res.append(empleado)
    return res       
def superan_salario1(M,umbral): 
     return M[M[:,3]>umbral]
print(superan_salario(Empleado_01,15000))
#1) Bastante poco la verdad no tenia gran dificultad


#2)
empleado_02=np.array([[20222333,45,2,20000],
                      [33456234,40,0,25000], 
                      [45432345,41,1,10000],
                      [43967304,37,0,12000],
                      [42236276,36,0,18000]])
#print(superan_salario1(empleado_02,15000))
#2)Ya probe cambiando la matriz y sigue andando


#3)
empleado_03=np.array([[20222333,20000,45,2],
                      [33456234,25000,40,0], 
                      [45432345,10000,41,1],
                      [43967304,12000,37,0],
                      [42236276,18000,36,0]])
Empleado_03=[[20222333,20000,45,2],
             [33456234,25000,40,0], 
             [45432345,10000,41,1],
             [43967304,12000,37,0],
             [42236276,18000,36,0]]
#print(superan_salario1(empleado_03,15000))
def superan_salario03(M,umbral):
    M=M[:,[0,2,3,1]]     #Reordeno las filas
    return M[M[:,3]>umbral]

def super3(M,umbral):
    res=[]
    for empleado in M:
        if empleado[1] > umbral:
            res.append([empleado[0],empleado[2],empleado[3],empleado[1]])
    return res        

print(super3(Empleado_03,15000))
#print(superan_salario03(empleado_03,15000))
#3) No funciona justamente porque la funcion utilizaba que conocia la ubicacion en la lista
#Primero ordeno las filas y despues hago lo mismo que para los otros

#4)
#No hace falta probar pero claramente no va a funcionar bien xq justamente esta armada para recorrer las filas y haciendolo por columna
# me va a chequear los dni o los hijos y no el salario
empleado_04=empleado_03.transpose()
#print(empleado_04)


def superan_salario04(M,umbral):
    M=M.transpose()
    M=M[:,[0,2,3,1]]     
    return M[M[:,3]>umbral]
#print(superan_salario04(empleado_04,15000))
#Primero traspongo la matriz, luego acomodo la columna y despues hago lo mismo que las otras funciones

Empleado_04=[[20222333,33456234,45432345,43967304,42236276],
             [20000,25000,10000,12000,18000],
             [45,40,41,37,36],
             [2,0,1,0,0]]
def super4(M,umbral):
    traspuesta=[]
    for i in range(len(M[0])):
        nuevaFila=[]
        for fila in M:
            nuevaFila.append(fila[i])
        traspuesta.append(nuevaFila)
    return super3(traspuesta,umbral)        
print(super4(Empleado_04,15000))
#5) PREGUNTAS
#1)a) Cuando agregamos mas filas no mucho realmente ya que basicamente buscaba en la misma posicion mas veces
#1)b) Cuando cambio el orden si hubo que aplicar un paso extra pero era basicamente lo mismo con un agregado 
#
#2) Ahi si agregamos algo mas y de no poder usarse la funcion habria que programar bastante mas pero el final en todos los casos es el mismo 
# 
# 3) Mucha comodidad sabiendo que solo aplicandolo funciona como tiene que funcionar y no perder tiempo en esa parte
#
#
#