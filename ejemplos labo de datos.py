import numpy as np
import pandas as pd
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.concatenate((a,b)))
altura_inicial = 100
# Factor de rebote
factor_rebote = 3 / 5
# Número de rebotes
rebotes = 10

print("Rebote\tAltura (m)")
print("----------------------")
altura = altura_inicial
for i in range(1, rebotes + 1):
    # Calcular la nueva altura
    altura *= factor_rebote
    # Imprimir el número de rebote y la altura correspondiente
    print(f"{i}\t{altura:0.2f} m")

def traductor_geringoso(lista)->dict:
    dicci=dict()
    cont=0
    palabra=""
    for elem in lista:
        for letra in elem:
            if cont<2:
                palabra+=letra
                cont+=1
            else:
                palabra+="pa"+letra  
                cont=1  
        dicci[elem]=palabra+"pa"
        palabra=""
        cont=0
    return dicci


a=["banana","manzana","mandarina"]
print(traductor_geringoso(a))

print("\t0   1   2   3   4   5   6   7   8   9")
print("------------------------------------------------")
j=0
for i in range(10):
    print(f"{i}:\t{i*0}   {i*1}   {i*2}   {i*3}   {i*4}   {i*5}   {i*6}   {i*7}   {i*8}   {i*9}")

with open("datame.txt","rt",encoding="utf-8") as texto:
    for linea in texto:
        if " estudiantes " in linea:   
            print(linea.strip())

def cuantas_materias(n):
    cont=0
    with open("cronograma_sugerido.csv","rt",encoding="utf-8") as doc:
        for line in doc:
            datos=line.split(',')
            if datos[0]==str(n):
                cont+=1
    return cont            

print(cuantas_materias(7))

def pisar_elemento(M,e):
    for vector in M:
        for i in range(0,len(vector)):
            if vector[i]==e:
                vector[i]=-1
    return M                

M=np.linalg.inv([[1,0,-1,0], [0,0,1,0],[2,1,-2,3],[3,1,-1,3]])
e=2
print(M)
print(pisar_elemento(M,e))



def leer_parque(nombre_archivo,parque):
    lista=[]
    dicci=dict()
    with open("arbolado-en-espacios-verdes.csv","rt",encoding="utf-8") as doc:
        for line in doc:
            data=line.split(',')
            if data[10]==parque:
                dicci[data[2]]=data
                lista.append(dicci) 
            dicci=dict()
    print(len(lista))
    return lista                   
    
print(leer_parque("arbolado-en-espacios-verdes.csv",'GENERAL PAZ'))
b=leer_parque("arbolado-en-espacios-verdes.csv",'GENERAL PAZ')


def especies(lista_arboles):
    especias=set()
    for dicci in lista_arboles:
        for clave,valor in dicci.items():
                especias.add(valor[7])
    return especias            

a=[{'1380': ['-58.5050933042', '-34.5675817714', '1380', '20', '33', '5', '330', 'Eucalipto', 'Eucalyptus sp.', 'Árbol Latifoliado Perenne', 'GENERAL PAZ', '"LARRALDE', ' CRISOLOGO', ' AV. - PAZ', ' GRAL.', ' AV.- AIZPURUA"', 'Mirtáceas', 'Eucalyptus', 'Exótico', '96164.756178', '106842.46540700001\n']},
    {'1381': ['-58.5051308009', '-34.5676013198', '1381', '178', '31', '78', '330', 'Eucalipto', 'Eucalyptus sp.', 'Árbol Latifoliado Perenne', 'GENERAL PAZ', '"LARRALDE', ' CRISOLOGO', ' AV. - PAZ', ' GRAL.', ' AV.- AIZPURUA"', 'Mirtáceas', 'Eucalyptus', 'Exótico', '96161.310446', '106840.291945\n']}, 
    {'1384': ['-58.5045414367', '-34.5673084626', '1384', '18', '142', '0', '330', 'GUILLI', 'Eucalyptus sp.', 'Árbol Latifoliado Perenne', 'GENERAL PAZ', '"LARRALDE', ' CRISOLOGO', ' AV. - PAZ', ' GRAL.', ' AV.- AIZPURUA"', 'Mirtáceas', 'Eucalyptus', 'Exótico', '96215.38080700001', '106872.80533300001\n']},
   {'1385': ['-58.504455406000005', '-34.5669880274', '1385', '7', '31', '0', '302', 'Cedro de San Juan', 'Cupressus lusitanica', 'Árbol Conífero Perenne', 'GENERAL PAZ', '"LARRALDE', ' CRISOLOGO', ' AV. - PAZ', ' GRAL.', ' AV.- AIZPURUA"', 'Cupresáceas', 'Cupressus', 'Exótico', '96223.26193', '106908.35795599999\n']},{'1380': ['-58.5050933042', '-34.5675817714', '1380', '20', '33', '5', '330', 'Eucalipto', 'Eucalyptus sp.', 'Árbol Latifoliado Perenne', 'GENERAL PAZ', '"LARRALDE', ' CRISOLOGO', ' AV. - PAZ', ' GRAL.', ' AV.- AIZPURUA"', 'Mirtáceas', 'Eucalyptus', 'Exótico', '96164.756178', '106842.46540700001\n']},
    {'1381': ['-58.5051308009', '-34.5676013198', '1381', '98', '31', '12', '330', 'Eucalipto', 'Eucalyptus sp.', 'Árbol Latifoliado Perenne', 'GENERAL PAZ', '"LARRALDE', ' CRISOLOGO', ' AV. - PAZ', ' GRAL.', ' AV.- AIZPURUA"', 'Mirtáceas', 'Eucalyptus', 'Exótico', '96161.310446', '106840.291945\n']}, 
    {'1384': ['-58.5045414367', '-34.5673084626', '1384', '18', '142', '172', '330', 'GUILLI', 'Eucalyptus sp.', 'Árbol Latifoliado Perenne', 'GENERAL PAZ', '"LARRALDE', ' CRISOLOGO', ' AV. - PAZ', ' GRAL.', ' AV.- AIZPURUA"', 'Mirtáceas', 'Eucalyptus', 'Exótico', '96215.38080700001', '106872.80533300001\n']},
   {'1385': ['-58.504455406000005', '-34.5669880274', '1385', '7', '31', '5', '302', 'Cedro de San Juan', 'Cupressus lusitanica', 'Árbol Conífero Perenne', 'GENERAL PAZ', '"LARRALDE', ' CRISOLOGO', ' AV. - PAZ', ' GRAL.', ' AV.- AIZPURUA"', 'Cupresáceas', 'Cupressus', 'Exótico', '96223.26193', '106908.35795599999\n']}
   ]
print(especies(a))

def contar_ejemplares(lista_arboles):
    diccionario=dict()
    for dicci in lista_arboles:
        for clave,valor in dicci.items():
            if valor[7] in diccionario:
              diccionario[valor[7]]+=1
            else:
               diccionario[valor[7]]=1   
    return diccionario
print(contar_ejemplares(b))        

def obtener_alturas(lista_arboles,especie):
    lista=[]
    for dicci in lista_arboles:
        for clave,valor in dicci.items():
            if valor[7]==especie:
                lista.append(float(valor[3]))
    return lista            
print(obtener_alturas(a,'Eucalipto'))

def obtener_inclinacion(lista_arboles,especie):
    lista=[]
    for dicci in lista_arboles:
        for clave,valor in dicci.items():
            if valor[7]==especie:
                lista.append(float(valor[5]))
    return lista 

def especimen_mas_inclinado(lista_arboles):
    maximo=("",0)
    especias=especies(lista_arboles)
    for tipo in especias:
        inclinacion =max(obtener_inclinacion(lista_arboles,tipo))
        if inclinacion > maximo[1]:
           maximo=(tipo,inclinacion)
    return maximo
print(especimen_mas_inclinado(a))


d = {'nombre':['Antonio', 'Brenda', 'Camilo', 'David'], 'apellido': ['Restrepo', 'Saenz', 
'Torres', 'Urondo'], 'lu': ['78/23', '449/22', '111/24', '1/21']}
d7 = pd.DataFrame(data = d)
d7.set_index('lu', inplace = True)
print(d7)

df2=pd.DataFrame(data=M)
df2=pd.DataFrame(M,columns=['a','b','c','d'],index=['primer','segundo',"3","4"])
print(df2)

fname='/users/Guille/Desktop/tpguille/python/arbolado-publico-lineal-2017-2018.csv'
df=pd.read_csv(fname)
especies_seleccionadas = ['Tilia x moltkei', 'Jacaranda mimosifolia', 'Tipuana tipu']

filename='/users/Guille/Desktop/tpguille/python/arbolado-en-espacios-verdes.csv'
data=pd.read_csv(filename)
data_arboles_parque=data[['nombre_cie','diametro','altura_tot']]
print(data_arboles_parque)



nuevo_data=df[['nombre_cientifico', 'ancho_acera','diametro_altura_pecho','altura_arbol']]

data_arboles_veredas = nuevo_data[(nuevo_data['nombre_cientifico'] == 'Tilia x moltkei') |
                                   (nuevo_data['nombre_cientifico'] == 'Jacaranda mimosifolia') |
                                   (nuevo_data['nombre_cientifico'] == 'Tipuana tipu')]
data_arboles_veredas["Ambiente"]="Vereda"
print(data_arboles_veredas)


archivo="arbolado-en-espacios-verdes.csv"
Dat=pd.read_csv(archivo)
Data=Dat[(Dat["nombre_com"]=="Palo borracho rosado")]
Cantidad=len(Data) #cantidad de filas
Maximo=Data["altura_tot"].max()
Minimo=Data["altura_tot"].min()


pruebas=pd.read_csv("cronograma_sugerido.csv")

new=pruebas.replace({'Correlatividad de Asignaturas': {"CBC":"Anterior"}})
pd.concat([pruebas,new])
print(pruebas)