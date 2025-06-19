import pandas as pd
from inline_sql import sql, sql_val

f1='/users/Guille/Desktop/Downloads/casos.csv'
f2='/users/Guille/Desktop/Downloads/departamento.csv'
f3='/users/Guille/Desktop/Downloads/grupoetario.csv'
f4='/users/Guille/Desktop/Downloads/provincia.csv'
f5='/users/Guille/Desktop/Downloads/tipoevento.csv'

casos=pd.read_csv(f1)
depatamento=pd.read_csv(f2)
grupoetario=pd.read_csv(f3)
provincia=pd.read_csv(f4)
tipoevento=pd.read_csv(f5)
