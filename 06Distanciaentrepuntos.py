import numpy as np 
import pandas as pd
from scipy.spatial import distance

#Definimos las coordenadas de nuestro sistema de tiendas

tiendas ={
    'Tienda A': (1,1),
    'Tienda B': (1,5),
    'Tienda C': (7,1),
    'Tienda D': (3,3),
    'Tienda E': (4,8),
}

#Convertir las coordenadas en un frame para faciltar el calculo

df_tiendas = pd.DataFrame(tiendas).T
df_tiendas.columns = ['X', 'Y']
print('Coordenadas de las tiendas: ')
print(df_tiendas)

#Inicializamos los dataframe de lo que vamos a obtener para el calculo de distancias 

distancias_punto1 = pd.DataFrame(index=df_tiendas.index, columns=df_tiendas.index)

distancias_punto2 = pd.DataFrame(index=df_tiendas.index, columns=df_tiendas.index)

distancias_punto3 = pd.DataFrame(index=df_tiendas.index, columns=df_tiendas.index)

#Vamos a calcular las distancias 
for i in df_tiendas.index:
    for j in df_tiendas.index:
        #Defino la distancia euclidiana del primer punto
        distancias_punto1.loc[i, j] = distance.euclidean(df_tiendas.loc[i], df_tiendas.loc[j])
        
        distancias_punto2.loc[i, j] = distance.cityblock(df_tiendas.loc[i], df_tiendas.loc[j])
        
        distancias_punto3.loc[i, j] = distance.chebyshev(df_tiendas.loc[i], df_tiendas.loc[j])
        
        #Mostrar resultados 
        
        print('/n Distancia Euclidiana entre cada una de las tiendas: ')
        
        print(distancias_punto1)
        
        print('/n Distancia Manhattan entre cada una de las tiendas: ')
        
        print(distancias_punto2)
        
        print('/n Distancia Chebyshev entre cada una de las tiendas: ')
        
        print(distancias_punto3)