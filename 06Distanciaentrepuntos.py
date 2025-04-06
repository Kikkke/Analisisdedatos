import numpy as np
import pandas as pd
from scipy.spatial import distance

#Definimos las coordenadas de nuestros puntos
puntos = {
    'Punto A': (2, 3),
    'Punto B': (5, 4),
    'Punto C': (1, 1),
    'Punto D': (6, 7),
    'Punto E': (3, 5),
    'Punto F': (8, 2),
    'Punto G': (4, 6),
    'Punto H': (2, 1)
}

#convertir las coordenadas en un DataFrame para facilitar el cálculo
df_puntos = pd.DataFrame(puntos).T
df_puntos.columns = ['X', 'Y']
print('Coordenadas de los puntos:')
print(df_puntos)

#inicializamos los DataFrames para almacenar las distancias
distancias_euclidiana = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)
distancias_manhattan = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)
distancias_chebyshev = pd.DataFrame(index=df_puntos.index, columns=df_puntos.index)

#calculamos las distancias entre todos los pares de puntos
for i in df_puntos.index:
    for j in df_puntos.index:
        #Distancia Euclidiana (norma L2)
        distancias_euclidiana.loc[i, j] = distance.euclidean(df_puntos.loc[i], df_puntos.loc[j])
        #Distancia Manhattan (norma L1)
        distancias_manhattan.loc[i, j] = distance.cityblock(df_puntos.loc[i], df_puntos.loc[j])
        #Distancia Chebyshev (norma L∞)
        distancias_chebyshev.loc[i, j] = distance.chebyshev(df_puntos.loc[i], df_puntos.loc[j])

#mostramos las matrices de distancia
print('\nMatriz de distancias Euclidianas:')
print(distancias_euclidiana)

print('\nMatriz de distancias Manhattan:')
print(distancias_manhattan)

print('\nMatriz de distancias Chebyshev:')
print(distancias_chebyshev)

#función para encontrar los pares más cercanos y más alejados
def encontrar_extremos(distancias, nombre_metrica):
    # Ignoramos la diagonal (distancias entre el mismo punto)
    mascara = ~np.eye(len(distancias), dtype=bool)
    valores = distancias.values[mascara]
    
    # Encontramos la mínima y máxima distancia
    min_dist = np.min(valores)
    max_dist = np.max(valores)
    
    # Encontramos todos los pares con esas distancias
    pares_minimos = np.where(distancias == min_dist)
    pares_maximos = np.where(distancias == max_dist)
    
    # Filtramos para evitar duplicados (A-B vs B-A)
    pares_minimos = [(df_puntos.index[i], df_puntos.index[j]) 
                     for i, j in zip(*pares_minimos) if i < j]
    pares_maximos = [(df_puntos.index[i], df_puntos.index[j]) 
                     for i, j in zip(*pares_maximos) if i < j]
    
    print(f'\nAnálisis para {nombre_metrica}:')
    print(f'Distancia mínima: {min_dist:.2f} entre los pares: {pares_minimos}')
    print(f'Distancia máxima: {max_dist:.2f} entre los pares: {pares_maximos}')

#aplicamos la función a cada tipo de distancia
encontrar_extremos(distancias_euclidiana, 'Distancia Euclidiana')
encontrar_extremos(distancias_manhattan, 'Distancia Manhattan')
encontrar_extremos(distancias_chebyshev, 'Distancia Chebyshev')