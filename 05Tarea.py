import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')

data = df['median_house_value']

media = np.mean(data)
mediana = np.median(data)

moda_result = stats.mode(data, keepdims=True)
moda = moda_result.mode[0] if moda_result.count[0] > 1 else "No hay moda significativa"

rango = np.ptp(data)
varianza = np.var(data, ddof=1)  
desviacion_estandar = np.std(data, ddof=1)  

print("Estadísticas Descriptivas:")
print(f"Media: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Moda: {moda}")
print(f"Rango: {rango:.2f}")
print(f"Varianza: {varianza:.2f}")
print(f"Desviación Estándar: {desviacion_estandar:.2f}")

frecuencias = pd.cut(data, bins=10).value_counts().sort_index()
print("\nTabla de Frecuencias:")
print(frecuencias)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(media, color='red', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')
plt.axvline(mediana, color='green', linestyle='dashed', linewidth=2, label=f'Mediana: {mediana:.2f}')
plt.xlabel('Median House Value')
plt.ylabel('Frecuencia')
plt.title('Distribución de los valores medianos de las casas')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(df['population'], df['median_house_value'], alpha=0.5, color='blue')
plt.xlabel('Población')
plt.ylabel('Valor Mediano de la Casa')
plt.title('Valor Mediano de la Casa vs Población')

plt.tight_layout()
plt.show()
