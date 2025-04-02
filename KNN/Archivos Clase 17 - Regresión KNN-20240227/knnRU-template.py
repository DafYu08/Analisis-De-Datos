# -*- coding: utf-8 -*-
"""
Materia     : Laboratorio de datos - FCEyN - UBA
Clase       : Clase Regresion Lineal
Detalle     : Modelo KNN
Autores     : Manuela Cerdeiro y Pablo Turjanski
Modificacion: 2023-10-21
"""

# Importamos bibliotecas
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns

#%%
####################################################################
########  DEFINICION DE FUNCIONES AUXILIARES
####################################################################

# Dibuja una recta. Toma como parametros pendiente e intercept
def plotRectaRegresion(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

#%%
####################################################################
########  MAIN
####################################################################
# Cargamos el archivo 
carpeta = '/home/Estudiante/Descargas/Archivos Clase 17 - Regresión KNN-20240227/'
data_train = pd.read_csv(carpeta+"datos_roundup.txt", sep=" ", encoding='utf-8')
#%%

# ----------------------------------
# ----------------------------------
#       Modelo KNNred
# ----------------------------------
# ----------------------------------
#  X = RU (variable predictora) [Dosis de Roundup]
#  Y = ID (variable a predecir) [Damage Index]
########################
## Generacion del modelo
########################
# Declaramos las variables
X = data_train[['RU']]
Y = data_train[['ID']]
k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsRegressor(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X, Y)
#####################################
## Evaluacion del modelo contra TRAIN
#####################################
# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("R^2 (test ): %.2f" % neigh.score(X, Y))

#####################################
## Prediccion
#####################################
# Cargamos el archivo (no posee valores para ID)
carpeta = '/home/Estudiante/Descargas/Archivos Clase 17 - Regresión KNN-20240227/'
data_a_predecir = pd.read_csv(carpeta+"datos_a_predecir.txt", sep=" ", encoding='utf-8')

# Realizamos la prediccion de ID utilizando el modelo y
# la asignamos a la columna ID
data_a_predecir[['ID']] = neigh.predict(data_a_predecir[['RU']])
# Graficamos una dispersion de puntos de ID en funcion de la Dosis de RU
# Graficamos tanto los puntos de entrenamiento del modelo como los predichos

ax = sns.scatterplot(data = data_train, x = 'RU', y = 'ID', s = 40, color = 'blue')

ax = sns.scatterplot(data = data_a_predecir, x = 'RU', y = 'ID', s = 40, color = 'green')

ax.set_xlabel("Dosis de RU (ug/huevo)")
ax.set_ylabel("Indice de daño")

plotRectaRegresion(0.037, 106.5)
# Graficamos la recta que habiamos obtenido en el modelo RLS la clase pasada



