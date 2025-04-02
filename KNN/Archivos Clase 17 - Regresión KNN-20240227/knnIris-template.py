# -*- coding: utf-8 -*-
"""
Materia     : Laboratorio de datos - FCEyN - UBA
Clase       : Clase Clasificacion
Detalle     : Modelo KNN
Autores     : Manuela Cerdeiro y Pablo Turjanski
Modificacion: 2023-10-24
"""

# Importamos bibliotecas
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#%%
####################################################################
########  MAIN
####################################################################
# Cargamos el archivo 
iris     = load_iris(as_frame = True)
dataIris = iris.frame # Iris en formato dataframe (5 variables)

# Para comprender la variable target armamos un diccionario con equivalencia
print(iris.target_names)
diccionario = dict(zip( [0,1,2], iris.target_names)) # Diccionario con equivalencia


#%%
# ----------------------------------
# ----------------------------------
#       Modelo KNN
# ----------------------------------
# ----------------------------------
#  X = RU (variable predictora) [Dosis de Roundup]
#  Y = ID (variable a predecir) [Damage Index]
########################
## Generacion del modelo
########################
# Declaramos las variables
X = dataIris.iloc[:,0:4]
Y = dataIris.iloc[:,  4]
k = 5
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k)
# Entrenamos el modelo
neigh.fit(X, Y)
#################################################
## Evaluacion del modelo contra dataIris completo
#################################################
# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("R^2 (test ): %.2f" % neigh.score(X, Y))


#%%
#################################################
## Generacion archivos TEST / TRAIN
#################################################
# Dividimos en test(30%) y train(70%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=314)

##################################################################
## Repetimos todo, pero en funcion de TEST y TRAIN y variando el k
##################################################################
# Rango de valores por los que se va a mover k
valores_k = range(1, 10)

resultados_test = np.zeros(len(valores_k))
resultados_train = np.zeros(len(valores_k))

for k in valores_k:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, Y)
    resultados_train[k-1] = neigh.score(X_train, Y_train)
    resultados_test[k-1] = neigh.score(X_test, Y_test)

promedios_train = np.mean(resultados_train, axis=0)
promedios_test = np.mean(resultados_test, axis=0)
##################################################################
## Graficamos R2 en funcion de k (para train y test)
##################################################################


#%%
#############################################################
## Repetimos todo, realizando varios experimentos para cada k
#############################################################
# Rango de valores por los que se va a mover k

#  Cantidad de veces que vamos a repetir el experimento

# Matrices donde vamos a ir guardando los resultados

# Realizamos la combinacion de todos los modelos (Nrep x k)

# Promediamos los resultados de cada repeticion

##################################################################
## Graficamos R2 en funcion de k (para train y test)
##################################################################
