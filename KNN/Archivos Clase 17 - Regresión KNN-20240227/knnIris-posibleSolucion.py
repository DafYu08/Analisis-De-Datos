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
X = dataIris.iloc[:,0:4]     # X = 4 variables predictoras (sepalo/petalo, ancho/alto)
Y = dataIris.iloc[:,  4]     # Y = 1 variable a predecir, la clase de flor (en formato numerico)
k = 5             # Cantidad de vecinos

# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors = k) 

# Entrenamos el modelo
neigh.fit(X, Y) 

#################################################
## Evaluacion del modelo contra dataIris completo
#################################################
# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("R^2 (train ): %.2f" % neigh.score(X, Y))



#%%
#################################################
## Generacion archivos TEST / TRAIN
#################################################
# Dividimos en test(30%) y train(70%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) 


##################################################################
## Repetimos todo, pero en funcion de TEST y TRAIN y variando el k
##################################################################
# Rango de valores por los que se va a mover k
valores_k = range(1, 10)

resultados_test  = np.zeros(len(valores_k))
resultados_train = np.zeros(len(valores_k))

for k in valores_k:
    # Declaramos el tipo de modelo
    neigh = KNeighborsClassifier(n_neighbors = k)
    # Entrenamos el modelo (con datos de train)
    neigh.fit(X_train, Y_train) 
    # Evaluamos el modelo con datos de train y luego de test
    resultados_train[k-1] = neigh.score(X_train, Y_train)
    resultados_test[k-1]  = neigh.score(X_test , Y_test )

##################################################################
## Graficamos R2 en funcion de k (para train y test)
##################################################################
plt.plot(valores_k, resultados_train, label = 'Train')
plt.plot(valores_k, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de knn')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('R^2')
plt.xticks(valores_k)
plt.ylim(0.90,1.00)


#%%
#############################################################
## Repetimos todo, realizando varios experimentos para cada k
#############################################################
# Rango de valores por los que se va a mover k
valores_k = range(1, 20)
#  Cantidad de veces que vamos a repetir el experimento
Nrep = 100
# Matrices donde vamos a ir guardando los resultados
resultados_test  = np.zeros(( Nrep , len(valores_k)))
resultados_train = np.zeros(( Nrep , len(valores_k)))

# Realizamos la combinacion de todos los modelos (Nrep x k)
for i in range(Nrep):
# Matrices donde vamos a ir guardando los resultados
    # Dividimos en test(30%) y train(70%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) 
    # Generamos el modelo y lo evaluamos
    for k in valores_k:
        # Declaramos el tipo de modelo
        neigh = KNeighborsClassifier(n_neighbors = k)
        # Entrenamos el modelo (con datos de train)
        neigh.fit(X_train, Y_train) 
        # Evaluamos el modelo con datos de train y luego de test
        resultados_train[i,k-1] = neigh.score(X_train, Y_train)
        resultados_test[i,k-1]  = neigh.score(X_test , Y_test )

# Promediamos los resultados de cada repeticion
promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test  = np.mean(resultados_test , axis = 0) 

##################################################################
## Graficamos R2 en funcion de k (para train y test)
##################################################################
plt.plot(valores_k, promedios_train, label = 'Train')
plt.plot(valores_k, promedios_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de knn')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('R^2')
plt.xticks(valores_k)
plt.ylim(0.90,1.00)