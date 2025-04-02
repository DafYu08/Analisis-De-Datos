#!/usr/bin/env python
# coding: utf-8

# # Evaluación de Modelos
# 
# **Objetivo:** dada los datos de una canción (una fila en nuestro dataset) poder predecir si esta en Folklore o Evermore o es de otro álbum.
# 
# **Datos:** dataset con distintas variables de las canciones de Taylor Swift.

# In[1]:

import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# #### Cargamos el dataset -- la función load_dataset limpia un poco los datos

# In[2]:


df_taylor = utils.load_dataset_taylor("taylor_album_songs.csv")
df_taylor.head()


# ### Separemos los labels y eliminamos el nombre de la canción


#%%

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Dividimos en test(30%) y train(70%)
data_train = df_taylor.drop(columns = ['track_name', 'is_folklore_or_evermore'])
X = data_train[['acousticness']] # doble corchete es importante
Y = df_taylor['is_folklore_or_evermore']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True) 

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_train, Y_train)
#####################################
## Evaluacion del modelo contra TRAIN
#####################################
# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test ): %.2f" % neigh.score(X_test, Y_test))

#%%

#################################################
## Generacion archivos TEST / TRAIN
#################################################
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Dividimos en test(30%) y train(70%)
data_train = df_taylor.drop(columns = ['track_name', 'is_folklore_or_evermore'])
X = data_train[['acousticness']] # doble corchete es importante
Y = df_taylor['is_folklore_or_evermore']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True) 


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
plt.title('Prediccion Folklore/Evermore, otro')
plt.xlabel('Acousticness')
plt.ylabel('Evermore/Folklore u otras')
plt.xticks(valores_k)
plt.ylim(0,1.00)


# In[4]:

# Complete aqui con su clasificador de preferencia!

#################################################
## Generacion archivos TEST / TRAIN
#################################################
# Dividimos en test(30%) y train(70%)
X = df_taylor.drop(columns = ['track_name', 'is_folklore_or_evermore']) # doble corchete es importante
Y = df_taylor['is_folklore_or_evermore']
X_dev, X_eval, Y_dev, Y_eval = train_test_split(X, Y, test_size = 0.3, shuffle=True) 


from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

arbol = DecisionTreeClassifier(max_depth=3)

hyper_params = {'criterion': ["gini", "entropy"],
                'max_depth' : [2,3,4,5]}

cross_val_score(arbol, X_dev, Y_dev, cv=5)

#kf = KFold(n_splits=2, shuffle=True)
#list(kf.split(X, Y))

from sklearn.model_selection import RandomizedSearchCV



clf = RandomizedSearchCV(arbol, hyper_params, random_state=0, n_iter=3)
clf.fit(X_dev, Y_dev)

clf.best_params_

clf.best_score_ #lo mide con los mejores hiperparámetros

clf.score(X_eval, Y_eval)






























