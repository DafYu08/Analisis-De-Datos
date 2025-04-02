# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:23:02 2024

@Autores: 
"""

#utf-8
#TP2
#Autores: Manu, Sofi, Dafne

import pandas as pd
from inline_sql import sql, sql_val
import matplotlib.pyplot as plt
from matplotlib import ticker   
from matplotlib import rcParams
import seaborn as sns
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# %%
carpeta = 'C:/Users/naemo/Desktop/UBA/Tercer Anio/Labo de datos/TP 2/Archivos/'
df = pd.read_csv(carpeta+'sign_mnist_train.csv')

#%% EJERCICIO1: Falta hacer lo de las imágenes promedio como hicimos con A y L Pero para las letras C, K, M, N y U


df_pixeles = df.drop(columns ='label') #Xs del modelo
pixels = np.array(df_pixeles.iloc[10000].tolist()) #convierte toda la fila en lista
pixels = pixels.reshape((28, 28))


# Mostrar la imagen usando matplotlib
plt.imshow(pixels, cmap='grey')
plt.axis('off')  # Para ocultar los ejes

#Label
#Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).
#%%
pixeles = df.drop(columns ='label') #Xs del modelo
letra = df[['label']] #y del modelo

#Dejo todas las letras que hice, pero solo use D, M, N, U y K
#C promedio
frankenC=pixeles[letra['label'] == 2].mean(axis=0).to_frame()
plt.matshow(frankenC.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#D promedio
frankenD=pixeles[letra['label'] == 3].mean(axis=0).to_frame()
plt.matshow(frankenD.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#E promedio
frankenE=pixeles[letra['label'] == 4].mean(axis=0).to_frame()
plt.matshow(frankenE.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#K promedio
frankenK=pixeles[letra['label'] == 10].mean(axis=0).to_frame()
plt.matshow(frankenK.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#M promedio
frankenM=pixeles[letra['label'] == 12].mean(axis=0).to_frame()
plt.matshow(frankenM.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#N promedio
frankenN=pixeles[letra['label'] == 13].mean(axis=0).to_frame()
plt.matshow(frankenN.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#U promedio
frankenU=pixeles[letra['label'] == 20].mean(axis=0).to_frame()
plt.matshow(frankenU.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#Con los tres graficos que siguen podemos justificar el 1a y 1b
#Decimos que tomamos algunas a ojo, D y U son parecidas, M y N tambien, pero
#D y M no.
#Mientras mas oscuro es el pixel, menor es la diferencia,
#la foto tiende a ser mas grisacea en los bordes y mas clara en el centro,
#entonces, las diferencias estan en el centro
#Los graficos que son mas oscuros son mas dificiles de diferenciar
diferenciaPromedioDU=abs(frankenD-frankenU)
plt.axis('off')
plt.matshow(diferenciaPromedioDU.values.reshape(28,28), cmap = 'gray')

diferenciaPromedioMN=abs(frankenM-frankenN)
plt.axis('off')
plt.matshow(diferenciaPromedioMN.values.reshape(28,28), cmap = 'gray')

diferenciaPromedioDM=abs(frankenD-frankenM)
plt.axis('off')
plt.matshow(diferenciaPromedioDM.values.reshape(28,28), cmap = 'gray')

#Para justificar el 1c
#Calculamos el desvio estandar para ver que tanto varia cada pixel respecto
#del promedio, mientras mas clara la imagen, mas distintas son entre si en
#ese pixel
#Las fotos de la E son mas parecidas (excepto por la parte de abajo) que las
#de la K
#Se ve claramente si miramos los graficos
#Desvio E
stdE=pixeles[letra['label'] == 4].std(axis=0).to_frame()
plt.matshow(stdE.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#Desvio K
stdK=pixeles[letra['label'] == 10].std(axis=0).to_frame()
plt.matshow(stdK.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()
#%%

#Cada pixel x tiene un número entero entre 0 y 255, que representa el tono de gris que tiene el pixel x.

#Organiza por clasificación 
fotos = []
mapeo = {}
for i in range(25):
    mapeo[i] = []
for i in range(df.shape[0]):
    pixels = np.array(df.iloc[i, 1:])  # Excluimos la primera columna
    pixels = pixels.reshape((28, 28))
    fotos.append(pixels)
    mapeo[df.iloc[i][0]].append(i)

#Mira imágenes    
for i in range(1):
    plt.imshow(fotos[mapeo[2][i]], cmap= 'gray')
    plt.axis('off')
    plt.show()

#%%EJERCICIO2

#Objetivo:  Decidir si la imagen corresponde a una seña de la L o a una seña de la A
'''
a. A partir del dataframe original, construir un nuevo dataframe que
contenga sólo al subconjunto de imágenes correspondientes a señas
de las letras L o A.
'''

df_AyL = df[(df['label'] == 0) | (df['label'] == 11)]

'''
b. Sobre este subconjunto de datos, analizar cuántas muestras se tienen
y determinar si está balanceado con respecto a las dos clases a
predecir (la seña es de la letra L o de la letra A).
'''

cuantas_AyL = df_AyL['label'].value_counts()
print(cuantas_AyL)

#Hay 1241 L y 1126 A => están bastante parejas las cantidades (hay un 10% más de L en comparación con las A)
#pero ambos son números buenos para entrenar un modelo

#%%
'''
c. Separar os datos en conjuntos de train y test.
'''
X = df_AyL.drop(columns ='label') #Xs del modelo
Y = df_AyL[['label']] #y del modelo
X_dev, X_eval, Y_dev, Y_eval = train_test_split(X, Y, test_size = 0.1, shuffle=True, stratify= Y, random_state=1) 

#%%
#A y L promedio

#A promedio
frankenA=X_dev[Y_dev['label'] == 0].mean(axis=0).to_frame()
plt.matshow(frankenA.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()

#L promedio
frankenL=X_dev[Y_dev['label'] == 11].mean(axis=0).to_frame()
plt.matshow(frankenL.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
plt.show()
#%%
'''
d. Ajustar un modelo de KNN considerando pocos atributos, por ejemplo
3. Probar con distintos conjuntos de 3 atributos y comparar resultados.
Analizar utilizando otras cantidades de atributos.
Importante: Para evaluar los resultados de cada modelo usar el
conjunto de test generado en el punto anterior.
OBS: Utilicen métricas para problemas de clasificación como por
ejemplo, exactitud.

'''

#%%
''' k = 3 tomando los 3 pixeles que más diferencia tienen'''

#los valos con mayor diferencia son los con mas diferencia en color
diferencia=abs(frankenA-frankenL)
diferencia = diferencia.sort_values(ascending = False, by = 0)
dif=(diferencia.head(3)).T #.T transpone
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test): %.2f" % neigh.score(X_eval_copy, Y_eval))


#%%
''' k = 3 tomando los 5 pixeles que más diferencia tienen'''

#los valos con mayor diferencia son los con mas diferencia en color
diferencia=abs(frankenA-frankenL)
diferencia = diferencia.sort_values(ascending = False, by = 0)
dif=(diferencia.head(5)).T #.T transpone
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test ): %.2f" % neigh.score(X_eval_copy, Y_eval))

#%%
''' k = 3 tomando los 8 pixeles que más diferencia tienen'''

#los valos con mayor diferencia son los con mas diferencia en color
diferencia=abs(frankenA-frankenL)
diferencia = diferencia.sort_values(ascending = False, by = 0)
dif=(diferencia.head(8)).T #.T transpone
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test ): %.2f" % neigh.score(X_eval_copy, Y_eval))

#%%
''' k = 3 tomando los 3 pixeles que menos diferencia tienen'''

#los valos con mayor diferencia son los con mas diferencia en color
diferencia=abs(frankenA-frankenL)
diferencia = diferencia.sort_values(ascending = False, by = 0)
dif=(diferencia.tail(3)).T #.T transpone
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test ): %.2f" % neigh.score(X_eval_copy, Y_eval))


#%%
''' k = 3 tomando los 5 pixeles que menos diferencia tienen'''

#los valos con mayor diferencia son los con mas diferencia en color
diferencia=abs(frankenA-frankenL)
diferencia = diferencia.sort_values(ascending = False, by = 0)
dif=(diferencia.tail(5)).T #.T transpone
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test ): %.2f" % neigh.score(X_eval_copy, Y_eval))

#%%
''' k = 3 tomando los 8 pixeles que menos diferencia tienen'''

#los valos con mayor diferencia son los con mas diferencia en color
diferencia=abs(frankenA-frankenL)
diferencia = diferencia.sort_values(ascending = False, by = 0)
dif=(diferencia.tail(8)).T #.T transpone
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el R2. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test ): %.2f" % neigh.score(X_eval_copy, Y_eval))

#%%
'''
e. Comparar modelos de KNN utilizando distintos atributos y distintos
valores de k (vecinos). Para el análisis de los resultados, tener en
cuenta las medidas de evaluación (por ejemplo, la exactitud) y la
cantidad de atributos.
'''

''' Grid search para cantidad de pixeles y vecinos cercanos desde 1 hasta 10 eligiendo pixeles aleatorios'''

diferencia=abs(frankenA-frankenL)
diferencia = diferencia.sample(frac=1, random_state= 272)

score_pixeles_x_vecinos = np.zeros((10, 10))

i = 10;
while(i > 0):
    dif = (diferencia.head(i)).T
    #random_pixels = np.random.choice(diferencia.columns, i, replace=False)  # Seleccionar i píxeles aleatorios
    X_dev_copy = X_dev[dif.columns]
    X_eval_copy = X_eval[dif.columns]
    
    for k in range(1,11):
            # Declaramos el tipo de modelo
            neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
            # Entrenamos el modelo
            neigh.fit(X_dev_copy, Y_dev.values.ravel())
            score_pixeles_x_vecinos[i-1, k-1] = neigh.score(X_eval_copy, Y_eval)
    i = i -1

#print("Matriz de scores:")
#print(score_pixeles_x_vecinos)

score_maximo = score_pixeles_x_vecinos.max()
#print("Mejor score:", score_maximo)

mejor_k = 0
mejor_cantidad_de_pixeles = 0
for i in range(len(score_pixeles_x_vecinos)):
    for j in range(len(score_pixeles_x_vecinos[0])):
        if score_pixeles_x_vecinos[i][j] == score_maximo:
            mejor_cantidad_de_pixeles = i + 1
            mejor_k = j + 1
            break
    if (mejor_k!= 0): break#Nos quedamos con el primero que encuentra

print("Mejor K: ", mejor_k , "\nMejor cantidad de pixeles: ", mejor_cantidad_de_pixeles)

del X, Y, X_dev, X_eval, Y_dev, Y_eval, X_dev_copy, X_eval_copy

#%%
#Ejercicio 3

#Filtro los datos
df_vocales = df[(df['label'] == 0) | (df['label'] == 4) | (df['label'] == 8) | (df['label'] == 14) | (df['label'] == 20)]

#Separo los datos
X = df_vocales.drop(columns ='label') #Xs del modelo
Y = df_vocales[['label']] #y del modelo
X_dev, X_eval, Y_dev, Y_eval = train_test_split(X, Y, test_size = 0.1, shuffle=True, stratify= Y, random_state=1)

#%%
arbol = DecisionTreeClassifier()
arbol.fit(X_dev, Y_dev)
print(arbol.get_depth())
#La profundidad maxima del arbol es 16
#%%
arbol = DecisionTreeClassifier()
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

hyper_params = {'max_depth' : list(range(1,17)), 'criterion' : ['entropy', 'gini']}
#Hay 32 combinaciones posibles = |max_depth| * |criterion| = 16*2 = 32
#%%
#Probamos la mitad de las combinaciones, tarda alrededor de 2 minutos y medio
clf = RandomizedSearchCV(arbol, hyper_params, cv=cross_val, n_iter = 16)
search = clf.fit(X_dev, Y_dev)

print(search.best_score_)
print(search.best_params_)
#Mejor modelo:
#Score = 0.9603237528906508
#Parametros = {'max_depth': 12, 'criterion': 'entropy'}
#%%
arbol = DecisionTreeClassifier(max_depth = 12, criterion = 'entropy')
arbol.fit(X_dev, Y_dev)
arbol.score(X_eval, Y_eval)
prediccion = arbol.predict(X_eval)

#%%
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
precision = accuracy_score(Y_eval, prediccion)
recall = recall_score(Y_eval, prediccion, average=None)
matriz_de_confusion = confusion_matrix(Y_eval, prediccion)
print("\nMétricas de evaluación:")
print("Precisión: ", precision)
print("Recall: ", recall)
print("Matriz de confusión:")
print(matriz_de_confusion)
#En la diagonal estan los true positives
#En el resto estan los false positives
#La primera fila son los valores que eran A y el modelo predijo la letra de la columna
#Por ejemplo, dijo que 111 eran A y 2 eran I