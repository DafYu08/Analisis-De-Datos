# -*- coding: utf-8 -*-
"""
Laboratorio de Datos v2024
Trabajo Práctico 02
intregrantes:
    --> Manuel Andres Beren
    --> Sofia Roitman
    --> Dafne Sol Yudcovsky
"""

#Importamos bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
#%%
#Importamos los datos
carpeta = ''
df = pd.read_csv(carpeta+'sign_mnist_train.csv')

#%% Ejercicio 1
#Separamos los datos para poder graficar imagenes promedio
pixeles = df.drop(columns ='label') #Xs del modelo
letra = df[['label']] #y del modelo

#%%Hacemos imagenes promedio
#D promedio
frankenD=pixeles[letra['label'] == 3].mean(axis=0).to_frame()

#M promedio
frankenM=pixeles[letra['label'] == 12].mean(axis=0).to_frame()

#N promedio
frankenN=pixeles[letra['label'] == 13].mean(axis=0).to_frame()

#U promedio
frankenU=pixeles[letra['label'] == 20].mean(axis=0).to_frame()

#%%Calculamos las diferencias de imagenes promedio entre letras
#Diferencia D U
diferenciaPromedioDU=abs(frankenD-frankenU)
plt.matshow(diferenciaPromedioDU.values.reshape(28,28), cmap = 'gray')
#quiero los pixels en int para encontarlos en la imagen
graphDU=diferenciaPromedioDU.sort_values(ascending = False, by = 0).head(10).T
pixels=graphDU.columns.to_frame()
pixels=pixels[0].str.split(pat='pixel', expand=True)
pixels=np.array(pixels[1].to_list(),dtype=int)-1
#saco las coords en la imagen 28x28
x=pixels%28
y=pixels//28 
plt.scatter(x,y,color='red',marker='.') #marco los 10 pixels con mayor diferencia
plt.axis('off')

plt.show()
# grafico de barras que muestra la diferencia en tono en ambas imagenes de los 10 pixels representativos
X_axis = np.arange(10) 
  
plt.bar(X_axis - 0.2, frankenD.T[graphDU.columns].T[0], 0.4, label = 'D',color='purple') 
plt.bar(X_axis + 0.2, frankenU.T[graphDU.columns].T[0], 0.4, label = 'U',color='violet') 
  
plt.xticks(X_axis, graphDU.columns,rotation=45) 

plt.title('10 pixeles con mayor diferencia entre las letras D y U',fontsize=12)
plt.ylabel('Tono de pixel',fontsize=11)
plt.xlim(-1,10)
plt.ylim(0,200)
   
plt.legend()
del x,y,pixels
#%%
#Diferencia M N
diferenciaPromedioMN=abs(frankenM-frankenN)
plt.matshow(diferenciaPromedioMN.values.reshape(28,28), cmap = 'gray')

graphMN=diferenciaPromedioMN.sort_values(ascending = False, by = 0).head(10).T
pixels=graphMN.columns.to_frame()
pixels=pixels[0].str.split(pat='pixel', expand=True)
pixels=np.array(pixels[1].to_list(),dtype=int)-1

x=pixels%28
y=pixels//28 

plt.scatter(x,y,color='red',marker='.')
plt.axis('off')

plt.show()

plt.bar(X_axis - 0.2, frankenN.T[graphMN.columns].T[0], 0.4, label = 'M',color='purple') 
plt.bar(X_axis + 0.2, frankenM.T[graphMN.columns].T[0], 0.4, label = 'N',color='violet') 
  
plt.xticks(X_axis, graphDU.columns,rotation=45) 

plt.title('10 pixeles con mayor diferencia entre las letras M y N',fontsize=12)
plt.ylabel('Tono de pixel',fontsize=11)
plt.xlim(-1,10)
plt.ylim(0,200)
   
plt.legend()
del x,y,pixels
#%%
#Diferencia D M
diferenciaPromedioDM=abs(frankenD-frankenM)
plt.matshow(diferenciaPromedioDM.values.reshape(28,28), cmap = 'gray')

graphDM=diferenciaPromedioDM.sort_values(ascending = False, by = 0).head(10).T
pixels=graphDM.columns.to_frame()
pixels=pixels[0].str.split(pat='pixel', expand=True)
pixels=np.array(pixels[1].to_list(),dtype=int)-1

x=pixels%28
y=pixels//28 

plt.scatter(x,y,color='red',marker='.')
plt.axis('off')
plt.show()
 
plt.bar(X_axis - 0.2, frankenD.T[graphDM.columns].T[0], 0.4, label = 'D',color='purple') 
plt.bar(X_axis + 0.2, frankenM.T[graphDM.columns].T[0], 0.4, label = 'M',color='violet') 
  
plt.xticks(X_axis, graphDU.columns,rotation=45) 

plt.title('10 pixeles con mayor diferencia entre las letras D y M',fontsize=12)
plt.ylabel('Tono de pixel',fontsize=11)
plt.xlim(-1,10)
plt.ylim(0,200)
   
plt.legend()
del x,y,pixels,X_axis
#%%Calculamos el desvio estandar para ver que tanto varia cada pixel respecto
#de la imagen promedio

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
#%%Ejercicio 2

#Separamos los datos para tener unicamente las A y las L
#0=A y 11=L
df_AyL = df[(df['label'] == 0) | (df['label'] == 11)]

#Calculamos cuantas A y cuantas L hay
print(df_AyL['label'].value_counts())

#Separamos los datos en train y test
X = df_AyL.drop(columns ='label') #Xs del modelo
Y = df_AyL[['label']] #y del modelo
X_dev, X_eval, Y_dev, Y_eval = train_test_split(X, Y, test_size = 0.5, shuffle=True, stratify= Y, random_state=1) 

#%%Armamos imagenes promedio de A y L

#A promedio
frankenA=X_dev[Y_dev['label'] == 0].mean(axis=0).to_frame()
plt.matshow(frankenA.values.reshape(28, 28), cmap = "gray")
plt.axis('off')

#L promedio
frankenL=X_dev[Y_dev['label'] == 11].mean(axis=0).to_frame()
plt.matshow(frankenL.values.reshape(28, 28), cmap = "gray")
plt.axis('off')
#%% k = 3 tomando los 3 pixeles que más diferencia tienen

#los valores con mayor diferencia son los que mas diferencia tienen en color
diferencia=abs(frankenA-frankenL)
diferencia = diferencia.sort_values(ascending = False, by = 0)#ordeno los pixeles de mayor a menor diferencia

#me quedo con los 3 pixeles de mayor diferencia
dif=diferencia.head(3).T #.T transpone
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el score. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test): %.2f" % neigh.score(X_eval_copy, Y_eval))

#Borramos variables
del dif, X_dev_copy, X_eval_copy

#%% k = 3 tomando los 5 pixeles que más diferencia tienen
dif=(diferencia.head(5)).T 
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el score. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test): %.2f" % neigh.score(X_eval_copy, Y_eval))

#Borramos variables
del dif, X_dev_copy, X_eval_copy,neigh

#%% k = 3 tomando los 8 pixeles que más diferencia tienen
dif=(diferencia.head(8)).T 
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el score. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test): %.2f" % neigh.score(X_eval_copy, Y_eval))

#Borramos variables
del dif, X_dev_copy, X_eval_copy,neigh

#%% k = 3 tomando los 3 pixeles que menos diferencia tienen
dif=(diferencia.tail(3)).T 
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el score. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test): %.2f" % neigh.score(X_eval_copy, Y_eval))

#Borramos variables
del dif, X_dev_copy, X_eval_copy,neigh

#%% k = 3 tomando los 5 pixeles que menos diferencia tienen

dif=(diferencia.tail(5)).T
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el score. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test): %.2f" % neigh.score(X_eval_copy, Y_eval))

#Borramos variables
del dif, X_dev_copy, X_eval_copy,neigh

#%% k = 3 tomando los 8 pixeles que menos diferencia tienen

dif=(diferencia.tail(8)).T #.T transpone
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

k = 3
# Declaramos el tipo de modelo
neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
# Entrenamos el modelo
neigh.fit(X_dev_copy, Y_dev.values.ravel())

# Calculamos el score. Recordar que 1 es en el caso de una prediccion perfecta
print("Score (test): %.2f" % neigh.score(X_eval_copy, Y_eval))

#Borramos variables
del dif, X_dev_copy, X_eval_copy,neigh

#%% Grid search para cantidad de pixeles y vecinos cercanos (k) desde 1 hasta 10 eligiendo pixeles aleatorios
diferencia = diferencia.sample(frac=1, random_state= 1)

score_pixeles_x_vecinos = np.zeros((10, 10))

i = 1;
while(i <= 10):
    dif = (diferencia.head(i)).T
    X_dev_copy = X_dev[dif.columns]
    X_eval_copy = X_eval[dif.columns]
    
    for k in range(1,11):
            # Declaramos el tipo de modelo
            neigh = KNeighborsClassifier(n_neighbors=k) #KNN con k = 3
            # Entrenamos el modelo
            neigh.fit(X_dev_copy, Y_dev.values.ravel())
            score_pixeles_x_vecinos[i-1, k-1] = neigh.score(X_eval_copy, Y_eval)
    i = i +1

print("Matriz de scores:")
print(score_pixeles_x_vecinos)

score_maximo = score_pixeles_x_vecinos.max()
print("Mejor score:", score_maximo)

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

#Borramos variables
del  X_dev_copy, X_eval_copy, dif,neigh
#%%
dif=(diferencia.head(mejor_cantidad_de_pixeles)).T 
X_dev_copy =X_dev[dif.columns]
X_eval_copy =X_eval[dif.columns]

neigh = KNeighborsClassifier(n_neighbors=mejor_k)

neigh.fit(X_dev_copy, Y_dev.values.ravel())


print("Score (test): %.2f" % neigh.score(X_eval_copy, Y_eval))
#da igual que con K=3 y 8 atributos, entonces me quedo con el que tiene menos atributos

del  X_dev_copy, X_eval_copy, dif,neigh,X,Y,X_dev, X_eval, Y_dev, Y_eval
#%% Ejercicio 3

#Filtramos los datos
df_vocales = df[(df['label'] == 0) | (df['label'] == 4) | (df['label'] == 8) | (df['label'] == 14) | (df['label'] == 20)]

#Separamos los datos
X = df_vocales.drop(columns ='label') #Xs del modelo
Y = df_vocales[['label']] #y del modelo
X_dev, X_eval, Y_dev, Y_eval = train_test_split(X, Y, test_size = 0.1, shuffle=True, stratify= Y, random_state=1)

#%% Armamos el arbol y vemos la maxima profundidad
arbol = DecisionTreeClassifier()
arbol.fit(X_dev, Y_dev)
print(arbol.get_depth())

#La profundidad maxima del arbol es 16
#%% Preparamos las variables para hacer Randomized Search
arbol = DecisionTreeClassifier()
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

hyper_params = {'max_depth' : list(range(1,17)), 'criterion' : ['entropy', 'gini'], 'random_state' : [1]}
#Hay 32 combinaciones posibles = |max_depth| * |criterion| * |random_state| = 16*2*1 = 32
#%% Probamos la mitad de las combinaciones, tarda alrededor de 2 minutos y medio
clf = RandomizedSearchCV(arbol, hyper_params, cv=cross_val, n_iter = 16, random_state = 1)
search = clf.fit(X_dev, Y_dev)

print(search.best_score_)
print(search.best_params_)
#Mejor modelo:
#Parametros = {'max_depth': 15, 'criterion': 'entropy'}
#%% Armamos el arbol con los mejores parametros
arbol = DecisionTreeClassifier(max_depth = 15, criterion = 'entropy', random_state = 1)
arbol.fit(X_dev, Y_dev)

#Probamos el mejor modelo obtenido
arbol.score(X_eval, Y_eval)
prediccion = arbol.predict(X_eval)

#ploteo los primeros 3 niveles del arbol de deciociones
plt.figure(figsize=(16, 9))
plot_tree(arbol,max_depth=2,filled=True ,fontsize=12)
#%% Armamos las metricas de clasificacion multiclase
precision = accuracy_score(Y_eval, prediccion)
recall = recall_score(Y_eval, prediccion, average=None)
matriz_de_confusion = confusion_matrix(Y_eval, prediccion)
print("Precisión: ", precision)
print("Recall: ", recall)
print("Matriz de confusión:")
print(matriz_de_confusion)
