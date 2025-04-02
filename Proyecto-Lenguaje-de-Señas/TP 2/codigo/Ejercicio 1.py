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
plt.matshow(diferenciaPromedioDU.values.reshape(28,28), cmap = 'gray')

diferenciaPromedioMN=abs(frankenM-frankenN)
plt.matshow(diferenciaPromedioMN.values.reshape(28,28), cmap = 'gray')

diferenciaPromedioDM=abs(frankenD-frankenM)
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
