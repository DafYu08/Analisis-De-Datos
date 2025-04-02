# -*- coding: utf-8 -*-
'''
Laboratorio de Datos v2024
Trabajo Práctico 01
intregrantes:
    --> Manuel Andres Beren
    --> Sofia Roitman
    --> Dafne Sol Yudcovsky
'''
import pandas as pd
from inline_sql import sql, sql_val
import matplotlib.pyplot as plt
from matplotlib import ticker   
from matplotlib import rcParams
import seaborn as sns

#insertar carpeta que quieran vvv
carpeta = "C:/Users/naemo/Desktop/UBA/Tercer Anio/Labo de datos/TP 1/Archivos/"

pbi_anual_pais_original = pd.read_csv(carpeta+"API_NY.GDP.PCAP.CD_DS2_en_csv_v2_73.csv")
metadata_country_original = pd.read_csv(carpeta+"Metadata_Country_API_NY.GDP.PCAP.CD_DS2_en_csv_v2_73.csv") #no se usa
metadata_indicator_original = pd.read_csv(carpeta+"Metadata_Indicator_API_NY.GDP.PCAP.CD_DS2_en_csv_v2_73.csv") #no se usa
lista_secciones_original = pd.read_csv(carpeta+"lista-secciones.csv")
lista_sedes_datos_original = pd.read_csv(carpeta+"lista-sedes-datos.csv")
lista_sedes_original = pd.read_csv(carpeta+"lista-sedes.csv")

#%%GQM


#lista_sedes_datos_original: esta abajo en el codigo porque  primero necesitamos separar 
#en unidades atomicas la columna redes_sociales

#lista_secciones_original:secciones de una misma sede con el mismo nombre
problema2=sql^"""
               SELECT DISTINCT s1.sede_id,s1.sede_desc_castellano,s1.nombre_titular,s1.apellido_titular
               FROM lista_secciones_original AS s1
               INNER JOIN lista_secciones_original AS s2
               ON s1.sede_id=s2.sede_id AND s1.sede_desc_castellano=s2.sede_desc_castellano AND
               (s1.nombre_titular!=s2.nombre_titular OR s1.apellido_titular!=s2.apellido_titular)
               ORDER BY s1.sede_desc_castellano,s1.sede_id;
               """
print(problema2)

#pbi_anual_pais_original:problema tiene regiones que los trata como pais
problema3=sql^"""
               SELECT "Country Name"
               FROM pbi_anual_pais_original AS p
               INTERSECT
               SELECT region
               FROM metadata_country_original AS m;
              """
print(problema3)

#%%Limpieza de datos

#Para este TP solo nos vamos a concentrar en los paises con sedes argentinas y sus respectivos PBIs del año 2022

pais = sql^"""
                      SELECT DISTINCT pais_iso_3 AS id_pais, UPPER(Pais_castellano) AS nombre_pais,region_geografica,"2022" AS PBI
                      FROM pbi_anual_pais_original
                      INNER JOIN lista_sedes_datos_original
                      ON pais_iso_3="Country Code"  AND "2022"  IS NOT NULL;
                     """
#pais.to_csv(r"D:\Python\Labo de datos\TP 1\pais.csv", index=False)
#Aca descartamos las sedes inactivas porque no tiene sentido contarlas
sedes = sql^"""
                   SELECT DISTINCT sede_id, pais_iso_3 AS id_pais
                   FROM lista_sedes_datos_original
                   WHERE sede_id IS NOT NULL AND estado = 'Activo';
                  """
#sedes.to_csv(r"D:\Python\Labo de datos\TP 1\sedes.csv", index=False)
#Si una sede tiene dos secciones con un mismo nombre, entonces
#hacen referencia a lo mismo y solo se cambia o agrega un titular
#como por ejemplo,  sede_id=ERUNI, sede_desc_castellano = Administración
secciones = sql^"""
                       SELECT DISTINCT sede_id, LOWER(sede_desc_castellano) AS descripcion
                       FROM lista_secciones_original
                       WHERE sede_id IS NOT NULL AND sede_desc_castellano IS NOT NULL;                 
                      """
#secciones.to_csv(r"D:\Python\Labo de datos\TP 1\secciones.csv", index=False)
#A continuacion todo el proceso de separar url en unidades atomicas y rescatar el tipo de red social                 
redes = sql^"""
             SELECT DISTINCT sede_id ,LOWER(redes_sociales) AS URL
             FROM lista_sedes_datos_original
             WHERE redes_sociales IS NOT NULL;
            """

#Con pandas me separa los URL en columnas
redes[['r1','r2','r3','r4','r5','r6','r7']]= redes['URL'].str.split(pat='  //  ', expand=True)
redes=redes[['sede_id','r1','r2','r3','r4','r5','r6']] #r7 es todo null y tiene una unica celda vacia

#Uno las columnas de url que me dio pandas y me quedo con las no vacias
redes = sql^"""
                     SELECT DISTINCT sede_id ,r1 AS URL
                     FROM redes
                     WHERE r1 like '%_%'
                     UNION
                     SELECT DISTINCT sede_id ,r2 AS URL
                     FROM redes
                     WHERE r2 like '%_%'
                     UNION
                     SELECT DISTINCT  sede_id ,r3 AS URL
                     FROM redes
                     WHERE r3 like '%_%'
                     UNION
                     SELECT DISTINCT  sede_id ,r4 AS URL
                     FROM redes
                     WHERE r4 like '%_%'
                     UNION
                     SELECT DISTINCT  sede_id ,r5 AS URL
                     FROM redes
                     WHERE r5 like '%_%'
                     UNION
                     SELECT DISTINCT  sede_id ,r6 AS URL
                     FROM redes
                     WHERE r6 like '%_%';
                     
                  """
#GQM problema 1
#Redes que no son URL
problema1 = sql^"""
                       SELECT DISTINCT *
                       FROM redes
                       WHERE url NOT LIKE '%.com%' ;            
                      """
print(problema1)

#selecciono los url y rescato el tipo de red social quedandome con la parte de adelante del .com
tipo=sql^"""
            SELECT DISTINCT URL
            FROM redes
            WHERE  url LIKE '%.com%'  AND url NOT LIKE '%mail%'          
            """

tipo[['red','trash']]= tipo['URL'].str.split(pat='.com',n=1, expand=True)
tipo=tipo[['red']]
tipo= tipo['red'].str.split(pat='\.|//',n=2, expand=True)

#Me quedo con el ultimo no null
#El primero son los que tienen http y www, el segundo alguno de los 2 y el tercero ninguno

tipo=sql^"""
                SELECT DISTINCT "2" AS red_social 
                FROM tipo
                WHERE  "2" IS NOT NULL 
                UNION
                SELECT DISTINCT "1" AS red_social 
                FROM tipo
                WHERE  "2" IS NULL AND "1" IS NOT NULL
                UNION
                SELECT DISTINCT "0" AS red_social 
                FROM tipo
                WHERE  "1" IS NULL AND "0" IS NOT NULL;
                """
print(tipo)

#Por tipos_red sabemos que solo hay 6 tipos de redes sociales: facebook,twitter,instagram,youtube,linkedin y flickr

redes = sql^"""
                    SELECT DISTINCT sede_id,'Facebook' AS Red_Social, URL
                    FROM redes
                    WHERE url LIKE '%facebook%'
                    UNION
                    SELECT DISTINCT sede_id,'Twitter' AS Red_Social, URL
                    FROM redes
                    WHERE url LIKE '%twitter%'
                    UNION
                    SELECT DISTINCT sede_id,'Instagram' AS Red_Social, URL
                    FROM redes
                    WHERE url LIKE '%instagram%'
                    UNION
                    SELECT DISTINCT sede_id,'Youtube' AS Red_Social, URL
                    FROM redes
                    WHERE url LIKE '%youtube%'
                    UNION
                    SELECT DISTINCT sede_id,'Linkedin' AS Red_Social, URL
                    FROM redes
                    WHERE url LIKE '%linkedin%'
                    UNION
                    SELECT DISTINCT sede_id,'Flickr' AS Red_Social, URL
                    FROM redes
                    WHERE url LIKE '%flickr%'
                    
                    ORDER BY sede_id,red_social;
                   """

#confirmo q son clave
test = sql^"""
                    select distinct sede_id,red_social
                    from redes
                    
                   """

#redes.to_csv(r"D:\Python\Labo de datos\TP 1\redes.csv", index=False)

#%% Data frames (ej. h)

#---------------------------------------------Ejercicio 1----------------------------------------------
cant_sedes_pais = sql^"""
                       SELECT DISTINCT p.id_pais,p.nombre_pais, count(s.id_pais) AS cant_sedes
                       FROM pais AS p
                       INNER JOIN sedes AS s
                       ON s.id_pais = p.id_pais
                       GROUP BY  p.nombre_pais, p.id_pais;
                      """

cant_secciones_pais = sql^"""
                           SELECT DISTINCT sed.id_pais,count(sec.sede_id) AS cant_secciones
                           FROM  sedes AS sed
                           LEFT OUTER JOIN secciones AS sec
                           ON sec.sede_id = sed.sede_id
                           GROUP BY sed.id_pais;
                          """


cantidades=sql^"""
                SELECT DISTINCT sed.*,cant_secciones
                FROM cant_sedes_pais AS sed
                INNER  JOIN cant_secciones_pais AS sec
                ON sed.id_pais = sec.id_pais;
                """


info_paises = sql^"""
                   SELECT DISTINCT c.nombre_pais AS Pais,
                                   cant_sedes AS Sedes,
                                   ROUND(cant_secciones/cant_sedes,1) AS 'Secciones promedio',
                                   PBI AS 'PBI per cápita 2022 (U$S)'
                   FROM pais AS p
                   INNER JOIN cantidades AS c
                   ON c.id_pais = p.id_pais
                   ORDER BY sedes DESC, c.nombre_pais ASC;
                  """
#info_paises.to_excel(r'D:\Python\Labo de datos\TP 1\ej1.xlsx', index=False)


#---------------------------------------------Ejercicio 2----------------------------------------------
pais_region_pbi = sql^"""
                           SELECT DISTINCT sed.nombre_pais, region_geografica, PBI
                           FROM cant_sedes_pais AS sed
                           INNER JOIN pais AS p
                           ON p.id_pais = sed.id_pais ;
                          """

info_regiones = sql^"""
                    SELECT DISTINCT region_geografica AS 'Región geográfica',
                                    count(region_geografica) AS 'Países Con Sedes Argentinas',
                                    AVG(PBI) AS 'Promedio PBI per cápita 2022 (U$S)'
                    FROM pais_region_pbi
                    GROUP BY region_geografica
                    ORDER BY 'Promedio PBI per cápita 2022 (U$S)' DESC;
                   """
#info_regiones.to_excel(r'D:\Python\Labo de datos\TP 1\ej2.xlsx', index=False)
#---------------------------------------------Ejercicio 4----------------------------------------------
pais_sede= sql^"""
                    SELECT DISTINCT nombre_pais,sede_id
                    FROM pais
                    INNER JOIN sedes
                    ON pais.id_pais=sedes.id_pais;
                   """

info_sedes_redes = sql^"""
                    SELECT DISTINCT nombre_pais AS País, ps.sede_id AS Sede, red_social AS 'Red Social',URL
                    FROM pais_sede AS ps
                    INNER JOIN redes
                    ON ps.sede_id=redes.sede_id
                    ORDER BY nombre_pais,sede,red_social,url;
                   """
#info_sedes_redes.to_excel(r'D:\Python\Labo de datos\TP 1\ej4.xlsx', index=False)
#---------------------------------------------Ejercicio 3----------------------------------------------
cantidad_redes_pais = sql^"""
                    SELECT DISTINCT País,COUNT (*) AS 'Cantidad distinta de redes sociales por pais'
                    FROM (
                        SELECT DISTINCT País,"Red Social"
                        FROM info_sedes_redes
                        )
                    GROUP BY País
                    ORDER BY "Cantidad distinta de redes sociales por pais" DESC,País;
                   """
#cantidad_redes_pais.to_excel(r'D:\Python\Labo de datos\TP 1\ej3.xlsx', index=False)

#%% Graficos (ej. i)
#---------------------------------------------Grafico 1----------------------------------------------
info_regiones = info_regiones.sort_values('Países Con Sedes Argentinas')
sns.set_style("whitegrid")  # Fondo blanco con líneas de rejilla
sns.set_context("notebook")  # Ajustar el tamaño de la fuent

fig, ax = plt.subplots()    

plt.rcParams['font.family'] = 'sans-serif'                
ax.bar(data=info_regiones, 
       x='Región geográfica', 
       height='Países Con Sedes Argentinas',
       edgecolor='k',
       color = "#52BCA3"
      )
ax.set_title('Cantidad de Países Con Sedes Argentinas por Región', fontsize=14, fontweight='bold')                   
#ax.set_xlabel('Región', fontsize=18, fontweight='bold')   #este lo sacaria      
ax.set_ylabel('Cantidad de países', fontsize=14)  
plt.xticks(fontsize=12,rotation=25, horizontalalignment='right')
ax.set_ylim(0, 22)
                                  
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}")); 
plt.gca().set_facecolor('#FFF8DC')  
plt.gcf().set_facecolor('#F5F5DC')


plt.show()
#fig.savefig('histograma.png', dpi=300)

#---------------------------------------------Grafico 2----------------------------------------------

order = pais_region_pbi.groupby('region_geografica')['PBI'].median().sort_values().index

sns.set_style("whitegrid")  # Fondo blanco con líneas de rejilla
sns.set_context("notebook")  # Ajustar el tamaño de la fuente

plt.figure(figsize=(16, 9))


ax = sns.boxplot(y="region_geografica", 
                x="PBI",  
                 data=pais_region_pbi, 
                 order = order,
                 showmeans = True,
                 meanprops=dict(marker = '*',markerfacecolor='white',markeredgecolor='k',markersize=12),
                 palette = ["#52BCA3"],
                 linewidth=1.5,  # Grosor de los bordes
                 boxprops=dict(edgecolor='black'),   # Color del borde de los boxplots
                 whiskerprops=dict(color='black'),   # Color de las líneas que representan los bigotes
                 medianprops=dict(color='black'),  # Color de la línea que representa la mediana
                 capprops=dict(color='black'))      # Color de las líneas que representan los extremos de los bigotes)

# Ajustar el tamaño de las etiquetas de las regiones
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_title('PBI per cápita por Región', fontsize=20, fontweight='bold')
ax.set_ylabel('Región', fontsize=18, fontweight='bold')
ax.set_xlabel('PBI per cápita 2022 (U$S)', fontsize=18, fontweight='bold')
ax.set_xlim(0,110000) 
plt.gca().set_facecolor('#FFF8DC')  # Blanco crema
plt.gcf().set_facecolor('#F5F5DC')  # Blanco crema más oscuro
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"));

# Mostrar el boxplot
plt.show()


#---------------------------------------------Grafico 3----------------------------------------------

fig,ax= plt.subplots()
plt.rcParams['font.family'] = 'sans-serif'
ax.scatter(data=info_paises,x='Sedes',y='PBI per cápita 2022 (U$S)',marker='o',edgecolor='k',color="#52BCA3")
ax.set_title('Relación entre el PBI per cápita y cantidad de sedes Argentinas por países', fontsize=14, fontweight='bold')
ax.set_xlabel('Cantidad de sedes',fontsize=14)
ax.set_ylabel('PBI per cápita 2022 (U$S)',fontsize=14)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"));
plt.gca().set_facecolor('#FFF8DC')  # Blanco crema
plt.gcf().set_facecolor('#F5F5DC') 
plt.show()
#fig.savefig('scatter.png', dpi=300)