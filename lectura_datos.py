import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd
import random
import ntpath

def lectura_csv():
    #Lectura en crudo del fichero CSV generado, añadiendo los labels a cada columna
    columnas = ['centro', 'izquierda', 'derecha', 'angulo', 'acelerador', 'freno', 'velocidad']
    try:
        datos = pd.read_csv('driving_log.csv', names = columnas)
    except:
        print("Ha ocurrido un error leyendo el dataset.")

    return datos
    

#Generación del histograma con la frecuencia de los ángulos
def genera_histograma(datos, num_bins):
    hist, bins = np.histogram(datos['angulo'], num_bins)
    
    return bins


#Selección de las imágenes y los ángulos
def get_imgs_y_angulos(datos):
    imagenes = []
    angulos = []
    for i in range(len(datos)):
        elemento = datos.iloc[i]
        
        angulo = elemento[3]
        
        #Imagen central
        imagen_centro = elemento[0]
        imagenes.append(os.path.join('IMG/', imagen_centro))
        angulos.append(angulo)
        
        #Imagen izquierda
        imagen_izquierda = elemento[1]
        imagenes.append(os.path.join('IMG/', imagen_izquierda))
        angulos.append(angulo + 0.2)
        
        #Imagen derecha
        imagen_derecha = elemento[2]
        imagenes.append(os.path.join('IMG/', imagen_derecha))
        angulos.append(angulo - 0.2)
        
    imagenes = np.asarray(imagenes)
    angulos = np.asarray(angulos)
    
    return imagenes, angulos