from sklearn.utils import shuffle
import ntpath
import os
import numpy as np
from sklearn.model_selection import train_test_split

#Función para obtener la 'hoja' del árbol perteneciente a la ruta absoluta de la imagen, que corresponde con su nombre
def get_tail(path):
    head, tail = ntpath.split(path) #Separando por "/", tail es el último elemento, y head todo lo anterior

    return tail


def renombra_imagenes(datos):
    #Modificación de las 3 primeras columnas, correspondiente a la ruta de las imágenes
    datos['centro'] = datos['centro'].apply(get_tail)
    datos['izquierda'] = datos['izquierda'].apply(get_tail)
    datos['derecha'] = datos['derecha'].apply(get_tail)

    return datos



#--------------BALANCEO DEL DATASET-------------
def balancea_dataset(datos, num_bins, bins, max_muestras_por_bin):
    datos_preprocesados = datos.copy() #Copia del dataset original
    elementos_a_eliminar = []

    #Recorrido de cada bin
    for i in range(num_bins):
        lista = []
        #Recorrido de cada elemento del dataset: si el ángulo pertenece al bin que estamos recorriendo, lo 
        #metemos en la lista
        for j in range(len(datos['angulo'])):
            if datos['angulo'][j] >= bins[i] and datos['angulo'][j] <= bins[i+1]:
                lista.append(j)
        
    #Mezclamos los elementos que pertenecen al bin
    lista = shuffle(lista)
    #Nos quedamos con las "sobras" basado en el umbral que hayamos decidido (max_muestras_por_bin), para posteriormente
    #añadirlo a elementos_a_eliminar y utilizarla como lista con los elementos a eliminar en el dataset
    lista = lista[max_muestras_por_bin:]
    elementos_a_eliminar.extend(lista)

    #Eliminamos dichos datos del dataset
    datos_preprocesados.drop(datos_preprocesados.index[elementos_a_eliminar], inplace=True)

    return datos_preprocesados


# División del dataset en conjuntos de entrenamiento y prueba
def divide_dataset(imagenes, angulos, train_size):
    x_entrenamiento, x_test, y_entrenamiento, y_test = train_test_split(imagenes, angulos, train_size=train_size)

    return x_entrenamiento, x_test, y_entrenamiento, y_test