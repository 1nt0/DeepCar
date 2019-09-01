import matplotlib.pyplot as plt
import numpy as np

#Dibujado del histograma de un dataframe de Pandas
def dibujar_histograma(datos, num_bins, max_muestras_por_bin):
    plt.hist(datos['angulo'], num_bins)

    plt.xlabel('Ángulo')
    plt.ylabel('Frecuencia')
    plt.title("Distribución de los ángulos") 

    #Línea representando max_muestras_por_bin sobre el histograma
    plt.plot((np.min(datos['angulo']), np.max(datos['angulo'])), (max_muestras_por_bin, max_muestras_por_bin))

    plt.show()

#Representación de los valores de LOSS de un modelo recién entrenado
def representa_loss(entrenamiento):
    plt.plot(entrenamiento.history['loss'])
    plt.plot(entrenamiento.history['val_loss'])

    plt.legend(['Entrenamiento', 'Validación'])
    plt.title('Loss')
    plt.xlabel('Época')

    plt.show()

