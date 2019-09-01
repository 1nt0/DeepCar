from lectura_datos import lectura_csv, generar_histograma, get_imgs_y_angulos
from preprocesamiento_datos import renombra_imagenes, balancea_dataset, divide_dataset
from redes import modelo_lenet5, modelo_nvidia
from entrenamiento import entrena_modelo
from graficas import representa_loss
from utilidades import exporta_modelo
import img_utilidades
import sys

#----------------------------------------------------CONFIGURACIÓN----------------------------------------------------------------
#HISTOGRAMA
num_bins = 23               #Número de "barras" (bins) en las que dividir el histograma
max_muestras_por_bin = 300  #Usado para eliminar los elementos del histograma cuya frecuencia sobrepasen este valor

#CONJUNTOS ENTRENAMIENTO/TEST
train_size = 0.8            #Porcentaje del tamaño que va a tener el subconjunto de entrenamiento al dividir el dataset

#ENTRENAMIENTO
red = 'nvidia'              #Red con la que entrenar el dataset. Los valores son 'nvidia' o 'lenet5'
epochs = 10                 #Número de épocas de entrenamiento
training_batch_size = 100   #Número de imágenes de entrenamiento generadas por cada "step"
steps_per_epoch = 300       #Número de steps por cada época
validation_batch_size = 100 #Número de imágenes de validación generadas por cada "step"
validation_steps = 200      #Número de steps de validación

nombre_modelo_exportado = 'modelo.h5'   #Debe tener la extensión .h5 y se guardará en la misma carpeta que este fichero
#---------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    #1- Lectura del fichero CSV---------------------------------------------------------------------------------------------------
    print("-----------")
    print("1- Lectura del fichero driving_log.csv...... (Debe estar en el mismo directorio, junto con la carpeta IMG)")
    try:
        datos = lectura_csv()
        print("Hay un total de {} elementos en el dataset.".format(len(datos)))
    except:
        print("Error leyendo el fichero CSV")
        sys.exit()
    if datos.empty:
        "El dataset está vacío!"
    
    print("Lectura efectuada con éxito.")

    #2- Renombración de la ruta de las imágenes-----------------------------------------------------------------------------------
    print("-----------")
    print("2- Renombrado de la ruta de las imágenes....")
    try:
        datos = renombra_imagenes(datos)
    except:
        print("Error renombrando la ruta de las imágenes")
        sys.exit()
    print("Renombrado efectuado con éxito.")

    #3- Generación del histograma-------------------------------------------------------------------------------------------------
    print("-----------")
    print("3- Generación del histograma....")
    try:
        bins = generar_histograma(datos, num_bins)
    except:
        print("Error generando el histograma.")
        sys.exit()
    print("Histograma generado con éxito.")

    #4- Balanceo del dataset------------------------------------------------------------------------------------------------------
    print("-----------")
    print("4- Balanceo del dataset....")
    try:
        datos_preprocesados = balancea_dataset(datos, num_bins, bins, max_muestras_por_bin)
    except:
        print("Error balanceando el dataset.")
        sys.exit()
    print("Balanceo efectuado con éxito.")

    #5- División del dataset en conjuntos de entrenamiento y prueba---------------------------------------------------------------
    print("-----------")
    print("5- División del dataset en conjuntos de entrenamiento y prueba....")
    try:
        #Obtención de las imágenes y los ángulos
        imagenes, angulos = get_imgs_y_angulos(datos_preprocesados)
        x_entrenamiento, x_test, y_entrenamiento, y_test = divide_dataset(imagenes, angulos, train_size)
        print("El conjunto de entrenamiento tiene {} elementos. ".format(len(x_entrenamiento)+len(y_entrenamiento)))
        print("El conjunto de testeo tiene {} elementos. ".format(len(x_test)+len(y_test)))
    except:
        print("Error al dividir el dataset.")
        sys.exit()
    print("División del dataset efectuada con éxito.")

    #6- Creación del modelo neuronal----------------------------------------------------------------------------------------------
    print("-----------")
    print("6- Creación del modelo neuronal....")
    try:
        if red == 'nvidia':
            modelo = modelo_nvidia()
        elif red == 'lenet5':
            modelo = modelo_lenet5()
        else:
            print("No se ha definido correctamente la variable red. Los posibles valores son 'nvidia' y 'lenet5'")
            sys.exit()
        print("Resumen del modelo neuronal:")
        print(modelo.summary())
    except:
        print("Error al crear el modelo neuronal.")
        sys.exit()
    print("\n División del dataset efectuada con éxito.")

    #7- Entrenamiento del modelo neuronal-----------------------------------------------------------------------------------------
    print("-----------")
    print("7- Entrenamiento del modelo neuronal....")
    try:
        entrenamiento = entrena_modelo(modelo, red, x_entrenamiento, y_entrenamiento, x_test, y_test, epochs, training_batch_size, steps_per_epoch, 
                                                                                                                    validation_batch_size, validation_steps)
    except:
        print("Error al entrenar el modelo.")
        sys.exit()
    print("Entrenamiento del modelo efectuado con éxito.")

    #8- Gráfica con la evolución de loss------------------------------------------------------------------------------------------
    print("-----------")
    print("8- Loss durante el entrenamiento en los conjuntos de entrenamiento y validación....")
    try:
        print("Cierre el gráfico para continuar...")
        representa_loss(entrenamiento)
    except:
        print("Error al generar la gráfica con los valores de loss.")
        sys.exit()
    print("Gráfica generada con éxito.")

    #9- Exportación del modelo entrenado------------------------------------------------------------------------------------------
    print("-----------")
    print("9- Exportación del modelo entrenado....")
    try:
        exporta_modelo(modelo, nombre_modelo_exportado)
    except:
        print("Error al exportar el modelo.")
        sys.exit()
    print("Exportación del modelo con nombre {} efectuada con éxito.".format(nombre_modelo_exportado))
