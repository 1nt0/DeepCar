import numpy as np
import random 
import matplotlib.image as mpimg
from img_utilidades import aumento_aleatorio, img_preproceso_lenet5, img_preproceso_nvidia



#ENTRENAMIENTO MODELO
def entrena_modelo(modelo, red, x_entrenamiento, y_entrenamiento, x_test, y_test, epochs, training_batch_size, steps_per_epoch, validation_batch_size, validation_steps):
    entrenamiento = modelo.fit_generator(batch_generator(x_entrenamiento, y_entrenamiento, training_batch_size, 'entrenamiento', red), 
                                                   steps_per_epoch=steps_per_epoch, 
                                                   epochs=epochs, 
                                                   validation_data=batch_generator(x_test, y_test, validation_batch_size, '', red),
                                                   validation_steps=validation_steps, 
                                                   verbose=1, 
                                                   shuffle=1)
                                                   
    return entrenamiento



#modo: 'entrenamiento' o ''. red: 'nvidia' o 'lenet5'
def batch_generator(imgs, angulos, batch_size, modo, red):
    
    while True:
        batch_imgs = []
        batch_angulos = []
        
        for i in range(batch_size):
            #Número aleatorio que actuará de índice para el conjunto de imágenes
            num_aleatorio = random.randint(0, len(imgs)-1)
            
            if modo == 'entrenamiento':
                img, angulo = aumento_aleatorio(imgs[num_aleatorio], angulos[num_aleatorio])
                
            else:
                img = mpimg.imread(imgs[num_aleatorio])
                angulo = angulos[num_aleatorio]
            
            if red == 'nvidia':
                img = img_preproceso_nvidia(img)
            elif red == 'lenet5':
                img = img_preproceso_lenet5(img)
            else:
                raise ValueError('No se ha seleccionado ninguna de las dos redes neuronales!')
            
            batch_imgs.append(img) 
            batch_angulos.append(angulo)
            
        yield(np.asarray(batch_imgs), np.asarray(batch_angulos))