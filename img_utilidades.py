import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.image as mpimg

#----PREPROCESO DE IMÁGENES DE CADA RED NEURONAL-----
def img_preproceso_lenet5(img):
    img = recorte(img)
    img = RGBaGRAY(img)
    img = redimensionar_imagen(img, 32, 32)
    img = img[:, :, np.newaxis]
    img = normalizar_imagen(img)

    return img

def img_preproceso_nvidia(img):
    img = recorte(img)
    img = RGBaYUV(img)
    img = redimensionar_imagen(img, 200, 66)
    img = normalizar_imagen(img)

    return img


#Implementación de cada una de las funciones usadas anteriormente
def recorte(img):
    img = img[70:135,:,:] 

    return img

def RGBaYUV(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) 

    return img

def RGBaGRAY(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img

def redimensionar_imagen(img, x, y):
    img = cv2.resize(img, (x, y))

    return img

def normalizar_imagen(img):
    img = img/255
    
    return img



#----------------DATA AUGMENTATION---------------------
def aumento_aleatorio(img, angulo):
    img = mpimg.imread(img)
    
    if np.random.random_sample() < 0.5:
        img = iluminacion(img)
        
    if np.random.random_sample() < 0.5:
        img = zooming(img)
        
    if np.random.random_sample() < 0.5:
        img = desplazamiento(img)
        
    if np.random.random_sample() < 0.5:
        img, angulo = volteo(img, angulo)
    
    return img, angulo


#Implementación de cada una de las funciones usadas anteriormente
def iluminacion(img):
    iluminacion = iaa.Multiply((0.2, 1.3))
    img = iluminacion.augment_image(img)
    
    return img

def zooming(img):
    zooming = iaa.Affine(scale={"x": (1.0, 1.4), "y": (1.0, 1.4)})
    img = zooming.augment_image(img)
    
    return img

def desplazamiento(img):
    desplazamiento = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
    img = desplazamiento.augment_image(img)
    
    return img

def volteo(img, angulo):
    volteo = iaa.Fliplr(1.0)
    img = volteo.augment_image(img)
    angulo = -angulo
    
    return img, angulo