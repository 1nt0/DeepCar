from keras.models import load_model

#Leer modelo
def lee_modelo(nombre):
    modelo = load_model(nombre)
    
    return modelo

#Exportar modelo
def exporta_modelo(modelo, nombre):
    modelo.save(nombre)