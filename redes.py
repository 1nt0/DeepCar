import keras
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

#Modelo LeNet-5
def modelo_lenet5():
    modelo = Sequential()

    modelo.add(Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    modelo.add(Conv2D(16, (5, 5), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    modelo.add(Conv2D(120, (5, 5), activation='relu'))
    
    modelo.add(Flatten())

    modelo.add(Dense(84, activation='relu'))

    modelo.add(Dense(1))
    
    optimizador = Adam(lr=1e-3)
    modelo.compile(loss='mse', optimizer=optimizador)
    
    return modelo

#Modelo NVIDIA
def modelo_nvidia():
    modelo = Sequential()
    
    modelo.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    modelo.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    modelo.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    modelo.add(Conv2D(64, (3, 3), activation='elu'))
    modelo.add(Conv2D(64, (3, 3), activation='elu'))
    
    modelo.add(Flatten())
    
    modelo.add(Dense(100, activation='elu'))
    modelo.add(Dense(50, activation='elu'))
    modelo.add(Dense(10, activation='elu'))
    modelo.add(Dense(1))
    
    optimizador = Adam(lr=1e-3)
    modelo.compile(loss='mse', optimizer=optimizador)
    
    return modelo