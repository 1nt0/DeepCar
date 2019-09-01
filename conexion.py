import base64
import numpy as np
import socketio
import eventlet
from flask import Flask
from keras.models import load_model
from io import BytesIO
from PIL import Image
import cv2
from img_utilidades import img_preproceso_lenet5, img_preproceso_nvidia

sio = socketio.Server()

app = Flask(__name__)

speed_max = 20


@sio.on('connect')
def connect(sid, environ):
    print('Conectado. sid:', sid)
    send_control(0, 0)


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preproceso_nvidia(image) #Cambiar en función del modelo usado: img_preproceso_nvidia o img_preproceso_lenet5
    image = np.array([image])
    steering_angle = float(model.predict(image))

    throttle = 1.0 - speed/speed_max
    print('Ángulo: {} Aceleración: {} Velocidad: {}'.format(steering_angle, throttle, speed))

    send_control(steering_angle, throttle)
 
 
def send_control(steering_angle, throttle):
    sio.emit('steer', 
        data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 

 
if __name__ == '__main__':
    model = load_model('modelo.h5')

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    