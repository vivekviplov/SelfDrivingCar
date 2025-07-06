import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# Create SocketIO server and Flask app
sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10  # Max speed for throttle control logic

# Image preprocessing for input to the model
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop sky and car hood
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))  # NVIDIA model input size
    img = img / 255.0  # Normalize
    return img

# Telemetry event from simulator (main loop)
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data['speed'])
        # Convert base64 image to NumPy array
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])
        
        # Predict steering angle using the model
        steering_angle = float(model.predict(image))
        
        # Throttle control based on speed
        throttle = 1.0 - speed / speed_limit
        print(f'{steering_angle:.4f}, {throttle:.4f}, Speed: {speed:.2f}')
        
        send_control(steering_angle, throttle)

# When a client connects
@sio.on('connect')
def connect(sid, environ):
    print('Client connected')
    send_control(0, 0)  # Send neutral command initially

# Function to send steering and throttle commands
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Run the app
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
