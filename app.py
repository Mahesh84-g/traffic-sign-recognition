from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'model.h5'

model = load_model(MODEL_PATH)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255  # Normalize the image
    return img

def getClassName(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h', 
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 
        'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons', 
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 
        'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 
        'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[classNo]

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Load image with target size
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))  # Resize image to model's input size (32x32)
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)  # Reshape image to fit the model input shape
    predictions = model.predict(img)  # Get predictions from the model
    classIndex = np.argmax(predictions, axis=-1)  # Get the index of the highest predicted class
    return getClassName(classIndex[0])  # Return the class name corresponding to the predicted class

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']  # Get the uploaded file
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)  # Save the file in the 'uploads' folder
        preds = model_predict(file_path, model)  # Predict the class of the uploaded image
        return preds  # Return the result as response
    return None

if __name__ == '__main__':
    app.run(port=5001, debug=True)
