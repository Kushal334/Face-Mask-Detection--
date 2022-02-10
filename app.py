from __future__ import division, print_function
import os
import numpy as np

# Keras
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'face_mask_detection.h5'

# Loading the trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150, 3)) # Loads the image at size (150, 150, 3)
        
    img_batch = np.expand_dims(img, axis=0)  # Expanding the dimensions of the array
    
    img_batch = img_batch / 255.  # Scaling down the value to bring it in range[0, 1]
    
    pred = model.predict(img_batch)  # Will get a value in range[0, 1]
    
    if pred < 0.5 :    # Giving names to the Predictions
        return 'Wearing a Mask'
    else:
        return 'Not wearing a Mask'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction

        preds = model_predict(file_path, model)
        
        result = str(preds)
        
        return result
    

if __name__ == '__main__':
    app.run(debug=True)
