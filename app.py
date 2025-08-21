from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model once (at the start)
model = tf.keras.models.load_model('sugarcane_leaf_model.h5')

# Class names (order should match your training)
class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# Image size (same as training)
IMG_SIZE = 150

app = Flask(__name__)

def preprocess_image(file):
    """Preprocess the image for prediction."""
    # Read image from the uploaded file
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Resize image to the target size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize the image to the range [0, 1]
    img = img.astype('float32') / 255.0
    
    # Add an extra dimension to match model input (batch size, height, width, channels)
    img = np.expand_dims(img, axis=0)  # shape (1, 150, 150, 3)
    return img

@app.route('/', methods=['GET'])
def home():
    """Render the home page with the form to upload images."""
    return render_template('index.html')  # Ensure index.html exists in your templates folder

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the image prediction request."""
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    # Process the image
    img = preprocess_image(file)
    
    # Make the prediction using the loaded model
    preds = model.predict(img)
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    # Render the result in the HTML page
    return render_template('index.html', prediction=pred_class, confidence=f"{confidence:.2f}")

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
