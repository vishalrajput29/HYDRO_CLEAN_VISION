from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
from PIL import Image
import io
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained YOLO model (update the path to your model)
model = YOLO('best (1).pt')

# Route to serve the home page
@app.route('/')
def home():
    return render_template('index1.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return {"error": "No image file found in request"}, 400

    uploaded_file = request.files['image']
    image = Image.open(uploaded_file)

    # Perform inference
    results = model(image)

    # Get the first result (YOLO returns a list of results)
    result = results[0]

    # Convert the result to a numpy array (this is the bounding-boxed image)
    result_image = result.plot()  # Get the image with bounding boxes

    # Convert numpy array to a PIL Image
    result_image_pil = Image.fromarray(result_image)

    # Save the result to a BytesIO object
    output = io.BytesIO()
    result_image_pil.save(output, format="JPEG")
    output.seek(0)

    # Return the processed image
    return send_file(output, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
