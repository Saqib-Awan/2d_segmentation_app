from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = "C:/Users/Saqib 1/project_folder/files4/model.h5"
model = load_model(model_path, compile=False)

# Define utility functions
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_mask(mask):
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (512, 512))
    return mask * 255

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    if request.method == 'POST':
        # Get uploaded image file
        image_file = request.files['image']

        if image_file:
            # Save the uploaded image
            image_path = os.path.join("uploads", image_file.filename)
            image_file.save(image_path)

            # Preprocess the image
            image = preprocess_image(image_path)

            # Perform segmentation
            segmented_mask = model.predict(image)[0]

            # Postprocess the mask
            segmented_mask = postprocess_mask(segmented_mask)

            # Save the segmented mask
            segmented_mask_path = os.path.join("static", "segmented_masks", image_file.filename)
            cv2.imwrite(segmented_mask_path, segmented_mask)

            return render_template('result.html', image_file=image_file.filename)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
