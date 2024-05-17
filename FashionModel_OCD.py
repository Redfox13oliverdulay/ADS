# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WwUEhg8KjCVWKPXdplWjccbBReRHc71M
"""

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

# Directly load the model without caching
def load_model():
    return tf.keras.models.load_model('/content/drive/MyDrive/Advance_Data_Science/Final Exam/final_model.h5')

model = load_model()

# App title
st.write("# Fashion Item Detection System by Oliver Dulay")

# File uploader for images
file = st.file_uploader("Choose a fashion item photo from your computer", type=["jpg", "png"])

# Predict function that processes the image and makes a prediction using the model
def import_and_predict(image_data, model):
    size = (28, 28)  # Target image size for Fashion MNIST
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = ImageOps.grayscale(image)  # Convert image to grayscale
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image
    img_reshape = img[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    prediction = model.predict(img_reshape)
    return prediction

# Interaction logic based on file upload
if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    result_text = "OUTPUT: " + class_names[np.argmax(prediction)]
    st.success(result_text)
