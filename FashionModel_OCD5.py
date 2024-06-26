# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YSwP2N0BM_wISfwdJfmEIwT2l0zXHb9c
"""

from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Attempt to load the model from a specified path
try:
    model = load_model('final_model.h5')
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")
    model = None  # Ensure model is set to None if load fails

def import_and_predict(image_data, model):
    if model is None:
        st.error("Model is not loaded, please check the model path.")
        return None  # Return None if the model isn't loaded

    size = (28, 28)  # Target image size for Fashion MNIST
    try:
        # Use Image.Resampling.LANCZOS for high-quality downsampling
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        image = ImageOps.grayscale(image)  # Convert image to grayscale
        img = np.asarray(image)
        img = img / 255.0  # Normalize the image
        img_reshape = img[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit web interface
st.title('Fashion MNIST Prediction App')
st.write('This app predicts the clothing category from images.')

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    prediction = import_and_predict(image, model)
    if prediction is not None:
        label_map = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }
        predicted_class = label_map[np.argmax(prediction)]
        st.write(f"Prediction: {predicted_class}")

