# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cvwyjexW4k-cQa_ctUWh8vzm4gDHUI_8
"""


import streamlit as st
import numpy as np
import requests
from tensorflow.keras.models import load_model
from io import BytesIO

# URL of the model on GitHub
model_url = 'https://github.com/Redfox13oliverdulay/ADS/blob/main/iris_model.h5'

# Function to load model from a URL
def load_model_from_url(url):
    r = requests.get(url)
    if r.ok:
        model_file = BytesIO(r.content)
        model = load_model(model_file)
        return model
    else:
        st.error("Failed to download the model. Please check the URL.")
        return None

# Load the model
model = load_model_from_url(model_url)

if model:
    st.title('Iris Species Prediction App')
    st.sidebar.image("https://yourimageurl/petal_image.png", caption='Please refer to this image to estimate petal dimensions.', use_column_width=True)

    # User input section
    st.subheader('Please input the measurements for your flower:')
    petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    # Prediction button
    if st.button('Predict'):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        species_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        species = np.argmax(prediction)
        st.write(f'The predicted Iris species for your flower is: {species_dict[species]}')
else:
    st.error("Unable to load the model. Check the setup and try again.")

