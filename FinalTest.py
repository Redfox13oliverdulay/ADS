from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Setup the page configuration using wide page setting
st.set_page_config(layout="wide")

# Try to load the model
try:
    model = load_model('final_model.h5')
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")
    model = None

def import_and_predict(image_data, model):
    if model is None:
        st.error("Model is not loaded, please check the model path.")
        return None
    
    size = (28, 28)  # Target image size for Fashion MNIST
    try:
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        image = ImageOps.grayscale(image)
        img = np.asarray(image)
        img = img / 255.0
        img_reshape = img[np.newaxis, ..., np.newaxis]
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Define the label map
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

# UI layout
st.title('Fashion MNIST Prediction App')
st.write('Upload an image and the app will predict the clothing category.')

# Sidebar for user inputs
with st.sidebar:
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    st.write("Please upload an image of clothing from the Fashion MNIST categories.")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    with st.spinner('Model processing...'):
        prediction = import_and_predict(image, model)
        if prediction is not None:
            predicted_class = label_map[np.argmax(prediction)]
            with col2:
                st.success(f"Prediction: {predicted_class}")
