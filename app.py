import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import joblib

# Load model and class labels
MODEL_PATH = 'C:/AI lab/leaf_disease_model.h5'
LABELS_PATH = 'C:/AI lab/class_labels.joblib'

model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = joblib.load(LABELS_PATH)

# Streamlit UI
st.title("Leaf Disease Detection")
st.write("Upload an image of a leaf to detect its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)

    # Prediction
    prediction = model.predict(img)
    predicted_class = list(CLASS_NAMES.keys())[np.argmax(prediction)]

    st.write(f"**Detected Disease:** {predicted_class}")
    st.write("Confidence:", round(np.max(prediction) * 100, 2), "%")
