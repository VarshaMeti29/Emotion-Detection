import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "emotion_model.h5"     # Make sure this file is in same folder
model = load_model(MODEL_PATH, compile=False)

class_labels = ['fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("Facial Emotion Recognition")
st.write("Upload an image to detect the emotion.")

# ----------------------------------------
# File Upload
# ----------------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg","webp"])

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, channels="BGR", caption="Uploaded Image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for model
    resized = cv2.resize(gray, (48, 48))
    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # ------------ Prediction ---------------
    prediction = model.predict(img_array)
    label_index = np.argmax(prediction)
    emotion = class_labels[label_index]

    st.subheader("Predicted Emotion:")
    st.success(emotion)
