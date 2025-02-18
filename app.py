import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the Teachable Machine model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_model.h5")  # Change to your model's folder
    return model

model = load_model()

labels = ["Fresh Apple","Stale Apple", "Fresh Banana", "Stale Banana"]
# Streamlit UI
st.title("Teachable Machine Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Adjust based on your model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)

    st.write(f"Prediction: {labels[class_index]}")
    st.write(f"Confidence: {predictions[0][class_index]:.2f}")
