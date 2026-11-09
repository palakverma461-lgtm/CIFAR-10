import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("custom_cifar_model.keras")

model = load_model()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its class.")

# 1. Drag and drop image uploader
uploaded_file = st.file_uploader("Drag and drop image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 2. Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)
    
    # 3. Auto-resize to 32x32 and preprocess
    image_resized = image.resize((32, 32))
    img_array = np.array(image_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)[0]
    max_prob = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]
    
    # 4. Green Success Box with prediction & confidence (jaise screenshot me hai)
    st.success(f"Prediction: **{predicted_class}** ({max_prob * 100:.2f}% confidence)")
    
    # Confidence scores bar chart
    st.write("### Confidence Scores")
    chart_data = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    st.bar_chart(chart_data)