import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image

# model yükle
model = keras.models.load_model("emotion_model.h5")

class_names = ["happy", "sad", "neutral"]

st.title("Emotion Detection CNN")

uploaded_file = st.file_uploader("Bir yüz görüntüsü yükle", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img_resized = img.resize((48,48))

    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(img, caption="Yüklenen görüntü", width=200)
    st.write(f"### Tahmin: {predicted_class}")
