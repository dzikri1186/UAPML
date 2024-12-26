import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# Konfigurasi halaman
st.set_page_config(
    page_title="Chicken Breed Classification",
    page_icon="🐔",
    layout="wide",  # Memanfaatkan tata letak lebar
)

# CSS untuk mempercantik tampilan
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f7;  /* Warna latar belakang */
    }
    .main {
        font-family: Arial, sans-serif;
        color: #2c3e50;
    }
    .header {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        font-size: 20px;
        color: #34495e;
        margin-bottom: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header aplikasi
st.markdown('<div class="header">Chicken Breed Classification 🐔</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Klasifikasi jenis ayam berdasarkan citra menggunakan CNN dan ResNet.</div>', unsafe_allow_html=True)

# Tambahkan gambar header
header_image = "../assets/sample_image.jpg"  # Ganti dengan gambar di folder assets
try:
    image = Image.open(header_image)
    st.image(image, caption="Contoh Gambar Ayam", use_column_width=True)
except:
    st.warning("Gambar header tidak ditemukan!")

# Load models
cnn_model = load_model("models/cnn_model.h5")
resnet_model = load_model("models/resnet_model.h5")

# Class labels
class_labels = ["Bielefeld", "Blackorpington", "Brahma", "Buckeye", "Fayoumi", "Leghorn", "Newhampshire", "Plymouthrock", "Sussex", "Turken"]

# Upload gambar ayam
uploaded_file = st.file_uploader("Upload gambar ayam untuk klasifikasi 🐓", type=["jpg", "png"])
if uploaded_file:
    st.markdown("---")
    st.subheader("Gambar yang Diupload")
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # Prediksi menggunakan CNN
    cnn_prediction = cnn_model.predict(img_array)
    cnn_class = np.argmax(cnn_prediction)

    # Prediksi menggunakan ResNet
    resnet_prediction = resnet_model.predict(img_array)
    resnet_class = np.argmax(resnet_prediction)

    # Tampilkan hasil prediksi
    st.markdown("---")
    st.subheader("Hasil Prediksi")
    st.write(f"**Prediksi CNN:** {class_labels[cnn_class]} (Kepercayaan: {cnn_prediction[0][cnn_class]:.2f})")
    st.write(f"**Prediksi ResNet:** {class_labels[resnet_class]} (Kepercayaan: {resnet_prediction[0][resnet_class]:.2f})")

    # Visualisasi Confidence
    st.markdown("---")
    st.subheader("Confidence Level")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.barplot(x=class_labels, y=cnn_prediction[0], ax=ax[0], palette="coolwarm")
    ax[0].set_title("CNN Confidence")
    ax[0].tick_params(axis="x", rotation=45)

    sns.barplot(x=class_labels, y=resnet_prediction[0], ax=ax[1], palette="coolwarm")
    ax[1].set_title("ResNet Confidence")
    ax[1].tick_params(axis="x", rotation=45)

    st.pyplot(fig)

# Footer aplikasi
st.markdown("---")
st.markdown("<center><b>Chicken Breed Classification</b> | Powered by Streamlit</center>", unsafe_allow_html=True)
