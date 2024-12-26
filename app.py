import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
print(f"Matplotlib Backend: {matplotlib.get_backend()}")

# CSS untuk mempercantik tampilan
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f9;
    }
    .main {
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-image: url("https://img.lovepik.com/photo/45010/4273.jpg_wh860.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
.title-container {
    text-align: center;
    background-color: rgba(0, 0, 0, 0.8); /* Latar belakang hitam */
    padding: 20px 30px; /* Tambahkan padding */
    border-radius: 15px; /* Membuat sudut melengkung */
    margin: auto; /* Otomatis rata tengah secara horizontal */
    width: 80%; /* Lebar kontainer */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Tambahkan bayangan */
    position: relative; /* Pastikan posisi relatif */
    top: 100%; /* Geser elemen ke tengah secara vertikal */
    transform: translateY(-50%); /* Koreksi agar benar-benar berada di tengah */
}

.title {
    font-size: 36px;
    font-weight: bold;
    color: #ffffff;
    margin: 0;
}
.subtitle {
    font-size: 15px;
    color: #ffffff;
    margin-top: 5px;
}

    .stFileUploader {
        border: 2px dashed #4caf50 !important;
        border-radius: 10px;
        padding: 10px;
        background-color: #e8f5e9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load models
cnn_model = load_model("C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/models/cnn_model.h5")
resnet_model = load_model("C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/models/resnet_model.h5")

# Class labels
class_labels = ["Bielefeld", "Blackorpington", "Brahma", "Buckeye", "Fayoumi", "Leghorn", "Newhampshire", "Plymouthrock", "Sussex", "Turken"]

# Function to preprocess image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# Streamlit UI
st.markdown(
    """
    <div class="title-container">
        <div class="title">Chicken Breed Classification</div>
        <div class="subtitle">Klasifikasi jenis ayam berdasarkan citra menggunakan CNN dan ResNet.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload Image
uploaded_file = st.file_uploader("Upload an image of a chicken", type=["jpg", "png"])
if uploaded_file:
    img, img_array = preprocess_image(uploaded_file)

    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict with both models
    cnn_prediction = cnn_model.predict(img_array)
    resnet_prediction = resnet_model.predict(img_array)

    # Get predicted class
    cnn_class = np.argmax(cnn_prediction)
    resnet_class = np.argmax(resnet_prediction)

    # Wrap results in a styled container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.subheader("Predictions:")
    st.write(f"**CNN Prediction:** {class_labels[cnn_class]} (Confidence: {cnn_prediction[0][cnn_class]:.2f})")
    st.write(f"**ResNet Prediction:** {class_labels[resnet_class]} (Confidence: {resnet_prediction[0][resnet_class]:.2f})")
    st.markdown('</div>', unsafe_allow_html=True)

    # Visualize Confidence
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x=class_labels, y=cnn_prediction[0], ax=ax[0])
    ax[0].set_title("CNN Confidence")
    ax[0].tick_params(axis='x', rotation=45)

    sns.barplot(x=class_labels, y=resnet_prediction[0], ax=ax[1])
    ax[1].set_title("ResNet Confidence")
    ax[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)
