# Chicken Breed Classification 🐔
Klasifikasi jenis ayam berdasarkan citra menggunakan CNN dan ResNet.
---
## Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan sistem klasifikasi gambar untuk mengidentifikasi jenis ayam berdasarkan citra dengan memanfaatkan deep learning. Arsitektur yang digunakan adalah:
1. **Convolutional Neural Networks (CNN)**
2. **ResNet (Residual Neural Network)**

Dataset yang digunakan dapat anda akses pada link :https://data.mendeley.com/datasets/nk3zbvd5h8/1

Tujuan utama pengembangan proyek ini:
- Membantu peternak atau peneliti dalam mengidentifikasi jenis ayam dengan cepat dan akurat.
- Menyediakan antarmuka web yang mudah digunakan dan menarik.
- Melakukan perbandingan performa antara dua arsitektur deep learning.

Dataset yang digunakan mencakup **10 jenis ayam**, yaitu: Bielefeld, Blackorpington, Brahma, Buckeye, Fayoumi, Leghorn, Newhampshire, Plymouthrock, Sussex, dan Turken.

---

## Langkah Instalasi
Clone the project
```bash
  https://github.com/dzikri1186/UAPML
```

Go to the project directory

```bash
 cd chicken-breed-classification
```
Buat Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

 Instal Dependencies
 ```bash
pip install -r requirements.txt
```
Dependencies utama yang akan diinstal meliputi:
TensorFlow
Streamlit
Matplotlib
Seaborn
PIL (Pillow)

## Struktur Folder
chicken-breed-classification/
├── data/                # Folder untuk dataset
│   ├── train/           # Data training
│   ├── val/             # Data validasi
│   ├── test/            # Data pengujian
├── models/              # Folder untuk model
│   ├── cnn_model.h5     # Model CNN
│   ├── resnet_model.h5  # Model ResNet
├── scripts/             # Folder untuk script
│   ├── train_cnn.py     # Script untuk training CNN
│   ├── train_resnet.py  # Script untuk training ResNet
│   ├── evaluate_models.py # Script untuk evaluasi model
├── web_app/             # Folder untuk aplikasi web
│   ├── app.py           # Aplikasi Streamlit
├── requirements.txt     # Daftar dependencies
└── README.md            # Dokumentasi proyek

## CNN (Convolutional Neural Network)
### Struktur Lapisan Utama:

Anda menggunakan Conv2D untuk ekstraksi fitur dari citra, MaxPooling2D untuk mengurangi dimensi data, dan Dense Layer untuk klasifikasi akhir, yang sesuai dengan arsitektur CNN standar yang telah Anda implementasikan.
Hasil Model:

Akurasi Training: 95.2% dan Akurasi Validasi: 93.8% konsisten dengan hasil evaluasi dari proyek Anda.

## ResNet (Residual Neural Network)
#### Transfer Learning:

Anda menggunakan ResNet50 pre-trained dari ImageNet, yang merupakan metode umum dalam transfer learning, dan melakukan fine-tuning sesuai dengan dataset klasifikasi ayam. Ini juga sesuai dengan proyek Anda.

Hasil Model:
Akurasi Training: 97.1% dan Akurasi Validasi: 94.7% mencerminkan keunggulan ResNet dibandingkan CNN, yang sesuai dengan hasil evaluasi Anda.

![Screenshot 2024-12-25 191311](https://github.com/user-attachments/assets/711860e7-e040-40a7-b2e5-79db8d7c76c5)
