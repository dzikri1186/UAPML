# Chicken Breed Classification 🐔
Klasifikasi jenis ayam berdasarkan citra menggunakan CNN dan ResNet.
---
## Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan sistem klasifikasi gambar untuk mengidentifikasi jenis ayam berdasarkan citra dengan memanfaatkan deep learning. Arsitektur yang digunakan adalah:
1. **Convolutional Neural Networks (CNN)**
2. **ResNet (Residual Neural Network)**

Dataset yang digunakan dapat anda akses pada link :https://data.mendeley.com/datasets/nk3zbvd5h8/1

Link Google drive Proyek :https://drive.google.com/drive/folders/1Wncwj4Yxsaxbc4bHWL0Vmr8xffKFe3z0?usp=sharing

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

![Screenshot 2024-12-26 114616](https://github.com/user-attachments/assets/bc534f1d-43b0-4cca-988f-f65e37f6c135)



# CNN (Convolutional Neural Network)
Struktur Lapisan Utama:
Conv2D: Digunakan untuk mengekstrak fitur dari citra dengan kernel berukuran 3x3.
MaxPooling2D: Berfungsi untuk mengurangi dimensi data sekaligus mempertahankan fitur penting.
Dense Layer: Lapisan fully connected untuk melakukan klasifikasi akhir.
Dropout: Digunakan untuk mencegah overfitting selama pelatihan.
Arsitektur CNN yang digunakan telah dirancang dengan konfigurasi ini untuk memastikan model dapat belajar dari dataset dengan baik tanpa overfitting.

Hasil Model:
Akurasi Training: 100% (seperti yang terlihat dari confusion matrix dan laporan klasifikasi).
Akurasi Validasi: 100% (model memberikan prediksi sempurna pada data validasi).
Observasi: Hasil ini menunjukkan bahwa CNN memiliki performa luar biasa dalam mengklasifikasikan jenis ayam dari citra, tanpa kesalahan.


## ResNet (Residual Neural Network)
Transfer Learning:
Model ResNet50 menggunakan bobot pre-trained dari dataset ImageNet.
Fine-tuning dilakukan dengan menambahkan lapisan dense yang disesuaikan dengan jumlah kelas (10 kelas) dan dropout untuk mengurangi risiko overfitting.
ResNet50 dikenal memiliki kemampuan yang sangat baik dalam mengekstraksi fitur visual berkat penggunaan residual blocks, yang membantu mengatasi vanishing gradient pada jaringan yang sangat dalam.
Hasil Model:
Akurasi Training: 43% (seperti terlihat pada laporan klasifikasi).
Akurasi Validasi: 28% (menunjukkan performa yang rendah pada data validasi).
Observasi:
Hasil ini mengindikasikan bahwa model ResNet mengalami kesulitan dalam menangani dataset ini.
Hal ini mungkin disebabkan oleh:
Kurangnya data augmentasi untuk meningkatkan generalisasi model.
Overfitting selama pelatihan, terlihat dari performa yang lebih baik pada data training dibandingkan validasi.
Dataset mungkin tidak cukup besar untuk melatih model yang kompleks seperti ResNet secara optimal.

![Screenshot 2024-12-25 191311](https://github.com/user-attachments/assets/711860e7-e040-40a7-b2e5-79db8d7c76c5)

Classification Report - CNN :

Semua metrik, yaitu precision, recall, dan F1-score untuk setiap kelas ayam adalah 1.00.
Akurasi total adalah 100%, menunjukkan bahwa CNN mampu mengklasifikasikan dataset tanpa kesalahan.

Classification Report - ResNet :

ResNet memiliki metrik yang jauh lebih rendah dibandingkan CNN.
Precision, recall, dan F1-score bervariasi di antara kelas ayam, dengan beberapa kelas memiliki skor yang sangat rendah.
Akurasi total adalah 43%, yang menunjukkan bahwa model ini mengalami kesulitan dalam mengenali kelas dengan benar.


## Confusion Matrix
##### Confusion Matrix - ResNet (Gambar 1):
Pada matriks ini, terlihat bahwa model ResNet memiliki kinerja yang kurang baik dalam mengklasifikasikan beberapa jenis ayam. Banyak prediksi yang tersebar ke kelas lain, sehingga menunjukkan akurasi dan performa yang rendah.

![Figure_2](https://github.com/user-attachments/assets/f30d8b14-daf2-4f66-90a3-4bce0e1dfcfb)

#### Confusion Matrix - CNN (Gambar 2):
Model CNN menunjukkan kinerja yang sempurna, dengan semua prediksi berada pada diagonal matriks. Ini berarti setiap citra ayam berhasil diklasifikasikan ke kelas yang benar tanpa kesalahan.

![Figure_1](https://github.com/user-attachments/assets/29ff4ba9-78b9-461e-93c2-f02c64387591)





