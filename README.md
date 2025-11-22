# Vision Transformer Comparison

This project compares different Vision Transformer models.
# Vision Transformer (ViT) vs DeiT (Data-efficient Image Transformer)

## Deskripsi
Percobaan ini bertujuan untuk membandingkan kinerja dua model transformer untuk tugas klasifikasi gambar, yaitu **Vision Transformer (ViT)** dan **Data-efficient Image Transformer (DeiT)**. Model-model ini diimplementasikan dengan menggunakan dataset gambar, dan hasil eksperimen akan dievaluasi berdasarkan metrik seperti akurasi, waktu inferensi, jumlah parameter, dan visualisasi confusion matrix.

## Tujuan Percobaan
- Membandingkan dua model Vision Transformer (ViT dan DeiT) pada tugas klasifikasi gambar.
- Menganalisis perbedaan antara model ViT dan DeiT dalam hal akurasi, jumlah parameter, dan waktu inferensi.
- Melakukan evaluasi model menggunakan metrik seperti akurasi, precision, recall, dan F1-score.
- Menyusun laporan ilmiah yang sistematis berdasarkan hasil eksperimen.

## Dataset
Dataset yang digunakan dalam percobaan ini adalah dataset klasifikasi gambar yang terdiri dari 5 kelas makanan Indonesia:
1. Bakso
2. Gado-gado
3. Nasi Goreng
4. Rendang
5. Soto Ayam

Dataset ini memiliki 838 gambar untuk **training** dan 270 gambar untuk **testing**. Gambar-gambar tersebut disimpan dalam folder `data/train` dan `data/test`.

## Model yang Digunakan
1. **Vision Transformer (ViT):**
   - Model transformer murni yang menggunakan self-attention untuk memproses gambar.
   - Model ini digunakan untuk membandingkan performa pada dataset gambar dengan arsitektur berbasis transformer.

2. **Data-efficient Image Transformer (DeiT):**
   - Model yang dikembangkan untuk meningkatkan efisiensi model transformer dengan menggunakan teknik knowledge distillation.
   - Model ini diharapkan dapat mencapai akurasi tinggi dengan penggunaan data yang lebih efisien.

## Langkah-langkah Percobaan
1. **Preprocessing Data:**
   - Dataset gambar di-resize menjadi 224x224 piksel.
   - Gambar diproses menggunakan normalisasi untuk menyesuaikan dengan pre-trained model.
   
2. **Training Model:**
   - Model ViT dan DeiT dilatih menggunakan dataset yang telah diproses.
   - Fungsi loss yang digunakan adalah **CrossEntropyLoss**.
   - Optimizer yang digunakan adalah **Adam** dengan learning rate 1e-4.
   
3. **Evaluasi Model:**
   - Model dievaluasi pada dataset **test** menggunakan metrik seperti **accuracy**, **precision**, **recall**, dan **F1-score**.
   - Visualisasi confusion matrix untuk mengamati hasil klasifikasi yang salah dan benar.
   
4. **Penyimpanan Model:**
   - Setelah training selesai, model dan optimizer disimpan dalam file `.pth` untuk digunakan kembali.
   
5. **Visualisasi Hasil:**
   - Kurva akurasi dan loss selama training.
   - Confusion matrix untuk mengevaluasi hasil klasifikasi.

# Struktur Direktori

Tugas-Deep-Learning-Eksplorasi-Vision-Transformer/ (root folder)
├── .git/ (Folder git yang berisi konfigurasi repositori)
├── data/ (Folder untuk dataset)
├── train/ (Gambar untuk pelatihan)
├── test/ (Gambar untuk pengujian)
├── models/ (Folder untuk model ViT dan DeiT)
├── vit_model.py (Definisi model Vision Transformer)
├── deit_model.py (Definisi model Data-efficient Transformer)
├── train.py (Script untuk proses training)
├── eval.py (Script untuk proses evaluasi)
├── output/ (Folder untuk output hasil model)
├── results/ (Hasil percakapan dan laporan, grafik, confusion matrix)
├── logs/ (Log training dan testing)
├── main.py (Script utama untuk menjalankan eksperimen)
├── requirements.txt (File dependencies Python)
├── README.md (Deskripsi tentang proyek)
└── .gitignore (File untuk menentukan file apa yang tidak dilacak oleh Git)

