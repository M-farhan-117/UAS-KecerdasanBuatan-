# UAS Kecerdasan Buatan — Klasifikasi Risiko Penyakit Jantung Menggunakan KNN dan Naive Bayes

Repository ini berisi laporan dan implementasi Ujian Akhir Semester (UAS) mata kuliah Kecerdasan Buatan, dengan topik klasifikasi risiko penyakit jantung menggunakan algoritma **K-Nearest Neighbors (KNN)** dan **Naive Bayes**.

## Anggota Kelompok

| Nama | NIM |
|------|-----|
| [Muhamad Farhan] | [2406176] |
| [Cakra Nur Ikhwan] | [2406177] |

## Struktur Repository

```
UAS-KecerdasanBuatan/
├── README.md              # Penjelasan proyek & langkah pengerjaan (file ini)
├── Laporan_uas.md          # Laporan lengkap (poin 1-10 sesuai rubrik)
├── uas_model.ipynb         # Notebook implementasi (EDA, preprocessing, modeling, evaluasi)
└── data/
    ├── dataset/             # Dataset mentah & hasil olahan
    └── Jurnal/              # Kumpulan jurnal referensi (PDF)
```

## Ringkasan Proyek

Dataset yang digunakan adalah **Heart Disease Dataset (Cleveland subset)** dari UCI Machine Learning Repository, berisi 303 data pasien dengan 13 fitur klinis (usia, tekanan darah, kolesterol, hasil EKG, dll.) dan 1 target klasifikasi biner (berisiko / tidak berisiko penyakit jantung).

Dua algoritma dibandingkan:
- **K-Nearest Neighbors (KNN)** — pendekatan berbasis jarak
- **Naive Bayes (Gaussian)** — pendekatan probabilistik

Detail lengkap latar belakang, business understanding, data understanding, EDA, data preparation, modeling, evaluasi, kesimpulan, dan referensi ada di [`Laporan_uas.md`](./Laporan_uas.md).

## Langkah Pengerjaan

1. **Persiapan dataset**
   - Dataset diunduh dari [Kaggle — Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) atau [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
   - File disimpan di `data/dataset/heart.csv`.

2. **Exploratory Data Analysis (EDA)**
   - Melihat distribusi tiap fitur (histogram, bar chart, pie chart).
   - Menganalisis korelasi antar fitur menggunakan heatmap dan pairplot.
   - Mengecek keseimbangan kelas target (imbalanced classes check).

3. **Data Preparation**
   - Membersihkan data (menangani missing value pada kolom `ca` dan `thal`, mengecek duplikasi).
   - Melakukan one-hot encoding pada fitur kategorik nominal (`cp`, `restecg`, `slope`, `thal`).
   - Melakukan standardisasi (StandardScaler) pada fitur numerik, terutama untuk KNN.
   - Membagi data menjadi data latih (80%) dan data uji (20%) dengan stratifikasi berdasarkan target.

4. **Modeling**
   - Melatih model **KNN** dengan tuning nilai K menggunakan GridSearchCV.
   - Melatih model **Gaussian Naive Bayes**.
   - Membandingkan kedua model dari sisi pendekatan, asumsi, dan waktu komputasi.

5. **Evaluation**
   - Menghitung confusion matrix untuk masing-masing model.
   - Menghitung metrik accuracy, precision, recall, dan F1-score.
   - Menentukan model terbaik berdasarkan metrik tersebut (lihat pembahasan di `Laporan_uas.md`).

6. **Kesimpulan dan Rekomendasi**
   - Merangkum hasil, mengevaluasi ketercapaian tujuan proyek, kelebihan-keterbatasan model, serta rekomendasi pengembangan selanjutnya.

## Cara Menjalankan

1. Clone repository ini:
   ```bash
   git clone https://github.com/[username]/UAS-KecerdasanBuatan.git
   cd UAS-KecerdasanBuatan
   ```
2. Install dependency yang dibutuhkan:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Jalankan notebook:
   ```bash
   jupyter notebook uas_model.ipynb
   ```
4. Pastikan file dataset (`data/dataset/heart.csv`) sudah tersedia sebelum menjalankan notebook.

## Referensi

Referensi jurnal ilmiah (minimal 5 sumber, gaya APA) tercantum lengkap di bagian akhir [`Laporan_uas.md`](./Laporan_uas.md#9-referensi). File PDF jurnal yang digunakan disimpan di folder `data/Jurnal/`.

## Lisensi

Proyek ini dibuat untuk keperluan akademik (Tugas UAS Kecerdasan Buatan) dan menggunakan dataset publik Heart Disease UCI yang berlisensi CC BY 4.0.
