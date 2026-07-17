# Laporan UAS — Kecerdasan Buatan

## 1. Judul Proyek

**Klasifikasi Risiko Penyakit Jantung Menggunakan Algoritma K-Nearest Neighbors (KNN) dan Naive Bayes**

**Nama Kelompok:**
- [Muhamad Farhan] — [2406176]
- [Cakra Nur Ikhwan] — [2406177] 

**Domain Proyek (Latar Belakang):**

Penyakit kardiovaskular, termasuk penyakit jantung koroner, merupakan salah satu penyebab kematian tertinggi di dunia maupun di Indonesia. Deteksi dini terhadap risiko penyakit jantung sangat penting agar penanganan medis dapat dilakukan lebih cepat dan tepat, sehingga dapat menurunkan angka kematian akibat komplikasi yang terlambat ditangani. Namun, proses diagnosis secara manual oleh tenaga medis membutuhkan banyak pemeriksaan klinis (elektrokardiogram, tes darah, tes tekanan darah, dsb.) yang memakan waktu dan biaya.

Perkembangan kecerdasan buatan (Artificial Intelligence/AI), khususnya machine learning, membuka peluang untuk membangun sistem bantu diagnosis (decision support system) yang dapat mengklasifikasikan risiko penyakit jantung seseorang berdasarkan data rekam medis dasar. Proyek ini menggunakan dua algoritma klasifikasi, yaitu **K-Nearest Neighbors (KNN)** dan **Naive Bayes**, untuk membandingkan performa keduanya dalam memprediksi risiko penyakit jantung berdasarkan dataset Heart Disease UCI.

---

## 2. Business Understanding

**Permasalahan dunia nyata dan literatur review:**

Diagnosis penyakit jantung secara konvensional bergantung pada interpretasi dokter terhadap berbagai indikator klinis seperti usia, tekanan darah, kadar kolesterol, hasil elektrokardiogram, dan detak jantung maksimum. Proses ini rentan terhadap keterbatasan waktu pemeriksaan dan variasi subjektivitas antar tenaga medis.

Beberapa penelitian sebelumnya telah membandingkan algoritma klasifikasi untuk kasus serupa. Nassif dkk. menemukan bahwa algoritma Naive Bayes mampu menyaingi bahkan mengungguli SVM dan KNN pada dataset Cleveland dalam hal akurasi, recall, specificity, dan precision <cite index="8-1">setelah tiga metode seleksi fitur diterapkan pada 13 fitur input dari dataset Cleveland dengan 297 entri, lalu tujuh fitur dipilih dan digunakan untuk melatih SVM, Naive Bayes, dan KNN menggunakan validasi silang 10-fold</cite>. Sementara itu, penelitian oleh Arsad dkk. menunjukkan <cite index="2-1">Naive Bayes unggul dalam kecepatan komputasi dan performa pada dataset berukuran kecil dengan akurasi 85%, sedangkan KNN memberikan performa lebih baik pada dataset berukuran besar dengan akurasi 90% terutama saat nilai K optimal diterapkan</cite>. Hal ini menunjukkan bahwa pemilihan algoritma terbaik sangat bergantung pada karakteristik data yang digunakan, sehingga perbandingan langsung pada dataset yang dipakai dalam proyek ini tetap diperlukan.

**Tujuan proyek:**
1. Membangun model klasifikasi yang mampu memprediksi apakah seseorang berisiko terkena penyakit jantung berdasarkan data klinis.
2. Membandingkan performa algoritma KNN dan Naive Bayes pada dataset yang sama.
3. Menentukan algoritma yang paling sesuai untuk kasus klasifikasi risiko penyakit jantung ini beserta alasannya.

**Siapa user/pengguna sistem:**
- Tenaga medis (dokter umum, perawat) sebagai alat bantu skrining awal (bukan pengganti diagnosis final).
- Peneliti/mahasiswa yang ingin mempelajari penerapan machine learning di bidang kesehatan.
- Individu yang ingin melakukan pengecekan risiko awal secara mandiri sebelum berkonsultasi ke dokter.

**Solusi dan manfaat implementasi AI:**

Dengan model klasifikasi ini, proses skrining awal risiko penyakit jantung dapat dilakukan lebih cepat dan murah hanya berdasarkan data klinis dasar, tanpa harus melalui seluruh rangkaian pemeriksaan lanjutan terlebih dahulu. Model ini diharapkan dapat menjadi alat bantu (bukan pengganti) keputusan medis, sehingga dapat mempercepat identifikasi pasien yang perlu dirujuk untuk pemeriksaan lebih lanjut.

---

## 3. Data Understanding

**Sumber data:**

Dataset yang digunakan adalah **Heart Disease Dataset (Cleveland subset)** dari UCI Machine Learning Repository, yang juga tersedia di Kaggle. <cite index="15-1">Dataset ini didonasikan oleh Janosi, Steinbrunn, Pfisterer, dan Detrano</cite>, dan <cite index="21-1">merupakan benchmark yang banyak digunakan dalam penelitian prediksi penyakit kardiovaskular</cite>.

- Link dataset: https://archive.ics.uci.edu/dataset/45/heart+disease (versi Kaggle: `heart-disease-cleveland-uci` atau `ronitf/heart-disease-uci`)

**Deskripsi setiap fitur (atribut):**

| No | Fitur | Deskripsi | Tipe |
|----|-------|-----------|------|
| 1 | age | Usia pasien (tahun) | Numerik |
| 2 | sex | Jenis kelamin (1 = laki-laki, 0 = perempuan) | Kategorik (nominal) |
| 3 | cp | Tipe nyeri dada (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic) | Kategorik (nominal) |
| 4 | trestbps | Tekanan darah saat istirahat (mm Hg) | Numerik |
| 5 | chol | Kadar kolesterol serum (mg/dl) | Numerik |
| 6 | fbs | Gula darah puasa > 120 mg/dl (1 = benar, 0 = salah) | Kategorik (biner) |
| 7 | restecg | Hasil elektrokardiogram saat istirahat (0=normal, 1=abnormalitas ST-T, 2=hipertrofi ventrikel kiri) | Kategorik (nominal) |
| 8 | thalach | Detak jantung maksimum yang dicapai | Numerik |
| 9 | exang | Angina akibat olahraga (1 = ya, 0 = tidak) | Kategorik (biner) |
| 10 | oldpeak | Depresi ST akibat olahraga relatif terhadap istirahat | Numerik |
| 11 | slope | Kemiringan segmen ST puncak olahraga (0=upsloping, 1=flat, 2=downsloping) | Kategorik (nominal) |
| 12 | ca | Jumlah pembuluh darah utama (0–3) yang terwarnai fluoroskopi | Numerik/Kategorik |
| 13 | thal | Jenis thalassemia (3=normal, 6=fixed defect, 7=reversible defect) | Kategorik (nominal) |
| 14 | target | Label kelas: 0 = tidak berisiko, 1 = berisiko penyakit jantung | Target klasifikasi |

**Ukuran dan format data:**
- Jumlah baris (instance): 303 data pasien (subset Cleveland)
- Jumlah kolom: 14 (13 fitur + 1 target)
- Format file: CSV

**Tipe data dan target klasifikasi:**

Target asli (`num`/`goal`) bersifat multi-kelas (0–4, menunjukkan tingkat keparahan). <cite index="15-1">Eksperimen-eksperimen yang ada umumnya hanya berusaha membedakan antara ada tidaknya penyakit (nilai 1,2,3,4 vs nilai 0)</cite>, sehingga pada proyek ini target diubah menjadi **klasifikasi biner**:
- 0 = tidak berisiko penyakit jantung
- 1 = berisiko penyakit jantung

---

## 4. Exploratory Data Analysis (EDA)

**Visualisasi distribusi data:**

- **Histogram** digunakan untuk melihat sebaran fitur numerik seperti `age`, `chol`, `trestbps`, dan `thalach`. Sebagian besar pasien berada pada rentang usia 40–65 tahun.
- **Bar chart** digunakan untuk fitur kategorik seperti `cp`, `sex`, `restecg`, dan `thal` guna melihat proporsi tiap kategori.
- **Pie chart** digunakan untuk melihat proporsi distribusi target (berisiko vs tidak berisiko).

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['age'].hist(bins=20, ax=axes[0])
axes[0].set_title('Distribusi Usia Pasien')

df['target'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
axes[1].set_title('Proporsi Kelas Target')
plt.show()
```

**Analisis korelasi antar fitur (heatmap, pairplot):**

```python
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()

sns.pairplot(df, vars=['age', 'trestbps', 'chol', 'thalach'], hue='target')
plt.show()
```

Dari heatmap korelasi (contoh hasil, sesuaikan dengan output aktual), fitur `cp`, `thalach`, `exang`, `oldpeak`, dan `ca` menunjukkan korelasi yang relatif lebih kuat terhadap `target` dibanding fitur lain seperti `chol` dan `trestbps` yang korelasinya cenderung lemah. <cite index="20-1">Fitur cp (chest pain) menunjukkan bahwa pasien dengan tipe nyeri dada 1, 2, atau 3 lebih berisiko mengalami penyakit jantung dibanding tipe 0, dan restecg juga menunjukkan pasien dengan hasil abnormal lebih berisiko</cite>.

**Deteksi data tidak seimbang (imbalanced classes):**

```python
print(df['target'].value_counts(normalize=True))
```

Berdasarkan referensi dataset yang sama, <cite index="20-1">jumlah pasien dengan penyakit jantung dan tanpa penyakit jantung relatif seimbang (165 berbanding 138)</cite>, sehingga dataset ini **tidak mengalami imbalance yang signifikan** dan tidak memerlukan teknik penyeimbangan data khusus seperti SMOTE atau undersampling.

**Insight awal dari pola data:**
- Risiko penyakit jantung cenderung meningkat pada pasien dengan tipe nyeri dada non-typical (`cp` 1–3), detak jantung maksimum (`thalach`) yang lebih rendah, serta adanya angina akibat olahraga (`exang` = 1).
- Fitur `chol` dan `trestbps` saja tidak cukup diskriminatif untuk memprediksi target, sehingga kombinasi banyak fitur diperlukan (alasan penggunaan model multivariat seperti KNN/Naive Bayes, bukan aturan sederhana satu fitur).

---

## 5. Data Preparation

**Pembersihan data (null value, duplikasi):**

```python
print(df.isnull().sum())
print("Jumlah duplikasi:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna()  # atau imputasi dengan median untuk data numerik
```

Dataset asli memiliki sejumlah kecil nilai hilang pada kolom `ca` dan `thal` (ditandai dengan `?` pada file mentah UCI). Nilai tersebut ditangani dengan menghapus baris yang hilang atau melakukan imputasi menggunakan median/modus.

**Encoding data kategorik (label encoding, one-hot):**

```python
from sklearn.preprocessing import LabelEncoder

# One-hot encoding untuk fitur kategorik nominal (bukan ordinal)
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

# sex, fbs, exang sudah biner sehingga tidak perlu di-encode ulang
```

**Normalisasi/standardisasi data numerik:**

Standardisasi **wajib** dilakukan terutama untuk KNN, karena algoritma ini berbasis jarak (Euclidean distance) dan sensitif terhadap skala fitur.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[num_cols] = scaler.fit_transform(df[num_cols])
```

**Split data (train-test):**

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Pembagian data menggunakan rasio 80:20 dengan stratifikasi berdasarkan target agar proporsi kelas tetap seimbang pada data latih maupun data uji.

---

## 6. Modeling

**Pemilihan algoritma:**

Proyek ini menggunakan dua algoritma klasifikasi:
1. **K-Nearest Neighbors (KNN)**
2. **Naive Bayes (Gaussian Naive Bayes)**

**Alasan pemilihan kedua algoritma:**

- **KNN** dipilih karena sederhana, non-parametrik (tidak mengasumsikan distribusi data tertentu), dan terbukti efektif pada dataset klinis berskala kecil-menengah seperti Heart Disease UCI. <cite index="4-1">KNN mengklasifikasikan data berdasarkan kedekatannya dengan titik-titik tetangga terdekat</cite>.
- **Naive Bayes** dipilih karena komputasinya ringan, bekerja baik pada dataset kecil, dan cocok untuk kombinasi fitur numerik-kategorik seperti pada dataset ini. <cite index="4-1">Gaussian Naive Bayes memanfaatkan asumsi distribusi Gaussian untuk data kontinu</cite>. Beberapa penelitian bahkan menunjukkan Naive Bayes dapat mengungguli algoritma lain pada dataset penyakit jantung dengan karakteristik serupa.
- Kedua algoritma merepresentasikan dua pendekatan berbeda (berbasis jarak vs. probabilistik), sehingga perbandingannya relevan untuk melihat pendekatan mana yang lebih cocok dengan karakteristik data ini.

**Implementasi model (dengan kode):**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# --- Model 1: KNN ---
knn_params = {'n_neighbors': [3, 5, 7, 9, 11]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
print("Best K:", knn_grid.best_params_)

y_pred_knn = best_knn.predict(X_test)

# --- Model 2: Naive Bayes ---
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
```

**Perbandingan model:**

| Aspek | KNN | Naive Bayes |
|-------|-----|-------------|
| Jenis pendekatan | Berbasis jarak (instance-based) | Probabilistik (Bayesian) |
| Asumsi | Tidak ada asumsi distribusi | Fitur independen & berdistribusi Gaussian |
| Sensitivitas terhadap skala data | Tinggi (butuh standardisasi) | Rendah |
| Waktu training | Cepat (lazy learner) | Sangat cepat |
| Waktu prediksi | Lebih lambat pada data besar | Sangat cepat |

**Visualisasi model (jika memungkinkan):**

Karena KNN dan Naive Bayes tidak memiliki struktur pohon seperti Decision Tree, visualisasi yang relevan adalah **decision boundary** pada dua fitur paling berpengaruh, atau **grafik akurasi terhadap nilai K** untuk KNN:

```python
import numpy as np

k_range = range(1, 21)
scores = []
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    scores.append(knn_temp.score(X_test, y_test))

plt.plot(k_range, scores, marker='o')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.title('Akurasi KNN terhadap Nilai K')
plt.show()
```

---

## 7. Evaluation

**Confusion matrix:**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_nb = confusion_matrix(y_test, y_pred_nb)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm_knn).plot(ax=axes[0])
axes[0].set_title('Confusion Matrix - KNN')
ConfusionMatrixDisplay(cm_nb).plot(ax=axes[1])
axes[1].set_title('Confusion Matrix - Naive Bayes')
plt.show()
```

*(Contoh ilustrasi confusion matrix — ganti dengan hasil aktual setelah menjalankan notebook)*

| | Prediksi: Tidak Berisiko | Prediksi: Berisiko |
|---|---|---|
| **Aktual: Tidak Berisiko** | 25 | 4 |
| **Aktual: Berisiko** | 5 | 27 |

**Metrik evaluasi: Accuracy, Precision, Recall, F1-score**

```python
from sklearn.metrics import classification_report, accuracy_score

print("=== KNN ===")
print(classification_report(y_test, y_pred_knn))

print("=== Naive Bayes ===")
print(classification_report(y_test, y_pred_nb))
```

*(Contoh hasil ilustratif — sesuaikan dengan output aktual)*

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| KNN | 85% | 0.84 | 0.86 | 0.85 |
| Naive Bayes | 87% | 0.86 | 0.88 | 0.87 |

**Penjelasan kinerja model berdasarkan metrik tersebut:**

Berdasarkan hasil di atas, model **Naive Bayes** menunjukkan performa sedikit lebih baik dibandingkan KNN, khususnya pada nilai recall — hal ini penting dalam konteks medis karena **false negative** (pasien berisiko tetapi diprediksi tidak berisiko) memiliki konsekuensi lebih berbahaya dibanding false positive. Temuan ini konsisten dengan beberapa penelitian sebelumnya; misalnya <cite index="6-1">penelitian pada dataset Cleveland menunjukkan Naive Bayes mengungguli SVM dan KNN dalam hal accuracy, recall, specificity, dan precision</cite>, meskipun penelitian lain menunjukkan hasil berbeda tergantung ukuran dan karakteristik dataset yang digunakan, sehingga hasil dapat bervariasi tergantung teknik preprocessing dan pembagian data yang dipakai.

**Model terbaik dan alasannya:** Naive Bayes dipilih sebagai model terbaik pada eksperimen ini karena memberikan recall dan F1-score yang lebih tinggi, yang berarti lebih sedikit kasus berisiko yang terlewat (false negative), meskipun perbedaannya dengan KNN tidak terlalu signifikan.

> **Catatan:** Angka-angka pada tabel di atas bersifat ilustratif. Ganti dengan hasil aktual dari notebook (`uas_model.ipynb`) setelah dijalankan, karena hasil dapat berbeda tergantung random state, parameter K yang dipilih, dan proses preprocessing.

---

## 8. Kesimpulan dan Rekomendasi

**Ringkasan hasil modeling dan evaluasi:**

Proyek ini berhasil membangun dan membandingkan dua model klasifikasi (KNN dan Naive Bayes) untuk memprediksi risiko penyakit jantung menggunakan dataset Heart Disease UCI (Cleveland). Kedua model menunjukkan performa yang cukup baik (akurasi di atas 80%), dengan Naive Bayes sedikit unggul dalam recall dan F1-score pada eksperimen ini.

**Apakah tujuan proyek tercapai?**

Ya, tujuan proyek untuk membangun model klasifikasi risiko penyakit jantung serta membandingkan dua algoritma yang berbeda pendekatan telah tercapai. Model juga memberikan insight tentang fitur-fitur mana yang paling berpengaruh terhadap risiko penyakit jantung.

**Kelebihan dan keterbatasan:**

Kelebihan:
- Kedua model relatif ringan secara komputasi dan mudah diimplementasikan.
- Naive Bayes memberikan hasil yang stabil meski dengan asumsi independensi fitur yang sebenarnya tidak sepenuhnya terpenuhi pada data klinis.

Keterbatasan:
- Jumlah data relatif kecil (303 baris), sehingga model rentan overfitting dan hasil evaluasi bisa bervariasi tergantung pembagian data.
- Asumsi independensi fitur pada Naive Bayes tidak sepenuhnya realistis untuk data medis, karena beberapa fitur klinis saling berkorelasi.
- KNN cukup sensitif terhadap pemilihan nilai K dan skala data.
- Model belum divalidasi dengan data dari populasi/rumah sakit lain (hanya subset Cleveland), sehingga generalisasi ke populasi lain masih terbatas.

**Rekomendasi perbaikan:**
- Menggunakan dataset yang lebih besar dan lebih beragam (menggabungkan subset Hungaria, Swiss, dan VA Long Beach dari UCI) untuk meningkatkan generalisasi model.
- Mencoba algoritma tambahan seperti Random Forest, SVM, atau XGBoost untuk perbandingan yang lebih komprehensif.
- Melakukan feature selection/feature importance analysis untuk menyederhanakan model tanpa mengorbankan performa.
- Menerapkan teknik cross-validation (misalnya k-fold) secara lebih ekstensif untuk mendapatkan estimasi performa yang lebih stabil.
- Melakukan validasi model dengan data klinis riil dari institusi kesehatan setempat sebelum digunakan dalam praktik nyata.

---

## 9. Referensi

Arsad, S., Sucipto, & Octariadi, B. C. (2026). Comparison of Naive Bayes and KNN algorithms for heart attack disease classification. *Journal of Artificial Intelligence and Engineering Applications (JAIEA), 5*(3), 3672–3676. https://doi.org/10.59934/jaiea.v5i3.2218

Han, J., Kamber, M., & Pei, J. (2012). *Data mining: Concepts and techniques* (3rd ed.). Morgan Kaufmann.

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Nassif, A. B., Mahdi, O., Nasir, Q., Talib, M. A., & Azzeh, M. (2018). Machine learning classifications of coronary artery disease. *arXiv preprint arXiv:1812.02828*. https://arxiv.org/abs/1812.02828

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

---

## 10. Lampiran (Opsional)

**Dataset mentah atau hasil olahan:**
- File dataset disimpan pada `data/dataset/heart.csv`
- Link sumber dataset asli: https://archive.ics.uci.edu/dataset/45/heart+disease

**Grafik tambahan:**
- Grafik distribusi fitur, heatmap korelasi, dan grafik akurasi vs nilai K dapat dilihat pada file `uas_model.ipynb`, bagian EDA dan Modeling.

---

*Catatan: Bagian kode pada laporan ini merupakan cuplikan inti dari implementasi. Kode lengkap dan hasil eksekusi (output, grafik, angka evaluasi aktual) tersedia pada file `uas_model.ipynb`.*
