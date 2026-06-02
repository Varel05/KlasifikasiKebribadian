# 🧠 Klasifikasi Kepribadian Introvert & Ekstrovert Berdasarkan Kebiasaan

Proyek ini menggunakan algoritma **Decision Tree** untuk mengklasifikasikan kepribadian seseorang — **Introvert** atau **Ekstrovert** — berdasarkan data kebiasaan dan perilaku sosial sehari-hari.

---

## 📋 Daftar Isi

- [Gambaran Proyek](#gambaran-proyek)
- [Dataset](#dataset)
- [Penjelasan Fitur](#penjelasan-fitur)
- [Persyaratan](#persyaratan)
- [Cara Menggunakan](#cara-menggunakan)
- [Alur Kerja Kode](#alur-kerja-kode)
- [Output & Visualisasi](#output--visualisasi)
- [Metrik Evaluasi](#metrik-evaluasi)

---

## 🔍 Gambaran Proyek

Model ini dilatih menggunakan algoritma **Decision Tree Classifier** dari scikit-learn untuk memprediksi tipe kepribadian berdasarkan 7 fitur perilaku. Proyek ini dijalankan di **Google Colab** dan mendukung dua cara memuat data: unggah langsung atau via Google Drive.

---

## 📁 Dataset

File dataset yang dibutuhkan: `personality_datasert.csv`

Dataset harus memiliki kolom-kolom berikut:

| Nama Kolom | Tipe | Keterangan |
|---|---|---|
| `Time_spent_Alone` | Numerik | Waktu yang dihabiskan sendirian |
| `Stage_fear` | Kategorikal | Ketakutan tampil di depan umum |
| `Social_event_attendance` | Numerik | Frekuensi menghadiri acara sosial |
| `Going_outside` | Numerik | Frekuensi keluar rumah |
| `Drained_after_socializing` | Kategorikal | Merasa lelah setelah bersosialisasi |
| `Friends_circle_size` | Numerik | Ukuran lingkaran pertemanan |
| `Post_frequency` | Numerik | Frekuensi posting di media sosial |
| `Personality` | Kategorikal | **Label target**: `Introvert` / `Extrovert` |

---

## 🏷️ Penjelasan Fitur

### 1. `Time_spent_Alone` — Waktu Dihabiskan Sendirian
Mengukur seberapa banyak waktu seseorang dihabiskan dalam kondisi menyendiri. Introvert cenderung lebih banyak menghabiskan waktu sendirian sebagai cara mengisi energi, sementara ekstrovert lebih sedikit.

### 2. `Stage_fear` — Ketakutan Tampil di Depan Umum
Merepresentasikan apakah seseorang mengalami rasa takut atau cemas saat harus tampil atau berbicara di depan orang banyak. Nilai bertipe kategorikal (misal: `Yes` / `No`).

### 3. `Social_event_attendance` — Kehadiran di Acara Sosial
Mengukur seberapa sering seseorang menghadiri acara sosial seperti pesta, pertemuan, atau kegiatan kelompok. Ekstrovert umumnya memiliki nilai lebih tinggi pada fitur ini.

### 4. `Going_outside` — Frekuensi Keluar Rumah
Mengukur seberapa sering seseorang keluar dari rumah untuk aktivitas apapun. Berkorelasi dengan tingkat keterbukaan terhadap lingkungan sosial.

### 5. `Drained_after_socializing` — Kelelahan Setelah Bersosialisasi
Menggambarkan apakah seseorang merasa terkuras energinya setelah berinteraksi sosial. Ciri khas introvert adalah merasa lelah setelah sosialisasi, sementara ekstrovert justru merasa berenergi. Nilai bertipe kategorikal (misal: `Yes` / `No`).

### 6. `Friends_circle_size` — Ukuran Lingkaran Pertemanan
Mengukur jumlah teman dekat atau kenalan yang dimiliki seseorang. Ekstrovert cenderung memiliki lingkaran sosial yang lebih luas, sedangkan introvert lebih sedikit namun lebih dalam.

### 7. `Post_frequency` — Frekuensi Posting di Media Sosial
Mengukur seberapa sering seseorang membuat postingan di media sosial. Dapat merefleksikan keinginan untuk berbagi dan berinteraksi secara publik.

---

## ⚙️ Persyaratan

### Library Python
```
pandas
scikit-learn
numpy
matplotlib
seaborn
graphviz
```

### Instalasi
Jalankan cell pertama di notebook untuk menginstal semua library:
```python
!pip install pandas scikit-learn numpy matplotlib seaborn graphviz
```

### Lingkungan
- **Google Colab** (direkomendasikan)
- Python 3.7+

---

## 🚀 Cara Menggunakan

### Langkah 1: Buka Notebook
Buka file `klasifikasi.ipynb` di Google Colab.

### Langkah 2: Instal Library
Jalankan cell pertama untuk menginstal semua dependensi yang dibutuhkan.

### Langkah 3: Muat Dataset

Tersedia **dua pilihan** untuk memuat data:

**Pilihan A — Unggah File Langsung (Default)**
```python
# Jalankan cell ini, lalu pilih file personality_datasert.csv dari komputer Anda
from google.colab import files
uploaded = files.upload()
```

**Pilihan B — Dari Google Drive**
Nonaktifkan Pilihan A (comment out), lalu aktifkan blok kode Pilihan B dan sesuaikan path file:
```python
# from google.colab import drive
# drive.mount('/content/drive')
# file_path = '/content/drive/MyDrive/path/ke/folder/personality_datasert.csv'
```

### Langkah 4: Jalankan Semua Cell Secara Berurutan
Klik **Runtime → Run all** atau jalankan setiap cell satu per satu dari atas ke bawah.

### Langkah 5: Lihat Hasil
Hasil evaluasi model dan visualisasi akan tampil otomatis setelah semua cell selesai dijalankan.

---

## 🔄 Alur Kerja Kode

```
1. Instalasi Library
       ↓
2. Memuat Data (CSV)
       ↓
3. Prapemrosesan Data
   - Pemisahan fitur (X) dan target (y)
   - Identifikasi kolom numerik & kategorikal
   - Encoding label target (LabelEncoder)
   - Penanganan nilai hilang (SimpleImputer)
   - One-Hot Encoding fitur kategorikal
   - Penskalaan fitur (StandardScaler)
       ↓
4. Pembagian Data (80% latih / 20% uji)
       ↓
5. Pelatihan Model Decision Tree
   (max_depth=5, random_state=42)
       ↓
6. Evaluasi Model
       ↓
7. Visualisasi Hasil
```

---

## 📊 Output & Visualisasi

Notebook menghasilkan 4 visualisasi:

| No | Visualisasi | Keterangan |
|---|---|---|
| 1 | **Confusion Matrix Heatmap** | Matriks perbandingan antara prediksi model dan label sebenarnya |
| 2 | **ROC Curve & AUC** | Kurva performa klasifikasi biner beserta nilai Area Under Curve |
| 3 | **Feature Importance Chart** | Bar chart yang menampilkan fitur mana yang paling berpengaruh terhadap keputusan model |
| 4 | **Pohon Keputusan** | Visualisasi struktur lengkap Decision Tree dengan kedalaman maksimal 5 |

---

## 📈 Metrik Evaluasi

Model dievaluasi menggunakan metrik berikut:

| Metrik | Keterangan |
|---|---|
| **Akurasi** | Persentase prediksi yang benar dari keseluruhan data uji |
| **Presisi** | Proporsi prediksi positif yang benar-benar positif |
| **Recall** | Kemampuan model menemukan semua sampel positif yang sebenarnya |
| **F1-Score** | Rata-rata harmonik antara Presisi dan Recall |
| **Classification Report** | Laporan lengkap per kelas (Introvert & Extrovert) |

---

## 📝 Catatan

- Pastikan nama file CSV adalah `personality_datasert.csv` (perhatikan penulisan *datasert*) agar sesuai dengan variabel di kode, atau ubah variabel `file_name` di Cell 3.
- Nama kolom di CSV harus **sama persis** (case-sensitive) dengan yang didefinisikan di kode.
- Parameter `max_depth=5` pada Decision Tree dapat diubah untuk menyesuaikan kompleksitas model.
