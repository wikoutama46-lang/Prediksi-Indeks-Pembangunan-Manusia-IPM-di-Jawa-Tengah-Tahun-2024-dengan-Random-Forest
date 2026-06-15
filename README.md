# 🌍 Prediksi Indeks Pembangunan Manusia (IPM) dengan Random Forest

> Sistem prediksi IPM Kabupaten/Kota Jawa Tengah berbasis **Random Forest Regressor** dengan antarmuka CLI interaktif.

---

## 📋 Deskripsi Proyek

Proyek ini membangun model machine learning untuk **memprediksi Indeks Pembangunan Manusia (IPM)** di tingkat Kabupaten/Kota se-Jawa Tengah menggunakan algoritma **Random Forest Regressor**. Data yang digunakan mencakup indikator-indikator pembangunan manusia tahun 2024 yang diterbitkan oleh BPS.

Program ini dilengkapi dengan antarmuka baris perintah (CLI) yang interaktif, memungkinkan pengguna untuk:
- Melihat performa model secara detail
- Menganalisis kontribusi setiap indikator
- Menelusuri prediksi per daerah
- Melakukan simulasi prediksi IPM berdasarkan input baru

---

## 📁 Struktur Direktori

```
Prediksi Indeks Pembangunan Manusia (IPM) dengan Random Forest/
│
├── Data/
│   └── indeks-pembangunan-manusia-ipm-menurut-kabupatenkota-tahun-2024.csv
│
├── prediksi_ipm_rf.py       # Script utama
└── README.md                # Dokumentasi ini
```

---

## ⚙️ Persyaratan Sistem

| Komponen     | Versi Minimum |
|-------------|---------------|
| Python       | 3.8+          |
| pandas       | 1.3+          |
| scikit-learn | 1.0+          |
| numpy        | 1.21+         |

---

## 🚀 Instalasi & Cara Menjalankan

### 1. Clone atau Unduh Proyek

```bash
git clone https://github.com/username/prediksi-ipm-rf.git
cd prediksi-ipm-rf
```

### 2. Install Dependensi

```bash
pip install pandas scikit-learn numpy
```

Atau jika menggunakan file `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Pastikan Path Data Sudah Benar

Buka `prediksi_ipm_rf.py` dan sesuaikan variabel `FILE_PATH` di bagian konfigurasi:

```python
FILE_PATH = r"C:\...\Data\indeks-pembangunan-manusia-ipm-menurut-kabupatenkota-tahun-2024.csv"
```

Ganti path tersebut dengan lokasi file CSV di komputer Anda.

### 4. Jalankan Program

```bash
python prediksi_ipm_rf.py
```

---

## 📊 Sumber Data

| Atribut       | Keterangan |
|--------------|------------|
| **Nama file** | `indeks-pembangunan-manusia-ipm-menurut-kabupatenkota-tahun-2024.csv` |
| **Sumber**    | Badan Pusat Statistik (BPS) Provinsi Jawa Tengah |
| **Format**    | CSV dengan separator titik koma (`;`) |
| **Cakupan**   | Seluruh Kabupaten/Kota di Jawa Tengah, Tahun 2024 |

### Struktur CSV

| Kolom         | Keterangan |
|--------------|------------|
| `tahun_data` | Tahun pengamatan |
| `kod_wil`    | Kode wilayah Kabupaten/Kota |
| `kab_kota`   | Nama Kabupaten/Kota |
| `indikator`  | Nama indikator (lihat tabel di bawah) |
| `indeks`     | Nilai indikator (angka desimal, separator koma) |

### Indikator yang Digunakan

| Nama Indikator (CSV)                                                    | Alias Model |
|-------------------------------------------------------------------------|-------------|
| Usia Harapan Hidup saat Lahir (tahun)                                   | `UHH`       |
| Harapan Lama Sekolah (tahun)                                            | `HLS`       |
| Rata-rata Lama Sekolah (tahun)                                          | `RLS`       |
| Pengeluaran per kapita Disesuaikan (ribu rupiah/orang/tahun)            | `Pengeluaran` |
| Indeks Pembangunan Manusia                                              | `IPM` *(target)* |

---

## 🤖 Penjelasan Model

### Algoritma: Random Forest Regressor

Random Forest adalah metode *ensemble learning* yang membangun banyak pohon keputusan (decision tree) secara paralel dan menggabungkan hasilnya. Keunggulan utama:

- **Tahan terhadap overfitting** berkat mekanisme *bagging* dan *random feature selection*
- **Robust** terhadap outlier dan data yang tidak normal
- **Menghasilkan feature importance** untuk interpretasi model

### Konfigurasi Model

```python
RandomForestRegressor(
    n_estimators=500,        # 500 pohon keputusan
    max_features="sqrt",     # Fitur per split = √(jumlah fitur)
    random_state=42,         # Seed untuk reprodusibilitas
    n_jobs=-1                # Gunakan semua core CPU
)
```

### Strategi Validasi

| Jumlah Data | Metode Validasi           |
|-------------|---------------------------|
| ≤ 50 observasi | Leave-One-Out CV (LOO-CV) |
| > 50 observasi | 5-Fold Cross Validation   |

LOO-CV dipilih untuk dataset kecil karena memaksimalkan data training dan menghasilkan estimasi bias yang lebih rendah.

---

## 🖥️ Panduan Penggunaan CLI

Setelah program berjalan dan model selesai dilatih, Anda akan masuk ke **Menu Utama**:

```
=================================================================
                  PREDIKSI IPM — RANDOM FOREST
=================================================================
  MENU UTAMA
-----------------------------------------------------------------
  [1] Lihat hasil evaluasi model
  [2] Lihat feature importance
  [3] Lihat prediksi per kabupaten/kota
  [4] Simulasi prediksi IPM baru
  [0] Keluar
-----------------------------------------------------------------
```

### Menu 1 — Evaluasi Model

Menampilkan metrik performa model hasil cross-validation:

| Metrik | Keterangan |
|--------|-----------|
| **R²** | Koefisien determinasi (seberapa baik model menjelaskan variasi data). Nilai mendekati 1 = sangat baik |
| **RMSE** | Root Mean Squared Error — rata-rata error dalam satuan IPM |
| **MAE** | Mean Absolute Error — rata-rata absolut selisih prediksi dan aktual |

Interpretasi R²:
- `R² > 0.95` → Sangat baik ✓
- `R² > 0.85` → Baik
- `R² ≤ 0.85` → Perlu penyesuaian hyperparameter

### Menu 2 — Feature Importance

Menampilkan kontribusi masing-masing indikator terhadap prediksi IPM dalam bentuk **bar chart teks** beserta peringkatnya. Berguna untuk memahami indikator mana yang paling menentukan nilai IPM.

### Menu 3 — Prediksi per Kabupaten/Kota

Menampilkan tabel perbandingan nilai IPM aktual vs. prediksi model untuk setiap daerah, disertai:
- **Residual** = IPM Aktual − IPM Prediksi
- **APE (%)** = Persentase error absolut
- Tanda `⚠` jika residual > 1.5 (memerlukan perhatian)

Tersedia filter berdasarkan tahun.

### Menu 4 — Simulasi Prediksi IPM Baru

Pengguna dapat memasukkan nilai indikator secara manual untuk mendapatkan prediksi IPM beserta:
- **Kategori IPM**: Rendah (< 70) / Sedang (70–79) / Tinggi (≥ 80)
- **3 Daerah Referensi Terdekat**: Kabupaten/Kota yang IPM aktualnya paling mendekati hasil prediksi

Contoh input simulasi:

```
  UHH  - Usia Harapan Hidup (tahun)           : 74.5
  HLS  - Harapan Lama Sekolah (tahun)         : 13.2
  RLS  - Rata-rata Lama Sekolah (tahun)       : 8.7
  Pengeluaran per kapita (ribu Rp/orang/tahun): 11500
```

Contoh output:

```
  ───────────────────────────────────────────────────────
  HASIL PREDIKSI
  ───────────────────────────────────────────────────────
  Prediksi IPM  : 73.45
  Kategori      : SEDANG (70 – 79)
  ───────────────────────────────────────────────────────
```

---

## 📈 Kategori IPM

Klasifikasi IPM mengacu pada standar UNDP/BPS:

| Kategori | Rentang Nilai |
|----------|--------------|
| 🔴 Rendah | < 70 |
| 🟡 Sedang | 70 – 79 |
| 🟢 Tinggi | ≥ 80 |

---

## 🔧 Kustomisasi

### Mengganti Dataset

Ganti `FILE_PATH` dan `SEPARATOR` di bagian konfigurasi:

```python
FILE_PATH = r"path/ke/file/data_baru.csv"
SEPARATOR = ";"   # atau "," tergantung format CSV
```

Pastikan kolom CSV sesuai: `tahun_data`, `kod_wil`, `kab_kota`, `indikator`, `indeks`.

### Mengganti Hyperparameter Model

```python
rf = RandomForestRegressor(
    n_estimators=1000,     # Tambah jumlah pohon untuk akurasi lebih tinggi
    max_depth=10,          # Batasi kedalaman pohon untuk mengurangi overfitting
    min_samples_split=5,   # Minimum sampel untuk split node
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

---

## 🐛 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `File tidak ditemukan` | Periksa variabel `FILE_PATH` di konfigurasi |
| `ModuleNotFoundError: pandas` | Jalankan `pip install pandas scikit-learn` |
| `KeyError: 'indikator'` | Pastikan nama kolom CSV sudah sesuai (huruf kecil, tanpa spasi ekstra) |
| `Hasil prediksi tidak masuk akal` | Periksa satuan input — Pengeluaran dalam **ribu rupiah**, bukan rupiah |
| Program berjalan lambat | Kurangi `n_estimators` atau gunakan `n_jobs=-1` (sudah diset default) |

---

## 📚 Referensi

- [Scikit-learn: Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [BPS Jawa Tengah — Indeks Pembangunan Manusia](https://jateng.bps.go.id)
- [UNDP — Human Development Index](https://hdr.undp.org/data-center/human-development-index)
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.

---

## 👤 Informasi Proyek

| Atribut       | Detail |
|--------------|--------|
| **Mata Kuliah** | Statistika dan Probabilitas |
| **Topik**     | Prediksi IPM Kabupaten/Kota Jawa Tengah |
| **Metode**    | Random Forest Regressor |
| **Bahasa**    | Python 3 |

---

> *Program ini dibuat untuk keperluan akademik. Hasil prediksi bersifat estimasi dan tidak menggantikan data resmi BPS.*