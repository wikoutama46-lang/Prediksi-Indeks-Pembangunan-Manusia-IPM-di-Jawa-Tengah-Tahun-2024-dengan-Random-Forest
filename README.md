# Prediksi Indeks Pembangunan Manusia (IPM) di Jawa Tengah Tahun 2024 dengan Random Forest

## Deskripsi Project
Project ini merupakan implementasi machine learning menggunakan algoritma Random Forest Regression untuk memprediksi nilai Indeks Pembangunan Manusia (IPM) berdasarkan beberapa indikator pembangunan daerah di Indonesia. Dataset yang digunakan berasal dari data IPM kabupaten/kota tahun 2024, yang kemudian diproses menggunakan Python dan library seperti pandas serta scikit-learn. <br>
Tahapan project meliputi proses pembersihan data, konversi format numerik, transformasi data menggunakan pivot table, pemilihan fitur, pelatihan model, hingga evaluasi performa model menggunakan metrik MAE dan R² Score. Model dilatih menggunakan variabel Usia Harapan Hidup, Harapan Lama Sekolah, Rata-rata Lama Sekolah, dan Pengeluaran per Kapita untuk memprediksi nilai IPM. Selain itu, program juga menyediakan fitur input data secara interaktif melalui terminal sehingga pengguna dapat memasukkan data baru dan memperoleh hasil prediksi IPM secara langsung.

## Tujuan
1. Membangun model prediksi Indeks Pembangunan Manusia (IPM) menggunakan algoritma Random Forest.
2. Menganalisis pengaruh indikator kesehatan, pendidikan, dan ekonomi terhadap nilai IPM.
3. Melakukan pengolahan dan transformasi data IPM kabupaten/kota menjadi format yang siap digunakan untuk machine learning.
4. Mengevaluasi performa model menggunakan metrik MAE dan R² Score.
5. Membuat sistem prediksi IPM sederhana yang dapat menerima input data baru melalui terminal.
6. Memberikan gambaran penerapan machine learning dalam     analisis data pembangunan manusia di Indonesia.
## Dataset
## Dataset Source

Data yang digunakan berasal dari publikasi resmi Badan Pusat Statistik (BPS) terkait Indeks Pembangunan Manusia (IPM) Jawa Tengah Tahun 2024 yang mencakup indikator kesehatan, pendidikan, dan pengeluaran per kapita pada tingkat kabupaten/kota.

Sumber:
BPS Jawa Tengah – Indeks Pembangunan Manusia (IPM) 2024  
https://data.jatengprov.go.id/dataset/6c4-indeks-pembangunan-manusia-ipm-menurut-kabupaten-kota?utm_source=chatgpt.com
### Struktur Dataset

| Kolom | Tipe Data | Deskripsi |
|---|---|---|
| `tahun_data` | Integer | Tahun data IPM |
| `kod_wil` | Integer | Kode wilayah kabupaten/kota |
| `kab_kota` | String | Nama kabupaten/kota |
| `indikator` | String | Jenis indikator pembangunan manusia |
| `indeks` | Float | Nilai indikator |

---

### Contoh Dataset

| tahun_data | kod_wil | kab_kota | indikator | indeks |
|---|---|---|---|---|
| 2024 | 3301 | Kabupaten Cilacap | Usia Harapan Hidup saat Lahir (tahun) | 74.57 |
| 2024 | 3301 | Kabupaten Cilacap | Harapan Lama Sekolah (tahun) | 12.69 |
| 2024 | 3301 | Kabupaten Cilacap | Rata-rata Lama Sekolah (tahun) | 7.40 |
| 2024 | 3301 | Kabupaten Cilacap | Pengeluaran per kapita Disesuaikan | 11868 |
| 2024 | 3301 | Kabupaten Cilacap | Indeks Pembangunan Manusia | 72.38 |
## Workflow Program

## Cara Menjalankan Program

## Algoritma yang Digunakan

## Kesimpulan
