"""
=============================================================
  PREDIKSI IPM - RANDOM FOREST REGRESSOR
  Data: Kabupaten/Kota Jawa Tengah
  Target: Indeks Pembangunan Manusia (IPM)
=============================================================
Cara pakai:
  1. Install library:
       pip install pandas scikit-learn
  2. Jalankan: python prediksi_ipm_rf.py
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
FILE_PATH = r"C:\Statistika dan Probabilitas - Project Random Forest\Prediksi Indeks Pembangunan Manusia (IPM) dengan Random Forest\Data\indeks-pembangunan-manusia-ipm-menurut-kabupatenkota-tahun-2024.csv"
SEPARATOR = ";"


# ─────────────────────────────────────────────
# UTILITAS TAMPILAN
# ─────────────────────────────────────────────
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def garis(char="=", lebar=65):
    print(char * lebar)


def header(judul, char="=", lebar=65):
    garis(char, lebar)
    padding = (lebar - len(judul) - 2) // 2
    print(
        f"{char}{' ' * padding}{judul}{' ' * (lebar - padding - len(judul) - 2)}{char}"
    )
    garis(char, lebar)


def input_angka(prompt, min_val=None, max_val=None):
    while True:
        try:
            raw = input(prompt).strip()
            if raw.lower() in ("keluar", "exit", "q"):
                return None
            val = float(raw.replace(",", "."))
            if min_val is not None and val < min_val:
                print(f"  [!] Nilai minimal adalah {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Nilai maksimal adalah {max_val}")
                continue
            return val
        except ValueError:
            print("  [!] Input tidak valid. Masukkan angka (contoh: 72.5 atau 72,5)")


def tampilkan_progress(label, persen, lebar=30):
    isi = int(persen * lebar)
    kosong = lebar - isi
    bar = "█" * isi + "░" * kosong
    print(f"  {label:<15} [{bar}] {persen*100:.1f}%")


# ─────────────────────────────────────────────
# 1. LOAD & VALIDASI DATA
# ─────────────────────────────────────────────
def load_data():
    clear()
    header("PREDIKSI IPM — RANDOM FOREST")
    print()

    if not os.path.exists(FILE_PATH):
        print(f"  [!] File tidak ditemukan:")
        print(f"      {FILE_PATH}")
        print()
        print("  Pastikan path file sudah benar lalu jalankan ulang program.")
        sys.exit(1)

    print(f"  [+] Memuat data dari:")
    print(f"      {FILE_PATH}")
    print()

    df_raw = pd.read_csv(FILE_PATH, sep=SEPARATOR)
    df_raw.columns = df_raw.columns.str.strip().str.lower()
    df_raw["indeks"] = df_raw["indeks"].astype(str).str.replace(",", ".").astype(float)

    print(f"  [+] Data dimuat      : {len(df_raw)} baris")
    print(f"  [+] Kabupaten/Kota   : {df_raw['kab_kota'].nunique()}")
    print(f"  [+] Tahun tersedia   : {sorted(df_raw['tahun_data'].unique())}")

    df = df_raw.pivot_table(
        index=["tahun_data", "kod_wil", "kab_kota"],
        columns="indikator",
        values="indeks",
    ).reset_index()
    df.columns.name = None

    rename_map = {
        "Usia Harapan Hidup saat Lahir (tahun)": "UHH",
        "Harapan Lama Sekolah (tahun)": "HLS",
        "Rata-rata Lama Sekolah (tahun)": "RLS",
        "Pengeluaran per kapita Disesuaikan (ribu rupiah/orang/tahun)": "Pengeluaran",
        "Indeks Pembangunan Manusia": "IPM",
    }
    df.rename(columns=rename_map, inplace=True)

    FEATURES = ["UHH", "HLS", "RLS", "Pengeluaran"]
    TARGET = "IPM"

    df.dropna(subset=FEATURES + [TARGET], inplace=True)
    print(f"  [+] Data siap        : {len(df)} observasi")

    return df, FEATURES, TARGET


# ─────────────────────────────────────────────
# 2. TRAINING & EVALUASI
# ─────────────────────────────────────────────
def train_model(df, FEATURES, TARGET):
    print()
    print("  [*] Melatih model Random Forest...")

    X = df[FEATURES].values
    y = df[TARGET].values

    rf = RandomForestRegressor(
        n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1
    )

    if len(df) <= 50:
        cv = LeaveOneOut()
        cv_name = "Leave-One-Out CV"
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_name = "5-Fold Cross Validation"

    y_pred_cv = cross_val_predict(rf, X, y, cv=cv)
    rf.fit(X, y)

    r2 = r2_score(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    mae = mean_absolute_error(y, y_pred_cv)

    print(f"  [+] Selesai. Metode validasi: {cv_name}")
    print()

    return rf, y_pred_cv, r2, rmse, mae, cv_name


# ─────────────────────────────────────────────
# MENU UTAMA
# ─────────────────────────────────────────────
def menu_utama():
    print()
    garis("-")
    print("  MENU UTAMA")
    garis("-")
    print("  [1] Lihat hasil evaluasi model")
    print("  [2] Lihat feature importance")
    print("  [3] Lihat prediksi per kabupaten/kota")
    print("  [4] Simulasi prediksi IPM baru")
    print("  [0] Keluar")
    garis("-")
    return input("  Pilih menu: ").strip()


# ─────────────────────────────────────────────
# MENU 1: EVALUASI MODEL
# ─────────────────────────────────────────────
def menu_evaluasi(r2, rmse, mae, cv_name):
    clear()
    header(f"EVALUASI MODEL ({cv_name})")
    print()

    label_r2 = (
        "Sangat baik ✓" if r2 > 0.95 else ("Baik" if r2 > 0.85 else "Perlu perhatian")
    )

    print(f"  {'Metrik':<12} {'Nilai':>10}   Keterangan")
    garis("-")
    print(f"  {'R²':<12} {r2:>10.4f}   {label_r2}")
    print(f"  {'RMSE':<12} {rmse:>10.4f}   Akar rata-rata kuadrat error")
    print(f"  {'MAE':<12} {mae:>10.4f}   Rata-rata absolut error")
    garis("-")

    if r2 > 0.95:
        print()
        print("  Model memiliki performa sangat baik (R² > 0.95).")
        print("  Prediksi sangat akurat terhadap data aktual.")
    elif r2 > 0.85:
        print()
        print("  Model memiliki performa baik (R² > 0.85).")
    else:
        print()
        print("  Model mungkin perlu penyesuaian hyperparameter.")

    print()
    input("  Tekan Enter untuk kembali ke menu...")


# ─────────────────────────────────────────────
# MENU 2: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def menu_importance(rf, FEATURES):
    clear()
    header("FEATURE IMPORTANCE")
    print()

    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(
        ascending=False
    )
    total = importances.sum()

    print(f"  {'Indikator':<20} {'Bobot':>7}   {'Kontribusi':<32} Peringkat")
    garis("-")
    for rank, (feat, val) in enumerate(importances.items(), 1):
        bar_len = int((val / total) * 35)
        bar = "█" * bar_len + "░" * (35 - bar_len)
        print(f"  {feat:<20} {val:>7.4f}   {bar}  #{rank}")
    garis("-")
    print()
    print("  Interpretasi:")
    top = importances.idxmax()
    print(f"  • Indikator paling berpengaruh : {top}")
    print(
        f"  • Kontribusinya                : {importances[top]*100:.1f}% dari total prediksi"
    )

    print()
    input("  Tekan Enter untuk kembali ke menu...")


# ─────────────────────────────────────────────
# MENU 3: PREDIKSI PER DAERAH
# ─────────────────────────────────────────────
def menu_detail(df, y_pred_cv, TARGET):
    clear()
    header("PREDIKSI PER KABUPATEN/KOTA")
    print()

    df_result = df[["tahun_data", "kab_kota", TARGET]].copy()
    df_result["IPM_Prediksi"] = np.round(y_pred_cv, 2)
    df_result["Residual"] = np.round(df_result[TARGET] - df_result["IPM_Prediksi"], 2)
    df_result["APE(%)"] = np.round(
        np.abs(df_result["Residual"] / df_result[TARGET]) * 100, 2
    )

    # Opsi filter
    tahun_list = sorted(df_result["tahun_data"].unique())
    print("  Tahun tersedia:", ", ".join(str(t) for t in tahun_list))
    print("  Ketik tahun untuk filter, atau Enter untuk semua tahun:")
    filter_input = input("  Tahun: ").strip()

    if filter_input.isdigit() and int(filter_input) in tahun_list:
        df_result = df_result[df_result["tahun_data"] == int(filter_input)]
        print(f"\n  Filter: Tahun {filter_input}")

    print()
    print(
        f"  {'No':<4} {'Daerah':<28} {'Tahun':>5} {'Aktual':>8} {'Prediksi':>9} {'Residual':>9} {'APE%':>6}"
    )
    garis("-")

    for i, (_, row) in enumerate(df_result.iterrows(), 1):
        flag = " ⚠" if abs(row["Residual"]) > 1.5 else ""
        print(
            f"  {i:<4} {row['kab_kota']:<28} {int(row['tahun_data']):>5} "
            f"{row[TARGET]:>8.2f} {row['IPM_Prediksi']:>9.2f} "
            f"{row['Residual']:>9.2f} {row['APE(%)']:>6.2f}{flag}"
        )

    garis("-")
    print(
        f"  Rata-rata APE: {df_result['APE(%)'].mean():.2f}%   "
        f"  Max residual: {df_result['Residual'].abs().max():.2f}"
    )
    print()
    print("  ⚠ = residual > 1.5 (perlu perhatian)")
    print()
    input("  Tekan Enter untuk kembali ke menu...")


# ─────────────────────────────────────────────
# MENU 4: SIMULASI PREDIKSI
# ─────────────────────────────────────────────
def menu_simulasi(rf, df, TARGET, FEATURES):
    while True:
        clear()
        header("SIMULASI PREDIKSI IPM")
        print()
        print("  Masukkan nilai indikator di bawah ini.")
        print("  Ketik 'keluar' untuk kembali ke menu utama.")
        print()

        # Tampilkan rentang data referensi
        print(f"  {'Indikator':<15} {'Min':>8} {'Rata-rata':>10} {'Max':>8}")
        garis("-", 50)
        for feat in FEATURES:
            mn = df[feat].min()
            avg = df[feat].mean()
            mx = df[feat].max()
            print(f"  {feat:<15} {mn:>8.2f} {avg:>10.2f} {mx:>8.2f}")
        garis("-", 50)
        print()

        uhh = input_angka("  UHH  - Usia Harapan Hidup (tahun)           : ", 40, 90)
        if uhh is None:
            break

        hls = input_angka("  HLS  - Harapan Lama Sekolah (tahun)         : ", 5, 25)
        if hls is None:
            break

        rls = input_angka("  RLS  - Rata-rata Lama Sekolah (tahun)       : ", 0, 20)
        if rls is None:
            break

        pnk = input_angka(
            "  Pengeluaran per kapita (ribu Rp/orang/tahun): ", 1000, 50000
        )
        if pnk is None:
            break

        # Prediksi
        simulasi = np.array([[uhh, hls, rls, pnk]])
        prediksi = rf.predict(simulasi)[0]

        # Kategori
        if prediksi >= 80:
            kategori = "TINGGI"
            ket = "(≥ 80)"
        elif prediksi >= 70:
            kategori = "SEDANG"
            ket = "(70 – 79)"
        else:
            kategori = "RENDAH"
            ket = "(< 70)"

        # Daerah referensi terdekat
        df["_selisih"] = np.abs(df[TARGET] - prediksi)
        ref = df.loc[df["_selisih"].idxmin()]
        df.drop(columns="_selisih", inplace=True)

        # Daerah 3 teratas terdekat
        df["_selisih"] = np.abs(df[TARGET] - prediksi)
        top3 = df.nsmallest(3, "_selisih")[["kab_kota", TARGET, "tahun_data"]]
        df.drop(columns="_selisih", inplace=True)

        # Tampilkan hasil
        print()
        garis("─", 55)
        print(f"  HASIL PREDIKSI")
        garis("─", 55)
        print(f"  Prediksi IPM  : {prediksi:.2f}")
        print(f"  Kategori      : {kategori} {ket}")
        garis("─", 55)
        print()
        print(f"  Input yang digunakan:")
        print(f"    UHH          = {uhh:.2f} tahun")
        print(f"    HLS          = {hls:.2f} tahun")
        print(f"    RLS          = {rls:.2f} tahun")
        print(f"    Pengeluaran  = {pnk:,.0f} ribu Rp/orang/tahun")
        print()
        print(f"  3 Daerah dengan IPM paling mirip:")
        print(f"  {'Daerah':<28} {'Tahun':>5} {'IPM Aktual':>11} {'Selisih':>8}")
        garis("-", 55)
        for _, r in top3.iterrows():
            selisih = abs(r[TARGET] - prediksi)
            print(
                f"  {r['kab_kota']:<28} {int(r['tahun_data']):>5} {r[TARGET]:>11.2f} {selisih:>8.2f}"
            )
        garis("-", 55)

        print()
        ulang = input("  Simulasi lagi? (y/n): ").strip().lower()
        if ulang != "y":
            break


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Load dan train
    df, FEATURES, TARGET = load_data()
    rf, y_pred_cv, r2, rmse, mae, cv_name = train_model(df, FEATURES, TARGET)

    print()
    print("  [✓] Model siap digunakan.")
    input("  Tekan Enter untuk masuk ke menu utama...")

    while True:
        clear()
        header("PREDIKSI IPM — RANDOM FOREST")
        pilihan = menu_utama()

        if pilihan == "1":
            menu_evaluasi(r2, rmse, mae, cv_name)
        elif pilihan == "2":
            menu_importance(rf, FEATURES)
        elif pilihan == "3":
            menu_detail(df, y_pred_cv, TARGET)
        elif pilihan == "4":
            menu_simulasi(rf, df, TARGET, FEATURES)
        elif pilihan == "0":
            clear()
            print()
            print("  Terima kasih. Program selesai.")
            print()
            break
        else:
            print("  [!] Pilihan tidak valid.")
            import time

            time.sleep(1)


if __name__ == "__main__":
    main()
