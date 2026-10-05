# Bitcoin 24-Hour Multi-Horizon Price Forecasting with Custom Seq2Seq LSTM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zakski-bit/bitcoin-seq2seq-forecasting/blob/main/bitcoin_seq2seq_forecasting.ipynb)
[![Vercel Live Demo](https://img.shields.io/badge/Vercel-Live%20Demo-black?logo=vercel)](https://bitcoin-seq2seq-forecasting-41sf.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0%2B-red.svg)](https://keras.io/)
[![Dicoding Rating](https://img.shields.io/badge/Dicoding-Bintang%205%20(Advanced)-success.svg)](https://www.dicoding.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Proyek Akhir Deep Learning Tingkat Mahir (Dicoding Indonesia)**  
> Membangun sistem peramalan harga Bitcoin (*Close Price*) multi-step 24 jam ke depan menggunakan arsitektur **Sequence-to-Sequence (Seq2Seq) LSTM**, mekanisme **Self-Attention kustom**, **Custom Training Loop (`tf.GradientTape`)**, dan **Autoregressive Decoding**.

---

## 🚀 Live Demo & Interactive Showcase

* **Live Web App (Vercel)**: 👉 [**https://bitcoin-seq2seq-forecasting-41sf.vercel.app**](https://bitcoin-seq2seq-forecasting-41sf.vercel.app)
* **Interactive Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zakski-bit/bitcoin-seq2seq-forecasting/blob/main/bitcoin_seq2seq_forecasting.ipynb)

---

## 📌 Ringkasan Masalah & Solusi Arsitektur

Peramalan multi-step (*multi-horizon forecasting*) pada aset kripto dengan volatilitas tinggi sering kali mengalami **degradasi error yang cepat** jika menggunakan model *direct output* sederhana. 

Proyek ini mengatasi masalah tersebut dengan menerapkan pendekatan **Sequence-to-Sequence Autoregressive**:
1. **Encoder**: Memproses deret waktu 48 jam historis dengan 7 fitur multivariate, diintegrasikan dengan layer `CustomMultiHeadAttention` untuk membobot relevansi sinyal teknikal (RSI, ATR, MACD, dan Rolling Statistics).
2. **Decoder**: Menggunakan *Teacher Forcing* saat fase pelatihan dan *Autoregressive Decoding* saat fase pengujian. Pada setiap langkah waktu $t$, model menghasilkan prediksi $1$ jam, lalu menjadikannya input untuk memprediksi langkah $t+1$.

---

## 📊 Benchmark & Hasil Evaluasi Model

Evaluasi dilakukan pada dataset uji independen secara kronologis (15% data terbelakang) tanpa adanya kebocoran data (*data leakage*):

| Model | Arsitektur | Teknik Inferensi | Test MAE (Normalized) | Rata-rata Error (USD) | Status Kriteria Dicoding |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **LSTM Baseline** | Single-layer LSTM + Dense | Direct Forecasting (24 steps) | `0.22624` | $\sim \$4,200$ | Baseline |
| **Seq2Seq LSTM (Custom)** | Encoder-Decoder + MHA | **Autoregressive Decoding** | **`0.00661`** | **$\sim \$105$** | **Lulus Level Advanced (< 0.015)** |

> **Peningkatan Performa:** Model Seq2Seq Autoregressive menghasilkan **penurunan error sebesar 97.08%** dibandingkan model baseline LSTM pada jendela pengujian yang sama.

### Sampel Evaluasi Prediksi 24 Jam (Sample Test #10)

```text
Jam ke   Data Aktual (USD)   Prediksi Baseline (USD)   Prediksi Seq2Seq (USD)   Selisih Seq2Seq (USD)
  t+1       $16,507.19             $12,406.84                $16,305.65               $201.54
  t+6       $16,510.38             $12,454.76                $16,386.81               $123.57
 t+12       $16,522.14             $12,649.30                $16,446.96                $75.18
 t+18       $16,625.48             $11,511.23                $16,509.24               $116.24
 t+24       $16,580.31             $11,429.51                $16,589.79                 $9.48
```

---

## 🛠️ Kustomisasi Tingkat Rendah (Low-Level Implementations)

1. **`CustomMultiHeadAttention` (Subclassing `tf.keras.layers.Layer`)**
   - Mengimplementasikan proyeksi linear Query ($Q$), Key ($K$), Value ($V$).
   - Menghitung matriks *Scaled Dot-Product Attention*: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
2. **`CustomLayerNormalization` & `CustomDropout`**
   - Menghitung rata-rata dan varians tensor dinamis untuk stabilisasi representasi representasi internal encoder.
3. **`CustomDense`**
   - Inisialisasi bobot manual Glorot Uniform dan bias zeros dengan `add_weight`.
4. **`HorizonWeightedMAE` (Custom Loss Function)**
   - Memberikan bobot penalti bertingkat: $\text{Weight}(t) = 1.0 + \frac{t}{\text{Horizon}}$, sehingga penyimpangan pada jam-jam terjauh ($t+18$ s/d $t+24$) dihukum lebih berat untuk meminimalkan *drift*.
5. **Custom Training Loop (`tf.GradientTape`)**
   - Menghitung gradien secara eksplisit dengan *gradient clipping* (`tf.clip_by_norm(g, 1.0)`).
   - Dilengkapi callback kustom `CustomEarlyStopping` dan `CustomReduceLROnPlateau`.

---

## 📁 Struktur Repositori

```text
bitcoin-seq2seq-forecasting/
├── index.html               # Frontend dashboard interaktif untuk Vercel
├── app.js                   # Logika visualisasi Chart.js & simulasi multi-skenario
├── vercel.json              # Konfigurasi zero-config deployment Vercel
├── data/
│   └── forecast_data.json   # Dataset hasil inferensi untuk demo web
├── models/
│   ├── best_model_seq2seq_LSTM.keras  # Model Seq2Seq Subclassing (GradientTape)
│   ├── model_seq2seq_LSTM.keras       # Model Seq2Seq Functional API
│   └── model_baseline_LSTM.keras      # Model Baseline LSTM
├── notebooks/
│   └── bitcoin_seq2seq_forecasting.ipynb  # Notebook lengkap + output eksekusi
├── src/
│   └── generate_final_submission.py       # Pipeline end-to-end training & eval
├── requirements.txt         # Dependensi Python
└── README.md                # Dokumentasi proyek
```

---

## 💻 Panduan Menjalankan Proyek

### 1. Menjalankan di Google Colab (Paling Mudah)
Cukup klik tombol berikut untuk langsung membuka notebook di Google Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zakski-bit/bitcoin-seq2seq-forecasting/blob/main/notebooks/bitcoin_seq2seq_forecasting.ipynb)

*Seluruh sel sudah terisi output (grafik, heatmap korelasi, dekomposisi tren, log training tape, dan tabel).*

### 2. Menjalankan di Komputer Lokal
```bash
# 1. Clone repositori
git clone https://github.com/zakski-bit/bitcoin-seq2seq-forecasting.git
cd bitcoin-seq2seq-forecasting

# 2. Buat virtual environment & install dependensi
python -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Jalankan pipeline pelatihan dan evaluasi
python src/generate_final_submission.py
```

### 3. Deploy Live Demo ke Vercel (Gratis & 1 Menit Selesai)
1. Buka [vercel.com](https://vercel.com) dan login dengan akun GitHub Anda.
2. Klik tombol **"Add New..."** &rarr; **"Project"**.
3. Pilih repositori `zakski-bit/bitcoin-seq2seq-forecasting`.
4. Pada bagian *Framework Preset*, pilih **Other** (otomatis mendeteksi static web).
5. Klik **Deploy**!
6. Web dashboard Anda akan langsung online dengan domain gratis seperti `bitcoin-seq2seq-forecasting.vercel.app`.

---

## 👤 Penulis

* **Nama:** Zaki Abdussalam
* **GitHub:** [@zakski-bit](https://github.com/zakski-bit)
* **Kelas:** Deep Learning Tingkat Mahir — Dicoding Indonesia
