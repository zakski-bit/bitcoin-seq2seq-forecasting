# Bitcoin 24-Hour Multi-Horizon Price Forecasting

Multi-horizon time-series forecasting system for Bitcoin (BTC/USD) closing prices, leveraging an autoregressive **Sequence-to-Sequence (Seq2Seq) LSTM** architecture augmented with a custom **Self-Attention** layer and a low-level **Custom Training Loop (`tf.GradientTape`)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dHQX7Zh0oi9HBnhiZxZ1-blRG87GEFjz?usp=sharing)
[![Vercel Live Demo](https://img.shields.io/badge/Vercel-Live%20Demo-black?logo=vercel)](https://bitcoin-seq2seq-forecasting-41sf.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Live Demos

- **Interactive Web Dashboard**: [https://bitcoin-seq2seq-forecasting-41sf.vercel.app](https://bitcoin-seq2seq-forecasting-41sf.vercel.app)
- **Google Colab Research Notebook**: [Open in Google Colab](https://colab.research.google.com/drive/1dHQX7Zh0oi9HBnhiZxZ1-blRG87GEFjz?usp=sharing)

## Problem Formulation & Architecture

Direct multi-step forecasting in high-volatility cryptocurrency markets frequently suffers from rapid error accumulation and gradient degradation. This project addresses drift through a multi-input autoregressive sequence-to-sequence design:

1. **Multivariate Feature Engineering**: Ingests historical 48-hour observation windows across 7 engineered signals: Log Returns, Relative Strength Index (RSI), Average True Range (ATR), Moving Average Convergence Divergence (MACD), and Rolling Volatility statistics, normalized via robust MinMax scaling.
2. **Encoder with Self-Attention**: Encodes temporal dependencies while dynamically weighting high-signal time steps via a custom multi-head scaled dot-product attention block.
3. **Autoregressive Decoder**: Implements Teacher Forcing during optimization and recursive autoregressive inference during evaluation: predicting $t+1$ and re-injecting forecasts into the recurrent state to generate the full 24-hour horizon.

## Empirical Benchmark & Evaluation

Models were evaluated on a chronologically separated test split (final 15% out-of-sample window) with strict safeguards against data leakage:

| Model Architecture | Inference Technique | Test MAE (Normalized) | Mean Absolute Error (USD) | Relative Error Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline LSTM** | Direct Multi-Horizon (Dense 24) | `0.22624` | $\sim \$4,200$ | Baseline |
| **Seq2Seq LSTM + Attention** | **Autoregressive Decoding** | **`0.00661`** | **$\sim \$105$** | **-97.08%** |

### 24-Hour Horizon Trajectory Sample (Test Sample #10)

```text
Horizon Step   Actual Price (USD)   Baseline Prediction   Seq2Seq Prediction   Absolute Deviation
    t+1           $16,507.19             $12,406.84           $16,305.65            $201.54
    t+6           $16,510.38             $12,454.76           $16,386.81            $123.57
    t+12          $16,522.14             $12,649.30           $16,446.96             $75.18
    t+18          $16,625.48             $11,511.23           $16,509.24            $116.24
    t+24          $16,580.31             $11,429.51           $16,589.79              $9.48
```

## Low-Level Implementations

- **`CustomMultiHeadAttention` (Subclassed `tf.keras.layers.Layer`)**: Explicit Query ($Q$), Key ($K$), Value ($V$) projections with scaled dot-product computation:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **`HorizonWeightedMAE` (Custom Loss)**: Time-weighted penalty function scaling loss dynamically across the forecast horizon:
  $$\text{Weight}(t) = 1.0 + \frac{t}{\text{Horizon}}$$
  penalizing deviations at late steps ($t+18$ to $t+24$) to prevent trajectory collapse.
- **Custom Optimization Loop (`tf.GradientTape`)**: Fine-grained gradient computation with $L_2$ gradient norm clipping (`clip_by_norm=1.0`), adaptive learning rate decay, and custom early stopping hooks.

## Repository Structure

```
├── index.html               # Chart.js visualization client for production demo
├── app.js                   # Client-side multi-horizon inference simulation
├── vercel.json              # Edge deployment configuration
├── submission_dltm.ipynb    # Complete research pipeline & model artifacts
└── requirements.txt         # Python runtime dependencies
```

## Quickstart

### Running the Research Notebook

Open [`submission_dltm.ipynb`](submission_dltm.ipynb) locally or run directly on [Google Colab](https://colab.research.google.com/drive/1dHQX7Zh0oi9HBnhiZxZ1-blRG87GEFjz?usp=sharing).

### Local Web Simulator

```bash
# Serve static visualization client locally
python -m http.server 8000
```
Navigate to `http://localhost:8000` to inspect interactive trajectory comparisons.

## License

MIT License. See [LICENSE](LICENSE) for details.
