import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

print("Loading data and model for web export...")
CSV_PATH = '../submission_dltm_bitcoin/crypto_hourly.csv'
if not os.path.exists(CSV_PATH):
    CSV_PATH = 'crypto_hourly.csv'

df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Close_Rolling_Mean_24'] = df['Close'].rolling(window=24).mean()
df['Close_Rolling_Std_24'] = df['Close'].rolling(window=24).std()
df = df.dropna().reset_index(drop=True)

features = ['Close', 'Volume USDT', 'RSI', 'MACD_Hist', 'ATR', 'Close_Rolling_Mean_24', 'Close_Rolling_Std_24']
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:].reset_index(drop=True)

feature_scaler = MinMaxScaler()
feature_scaler.fit(train_df[features])
target_scaler = MinMaxScaler()
target_scaler.fit(train_df[['Close']])

LOOKBACK = 48
HORIZON = 24

# Prepare test sequences
test_features_scaled = feature_scaler.transform(test_df[features])
test_close_scaled = target_scaler.transform(test_df[['Close']])

X_test, y_test, dec_test = [], [], []
for i in range(len(test_features_scaled) - LOOKBACK - HORIZON + 1):
    X_test.append(test_features_scaled[i:i+LOOKBACK])
    y_test.append(test_close_scaled[i+LOOKBACK:i+LOOKBACK+HORIZON])
    last_hist_val = test_close_scaled[i+LOOKBACK-1]
    target_hist = test_close_scaled[i+LOOKBACK:i+LOOKBACK+HORIZON-1]
    dec_test.append(np.vstack([last_hist_val, target_hist]))

X_test = np.array(X_test)
y_test = np.array(y_test)
dec_test = np.array(dec_test)

print(f"X_test shape: {X_test.shape}")

# Custom layers definitions needed for loading models
@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.w = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", trainable=True, name="kernel")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": tf.keras.activations.serialize(self.activation)})
        return config

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class CustomLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.gamma = self.add_weight(shape=(dim,), initializer="ones", trainable=True, name="gamma")
        self.beta = self.add_weight(shape=(dim,), initializer="zeros", trainable=True, name="beta")
        super().build(input_shape)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        return self.gamma * ((inputs - mean) / tf.sqrt(variance + self.epsilon)) + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def call(self, inputs, training=None):
        if training and self.rate > 0.0:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.q_proj = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.k_proj = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.v_proj = tf.keras.layers.Dense(self.num_heads * self.key_dim)

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        self.out_proj = tf.keras.layers.Dense(d_model)
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        q = tf.transpose(tf.reshape(q, (batch_size, seq_len, self.num_heads, self.key_dim)), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, (batch_size, seq_len, self.num_heads, self.key_dim)), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, (batch_size, seq_len, self.num_heads, self.key_dim)), [0, 2, 1, 3])
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.key_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, v)
        context = tf.reshape(tf.transpose(context, [0, 2, 1, 3]), (batch_size, seq_len, self.num_heads * self.key_dim))
        return self.out_proj(context)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config

@tf.keras.utils.register_keras_serializable(package="CustomLosses")
class CustomMAE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

# Load models
baseline_model = tf.keras.models.load_model('models/model_baseline_LSTM.keras')
seq2seq_model = tf.keras.models.load_model('models/model_seq2seq_LSTM.keras')

# Reconstruct encoder and decoder step models for autoregressive inference
enc_in = seq2seq_model.get_layer('encoder_input').input
attn_layer = seq2seq_model.get_layer('custom_multi_head_attention_1')
enc_lstm = seq2seq_model.get_layer('encoder_lstm')
attn_out = attn_layer(enc_in)
enc_comb = tf.keras.layers.add([enc_in, attn_out])
_, state_h, state_c = enc_lstm(enc_comb)
encoder_inf = tf.keras.Model(enc_in, [state_h, state_c])

dec_step_in = tf.keras.Input(shape=(1,), name='dec_step_in')
h_in = tf.keras.Input(shape=(64,), name='h_in')
c_in = tf.keras.Input(shape=(64,), name='c_in')
dec_lstm = seq2seq_model.get_layer('decoder_lstm')
out_dense = seq2seq_model.get_layer('output_dense')
lstm_out, [new_h, new_c] = dec_lstm.cell(dec_step_in, states=[h_in, c_in])
step_pred = out_dense(lstm_out)
decoder_step = tf.keras.Model([dec_step_in, h_in, c_in], [step_pred, new_h, new_c])

# Samples to export for live demo
scenarios = [
    {"id": "scenario_1", "name": "Sideways Consolidation (Test Sample #10)", "idx": 10, "tag": "Range-bound"},
    {"id": "scenario_2", "name": "Bullish Trend Run (Test Sample #100)", "idx": 100, "tag": "Bullish Rally"},
    {"id": "scenario_3", "name": "Bearish Price Dip (Test Sample #250)", "idx": 250, "tag": "Correction"}
]

export_data = {
    "project_name": "Bitcoin 24-Hour Multi-Horizon Price Forecasting",
    "course": "Deep Learning Tingkat Mahir (Dicoding)",
    "overall_metrics": {
        "seq2seq_mae_norm": 0.00661,
        "baseline_mae_norm": 0.22624,
        "threshold_mae": 0.015,
        "error_reduction_pct": 97.1,
        "lookback_hours": 48,
        "forecast_hours": 24,
        "features": features
    },
    "scenarios": []
}

for sc in scenarios:
    idx = sc['idx']
    sample_x = X_test[idx:idx+1]
    sample_y = y_test[idx:idx+1]
    sample_dec = dec_test[idx:idx+1]

    # Baseline prediction
    p_base_norm = baseline_model.predict(sample_x, verbose=0)[0]
    
    # Seq2Seq Autoregressive
    h, c = encoder_inf.predict(sample_x, verbose=0)
    curr_in = sample_dec[:, 0, 0:1]
    step_preds = []
    for _ in range(HORIZON):
        pred, h, c = decoder_step.predict([curr_in, h, c], verbose=0)
        step_preds.append(pred[0, 0])
        curr_in = pred
    p_seq_norm = np.array(step_preds).reshape(HORIZON, 1)

    # Inverse transform to USD
    actual_usd = target_scaler.inverse_transform(sample_y[0]).flatten()
    baseline_usd = target_scaler.inverse_transform(p_base_norm).flatten()
    seq2seq_usd = target_scaler.inverse_transform(p_seq_norm).flatten()

    diff_base_usd = np.abs(actual_usd - baseline_usd)
    diff_seq_usd = np.abs(actual_usd - seq2seq_usd)

    # Historical 48h
    hist_raw_usd = target_scaler.inverse_transform(sample_x[0, :, 0:1]).flatten()
    hist_dates = test_df['Date'].iloc[idx:idx+LOOKBACK].dt.strftime('%b %d, %H:00').tolist()
    target_dates = test_df['Date'].iloc[idx+LOOKBACK:idx+LOOKBACK+HORIZON].dt.strftime('%b %d, %H:00').tolist()

    scenario_entry = {
        "id": sc["id"],
        "name": sc["name"],
        "tag": sc["tag"],
        "historical": {
            "dates": hist_dates,
            "prices": np.round(hist_raw_usd, 2).tolist()
        },
        "forecast": {
            "hours": [f"t+{i+1}" for i in range(HORIZON)],
            "dates": target_dates,
            "actual_usd": np.round(actual_usd, 2).tolist(),
            "baseline_usd": np.round(baseline_usd, 2).tolist(),
            "seq2seq_usd": np.round(seq2seq_usd, 2).tolist(),
            "diff_baseline_usd": np.round(diff_base_usd, 2).tolist(),
            "diff_seq2seq_usd": np.round(diff_seq_usd, 2).tolist()
        },
        "metrics": {
            "seq2seq_mae_usd": float(np.round(np.mean(diff_seq_usd), 2)),
            "baseline_mae_usd": float(np.round(np.mean(diff_base_usd), 2)),
            "accuracy_ratio": float(np.round(np.mean(diff_base_usd) / max(np.mean(diff_seq_usd), 1e-4), 1))
        }
    }
    export_data["scenarios"].append(scenario_entry)

os.makedirs('data', exist_ok=True)
with open('data/forecast_data.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2)

print("Export completed! Saved to data/forecast_data.json")
