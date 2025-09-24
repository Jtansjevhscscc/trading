# dashboard_trading.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.title("📈 Mini-dashboard Trading")

# ---------- Paramètres ----------
TICKER = st.selectbox("Choisir un actif :", ["BTC-USD", "AAPL"])
WINDOW = 20
FEATURES = ["Close","SMA50","SMA200","RSI","Boll_Mid","Boll_Std","Boll_Upper","Boll_Lower"]

risk_factor = st.slider("Facteur de risque", 1.0, 20.0, 10.0)
max_fraction = st.slider("Fraction max du capital par trade", 0.01, 1.0, 0.2)

capital_init = st.number_input("Capital initial", 1000.0)

# ---------- Indicators ----------
def compute_indicators(df):
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(100).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Boll_Mid"] = df["Close"].rolling(20).mean()
    df["Boll_Std"] = df["Close"].rolling(20).std()
    df["Boll_Upper"] = df["Boll_Mid"] + 2 * df["Boll_Std"]
    df["Boll_Lower"] = df["Boll_Mid"] - 2 * df["Boll_Std"]
    return df

# ---------- Load model & scalers ----------
@st.cache_resource
def load_model_scalers():
    model = load_model("lstm_delta_model.keras")
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model_scalers()

st.success("✅ Modèle et scalers chargés")

# ---------- Download data ----------
data_load_state = st.text("Chargement des données...")
data = yf.download(TICKER, period="1y", auto_adjust=True)
data["Close"] = np.log(data["Close"])
data = compute_indicators(data)
data = data.dropna(subset=FEATURES)
data_load_state.text("Données chargées ✔")

arr = data[FEATURES].values
closes = data["Close"].values

# ---------- Build sequences ----------
n_samples = len(arr) - WINDOW
X_new = np.zeros((n_samples, WINDOW, arr.shape[1]), dtype=np.float32)
y_true = np.zeros((n_samples,), dtype=np.float32)

for i in range(n_samples):
    X_new[i] = arr[i:i+WINDOW]
    y_true[i] = closes[i+WINDOW]

# Scale
n_features = arr.shape[1]
X_new_scaled = scaler_X.transform(X_new.reshape(-1, n_features)).reshape(X_new.shape)

# ---------- Forecast ----------
y_pred_scaled = model.predict(X_new_scaled, batch_size=32)
pred_delta = scaler_y.inverse_transform(y_pred_scaled).flatten()
pred_log_prices = closes[WINDOW-1: -1] + pred_delta
pred_price = np.exp(pred_log_prices)
true_price = np.exp(y_true)

# ---------- Trading simulation pondérée ----------
capital = capital_init
cash, position = capital, 0.0
equity_curve = []
trades = []
wins, losses = 0, 0

for i in range(len(pred_price)-1):
    current_price = true_price[i]
    predicted = pred_price[i]
    score = (predicted - current_price) / current_price
    weight = np.clip(abs(score) * risk_factor, 0, max_fraction)

    # BUY
    if predicted > current_price and cash > 0:
        invest_amount = cash * weight
        qty = invest_amount / current_price
        position += qty
        cash -= invest_amount
        trades.append(("BUY", current_price, qty))

    # SELL
    elif predicted < current_price and position > 0:
        sell_qty = position * weight
        if sell_qty > 0:
            sell_amount = sell_qty * current_price
            buy_price = trades[-1][1] if trades else current_price
            pnl = (current_price - buy_price) * sell_qty
            cash += sell_amount
            position -= sell_qty
            trades.append(("SELL", current_price, sell_qty, pnl))
            if pnl > 0: wins += 1
            elif pnl < 0: losses += 1

    total_value = cash + position * current_price
    equity_curve.append(total_value)

# ---------- Stats ----------
final_value = cash + position * true_price[-1]
total_trades = len([t for t in trades if t[0]=="SELL"])
win_rate = wins / total_trades * 100 if total_trades > 0 else 0
avg_gain = np.mean([t[3] for t in trades if t[0]=="SELL" and t[3] > 0]) if wins > 0 else 0
avg_loss = np.mean([t[3] for t in trades if t[0]=="SELL" and t[3] < 0]) if losses > 0 else 0
total_return_pct = (final_value - capital_init) / capital_init * 100

st.subheader("📊 Stats Trading")
st.write(f"Capital final : {final_value:.2f} USD")
st.write(f"Nombre de trades : {total_trades}")
st.write(f"% trades gagnants : {win_rate:.2f}%")
st.write(f"Gain moyen : {avg_gain:.2f}, Perte moyenne : {avg_loss:.2f}")
st.write(f"Rendement total : {total_return_pct:.2f}%")

# ---------- Plot equity curve ----------
fig = go.Figure()
fig.add_trace(go.Scatter(y=equity_curve, mode='lines', name='Equity Curve'))
fig.update_layout(title="Évolution du capital", yaxis_title="USD")
st.plotly_chart(fig)

# ---------- Plot prix ----------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=true_price, mode='lines', name='Prix réel'))
fig2.add_trace(go.Scatter(y=pred_price, mode='lines', name='Prix prédit'))
fig2.update_layout(title=f"Prix réel vs prédit ({TICKER})", yaxis_title="USD")
st.plotly_chart(fig2)
