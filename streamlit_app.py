# dashboard_trading.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ----------------- 1. Interface -----------------
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📈 Dashboard Trading LSTM Delta")

# ----------------- 2. Paramètres utilisateur -----------------
TICKER = st.text_input("Symbole boursier", value="BTC-USD")
WINDOW = st.number_input("Fenêtre LSTM (jours)", value=50, min_value=5)
capital_init = st.number_input("Capital initial", value=10000.0)
risk_factor = st.slider("Risk factor", 1.0, 50.0, 10.0)
max_fraction = st.slider("Fraction max par trade", 0.0, 1.0, 0.2)

START = st.date_input("Date début données", value=pd.to_datetime("2025-01-01"))

# ----------------- 3. Load model & scalers -----------------
@st.cache_resource
def load_models():
    model = load_model("lstm_delta_model_corrected.keras")
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_models()
st.success("Modèle et scalers rechargés ✔")

# ----------------- 4. Download data -----------------
@st.cache_data(ttl=3600)
def download_data(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True)
    df["Close_log"] = np.log(df["Close"])
    return df

data = download_data(TICKER, START)
closes = data["Close_log"].values

if len(closes) <= WINDOW:
    st.warning("Pas assez de données pour cette fenêtre.")
    st.stop()

# ----------------- 5. Build sequences -----------------
n_samples = len(closes) - WINDOW
X_new = np.zeros((n_samples, WINDOW))
y_true = np.zeros((n_samples,))
for i in range(n_samples):
    X_new[i] = closes[i:i+WINDOW]
    y_true[i] = closes[i+WINDOW] - closes[i+WINDOW-1]

X_new_scaled = scaler_X.transform(X_new).reshape(X_new.shape[0], X_new.shape[1], 1)

# ----------------- 6. Predict -----------------
y_pred_scaled = model.predict(X_new_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()  # delta prédit
true_delta = y_true

# Reconstruction prix
base_log_prices = closes[WINDOW-1:-1]
pred_log_prices = base_log_prices + y_pred
true_log_prices = base_log_prices + true_delta

pred_price = np.exp(pred_log_prices)
true_price = np.exp(true_log_prices)

# ----------------- 7. Backtest pondéré -----------------
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
    if score > 0 and cash > 0:
        invest_amount = cash * weight
        qty = invest_amount / current_price
        position += qty
        cash -= invest_amount
        trades.append(("BUY", current_price, qty))

    # SELL
    elif score < 0 and position > 0:
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

    equity_curve.append(cash + position*current_price)

final_value = cash + position*true_price[-1]
total_trades = len([t for t in trades if t[0]=="SELL"])
win_rate = wins / total_trades * 100 if total_trades > 0 else 0
avg_gain = np.mean([t[3] for t in trades if t[0]=="SELL" and t[3]>0]) if wins > 0 else 0
avg_loss = np.mean([t[3] for t in trades if t[0]=="SELL" and t[3]<0]) if losses > 0 else 0
total_return_pct = (final_value - capital_init) / capital_init * 100

# ----------------- 8. Stats -----------------
st.subheader("📊 Résultats Backtest")
st.write(f"Capital final simulé : {final_value:.2f} USD")
st.write(f"Nombre total de trades : {total_trades}")
st.write(f"% de trades gagnants : {win_rate:.2f}%")
st.write(f"Gain moyen par trade gagnant : {avg_gain:.2f}")
st.write(f"Perte moyenne par trade perdant : {avg_loss:.2f}")
st.write(f"Rendement total sur capital : {total_return_pct:.2f}%")

# ----------------- 9. Graphiques -----------------
st.subheader("📈 Evolution prix")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(y=true_price, name="Prix réel"))
fig_price.add_trace(go.Scatter(y=pred_price, name="Prix prédit"))
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("💰 Evolution capital")
fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(y=equity_curve, name="Equity curve"))
st.plotly_chart(fig_equity, use_container_width=True)
