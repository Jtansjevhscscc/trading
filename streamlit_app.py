import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from jinja2 import Environment, FileSystemLoader
from supabase import create_client
import time
import json
import os

st.set_page_config("Trading Dashboard", layout="wide")

def is_cloud():
	return os.getenv("STREAMLIT_RUNTIME") is not None

if is_cloud():
	url = st.secrets["SUPABASE_URL"]
	key = st.secrets["SUPABASE_KEY"]
else:
	from dotenv import load_dotenv

	load_dotenv()
	url = os.getenv("url")
	key = os.getenv("key")

sb = create_client(url, key)
bucket = "data"

supabase = create_client(url, key)

def load_historic():
	try:
		res = sb.storage.from_(bucket).download("historic.json")
		return json.loads(res.decode())
	except Exception:
		return {}

def save_historic(historic):
	data = json.dumps(historic).encode("utf-8")
	sb.storage.from_(bucket).update(
		"historic.json",
		data,
		{"content-type": "application/json"}
	)

ticker = "BTC-USD"
window = 50

# def load_historic():
	# with open("historic.json", "r") as f:
		# historic = json.load(f)
	# return historic

@st.cache_resource
def load_models():
	model = load_model("lstm_delta_model_corrected.keras")
	with open("scaler_X.pkl", "rb") as f:
		scaler_X = pickle.load(f)
	with open("scaler_y.pkl", "rb") as f:
		scaler_y = pickle.load(f)
	return model, scaler_X, scaler_y

def update_historic(new_trade):
	historic[str(day)] = new_trade
	# with open("historic.json", "w", encoding="utf-8") as f:
	#   json.dump(historic, f, indent=4)
	save_historic(historic)





def simulate_actions(portfolio, current_price, predicted, risk_factor=10, max_fraction=0.2, n_samples=20):
	score = (predicted - current_price) / current_price
	weights = np.linspace(0, max_fraction, n_samples)

	results = []

	for w in weights:
		new_line = portfolio.copy()

		if score > 0 and new_line["cash"] > 0:
			invest_amount = new_line["cash"] * w
			qty = invest_amount / current_price
			new_line["position"] += qty
			new_line["cash"] -= invest_amount
			action = f"buy {w:.2f}"

		elif score < 0 and new_line["position"] > 0:
			sell_qty = new_line["position"] * w
			sell_amount = sell_qty * current_price
			new_line["cash"] += sell_amount
			new_line["position"] -= sell_qty
			action = f"sell {w:.2f}"

		else:
			action = f"skip {w:.2f}"

		new_line["capital"] = new_line["cash"] + new_line["position"] * current_price
		new_line["action"] = action
		results.append(new_line)

	idle_line = portfolio.copy()
	idle_line["capital"] = portfolio["cash"] + portfolio["position"] * current_price
	idle_line["action"] = "idle"
	results.append(idle_line)

	return results


def trade(current_price, predicted, risk_factor=10, max_fraction=0.2):
	results = simulate_actions(portfolio, current_price, predicted)

	new_line = max(results, key=lambda x: x["capital"])

	# score = (predicted - current_price) / current_price
	# weight = np.clip(abs(score) * risk_factor, 0, max_fraction)
	# new_line = portfolio.copy()

	# if score > 0 and new_line["cash"] > 0:
	#   invest_amount = new_line["cash"] * weight
	#   qty = invest_amount / current_price
	#   new_line["position"] += qty
	#   new_line["cash"] -= invest_amount
	#   st.markdown("Achete !")

	# elif score < 0 and new_line["position"] > 0:
	#   sell_qty = new_line["position"] * weight
	#   sell_amount = sell_qty * current_price
	#   new_line["cash"] += sell_amount
	#   new_line["position"] -= sell_qty
	#   st.markdown("Vend !")

	# else:
	#   st.markdown("rien ajd")

	# new_line["capital"] = new_line["cash"] + new_line["position"] * current_price
	# new_line["value"] = current_price
	# new_line["passive"] = new_line["init_position"] * current_price + new_line["cash"]

	update_historic(new_line)

def predict_next_value(df):
	closes = df["Close_log"].values
	closes_scaled = scaler_X.transform([closes])
	pred_scaled = model.predict(closes_scaled)
	pred_delta = scaler_y.inverse_transform(pred_scaled).flatten()
	pred_log = closes[-1] + pred_delta
	pred_value = np.exp(pred_log)

	if last_day != day:
		trade(np.exp(closes[-1]), pred_value[0])

# @st.cache_resource
def verify_trad():
	last_trade = historic.get(str(day), False)

	if not last_trade:
		yesterday = day - timedelta(days=1)
		start = yesterday - timedelta(days=window-1)
		
		df = yf.download("BTC-USD", start=start, end=day, auto_adjust=True)
		df["Close_log"] = np.log(df["Close"])
		predict_next_value(df)
		st.rerun()


day = datetime.now(timezone.utc).date()
historic = load_historic()
first_day, first_portfolio = next(iter(historic.items()))
last_day, portfolio = next(reversed(historic.items()))
dates = list(historic.keys())

st.write(f"Aujourd'hui : {day}")
st.write(f"Hier : {day - timedelta(days=1)}")
st.write(f"-window_size : {day - timedelta(days=window)}")

capital = portfolio["capital"]
capitals = [i["capital"] for i in historic.values()]
model, scaler_X, scaler_y = load_models()
verify_trad()

env = Environment(loader=FileSystemLoader("."))
template = env.get_template("template.html")
countdown_tplt = env.get_template("countdown.html")

with open("style.css", "r") as f:
	css = f.read()

html = template.render(capital=capital)
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
st.markdown(html, unsafe_allow_html=True)







def compute_sma_baseline(df):
	cash = first_portfolio["cash"]
	position = first_portfolio["position"]
	caps = []

	df["SMA50"] = df["Close"].rolling(50).mean()
	df["SMA200"] = df["Close"].rolling(200).mean()

	for i in range(len(df)):
		price = df["Close"].iloc[i][0]
		sma50 = df["SMA50"].iloc[i]
		sma200 = df["SMA200"].iloc[i]

		if np.isnan(sma50) or np.isnan(sma200):
			caps.append(cash + position * price)
			continue

		if sma50 > sma200 and cash > 0:
			invest_amount = cash * 0.1
			qty = invest_amount / price
			position += qty
			cash -= invest_amount

		elif sma50 < sma200 and position > 0:
			cash += position * price
			position = 0

		caps.append(cash + position * price)

	return caps

start_date = (datetime.strptime(list(historic.keys())[0], "%Y-%m-%d") - timedelta(days=250))
df_hist = yf.download(ticker, start=start_date, end=str(day), auto_adjust=True)
df_sim = df_hist
df_sim["SMA50"] = df_sim["Close"].rolling(50).mean()
df_sim["SMA200"] = df_sim["Close"].rolling(200).mean()
sma_baseline_capitals = compute_sma_baseline(df_sim)

active = (portfolio["capital"] - first_portfolio["capital"]) / first_portfolio["capital"] * 100
passive = (portfolio["value"] - first_portfolio["value"]) / first_portfolio["value"] * 100
baseline = (sma_baseline_capitals[-1] - first_portfolio["capital"]) / first_portfolio["capital"] * 100
values = [active, passive, baseline]

comparison_plot = go.Figure()

comparison_plot.add_trace(go.Bar(
	x=["Actif", "Passif", "Baseline"],
	y=values,
	text=[f"{v:.2f}%" for v in values],
	textposition="outside",
	marker_color=["green" if v >= 0 else "red" for v in values] 
))

comparison_plot.update_layout(
	title="Comparaison passif/actif/baseline",
	yaxis=dict(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor="black"
	),
	bargap=0.4
)

st.plotly_chart(comparison_plot, use_container_width=True)

st.subheader("Historique du capital")
passives = [i["passive"] for i in historic.values()]

historic_plot = go.Figure()

historic_plot.add_trace(go.Scatter(
	x=dates, y=capitals, 
	name="Capital",
	line=dict(color="royalblue", width=2, dash="solid"),
	mode="lines+markers",
	marker=dict(size=6, symbol="circle", color="royalblue"),
	fill="tozeroy",
	fillcolor="rgba(65,105,225,0.2)",
	showlegend=False
))

historic_plot.add_trace(go.Scatter(
	x=dates, y=passives,
	name="Capital passif",
	line=dict(color="red", width=2, dash="solid"),
	mode="lines+markers",
	marker=dict(size=6, symbol="circle", color="red"),
	showlegend=False
))

historic_plot.add_trace(go.Scatter(
	x=dates, y=sma_baseline_capitals,
	name="Capital baseline",
	line=dict(color="green", width=2, dash="solid"),
	mode="lines+markers",
	marker=dict(size=6, symbol="circle", color="green"),
	showlegend=False
))


st.plotly_chart(historic_plot, use_container_width=True)

st.subheader("Historique des positions")
btc_plot = go.Figure()

btc = [i["position"] for i in historic.values()]

btc_plot.add_trace(go.Scatter(
	x=dates, y=btc, 
	name="BTC",
	line=dict(color="orange", width=3, dash="solid"),
	mode="lines+markers",
	fill="tozeroy",
	marker=dict(size=6, symbol="circle", color="orange"),
	showlegend=False
))

st.plotly_chart(btc_plot, use_container_width=True)



def time_until_midnight_utc():
	now_utc = datetime.now(timezone.utc)
	midnight_utc = datetime(
		now_utc.year, now_utc.month, now_utc.day, 0, 0, 0, tzinfo=timezone.utc
	) + timedelta(days=1)
	return midnight_utc - now_utc

placeholder = st.empty()

while True:
	delta = time_until_midnight_utc()
	secs = int(delta.total_seconds())
	hours, secs = divmod(secs, 3600)
	mins, secs = divmod(secs, 60)
	time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

	countdown = countdown_tplt.render(time=time_str)
	placeholder.markdown(countdown, unsafe_allow_html=True)

	time.sleep(1)





