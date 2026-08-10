# Pocket Option OTC High-Frequency Tick Database 📈

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Format](https://img.shields.io/badge/format-Parquet%20%7C%20CSV.GZ-emerald.svg)]()
[![Precision](https://img.shields.io/badge/granularity-Sub--Second%20Ticks-cyan.svg)]()
[![License](https://img.shields.io/badge/license-Educational%20%2F%20Research-amber.svg)]()

A curated, high-frequency tick dataset captured from **Pocket Option OTC** currency pairs, indices, commodities, and crypto streams via low-latency WebSocket hooks.

Engineered specifically for **Quantitative Researchers, Binary Options Algorithmic Traders, and Machine Learning Modelers** who require sub-second empirical price action without broker historical API limitations.

---

## 🚀 Key Features

- **Sub-Second Tick Resolution:** Captures real-time micro-price fluctuations (~500ms arrival rate).
- **Institutional Market Manipulation Detection:** Real-time metrics for **Push & Snap** (liquidity grab velocity spikes with exponential time decay) and **Pinning** (artificial price clustering and stall zones).
- **Gap-Aware Session Segmentation:** Includes an automated `session_id` column that groups continuous trading sessions, preventing cross-gap indicator distortion during backtests.
- **Enriched Indicator Features:** Pre-calculated rolling **Tick Density (tpm)**, **Sigmoid Liquidity %**, **Normalized Volatility Score**, and **Realized Micro-Spread**.
- **Dual Optimized Formats:** Delivered in ultra-fast, compressed **Apache Parquet (`.parquet`)** and universal **Gzip CSV (`.csv.gz`)**.

---

## 📊 Dataset Schema

| Column Name | Type | Description | Example |
|---|---|---|---|
| `timestamp` | `float64` | Unix Epoch timestamp with millisecond precision | `1786268418.794` |
| `datetime_utc` | `string` | Human-readable UTC timestamp (ISO 8601) | `2026-08-09 18:20:18.794` |
| `asset` | `string` | Standardized OTC Pair identifier | `EURUSD_otc` |
| `price` | `float64` | Executed tick price | `1.16885` |
| `direction` | `int8` | `1` (Up/Call), `-1` (Down/Put), `0` (Flat) | `1` |
| `session_id` | `int32` | Continuous session ID (increments on gaps $>60\text{s}$) | `1` |
| `ticks_per_min` | `int32` | Rolling 15-tick instantaneous frequency | `126` |
| `sigmoid_liquidity` | `float32` | Normalized Liquidity Score ($0.0\%–100.0\%$, Midpoint $120\text{ tpm} = 50\%$) | `54.2` |
| `liquidity_level` | `string` | Discrete liquidity classification (`LOW`, `MEDIUM`, `HIGH`) | `MEDIUM` |
| `volatility_score` | `float32` | Normalized return standard deviation ($0.0\%–100.0\%$) | `42.5` |
| `spread_pts` | `float32` | Realized High-Low micro-spread in basis points | `1.45` |
| `push_snap_severity` | `float32` | Push & Snap severity ($0.000$–$1.000$, velocity spike vs 300-tick MAV with $\tau=5\text{s}$ decay) | `0.425` |
| `pinning_severity` | `float32` | Pinning severity ($0.000$–$1.000$, 20-tick clustering within $<0.005\%$ threshold) | `0.000` |
| `manipulation_type` | `string` | Discrete manipulation classification (`NONE`, `PUSH_SNAP`, `PINNING`, `BOTH`) | `PUSH_SNAP` |
| `is_manipulated` | `int8` | Binary manipulation filter flag (`1` if severity $>0.01$, otherwise `0`) | `1` |

---

## ⚡ Quick-Start Python Example

### 1. Installation
```bash
pip install pandas pyarrow
```

### 2. Loading & Analyzing Ticks
```python
import pandas as pd

# Load sample Parquet file in < 100ms
df = pd.read_parquet("data/samples/EURUSD_otc_sample.parquet")

# 1. Inspect Continuous Sessions & Manipulation
for session_id, session_df in df.groupby("session_id"):
    manip_pct = (session_df["is_manipulated"] == 1).mean() * 100.0
    print(f"Session {session_id}: {len(session_df)} ticks | Manipulation: {manip_pct:.1f}%")

# 2. Filter Clean Execution Ticks (Excluding Manipulation)
clean_ticks = df[df["is_manipulated"] == 0]
print(f"Clean Non-Manipulated Ticks: {len(clean_ticks):,} / {len(df):,}")

# 3. Resample Session 1 into 5-Second OHLCV Candles
session_1 = df[df["session_id"] == 1].copy()
session_1["dt"] = pd.to_datetime(session_1["timestamp"], unit="s", utc=True)
session_1.set_index("dt", inplace=True)

candles_5s = session_1["price"].resample("5s").ohlc()
candles_5s["volume_ticks"] = session_1["price"].resample("5s").count()
print(candles_5s.head())
```

---

## 📓 Interactive Jupyter Notebook (`otc_tick_analysis.ipynb`)

We provide a plug-and-play Jupyter Notebook with visual charts and backtests:

1. **Launch Jupyter Lab / Notebook**:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
2. Open [`otc_tick_analysis.ipynb`](otc_tick_analysis.ipynb) to interactively:
   - Resample sub-second ticks into 5-second & 1-minute OHLCV candles.
   - Visualize price charts with tick density volume overlays.
   - Plot Sigmoid Liquidity and Volatility distribution histograms.
   - Run a vectorized 60-second binary options strategy simulation.

---

## 📦 What's in the Full Package?

- **41,804,930 Sub-Second Ticks** across **78 Unique OTC Assets**.
- **Major & Cross Currency Pairs:** `EURUSD` (2.91M ticks), `AUDCAD` (2.35M ticks), `AUDCHF` (2.23M ticks), `AUDUSD` (1.89M ticks), `CADCHF` (1.84M ticks), `EURCHF` (1.81M ticks), `EURGBP` (1.77M ticks), `EURJPY` (1.61M ticks), `GBPUSD` (1.39M ticks), `USDJPY` (1.08M ticks), `GBPJPY` (1.06M ticks).
- **Exotic OTC Pairs:** `USDARS` (610k ticks), `USDBRL` (83k ticks), `MADUSD` (987k ticks), `LBPUSD` (791k ticks), `KESUSD` (665k ticks), `NGNUSD` (479k ticks), `USDMXN` (100k ticks), `USDTRY` (683k ticks), `SARCNY` (119k ticks), `QARCNY` (332k ticks), `AEDCNY` (674k ticks), `ZARUSD` (783k ticks).
- **Crypto OTC:** `BTCUSD` (196k ticks), `SOLUSD` (92k ticks), `BNBUSD` (75k ticks), `DOTUSD` (48k ticks), `LTCUSD` (37k ticks), `ETHUSD` (25k ticks), `DOGE` (17k ticks), `AVAX` (11k ticks).
- **Equities & Commodities OTC:** `AAPL` (203k ticks), `VISA` (166k ticks), `TSLA` (144k ticks), `AMZN` (107k ticks), `MSFT` (59k ticks), `FB` (42k ticks), `VIX` (21k ticks).
- **Complete Manifest (`manifest.json`):** 78 verified assets with continuous session breakdowns, UTC timestamps, rolling liquidity, and zero micro-session noise.

👉 **[Get the Full Multi-Asset Database & Live Updates on Whop](https://whop.com)** *(Insert Your Whop Link Here)*

---

## 🔬 Research & Academic Disclaimer

This dataset is provided for **quantitative research, statistical modeling, algorithm benchmarking, and educational backtesting purposes only**. All data points represent synthetic OTC quote streams intercepted from publicly connected WebSockets. No trading advice or guarantees of future market performance are implied.
