# AI Supply Chain Disruption Predictor

A machine learning pipeline that predicts supply chain disruptions by continuously monitoring global news, port congestion, and weather signals — giving companies early warning before disruptions hit their operations.

---

## The Problem

Supply chain managers are constantly blindsided. A typhoon shuts down the port of Shanghai. Houthi attacks reroute vessels away from the Red Sea. A trade embargo blocks a critical shipping lane. By the time these events appear in internal reports, the damage is already done — shipments are delayed, production lines are halted, and customers are waiting.

The question this project addresses: **can we predict disruptions before they happen, using publicly available data?**

---

## What Was Built

A fully automated Python pipeline that:

1. **Ingests live data** from three independent global signal sources
2. **Cleans and normalizes** each source into a consistent risk score
3. **Merges** all signals into a unified feature matrix
4. **Trains a machine learning model** to identify disruption patterns
5. **Generates daily risk predictions** per country and maritime chokepoint
6. **Fires Slack alerts** when any region crosses a HIGH or CRITICAL risk threshold

---

## Data Sources

### 1. GDELT Project — News & Geopolitical Risk
GDELT scrapes over 4 million news sources every 15 minutes across 100 languages. Each news event is tagged with a location, event type (protest, conflict, trade embargo, sanctions), a Goldstein destabilization score (-10 to +10), and news sentiment tone.

We filter to supply-chain-relevant CAMEO event codes — protests, sanctions, trade embargoes, armed conflict — and compute a `news_risk_score` per country per day. Higher score = more destabilizing geopolitical environment.

**Why it matters:** Geopolitical events precede port disruptions. Trade sanctions, border closures, and labor protests show up in news days before ships start diverting.

### 2. IMF Portwatch — Port Congestion & Shipping
The IMF and UN jointly maintain Portwatch — a real-time database of vessel activity at 2,065 ports and 28 critical maritime chokepoints (Suez Canal, Strait of Malacca, Bab el-Mandeb, Dover Strait etc.), updated daily.

We track total vessel calls and trade volume per port and compute a `port_congestion_score`. The key insight: **a sudden drop in vessel activity is a stronger disruption signal than a spike** — it means ships are actively avoiding a port or chokepoint.

**Why it matters:** Portwatch gives direct operational signal. When the Bering Strait score spikes, vessels are diverting. When Shanghai portcalls drop, factories aren't getting their components.

### 3. Open-Meteo — Weather Severity
Open-Meteo provides historical and forecast weather data globally from ERA5 reanalysis — no API key required. We query 43 major trade cities and ports, covering the world's largest manufacturing hubs and port cities.

We combine precipitation, wind speed, WMO weather codes, and extreme temperature into a `weather_severity_score`. Aggregation uses the maximum across all cities in a country — because a single hurricane at one port disrupts the entire country's trade flow regardless of what the weather is like elsewhere.

**Why it matters:** Extreme weather is the number one cause of supply chain disruption globally. Typhoon Hagibis (Japan, 2019), Hurricane Ida (US Gulf, 2021), and the 2022 Pakistan floods all caused multi-week port closures.

---

## Technical Architecture

```
Data Layer                Processing Layer           Output Layer
──────────────            ─────────────────          ────────────
GDELT API      →  gdelt.py      → news_risk_score    ┐
Portwatch API  →  portwatch.py  → port_congestion    ├→ main.py → feature_matrix.csv
Open-Meteo API →  weather.py    → weather_severity   ┘
                                                           ↓
                                                      label.py → labeled_matrix.csv
                                                           ↓
                                                      model/train.py → model.pkl
                                                           ↓
                                                      model/predict.py → predictions.csv
                                                           ↓
                                                      alerts/slack.py → Slack alert
```

Each data source has its own self-contained script with three functions:
- `pull()` — fetch raw data from API, save to `data/raw/`
- `clean()` — normalize, filter, handle nulls
- `extract_features()` — produce a `(date, region, score)` output

This modular design means adding a new data source only requires writing one new file.

---

## Machine Learning Model

**Algorithm:** XGBoost Classifier

XGBoost was chosen for its strong performance on tabular data, built-in handling of class imbalance, fast training, and native SHAP support for explainability.

**Training approach:**
- 80/20 chronological split — test set uses the most recent data, simulating real deployment
- `scale_pos_weight` adjusted for class imbalance (22% disruption rate)
- No feature scaling needed — XGBoost is tree-based and scale-invariant

**Results on held-out test set:**

| Metric | Score |
|---|---|
| ROC-AUC | 0.9994 |
| Recall (disruptions) | 94% |
| Precision (disruptions) | 90% |
| Accuracy | 99% |

94% recall means the model catches 94 out of every 100 real disruptions — only 6 are missed. For a risk tool, recall matters more than precision: a false alarm is an inconvenience, a missed disruption is a crisis.

**SHAP Feature Importance:**

| Feature | Importance |
|---|---|
| port_congestion_score | 4.54 (dominant) |
| news_risk_score | 2.93 |
| weather_severity_score | 1.04 |

Port congestion is the strongest predictor — when ships start backing up, a disruption is already underway or imminent. News risk is the second strongest — geopolitical tension precedes operational disruption. Weather is weakest in this 30-day dataset but would strengthen significantly with multi-year historical data.

---

## Risk Level Classification

Predictions are bucketed into four actionable levels:

| Level | Probability | Recommended Action |
|---|---|---|
| 🚨 CRITICAL | ≥ 85% | Activate contingency plan immediately |
| ⚠️ HIGH | ≥ 65% | Monitor closely, alert procurement team |
| 🟡 MEDIUM | ≥ 40% | Flag for review in daily standup |
| 🟢 LOW | < 40% | Normal operations |

---

## Sample Alert (Slack)

```
🌐 Supply Chain Risk Alert — 2026-05-27

1 region(s) flagged as HIGH or CRITICAL risk
─────────────────────────────────────────
⚠️ USA — HIGH (probability: 78%)
🌪️ Weather severity: 0.28
─────────────────────────────────────────
Generated by Supply Chain Disruption Predictor • 2026-05-27 15:47 UTC
```

---

## Project Structure

```
supply_chain_predictor/
│
├── data/
│   ├── raw/                        # raw API responses
│   └── processed/                  # cleaned features, predictions
│
├── gdelt.py                        # news & geopolitical signal
├── portwatch.py                    # shipping & port congestion signal
├── weather.py                      # weather severity signal
├── main.py                         # merges all features
├── label.py                        # generates disruption labels
│
├── model/
│   ├── train.py                    # XGBoost training + SHAP
│   ├── predict.py                  # generates daily predictions
│   └── model.pkl                   # saved trained model
│
├── alerts/
│   └── slack.py                    # Slack webhook alerts
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull and process data
python gdelt.py
python portwatch.py
python weather.py

# 3. Merge and label
python main.py
python label.py

# 4. Train model
python model/train.py

# 5. Predict and alert
python model/predict.py
python alerts/slack.py
```

---

## Honest Limitations of This MVP

Being transparent about limitations is as important as the results themselves.

**Rule-based labels:** Disruption labels were generated using threshold rules on the same features used for training. This explains the near-perfect AUC of 0.9994 — the model is learning the rules back rather than discovering independent patterns. In production, labels would come from verified disruption databases such as Lloyd's List, Resilinc EventWatchAI, or ACLED conflict data, producing a more realistic AUC of 0.75–0.85.

**30-day data window:** All three sources were pulled for only 30 days due to API and download constraints. A production system would use 3–5 years of historical data, giving the model exposure to major disruption cycles — COVID port closures, Suez Canal blockage, Red Sea crisis — rather than a quiet recent period.

**43-city weather coverage:** Weather is currently sampled from 43 cities. Many countries and all maritime chokepoints receive median-filled weather scores. Full coverage would require integrating NOAA storm track data and expanding the city list significantly.

**No real-time streaming:** The current pipeline runs as a batch job. A production system would use Apache Kafka for real-time event streaming from GDELT's 15-minute update feed, with Apache Airflow orchestrating scheduled pulls from Portwatch and weather APIs.

---

## What a Production System Would Look Like

This project was built as a proof of concept. Scaling it to production would involve:

**Richer data sources:**
- Satellite AIS vessel tracking for real-time ship positions and diversions
- Freight rate indices (Drewry WCI, Baltic Dry Index) as leading indicators
- Social media monitoring for early protest and labor dispute signals
- Customs and trade flow data from UN Comtrade

**Better labels:**
- Integration with commercial disruption databases (Resilinc, Everstream Analytics)
- Semi-supervised learning using small verified label sets + large unlabeled data
- Human-in-the-loop review of model predictions to continuously improve labels

**Infrastructure:**
- Apache Kafka for real-time GDELT streaming
- Apache Airflow for pipeline orchestration and scheduling
- PostgreSQL + TimescaleDB for time-series storage
- MLflow for model versioning and experiment tracking
- Kubernetes for containerized deployment
- Grafana dashboards for operational monitoring

**Model improvements:**
- Time-series features — rolling averages, rate of change, lagged signals
- Anomaly detection layer — flag deviations from each port's historical baseline
- Claude API integration — LLM-based reasoning layer to summarize and contextualize high-risk alerts in plain English for non-technical supply chain managers

---

## Why This Matters for Supply Chain Analysts

Modern supply chain analysis is shifting from reactive to predictive. The tools exist — GDELT, Portwatch, satellite AIS, freight indices — and they are largely free. The gap is in knowing how to combine them, clean them, and extract signal from noise.

This project demonstrates that gap can be closed with Python, domain knowledge, and a structured ML pipeline. The same approach scales directly to:

- **Supplier risk scoring** — replace country-level aggregation with supplier-level data
- **Inventory optimization** — feed disruption probabilities into safety stock models
- **Route planning** — use chokepoint risk scores to recommend alternative shipping routes
- **Procurement strategy** — identify high-risk sourcing geographies before contracts are signed

The goal is not to replace supply chain analysts — it is to give them a 72-hour early warning system so decisions are made on signal, not surprise.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| ML model | XGBoost |
| Explainability | SHAP |
| HTTP requests | Requests |
| Alerting | Slack Webhooks (Block Kit) |
| Data sources | GDELT, IMF Portwatch, Open-Meteo |

---

*Built as part of a portfolio project exploring the intersection of data engineering, machine learning, and global supply chain risk management.*
