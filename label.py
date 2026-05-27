"""
label.py
--------
Adds a binary disruption label to the feature matrix using rule-based labeling.

Rule:
    disruption = 1 if:
        news_risk_score > 0.75 (high geopolitical tension)
        OR port_congestion_score > 0.70 (severe port congestion)
        OR weather_severity_score > 0.25 (significant weather event)

    disruption = 0 otherwise

Input:  data/processed/feature_matrix.csv
Output: data/processed/labeled_feature_matrix.csv
"""

import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
PROCESSED_DIR   = Path("data/processed")
INPUT_PATH      = PROCESSED_DIR / "feature_matrix.csv"
OUTPUT_PATH     = PROCESSED_DIR / "labeled_feature_matrix.csv"

# ─────────────────────────────────────────
# THRESHOLDS
# Tune these if your disruption rate looks off
# Target: 10-20% disruption rate (too high = too noisy, too low = too rare)
# ─────────────────────────────────────────
NEWS_THRESHOLD    = 0.75
PORT_THRESHOLD    = 0.70
WEATHER_THRESHOLD = 0.25


def label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based disruption labels.

    Why OR not AND?
    - Any single severe signal is enough to indicate disruption risk
    - A hurricane (high weather) alone shuts ports even if news is quiet
    - A trade embargo (high news) is a disruption even in calm weather
    - AND would be too strict — very few rows would get labeled 1
    """
    df["disruption"] = (
        (df["news_risk_score"]        > NEWS_THRESHOLD)    |
        (df["port_congestion_score"]  > PORT_THRESHOLD)    |
        (df["weather_severity_score"] > WEATHER_THRESHOLD)
    ).astype(int)

    return df


def run():
    print("[LABEL] Loading feature matrix...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    print(f"[LABEL] Loaded {len(df)} rows")

    df = label(df)

    # ── Stats ──
    total       = len(df)
    disruptions = df["disruption"].sum()
    rate        = disruptions / total * 100

    print(f"\n[LABEL] Labeling complete:")
    print(f"        Total rows:    {total}")
    print(f"        Disruptions:   {disruptions} ({rate:.1f}%)")
    print(f"        Normal:        {total - disruptions} ({100 - rate:.1f}%)")

    # Warn if label distribution looks off
    if rate < 5:
        print("[LABEL] WARNING: Disruption rate very low (<5%) — consider lowering thresholds")
    elif rate > 40:
        print("[LABEL] WARNING: Disruption rate very high (>40%) — consider raising thresholds")
    else:
        print("[LABEL] Label distribution looks healthy ✓")

    # ── Breakdown by signal ──
    print(f"\n[LABEL] Breakdown — what triggered disruption=1:")
    print(f"        High news only:    {((df['news_risk_score'] > NEWS_THRESHOLD) & (df['port_congestion_score'] <= PORT_THRESHOLD) & (df['weather_severity_score'] <= WEATHER_THRESHOLD)).sum()}")
    print(f"        High port only:    {((df['port_congestion_score'] > PORT_THRESHOLD) & (df['news_risk_score'] <= NEWS_THRESHOLD) & (df['weather_severity_score'] <= WEATHER_THRESHOLD)).sum()}")
    print(f"        High weather only: {((df['weather_severity_score'] > WEATHER_THRESHOLD) & (df['news_risk_score'] <= NEWS_THRESHOLD) & (df['port_congestion_score'] <= PORT_THRESHOLD)).sum()}")
    print(f"        Multiple signals:  {((df['disruption'] == 1) & ((df['news_risk_score'] > NEWS_THRESHOLD).astype(int) + (df['port_congestion_score'] > PORT_THRESHOLD).astype(int) + (df['weather_severity_score'] > WEATHER_THRESHOLD).astype(int) > 1)).sum()}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[LABEL] Saved → {OUTPUT_PATH}")
    print(f"\n[LABEL] Sample output:\n{df.head(10).to_string()}")

    return df


if __name__ == "__main__":
    run()