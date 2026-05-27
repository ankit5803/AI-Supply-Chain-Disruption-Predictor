"""
main.py
-------
Merges all three feature files into one unified feature matrix.

Input files:
  - data/processed/gdelt_features.csv
  - data/processed/portwatch_features.csv
  - data/processed/weather_features.csv

Output:
  - data/processed/feature_matrix.csv

Columns:
  date | region | news_risk_score | port_congestion_score | weather_severity_score

This is the input to the ML model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")

GDELT_PATH     = PROCESSED_DIR / "gdelt_features.csv"
PORTWATCH_PATH = PROCESSED_DIR / "portwatch_features.csv"
WEATHER_PATH   = PROCESSED_DIR / "weather_features.csv"
OUTPUT_PATH    = PROCESSED_DIR / "feature_matrix.csv"


# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
def load_features():
    """
    Load all three feature files.
    Keep only the columns needed for merging and modeling.
    """
    print("[MAIN] Loading feature files...")

    gdelt = pd.read_csv(GDELT_PATH, parse_dates=["date"])
    gdelt = gdelt[["date", "region", "news_risk_score"]]
    print(f"[MAIN] GDELT:     {len(gdelt)} rows | {gdelt['region'].nunique()} regions | {gdelt['date'].min().date()} to {gdelt['date'].max().date()}")

    portwatch = pd.read_csv(PORTWATCH_PATH, parse_dates=["date"])
    portwatch = portwatch[["date", "region", "port_congestion_score"]]
    print(f"[MAIN] Portwatch: {len(portwatch)} rows | {portwatch['region'].nunique()} regions | {portwatch['date'].min().date()} to {portwatch['date'].max().date()}")

    weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    weather = weather[["date", "region", "weather_severity_score"]]
    print(f"[MAIN] Weather:   {len(weather)} rows | {weather['region'].nunique()} regions | {weather['date'].min().date()} to {weather['date'].max().date()}")

    return gdelt, portwatch, weather


# ─────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────
def merge_features(gdelt, portwatch, weather):
    """
    Merge all three DataFrames on (date, region).

    Uses outer join so no data is lost — if a country appears in
    portwatch but not in GDELT on a given date, we keep that row
    and fill the missing score with the dataset median.

    Why median not 0?
    - Filling with 0 implies "no risk" which is misleading
    - Filling with median implies "average/unknown risk" which is honest
    - 0 would bias the model toward thinking missing = safe
    """
    print("\n[MAIN] Merging features...")

    # ── Outer merge GDELT + Portwatch ──
    merged = pd.merge(gdelt, portwatch, on=["date", "region"], how="outer")
    print(f"[MAIN] After GDELT + Portwatch merge: {len(merged)} rows")

    # ── Outer merge + Weather ──
    merged = pd.merge(merged, weather, on=["date", "region"], how="outer")
    print(f"[MAIN] After + Weather merge: {len(merged)} rows")

    return merged


# ─────────────────────────────────────────
# HANDLE MISSING VALUES
# ─────────────────────────────────────────
def handle_missing(df):
    """
    Fill missing scores after merge.

    Three strategies depending on why data is missing:

    1. news_risk_score missing:
       Country appears in portwatch/weather but not in GDELT that day.
       Fill with median — "we don't know, assume average risk"

    2. port_congestion_score missing:
       Some landlocked countries have no port data.
       Fill with 0 — no port = no port risk, 0 is accurate here.

    3. weather_severity_score missing:
       Country not in our 43-city list.
       Fill with median — "we don't know, assume average risk"
    """
    print("\n[MAIN] Handling missing values...")

    # Show missing counts before
    print("[MAIN] Missing before fill:")
    print(df[["news_risk_score", "port_congestion_score", "weather_severity_score"]].isnull().sum().to_string())

    # ── Fill strategies ──
    news_median    = df["news_risk_score"].median()
    weather_median = df["weather_severity_score"].median()

    df["news_risk_score"]        = df["news_risk_score"].fillna(news_median)
    df["port_congestion_score"]  = df["port_congestion_score"].fillna(0)
    df["weather_severity_score"] = df["weather_severity_score"].fillna(weather_median)

    # Show missing counts after
    print("[MAIN] Missing after fill:")
    print(df[["news_risk_score", "port_congestion_score", "weather_severity_score"]].isnull().sum().to_string())

    return df


# ─────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────
def validate(df):
    """
    Run basic sanity checks on the merged feature matrix.
    Warns if anything looks wrong — doesn't crash, just informs.
    """
    print("\n[MAIN] Validating feature matrix...")

    checks_passed = True

    # Check all scores are in [0, 1]
    for col in ["news_risk_score", "port_congestion_score", "weather_severity_score"]:
        out_of_range = df[(df[col] < 0) | (df[col] > 1)]
        if len(out_of_range) > 0:
            print(f"[MAIN] WARNING: {col} has {len(out_of_range)} values outside [0, 1]")
            checks_passed = False

    # Check no remaining nulls
    null_counts = df[["news_risk_score", "port_congestion_score", "weather_severity_score"]].isnull().sum()
    if null_counts.sum() > 0:
        print(f"[MAIN] WARNING: Nulls still present:\n{null_counts}")
        checks_passed = False

    # Check date column is valid
    if df["date"].isnull().sum() > 0:
        print(f"[MAIN] WARNING: {df['date'].isnull().sum()} null dates")
        checks_passed = False

    # Check region column has no empty values
    empty_regions = df[df["region"].astype(str).str.strip().eq("")]
    if len(empty_regions) > 0:
        print(f"[MAIN] WARNING: {len(empty_regions)} empty region values")
        checks_passed = False

    if checks_passed:
        print("[MAIN] All checks passed ✓")

    return checks_passed


# ─────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────
def run():
    """
    Full pipeline:
    load → align dates → merge → handle missing → validate → save
    """

    # ── Load ──
    gdelt, portwatch, weather = load_features()

    # ── Align date ranges — trim everything to GDELT start date ──
    # GDELT only covers ~30 days. Trimming portwatch and weather to the
    # same window ensures we're not training on mostly median-filled rows.
    cutoff = gdelt["date"].min()
    portwatch = portwatch[portwatch["date"] >= cutoff]
    weather   = weather[weather["date"] >= cutoff]
    print(f"\n[MAIN] Date cutoff applied: {cutoff.date()} — all datasets trimmed to match GDELT range")

    # ── Merge ──
    merged = merge_features(gdelt, portwatch, weather)

    # ── Handle missing ──
    merged = handle_missing(merged)

    # ── Sort for cleanliness ──
    merged = merged.sort_values(["date", "region"]).reset_index(drop=True)

    # ── Validate ──
    validate(merged)

    # ── Save ──
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[MAIN] Feature matrix saved → {OUTPUT_PATH}")
    print(f"[MAIN] Shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
    print(f"[MAIN] Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"[MAIN] Unique regions: {merged['region'].nunique()}")
    print(f"\n[MAIN] Sample output:\n{merged.head(10).to_string()}")
    print(f"\n[MAIN] Summary stats:\n{merged[['news_risk_score', 'port_congestion_score', 'weather_severity_score']].describe().to_string()}")

    return merged


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    feature_matrix = run()