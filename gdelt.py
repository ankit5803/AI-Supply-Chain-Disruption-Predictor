"""
gdelt.py
--------
Handles everything for GDELT news/geopolitical data:
  - pull()             : download raw GDELT files (historical or live)
  - clean()            : clean raw data, handle nulls/types
  - extract_features() : compute a news_risk_score per (date, region)
  - run(mode)          : orchestrates all three steps

Output: data/processed/gdelt_features.csv
Columns: date | region | news_risk_score
"""

import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = RAW_DIR / "gdelt_raw.csv"
PROCESSED_PATH = PROCESSED_DIR / "gdelt_features.csv"

# ─────────────────────────────────────────
# GDELT COLUMN NAMES (58 columns, no header in file)
# Source: https://github.com/linwoodc3/gdelt2HeaderRows
# ─────────────────────────────────────────
GDELT_COLUMNS = [
    "GlobalEventID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
    "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat",
    "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type",
    "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"
]

# ─────────────────────────────────────────
# SUPPLY CHAIN RELEVANT CAMEO EVENT CODES
# These event types are most likely to signal disruptions
# Full list: https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt
# ─────────────────────────────────────────
RELEVANT_EVENT_CODES = [
    "14",   # Protest / demonstrate
    "145",  # Protest violently
    "17",   # Coerce
    "18",   # Assault
    "19",   # Fight
    "20",   # Use conventional mass violence
    "112",  # Criticize or denounce (trade disputes)
    "1122", # Accuse of human rights abuses
    "172",  # Impose embargo / boycott / sanctions
    "1721", # Impose embargo on trade
    "173",  # Halt negotiations
    "174",  # Halt mediation
    "175",  # Break relations
]


# ─────────────────────────────────────────
# STEP 1: PULL
# ─────────────────────────────────────────
def pull(mode: str = "historical", start_date: str = None, end_date: str = None, days: int = 365):
    """
    Download GDELT 1.0 event files and save to data/raw/gdelt_raw.csv.

    Args:
        mode       : "historical" (date range) or "live" (latest 15-min file)
        start_date : "YYYY-MM-DD" — used in historical mode
        end_date   : "YYYY-MM-DD" — used in historical mode (defaults to today)
        days       : how many days back from today if start_date not given (default 365)
    """
    print(f"[GDELT] Starting pull in '{mode}' mode...")

    if mode == "live":
        _pull_live()
    elif mode == "historical":
        if start_date is None:
            # Default: go back `days` days from today
            end_dt = datetime.today()
            start_dt = end_dt - timedelta(days=days)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()

        _pull_historical(start_dt, end_dt)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'historical' or 'live'.")


def _pull_live():
    """Pull the most recent 15-minute GDELT 2.0 update."""
    # This file always points to the latest update
    master_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    print("[GDELT] Fetching latest update index...")

    response = requests.get(master_url, timeout=30)
    response.raise_for_status()

    # The file has 3 lines; first line contains the events CSV zip URL
    # Format: "<size> <md5> <url>"
    first_line = response.text.strip().split("\n")[0]
    zip_url = first_line.split(" ")[2]

    print(f"[GDELT] Downloading live file: {zip_url}")
    df = _download_and_parse_zip(zip_url, source="gdelt2")
    df.to_csv(RAW_PATH, index=False)
    print(f"[GDELT] Live data saved → {RAW_PATH} ({len(df)} rows)")


def _pull_historical(start_dt: datetime, end_dt: datetime):
    """
    Download GDELT 1.0 daily files for a date range.
    Files are at: http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
    """
    base_url = "http://data.gdeltproject.org/events/{date}.export.CSV.zip"
    all_frames = []

    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        url = base_url.format(date=date_str)

        try:
            print(f"[GDELT] Downloading {date_str}...")
            df = _download_and_parse_zip(url, source="gdelt1")
            all_frames.append(df)
        except Exception as e:
            print(f"[GDELT] WARNING: Could not download {date_str} — {e}")

        current += timedelta(days=1)

    if not all_frames:
        raise RuntimeError("[GDELT] No data downloaded. Check your date range or internet connection.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(RAW_PATH, index=False)
    print(f"[GDELT] Historical data saved → {RAW_PATH} ({len(combined)} rows)")


def _download_and_parse_zip(url: str, source: str = "gdelt1") -> pd.DataFrame:
    """
    Download a zip from GDELT, extract the CSV inside, return as DataFrame.

    Args:
        url    : full URL to the .zip file
        source : "gdelt1" (58 cols, tab-delimited) or "gdelt2" (same + extra cols)
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Unzip in memory — no need to write zip to disk
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]  # always one file inside
        with z.open(csv_filename) as f:
            # GDELT files are tab-delimited despite the .csv extension
            # No header row — we supply column names manually
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=GDELT_COLUMNS[:],  # GDELT 2.0 may have extra cols, truncate
                on_bad_lines="skip",     # skip malformed rows
                low_memory=False
            )

    return df


# ─────────────────────────────────────────
# STEP 2: CLEAN
# ─────────────────────────────────────────
def clean():
    """
    Load raw GDELT data and clean it:
    - Keep only columns relevant to supply chain analysis
    - Parse dates
    - Drop nulls in critical columns
    - Filter to supply-chain-relevant event codes
    - Normalize GoldsteinScale and AvgTone to [0, 1]
    """
    print("[GDELT] Cleaning raw data...")

    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"[GDELT] Loaded {len(df)} raw rows")

    # ── Keep only the columns we actually need ──
    cols_needed = [
        "SQLDATE",
        "Actor1CountryCode",   # where the event happened
        "EventCode",               # type of event (CAMEO code)
        "GoldsteinScale",          # destabilization score (-10 to +10)
        "AvgTone",                 # news sentiment (-100 to +100)
        "NumMentions",             # how widely reported
        "NumArticles",
    ]
    df = df[cols_needed].copy()

    # ── Parse date ──
    # SQLDATE is YYYYMMDD integer format
    df["date"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["date"] >= pd.Timestamp.today() - pd.Timedelta(days=35)]
    df.drop(columns=["SQLDATE"], inplace=True)

    # ── Rename for clarity ──
    df.rename(columns={"Actor1CountryCode": "region"}, inplace=True)

    # ── Drop rows missing critical fields ──
    df.dropna(subset=["date", "region", "GoldsteinScale", "AvgTone"], inplace=True)

    # ── Drop rows with empty/unknown regions ──
    # Cast to string first — column may contain mixed types (floats/NaN from parsing)
    df["region"] = df["region"].astype(str).str.strip()
    df = df[df["region"].ne("")]
    df = df[df["region"].ne("nan")]

    # ── Convert EventCode to string for filtering ──
    df["EventCode"] = df["EventCode"].astype(str).str.strip()

    # ── Filter: keep only supply-chain-relevant events ──
    # We keep rows where the EventCode STARTS WITH any of our relevant codes
    # e.g. "172" matches "1721", "1722" etc.
    def is_relevant(code):
        return any(code.startswith(c) for c in RELEVANT_EVENT_CODES)

    mask = df["EventCode"].apply(is_relevant)
    df = df[mask].copy()
    print(f"[GDELT] After event code filter: {len(df)} rows")

    # ── Normalize NumMentions ──
    # More mentions = more significant event
    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(1)
    df["NumMentions"] = np.log1p(df["NumMentions"])  # log scale to reduce outlier effect

    # ── Normalize GoldsteinScale from [-10, +10] to [0, 1] ──
    # Lower Goldstein = more destabilizing → higher risk
    # So we INVERT: risk = (10 - GoldsteinScale) / 20
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce").fillna(0)
    df["goldstein_risk"] = (10 - df["GoldsteinScale"]) / 20  # 0 = stable, 1 = very destabilizing

    # ── Normalize AvgTone to [0, 1] ──
    # More negative tone = higher risk
    # AvgTone range is roughly [-100, +100]
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce").fillna(0)
    df["tone_risk"] = (-df["AvgTone"] + 100) / 200  # 0 = very positive, 1 = very negative
    df["tone_risk"] = df["tone_risk"].clip(0, 1)     # safety clip

    print(f"[GDELT] Clean complete. {len(df)} rows retained.")
    return df


# ─────────────────────────────────────────
# STEP 3: EXTRACT FEATURES
# ─────────────────────────────────────────
def extract_features(df: pd.DataFrame = None):
    """
    Aggregate cleaned GDELT rows into one risk score per (date, region).

    Risk score formula:
        news_risk_score = 0.5 * goldstein_risk + 0.3 * tone_risk + 0.2 * mention_weight

    Output: data/processed/gdelt_features.csv
    Columns: date | region | news_risk_score
    """
    print("[GDELT] Extracting features...")

    if df is None:
        # If called standalone, reload clean data
        df = clean()

    # ── Mention weight: normalize per group so high-volume days don't dominate ──
    df["mention_weight"] = df.groupby(["date", "region"])["NumMentions"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )

    # ── Composite risk score per row ──
    df["row_risk"] = (
        0.5 * df["goldstein_risk"] +
        0.3 * df["tone_risk"] +
        0.2 * df["mention_weight"]
    )

    # ── Aggregate: mean risk per (date, region) ──
    features = (
        df.groupby(["date", "region"])
        .agg(
            news_risk_score=("row_risk", "mean"),
            event_count=("row_risk", "count"),        # how many events that day/region
            avg_tone=("AvgTone", "mean"),              # keep for interpretability
            avg_goldstein=("GoldsteinScale", "mean"),  # keep for interpretability
        )
        .reset_index()
    )

    # ── Final clip to [0, 1] ──
    features["news_risk_score"] = features["news_risk_score"].clip(0, 1).round(4)

    features.to_csv(PROCESSED_PATH, index=False)
    print(f"[GDELT] Features saved → {PROCESSED_PATH}")
    print(f"[GDELT] Sample output:\n{features.head(10).to_string()}")

    return features


# ─────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────
def run(mode: str = "historical", start_date: str = None, end_date: str = None, days: int = 365):
    """
    Full pipeline: pull → clean → extract_features

    Args:
        mode       : "historical" or "live"
        start_date : "YYYY-MM-DD" (historical mode)
        end_date   : "YYYY-MM-DD" (historical mode, defaults to today)
        days       : days back from today if start_date not given
    """
    pull(mode=mode, start_date=start_date, end_date=end_date, days=days)
    df_clean = clean()
    features = extract_features(df_clean)
    return features


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Example: pull last 30 days of historical data
    features = run(mode="historical", days=30)
    print(f"\n[GDELT] Done. {len(features)} (date, region) pairs produced.")
    print(features.describe())