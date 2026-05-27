"""
portwatch.py
------------
Handles everything for IMF Portwatch shipping/port congestion data:
  - pull()             : download raw port + chokepoint data from ArcGIS API
  - clean()            : clean raw data, handle nulls/types
  - extract_features() : compute a port_congestion_score per (date, region)
  - run()              : orchestrates all three steps

Output: data/processed/portwatch_features.csv
Columns: date | region | port_congestion_score

No API key needed — Portwatch is completely free and open.
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_PORTS_PATH = RAW_DIR / "portwatch_ports_raw.csv"
RAW_CHOKE_PATH = RAW_DIR / "portwatch_chokepoints_raw.csv"
PROCESSED_PATH = PROCESSED_DIR / "portwatch_features.csv"

# ─────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────
BASE_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/{dataset}/FeatureServer/0/query"

DATASETS = {
    "ports":       "Daily_Ports_Data",
    "chokepoints": "Daily_Chokepoints_Data",
}

PAGE_SIZE = 1000

PORTS_FIELDS = [
    "date", "portname", "country", "ISO3",
    "portcalls_container", "portcalls_dry_bulk",
    "portcalls_general_cargo", "portcalls_tanker",
    "portcalls_cargo", "portcalls",
    "import", "export",
]


# ─────────────────────────────────────────
# STEP 1: PULL
# ─────────────────────────────────────────
def pull(days: int = 30):
    """
    Pull port and chokepoint data from Portwatch ArcGIS API.

    Args:
        days: how many days of historical data to pull (default 30)
    """
    print(f"[PORTWATCH] Starting pull — last {days} days...")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    date_filter = f"date >= TIMESTAMP '{start_str}' AND date <= TIMESTAMP '{end_str}'"

    # ── Pull ports ──
    print("[PORTWATCH] Pulling ports data...")
    ports_df = _paginate_arcgis(
        dataset=DATASETS["ports"],
        fields=PORTS_FIELDS,
        where=date_filter
    )
    if ports_df.empty:
        print("[PORTWATCH] Date filter returned 0 rows for ports. Pulling all available...")
        ports_df = _paginate_arcgis(
            dataset=DATASETS["ports"],
            fields=PORTS_FIELDS,
            where="1=1"
        )
    ports_df.to_csv(RAW_PORTS_PATH, index=False)
    print(f"[PORTWATCH] Ports saved → {RAW_PORTS_PATH} ({len(ports_df)} rows)")

    # ── Pull chokepoints ──
    # Use outFields=* because chokepoint schema differs from ports
    print("[PORTWATCH] Pulling chokepoints data...")
    choke_df = _paginate_arcgis(
        dataset=DATASETS["chokepoints"],
        fields=["*"],
        where="1=1"
    )
    choke_df.to_csv(RAW_CHOKE_PATH, index=False)
    print(f"[PORTWATCH] Chokepoints saved → {RAW_CHOKE_PATH} ({len(choke_df)} rows)")


def _paginate_arcgis(dataset: str, fields: list, where: str = "1=1") -> pd.DataFrame:
    """
    Fetch all records from an ArcGIS FeatureServer with pagination.
    Portwatch caps at 1000 rows per request — we paginate until done.
    """
    url = BASE_URL.format(dataset=dataset)
    all_records = []
    offset = 0

    while True:
        params = {
            "where":             where,
            "outFields":         ",".join(fields),
            "f":                 "json",
            "resultOffset":      offset,
            "resultRecordCount": PAGE_SIZE,
            "outSR":             "4326",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[PORTWATCH] ERROR fetching offset {offset}: {e}")
            break

        if "error" in data:
            print(f"[PORTWATCH] ArcGIS API error: {data['error']}")
            break

        features = data.get("features", [])
        if not features:
            break

        records = [f["attributes"] for f in features]
        all_records.extend(records)
        print(f"[PORTWATCH]   Fetched {offset + len(records)} records...")

        if len(features) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(0.5)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


# ─────────────────────────────────────────
# STEP 2: CLEAN
# ─────────────────────────────────────────
def clean():
    """
    Load and clean raw Portwatch data.
    Handles empty chokepoints gracefully.
    """
    print("[PORTWATCH] Cleaning raw data...")

    ports = pd.read_csv(RAW_PORTS_PATH)
    print(f"[PORTWATCH] Loaded {len(ports)} port rows")

    try:
        choke = pd.read_csv(RAW_CHOKE_PATH)
        print(f"[PORTWATCH] Loaded {len(choke)} chokepoint rows")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("[PORTWATCH] No chokepoint data, skipping.")
        choke = pd.DataFrame()

    ports = _clean_ports(ports)

    if not choke.empty:
        choke = _clean_chokepoints(choke)
        combined = pd.concat([ports, choke], ignore_index=True)
    else:
        combined = ports

    print(f"[PORTWATCH] Combined: {len(combined)} rows after cleaning")
    return combined


def _clean_ports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the ports dataset.

    Confirmed schema from debug:
      date (string YYYY-MM-DD), portname, country, ISO3,
      portcalls (int), import (int), export (int)
    """
    if df.empty:
        return df

    # ── Parse date — already a clean string like "2026-04-23" ──
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── Use ISO3 as region (confirmed 3-letter code: JPN, USA, CHN) ──
    df.rename(columns={"ISO3": "region", "portname": "port_name"}, inplace=True)
    df["region"] = df["region"].astype(str).str.strip().str.upper()

    # ── Drop rows missing critical fields ──
    df.dropna(subset=["date", "region"], inplace=True)
    df = df[df["region"].ne("nan")]
    df = df[df["region"].ne("")]

    # ── Total port calls — confirmed column name: portcalls ──
    df["total_portcalls"] = pd.to_numeric(df["portcalls"], errors="coerce").fillna(0)

    # ── Trade volume — confirmed columns: import, export ──
    df["import"] = pd.to_numeric(df["import"], errors="coerce").fillna(0)
    df["export"] = pd.to_numeric(df["export"], errors="coerce").fillna(0)
    df["trade_vol"] = df["import"] + df["export"]

    return df[["date", "region", "port_name", "total_portcalls", "trade_vol"]].copy()


def _clean_chokepoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the chokepoints dataset.

    Confirmed schema from debug:
      date (string YYYY-MM-DD), portname, portid,
      n_total, n_container, n_dry_bulk, n_tanker etc.
      No import/export columns.
    """
    if df.empty:
        return df

    # ── Parse date — already a clean string like "2019-01-01" ──
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── Use portname as identifier, prefix with CHOKE_ ──
    df["port_name"] = df["portname"].astype(str)
    df["region"] = "CHOKE_" + df["port_name"].str.upper().str.replace(" ", "_")

    df.dropna(subset=["date"], inplace=True)

    # ── Total vessel transits — chokepoints use n_ prefix (confirmed) ──
    # n_total is the pre-summed total across all vessel types
    if "n_total" in df.columns:
        df["total_portcalls"] = pd.to_numeric(df["n_total"], errors="coerce").fillna(0)
    else:
        # fallback: sum all n_ columns
        n_cols = [c for c in df.columns if c.startswith("n_")]
        df["total_portcalls"] = df[n_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

    # ── No import/export in chokepoints — set to 0 ──
    df["trade_vol"] = 0

    return df[["date", "region", "port_name", "total_portcalls", "trade_vol"]].copy()


# ─────────────────────────────────────────
# STEP 3: EXTRACT FEATURES
# ─────────────────────────────────────────
def extract_features(df: pd.DataFrame = None):
    """
    Aggregate cleaned Portwatch data into one score per (date, region).

    Risk logic:
    - Low portcalls = ships avoiding the port = disruption signal
    - Low trade volume = goods not moving = disruption signal
    - disruption_score = 1 - normalized_activity

    For chokepoints trade_vol is 0, so their score is driven purely
    by vessel transit counts — a drop in transits = high risk.

    Output: data/processed/portwatch_features.csv
    """
    print("[PORTWATCH] Extracting features...")

    if df is None:
        df = clean()

    # ── Log-normalize portcalls and trade ──
    df["portcalls_log"] = np.log1p(df["total_portcalls"])
    df["trade_log"] = np.log1p(df["trade_vol"])

    # ── Aggregate per (date, region) ──
    features = (
        df.groupby(["date", "region"])
        .agg(
            avg_portcalls=("portcalls_log", "mean"),
            avg_trade=("trade_log", "mean"),
            port_count=("port_name", "count"),
        )
        .reset_index()
    )

    # ── Scale to [0, 1] across full dataset ──
    for col in ["avg_portcalls", "avg_trade"]:
        max_val = features[col].max()
        if max_val > 0:
            features[col] = features[col] / max_val

    # ── Disruption proxy: low activity = high risk ──
    features["port_congestion_score"] = (
        0.5 * (1 - features["avg_portcalls"]) +
        0.5 * (1 - features["avg_trade"])
    ).clip(0, 1).round(4)

    features.to_csv(PROCESSED_PATH, index=False)
    print(f"[PORTWATCH] Features saved → {PROCESSED_PATH}")
    print(f"[PORTWATCH] Sample output:\n{features.head(10).to_string()}")

    return features


# ─────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────
def run(days: int = 30):
    """Full pipeline: pull → clean → extract_features"""
    if not RAW_PORTS_PATH.exists():
        pull(days=days)
    else:
        print("[PORTWATCH] Raw files already exist, skipping download.")

    df_clean = clean()
    features = extract_features(df_clean)
    return features


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    features = run(days=30)
    print(f"\n[PORTWATCH] Done. {len(features)} (date, region) pairs produced.")
    print(features.describe())