"""
weather.py
----------
Handles everything for Open-Meteo historical weather data:
  - pull()             : fetch weather data for major trade cities/ports
  - clean()            : clean and normalize raw weather data
  - extract_features() : compute weather_severity_score per (date, region)
  - run()              : orchestrates all three steps

Output: data/processed/weather_features.csv
Columns: date | region | weather_severity_score

No API key needed — Open-Meteo is completely free.

Why weather matters for supply chains:
  - Hurricanes / typhoons shut down ports
  - Floods block roads and rail lines
  - Extreme cold freezes infrastructure
  - Heavy precipitation delays cargo handling
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = RAW_DIR / "weather_raw.csv"
PROCESSED_PATH = PROCESSED_DIR / "weather_features.csv"

# ─────────────────────────────────────────
# API CONFIG
# Historical weather archive endpoint — free, no auth
# ─────────────────────────────────────────
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily variables we care about for supply chain disruption
# Each one represents a different type of disruption risk
DAILY_VARIABLES = [
    "precipitation_sum",        # mm — flooding, port closures
    "windspeed_10m_max",        # km/h — storm conditions
    "weathercode",              # WMO code — categorical event type
    "temperature_2m_max",       # °C — extreme heat stress
    "temperature_2m_min",       # °C — extreme cold / freezing
]

# ─────────────────────────────────────────
# MAJOR TRADE CITIES
# Selected for: port importance, manufacturing hub, supply chain relevance
# Covers top trading nations and critical chokepoint regions
# Format: (city_name, ISO3_country_code, latitude, longitude)
# ─────────────────────────────────────────
TRADE_CITIES = [
    # East Asia — manufacturing core
    ("Shanghai",        "CHN",  31.2304,  121.4737),
    ("Shenzhen",        "CHN",  22.5431,  114.0579),
    ("Guangzhou",       "CHN",  23.1291,  113.2644),
    ("Ningbo",          "CHN",  29.8683,  121.5440),
    ("Tianjin",         "CHN",  39.3434,  117.3616),
    ("Hong Kong",       "HKG",  22.3193,  114.1694),
    ("Tokyo",           "JPN",  35.6762,  139.6503),
    ("Osaka",           "JPN",  34.6937,  135.5023),
    ("Busan",           "KOR",  35.1796,  129.0756),
    ("Kaohsiung",       "TWN",  22.6273,  120.3014),

    # Southeast Asia — electronics and textiles
    ("Singapore",       "SGP",   1.3521,  103.8198),
    ("Port Klang",      "MYS",   3.0000,  101.4000),
    ("Jakarta",         "IDN",  -6.2088,  106.8456),
    ("Ho Chi Minh",     "VNM",  10.8231,  106.6297),
    ("Bangkok",         "THA",  13.7563,  100.5018),
    ("Manila",          "PHL",  14.5995,  120.9842),

    # South Asia
    ("Mumbai",          "IND",  19.0760,   72.8777),
    ("Chennai",         "IND",  13.0827,   80.2707),
    ("Colombo",         "LKA",   6.9271,   79.8612),
    ("Karachi",         "PAK",  24.8607,   67.0011),

    # Middle East — oil and transit
    ("Dubai",           "ARE",  25.2048,   55.2708),
    ("Jeddah",          "SAU",  21.2854,   39.2376),

    # Europe — major trade ports
    ("Rotterdam",       "NLD",  51.9225,    4.4792),
    ("Hamburg",         "DEU",  53.5753,    9.9929),
    ("Antwerp",         "BEL",  51.2194,    4.4025),
    ("Felixstowe",      "GBR",  51.9630,    1.3514),
    ("Barcelona",       "ESP",  41.3851,    2.1734),
    ("Genoa",           "ITA",  44.4056,    8.9463),
    ("Piraeus",         "GRC",  37.9428,   23.6460),

    # North America
    ("Los Angeles",     "USA",  34.0522, -118.2437),
    ("New York",        "USA",  40.7128,  -74.0060),
    ("Houston",         "USA",  29.7604,  -95.3698),
    ("Savannah",        "USA",  32.0835,  -81.0998),
    ("Vancouver",       "CAN",  49.2827, -123.1207),
    ("Toronto",         "CAN",  43.6532,  -79.3832),

    # South America
    ("Santos",          "BRA", -23.9608,  -46.3336),
    ("Buenos Aires",    "ARG", -34.6037,  -58.3816),

    # Africa — chokepoint adjacent
    ("Cairo",           "EGY",  30.0444,   31.2357),
    ("Durban",          "ZAF", -29.8587,   31.0218),
    ("Lagos",           "NGA",   6.5244,    3.3792),

    # Chokepoint region cities
    ("Aden",            "YEM",  12.7797,   45.0095),  # Bab el-Mandeb
    ("Djibouti City",   "DJI",  11.5720,   43.1450),  # Bab el-Mandeb
    ("Colombo",         "LKA",   6.9271,   79.8612),  # Indian Ocean route
    ("Muscat",          "OMN",  23.5880,   58.3829),  # Strait of Hormuz
]

# Remove duplicate cities (Colombo appears twice)
seen = set()
TRADE_CITIES_UNIQUE = []
for city in TRADE_CITIES:
    if city[0] not in seen:
        seen.add(city[0])
        TRADE_CITIES_UNIQUE.append(city)


# ─────────────────────────────────────────
# WMO WEATHER CODE SEVERITY MAPPING
# WMO codes describe weather conditions categorically
# We map them to a severity score [0, 1] for supply chain risk
# Full reference: https://open-meteo.com/en/docs (weathercode section)
# ─────────────────────────────────────────
def wmo_to_severity(code) -> float:
    """
    Convert WMO weather code to a supply chain severity score [0, 1].

    WMO codes:
      0        = Clear sky           → 0.0
      1-3      = Partly cloudy       → 0.0
      45, 48   = Fog                 → 0.3  (visibility risk)
      51-57    = Drizzle             → 0.2
      61-67    = Rain                → 0.4
      71-77    = Snow                → 0.5
      80-82    = Rain showers        → 0.5
      85-86    = Snow showers        → 0.6
      95       = Thunderstorm        → 0.7
      96, 99   = Severe thunderstorm → 0.9
    """
    try:
        code = int(code)
    except (ValueError, TypeError):
        return 0.0

    if code == 0:
        return 0.0
    elif code <= 3:
        return 0.0
    elif code in (45, 48):
        return 0.3
    elif 51 <= code <= 57:
        return 0.2
    elif 61 <= code <= 67:
        return 0.4
    elif 71 <= code <= 77:
        return 0.5
    elif 80 <= code <= 82:
        return 0.5
    elif code in (85, 86):
        return 0.6
    elif code == 95:
        return 0.7
    elif code in (96, 99):
        return 0.9
    else:
        return 0.1  # unknown code — small baseline


# ─────────────────────────────────────────
# STEP 1: PULL
# ─────────────────────────────────────────
def pull(days: int = 30):
    """
    Fetch historical weather for all trade cities from Open-Meteo.
    Makes one API call per city — ~43 calls total.
    Each call returns daily weather variables for the date range.

    Args:
        days: how many days of historical data to pull (default 30)
    """
    print(f"[WEATHER] Starting pull — last {days} days, {len(TRADE_CITIES_UNIQUE)} cities...")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    all_records = []

    for i, (city, iso3, lat, lon) in enumerate(TRADE_CITIES_UNIQUE):
        print(f"[WEATHER]   ({i+1}/{len(TRADE_CITIES_UNIQUE)}) {city}, {iso3}...")

        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start_str,
            "end_date":   end_str,
            "daily":      ",".join(DAILY_VARIABLES),
            "timezone":   "UTC",
        }

        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[WEATHER]   WARNING: Failed for {city} — {e}")
            continue

        # ── Parse response ──
        # Open-Meteo returns {"daily": {"time": [...], "precipitation_sum": [...], ...}}
        daily = data.get("daily", {})
        dates = daily.get("time", [])

        if not dates:
            print(f"[WEATHER]   WARNING: No data returned for {city}")
            continue

        # Build one row per day for this city
        for j, date in enumerate(dates):
            record = {
                "date":    date,
                "city":    city,
                "region":  iso3,
                "lat":     lat,
                "lon":     lon,
            }
            # Add each weather variable — use None if missing
            for var in DAILY_VARIABLES:
                values = daily.get(var, [])
                record[var] = values[j] if j < len(values) else None

            all_records.append(record)

        # Be polite — Open-Meteo is free, don't hammer it
        time.sleep(0.2)

    if not all_records:
        raise RuntimeError("[WEATHER] No data downloaded. Check your internet connection.")

    df = pd.DataFrame(all_records)
    df.to_csv(RAW_PATH, index=False)
    print(f"[WEATHER] Raw data saved → {RAW_PATH} ({len(df)} rows)")


# ─────────────────────────────────────────
# STEP 2: CLEAN
# ─────────────────────────────────────────
def clean():
    """
    Load and clean raw weather data:
    - Parse dates
    - Handle nulls
    - Normalize each weather variable to [0, 1]
    - Convert WMO codes to severity scores
    """
    print("[WEATHER] Cleaning raw data...")

    df = pd.read_csv(RAW_PATH)
    print(f"[WEATHER] Loaded {len(df)} raw rows")

    # ── Parse date ──
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", "region"], inplace=True)

    # ── Clean region ──
    df["region"] = df["region"].astype(str).str.strip().str.upper()
    df = df[df["region"].ne("nan")]

    # ── Normalize precipitation ──
    # Cap at 100mm/day — anything above is extreme flooding (score = 1.0)
    # Normal rainy day ~5-10mm, heavy rain ~25mm, extreme ~50mm+
    df["precipitation_sum"] = pd.to_numeric(df["precipitation_sum"], errors="coerce").fillna(0)
    df["precip_norm"] = (df["precipitation_sum"] / 100).clip(0, 1)

    # ── Normalize wind speed ──
    # Cap at 120 km/h — hurricane force wind (score = 1.0)
    # Normal: 10-20 km/h, Strong: 40-60 km/h, Storm: 80km/h+, Hurricane: 120km/h+
    df["windspeed_10m_max"] = pd.to_numeric(df["windspeed_10m_max"], errors="coerce").fillna(0)
    df["wind_norm"] = (df["windspeed_10m_max"] / 120).clip(0, 1)

    # ── WMO weather code → severity score ──
    df["weathercode"] = pd.to_numeric(df["weathercode"], errors="coerce").fillna(0)
    df["wmo_severity"] = df["weathercode"].apply(wmo_to_severity)

    # ── Extreme temperature risk ──
    # High heat: above 40°C is dangerous for port workers / equipment
    # Extreme cold: below -20°C freezes infrastructure
    df["temperature_2m_max"] = pd.to_numeric(df["temperature_2m_max"], errors="coerce").fillna(20)
    df["temperature_2m_min"] = pd.to_numeric(df["temperature_2m_min"], errors="coerce").fillna(10)

    # Heat risk: 0 at 35°C and below, 1.0 at 50°C
    df["heat_risk"] = ((df["temperature_2m_max"] - 35) / 15).clip(0, 1)

    # Cold risk: 0 at -10°C and above, 1.0 at -30°C
    df["cold_risk"] = ((-10 - df["temperature_2m_min"]) / 20).clip(0, 1)

    # Combined temp risk = max of heat or cold (whichever is more extreme)
    df["temp_risk"] = df[["heat_risk", "cold_risk"]].max(axis=1)

    print(f"[WEATHER] Clean complete. {len(df)} rows retained.")
    return df


# ─────────────────────────────────────────
# STEP 3: EXTRACT FEATURES
# ─────────────────────────────────────────
def extract_features(df: pd.DataFrame = None):
    """
    Aggregate cleaned weather data into one severity score per (date, region).

    Risk score formula:
        weather_severity_score =
            0.35 * precip_norm      (flooding is the biggest port disruption)
            0.30 * wind_norm        (storm conditions halt operations)
            0.20 * wmo_severity     (categorical weather event type)
            0.15 * temp_risk        (extreme heat/cold — less common but severe)

    Output: data/processed/weather_features.csv
    Columns: date | region | weather_severity_score
    """
    print("[WEATHER] Extracting features...")

    if df is None:
        df = clean()

    # ── Composite severity score per city per day ──
    df["row_severity"] = (
        0.35 * df["precip_norm"] +
        0.30 * df["wind_norm"] +
        0.20 * df["wmo_severity"] +
        0.15 * df["temp_risk"]
    ).clip(0, 1)

    # ── Aggregate: max severity per (date, region) ──
    # We use MAX not mean here — if one city in a country has a hurricane,
    # that's a disruption regardless of what the other cities look like.
    # Mean would dilute a real disaster signal.
    features = (
        df.groupby(["date", "region"])
        .agg(
            weather_severity_score=("row_severity", "max"),
            avg_precipitation=("precipitation_sum", "mean"),   # interpretability
            max_windspeed=("windspeed_10m_max", "max"),        # interpretability
            cities_affected=("city", "count"),                 # how many cities hit
        )
        .reset_index()
    )

    features["weather_severity_score"] = features["weather_severity_score"].clip(0, 1).round(4)

    features.to_csv(PROCESSED_PATH, index=False)
    print(f"[WEATHER] Features saved → {PROCESSED_PATH}")
    print(f"[WEATHER] Sample output:\n{features.head(10).to_string()}")

    return features


# ─────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────
def run(days: int = 30):
    """Full pipeline: pull → clean → extract_features"""
    if not RAW_PATH.exists():
        pull(days=days)
    else:
        print("[WEATHER] Raw file already exists, skipping download.")

    df_clean = clean()
    features = extract_features(df_clean)
    return features


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    features = run(days=30)
    print(f"\n[WEATHER] Done. {len(features)} (date, region) pairs produced.")
    print(features.describe())