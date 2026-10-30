"""Shared constants and preprocessing for the Delay Intelligence pipeline.

Both functions are used for training (part_a_clean_eda.py) and for scoring new
parcels (predict_delay.py), so the same cleaning runs in both places.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100  # network accepts parcels up to 40 kg; anything above 100 is a scanner fault


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians).

    If medians is None the three fill-in medians are computed from this table
    (non-missing values after steps 1-6); otherwise the passed dictionary is used.
    """
    df = raw.copy()

    # 1. exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. service_type: strip spaces, title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # 3. payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value text (e.g. "₹1,290") -> numeric declared_value_inr
    value_text = (df["declared_value"].astype("string")
                  .str.replace("₹", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .str.strip())
    df["declared_value_inr"] = pd.to_numeric(value_text, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M")

    # 6. weight sentinel -> missing
    df["parcel_weight_kg"] = pd.to_numeric(df["parcel_weight_kg"], errors="coerce").astype(float)
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").astype(float)

    # 7. median fill
    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and derived model inputs. Never uses delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
