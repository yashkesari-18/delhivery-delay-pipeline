"""Part A - clean the raw export, build features, write the modelling table and delay evidence."""
import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, PROMISED_HOURS, build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# ---- "before" measurements (after removing duplicates, before any filling) ----
dedup = raw.drop_duplicates()
weight_num = pd.to_numeric(dedup["parcel_weight_kg"], errors="coerce")
before = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(weight_num.isna().sum()),  # sentinels not included
    "sentinel_weight_rows": int((weight_num > 100).sum()),
    "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": int(dedup["service_type"].nunique()),
}

# ---- clean + features ----
clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

# ---- checks ----
hours = table["delivery_hours"]
q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

checks = dict(before)
checks.update({
    "service_type_spellings_after": int(clean["service_type"].nunique()),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(table),
    "nulls_in_features_and_target": int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(table["is_delayed"].sum()),
    "delay_rate": table["is_delayed"].mean(),
    "hours_q1": q1,
    "hours_q3": q3,
    "hours_lower_bound": lower,
    "hours_upper_bound": upper,
    "hours_outlier_count": int(((hours < lower) | (hours > upper)).sum()),
    "declared_value_mean_inr": table["declared_value_inr"].mean(),
    "declared_value_median_inr": table["declared_value_inr"].median(),
    "log_declared_value_mean": table["log_declared_value"].mean(),
    "log_declared_value_median": table["log_declared_value"].median(),
})
checks_df = pd.DataFrame({"metric": list(checks.keys()),
                          "value": pd.Series([v if isinstance(v, int) else round(float(v), 4)
                                              for v in checks.values()], dtype=object)})
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# ---- delay evidence by group ----
table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=False).agg(shipments=("is_delayed", "size"),
                                               delayed=("is_delayed", "sum"),
                                               delay_rate=("is_delayed", "mean"),
                                               mean_hours=("delivery_hours", "mean")).reset_index()
    for r in g.to_dict("records"):
        rows.append({"dimension": dim, "group": str(r[dim]), "shipments": int(r["shipments"]),
                     "delayed": int(r["delayed"]), "delay_rate": round(float(r["delay_rate"]), 4),
                     "mean_hours": round(float(r["mean_hours"]), 4)})
pd.DataFrame(rows).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# ---- correlations of numeric columns ----
numeric = table.drop(columns=["distance_band"]).select_dtypes(include=[np.number])
corr = numeric.corr().round(4)
corr.index.name = "column"
corr.to_csv("outputs/part_a_correlations.csv", index=True)

print(checks_df.to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
top = corr["delivery_hours"].drop("delivery_hours").abs().sort_values(ascending=False).head(3).index
print("Top correlations with delivery_hours:", corr.loc[top, "delivery_hours"].to_dict())
