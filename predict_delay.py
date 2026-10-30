"""Score parcels about to be dispatched and turn them into an action list."""
import os

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute["column"], impute["median"].astype(float)))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

# same cleaning and features as training, with the stored historical medians
clean, _ = clean_shipments(new, medians=medians)
features = build_features(clean, hubs)
assert features[feature_order].notna().all().all(), "missing values in model inputs"

out = features[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = model.predict_proba(features[feature_order])[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out = out.merge(segments, on="hub_id", how="left")


def action(row):
    if row["flagged"] != 1:
        return "No action"
    return "Escalate to hub manager" if row["segment"] == "Escalate" else "Proactive customer call"


out["action"] = out.apply(action, axis=1)
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(out["action"].value_counts().to_string())
print("flagged:", int(out["flagged"].sum()), "of", len(out))
