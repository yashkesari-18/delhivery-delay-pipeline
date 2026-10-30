"""Part D (segments) - hub profile, K-Means segments, separation evidence, tier-versus-segment check."""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile, ascending hub_id
profile = (table.groupby("hub_id")
           .agg(shipments=("shipment_id", "count"),
                mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"),
                mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .reset_index()
           .sort_values("hub_id")
           .reset_index(drop=True))
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. scale the four profile columns (hubs in ascending hub_id order)
PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[PROFILE_COLS])

# 3. separation evidence for K = 2..5
dist = pairwise_distances(Z)
iu = np.triu_indices(len(Z), k=1)  # 276 unordered hub pairs
sep_rows = []
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
    same = (labels[:, None] == labels[None, :])[iu]
    ratio = dist[iu][~same].mean() / dist[iu][same].mean()
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    sep_rows.append({"K": k, "across_over_within": round(float(ratio), 4),
                     "cluster_sizes": "-".join(str(s) for s in sizes)})
pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)

# K = 3 is fixed by the business (three playbooks)
labels3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Z)
profile["cluster"] = labels3

# 4. name clusters by mean hub delay_rate: Standard < Watch < Escalate
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments.to_csv("outputs/hub_segments.csv", index=False)

seg_profile = (profile.groupby("segment")
               .agg(hubs=("hub_id", "count"),
                    mean_delivery_hours=("mean_delivery_hours", "mean"),
                    delay_rate=("delay_rate", "mean"),
                    mean_distance_km=("mean_distance_km", "mean"),
                    mean_n_stops=("mean_n_stops", "mean"))
               .reindex(["Standard", "Watch", "Escalate"])
               .reset_index())
seg_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

TIER_IMPLIES = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_IMPLIES)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(profile.round(4).to_string(index=False))
print(pd.DataFrame(sep_rows).to_string(index=False))
print(seg_profile.round(4).to_string(index=False))
print(mismatch.to_string(index=False))
