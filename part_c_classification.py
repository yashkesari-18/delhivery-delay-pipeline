"""Part C - delay classifier: baselines, feature signal, tuning, alert threshold, one test evaluation."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

# 1. stratified split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print("sizes:", len(X_train), len(X_val), len(X_test), "delayed:", y_train.sum(), y_val.sum(), y_test.sum())


def scores(y_true, y_pred):
    """Confusion-matrix cells and the four metrics derived from them."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": round((tp + tn) / (tn + fp + fn + tp), 4),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def f1_of(y_true, y_pred):
    return scores(y_true, y_pred)["f1"]


# 2. blind baseline + three classifiers at the default 0.5 rule
model_rows = [{"model": "blind_all_on_time", **scores(y_val, np.zeros(len(y_val), dtype=int))}]

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **scores(y_val, logreg.predict(X_val))})

best_d, best_tree_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f1 = f1_of(y_val, tree.predict(X_val))
    if f1 > best_tree_f1:  # strict '>' keeps the smaller depth on ties
        best_d, best_tree_f1 = d, f1
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
model_rows.append({"model": f"decision_tree_depth_{best_d}", **scores(y_val, tree.predict(X_val))})

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **scores(y_val, rf_default.predict(X_val))})
pd.DataFrame(model_rows).to_csv("outputs/part_c_models.csv", index=False)

# 3. which columns carry signal
selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
feat = pd.DataFrame({"feature": FEATURES,
                     "f_score": selector_all.scores_,
                     "rf_importance": rf_default.feature_importances_})
feat["f_rank"] = feat["f_score"].rank(ascending=False, method="first").astype(int)
feat["is_noise"] = feat["feature"].isin(NOISE_COLUMNS)
feat = feat.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
feat["f_score"] = feat["f_score"].round(4)
feat["rf_importance"] = feat["rf_importance"].round(4)
feat.to_csv("outputs/part_c_feature_scores.csv", index=False)
noise_share = float(rf_default.feature_importances_[[FEATURES.index(c) for c in NOISE_COLUMNS]].sum())
print("noise share of forest importance:", round(noise_share, 4))

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_of(y_val, pipe.predict(X_val))})
pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning on the 7 selected columns (selector fitted on training rows only; test rows untouched)
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xtr7, Xval7 = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}

gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5,
                        random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)


def fmt_params(p):
    return "; ".join(f"{k}={'unlimited' if v is None else v}" for k, v in sorted(p.items()))


default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": len(gs.cv_results_["params"]), "cv_folds": 5,
     "total_fits": len(gs.cv_results_["params"]) * 5, "best_params": fmt_params(gs.best_params_),
     "best_cv_accuracy": round(float(gs.best_score_), 4), "val_f1": f1_of(y_val, gs.predict(Xval7))},
    {"search": "randomized_search", "candidates": len(rs.cv_results_["params"]), "cv_folds": 5,
     "total_fits": len(rs.cv_results_["params"]) * 5, "best_params": fmt_params(rs.best_params_),
     "best_cv_accuracy": round(float(rs.best_score_), 4), "val_f1": f1_of(y_val, rs.predict(Xval7))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": fmt_params(default_params), "best_cv_accuracy": np.nan,
     "val_f1": f1_of(y_val, default7.predict(Xval7))},
])
tuning.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. choose final configuration (ties -> default, then grid, then randomized)
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
best_f1 = tuning["val_f1"].max()
chosen_from = min(tuning.loc[tuning["val_f1"] == best_f1, "search"], key=priority.get)
chosen_params = {"default_forest_same_7_features": default_params,
                 "grid_search": gs.best_params_, "randomized_search": rs.best_params_}[chosen_from]
final = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                  ("model", RandomForestClassifier(random_state=42, **chosen_params))]).fit(X_train, y_train)

# 6. alert threshold on validation probabilities
val_prob = final.predict_proba(X_val)[:, 1]
thr_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    thr_rows.append({"threshold": t, **scores(y_val, (val_prob >= t).astype(int))})
thr = pd.DataFrame(thr_rows)
thr.to_csv("outputs/part_c_thresholds.csv", index=False)
meets = thr.loc[thr["recall"] >= 0.80, "threshold"]
threshold = float(meets.max()) if len(meets) else 0.1

# 7. one test evaluation
test_prob = final.predict_proba(X_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **scores(y_test, (test_prob >= threshold).astype(int))}]) \
    .to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": chosen_params["n_estimators"],
               "max_depth": "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"],
               "min_samples_leaf": chosen_params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen_from}]).to_csv("outputs/part_c_final_config.csv", index=False)

print(pd.DataFrame(model_rows).to_string(index=False))
print(feat.to_string(index=False))
print(pd.DataFrame(selectk_rows).to_string(index=False))
print(tuning.to_string(index=False))
print(thr.to_string(index=False))
print("chosen:", chosen_from, chosen_params, "threshold:", threshold)
print(pd.read_csv("outputs/part_c_final_test.csv").to_string(index=False))
