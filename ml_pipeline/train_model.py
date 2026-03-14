"""
CLV Predictor - ML Pipeline
Train a RandomForestRegressor on Online Retail II UCI dataset
to predict 30-day customer lifetime value (future spend).

Key fixes vs v1:
  • Model is wrapped in a sklearn Pipeline (StandardScaler + RF).
    This means the model carries feature names, eliminating the
    "X does not have valid feature names" warning on predict().
  • feature_meta.pkl saves the exact column order so backend
    always builds a named DataFrame — not a raw numpy array.
  • All dropped / error / unused rows are exported to audit CSVs.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  CLV Predictor — ML Pipeline  (v2)")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. DATA LOADING & CLEANING  (with audit exports)
# ─────────────────────────────────────────────
csv_path = os.path.join(SCRIPT_DIR, "online_retail_II.csv")
print(f"\n[1/7] Loading dataset from: {csv_path}")

df_raw = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
print(f"      Raw rows: {len(df_raw):,}")

audit_frames = {}

# ── 1a. Missing Customer ID ────────────────────────────────────
missing_cid = df_raw[df_raw["Customer ID"].isna()].copy()
missing_cid["drop_reason"] = "missing_customer_id"
audit_frames["missing_customer_id"] = missing_cid
print(f"      Dropped (missing Customer ID) : {len(missing_cid):,}")
df = df_raw.dropna(subset=["Customer ID"]).copy()
df["Customer ID"] = df["Customer ID"].astype(int)

# ── 1b. Cancelled orders (Invoice starts with 'C') ─────────────
cancelled = df[df["Invoice"].astype(str).str.startswith("C")].copy()
cancelled["drop_reason"] = "cancelled_order"
audit_frames["cancelled_orders"] = cancelled
print(f"      Dropped (cancelled orders)    : {len(cancelled):,}")
df = df[~df["Invoice"].astype(str).str.startswith("C")].copy()

# ── 1c. Non-positive Quantity ──────────────────────────────────
bad_qty = df[df["Quantity"] <= 0].copy()
bad_qty["drop_reason"] = "non_positive_quantity"
audit_frames["bad_quantity"] = bad_qty
print(f"      Dropped (Quantity ≤ 0)        : {len(bad_qty):,}")
df = df[df["Quantity"] > 0].copy()

# ── 1d. Non-positive Price ─────────────────────────────────────
bad_price = df[df["Price"] <= 0].copy()
bad_price["drop_reason"] = "non_positive_price"
audit_frames["bad_price"] = bad_price
print(f"      Dropped (Price ≤ 0)           : {len(bad_price):,}")
df = df[df["Price"] > 0].copy()

# ── Save master audit CSV ──────────────────────────────────────
dropped_df = pd.concat(audit_frames.values(), ignore_index=True)
dropped_path = os.path.join(SCRIPT_DIR, "dropped_rows_audit.csv")
dropped_df.to_csv(dropped_path, index=False)
print(f"\n      ✓ dropped_rows_audit.csv   ({len(dropped_df):,} rows)")

# ── Derived columns ────────────────────────────────────────────
df["TotalAmount"] = df["Quantity"] * df["Price"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print(f"      Clean rows remaining: {len(df):,}")

# ─────────────────────────────────────────────
# 2. TEMPORAL SPLIT
# ─────────────────────────────────────────────
print("\n[2/7] Computing temporal split...")
max_date    = df["InvoiceDate"].max()
cutoff_date = max_date - pd.Timedelta(days=30)
print(f"      Max date    : {max_date.date()}")
print(f"      Cutoff date : {cutoff_date.date()}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING (PAST BEHAVIOR)
# ─────────────────────────────────────────────
print("\n[3/7] Engineering features from past transactions...")

past_df = df[df["InvoiceDate"] < cutoff_date].copy()

# Customers with data ONLY after the cutoff — cannot build features for them
future_only_ids = set(df["Customer ID"]) - set(past_df["Customer ID"])
future_only_rows = df[df["Customer ID"].isin(future_only_ids)].copy()
future_only_rows["drop_reason"] = "no_pre_cutoff_history"
future_only_path = os.path.join(SCRIPT_DIR, "future_only_customers.csv")
future_only_rows.to_csv(future_only_path, index=False)
print(f"      Customers with no pre-cutoff history (excluded): {len(future_only_ids):,}")
print(f"      ✓ future_only_customers.csv saved")

features = (
    past_df.groupby("Customer ID")
    .agg(
        last_purchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum"),
        Country=("Country", "first"),
    )
    .reset_index()
)
features["Recency"] = (cutoff_date - features["last_purchase"]).dt.days
features["Is_UK"]   = (features["Country"] == "United Kingdom").astype(int)
features = features[["Customer ID", "Recency", "Frequency", "Monetary", "Is_UK"]]
print(f"      Customers with past transactions: {len(features):,}")

# ─────────────────────────────────────────────
# 4. TARGET DEFINITION (30-DAY FUTURE SPEND)
# ─────────────────────────────────────────────
print("\n[4/7] Computing 30-day future spend (target)...")

future_df = df[df["InvoiceDate"] >= cutoff_date].copy()
future_spend = (
    future_df.groupby("Customer ID")["TotalAmount"]
    .sum()
    .reset_index()
    .rename(columns={"TotalAmount": "future_spend_30d"})
)
print(f"      Customers who returned in 30 days: {len(future_spend):,}")

# ─────────────────────────────────────────────
# 5. DATASET MERGING
# ─────────────────────────────────────────────
print("\n[5/7] Merging features and target...")

model_df = features.merge(future_spend, on="Customer ID", how="left")
model_df["future_spend_30d"] = model_df["future_spend_30d"].fillna(0.0)

model_df.to_csv(os.path.join(SCRIPT_DIR, "model_dataset.csv"), index=False)
print(f"      Final dataset shape: {model_df.shape}")
print(f"      Customers with future spend > 0: {(model_df['future_spend_30d'] > 0).sum():,}")
print(f"      ✓ model_dataset.csv saved")

# ─────────────────────────────────────────────
# 6. MODELING
# ─────────────────────────────────────────────
print("\n[6/7] Training RandomForestRegressor pipeline...")

# IMPORTANT: Keep X as a named DataFrame (not .values) so sklearn stores
# feature names inside the Pipeline — this is what eliminates the warning.
FEATURE_COLS = ["Recency", "Frequency", "Monetary", "Is_UK"]
X = model_df[FEATURE_COLS]
y = model_df["future_spend_30d"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )),
])
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\n  ┌──────────────────────────────────────┐")
print(f"  │  MAE  : ${mae:>10,.2f}                │")
print(f"  │  R²   : {r2:>10.4f}                │")
print("  └──────────────────────────────────────┘")

# ─────────────────────────────────────────────
# 7. EXPORT
# ─────────────────────────────────────────────
print("\n[7/7] Saving artifacts...")

model_path = os.path.join(SCRIPT_DIR, "clv_rf_model.pkl")
joblib.dump(model_pipeline, model_path)
print(f"      ✓ clv_rf_model.pkl  (sklearn Pipeline: StandardScaler + RF)")

# Save feature column order — backend reads this to build named DataFrames
meta_path = os.path.join(SCRIPT_DIR, "feature_meta.pkl")
joblib.dump({"feature_cols": FEATURE_COLS}, meta_path)
print(f"      ✓ feature_meta.pkl  (feature column names + order)")

print("\n  ─── Generated files ──────────────────────────────────")
print("    ml_pipeline/clv_rf_model.pkl          — model pipeline (scaler+RF)")
print("    ml_pipeline/feature_meta.pkl          — feature column metadata")
print("    ml_pipeline/dropped_rows_audit.csv    — all dropped / invalid rows")
print("    ml_pipeline/future_only_customers.csv — no-history customers")
print("    ml_pipeline/model_dataset.csv         — final RFM + target table")
print("\n  Pipeline complete ✓\n")