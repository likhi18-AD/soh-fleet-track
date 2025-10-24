import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

FEATURE_COLS = [
    "i_pack_mean", "i_pack_std",
    "v_pack_mean", "v_pack_std",
    "soc_mean", "soc_std",
    "t_min_mean", "t_max_mean",
]
RES_FEATURES = ["t_min_mean", "t_max_mean", "soc_mean"]

def load_vehicle_table(processed_folder, vehicle_id):
    f = os.path.join(processed_folder, f"{vehicle_id}_monthly_features.csv")
    df = pd.read_csv(f)
    df["month"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return df.sort_values("month").reset_index(drop=True)

def apply_gpr(vehicle_id, input_steps=12, output_steps=6,
              processed_folder="data/processed", models_folder="models"):
    base_path = os.path.join(processed_folder, f"{vehicle_id}_forecast_seq2seq.csv")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Baseline forecast not found: {base_path}")

    base = pd.read_csv(base_path)
    base["month"] = pd.to_datetime(base["month"])

    df = load_vehicle_table(processed_folder, vehicle_id)

    feat_scaler = joblib.load(os.path.join(models_folder, "feature_scaler.pkl"))
    gpr = joblib.load(os.path.join(models_folder, "gpr_residual.pkl"))
    xsc = joblib.load(os.path.join(models_folder, "gpr_xscaler.pkl"))

    if len(df) < input_steps:
        raise ValueError(f"Not enough history for {vehicle_id}")

    # Use last encoder row's raw explanatory vars (temps, soc) for all K steps
    last_row = df.iloc[-1]
    xr = last_row[RES_FEATURES].values.astype(float)[None, :]    # (1,3)
    xr = xsc.transform(xr)

    residuals = gpr.predict(np.repeat(xr, len(base), axis=0))    # (K,)

    hybrid = base.copy()
    hybrid["forecast_capacity_hybrid"] = hybrid["forecast_capacity"].values + residuals
    outp = os.path.join(processed_folder, f"{vehicle_id}_forecast_hybrid.csv")
    hybrid.to_csv(outp, index=False)
    print("Saved:", outp)
    return hybrid

if __name__ == "__main__":
    for vid in ["#1", "#2", "#3"]:
        apply_gpr(vid)
