import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

FEATURE_COLS = [
    "i_pack_mean","i_pack_std","v_pack_mean","v_pack_std",
    "soc_mean","soc_std","t_min_mean","t_max_mean",
]
TARGET_COL = "capacity_median"

def load_vehicle_table(processed_folder, vehicle_id):
    f = os.path.join(processed_folder, f"{vehicle_id}_monthly_features.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Not found: {f}")
    df = pd.read_csv(f)
    df["month"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    df = df.sort_values("month").reset_index(drop=True)
    return df

def forecast_next_k(vehicle_id, input_steps=12, output_steps=6,
                    processed_folder="data/processed", models_folder="models"):
    df = load_vehicle_table(processed_folder, vehicle_id)

    scaler = joblib.load(os.path.join(models_folder, "feature_scaler.pkl"))
    X_feat = df[FEATURE_COLS].copy()
    X_feat[FEATURE_COLS] = scaler.transform(X_feat[FEATURE_COLS])

    if len(df) < input_steps:
        raise ValueError(f"Not enough history for {vehicle_id}: have {len(df)} months, need {input_steps}")

    enc = X_feat.iloc[-input_steps:].values                            # (N,F)
    last_enc = enc[-1][None, :]                                        # (1,F)
    dec = np.repeat(last_enc, output_steps, axis=0)                    # (K,F)

    enc = enc[np.newaxis, :, :]                                        # (1,N,F)
    dec = dec[np.newaxis, :, :]                                        # (1,K,F)

    model = load_model(os.path.join(models_folder, "seq2seq_final.keras"), compile=False)
    yhat = model.predict([enc, dec], verbose=0)[0, :, 0]               # (K,)

    last_month = df["month"].iloc[-1]
    future_months = pd.date_range(last_month + pd.offsets.MonthEnd(1), periods=output_steps, freq="M")

    out = pd.DataFrame({
        "vehicle_id": vehicle_id,
        "month": future_months,
        "forecast_capacity": yhat
    })
    out_path = os.path.join(processed_folder, f"{vehicle_id}_forecast_seq2seq.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved forecast: {out_path}")
    return out

if __name__ == "__main__":
    for vid in ["#1", "#2", "#3"]:
        forecast_next_k(vid, input_steps=12, output_steps=6)
