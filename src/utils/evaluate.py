import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

FEATURE_COLS = [
    "i_pack_mean", "i_pack_std",
    "v_pack_mean", "v_pack_std",
    "soc_mean", "soc_std",
    "t_min_mean", "t_max_mean",
]
RES_FEATURES = ["t_min_mean", "t_max_mean", "soc_mean"]
TARGET_COL = "capacity_median"

def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def load_all(processed_folder="data/processed"):
    fs = [f for f in os.listdir(processed_folder) if f.endswith("_monthly_features.csv")]
    frames = []
    for f in fs:
        d = pd.read_csv(os.path.join(processed_folder, f))
        d["month"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

def evaluate(input_steps=12, output_steps=6, processed_folder="data/processed", models_folder="models"):
    os.makedirs("experiments", exist_ok=True)
    df = load_all(processed_folder).dropna(subset=FEATURE_COLS + [TARGET_COL, "vehicle_id", "month"])
    df = df.sort_values(["vehicle_id", "month"]).reset_index(drop=True)

    feat_scaler = joblib.load(os.path.join(models_folder, "feature_scaler.pkl"))
    seq_model = load_model(os.path.join(models_folder, "seq2seq_final.keras"), compile=False)
    gpr = joblib.load(os.path.join(models_folder, "gpr_residual.pkl"))
    xsc = joblib.load(os.path.join(models_folder, "gpr_xscaler.pkl"))

    rows = []
    for vid, g in df.groupby("vehicle_id"):
        g = g.reset_index(drop=True)
        if len(g) <= input_steps:
            continue

        feats = g[FEATURE_COLS].copy()
        feats[FEATURE_COLS] = feat_scaler.transform(feats[FEATURE_COLS])
        y = g[TARGET_COL].values

        yhat_base, yhat_hyb, y_true = [], [], []
        for i in range(input_steps, len(g)):
            enc = feats.iloc[i - input_steps:i].values
            last_raw = g.iloc[i][RES_FEATURES].values.astype(float)[None, :]   # explanatory vars at time i
            dec = np.repeat(enc[-1][None, :], output_steps, axis=0)

            pred = seq_model.predict([enc[np.newaxis, :, :], dec[np.newaxis, :, :]], verbose=0)[0, :, 0]
            base1 = pred[0]

            xr = xsc.transform(last_raw)
            res = gpr.predict(xr)[0]
            hyb1 = base1 + res

            yhat_base.append(base1)
            yhat_hyb.append(hyb1)
            y_true.append(y[i])

        if y_true:
            mae_b = mean_absolute_error(y_true, yhat_base)
            rmse_b = _rmse(y_true, yhat_base)
            mae_h = mean_absolute_error(y_true, yhat_hyb)
            rmse_h = _rmse(y_true, yhat_hyb)

            rows.append({
                "vehicle_id": vid,
                "n_samples": len(y_true),
                "mae_baseline": mae_b,
                "rmse_baseline": rmse_b,
                "mae_hybrid": mae_h,
                "rmse_hybrid": rmse_h,
                "mae_gain": mae_b - mae_h,
                "rmse_gain": rmse_b - rmse_h,
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join("experiments", "eval_metrics.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(out)
    return out

if __name__ == "__main__":
    evaluate()
