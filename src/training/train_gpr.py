import os
import numpy as np
import pandas as pd
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

FEATURE_COLS = [
    "i_pack_mean", "i_pack_std",
    "v_pack_mean", "v_pack_std",
    "soc_mean", "soc_std",
    "t_min_mean", "t_max_mean",
]
# Features used to explain residuals
RES_FEATURES = ["t_min_mean", "t_max_mean", "soc_mean"]

TARGET_COL = "capacity_median"

def load_processed(processed_folder="data/processed"):
    files = [f for f in os.listdir(processed_folder) if f.endswith("_monthly_features.csv")]
    frames = []
    for f in files:
        df = pd.read_csv(os.path.join(processed_folder, f))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["month"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return df

def make_one_step_pairs(df, seq_model, feat_scaler, input_steps=12, output_steps=6):
    Xr, yr = [], []

    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("month").reset_index(drop=True)
        if len(g) <= input_steps:
            continue

        feats = g[FEATURE_COLS].copy()
        feats[FEATURE_COLS] = feat_scaler.transform(feats[FEATURE_COLS])
        y = g[TARGET_COL].values.reshape(-1, 1)

        for i in range(input_steps, len(g)):
            enc = feats.iloc[i - input_steps:i].values                       # (N,F)
            last_enc = enc[-1][None, :]                                      # (1,F)
            dec = np.repeat(last_enc, output_steps, axis=0)                   # (K,F)

            yhat = seq_model.predict([enc[np.newaxis, :, :], dec[np.newaxis, :, :]], verbose=0)[0, :, 0]
            baseline_1 = yhat[0]                                             # one-step ahead baseline
            actual_1 = y[i, 0]
            residual = actual_1 - baseline_1

            xr = g.loc[i, RES_FEATURES].values.astype(float)                 # raw (unscaled) explanatory vars
            Xr.append(xr)
            yr.append(residual)

    if not Xr:
        raise ValueError("No residual samples created. Check input_steps or data length.")
    return np.array(Xr), np.array(yr)

def main(input_steps=12, output_steps=6, models_folder="models"):
    os.makedirs(models_folder, exist_ok=True)

    df = load_processed("data/processed").dropna(subset=FEATURE_COLS + [TARGET_COL, "vehicle_id", "month"])

    feat_scaler = joblib.load(os.path.join(models_folder, "feature_scaler.pkl"))
    seq_model = load_model(os.path.join(models_folder, "seq2seq_final.keras"), compile=False)

    Xr_raw, yr = make_one_step_pairs(df, seq_model, feat_scaler, input_steps, output_steps)

    x_scaler = StandardScaler()
    Xr = x_scaler.fit_transform(Xr_raw)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(Xr.shape[1]), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42)
    gpr.fit(Xr, yr)

    joblib.dump(gpr, os.path.join(models_folder, "gpr_residual.pkl"))
    joblib.dump(x_scaler, os.path.join(models_folder, "gpr_xscaler.pkl"))

    print("Saved:", os.path.join(models_folder, "gpr_residual.pkl"))
    print("Samples:", Xr.shape[0])

if __name__ == "__main__":
    main()
