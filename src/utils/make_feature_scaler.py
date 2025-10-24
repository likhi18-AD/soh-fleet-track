import os
import glob
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "i_pack_mean", "i_pack_std",
    "v_pack_mean", "v_pack_std",
    "soc_mean", "soc_std",
    "t_min_mean", "t_max_mean",
]

def main(processed_folder="data/processed", out_path="models/feature_scaler.pkl"):
    files = sorted(glob.glob(os.path.join(processed_folder, "*_monthly_features.csv")))
    if not files:
        raise FileNotFoundError(f"No *_monthly_features.csv files in {processed_folder}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    X = df[FEATURE_COLS].astype("float32").copy()
    scaler = StandardScaler()
    scaler.fit(X)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(scaler, out_path)
    print(f"Saved scaler: {out_path} (n={len(X)})")

if __name__ == "__main__":
    main()
