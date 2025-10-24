import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.seq2seq import build_seq2seq
import joblib

FEATURE_COLS = [
    "i_pack_mean", "i_pack_std",
    "v_pack_mean", "v_pack_std",
    "soc_mean", "soc_std",
    "t_min_mean", "t_max_mean",
]
TARGET_COL = "capacity_median"

def load_processed(processed_folder="data/processed"):
    files = [f for f in os.listdir(processed_folder) if f.endswith("_monthly_features.csv")]
    if not files:
        raise FileNotFoundError("No *_monthly_features.csv found in data/processed/")
    frames = []
    for f in files:
        df = pd.read_csv(os.path.join(processed_folder, f))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # normalize month to sortable timestamp
    if "month" in df.columns:
        df["month"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return df

def make_sequences_groupwise(df, input_steps=12, output_steps=6):
    X_enc, X_dec, Y = [], [], []

    for vid, g in df.groupby("vehicle_id"):
        g = g.sort_values("month").reset_index(drop=True)
        if len(g) < input_steps + output_steps + 1:
            continue

        feat = g[FEATURE_COLS].values
        tgt  = g[TARGET_COL].values.reshape(-1, 1)

        for i in range(len(g) - input_steps - output_steps + 1):
            enc = feat[i:i+input_steps]                              # (N, F)
            # simple decoder input: repeat last encoder feature row K times
            last_enc = enc[-1][None, :]                              # (1, F)
            dec = np.repeat(last_enc, output_steps, axis=0)          # (K, F)
            out = tgt[i+input_steps:i+input_steps+output_steps]      # (K, 1)

            X_enc.append(enc)
            X_dec.append(dec)
            Y.append(out)

    if not X_enc:
        raise ValueError("No sequences created. Reduce input_steps/output_steps or check data length.")
    return np.array(X_enc), np.array(X_dec), np.array(Y)

def main(input_steps=12, output_steps=6, hidden_units=64, epochs=50, batch_size=16):
    os.makedirs("models", exist_ok=True)

    df = load_processed()
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "vehicle_id", "month"])

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    joblib.dump(scaler, "models/feature_scaler.pkl")

    X_enc, X_dec, Y = make_sequences_groupwise(df, input_steps, output_steps)

    model = build_seq2seq(input_steps=input_steps, feature_dim=len(FEATURE_COLS),
                          output_steps=output_steps, hidden_units=hidden_units)

    ckpt = ModelCheckpoint("models/seq2seq_best.h5", save_best_only=True, monitor="val_loss")
    es   = EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")

    history = model.fit(
        [X_enc, X_dec], Y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, es],
        verbose=1
    )

    model.save("models/seq2seq_final.h5")
    print("Saved: models/seq2seq_final.h5")
    print(f"Train samples: {len(X_enc)}  Val split: 20%")
    print("Last epoch:",
          history.epoch[-1],
          "loss=", float(history.history['loss'][-1]),
          "val_loss=", float(history.history['val_loss'][-1]))

if __name__ == "__main__":
    main()
