import os
import pandas as pd
import numpy as np

def compute_monthly_capacity(interim_folder: str, processed_folder: str, vehicle_ids=None):
    os.makedirs(processed_folder, exist_ok=True)
    if vehicle_ids is None:
        vehicle_ids = [f.replace('.parquet', '') for f in os.listdir(interim_folder) if f.endswith('.parquet')]

    for vid in vehicle_ids:
        fpath = os.path.join(interim_folder, f"{vid}.parquet")
        if not os.path.exists(fpath):
            continue

        df = pd.read_parquet(fpath)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'i_pack', 'soc'])

        df['month'] = df['timestamp'].dt.to_period('M')
        df['delta_soc'] = df['soc'].diff().fillna(0)
        df['capacity_ah'] = np.abs(df['i_pack'] * df['delta_soc'])

        grouped = df.groupby('month')['capacity_ah'].median().reset_index()
        grouped.rename(columns={'capacity_ah': 'capacity_median'}, inplace=True)
        grouped['vehicle_id'] = vid
        grouped['capacity_mean'] = df.groupby('month')['capacity_ah'].mean().values

        out_path = os.path.join(processed_folder, f"{vid}_monthly_capacity.csv")
        grouped.to_csv(out_path, index=False)
        print(f"Saved capacity summary: {out_path}")

if __name__ == "__main__":
    compute_monthly_capacity("data/interim", "data/processed", ["#1", "#2", "#3"])
