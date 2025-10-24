import os
import pandas as pd

def build_monthly_features(interim_folder: str, processed_folder: str, vehicle_ids=None):
    os.makedirs(processed_folder, exist_ok=True)
    if vehicle_ids is None:
        vehicle_ids = [f.replace('.parquet', '') for f in os.listdir(interim_folder) if f.endswith('.parquet')]

    for vid in vehicle_ids:
        fpath = os.path.join(interim_folder, f"{vid}.parquet")
        if not os.path.exists(fpath):
            continue

        df = pd.read_parquet(fpath)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'i_pack', 'v_pack', 'soc'])
        df['month'] = df['timestamp'].dt.to_period('M')

        agg = df.groupby('month').agg({
            'i_pack': ['mean', 'std'],
            'v_pack': ['mean', 'std'],
            'soc': ['mean', 'std'],
            't_min': 'mean',
            't_max': 'mean'
        })
        agg.columns = ['_'.join(col) for col in agg.columns]
        agg.reset_index(inplace=True)

        cap_path = os.path.join(processed_folder, f"{vid}_monthly_capacity.csv")
        if os.path.exists(cap_path):
            cap = pd.read_csv(cap_path)
            cap['month'] = pd.PeriodIndex(cap['month'], freq='M')
            agg = pd.merge(agg, cap, on='month', how='left')

        out_path = os.path.join(processed_folder, f"{vid}_monthly_features.csv")
        agg.to_csv(out_path, index=False)
        print(f"Saved monthly feature file: {out_path}")

if __name__ == "__main__":
    build_monthly_features("data/interim", "data/processed", ["#1", "#2", "#3"])
