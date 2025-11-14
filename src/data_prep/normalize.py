import os
import re
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Patterns for flexible matching
CANDIDATES = {
    "timestamp": [r"^record[_ ]?time$", r"^timestamp$", r"^time(_stamp)?$", r"^datetime$", r"^date_time$"],
    "v_pack": [r"^pack[_ ]?voltage.*$", r"^voltage(_pack)?$"],
    "i_pack": [r"^charge[_ ]?current.*$", r"^current(_pack)?$"],
    "soc":    [r"^soc$", r"^state[_ ]?of[_ ]?charge.*$"],
    "t_min":  [r"^min[_ ]?temperature.*$"],
    "t_max":  [r"^max[_ ]?temperature.*$"],
}

# Exact-name fallbacks (units in names)
EXACT_RENAMES = {
    "pack_voltage (V)": "v_pack",
    "charge_current (A)": "i_pack",
    "max_temperature (°)": "t_max",
    "min_temperature (°)": "t_min",
}

def find_col(cols, patterns):
    cols_l = [c.strip().lower() for c in cols]
    for p in patterns:
        rgx = re.compile(p, re.IGNORECASE)
        for i, c in enumerate(cols_l):
            if rgx.match(c):
                return cols[i]
    return None

def parse_timestamp(series):
    # If already datetime-like strings
    ts = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if ts.notna().mean() > 0.9 and (ts.dt.year.dropna().median() >= 2000):
        return ts.dt.tz_localize(None) if getattr(ts.dt, "tz", None) is not None else ts

    s = series.astype(str).str.strip()

    # YYYYMMDDHHMMSS (14 digits)
    mask14 = s.str.fullmatch(r"\d{14}", na=False)
    if mask14.any():
        ts2 = pd.to_datetime(s[mask14], format="%Y%m%d%H%M%S", errors="coerce")
        ts.loc[mask14] = ts2.values

    # YYYYMMDDHHMM (12 digits)
    mask12 = s.str.fullmatch(r"\d{12}", na=False)
    if mask12.any():
        ts2 = pd.to_datetime(s[mask12], format="%Y%m%d%H%M", errors="coerce")
        ts.loc[mask12] = ts2.values

    # Unix epochs as s/ms/us/ns
    if ts.notna().mean() < 0.9:
        num = pd.to_numeric(s, errors="coerce")
        for unit in ["s", "ms", "us", "ns"]:
            ts_try = pd.to_datetime(num, unit=unit, errors="coerce")
            if ts_try.notna().mean() > ts.notna().mean() and (ts_try.dt.year.dropna().median() >= 2000):
                ts = ts_try

    return pd.to_datetime(ts, errors="coerce")

def normalize_vehicle_data(raw_folder: str, interim_folder: str, vehicle_ids=None):
    os.makedirs(interim_folder, exist_ok=True)
    if vehicle_ids is None:
        vehicle_ids = [f for f in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, f))]

    for vid in vehicle_ids:
        vpath = os.path.join(raw_folder, vid)
        if not os.path.isdir(vpath):
            continue
        csvs = [f for f in os.listdir(vpath) if f.lower().endswith(".csv")]
        if not csvs:
            continue

        dfs = []
        for fn in csvs:
            fp = os.path.join(vpath, fn)
            try:
                temp = pd.read_csv(fp)
            except Exception:
                continue

            # First do exact renames (handles unit suffixes)
            temp = temp.rename(columns=EXACT_RENAMES)

            # Then regex-based picks
            colmap = {}
            for std, pats in CANDIDATES.items():
                hit = find_col(temp.columns, pats)
                if hit:
                    colmap[hit] = std
            temp = temp.rename(columns=colmap)

            # Build timestamp
            if "timestamp" in temp.columns:
                temp["timestamp"] = parse_timestamp(temp["timestamp"])
            else:
                temp["timestamp"] = pd.NaT

            keep = ["timestamp", "v_pack", "i_pack", "soc", "t_min", "t_max"]
            present = [k for k in keep if k in temp.columns]
            temp = temp[present]
            temp["vehicle_id"] = vid
            dfs.append(temp)

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        out_path = os.path.join(interim_folder, f"{vid}.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_path)
        print(f"saved: {out_path} rows={len(df)} cols={list(df.columns)}")

if __name__ == "__main__":
    normalize_vehicle_data("data/raw", "data/interim", ["#1", "#2", "#3"])
