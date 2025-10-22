# SOH Fleet Track (Charging-Data Model)

Goal: predict the next **6 months** of battery capacity/SOH from past **12 months** of charging logs.
We use a main predictor (12→6) and a small temperature-based fixer for seasons.

## Steps
1) ETL: unzip car `#1..#3` → normalize columns (time, V, I, SOC, Tmin, Tmax)
2) Monthly label: one capacity value per month (median of sessions)
3) Monthly features: simple stats (avg current/voltage/temps, SOC range)
4) Model: predict 6 months ahead + temperature fixer for adjustment
5) Evaluate: train on 2 cars, test on 1 (rotate)
