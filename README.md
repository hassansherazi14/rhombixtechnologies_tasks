# Equipment Failure Prediction — Data Preprocessing & Feature Engineering

Data Science Internship Task · Rhombix Technologies

## Overview
Preprocessed and engineered features for predicting equipment (turbofan engine) failure
using NASA's C-MAPSS Turbofan Engine Degradation dataset (FD001 subset).

## Dataset
- **Source:** NASA C-MAPSS Turbofan Engine Degradation Simulation, FD001 subset
- **Size:** 100 engines, run to failure, 20,631 total cycle records
- **Features:** 3 operational settings + 21 sensor channels per cycle
- **Target:** Remaining Useful Life (RUL), derived from cycles-to-failure per engine

## Task 1 — Data Preprocessing
- Loaded raw sensor data, inspected structure and data types
- Checked for missing values
- Dropped zero-variance sensors (no predictive signal)
- Constructed the RUL target (and a clipped version, capped at 125 cycles)
- Added an optional binary "failure within 30 cycles" label
- Scaled sensor readings (MinMax)

## Task 2 — Feature Engineering
- Time-based: normalized cycle position per engine (0 = start of life, 1 = failure)
- Rolling statistics: 5-cycle rolling mean/std per sensor, per engine
- Lag features: sensor readings 1 and 3 cycles back
- Rate-of-change: cycle-to-cycle delta per sensor
- Cumulative degradation: running sum of absolute deltas per sensor

Result: 26 raw columns → 129 engineered features.

## Files
| File | Description |
|---|---|
| `equipment_failure_feature_engineering.ipynb` | Full notebook — preprocessing + feature engineering |
| `train_FD001.txt` | Raw C-MAPSS input data |
| `cmapss_fd001_features.csv` | Processed output with engineered features |

## Tools
Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Next Steps
Train/test split by engine ID, feature selection, and model training (RUL regression
or binary failure classification) — planned for a later task.
