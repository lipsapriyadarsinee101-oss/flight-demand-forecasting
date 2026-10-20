# Flight Demand Forecasting

A reproducible data science portfolio project for travel technology: predict tomorrow's route-level booking volume, evaluate against a weekly baseline, and serve predictions through a local HTTP API.

**Synthetic demonstration only.** No AERTiCKET data, affiliation, or real business results are claimed. The generator deliberately contains learnable seasonal patterns; these scores do not establish performance on real airline demand.

## Results

Held-out period: 2025-10-02 to 2025-12-30, 270 route-days across BER-LHR, BER-PMI, and FRA-JFK.

| Metric | Ridge regression | Same weekday last week |
|---|---:|---:|
| MAE (bookings) | 6.43 | 9.30 |
| RMSE (bookings) | 7.93 | 11.45 |
| WAPE | 3.91% | 5.65% |

MAE is about 31% lower than the baseline on this synthetic holdout. Full metrics, route breakdowns, and row-level predictions are in `artifacts/`.

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python forecast.py
python -m unittest -v
python serve.py
```

Training writes a JSON model, metrics, predictions, and generated data into `artifacts/`. No credentials or downloads are required beyond installing NumPy. The model uses standardized ridge regression implemented with NumPy, keeping its mathematics inspectable.

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"date":"2025-12-31","route":"BER-LHR","last_7_bookings":[118,130,142,125,110,115,120]}'
```

On Windows use `curl.exe` and your shell's JSON quoting, or PowerShell:

```powershell
$body = @{date='2025-12-31'; route='BER-LHR'; last_7_bookings=@(118,130,142,125,110,115,120)} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

Supply seven consecutive observed daily counts, oldest first, ending the day before the requested date. The API validates values but cannot verify those dates because only counts are supplied. It is a localhost demonstration, without authentication or production hardening.

## Method and leakage controls

1. Generate 1,095 days of daily counts for three routes with a fixed seed of 42.
2. Validate route identifiers, dates, duplicates, missing daily observations, and nonnegative integer targets.
3. Build calendar seasonality, trend, route indicators, lag-1, lag-7, and trailing seven-day mean. Lag features exclude the target day.
4. Reserve the last 90 dates for testing and the preceding 90 for validation. All routes share the same date boundaries.
5. Fit scaling only on training rows; choose ridge strength from 0.1, 1, 10, and 100 using validation MAE.
6. Refit on training plus validation, freeze the model, and evaluate rolling one-day-ahead predictions. Previous observed test-day targets may become later lag features. This is **not** a fixed-origin 90-day forecast.
7. Save JSON parameters, feature schema, training cutoff, and input-data hash. No pickle deserialization is needed.

## Bring your own data

```bash
python forecast.py --data bookings.csv --out artifacts
```

CSV schema: `date,route,bookings`, dates in YYYY-MM-DD format, one row per route per day. Supported routes are currently fixed to the three demo routes. At least 187 consecutive days per route are required; longer histories are preferable for annual seasonality. Aggregate individual bookings upstream and omit customer identifiers. The dates must represent booking dates, not departure dates, for this particular task. Adapt the feature schema and route validation before using other routes.

## Monitoring and operationalization

`metrics.json` includes a standardized feature mean-shift diagnostic, with a threshold of 2 training standard deviations. It is a simple heuristic, not a calibrated drift test; calendar and trend features can trigger expected shifts. Review alerts alongside realized MAE/WAPE, route-specific errors, missing data, and schema failures. Do not retrain or promote automatically from this signal.

A proposed production process: ingest daily aggregates, validate completeness, predict next-day demand, join predictions with subsequently observed counts, compare rolling error with the baseline, and retrain candidates when sustained degradation appears. Evaluate on fresh temporal windows before promotion and retain the prior model for rollback. Scheduled retraining, cloud deployment, a model registry, and real-data connectors are future work.

## Project map

- `forecast.py`: data generation, validation, features, tuning, evaluation, persistence, and drift diagnostic
- `serve.py`: local prediction API and input validation
- `test_forecast.py`: leakage, invalid data, model persistence, API input, and drift checks
- `.github/workflows/ci.yml`: automated tests and evaluation artifacts
- `docs/model-card.md`: intended use, limitations, and deployment considerations

## Why this project fits the role

Demonstrates a complete workflow from a business question to data preparation, supervised learning, time-based validation, baseline comparison, inference, monitoring, and technical communication. NumPy ridge is an interpretable starting point; gradient boosting and richer external features should be evaluated on real data before adding deep learning complexity.
