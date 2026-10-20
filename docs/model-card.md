# Model card

## Intended use
Next-day daily booking volume by supported route for an educational travel-technology portfolio. This project is not affiliated with AERTiCKET. Do not use its output for pricing, inventory commitments, or staffing without real-data validation.

## Training data and features
Seeded synthetic daily observations from 2023-01-01 through 2025-12-30. Synthetic demand includes trend, weekly and annual seasonality, route offsets, and Gaussian noise. It excludes cancellations, disruptions, fare elasticity, capacity constraints, marketing interventions, holidays, and competitor behavior.

Features: calendar sine/cosine terms, trend, route indicators, previous-day count, same weekday last week, and trailing seven-day mean. Numeric scaling is learned within each fit. Ridge strength is selected on a later validation window. The persisted evaluation model is frozen before the final holdout period.

## Evaluation
Final 90 calendar days, three routes, 270 observations. MAE 6.43, RMSE 7.93, WAPE 3.91%; weekly baseline MAE 9.30. See artifacts/metrics.json for exact values and per-route metrics. Results are synthetic and optimistic because the generator matches the chosen feature family. No confidence intervals or prediction intervals are estimated. One holdout window does not establish robustness across seasons or regimes.

## Limitations and risks
Unseen routes are rejected. Demand is bookings, not unconstrained passenger demand. Capacity-censored bookings can misrepresent underlying demand. Drift monitoring only measures feature mean shifts and can miss variance changes, interactions, and concept drift. Long extrapolation is unreliable. The API assumes correctly aligned seven-day histories and cannot independently verify their chronology.

## Before production
Use licensed, privacy-reviewed aggregates; evaluate multiple rolling origins, holidays, disruptions, route segments, missing-data policies, and cold starts. Compare stronger baselines and tree models. Add uncertainty estimates, access controls, request timeouts, concurrency limits, structured logs, schema version enforcement, monitoring ownership, and model rollback. Agree acceptable error thresholds with business stakeholders.
