"""Reproducible next-day route booking forecasts; synthetic data only."""
import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROUTES = ('BER-LHR', 'BER-PMI', 'FRA-JFK')
FEATURES = ['trend', 'annual_sin', 'annual_cos', 'weekly_sin', 'weekly_cos',
            'lag_1', 'lag_7', 'mean_7', 'BER-PMI', 'FRA-JFK']


def generate(days=1095, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        d = date(2023, 1, 1) + timedelta(days=day)
        for i, route in enumerate(ROUTES):
            demand = (100 + i * 45 + .035 * day
                      + (22 + i * 5) * np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
                      + 18 * np.cos(2 * np.pi * d.weekday() / 7)
                      + rng.normal(0, 8))
            rows.append({'date': d.isoformat(), 'route': route,
                         'bookings': max(0, int(round(demand)))})
    return rows


def validate(rows):
    seen = set()
    history = {r: [] for r in ROUTES}
    for row in rows:
        d = date.fromisoformat(row['date'])
        r = row['route']
        y = float(row['bookings'])
        if r not in history or not np.isfinite(y) or y < 0 or y != int(y):
            raise ValueError('Unknown route or invalid booking count')
        if (d, r) in seen:
            raise ValueError('Duplicate route/date')
        seen.add((d, r))
        history[r].append(d)
    for dates in history.values():
        dates.sort()
        if len(dates) < 8 or any((b-a).days != 1 for a, b in zip(dates, dates[1:])):
            raise ValueError('Each route needs at least eight consecutive daily observations')


def vector(d, route, past):
    return [(d-date(2023, 1, 1)).days / 365.25,
            np.sin(2*np.pi*d.timetuple().tm_yday/365.25),
            np.cos(2*np.pi*d.timetuple().tm_yday/365.25),
            np.sin(2*np.pi*d.weekday()/7), np.cos(2*np.pi*d.weekday()/7),
            past[-1], past[-7], float(np.mean(past[-7:])),
            float(route == ROUTES[1]), float(route == ROUTES[2])]


def dataset(rows):
    validate(rows)
    past = {r: [] for r in ROUTES}
    x, y, dates, routes, baseline = [], [], [], [], []
    for row in sorted(rows, key=lambda r: (r['date'], r['route'])):
        r = row['route']
        d = date.fromisoformat(row['date'])
        if len(past[r]) >= 7:
            x.append(vector(d, r, past[r]))
            y.append(float(row['bookings']))
            dates.append(row['date'])
            routes.append(r)
            baseline.append(past[r][-7])
        past[r].append(float(row['bookings']))
    return np.array(x), np.array(y), np.array(dates), np.array(routes), np.array(baseline)


def fit(x, y, alpha):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(z.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coef': coef.tolist(),
            'alpha': alpha, 'features': FEATURES, 'schema_version': 1}


def predict(model, x):
    z = (np.asarray(x)-model['mean'])/model['scale']
    return np.maximum(0, np.column_stack([np.ones(len(z)), z]) @ model['coef'])


def metrics(y, p):
    return {'mae': float(np.mean(np.abs(y-p))),
            'rmse': float(np.sqrt(np.mean((y-p)**2))),
            'wape': float(np.sum(np.abs(y-p))/max(np.sum(y), 1))}


def drift(model, x, threshold=2.0):
    shift = np.abs((np.mean(x, axis=0)-model['mean'])/model['scale'])
    return {'threshold': threshold, 'standardized_mean_shift': dict(zip(FEATURES, shift.tolist())),
            'alert': bool(np.any(shift > threshold))}


def run(out=Path('artifacts'), source=None):
    out.mkdir(parents=True, exist_ok=True)
    if source:
        with open(source, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    else:
        rows = generate()
    x, y, dates, routes, baseline = dataset(rows)
    unique = np.unique(dates)
    if len(unique) < 180:
        raise ValueError('Training requires at least 180 usable dates')
    test_start, val_start = unique[-90], unique[-180]
    train, val, test = dates < val_start, (dates >= val_start) & (dates < test_start), dates >= test_start
    candidates = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        m = fit(x[train], y[train], alpha)
        candidates.append((metrics(y[val], predict(m, x[val]))['mae'], alpha))
    alpha = min(candidates)[1]
    model = fit(x[~test], y[~test], alpha)
    p = predict(model, x[test])
    report = {'data_kind': 'external CSV' if source else 'synthetic demo',
              'task': 'rolling one-day-ahead forecasts; actual prior days available',
              'validation_start': val_start, 'test_start': test_start,
              'test_end': unique[-1], 'test_rows': int(test.sum()),
              'validation_candidates': [{'alpha': a, 'mae': s} for s, a in candidates],
              'model': metrics(y[test], p), 'weekly_baseline': metrics(y[test], baseline[test]),
              'by_route': {r: metrics(y[test][routes[test] == r], p[routes[test] == r]) for r in ROUTES},
              'drift': drift(model, x[test])}
    model['trained_through'] = unique[-91]
    model['data_sha256'] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    (out/'model.json').write_text(json.dumps(model, indent=2), encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (out/'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['date', 'route', 'actual', 'prediction', 'weekly_baseline'])
        w.writerows(zip(dates[test], routes[test], y[test], p, baseline[test]))
    with (out/'sample_data.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['date', 'route', 'bookings'])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    run(args.out, args.data)
