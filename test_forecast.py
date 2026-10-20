import tempfile
import unittest
from pathlib import Path

import numpy as np
from forecast import dataset, drift, fit, generate, predict, run, validate
from serve import forecast_request


class ForecastTests(unittest.TestCase):
    def test_future_targets_do_not_change_earlier_features(self):
        rows = generate(30)
        before = dataset(rows)[0]
        rows[-1]['bookings'] += 1000
        np.testing.assert_array_equal(before, dataset(rows)[0])

    def test_invalid_data(self):
        rows = generate(30)
        with self.assertRaises(ValueError):
            validate(rows + [rows[0]])
        rows[0]['bookings'] = -1
        with self.assertRaises(ValueError):
            validate(rows)

    def test_artifact_roundtrip_and_api(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            report = run(Path(d))
            model = json.loads((Path(d)/'model.json').read_text())
            self.assertLess(report['model']['mae'], report['weekly_baseline']['mae'])
            self.assertEqual(report['test_rows'], 270)
            result = forecast_request(model, {'date': '2026-01-01', 'route': 'BER-LHR',
                                             'last_7_bookings': [100]*7})
            self.assertGreaterEqual(result['predicted_bookings'], 0)
            with self.assertRaises(ValueError):
                forecast_request(model, {'date': '2026-01-01', 'route': 'BER-LHR',
                                         'last_7_bookings': [float('nan')]*7})

    def test_drift_and_finite_predictions(self):
        x, y, *_ = dataset(generate(60))
        m = fit(x, y, 1)
        self.assertTrue(np.isfinite(predict(m, x)).all())
        self.assertFalse(drift(m, x)['alert'])
        self.assertTrue(drift(m, x + 1000)['alert'])


if __name__ == '__main__':
    unittest.main()
