"""Local demonstration API. Run training first. No external exposure intended."""
import json
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from forecast import ROUTES, predict, vector


def forecast_request(model, payload):
    d = date.fromisoformat(payload['date'])
    route = payload['route']
    past = payload['last_7_bookings']
    if route not in ROUTES or not isinstance(past, list) or len(past) != 7:
        raise ValueError('Known route and exactly seven booking counts required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not np.isfinite(v) or v < 0 or v != int(v) for v in past):
        raise ValueError('Counts must be finite nonnegative integers')
    if d <= date.fromisoformat(model['trained_through']):
        raise ValueError('Prediction date must be after training cutoff')
    return {'route': route, 'date': d.isoformat(),
            'predicted_bookings': round(float(predict(model, [vector(d, route, past)])[0]), 2),
            'model_trained_through': model['trained_through']}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.respond(200 if self.path == '/health' else 404,
                     {'status': 'ok'} if self.path == '/health' else {'error': 'not found'})

    def do_POST(self):
        if self.path != '/predict':
            self.respond(404, {'error': 'not found'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 10000:
                raise ValueError('Request must be 1–10000 bytes')
            payload = json.loads(self.rfile.read(length))
            result = forecast_request(self.server.model, payload)
        except (ValueError, KeyError, TypeError, OverflowError):
            self.respond(400, {'error': 'Invalid input: use date, supported route and last_7_bookings'})
            return
        self.respond(200, result)


if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', 8000), Handler)
    server.model = json.loads(Path('artifacts/model.json').read_text())
    print('Demo API: http://127.0.0.1:8000/health')
    server.serve_forever()
