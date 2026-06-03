#!/usr/bin/env python3
"""Minimal HTTP API for the ARA-Net v6 research ensemble.

Run:
    python deployment/research_api.py --port 8080

POST /predict with JSON:
{
  "unit": "subject",
  "rows": [
    {
      "subject_id": "example_001",
      "scan_id": "scan_001",
      "<base_model>__prob_CN": 0.8,
      "<base_model>__prob_MCI": 0.15,
      "<base_model>__prob_AD": 0.05
    }
  ]
}
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from research_inference import (
    aggregate_subjects,
    ensemble_scan_probabilities,
    format_prediction_rows,
    json_payload,
    load_config,
    validate_rows,
)


def make_handler(config: dict):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": config["name"], "version": config["version"]})
                return
            self._send(404, {"error": "Use POST /predict or GET /health."})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send(404, {"error": "Unknown endpoint."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                rows = request.get("rows", [])
                if not isinstance(rows, list):
                    raise ValueError("`rows` must be a list of objects.")
                unit = request.get("unit", "subject")
                if unit not in {"scan", "subject"}:
                    raise ValueError("`unit` must be `scan` or `subject`.")
                validate_rows(rows, config)
                scan_probs = ensemble_scan_probabilities(rows, config)
                if unit == "scan":
                    predictions = format_prediction_rows(rows, scan_probs, config)
                else:
                    predictions = aggregate_subjects(rows, scan_probs, config)
                self._send(200, json_payload(predictions, config))
            except Exception as exc:  # noqa: BLE001 - API returns validation errors.
                self._send(400, {"error": str(exc), "clinical_use_notice": config["clinical_use_notice"]})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("deployment/final_ensemble_config.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    config = load_config(args.config)
    server = HTTPServer((args.host, args.port), make_handler(config))
    print(f"[serving] http://{args.host}:{args.port}")
    print(config["clinical_use_notice"])
    server.serve_forever()


if __name__ == "__main__":
    main()
