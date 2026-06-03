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
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from research_inference import (
    aggregate_subjects,
    ensemble_scan_probabilities,
    format_prediction_rows,
    json_payload,
    load_config,
    validate_rows,
)


def metadata_payload(config: dict) -> dict:
    return {
        "model": config["name"],
        "version": config["version"],
        "classes": config["classes"],
        "base_models": config["base_models"],
        "primary_evaluation": config.get("primary_evaluation", {}),
        "intended_use": config["intended_use"],
        "not_intended_for": config["not_intended_for"],
        "clinical_use_notice": config["clinical_use_notice"],
    }


def safe_public_path(base_dir: Path, relative_path: str) -> Path | None:
    base = base_dir.resolve()
    target = (base / relative_path).resolve()
    if target == base or base in target.parents:
        return target
    return None


def make_handler(config: dict, static_dir: Path | None = None, examples_dir: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _public_file(self, request_path: str) -> Path | None:
            route = unquote(request_path)
            if static_dir and route in {"/", "/index.html"}:
                return safe_public_path(static_dir, "index.html")
            if static_dir and route in {"/app.js", "/styles.css"}:
                return safe_public_path(static_dir, route.lstrip("/"))
            if static_dir and route.startswith("/assets/"):
                return safe_public_path(static_dir, route.lstrip("/"))
            if examples_dir and route.startswith("/examples/"):
                return safe_public_path(examples_dir, route.removeprefix("/examples/"))
            return None

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/health":
                self._send_json(200, {"status": "ok", "model": config["name"], "version": config["version"]})
                return
            if route == "/metadata":
                self._send_json(200, metadata_payload(config))
                return
            file_path = self._public_file(route)
            if file_path and file_path.is_file():
                content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
                if file_path.suffix == ".js":
                    content_type = "text/javascript; charset=utf-8"
                elif file_path.suffix in {".html", ".css", ".csv"}:
                    content_type = f"{content_type}; charset=utf-8"
                self._send_bytes(200, file_path.read_bytes(), content_type)
                return
            self._send_json(404, {"error": "Use GET /, GET /health, GET /metadata, or POST /predict."})

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            if route != "/predict":
                self._send_json(404, {"error": "Unknown endpoint."})
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
                self._send_json(200, json_payload(predictions, config))
            except Exception as exc:  # noqa: BLE001 - API returns validation errors.
                self._send_json(400, {"error": str(exc), "clinical_use_notice": config["clinical_use_notice"]})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("deployment/final_ensemble_config.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--static-dir", type=Path, default=Path("frontend"))
    parser.add_argument("--examples-dir", type=Path, default=Path("examples"))
    args = parser.parse_args()

    config = load_config(args.config)
    static_dir = args.static_dir if args.static_dir.exists() else None
    examples_dir = args.examples_dir if args.examples_dir.exists() else None
    server = HTTPServer((args.host, args.port), make_handler(config, static_dir, examples_dir))
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"[serving] http://{args.host}:{args.port}")
    if static_dir:
        print(f"[frontend] http://{display_host}:{args.port}/")
    print(config["clinical_use_notice"])
    server.serve_forever()


if __name__ == "__main__":
    main()
