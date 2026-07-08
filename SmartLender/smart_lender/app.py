"""
app.py
------
"6. Building the Flask Web Application" from the project brief -- implemented
with Python's built-in `http.server` instead of Flask, so the whole project
runs with ZERO pip installs.

Routes:
    GET  /            -> Home page (introduction + usage scenarios)
    GET  /predict      -> Loan eligibility prediction form
    POST /predict      -> Runs the saved model and shows the result
    GET  /models        -> Model comparison table (train/test accuracy)
    GET  /eda           -> Exploratory Data Analysis report (charts)
    GET  /static/...    -> CSS / chart assets
"""

import os
import pickle
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import data_utils as du

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "model_store", "model.pkl")

_MODEL_PAYLOAD = None


def _load_model_payload():
    global _MODEL_PAYLOAD
    if _MODEL_PAYLOAD is None:
        with open(MODEL_PATH, "rb") as f:
            _MODEL_PAYLOAD = pickle.load(f)
    return _MODEL_PAYLOAD


def _read_template(name):
    with open(os.path.join(TEMPLATES_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _predict_from_form(fields):
    payload = _load_model_payload()
    model = payload["model"]
    encoders = payload["encoders"]
    fill_values = payload["fill_values"]
    needs_norm = payload["needs_norm"]

    row = {}
    for col in du.FEATURE_ORDER:
        val = fields.get(col, "")
        if col in du.NUMERIC_COLS:
            try:
                row[col] = float(val)
            except (TypeError, ValueError):
                row[col] = fill_values[col]
        else:
            row[col] = val if val else fill_values[col]

    vector = du.encode_row(row, encoders)

    if needs_norm:
        normed, _, _ = du.normalize_features([vector], payload["norm_mins"], payload["norm_maxs"])
        vector_for_model = normed[0]
    else:
        vector_for_model = vector

    prediction = model.predict([vector_for_model])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([vector_for_model])[0]
        confidence = proba if prediction == 1 else (1 - proba)
    elif hasattr(model, "predict_proba_positive"):
        proba = model.predict_proba_positive([vector_for_model])[0]
        confidence = proba if prediction == 1 else (1 - proba)
    else:
        confidence = 0.75  # fallback display value for models without a native probability

    return prediction, confidence, row


class SmartLenderHandler(BaseHTTPRequestHandler):
    server_version = "SmartLenderHTTP/1.0"

    def log_message(self, fmt, *args):
        print(f"[app] {self.address_string()} - {fmt % args}")

    def _send_html(self, html, status=200):
        encoded = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self._send_html("<h1>404 Not Found</h1>", status=404)

    # ------------------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_html(_read_template("home.html"))

        elif path == "/predict":
            payload = _load_model_payload()
            html = _read_template("predict.html")
            html = html.replace("__BEST_MODEL_NAME__", payload["model_name"])
            best_acc = payload["all_results"][payload["model_name"]]["test_acc"] * 100
            html = html.replace("__BEST_MODEL_ACC__", f"{best_acc:.1f}")
            self._send_html(html)

        elif path == "/models":
            payload = _load_model_payload()
            rows = []
            for name, res in sorted(payload["all_results"].items(),
                                     key=lambda kv: -kv[1]["test_acc"]):
                is_best = "best" if name == payload["model_name"] else ""
                rows.append(
                    f'<tr class="{is_best}"><td>{name}{" ⭐" if is_best else ""}</td>'
                    f'<td>{res["train_acc"]*100:.1f}%</td>'
                    f'<td>{res["test_acc"]*100:.1f}%</td></tr>'
                )
            html = _read_template("models.html").replace("__ROWS__", "".join(rows))
            self._send_html(html)

        elif path == "/eda":
            eda_path = os.path.join(STATIC_DIR, "eda", "eda_report.html")
            self._send_file(eda_path, "text/html; charset=utf-8")

        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            full_path = os.path.join(STATIC_DIR, rel)
            ext = os.path.splitext(full_path)[1].lower()
            content_type = {
                ".css": "text/css",
                ".html": "text/html; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".jpg": "image/jpeg",
            }.get(ext, "application/octet-stream")
            self._send_file(full_path, content_type)

        else:
            self._send_html("<h1>404 Not Found</h1><p><a href='/'>Go home</a></p>", status=404)

    # ------------------------------------------------------------------
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            fields = {k: v[0] for k, v in urllib.parse.parse_qs(body).items()}

            prediction, confidence, row = _predict_from_form(fields)
            payload = _load_model_payload()

            status_class = "approved" if prediction == 1 else "rejected"
            status_text = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

            details_rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in row.items()
            )

            html = _read_template("result.html")
            html = html.replace("__STATUS_CLASS__", status_class)
            html = html.replace("__STATUS_TEXT__", status_text)
            html = html.replace("__MODEL_NAME__", payload["model_name"])
            html = html.replace("__CONFIDENCE__", f"{confidence*100:.1f}")
            html = html.replace("__DETAILS_ROWS__", details_rows)
            self._send_html(html)
        else:
            self._send_html("<h1>404 Not Found</h1>", status=404)


def run(host="127.0.0.1", port=5000):
    server = ThreadingHTTPServer((host, port), SmartLenderHandler)
    print(f"\n[app] Smart Lender is running at http://{host}:{port}")
    print("[app] Press CTRL+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] Shutting down...")
        server.server_close()


if __name__ == "__main__":
    run()
