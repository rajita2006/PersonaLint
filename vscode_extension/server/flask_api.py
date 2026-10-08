"""
Module 5 (part 1) -- Flask API

Exposes Modules 2 (AST feature extraction) and 4 (personalized z-score
detector) as a single local HTTP endpoint, /analyze. The VS Code
extension calls this on every file save.

Everything stays on localhost -- nothing is sent externally. Run this
once, in the background, before using the extension:

    python3 flask_api.py

Then in VS Code, saving a .py file sends its content here and gets back
a list of flagged functions with human-readable reasons.
"""

import json
import os
import sys

from flask import Flask, request, jsonify

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "feature_extraction"))
from extractor import extract_functions  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
LABELED_PATH = os.path.join(DATA_DIR, "labeled_corpus.json")
FINGERPRINTS_DIR = os.path.join(DATA_DIR, "fingerprints")

METRICS = ["loc", "max_nesting_depth", "num_params", "num_branches", "naming_quality"]
LOWER_IS_WORSE = {"naming_quality"}
Z_THRESHOLD = 2.0

METRIC_LABELS = {
    "loc": "length",
    "max_nesting_depth": "nesting depth",
    "num_params": "parameter count",
    "num_branches": "branching complexity",
    "naming_quality": "variable naming clarity",
}

app = Flask(__name__)

# Loaded once at startup, reused across requests.
_population_baseline = None
_developer_fingerprints = {}  # developer_id -> {metric: {"mean":..., "std":...}}


def load_population_baseline():
    """Fallback baseline for any developer without their own fingerprint yet
    (the 'cold start' case discussed in the paper's limitations)."""
    with open(LABELED_PATH) as f:
        data = json.load(f)
    baseline = {}
    for m in METRICS:
        values = [d[m] for d in data]
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        baseline[m] = {"mean": mean, "std": var ** 0.5 or 1e-6}
    return baseline


def load_developer_fingerprints():
    """Loads every pre-built fingerprint from data/fingerprints/*.json so
    any developer in the corpus can be selected in the extension's settings."""
    fingerprints = {}
    if not os.path.isdir(FINGERPRINTS_DIR):
        return fingerprints
    for fname in os.listdir(FINGERPRINTS_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(FINGERPRINTS_DIR, fname)) as f:
            fp = json.load(f)
        baseline = {}
        for m in METRICS:
            baseline[m] = {"mean": fp["baseline"][m], "std": fp["std"][m] or 1e-6}
        fingerprints[fp["author"]] = baseline
    return fingerprints


def flag_function(func, baseline):
    """Returns a list of human-readable reasons this function deviates from
    baseline by more than Z_THRESHOLD standard deviations. Empty list = clean."""
    reasons = []
    for m in METRICS:
        mean, std = baseline[m]["mean"], baseline[m]["std"]
        z = (func[m] - mean) / std
        label = METRIC_LABELS[m]
        if m in LOWER_IS_WORSE:
            if z < -Z_THRESHOLD:
                reasons.append(
                    f"{label} is unusually low ({func[m]:.2f} vs your typical {mean:.2f}, "
                    f"{abs(z):.1f} std below your norm)"
                )
        else:
            if z > Z_THRESHOLD:
                reasons.append(
                    f"{label} is unusually high ({func[m]:.2f} vs your typical {mean:.2f}, "
                    f"{z:.1f} std above your norm)"
                )
    return reasons


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(force=True)
    code = payload.get("code", "")
    developer_id = payload.get("developer_id")
    file_path = payload.get("file_path", "<buffer>")

    baseline = _developer_fingerprints.get(developer_id, _population_baseline)
    used_personalized = developer_id in _developer_fingerprints

    functions = extract_functions(code, file_path=file_path)
    results = []
    for func in functions:
        reasons = flag_function(func, baseline)
        results.append({
            "function_name": func["function_name"],
            "lineno": func["lineno"],
            "flagged": len(reasons) > 0,
            "reasons": reasons,
        })

    return jsonify({
        "developer_id": developer_id,
        "used_personalized_baseline": used_personalized,
        "functions": results,
    })


@app.route("/developers", methods=["GET"])
def list_developers():
    """Lets the extension show a dropdown of known developers to personalize against."""
    return jsonify({"developers": sorted(_developer_fingerprints.keys())})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "developers_loaded": len(_developer_fingerprints)})


if __name__ == "__main__":
    print("Loading population baseline...")
    _population_baseline = load_population_baseline()
    print("Loading developer fingerprints...")
    _developer_fingerprints = load_developer_fingerprints()
    print(f"Loaded {len(_developer_fingerprints)} developer fingerprints.")
    print("Starting server on http://localhost:5050 ...")
    app.run(host="127.0.0.1", port=5050, debug=False)
