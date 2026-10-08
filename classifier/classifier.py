"""
Module 4 -- Classifier / Detectors

Implements and evaluates three detection strategies against the same
labeled_corpus.json ground truth:

  A. Universal threshold  -- flag if a metric deviates from the POPULATION
                              average (across all developers) by more than
                              a margin. This is your non-personalized baseline.

  B. Personalized threshold -- flag if a metric deviates from that specific
                              DEVELOPER'S OWN average by more than a margin.
                              This is your core novel method.

  C. Random Forest          -- trained on the 5 raw AST metrics to predict
                              the label directly, as a more sophisticated
                              alternative to simple averaging (grouped
                              train/test split by developer_id, so the
                              model is never tested on a developer's code
                              it partially trained on).

NOTE on methodology (important for your Threats to Validity section):
The labels in labeled_corpus.json were generated using fixed thresholds
on loc/nesting (+ radon CC), so Approach A and the labeling rule share
some feature overlap -- treat any near-perfect Approach A score with
appropriate skepticism. The more meaningful, non-circular comparison is
NOT "who best predicts the label" but "how does moving the reference
point from population-average to personal-average change the false
positive rate, while recall on real (radon-independent) smells like high
cyclomatic complexity is preserved." The script below reports both the
overall metrics AND a CC-only recall breakdown for exactly this reason.

Usage:
    python classifier.py
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LABELED_PATH = os.path.join(DATA_DIR, "labeled_corpus.json")
RESULTS_PATH = os.path.join(DATA_DIR, "classifier_results.json")

METRICS = ["loc", "max_nesting_depth", "num_params", "num_branches", "naming_quality"]
# naming_quality is the one metric where HIGHER is BETTER -- smell means
# falling below baseline, not exceeding it. Everything else, smell means
# exceeding baseline.
LOWER_IS_WORSE = {"naming_quality"}

DEFAULT_MARGIN = 0.20
MARGIN_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

# z-score based flagging: flag if a metric is more than k standard
# deviations from baseline. More statistically defensible than a flat
# percentage-of-mean margin, especially for low-mean/high-variance
# metrics like num_branches (mean ~1.25) where a %-margin over-triggers
# on trivially small absolute changes.
DEFAULT_Z = 2.0
Z_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_corpus():
    with open(LABELED_PATH) as f:
        return json.load(f)


def compute_population_baseline(data):
    baseline = {}
    for m in METRICS:
        values = [d[m] for d in data]
        baseline[m] = {
            "mean": sum(values) / len(values),
            "std": float(np.std(values)) or 1e-6,
        }
    return baseline


def compute_developer_baselines(data):
    by_dev = defaultdict(list)
    for d in data:
        by_dev[d["developer_id"]].append(d)

    baselines = {}
    for dev, funcs in by_dev.items():
        baselines[dev] = {}
        for m in METRICS:
            values = [f[m] for f in funcs]
            baselines[dev][m] = {
                "mean": sum(values) / len(values),
                "std": float(np.std(values)) or 1e-6,
            }
    return baselines


def flag_against_baseline_zscore(func, baseline, k):
    """Flag if any metric is more than k standard deviations from baseline,
    in the 'worse' direction (respecting LOWER_IS_WORSE metrics)."""
    for m in METRICS:
        mean = baseline[m]["mean"]
        std = baseline[m]["std"]
        z = (func[m] - mean) / std
        if m in LOWER_IS_WORSE:
            if z < -k:
                return True
        else:
            if z > k:
                return True
    return False


def evaluate(y_true, y_pred, label=""):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "label": label,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    print("Loading labeled corpus...")
    data = load_corpus()
    y_true = [1 if d["label"] == "smelly" else 0 for d in data]

    population_baseline = compute_population_baseline(data)
    developer_baselines = compute_developer_baselines(data)

    print(f"Population baseline: {[(k, round(v['mean'],2), round(v['std'],2)) for k,v in population_baseline.items()]}")
    print()

    results = {"z_sweep": [], "random_forest": None, "cc_only_recall": {}}

    # --- A vs B: z-score threshold sweep ------------------------------------
    print("Running z-score threshold sweep (universal vs personalized)...")
    for k in Z_SWEEP:
        y_pred_universal = [
            1 if flag_against_baseline_zscore(d, population_baseline, k) else 0
            for d in data
        ]
        y_pred_personal = [
            1 if flag_against_baseline_zscore(d, developer_baselines[d["developer_id"]], k) else 0
            for d in data
        ]

        universal_metrics = evaluate(y_true, y_pred_universal, f"universal_z_{k}")
        personal_metrics = evaluate(y_true, y_pred_personal, f"personalized_z_{k}")

        results["z_sweep"].append({
            "z": k,
            "universal": universal_metrics,
            "personalized": personal_metrics,
        })

        print(f"  k={k:.1f}  "
              f"universal: P={universal_metrics['precision']:.3f} R={universal_metrics['recall']:.3f} "
              f"FPR={universal_metrics['false_positive_rate']:.3f}  |  "
              f"personalized: P={personal_metrics['precision']:.3f} R={personal_metrics['recall']:.3f} "
              f"FPR={personal_metrics['false_positive_rate']:.3f}")

    # --- Non-circular check: recall specifically on CC-flagged functions ---
    # This isolates the part of the label NOT derived from loc/nesting,
    # so it's a fairer signal of real detection ability.
    print()
    print("CC-only recall check (functions flagged solely for high complexity)...")
    cc_only_idx = [i for i, d in enumerate(data) if d["radon_cc"] is not None and d["radon_cc"] > 10]
    if cc_only_idx:
        for k in [DEFAULT_Z]:
            y_pred_universal = [
                1 if flag_against_baseline_zscore(data[i], population_baseline, k) else 0
                for i in cc_only_idx
            ]
            y_pred_personal = [
                1 if flag_against_baseline_zscore(data[i], developer_baselines[data[i]["developer_id"]], k) else 0
                for i in cc_only_idx
            ]
            universal_recall = sum(y_pred_universal) / len(y_pred_universal)
            personal_recall = sum(y_pred_personal) / len(y_pred_personal)
            results["cc_only_recall"] = {
                "n_high_complexity_functions": len(cc_only_idx),
                "universal_recall": round(universal_recall, 4),
                "personalized_recall": round(personal_recall, 4),
            }
            print(f"  n={len(cc_only_idx)} high-CC functions | "
                  f"universal catches {universal_recall:.1%} | "
                  f"personalized catches {personal_recall:.1%}")

    # --- C: Random Forest, grouped split by developer -----------------------
    print()
    print("Training Random Forest (grouped split by developer_id)...")
    X = np.array([[d[m] for m in METRICS] for d in data])
    y = np.array(y_true)
    groups = np.array([d["developer_id"] for d in data])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])
    y_pred_rf = clf.predict(X[test_idx])

    rf_metrics = evaluate(y[test_idx], y_pred_rf, "random_forest")
    rf_metrics["feature_importances"] = {
        m: round(float(imp), 4) for m, imp in zip(METRICS, clf.feature_importances_)
    }
    rf_metrics["n_train"] = len(train_idx)
    rf_metrics["n_test"] = len(test_idx)
    rf_metrics["n_developers_train"] = len(set(groups[train_idx]))
    rf_metrics["n_developers_test"] = len(set(groups[test_idx]))
    results["random_forest"] = rf_metrics

    print(f"  RF: P={rf_metrics['precision']:.3f} R={rf_metrics['recall']:.3f} "
          f"F1={rf_metrics['f1']:.3f} FPR={rf_metrics['false_positive_rate']:.3f}")
    print(f"  Feature importances: {rf_metrics['feature_importances']}")
    print(f"  Train devs: {rf_metrics['n_developers_train']}, Test devs: {rf_metrics['n_developers_test']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Full results saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
