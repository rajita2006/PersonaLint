"""
evaluate_against_human_labels.py -- The real test. Compares universal
threshold, personalized threshold, and the Random Forest classifier
against actual human judgment (data/hand_labels_*.json), not the
circular auto-generated labels used in classifier.py.

This also reports how often the auto-labeler (radon/fixed-threshold)
agreed with the human labeler in the first place -- a direct, honest
measure of how good the pseudo-labels were as a proxy.

Usage:
    python evaluate_against_human_labels.py --hand-labels ../data/hand_labels_rajita.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LABELED_PATH = os.path.join(DATA_DIR, "labeled_corpus.json")

METRICS = ["loc", "max_nesting_depth", "num_params", "num_branches", "naming_quality"]
LOWER_IS_WORSE = {"naming_quality"}
DEFAULT_Z = 2.0


def compute_population_baseline(data):
    baseline = {}
    for m in METRICS:
        values = [d[m] for d in data]
        baseline[m] = {"mean": sum(values) / len(values), "std": float(np.std(values)) or 1e-6}
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
            baselines[dev][m] = {"mean": sum(values) / len(values), "std": float(np.std(values)) or 1e-6}
    return baselines


def flag_zscore(func, baseline, k):
    for m in METRICS:
        mean, std = baseline[m]["mean"], baseline[m]["std"]
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
    return {"label": label, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "false_positive_rate": round(fpr, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def build_exclusion_key(d):
    """Same key format used in hand_label.py to identify a specific function
    instance -- lets us exclude the hand-labeled functions from training/
    baseline data so we're not testing on data the model has seen."""
    return f"{d['repo']}::{d['commit_sha']}::{d['file_path']}::{d['function_name']}::{d['lineno']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-labels", required=True)
    parser.add_argument("--second-labeler", default=None,
                         help="Optional second hand-labels file, for inter-rater agreement (Cohen's kappa)")
    args = parser.parse_args()

    with open(args.hand_labels) as f:
        hand_labels = json.load(f)

    # "unsure" is excluded from scoring, per the rubric -- it's kept in the
    # file for discussion but doesn't count as ground truth either way.
    scored = [h for h in hand_labels if h["human_label"] in ("smelly", "clean")]
    n_unsure = len(hand_labels) - len(scored)
    print(f"Loaded {len(hand_labels)} hand labels ({len(scored)} scored, {n_unsure} unsure/excluded)")

    y_true = [1 if h["human_label"] == "smelly" else 0 for h in scored]

    # --- How good were the pseudo-labels in the first place? ----------------
    y_auto = [1 if h.get("_auto_label") == "smelly" else 0 for h in scored]
    auto_metrics = evaluate(y_true, y_auto, "auto_pseudo_label_vs_human")
    print()
    print("Pseudo-label (radon/fixed-threshold) agreement with human judgment:")
    print(f"  Precision={auto_metrics['precision']:.3f}  Recall={auto_metrics['recall']:.3f}  "
          f"F1={auto_metrics['f1']:.3f}")
    print(f"  (This is the honesty check -- low numbers here justify why hand-labeling was necessary)")

    # --- Load full corpus for baselines --------------------------------------
    # CRITICAL: exclude the hand-labeled functions themselves from the
    # baseline/training data. They were sampled FROM this corpus, so leaving
    # them in means the model (RF) and the reference baselines (universal,
    # personalized) would be evaluated on data they were partly built from --
    # data leakage that inflates every result below. We hold them out.
    print()
    print("Loading full corpus for population/developer baselines...")
    with open(LABELED_PATH) as f:
        full_data_all = json.load(f)

    hand_labeled_keys = {build_exclusion_key(h) for h in hand_labels}
    full_data = [d for d in full_data_all if build_exclusion_key(d) not in hand_labeled_keys]
    n_excluded = len(full_data_all) - len(full_data)
    print(f"  Excluded {n_excluded} functions from baselines/training "
          f"(the hand-labeled sample) to avoid data leakage")
    print(f"  Training/baseline corpus: {len(full_data)} functions (was {len(full_data_all)})")

    population_baseline = compute_population_baseline(full_data)
    developer_baselines = compute_developer_baselines(full_data)

    # --- Universal vs personalized threshold, against REAL human labels -----
    print()
    print(f"Universal vs personalized threshold (z={DEFAULT_Z}), evaluated against human labels:")
    y_pred_universal = [1 if flag_zscore(h, population_baseline, DEFAULT_Z) else 0 for h in scored]
    y_pred_personal = [
        1 if flag_zscore(h, developer_baselines.get(h["developer_id"], population_baseline), DEFAULT_Z) else 0
        for h in scored
    ]
    universal_metrics = evaluate(y_true, y_pred_universal, "universal_vs_human")
    personal_metrics = evaluate(y_true, y_pred_personal, "personalized_vs_human")
    print(f"  Universal:    P={universal_metrics['precision']:.3f} R={universal_metrics['recall']:.3f} "
          f"F1={universal_metrics['f1']:.3f} FPR={universal_metrics['false_positive_rate']:.3f}")
    print(f"  Personalized: P={personal_metrics['precision']:.3f} R={personal_metrics['recall']:.3f} "
          f"F1={personal_metrics['f1']:.3f} FPR={personal_metrics['false_positive_rate']:.3f}")

    # --- Statistical significance: is universal vs personalized a real
    # difference, or could it be noise at this sample size? McNemar's test
    # is the standard choice for comparing two classifiers on the SAME
    # test set (paired predictions), using only the cases where they
    # disagreed with each other.
    from scipy.stats import binomtest
    correct_universal = [1 if p == t else 0 for p, t in zip(y_pred_universal, y_true)]
    correct_personal = [1 if p == t else 0 for p, t in zip(y_pred_personal, y_true)]
    # n01: universal wrong, personalized right. n10: universal right, personalized wrong.
    n01 = sum(1 for cu, cp in zip(correct_universal, correct_personal) if cu == 0 and cp == 1)
    n10 = sum(1 for cu, cp in zip(correct_universal, correct_personal) if cu == 1 and cp == 0)
    n_discordant = n01 + n10
    if n_discordant > 0:
        mcnemar_p = binomtest(min(n01, n10), n_discordant, 0.5).pvalue
    else:
        mcnemar_p = 1.0
    print()
    print(f"  McNemar's test (universal vs personalized, paired on same {len(y_true)} functions):")
    print(f"    Personalized-only-correct: {n01}  |  Universal-only-correct: {n10}  |  p-value: {mcnemar_p:.4f}")
    print(f"    {'SIGNIFICANT at p<0.05 -- personalization is a real effect, not noise' if mcnemar_p < 0.05 else 'NOT significant at p<0.05 at this sample size -- report as a trend, not a proven effect'}")

    # --- Random Forest, trained on full corpus MINUS the hand-labeled sample
    print()
    print("Random Forest (trained on corpus excluding hand-labeled sample) tested against human labels...")
    X_train = np.array([[d[m] for m in METRICS] for d in full_data])
    y_train = np.array([1 if d["label"] == "smelly" else 0 for d in full_data])
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    X_test = np.array([[h[m] for m in METRICS] for h in scored])
    y_pred_rf = clf.predict(X_test)
    rf_metrics = evaluate(y_true, y_pred_rf, "random_forest_vs_human")
    print(f"  RF: P={rf_metrics['precision']:.3f} R={rf_metrics['recall']:.3f} "
          f"F1={rf_metrics['f1']:.3f} FPR={rf_metrics['false_positive_rate']:.3f}")

    results = {
        "n_hand_labeled": len(hand_labels),
        "n_scored": len(scored),
        "n_unsure": n_unsure,
        "auto_label_vs_human": auto_metrics,
        "universal_vs_human": universal_metrics,
        "personalized_vs_human": personal_metrics,
        "random_forest_vs_human": rf_metrics,
        "mcnemar_universal_vs_personalized": {
            "n01_personalized_only_correct": n01,
            "n10_universal_only_correct": n10,
            "p_value": round(mcnemar_p, 4),
            "significant_at_0.05": bool(mcnemar_p < 0.05),
        },
    }

    # --- Inter-rater agreement, if a second labeler's file was provided -----
    if args.second_labeler:
        with open(args.second_labeler) as f:
            second = json.load(f)
        second_by_id = {h["_id"]: h["human_label"] for h in second if h["human_label"] in ("smelly", "clean")}
        first_by_id = {h["_id"]: h["human_label"] for h in scored}
        common_ids = set(first_by_id) & set(second_by_id)

        if common_ids:
            y1 = [1 if first_by_id[i] == "smelly" else 0 for i in common_ids]
            y2 = [1 if second_by_id[i] == "smelly" else 0 for i in common_ids]
            kappa = cohen_kappa_score(y1, y2)
            agreement_pct = sum(a == b for a, b in zip(y1, y2)) / len(y1)
            print()
            print(f"Inter-rater agreement on {len(common_ids)} shared functions:")
            print(f"  Raw agreement: {agreement_pct:.1%}")
            print(f"  Cohen's kappa: {kappa:.3f}")
            results["inter_rater"] = {
                "n_common": len(common_ids),
                "raw_agreement": round(agreement_pct, 4),
                "cohens_kappa": round(kappa, 4),
            }
        else:
            print("\nNo overlapping function IDs found between the two labelers' files.")

    out_path = os.path.join(DATA_DIR, "human_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved -> {out_path}")


if __name__ == "__main__":
    main()
