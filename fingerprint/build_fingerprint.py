"""
Module 3 -- Fingerprint Builder

Bridges Modules 1 and 2. Takes the JSON output of collect_commits.py
(a developer's commit history with full file snapshots), runs every
snapshot through the AST feature extractor, and computes the average
of each metric across every function the developer has ever written.

That average vector is the developer's "fingerprint" -- their personal
typical-coding baseline, saved as JSON for Module 4 (classifier) to
compare new code against.

Usage:
    python build_fingerprint.py --commits ../data/jane_commits.json --out ../data/jane_fingerprint.json
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))
from extractor import extract_functions  # noqa: E402

METRICS = ["loc", "max_nesting_depth", "num_params", "num_branches", "naming_quality"]


def build_fingerprint(commits: list[dict]) -> dict:
    """
    commits: the list of commit records produced by collect_commits.py

    Returns:
        {
            "author": str,
            "num_commits_analyzed": int,
            "num_functions_analyzed": int,
            "baseline": {metric: mean_value, ...},
            "std": {metric: stdev_value, ...},   # useful later for z-score style thresholds
            "all_function_features": [...]        # kept for classifier training in Module 4
        }
    """
    if not commits:
        raise ValueError("No commits provided -- nothing to fingerprint")

    author = commits[0]["author"]
    all_features = []

    for commit in commits:
        for file_entry in commit["files"]:
            funcs = extract_functions(file_entry["content"], file_path=file_entry["path"])
            for func in funcs:
                func["commit_sha"] = commit["sha"]
                func["commit_timestamp"] = commit["timestamp"]
                all_features.append(func)

    if not all_features:
        raise ValueError(f"No functions could be extracted for author '{author}'")

    baseline = {}
    std = {}
    for metric in METRICS:
        values = [f[metric] for f in all_features]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
        baseline[metric] = round(mean, 4)
        std[metric] = round(variance ** 0.5, 4)

    return {
        "author": author,
        "num_commits_analyzed": len(commits),
        "num_functions_analyzed": len(all_features),
        "baseline": baseline,
        "std": std,
        "all_function_features": all_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Build a developer fingerprint from collected commits")
    parser.add_argument("--commits", required=True, help="Path to JSON produced by collect_commits.py")
    parser.add_argument("--out", required=True, help="Output fingerprint JSON path")
    args = parser.parse_args()

    with open(args.commits) as f:
        commits = json.load(f)

    fingerprint = build_fingerprint(commits)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fingerprint, f, indent=2)

    print(f"Fingerprint for {fingerprint['author']}: "
          f"{fingerprint['num_functions_analyzed']} functions across "
          f"{fingerprint['num_commits_analyzed']} commits -> {args.out}")
    print("Baseline:", fingerprint["baseline"])


if __name__ == "__main__":
    main()
