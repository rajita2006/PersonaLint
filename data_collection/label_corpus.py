"""
label_corpus.py -- Pseudo-label every function in the corpus using radon,
a real, established, non-personalized static analysis tool. This produces:

  1. The ground-truth-proxy labels needed to evaluate your classifier
     (precision/recall/F1 against "does radon consider this smelly").
  2. Arm A of your paper's comparison -- the fixed, universal-threshold
     baseline that your personalized approach is being tested against.

We use radon's own complexity engine directly (pure Python, AST-based,
no subprocess needed -- fast even across 100k+ functions):

  - Cyclomatic Complexity (CC) via radon.complexity.cc_visit
    Standard McCabe interpretation: CC > 10 is commonly flagged as
    "complex" (radon's own rank C and above; ranks: A=1-5, B=6-10,
    C=11-20, D=21-30, E=31-40, F=41+).

  - Maintainability Index (MI) via radon.metrics.mi_visit, computed
    per-file (MI is a whole-file metric, not per-function) -- included
    for reference/discussion but not used as the per-function label.

A function is pseudo-labeled "smelly" if EITHER:
  - radon cyclomatic complexity rank is C or worse (CC > 10), OR
  - our own extracted max_nesting_depth > 4, OR
  - our own extracted loc > 50 (classic "Long Method" smell threshold)

These are standard, literature-grounded fixed thresholds (McCabe's CC>10;
Long Method as LOC>50 is used in multiple code-smell detection papers),
which is exactly what makes this a legitimate non-personalized baseline
to compare your personalized fingerprint approach against.

Usage:
    python label_corpus.py
"""

import json
import os
import sys

from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fingerprint"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))

from corpus_config import CORPUS, MAX_COMMITS_PER_AUTHOR  # noqa: E402
from collect_commits import collect_commits  # noqa: E402
from extractor import extract_functions  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPOS_DIR = os.path.join(DATA_DIR, "repos")

CC_SMELL_THRESHOLD = 10       # McCabe: CC > 10 => rank C or worse
NESTING_SMELL_THRESHOLD = 4   # commonly used "deeply nested" cutoff
LOC_SMELL_THRESHOLD = 50      # classic "Long Method" cutoff


def radon_cc_for_file(source_code: str) -> dict:
    """
    Returns {(function_name, lineno): complexity_int} for every function
    radon finds in this source file. Silently returns {} on parse failure
    (matches extractor.py's behavior for consistency).
    """
    try:
        blocks = cc_visit(source_code)
    except SyntaxError:
        return {}

    result = {}
    for block in blocks:
        # radon also reports methods nested under classes -- name and
        # lineno together give us a unique-enough key to match against
        # our own AST extractor's output for the same source string.
        result[(block.name, block.lineno)] = block.complexity
    return result


def label_function(ast_features: dict, radon_cc: int | None) -> dict:
    """Apply the fixed-threshold rules and record which rule(s) fired."""
    triggered = []

    if radon_cc is not None and radon_cc > CC_SMELL_THRESHOLD:
        triggered.append(f"high_complexity(CC={radon_cc})")

    if ast_features["max_nesting_depth"] > NESTING_SMELL_THRESHOLD:
        triggered.append(f"deep_nesting(depth={ast_features['max_nesting_depth']})")

    if ast_features["loc"] > LOC_SMELL_THRESHOLD:
        triggered.append(f"long_method(loc={ast_features['loc']})")

    return {
        "radon_cc": radon_cc,
        "label": "smelly" if triggered else "clean",
        "triggered_rules": triggered,
    }


def main():
    labeled_functions = []
    stats = {"total": 0, "smelly": 0, "clean": 0, "no_radon_match": 0}

    for repo_name, author in CORPUS:
        repo_path = os.path.join(REPOS_DIR, repo_name)
        if not os.path.isdir(repo_path):
            print(f"  [skip] repo not found: {repo_path}")
            continue

        print(f"Labeling {repo_name} / {author} ...")
        try:
            commits = collect_commits(repo_path, author, max_commits=MAX_COMMITS_PER_AUTHOR)
        except Exception as e:
            print(f"  [skip] {e}")
            continue

        if not commits:
            continue

        developer_id = f"{repo_name}::{author}"
        dev_count = 0

        for commit in commits:
            for file_entry in commit["files"]:
                content = file_entry["content"]
                path = file_entry["path"]

                ast_funcs = extract_functions(content, file_path=path)
                if not ast_funcs:
                    continue

                radon_map = radon_cc_for_file(content)

                for func in ast_funcs:
                    key = (func["function_name"], func["lineno"])
                    radon_cc = radon_map.get(key)
                    if radon_cc is None:
                        stats["no_radon_match"] += 1

                    labels = label_function(func, radon_cc)

                    record = {
                        **func,
                        **labels,
                        "repo": repo_name,
                        "developer_id": developer_id,
                        "commit_sha": commit["sha"],
                    }
                    labeled_functions.append(record)
                    dev_count += 1
                    stats["total"] += 1
                    stats[labels["label"] if labels["label"] == "smelly" else "clean"] += 1

        print(f"  -> {dev_count} functions labeled")

    out_path = os.path.join(DATA_DIR, "labeled_corpus.json")
    with open(out_path, "w") as f:
        json.dump(labeled_functions, f, indent=2)

    stats_path = os.path.join(DATA_DIR, "labeling_stats.json")
    stats["smelly_pct"] = round(100 * stats["smelly"] / stats["total"], 2) if stats["total"] else 0
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print(f"Labeled {stats['total']} functions.")
    print(f"  Smelly: {stats['smelly']} ({stats['smelly_pct']}%)")
    print(f"  Clean:  {stats['clean']}")
    print(f"  (radon had no per-function match for {stats['no_radon_match']} functions -- "
          f"labeled by nesting/LOC rules only)")
    print(f"Saved -> {out_path}")
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()
