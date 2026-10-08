"""
build_corpus.py -- runs the full Module 1 -> 3 pipeline across every
(repo, author) pair in corpus_config.py and produces:

    data/fingerprints/<repo>_<author>.json   -- one fingerprint per developer
    data/corpus_dataset.json                  -- every function from every
                                                  developer, in one flat file,
                                                  ready for Module 4 / labeling

This is the actual dataset-building step. Run it once your repos are
cloned into data/repos/<repo_name>.
"""

import json
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fingerprint"))

from corpus_config import CORPUS, MAX_COMMITS_PER_AUTHOR  # noqa: E402
from collect_commits import collect_commits  # noqa: E402
from build_fingerprint import build_fingerprint  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPOS_DIR = os.path.join(DATA_DIR, "repos")
FINGERPRINTS_DIR = os.path.join(DATA_DIR, "fingerprints")


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def main():
    os.makedirs(FINGERPRINTS_DIR, exist_ok=True)
    all_functions = []
    summary = []

    for repo_name, author in CORPUS:
        repo_path = os.path.join(REPOS_DIR, repo_name)
        if not os.path.isdir(repo_path):
            print(f"  [skip] repo not found: {repo_path}")
            continue

        print(f"Processing {repo_name} / {author} ...")
        try:
            commits = collect_commits(repo_path, author, max_commits=MAX_COMMITS_PER_AUTHOR)
            if not commits:
                print(f"  [skip] no commits found for {author} in {repo_name}")
                continue

            fingerprint = build_fingerprint(commits)
        except ValueError as e:
            print(f"  [skip] {e}")
            continue

        # Tag every function with repo + author + a unique developer_id so
        # Module 4 can group by developer for personalized thresholds.
        developer_id = f"{repo_name}::{author}"
        for func in fingerprint["all_function_features"]:
            func["repo"] = repo_name
            func["developer_id"] = developer_id
            all_functions.append(func)

        out_path = os.path.join(FINGERPRINTS_DIR, f"{safe_name(repo_name)}_{safe_name(author)}.json")
        with open(out_path, "w") as f:
            json.dump(fingerprint, f, indent=2)

        summary.append({
            "repo": repo_name,
            "author": author,
            "commits_analyzed": fingerprint["num_commits_analyzed"],
            "functions_analyzed": fingerprint["num_functions_analyzed"],
            "baseline": fingerprint["baseline"],
        })
        print(f"  -> {fingerprint['num_functions_analyzed']} functions, "
              f"fingerprint saved to {out_path}")

    combined_path = os.path.join(DATA_DIR, "corpus_dataset.json")
    with open(combined_path, "w") as f:
        json.dump(all_functions, f, indent=2)

    summary_path = os.path.join(DATA_DIR, "corpus_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Corpus complete: {len(all_functions)} total functions across "
          f"{len(summary)} developers.")
    print(f"Combined dataset -> {combined_path}")
    print(f"Per-developer summary -> {summary_path}")


if __name__ == "__main__":
    main()
