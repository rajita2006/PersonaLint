"""
Module 1 -- Data Collection

Walks a git repository's commit history and extracts, for a given author,
every Python file blob that existed at each of their commits. This gives
Module 3 (fingerprint builder) exact code snapshots to run feature
extraction against.

Usage:
    python collect_commits.py --repo /path/to/repo --author "Jane Doe" --out ../data/jane_commits.json
"""

import argparse
import json
import os
from datetime import datetime, timezone

from git import Repo
from git.exc import GitCommandError


def collect_commits(repo_path: str, author_name: str, max_commits: int | None = None):
    """
    Walk every commit in the repo, filter by author, and for each commit
    record which .py files changed plus their full content at that commit.

    Returns a list of dicts:
        {
            "sha": str,
            "timestamp": str (ISO 8601),
            "author": str,
            "message": str,
            "files": [{"path": str, "content": str}, ...]
        }
    """
    repo = Repo(repo_path)
    if repo.bare:
        raise ValueError(f"{repo_path} is a bare repo with no working tree history")

    records = []
    commit_iter = repo.iter_commits("HEAD")

    for commit in commit_iter:
        if author_name.lower() not in (commit.author.name or "").lower():
            continue

        # A commit's parent lets us diff to find exactly which files changed.
        # Root commits (no parent) are diffed against an empty tree.
        # NOTE: shallow clones (git clone --depth N) can leave a commit's
        # parent object missing from the local history -- guard against
        # GitCommandError from those "grafted" boundary commits and just
        # skip them rather than crashing the whole collection run.
        parents = commit.parents
        try:
            diffs = commit.diff(parents[0] if parents else None, create_patch=False)
        except GitCommandError:
            continue

        changed_py_files = []
        for diff_item in diffs:
            path = diff_item.b_path or diff_item.a_path
            if not path or not path.endswith(".py"):
                continue
            try:
                blob = commit.tree / path
                content = blob.data_stream.read().decode("utf-8", errors="ignore")
            except KeyError:
                # File was deleted in this commit -- nothing to extract features from.
                continue
            changed_py_files.append({"path": path, "content": content})

        if not changed_py_files:
            continue

        records.append({
            "sha": commit.hexsha,
            "timestamp": datetime.fromtimestamp(
                commit.authored_date, tz=timezone.utc
            ).isoformat(),
            "author": commit.author.name,
            "message": commit.message.strip().split("\n")[0],
            "files": changed_py_files,
        })

        if max_commits and len(records) >= max_commits:
            break

    return records


def main():
    parser = argparse.ArgumentParser(description="Collect per-author commit snapshots for PersonaLint")
    parser.add_argument("--repo", required=True, help="Path to a local clone of the git repo")
    parser.add_argument("--author", required=True, help="Author name (or substring) to filter by")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--max-commits", type=int, default=None, help="Optional cap on commits collected")
    args = parser.parse_args()

    records = collect_commits(args.repo, args.author, args.max_commits)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Collected {len(records)} commits by '{args.author}' -> {args.out}")


if __name__ == "__main__":
    main()
