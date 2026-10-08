"""
hand_label.py -- Interactive tool for building real, human-judgment ground
truth. Samples functions from labeled_corpus.json, shows you the actual
source code (re-fetched from git) plus its metrics, and asks you to judge
smelly / clean / unsure yourself.

Why this matters: classifier.py showed that the automatic (radon/threshold)
labels structurally favor the universal-threshold detector, because both
are population-defined. Human judgment is not tied to a fixed cutoff, so
it's the fair, non-circular ground truth needed to actually test whether
personalization helps.

Sampling strategy: stratified, not purely random --
  - half the sample drawn from functions the auto-labeler called "smelly"
  - half drawn from functions it called "clean"
  - within each half, spread across different developers so no single
    coding style dominates your judgment calibration
This avoids wasting your labeling time on an all-clean sample (since only
~4.7% of functions are auto-flagged) while still giving you a real,
independently-judged evaluation set.

OPTIONAL: LLM-assisted mode (--assisted flag on the `label` command).
This is a real, citable technique -- "LLM-assisted annotation with human
verification" -- used to speed up manual labeling. When enabled, an LLM
suggests a label + one-line rationale BEFORE you decide; you just confirm
or override. You are still the labeler of record: the LLM's suggestion is
shown, never auto-saved without your input. Disclose this in your paper's
methodology section if you use it -- it's standard practice, not a way to
avoid real human judgment, and your override rate is itself a reportable
number (how often you disagreed with the suggestion).

Usage:
    # Step 1: build the sample (deterministic given --seed, so you and a
    # second labeler can label the exact same set independently)
    python hand_label.py sample --n 200 --seed 42 --out ../data/hand_label_sample.json

    # Step 2: label it interactively (re-run any time to resume progress)
    python hand_label.py label --sample ../data/hand_label_sample.json \\
        --repo-root ../data/repos --out ../data/hand_labels_rajita.json

    # Step 2b: same, but with LLM-assisted suggestions (requires
    # ANTHROPIC_API_KEY set in your environment)
    python hand_label.py label --sample ../data/hand_label_sample.json \\
        --repo-root ../data/repos --out ../data/hand_labels_rajita.json --assisted
"""

import argparse
import json
import os
import random
import sys
import urllib.request
import urllib.error

sys.path.append(os.path.dirname(__file__))
from git import Repo  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LABELED_PATH = os.path.join(DATA_DIR, "labeled_corpus.json")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"  # cheap + fast, good enough for a suggestion

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
GEMINI_MODEL = "gemini-3.6-flash"  # 2.5-flash was retired for new users; this is the current fast/cheap tier

RUBRIC = """
Judge this function the way an experienced reviewer would in code review.
Consider (not a checklist -- use your judgment):
  - Is it doing too many things at once?
  - Would a new team member need real effort to understand it?
  - Is the nesting/branching genuinely hard to follow, or just long but linear?
  - Are names uninformative in a way that actually hurts understanding?

  [1] Smelly    -- yes, this would get a comment in real code review
  [2] Clean     -- no real issue, even if a bit long
  [3] Unsure    -- genuinely ambiguous, skip scoring but keep for discussion
  [s] Skip      -- can't judge (e.g. needs more context) -- excluded from set
  [q] Quit      -- save progress and exit
"""


def cmd_sample(args):
    with open(LABELED_PATH) as f:
        data = json.load(f)

    smelly = [d for d in data if d["label"] == "smelly"]
    clean = [d for d in data if d["label"] == "clean"]

    rng = random.Random(args.seed)
    n_each = args.n // 2

    def stratified_pick(pool, n):
        # Spread the sample across developers rather than letting one
        # prolific developer dominate the labeling set.
        by_dev = {}
        for item in pool:
            by_dev.setdefault(item["developer_id"], []).append(item)
        devs = list(by_dev.keys())
        rng.shuffle(devs)

        picked = []
        i = 0
        while len(picked) < n and devs:
            dev = devs[i % len(devs)]
            bucket = by_dev[dev]
            if bucket:
                idx = rng.randrange(len(bucket))
                picked.append(bucket.pop(idx))
            else:
                devs.remove(dev)
                continue
            i += 1
        return picked

    sample = stratified_pick(smelly, n_each) + stratified_pick(clean, args.n - n_each)
    rng.shuffle(sample)  # so the labeler doesn't see auto-label order

    # Strip the auto-label before saving the sample file, so labelers
    # aren't biased by seeing it during the interactive session.
    blinded = []
    for item in sample:
        blind_item = {k: v for k, v in item.items() if k not in ("label", "triggered_rules")}
        blind_item["_auto_label"] = item["label"]  # kept for later comparison, not shown during labeling
        blinded.append(blind_item)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(blinded, f, indent=2)

    print(f"Sampled {len(blinded)} functions ({n_each} auto-smelly, {args.n - n_each} auto-clean, blinded)")
    print(f"-> {args.out}")
    print("Share this exact file with a second labeler for inter-rater agreement.")


def get_function_source(repo_root, repo_name, commit_sha, file_path, lineno):
    """Re-fetch the exact source lines for this function from git history,
    so the labeler sees real code, not just metrics."""
    repo_path = os.path.join(repo_root, repo_name)
    try:
        repo = Repo(repo_path)
        commit = repo.commit(commit_sha)
        blob = commit.tree / file_path
        content = blob.data_stream.read().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        # Show a window around the function start -- good enough for judgment
        # without needing to re-run full AST end-line detection here.
        start = max(0, lineno - 1)
        end = min(len(lines), lineno + 40)
        return "\n".join(lines[start:end])
    except Exception as e:
        return f"[Could not fetch source: {e}]"


def get_llm_suggestion_anthropic(prompt):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set in environment"

    body = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return parse_llm_json(text)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError, KeyError) as e:
        return None, f"Anthropic suggestion failed ({e}) -- labeling unassisted for this one"


def get_llm_suggestion_gemini(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY not set in environment"

    url = GEMINI_API_URL.format(model=GEMINI_MODEL, key=api_key)
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 500,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return parse_llm_json(text)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        return None, f"Gemini HTTP {e.code}: {error_body[:200]} -- labeling unassisted for this one"
    except (urllib.error.URLError, json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
        return None, f"Gemini suggestion failed ({e}) -- labeling unassisted for this one"


def parse_llm_json(text):
    """Parse the LLM's response. Expects a simple 'LABEL: ...' / 'RATIONALE: ...'
    format rather than strict JSON -- LLMs sometimes emit unescaped quotes or
    stray newlines inside a rationale sentence, which breaks json.loads even
    when the intent was valid JSON. This plain-text format is far more robust
    to parse reliably in a lightweight script."""
    text = text.strip()

    label = None
    rationale = ""
    for line in text.splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith("label:"):
            label = line.split(":", 1)[1].strip().lower().strip('."\'')
        elif lower.startswith("rationale:"):
            rationale = line.split(":", 1)[1].strip()

    if label not in ("smelly", "clean", "unsure"):
        # Fall back: try to find the word anywhere in the response
        for candidate in ("smelly", "clean", "unsure"):
            if candidate in text.lower():
                label = candidate
                break

    if label is None:
        raise ValueError(f"Could not parse a label from response: {text[:150]!r}")

    return label, rationale


def get_llm_suggestion(function_name, source, metrics, provider="anthropic"):
    """Ask an LLM for a suggested label + one-line rationale. Returns
    (label, rationale) or (None, error_message) if the call fails --
    the caller should fall back to unassisted labeling on failure rather
    than blocking the whole session on an API hiccup."""
    prompt = f"""You are reviewing a function the way an experienced code reviewer would.

Function: {function_name}
Metrics: loc={metrics['loc']} nesting={metrics['max_nesting_depth']} params={metrics['num_params']} branches={metrics['num_branches']} naming_quality={metrics['naming_quality']}

Source:
{source}

Judge whether this function is "smelly" (would get a real code review comment -- doing too much, hard to follow, genuinely confusing) or "clean" (fine, even if long) or "unsure" (genuinely ambiguous).

Respond in EXACTLY this format, nothing else, no markdown, no extra text:
LABEL: smelly
RATIONALE: one short sentence, no quotes or special characters"""

    if provider == "gemini":
        return get_llm_suggestion_gemini(prompt)
    return get_llm_suggestion_anthropic(prompt)


def cmd_label(args):
    with open(args.sample) as f:
        sample = json.load(f)

    existing = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = {item["_id"]: item for item in json.load(f)}

    results = list(existing.values())
    labeled_keys = set(existing.keys())
    n_agreed, n_overridden, n_suggested = 0, 0, 0

    for item in sample:
        key = f"{item['repo']}::{item['commit_sha']}::{item['file_path']}::{item['function_name']}::{item['lineno']}"
        if key in labeled_keys:
            continue

        print("\n" + "=" * 70)
        print(f"{item['function_name']}  ({item['file_path']}:{item['lineno']})")
        print(f"Developer: {item['developer_id']}")
        print("-" * 70)
        source = get_function_source(
            args.repo_root, item["repo"], item["commit_sha"], item["file_path"], item["lineno"]
        )
        print(source)
        print("-" * 70)
        print(f"Metrics: loc={item['loc']} nesting={item['max_nesting_depth']} "
              f"params={item['num_params']} branches={item['num_branches']} "
              f"naming_quality={item['naming_quality']}")

        suggestion = None
        if args.assisted:
            n_suggested += 1
            print(f"\n[Asking {args.provider} for a suggestion...]")
            suggestion, rationale = get_llm_suggestion(item["function_name"], source, item, provider=args.provider)
            if suggestion:
                print(f"  Suggested: {suggestion.upper()}  -- {rationale}")
                print("  (Press Enter to accept, or type 1/2/3/s/q to override)")
            else:
                print(f"  [{rationale}]")

        print(RUBRIC)

        prompt_text = "Your judgment [1/2/3/s/q, Enter=accept suggestion]: " if suggestion else "Your judgment [1/2/3/s/q]: "
        choice = input(prompt_text).strip().lower()

        if choice == "q":
            break
        if choice == "s":
            continue

        if choice == "" and suggestion:
            judgment = suggestion
            n_agreed += 1
        else:
            judgment_map = {"1": "smelly", "2": "clean", "3": "unsure"}
            judgment = judgment_map.get(choice)
            if judgment is None:
                print("Not a valid choice, skipping.")
                continue
            if suggestion and judgment != suggestion:
                n_overridden += 1
            elif suggestion:
                n_agreed += 1

        record = dict(item)
        record["_id"] = key
        record["human_label"] = judgment
        if suggestion:
            record["_llm_suggestion"] = suggestion
        results.append(record)
        labeled_keys.add(key)

        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} hand labels -> {args.out}")
    if n_suggested:
        print(f"LLM-assisted: {n_agreed} accepted, {n_overridden} overridden "
              f"({100*n_overridden/max(n_suggested,1):.0f}% override rate)")


def main():
    parser = argparse.ArgumentParser(description="Hand-label a ground truth sample for PersonaLint")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="Build a blinded, stratified sample to label")
    p_sample.add_argument("--n", type=int, default=200)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--out", required=True)
    p_sample.set_defaults(func=cmd_sample)

    p_label = sub.add_parser("label", help="Interactively label a sample")
    p_label.add_argument("--sample", required=True)
    p_label.add_argument("--repo-root", required=True, help="Directory containing cloned repos (data/repos)")
    p_label.add_argument("--out", required=True)
    p_label.add_argument("--assisted", action="store_true",
                          help="Show an LLM-suggested label before you decide")
    p_label.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic",
                          help="Which LLM to use for --assisted mode. "
                               "anthropic needs ANTHROPIC_API_KEY, gemini needs GEMINI_API_KEY")
    p_label.set_defaults(func=cmd_label)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
