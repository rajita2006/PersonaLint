# PersonaLint

An LLM-augmented, developer-personalized code smell detector.
Flags code that's unusual *for a specific developer's own history*,
rather than against a universal rule set — and explains why in terms
of that developer's own baseline.

Status: Modules 1–3 (data collection, AST feature extraction, fingerprint
builder) are implemented and tested end-to-end below. Modules 4–5 and the
LLM explainer are next.

---

## Setup on your Mac (Apple Silicon, works fine on M-series incl. M5)

```bash
# 1. Unzip this project somewhere, then:
cd personalint

# 2. Create a virtual environment (recommended over global pip installs)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

No GPU, no special hardware needed for any of this — git parsing, AST
walking, and a small Random Forest are all trivial on any modern laptop.
The only network-dependent step is `git clone`-ing repos to analyze, and
later, API calls to an LLM for explanations.

---

## Verified working (tested on this machine before handing off to you)

Ran the full Module 1 → 3 pipeline against a real GitHub repo
(`psf/requests`) filtered to one real contributor's 60 most recent
commits. Result: 5,338 functions extracted, fingerprint built successfully.

```bash
# Clone a repo to analyze
git clone https://github.com/psf/requests.git data/test_repo

# Module 1: collect one developer's commits + file snapshots
cd data_collection
python3 collect_commits.py \
  --repo ../data/test_repo \
  --author "Nate Prewitt" \
  --out ../data/nate_commits.json \
  --max-commits 60

# Module 3 (runs Module 2 internally): build their personal fingerprint
cd ../fingerprint
python3 build_fingerprint.py \
  --commits ../data/nate_commits.json \
  --out ../data/nate_fingerprint.json
```

Output baseline for that developer:
```
loc: 12.20, max_nesting_depth: 0.63, num_params: 1.82, num_branches: 1.13, naming_quality: 0.84
```

That's a real personal coding-style fingerprint, derived from real commit
history — this is the object Module 4 will compare new code against.

---

## Project layout

```
personalint/
├── data_collection/       # Module 1 — GitPython commit walker
│   └── collect_commits.py
├── feature_extraction/    # Module 2 — AST-based metric extractor
│   └── extractor.py
├── fingerprint/           # Module 3 — per-developer baseline builder
│   └── build_fingerprint.py
├── classifier/            # Module 4 — TODO: threshold + Random Forest
├── llm_explainer/         # TODO: LLM-generated, personalized explanations
├── vscode_extension/      # Module 5 — TODO: TypeScript extension + Flask API
├── data/                  # local scratch space for cloned repos & JSON output
└── requirements.txt
```

---

## Next steps (in order)

1. **Module 4 — classifier.** Two variants to build and compare:
   - Simple threshold: flag if a metric exceeds the developer's `baseline[metric]`
     by more than X% (start at 20%, but sweep 10–40% for the paper).
   - Random Forest trained on `all_function_features` from multiple developers'
     fingerprints, using pseudo-labels from `radon`/`pylint` as targets.
2. **Baseline for comparison** — run `radon` and `pylint` on the same functions
   with fixed, non-personalized thresholds. This is your Arm A in the paper.
3. **Dataset expansion** — repeat the collect_commits → build_fingerprint
   pipeline across 8–12 real repos and their most active contributors.
4. **LLM explainer** — given a flagged function + the developer's fingerprint,
   ask an LLM to explain the deviation in terms of *their own* norms, not
   generic advice.
5. **VS Code extension** — Flask API wrapping Modules 2/4/LLM-explainer,
   TypeScript extension calling it on file save.
6. **Evaluation** — hand-label a sample, compute precision/recall/F1 for
   each arm, run the small user study comparing generic vs. personalized
   explanations.

See the "final idea" and evaluation design discussed in the research plan
for the full paper structure and dataset sourcing options.
