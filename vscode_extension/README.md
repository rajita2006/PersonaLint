# PersonaLint VS Code Extension -- Demo Setup

This is Module 5: a live VS Code linter that flags code unusual for a
specific developer's own history, using the fingerprint data you already
built (Modules 1-4).

## One-time setup (do this before your demo)

```bash
# 1. Python side: install Flask if you haven't
pip install flask --break-system-packages   # or just: pip install flask

# 2. Extension side: install Node dependencies and compile
cd vscode_extension
npm install
npm run compile
```

## Every time you demo

**Step 1 -- start the local analysis server** (leave this terminal running):
```bash
cd vscode_extension/server
python3 flask_api.py
```
You should see `Starting server on http://localhost:5050 ...` and a count
of developer fingerprints loaded. This reads from `data/fingerprints/`,
so make sure you've already run `build_corpus.py` at least once.

**Step 2 -- launch the extension in a VS Code Extension Development Host:**
1. Open the `vscode_extension` folder in VS Code (`code vscode_extension`)
2. Press `F5` (or Run > Start Debugging). This opens a second VS Code
   window with PersonaLint active.
3. In that new window, open any `.py` file (e.g. one of the cloned repos
   in `data/repos/`, if you still have them locally).

**Step 3 -- set which developer's fingerprint to use** (so the demo shows
real personalization, not just the population baseline):
1. In the Extension Development Host window: `Cmd+,` to open Settings
2. Search "personalint"
3. Set `Developer Id` to one from your corpus, e.g. `Ben Darnell` or
   `Nate Prewitt` (must match a name in `data/fingerprints/*.json` exactly)

**Step 4 -- trigger analysis:** save the file (`Cmd+S`). Flagged functions
get a yellow squiggly underline; hover over one to see the personalized
explanation (which metric deviated, by how much, compared to that
developer's own typical values). Flags also appear in the Problems panel
(`Cmd+Shift+M`).

## Good demo move

Open the same file with two different `Developer Id` settings (e.g. one
with a tight, low-nesting fingerprint vs. one with a looser style) and
save both times -- show that the exact same function gets flagged for one
developer and not the other. That's the entire thesis of the paper,
visible live in the editor.

## If nothing gets flagged

- Check the Flask server terminal is still running and shows no errors
- Check `Developer Id` in settings exactly matches a name under
  `data/fingerprints/` (case-sensitive)
- Try a deliberately messy function (deeply nested, many params, bad names)
  to confirm the pipeline works end-to-end before trusting a "clean" result
