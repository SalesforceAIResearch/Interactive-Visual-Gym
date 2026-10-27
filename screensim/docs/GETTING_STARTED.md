# Getting started

Run these commands from `interactive_visual_gym/screensim/`. ScreenSim does not require a separately running simulator.

## 1. Install

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all,dev]"
python -m playwright install chromium
```

On Linux, Chromium also needs system libraries. `python -m playwright install-deps chromium --dry-run` lists them; installation may require administrator access.

## 2. Check the installation

These commands do not call models:

```bash
screensim-case
screensim-case --episodes
screensim-roster
python -m pytest -q
```

## 3. Configure credentials

Export `GEMINI_API_KEY` for the simulated user and Gemini assistants. Export `OPENAI_API_KEY` for OpenAI assistants. The user client reads environment variables first; `~/.gemini_key` and `~/.openai_key` remain supported as legacy fallbacks. Never commit credentials.

Interaction-quality grading defaults to Gemini and GPT. See [grading](INTERACTION_QUALITY_GRADING.md) for the judge clients' credential requirements.

## 4. Run one episode

```bash
export GEMINI_API_KEY=...
screensim-episode --task hearing_setup \
    --episode hearing_setup__c2 --persona baseline \
    --assistant gemini-3.8-flash --out runs/demo
```

This selects the requested case and persona using the assistant's roster configuration. Add `--dry-run` to inspect the selection without creating records, opening a browser, or calling models. Without an explicit case, the command uses the task's first designed case; the default persona is `classic_novice`.

## 5. Inspect the record

`--out` accepts either a directory or an explicit `.json` summary filename. The command above produces:

- `runs/demo/composite_results_v2.json`: summary results.
- `runs/demo/composite_results_v2.records.jsonl`: full records, one JSON object per line.

Each record includes task/case and persona identifiers, the interaction trajectory, injected errors, and outcome fields. Keep the JSONL records for rescoring and visualizations.

## 6. Score

Pass the **directory containing the record files**:

```bash
screensim-score --results runs/demo
screensim-grade --results runs/demo --judge gemini,gpt
```

The first command performs objective scoring. The second calls the judges for missing interaction-quality judgments.

## 7. Troubleshooting

- **Missing browser:** run the Playwright installation with the same Python environment used for the episode.
- **Missing Gemini credentials:** export `GEMINI_API_KEY`; the simulated user needs this even when the assistant uses another provider.
- **No records found during scoring:** pass the record directory, not the summary JSON filename.
- **Check a selection before a batch:** use `--dry-run` and inspect its case, persona, and assistant tag.
- **Resume versus repeat:** reuse a directory to resume missing records; use a fresh directory for an independent run.

Continue with [Evaluate your model](EVALUATE_YOUR_MODEL.md).
