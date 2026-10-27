# Examples

Each example is self-contained and runs from the repository root after `pip install -e .` and
`bash simulator/install_virtualhome.sh`. Every episode needs `GEMINI_API_KEY` (the simulated person is a
Gemini model) and a running simulator instance.

**`run_one_episode.sh`** launches one simulator instance, plays the shipped case `cx_tidy_missing_rep` with
the `baseline` persona twice (once with the silent control, once with `gemini:gemini-3.8-flash` in the seat),
and scores both records objectively with `vhhome-score --only objective`. It is the smallest complete loop:
world, person, assistant, record, table. `CASE`, `PERSONA`, `ASSISTANT` and `OUT` can be overridden in the
environment.

**`my_cases.json`** is a case file written by `vhhome-case new` holding two user cases that pass the
composition checks: `my_tidy_1` (`tidy_hard`: the first bedroom lamp is skipped, then the fridge she already
shut is opened again after the television) and `my_bedroom_1` (`tidybedroom_hard`: the cabinet she closed is
opened again after the lamp, then the mug ends on the kitchen counter instead of the kitchen table). Pass it
with `--cases` to `vhhome-case`, `vhhome-episode` or `vhhome-benchmark`; how it was made is in
[docs/GENERATE_CASES.md](../docs/GENERATE_CASES.md).

**`custom_cells.json`** is a cells file over those two cases, each played by `classic_novice` (does not know
the house, trusts the assistant) and `classic_expert` (knows the house, distrusts the assistant): four
episodes. `vhhome-benchmark --cases examples/my_cases.json --cells examples/custom_cells.json ...` runs them in
place of the 75 benchmark cells; the format is the same as `benchmark/cells.json`.

**`evaluate_served_model.sh`** is the serve-your-own-model flow: any OpenAI-compatible endpoint is the
assistant (`--assistant openai:<name> --base-url ...`), run three times over the 75 cells on a single simulator
port (a served model's latency is charged to the clock, so it plays one episode at a time), then
`vhhome-score` pools the runs into the final table with the two reported judges. Set `MODEL` and `BASE_URL`;
for a model of the published roster, `bash serving/serve.sh <tag>` starts it on its roster port and the tag
alone names it.
