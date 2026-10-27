# VH-Home

**A household simulator and benchmark for interactive visual assistants.**

VH-Home puts a model in the seat of a household assistant. A simulated person, played by a language model with a persona, does one chore in a 3D apartment (VirtualHome, Unity): tidying the house, putting groceries away, shutting the place down for the night. At scheduled moments the engine takes over the person's hands and makes it commit a real mistake: leaving a lamp on, taking the juice instead of the milk, putting a mug down on the wrong surface, reopening a fridge it already shut. The assistant under test watches through the person's first-person camera plus two maps and can speak at any time. It is scored on whether the chore gets finished, whether it names each mistake (ideally before the mistake lands), how many false alarms it raises, and how good each thing it says is at the moment it says it. The seconds the assistant spends thinking are charged to the game clock, so a slow warning is a late warning.

## How an episode works

| Piece | What it is |
|---|---|
| **World** | VirtualHome stepped one command per tick. One tick is 1 second of animation (5 rendered frames). Beside the 3D scene the engine keeps a symbolic twin of the state (what is held, what is open, what is switched on, where things are), used for grading and for the person's own planning. |
| **Person** | A Gemini model with a persona (how well it knows this house, how much it trusts the assistant, its pace, and how much alerting and interruption it wants). It plans the chore itself from the instruction and acts one command at a time through an egocentric, visibility-gated action space: it can walk to the fridge only if a fridge is in view, and changes rooms with "go through the visible door". It talks rarely, and only when it needs something. |
| **Assistant** | The model under test. Every 2 ticks it receives the task instruction, the six mistake names, the person's questionnaire answers, the last 8 first-person frames, a top-down map of the whole house and one of the current room, the conversation so far and its own earlier notes. It returns a spoken line plus an optional typed flag naming a mistake, e.g. `{"error": "wrong_object"}`. A reply that took *x* seconds arrives `ceil(x / 1.2)` ticks later (at most 15). |
| **Case** | One chore plus a schedule of two or three mistakes of different classes, each anchored on a subgoal of the chore's plan (e.g. `switch_off(tablelamp@3)`). A mistake fires when the live world reaches its anchor, never on a timer. |
| **Takeover** | While a mistake is live the engine drives the person's body through the mistake and shows the person a system task so it talks about what it is doing in character. The person never repairs a mistake on its own. The takeover is lifted when the person is convinced by a correct assistant claim, when the mistake's sequence has been carried out, or when its queue is exhausted. |
| **Outcome** | `won` (every checkpoint of the chore is true at the end), `finished_wrong` (the person declared the chore done but a checkpoint is false), `stuck` (no progress for 60 ticks), `timeout` or `silent_fail`. |

The shipped benchmark has **17 tasks** (one chore in one of VirtualHome's five apartments, with a setup that leaves things out of place and programmatic checkpoints that define success), **25 cases** built on them, and **9 personas**. The six mistake classes an assistant may flag are `skipped_step`, `wrong_object`, `wrong_placement`, `repetition`, `stall` and `navigation_inefficiency`. The formal protocol plays 75 fixed cells (25 cases x 3 personas each, drawn once with a fixed seed) and averages 3 independent runs, 225 episodes per assistant. `vhhome-case list` prints the tasks, cases, personas and mistake classes; [docs/BENCHMARK.md](BENCHMARK.md) describes them and the metrics.

**Leaderboard:** [results/README.md](../results/README.md).

## Quickstart

### Install

Python 3.10 or newer. The simulator is the VirtualHome v2.3.0 Unity build, rendered with software OpenGL under a virtual X display, so no GPU is needed.

```bash
git clone https://github.com/hellomuffin/interactive_visual_gym.git && cd interactive_visual_gym/vhhome
pip install -e .                          # extras: .[anthropic] (the Claude judge), .[cluster] (cluster submission), .[dev] (pytest)
bash simulator/install_virtualhome.sh     # clones the VirtualHome python client and fetches the v2.3.0 Linux Unity build (a few GB)
```

`install_virtualhome.sh` needs `git`, `curl` and `unzip`; the Unity build needs Xvfb and the Mesa packages listed at the top of `simulator/launch_simulators.sh` (or, without root, `bash simulator/build_gl_stack.sh`). See [docs/SIMULATOR.md](SIMULATOR.md).

### API keys

```bash
export GEMINI_API_KEY=...        # required: the simulated person is a Gemini model; also Gemini assistants and the gemini judge
export OPENAI_API_KEY=...        # GPT assistants, the gpt judge, the person's voice for native streaming seats
```

Or put `KEY=value` lines in `~/.config/vhhome/keys.env` (`VHHOME_KEYS_FILE` overrides the path). The environment wins over the file; no key is ever printed.

### Start the simulator

```bash
bash simulator/launch_simulators.sh 2     # N headless instances on ports 8080, 8081, ...; HTTP 501 on a port means healthy
```

Leave them running. One episode per instance at a time.

### Run one episode

```bash
vhhome-episode --case cx_tidy_missing_rep --persona baseline \
    --assistant gemini:gemini-3.8-flash --out runs/demo
```

The record lands at `runs/demo/gemini-3.8-flash/gemini-3.8-flash__cx_tidy_missing_rep__baseline.json`, with the episode's log and a film of it beside it. `--assistant silent` runs the control that never speaks and needs no key beyond the person's. `vhhome-roster` lists the assistants we evaluated; any of their tags works in place of `gemini:gemini-3.8-flash`, as does any OpenAI-compatible endpoint (`--assistant openai:<served-name> --base-url http://127.0.0.1:18300/v1`). An episode takes roughly 10 to 30 minutes, most of it rendering and model calls.

### Run the benchmark and score it

```bash
vhhome-benchmark --assistant gemini:gemini-3.8-flash --out runs/gemini --engines 8080,8081 --repeat 1   # 75 cells, resumable
vhhome-benchmark --assistant gemini:gemini-3.8-flash --out runs/gemini --engines 8080,8081 --repeat 2
vhhome-benchmark --assistant gemini:gemini-3.8-flash --out runs/gemini --engines 8080,8081 --repeat 3
vhhome-score --run runs/gemini/r1 runs/gemini/r2 runs/gemini/r3
```

`vhhome-score` grades the task outcome, in-time success, efficiency and detection from the records alone, then the interaction-quality rubric with LLM judges, and prints the pooled table (`final_results.md`). A model you serve yourself must use a single `--engines` port, because its latency is charged to the game clock. See [docs/EVALUATE_YOUR_MODEL.md](EVALUATE_YOUR_MODEL.md).

## Make your own cases

`vhhome-case` composes a new case from an existing task and a schedule of mistakes, checks it against the composition rules (distinct classes, surviving anchors, separated detection windows, independent targets, visible evidence) and writes it to a case file you can run:

```bash
vhhome-case plan tidy_hard                                   # the subgoals a mistake can be anchored on
vhhome-case new --name my_tidy_1 --task tidy_hard \
    --mistake "skipped_step sid=switch_off(tablelamp@1)" \
    --mistake "repetition sid=shut(fridge) at=switch_off(tv) gap=delayed" \
    --story "She walks straight past the first bedroom lamp, and after the television she goes back and opens the fridge she had already shut." \
    --out my_cases.json
vhhome-episode --cases my_cases.json --case my_tidy_1 --persona classic_novice --assistant gemini:gemini-3.8-flash --out runs/mine
```

Details in [docs/GENERATE_CASES.md](GENERATE_CASES.md).

## Documentation

| Page | What it covers |
|---|---|
| [docs/GETTING_STARTED.md](GETTING_STARTED.md) | Installation in detail, keys, the simulator, your first episode and how to read its record, first scoring, the tests, common problems |
| [docs/EVALUATE_YOUR_MODEL.md](EVALUATE_YOUR_MODEL.md) | Putting your own model in the assistant seat; the full three-run protocol; what the assistant receives and must return |
| [docs/GENERATE_CASES.md](GENERATE_CASES.md) | Composing and verifying new cases with `vhhome-case` |
| [docs/BENCHMARK.md](BENCHMARK.md) | The tasks, cases, personas, cells, the metrics and the formal protocol |
| [docs/REPRODUCE.md](REPRODUCE.md) | Reproducing the published numbers |
| [docs/ASSISTANTS.md](ASSISTANTS.md) | Every evaluated assistant: backends, flags, serving the open-weight models |
| [docs/INTERACTION_QUALITY_GRADING.md](INTERACTION_QUALITY_GRADING.md) | The per-round interaction-quality rubric and its judges |
| [docs/ENGINE.md](ENGINE.md) | The engine on its own: world, action space, gates, plans, mistakes |
| [docs/SIMULATOR.md](SIMULATOR.md) | VirtualHome: installing the Unity build, launching instances, the software-GL stack, troubleshooting |

## Repository layout

```
vhhome/            engine: world (VirtualHome stepping + symbolic twin), actions, gates, topdown_map, render,
                   tasks, goals, plan, mistakes, paths, keys, data/ (reply schema, scene passages)
vhhome_bench/      harness: protocol (prompts, assistant call), person, episode (one rollout), lane (a run over
                   the cells), roster, native + native_backend (streaming seats), cases_spec, cases, case_checks,
                   preflight, check, scoring/ (metrics, metrics_table, results_table, rubric, rubric_table,
                   rubric_batch, final_table), video/ (film, film_reference), cluster/, cli/
benchmark/         cells.json (the 75 cells), clean_cells.json, t_ref.json (reference times), gpu_latency_ratio.json,
                   composite_ledger.json, admissibility_ledger.json, composite_vlm_audit.json, task_video_audit.json
results/           the published table and the numbers behind it
serving/           scripts and shims that serve the open-weight assistants (vLLM or the authors' code)
simulator/         install_virtualhome.sh, launch_simulators.sh, build_gl_stack.sh, Dockerfile
tests/             test_engine.py (offline engine suite), verify_cases.py (the case composition rules)
docs/, examples/
```

## Citation

Use the shared project metadata in [CITATION.cff](../../CITATION.cff).

## License

MIT. See [LICENSE](../LICENSE).
