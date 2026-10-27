# The engine

CookSim's engine is a deterministic, renderer-agnostic kitchen simulator. The benchmark harness drives it with a
simulated cook and a scheduled set of mistakes; it can equally be used on its own, from Python, in the browser, or
as a reinforcement-learning environment. This page covers the engine itself; the benchmark is described in
[BENCHMARK.md](BENCHMARK.md).

## What is simulated

| | |
|---|---|
| Recipes | caprese, cheeseburger, fish & chips, steak frites, burrito in the benchmark; 19 recipes in the recipe book (soups, salads, burgers, sushi, pizza, fried dishes, rice bowls), each a list of components in a required plating order |
| Ingredients | 12 ingredients plus plates, with raw / chopped / cooked / burnt states |
| Stations | pots, pans, ovens, fryers, cutting boards, sinks, ingredient crates, the serving pass, trash |
| Mechanics | chopping, cooking, baking, frying, burning (food left on the heat smokes, then burns), dish-washing, timed orders, multi-step assembly on a plate |
| Players | 1 to N cooks; the benchmark uses one |
| Maps | text layouts, a visual level editor, and procedural generation; the benchmark uses two hand-designed kitchens per recipe (`benchmark/grid.json`) |
| Rendering | a WebGL renderer (Three.js) with a top-down and a first-person view, used headlessly to produce the frames an assistant sees |

Time is discrete: one **tick**. The benchmark pace sets what one tick is worth to the heat physics (food that is
done stays safe for 40 ticks, then smokes for 45, then burns) and how long a tick lasts on the wall clock when a
model's latency is charged (1.2 s). `cooksim/core/config.py` holds every constant; `apply_pace()` selects the pace.

## From Python

```python
from cooksim import KitchenGame, load_layout, GameConfig
from cooksim.agents import GreedyChef

g = KitchenGame(load_layout("bistro"), config=GameConfig(horizon=1500), seed=0)
bots = [GreedyChef(seed=i) for i in range(g.n_players)]
while not g.done:
    g.step([bot(g, i) for i, bot in enumerate(bots)])
print(g.score, dict(g.stats))
state = g.render_state()   # the JSON-serialisable state the renderer draws
```

`KitchenGame` is the whole simulation: `step()` applies one action per cook and advances every station; `render_state()`
is what the renderer, the benchmark's state dumps and the graders read. Nothing in the engine depends on a browser
or a model.

## The benchmark's view of the engine

The harness (`cooksim_bench/`) adds, on top of `KitchenGame`:

* **high-level actions** (`cooksim/agents/highlevel.py`): "go to the pan", "scoop the meat onto the plate", "chop a
  tomato". The simulated cook talks in this space; the engine routes and validates each command and reports
  refusals in words (`cooksim/interact/checks.py`).
* **the kitchen rules** (`cooksim/interact/kitchen_rules.py`): the rulebook both the cook and the assistant receive,
  rendered from the configuration so the text always matches the physics.
* **personas** (`cooksim/interact/persona_v2.py`): a cook's skill, trust in the assistant and three questionnaire
  answers; they set what the cook does mechanically and what the assistant is told about the person.
* **takeover windows** (`cooksim/interact/windows.py`, `cooksim/core/error_lib.py`): the library of forceable
  mistakes and the rules for when a mistake is identifiable, when it commits, and when the cook's hands are released.
* **observation** (`cooksim/agents/observe.py`, `observe_text.py`): the rendered frames an assistant sees, or the
  text state map used by the `agent+++` setting and by the scripted verifier.

## In the browser

```bash
cooksim-server --port 8000        # then open http://127.0.0.1:8000
```

Play with the keyboard (cook 1: arrows + Space; cook 2: WASD + F), add scripted cooks to any seat, switch layouts or
spawn a random kitchen from the side panel. The level editor at `/editor` paints tiles, validates and exports layouts.
`/static/headless_render.html` is the page the benchmark drives headlessly to render frames.

## Reinforcement learning

```python
from cooksim.envs import CookSimParallelEnv          # PettingZoo, multi-agent
env = CookSimParallelEnv(layout="open_kitchen", reward_mode="shared", seed=0)
obs, info = env.reset(seed=0)
while env.agents:
    obs, rewards, terms, truncs, infos = env.step({a: env.action_space(a).sample() for a in env.agents})

from cooksim.envs import CookSimGymEnv, RandomPartner  # Gymnasium, single agent vs. a fixed partner
env = CookSimGymEnv("cramped_room", partner_policy=RandomPartner(), seed=0)
```

Observations are a `Dict` of a multi-channel grid, the order queue and time; actions are `Discrete(6)`
(north, south, east, west, interact, stay). `reward_mode` is `"shared"` or `"individual"`; `GameConfig.shaped_rewards`
adds dense signals. Pass `procedural={"width": 11, "height": 7, "style": "ring"}` to train on a fresh kitchen each
reset. Both wrappers pass the PettingZoo / Gymnasium API tests (`tests/test_envs.py`).

## Layouts

Built-in kitchens: `cramped_room`, `forced_coordination`, `open_kitchen`, `bistro`, `burger_bar`, `pizzeria`,
`diner` (`cooksim.list_layouts()`). Text grammar, one character per cell:

```
' ' floor   X counter   # wall
O onion  T tomato  L lettuce  M mushroom  E meat  F fish  B bun  C cheese  R rice  G dough  Z egg  Y potato
D plates  P pot  A pan  N oven  / cutting-board  W sink  S serving  U trash   1-9 cook spawns
```

`generate_layout(width, height, n_players, style="ring"|"divided", seed)` makes a random kitchen. The benchmark's
two kitchens per recipe live in `benchmark/grid.json` and are addressed by *cell* name (`<tier>_nops_<medium|hard>_map`).

## Package map

```
cooksim/
  core/        enums, items, recipes, stations, orders, layout, generator, plan_ir, error_lib, config, game
  interact/    kitchen_rules, persona_v2, protocol (prompts), checks (refusals), windows (takeovers), state, system
  agents/      heuristic cooks, high-level action executor, observers (frames, text state map), places
  envs/        Gymnasium + PettingZoo wrappers, observation encoder
  layouts/     built-in kitchens
  server/      FastAPI app: live sessions over WebSocket, REST for layouts/recipes, the static renderer
web/           Three.js renderer (render3d.js, food3d.js), game client, level editor, headless_render.html
```

## Tests and examples

```bash
pytest -q                               # engine mechanics, env API conformance, benchmark invariants
python examples/greedy_demo.py          # heuristic cooks across every layout
python examples/random_rollout.py       # a PettingZoo random rollout
python examples/procedural_and_custom.py
```
