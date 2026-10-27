# The benchmark

## An episode

One episode is one task, done once, by one persona, watched by one assistant.

1. The engine builds a phone whose initial state is perturbed — some settings already wrong, some
   already right.
2. The person is told the **outcome** they want, never the controls: *"stop TripSnap knowing where I
   am, following me across apps, or loading things in the background — only TripSnap."* Working out
   which switches that means is part of the task.
3. The person taps. A low-skill persona asks before every step; a high-skill one works alone.
4. At an exact screen state the engine **forces a wrong tap** and holds the person's hands until the
   assistant names it and the person accepts. Then the person repairs it as themselves.
5. The assistant sees the screen each turn and may speak. It never taps.

## Strict success

> the owner's goal is met **and nothing else on the phone changed**

A typed search query is not a change. Everything else is. This is the whole difficulty: the forced
tap leaves a setting the task never asked for, and an assistant that redirects to the right control
without ever asking for the wrong one to be put back does not pass.

## The four mistake kinds

| kind | the person… |
|---|---|
| `wrong_page` | opens the wrong page — nothing has changed yet, they are simply in the wrong place |
| `wrong_target` | hits the switch next to the right one |
| `wrong_value` | picks the right control with the wrong option (*Ask Next Time* where *Never* was needed) |
| `extra_write` | changes something the task never mentioned, often the one thing it said to keep |

Each designed episode carries **two**. Detection is scored only on mistakes that actually fired.

## The suite

11 benchmark tasks → 30 designed episodes (up to 3 per task) × 3 personas = **90 cells per seat**.
`screensim-case --episodes` prints each episode with the mistakes designed into it.

Tasks span lock-screen privacy, app permissions, accessibility setup, cellular and roaming, keyboard
behaviour, StandBy, iCloud protection and a payments flow with a one-time code. Every task has a
**trap**: a plausible wrong move that satisfies the surface of the request and breaks it — the master
*Cellular Data* switch when only two apps were meant to lose mobile data.

## The nine personas

Two private axes, **skill** with a phone and **trust** in the assistant. Published numbers average
`classic_novice`, `classic_expert` and `baseline`. The assistant is told nothing about the person.

The person is deliberately near-silent: they do not narrate their taps, do not repair a mistake they
were not told about, and do not confess one. A wrong setting left standing is the assistant's failure.

## The metrics

| | |
|---|---|
| **In-time success** | strict success within the tick cap *C = 2·T_ref + 8* |
| **Action efficiency (TT)** | clip((C − T)/(C − T_ref)); reported over all episodes and over the ones won |
| **Conversation efficiency (CC)** | 1 − clip(talk turns / assistant polls) |
| **Prevention timing (DT)** | per fired mistake: 1 prevented · 0.5 flagged but still standing · 0 missed |
| **Detection F1** | precision TP/(TP+FP), recall TP/fired |
| **False alarms** | the person's "that's wrong" calls on things that were not wrong |
| **Interaction quality** | the per-round rubric — see [INTERACTION_QUALITY_GRADING.md](INTERACTION_QUALITY_GRADING.md) |

Detection credit is the **person's** private judgement, not the engine's: a warning counts when it
made the person realise what they had done.

Latency is **not** charged. The phone does not move while the assistant thinks, so slow seats are not
penalised — unlike CookSim, where the kitchen keeps burning.
