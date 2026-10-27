# Proact-VL (`proact-vl`, port 18950): audit against the official guidance (2026-09-19)

**Verdict.** The serving mechanics are exactly the authors'. That covers the checkpoint recipe, attention, resolution, the chunk format, the FLAG head, per-chunk sampling and the KV budget. The one large deviation was the **system prompt**.

**Why the prompt mattered.** Proact-VL was trained and evaluated with one-paragraph system prompts: `construct_val_system_prompt`, and for egocentric step-by-step guidance `EGO4D_SYSTEM_PROMPT`. Given our 13k-character rulebook, it **recites the rulebook**, e.g. "You are the kitchen / assistant — you see / the cook's perspective through / a camera. / You never talk to…" across chunks, and its FLAG head fires on every chunk.

**The fix.** With the authors' own Ego4D guidance prompt, plus the recipe line and the unchanged benchmark user text as the query:
- the recitation disappears;
- the head chooses silence on 9/24 calls, and it speaks on 15/24, the same count as the API reference;
- replies are kitchen guidance: "Turn the pan on.", "The cook is chopping the potato on the cutting…";
- latency falls from 4.39 s to 2.78 s.

**Adopted as the shim default** (`PROACT_PROMPT=ego4d`). This changes what the model is told: it no longer sees the rulebook. The orchestrator should confirm that trade.

**Two more answers.**
- **FLAG threshold.** The README's eval threshold of 0.3 changes nothing on this input, because scores already sit at 0.8–1.0.
- **Chunk window.** Returning more than the 2 newest chunks does not follow the authors' intent better; see §Chunk window.

Sources:
- **[readme]** `src/Proact-VL/README.md` (github.com/microsoft/AnthropomorphicIntelligence/Proact-VL).
- **[qs]** `quickstart.py`.
- **[eval]** `evaluation/gaming/distributed_generate_gaming.py` and `script/evaluation/infer.sh`.
- **[conv]** `proactvl/utils/conversations.py`.
- **[vr]** `proactvl/infer/video_reader.py`.
- **[asst]** `proactvl/infer/assistant.py`.
- **[model]** `proactvl/model/modeling_proact.py`.
- **[cfg]** the adapter's `config.json`.

## Item table

| # | item | ours | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint | LoRA `oaaoaa/proact_lora_base_liveccbase_final` (with the trained `state_proj` FLAG head) on `chenjoya/LiveCC-7B-Base`, bf16, merged at load | Model Zoo row "Proact-VL_LiveCC-7B-Base → `oaaoaa/proactvl_base_liveccbase`" (full checkpoint, **gated**) | [readme] Model Zoo | **deviates by necessity.** The gated repo needs an HF token, which this machine lacks. The ungated LoRA is the authors' own upload in the same collection, trained by the same recipe. Report results as the LoRA variant. |
| 2 | engine | transformers 4.57.1, peft 0.18.1, flash-attn 2.8.3, the authors' `Assistant` class | `script/env/prepare_env.sh` pins; `ProActConfig.attn_implementation = 'flash_attention_2'` | [model] l.25 | matches |
| 3 | prompt format | `_make_user_prompt` logic: `<video><|query_start|>q<|query_end|><|FLAG|>`, system primed once per call | same | [asst], `multi_assistant_inference.py` | matches |
| 3b | **system prompt** | before: the 13k-char benchmark rulebook | README: "We recommend include the following content in the system prompts" (Default / Solo / Co-commentary / **Guidance**). Eval uses fixed one-paragraph prompts; for egocentric guidance it uses `EGO4D_SYSTEM_PROMPT` ("You are an AI assistant that provides real-time, step-by-step guidance from first-person (egocentric) video…") | [readme] §Demo; [conv] l.86–93 | **fixed: `PROACT_PROMPT=ego4d`.** System = `EGO4D_SYSTEM_PROMPT` verbatim. The query = "The cook was asked to make: …" + the benchmark user text. |
| 4 | frames / resolution | each tick frame → one 1-s chunk of 2 identical frames; smart_resize to 1036×392 (min 128·28², max 540·28²) | 2 frames per second (`nframes = chunk_seconds * 2`), 2-frame chunks, `MIN_PIXELS=128*28*28, MAX_PIXELS=540*28*28` | [vr] l.4–5, 30–45, 91–99 | resolution **matches** (the official resize of 1042×396 is also 1036×392). **Rate deviates by necessity:** the harness gives 1 frame per 1.2-s tick, so a chunk holds one frame twice (not 2 frames 0.5 s apart), and it stands for 1.2 s of game time, not 1 s. |
| 5 | per-chunk sampling | do_sample, T 0.7, top_p 0.9, repetition_penalty 1.15, max_new_tokens 12 | identical | [qs] `generate_config`; [eval] l.80–86 | matches |
| 5b | FLAG threshold | 0.5 | `state_threshold: 0.5` in the adapter config, `quickstart.py` and the eval-script default; README/`infer.sh` eval runs use **0.3** | [cfg]; [qs]; [readme] §Infer, `script/evaluation/infer.sh` | matches (0.5). **0.3 was measured:** no change in behaviour (below). |
| 6 | thinking | n/a | n/a | — | — |
| 7 | KV / cache | `max_kv_tokens 16384`, evict 20%, the model's own RoPE re-shift; stateless per call, so nothing is evicted in 8 chunks | `--max_kv_tokens 16384` | [readme] §Infer; [asst] | matches. The per-call re-prefill is the benchmark's stateless protocol (see SETUP.md "native session"). |
| 7b | attention | flash_attention_2 | `attn_implementation: str = 'flash_attention_2'` | [model] | matches |
| 8 | structured output | none: the shim wraps the speech into the schema (`flag` always null) | the model emits ≤12-token speech per active second; there is no JSON mode | — | deviates by design (documented in SETUP.md) |

## Bench (24 real calls, `--profile default --max-tokens 4000 --image-scale 1.0`, GPU 0)

| tag | change | median | p90 | prompt tok | compl tok | schema ok | spoke | prompt echo* |
|---|---|---|---|---|---|---|---|---|
| `proact_audit_base` (**before**) | bench system prompt, threshold 0.5 | 4.39 s | 5.32 s | 8,379 | 44 | 24/24 (wrapped) | 24/24 | 2–3 |
| `proact_audit_thr30` | threshold 0.3 (README eval) | 4.55 s | 5.04 s | 8,379 | 51 | 24/24 | 24/24 | 2 |
| `proact_audit_ego4d` (**after**) | authors' Ego4D guidance system prompt | **2.78 s** | **3.00 s** | 5,276 | 11 | 24/24 | **15/24** (reference 15/24) | 1 |

\* Echo counts replies that copy ≥6 consecutive prompt words. The metric undercounts here, because the recitation is split into ≤12-token pieces and many are shorter than 6 words. The sanity read below is the real evidence.

**Sanity read** (same calls; the reference is the round-4 API reply):

| call | reference | before (bench prompt) | after (ego4d) |
|---|---|---|---|
| easy_hard_1 t16 | "You just chopped the cheese instead of trashing it!" | "The kitchen's floor is brown, the walls are" | "Let's prepare the tomato by chopping it into smaller pieces." |
| easy_hard_2 t94 | "you're still at the cheese crate with the correct finished plate. Go to the serving pass" | "You're still standing by the cheese crate with the correct p…" | "You're about to serve the Caprese plate" |
| expert_hard_2 t362 | "Go to the cutting board and put the potato down. Chop it…" | "What you see is what the" | "The cook is chopping the potato on the cutting" |
| master_hard_2 t278 | (silent) | "You see the cook has turned the pan" | "Turn the pan on." |
| easy_hard_2 t10 | "If you're taking the plate now, that's okay…" | "You are also the kitchen's error detector. When you" | (silent) |
| expert_hard_4 t114 | "Finish chopping that potato…" | "1. Frame at t78:" | (silent) |

- **Before.** About 20 of 24 replies (by hand count) are pieces of the rulebook or transcript: "You are supervising the kitchen…", "It's your turn to respond to the cook.", "This is the end of the episode."
- **After.** Replies are about the kitchen. Some still echo the cook's transcript line ("t416 cook: I'm setting the plate on the counter"), and none detects the reference errors: the FLAG head decides *when* to speak, not *which* error.

## Chunk window: should the shim return more than the 2 newest chunks?

**The authors' intended use.** One persistent session, one chunk per second. Each utterance is emitted at the second it is generated, exactly once. A user query enters at the second it is asked, and the model replies from then on.

**Our protocol.** One stateless call every 2 ticks replays 8 chunks. Only the 2 newest are new to the harness. The query (the benchmark user text) sits on the first new chunk (`PROACT_QUERY_AT=new`), exactly where a query asked "now" would arrive.

The same generations re-assembled with a wider window (`pass3_setup/audit_chunks.py`, no re-run):

| run | K = 2 newest | K = 4 | K = 8 (all) |
|---|---|---|---|
| bench prompt | spoke 24/24, 7.8 words | 24/24, 14.8 words | 24/24, 26.0 words, more rulebook |
| ego4d prompt | spoke 15/24, 7.8 words | identical | identical |

With the ego4d prompt the FLAG head is **silent on every history chunk**. It speaks only once the query arrives, so K makes no difference. With the bench prompt, a wider K only adds more recitation.

Returning history chunks would also re-emit seconds that the previous poll already reported, which breaks the "once, at its second" semantics.

**Answer: keep 2** (`PROACT_NEW_CHUNKS=2`). It is the faithful mapping.

## Changes to apply
- **Serve script** (`pass3_setup/serve_proact_vl.sh`, in place): `export PROACT_PROMPT=${PROACT_PROMPT:-ego4d}`. Set `PROACT_PROMPT=bench` for the old behaviour. The pre-audit copy is `pass3_setup/logs/serve_proact_vl.sh.pre_audit`.
- **Shim** (`proact-vl/shim_server.py`, in place):
  - new `PROACT_PROMPT` (`bench` | `ego4d`);
  - per-request overrides `proact_threshold` / `proact_prompt` in the request body, used only by the audit;
  - `cfg` is logged per request.
  - The pre-audit copy is `proact-vl/logs/shim_server.py.pre_audit`.
- **Engine PROFILES**: none needed. `default` still works, because the shim ignores sampling and `response_format`. The SETUP.md suggestion `"proact-vl-shim": dict(temperature=None, extra_body={}, prefill=None)` still holds.
- **For the orchestrator.** `PROACT_PROMPT=ego4d` means Proact-VL is *not* given the rulebook. It gets the recipe line, the benchmark user text (memory, transcript, feedback) and the frames. The alternative is the authors' model reciting the rulebook. Say so where results are reported.
