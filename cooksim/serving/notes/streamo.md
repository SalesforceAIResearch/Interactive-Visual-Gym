# Streamo (roster row 18) — BLOCKED: weights not released

Checked 2026-09-18 (UTC).

| | |
|---|---|
| paper | *Streaming Video Instruction Tuning*, Xia et al., arXiv 2512.21334 (CVPR 2026) |
| code | github.com/maifoundations/Streamo (training code only, built on ms-swift; a fork at PINTO0309/Streamo is identical) |
| data | `maifoundations/Streamo-Instruct-465K` (HF dataset, released 2026-01-27) |
| weights | **none**. The README says: "Due to some restrictions, we are unable to publicly release the model weights at this time. If you have any request, please feel free to contact us." |

**What was checked**
- The HF Hub has no Streamo model repos: searched for `streamo` and listed every model under the `maifoundations` author (only Visionary-R1, IndexMark and DualMindVLM).
- ModelScope search also returned nothing.
- GitHub issue #6, "checkpoints and Streamo-Instruct-465K" (2026-02-11), is still open with no checkpoint link. In issue #9 a user trained their own 3B variant from the released data.
- The repo ships no inference script either. Issue #4 asks for one.

Nothing was downloaded, installed or served. Port 18900 is unused.

**What the model is (for when weights appear).** Streamo is a Qwen2.5-VL fine-tune in multi-turn stream format.
- Each 1-fps frame is one user turn: `<Xs-Ys>\n<stream>`.
- The assistant answers every frame with one of three state tokens:
  - `</Silence>`: nothing relevant.
  - `</Standby>`: the event is still unfolding.
  - `</Response> <text>`: speak.
- For our schema, a shim would map `</Silence>` and `</Standby>` to `text: ""` and `</Response>` to the text that follows it. That is the same adapter as the Proact-VL shim here (`../proact-vl/shim_server.py`).

**Closest open alternative (not substituted).** `wangyueqian/MMDuet2`.
- Qwen2.5-VL-3B post-trained with multi-turn RL for proactive replies during video playback.
- It is ungated and uses transformers-native Qwen2.5-VL weights; its demo and eval code are at github.com/yellow-binary-tree/MMDuet2.
- Like Streamo, the decision to reply or stay silent is made in the next-token stream, not by a separate head.

A second option is `yzy666/LiveStar_8B` (InternVL-based, streaming response–silence decoding), also ungated. Neither was set up. Adding one is a roster decision for the orchestrator.
