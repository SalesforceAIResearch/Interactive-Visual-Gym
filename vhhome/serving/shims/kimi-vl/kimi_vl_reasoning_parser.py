# Reasoning parser plugin for Kimi-VL-A3B-Thinking(-2506), vLLM 0.19.1.
#
# Kimi-VL thinks in "◁think▷ ... ◁/think▷" and then answers. Those markers are NOT single tokens in its
# tiktoken vocab ("◁think▷" = 5 tokens, "◁/think▷" = 6), so vLLM's BaseThinkingReasoningParser (which looks the
# markers up as single vocab entries) cannot be used, and vLLM 0.19.1 ships no Kimi-VL parser. This parser
# matches the markers on decoded text instead. It does two jobs:
#   1. the reply's `content` is only the answer after ◁/think▷ (thinking goes to `reasoning_content`), so the
#      benchmark's _parse_json never picks a "{" out of the thinking;
#   2. structured output (the reply json_schema) is applied only AFTER ◁/think▷, so the model can think first.
# A prompt that already ends in "◁think▷◁/think▷" (assistant prefill = thinking off) counts as reasoning ended,
# so the grammar applies from the first generated token.
#   serve: --reasoning-parser-plugin kimi_vl_reasoning_parser.py --reasoning-parser kimi_vl
from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser, ReasoningParserManager

START, END = "◁think▷", "◁/think▷"
_TAIL = 24          # tokens decoded to look for a marker; both markers are <= 6 tokens


@ReasoningParserManager.register_module("kimi_vl")
class KimiVLReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._tok = self.model_tokenizer

    def _dec(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids))

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # called on the prompt (prefill check) and on prompt+output while thinking: look at the tail only
        tail = self._dec(list(input_ids)[-_TAIL:])
        e = tail.rfind(END)
        return e >= 0 and e >= tail.rfind(START)

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        return self.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # content = tokens after the first point where the decoded text contains END
        for i in range(len(input_ids)):
            if END in self._dec(input_ids[max(0, i - _TAIL): i + 1]):
                return input_ids[i + 1:]
        return []

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        head, sep, rest = model_output.partition(START)
        body = rest if sep else head
        if END not in body:
            if sep:                                   # started thinking, never finished (budget/length)
                return body, None
            return None, model_output                 # no thinking at all (prefilled off): all content
        reasoning, _, content = body.partition(END)
        return reasoning, (content.strip() or None)

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text,
                                    previous_token_ids, current_token_ids, delta_token_ids) -> DeltaMessage | None:
        if END in previous_text:
            return DeltaMessage(content=delta_text)
        if END in current_text:
            r, c = current_text.split(END, 1)
            r_new = r[len(previous_text):] if len(r) > len(previous_text) else ""
            return DeltaMessage(reasoning=r_new.replace(START, "") or None, content=c or None)
        if START not in current_text and len(current_text) >= len(START):
            return DeltaMessage(content=delta_text)
        return DeltaMessage(reasoning=delta_text.replace(START, "") or None)
