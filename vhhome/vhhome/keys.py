"""API keys. The environment wins; otherwise `~/.config/vhhome/keys.env` (KEY=VALUE lines, no quotes needed) is read
once. Nothing here ever writes a key anywhere, and no key is ever printed.

    OPENAI_API_KEY          OpenAI assistants (GPT, Realtime), the OpenAI TTS voice of the person, the gpt judge
    GEMINI_API_KEY          the simulated person, Gemini assistants, the gemini judge
    ANTHROPIC_API_KEY       Claude assistants and the claude judge
    ANTHROPIC_WORKSPACE_ID  required with a user-scoped Anthropic key (sk-ant-usr-...)
    DASHSCOPE_API_KEY       Alibaba Cloud assistants
"""
import os

KEYS_FILE = os.environ.get("VHHOME_KEYS_FILE") or os.path.expanduser("~/.config/vhhome/keys.env")
_loaded = False


def _load_file():
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not os.path.exists(KEYS_FILE):
        return
    for line in open(KEYS_FILE):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v


def get(name, required=True):
    """The key called `name`, from the environment or the keys file."""
    v = os.environ.get(name)
    if not v:
        _load_file()
        v = os.environ.get(name)
    if not v and required:
        raise RuntimeError(f"{name} is not set: export it, or add a line to {KEYS_FILE}")
    return v or ""


def load_all():
    """Put every key from the keys file into the environment (what subprocesses and SDKs read)."""
    _load_file()
