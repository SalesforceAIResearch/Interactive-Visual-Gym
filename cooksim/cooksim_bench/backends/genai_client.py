"""Single source for Gemini access across all CookSim harnesses.

Two paradigms, auto-selected from the environment:

  VERTEX (Salesforce internal, 2026-07-26 default): set
      GOOGLE_GENAI_USE_VERTEXAI=true
      GOOGLE_CLOUD_PROJECT=<your-gcp-project>
      GOOGLE_CLOUD_LOCATION=us-central1
  and authenticate once with `gcloud auth application-default login`.
  No API key anywhere. Available models on this project: gemini-2.5-pro,
  gemini-2.5-flash, gemini-2.5-flash-lite (the 3.1-preview family is NOT
  allowlisted there).

  API KEY (legacy fallback): GEMINI_API_KEY set and vertex flag absent.

Model ids differ between the paradigms, so always pair make_client() with
default_model()/fast_model() instead of hardcoding names.
"""
import os

from google import genai
from google.genai import types


def use_vertex() -> bool:
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"


def make_client(timeout_ms: int = 120_000) -> "genai.Client":
    http = types.HttpOptions(timeout=timeout_ms)
    if use_vertex():
        # location MUST be "global": the 3.x preview models 404 in every regional
        # endpoint on this project (probed us-central1/us-east5/us-west1/europe-west4
        # on 2026-07-27) but serve fine from global.
        return genai.Client(vertexai=True,
                            project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                                   ""),   # set GOOGLE_CLOUD_PROJECT in the environment,
                            location="global", http_options=http)
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=http)


def default_model() -> str:
    """The strong model (supervisor / human / judge) - same id in BOTH paradigms,
    so Vertex-era results stay comparable with the historical grids."""
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")


def fast_model() -> str:
    """The latency-tier model (study fast display channel)."""
    return os.environ.get("GEMINI_FAST_MODEL",
                          "gemini-3-flash-preview" if use_vertex() else "gemini-flash-latest")


def live_model() -> str:
    """Audio-native live model (study live voice mode)."""
    return os.environ.get("LIVE_MODEL",
                          "gemini-live-2.5-flash" if use_vertex()
                          else "models/gemini-3.1-flash-live-preview")


def resolve_model(name: str) -> str:
    """Map historical model ids onto what this paradigm serves."""
    if not use_vertex():
        return name
    n = (name or "").strip()
    if n == "gemini-flash-latest" or ("flash" in n and "live" not in n and "3-flash" not in n):
        return "gemini-3-flash-preview"
    return n or "gemini-3.1-pro-preview"
