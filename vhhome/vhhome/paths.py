"""Where VH-Home keeps things. Every location can be overridden with an environment variable, so the package
runs from any checkout and the large, uncommitted pieces (the Unity build, run output) can live elsewhere.

    VHHOME_ROOT          the checkout (default: the parent of this package)
    VHHOME_SIMULATOR     directory holding the VirtualHome client checkout `virtualhome/` and the Unity build
                         `unity_vol/` (default: <root>/simulator)
    VHHOME_UNITY_DIR     the Unity build directory itself (default: <simulator>/unity_vol)
    VHHOME_VIRTUALHOME   the VirtualHome repository checkout (default: <simulator>/virtualhome)
    VHHOME_RUNS          run output: records, films, frame scratch, TTS cache (default: <root>/runs)
"""
import os

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("VHHOME_ROOT") or os.path.dirname(PKG)
DATA = os.path.join(PKG, "data")
BENCH = os.path.join(ROOT, "benchmark")
SIM = os.environ.get("VHHOME_SIMULATOR") or os.path.join(ROOT, "simulator")
UNITY_DIR = os.environ.get("VHHOME_UNITY_DIR") or os.path.join(SIM, "unity_vol")
VIRTUALHOME = os.environ.get("VHHOME_VIRTUALHOME") or os.path.join(SIM, "virtualhome")
RUNS = os.environ.get("VHHOME_RUNS") or os.path.join(ROOT, "runs")
FRAMES = os.path.join(RUNS, "frames")                # per-episode frame scratch (deleted after the film is cut)

CELLS = os.path.join(BENCH, "cells.json")            # the 75 benchmark cells (case x persona)
CLEAN_CELLS = os.path.join(BENCH, "clean_cells.json")
T_REF = os.path.join(BENCH, "t_ref.json")            # measured clean-plan time per case
GPU_LATENCY_RATIO = os.path.join(BENCH, "gpu_latency_ratio.json")
SCENE_PASSAGES = os.path.join(DATA, "scene_passages.json")
REPLY_SCHEMA = os.path.join(DATA, "assistant_reply_schema.json")


def virtualhome_client():
    """The directory that makes `unity_simulator` importable (VirtualHome's python client)."""
    for relative in ("simulation", os.path.join("virtualhome", "simulation")):
        candidate = os.path.join(VIRTUALHOME, relative)
        if os.path.isdir(os.path.join(candidate, "unity_simulator")):
            return candidate
    return os.path.join(VIRTUALHOME, "simulation")
