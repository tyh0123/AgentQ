"""
Central configuration + paths for the AgentQ multi-agent system.

Everything the deterministic tool layer and the agents need to find the
existing pipeline code, the trained inverse-ML model, and the physics
thresholds the critics judge against lives here.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings

# Qt must be head-less BEFORE qiskit-metal is imported anywhere (the real GDS
# build runs qiskit-metal in-process). Set it here, at first import of config.
os.environ.setdefault("QT_API", "pyqt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ── Quiet the noisy dependencies (set QMETAL_VERBOSE=1 to keep them) ──
# The SQuADDS DB query prints a wall of httpx/HuggingFace INFO logs and progress
# bars that look like errors but are just the DB download. A successful query
# still prints `source: squadds_db (...)`. Silence the noise so runs are legible.
if not os.environ.get("QMETAL_VERBOSE"):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    for _name in ("httpx", "huggingface_hub", "huggingface_hub.utils._http",
                  "datasets", "urllib3", "filelock", "fsspec", "qiskit_metal",
                  "py.warnings"):
        logging.getLogger(_name).setLevel(logging.CRITICAL)

# ── Reused pipeline code (do NOT reimplement) ──
# Local copies under tools_pool/ make the project self-contained; an external
# checkout (via env var) is the fallback when a local dir is missing.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _prefer_local(local: str, external: str) -> str:
    path = os.path.join(_BASE_DIR, "tools_pool", local)
    return path if os.path.isdir(path) else external


PYPROJECT_DIR = _prefer_local("qpipe",                                   # qpipe_* pipeline
                              os.environ.get("AGENTQ_QPIPE_DIR", ""))
ML_DIR        = _prefer_local("squadds_ml",                              # trained inverse model
                              os.environ.get("AGENTQ_ML_DIR", ""))

# Outputs (GDS + mask npy/png + Artemis input + sbatch + metafile) land here.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Make the reused modules importable.
for _p in (PYPROJECT_DIR, ML_DIR):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


# ── Geometry source policy ──
# "auto" : query the real SQuADDS DB first; fall back to the local offline
#          inverse ML model on a DB miss / too-far match / no network.
# "db"   : force the SQuADDS DB path (uses the DB's own remote-ML fallback).
# "ml"   : force the local offline inverse model (reproducible on-stage).
DEFAULT_SOURCE = "auto"


# ── LLM backend (pluggable) ──
# The critics use Claude when it's reachable, else deterministic physics rules.
LLM_MODEL  = "claude-opus-4-8"
LLM_EFFORT = "high"


# ── Physics thresholds the critics judge against ──
# Decision Point 2 (SimConfigCritic): the Artemis excitation source must sit at
# least this many cells inside the top boundary, clear of the PML region.
MIN_PML_CLEARANCE_CELLS = 8

# Decision Point 1 (MaskCritic): the lumped-Lj LC estimate must land within this
# fraction of the target qubit frequency, and metal fraction must be sane.
FREQ_TOL_PCT       = 0.20      # |f_expected - f_target| / f_target
METAL_FRACTION_MIN = 0.05
METAL_FRACTION_MAX = 0.99

# Assumed shunt capacitance for the quick LC frequency estimate (matches qpipe).
C_SHUNT_F = 80e-15

# How many designer→critic revision rounds the orchestrator will attempt per
# decision point before giving up.
MAX_REVISIONS = 3
