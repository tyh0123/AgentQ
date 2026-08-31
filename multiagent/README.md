# AgentQ · designer–critic multi-agent

A multi-agent reframing of the AgentQ pipeline for the **Berkeley AI Summit poster**.
One **designer** agent turns a natural-language-style spec into a full FDTD job;
independent **critic** agents gate each decision point and hand back concrete
fixes, and the designer revises — an **autonomous self-correction loop** that
catches config errors *before* they waste GPU-days on the cluster.

```
        target (f_q, α)   ◄── asked interactively, or --freq/--anharm
             │
   ┌─────────▼──────────┐         deterministic tool layer (the real qpipe_* pipeline)
   │   DesignerAgent     │  ─────  SQuADDS DB query → (local inverse-ML fallback)
   │                     │         → qiskit-metal GDS → npy mask + preview PNG
   └─────────┬──────────┘         → Artemis input → sbatch
             │ artifacts
   ┌─────────▼───────────────────────────────┐   LLM-judgment layer
   │  MaskCritic (DP1)   SimConfigCritic (DP2) │   (Claude if reachable, else rules)
   └─────────┬───────────────────────────────┘
             │ FAIL → concrete fix → designer.revise() → re-review
             ▼
   validated Artemis input + Perlmutter sbatch  ──►  [downstream agent runs it]
```

## The pipeline (every stage is the project's real code)

| # | stage | tool | real module |
|---|-------|------|-------------|
| 1 | ask the user for the design | `demo._ask_spec` | — |
| 2 | query SQuADDS DB | `tools.resolve_geometry` | `qpipe_squadds.query` |
| 3 | ML model on a DB miss / offline | `tools.resolve_geometry` | local `mlp_inverse` (offline) |
| 4 | qiskit-metal layout → GDS | `tools.build_gds` | `qpipe_layout` + `qpipe_gds` |
| 5 | mask visualization (PNG) | `tools.rasterize_mask` | `qpipe_mask.rasterize` |
| 6 | Artemis-compatible npy mask | `tools.rasterize_mask` | `qpipe_mask.rasterize` |
| 7 | Artemis-compatible input file | `tools.write_artemis` | `qpipe_artemis.write_input` |
| + | Perlmutter sbatch | `tools.write_slurm` | `qpipe_slurm.write_slurm` |

Geometry **source** is selectable: `--source auto` (query the DB, fall back to
the local model on a miss / no network), `--source db` (force SQuADDS), or
`--source ml` (force the local offline model — reproducible on-stage).

## Why this shape (the poster's story)

- **AI-for-Science + cost-aware self-correction.** The downstream cost is real and
  expensive (GPU-days of FDTD). A source placed inside the PML, or a mis-set
  Josephson inductance, would silently waste that run — the critics catch both.
- **Multi-agent, not monolithic.** A producer and *independent* critics, one per
  decision point — the natural upgrade from a single orchestrator.
- **Two layers.** Deterministic computation (geometry, rasterization, input
  generation) is code; *judgment* (is this design sane?) is the agent's — and is
  pluggable onto Claude.
- **The ML is ours and it's local.** Geometry comes from the project's own trained
  inverse model (`inverse_model.pt`, surrogate-defined loss), run offline via
  torch — no network, fully reproducible on stage.

## Run

```bash
conda activate mag_torch     # torch + qiskit-metal + squadds + the qpipe deps
cd ~/research/agentq_multiagent

python demo.py                                                 # asks you for the design
python demo.py --freq 5.0 --anharm -200                        # validation, clean run (auto source)
python demo.py --freq 5.0 --anharm -200 --source ml            # force offline model
python demo.py --freq 5.0 --anharm -200 --inject-fault source_in_pml   # DP2 catches + recovers
python demo.py --freq 5.0 --anharm -200 --inject-fault bad_lj          # DP1 catches + recovers
python demo.py --freq 7.0 --anharm -180 --inject-fault both            # both critics fire

# qubit + resonator: qubit + claw + coupler + quarter-wave resonator + feedline (needs the DB)
python demo.py --mode qubit_resonator --freq 5.0 --anharm -210 --cavity 7.5 --g 70 --source db
python demo.py --mode qubit_resonator --freq 5.0 --anharm -210 --cavity 7.5 --g 70 --source db --inject-fault bad_lj

# multi: N qubits on a shared feedline (needs the DB)
python demo.py --mode multi --freqs 5.0,5.5,6.0 --anharm -200 --source db
python demo.py --mode multi --freqs 5.0,5.5 --cavities 7.5,8.0 --spacing 2500 --source db
```

## Layout modes

All three modes run the full chain **geometry → GDS → mask/PNG → Artemis input →
sbatch**, with both critics active:

- **`qubit`** (default) — single qubit + straight CPW feedline + JJ. Vertical-CPW
  Ex source. Runs offline with `--source ml`.
- **`qubit_resonator`** — qubit + claw + CoupledLineTee coupler + RouteMeander
  quarter-wave resonator + launchpad feedline + JJ. Geometry from SQuADDS
  (`--source db`).
- **`multi`** — N qubits (each with its own claw + resonator) on one shared
  feedline; targets via `--freqs`. DP1 checks every qubit's lumped-Lj frequency.

### FDTD excitation source

- `qubit` uses the vertical-CPW source (`qpipe_artemis`): Ex at a fixed y across
  two x-gaps.
- `qubit_resonator` / `multi` have a **horizontal feedline**, so they use an
  **input-port Ey source** at a fixed x near `wb_in`, across the feedline's two
  y-gaps (`artemis_full.py`) — plus **one lumped inductor per qubit JJ** (JJ cells
  detected from the `(20,10)` bridge). DP2 checks the source's x-clearance from
  the PML. The source header in the input file flags it for physics validation
  before large runs.

### Coupler geometry (user input)

The DB doesn't provide the resonator↔feedline coupler, so its geometry is a
design input (defaults: gap 5 µm, overlap 200 µm, down 100 µm):

- `--coupling-gap <um>`    — gap between resonator-side line and feedline (`coupling_space`)
- `--coupling-length <um>` — overlap / coupling length (`coupling_length`)
- `--down-length <um>`     — drop length to the resonator (`down_length`)

```bash
python demo.py --mode qubit_resonator --freq 5.0 --anharm -210 --cavity 7.5 --g 70 \
    --source db --coupling-gap 10 --coupling-length 300
```

These set the qubit-readout / feedline-resonator coupling strength and apply to
`multi` as well.

### Ground plane + GDS layers

- The chip auto-sizes so the **`(1,0)` ground plane encloses every structure**
  (`layouts._fit_chip`, 500 µm margin). The fixed default chip was too small once
  a resonator + feedline were present, leaving them outside the ground plane; now
  the mask is full ground metal with the CPW/qubit/resonator etched into it.
- The exported GDS is trimmed to the **fab layers `1/0` (ground), `1/10` (etch),
  `1/11` (CPW), `20/10` (JJ bridge)** — cheese holes (`1/101`), marker frames
  (`1/99`, `1/100`, `1/102`) and the JJ window (`60/10`) are dropped
  (`tools.clean_gds_layers`, after rasterization). Pass `--keep-all-layers` to
  keep the raw qiskit-metal output.

**SQuADDS coverage note:** this DB build supplies qubit + claw + quarter-wave
resonator (`total_length`) + `Lj` for `cavity_claw/RouteMeander/quarter` rows, but
its `coupler_options` are **all null** — so the feedline coupler + feedline
geometry are filled from sensible defaults (`tools._fill_coupler_defaults`), not
the DB.

Artifacts land in `out/`: `<prefix>.gds` (layout), `<prefix>_mask_preview.png`
(mask visualization), `<prefix>_mask_{no,with}_jj.npy` (Artemis masks),
`<prefix>_mask_meta.json`, `input_<prefix>` (Artemis FDTD input), and
`perl_<prefix>.run` (sbatch). For the offline poster demo use `--source ml`;
the full GDS + PNG + Artemis chain still runs, just without the network DB query.

## Judgment backend (pluggable)

`llm.available()` decides at runtime:

- **Claude** (`claude-opus-4-8`, adaptive thinking, structured verdict) — used when
  the `anthropic` SDK is installed **and** a credential is present
  (`export ANTHROPIC_API_KEY=…`, or `ant auth login`).
- **Deterministic physics rules** — the offline fallback, so the demo always runs.

Either way the *fix* is computed from physics by the critic; only the *judgment*
moves to Claude. Verdicts print the backend they came from.

## Scope

This system stops at a **validated Artemis input + sbatch** — everything that is
locally testable. Running on Perlmutter, collecting plotfiles, and **Decision
Point 3** (FFT / mode identification) belong to a separate downstream agent.

## Layout

| File | Role |
|------|------|
| `demo.py` | CLI entry + narrated trace |
| `agents/designer.py` | producer: builds & revises the artifact chain |
| `agents/critics.py` | `MaskCritic` (DP1), `SimConfigCritic` (DP2) |
| `agents/orchestrator.py` | runs designer, gates each DP, drives the revise loop |
| `tools.py` | deterministic tool layer — drives the real `qpipe_*` pipeline (SQuADDS → GDS → mask/PNG → Artemis → sbatch) |
| `mlp_inverse.py` | local trained inverse model (geometry) + analytic L_J — the offline fallback |
| `llm.py` | pluggable Claude-or-rules judgment |
| `faults.py` | fault injectors for the fail→detect→recover demo |
| `config.py` | paths to the reused repos + physics thresholds |

The qpipe pipeline and the trained inverse model are vendored under
`tools_pool/` (`qpipe/`, `squadds_ml/`), so the repo is self-contained. To use
an external checkout instead, remove the vendored dir and set
`AGENTQ_QPIPE_DIR` / `AGENTQ_ML_DIR`.

Dependencies: `pip install -r requirements.txt` (Python 3.11).
