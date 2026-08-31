#!/usr/bin/env python3
"""
AgentQ multi-agent demo — designer + critics, with self-correction.

A design spec in (asked interactively, or via flags); a *validated* Artemis FDTD
input + Perlmutter sbatch out — built by the project's real pipeline:

    SQuADDS DB query → (local inverse-ML fallback) → qiskit-metal GDS
      → npy mask + preview PNG → Artemis input → sbatch

Inject a fault to watch a critic catch a config error before it would waste
GPU-days on the cluster.

Examples
--------
    python demo.py                                  # asks you for the design
    python demo.py --freq 5.0 --anharm -200
    python demo.py --freq 5.0 --anharm -200 --source ml         # force offline model
    python demo.py --freq 5.0 --anharm -200 --inject-fault source_in_pml
    python demo.py --freq 7.0 --anharm -180 --inject-fault both
"""
from __future__ import annotations

import argparse
import os
import sys

import config
import faults as fault_mod
import llm
from blackboard import DesignRecord
from agents import DesignerAgent, Orchestrator

C = {"dim": "\033[2m", "b": "\033[1m", "g": "\033[32m", "r": "\033[31m",
     "y": "\033[33m", "c": "\033[36m", "x": "\033[0m"}


def _p(s=""):
    print(s)


def on_event(kind, payload):
    if kind == "phase":
        _p(f"\n{C['b']}{C['c']}▶ {payload}{C['x']}")
    elif kind == "verdict":
        col = C["g"] if payload.ok else C["r"]
        _p(f"  {col}{payload.summary()}{C['x']}")
    elif kind == "revise":
        _p(f"  {C['y']}↻ designer applies fix: {payload}{C['x']}")


def _ask_spec(args):
    """Ask the user for the design when it wasn't given on the command line.
    This is AgentQ's front door: natural-language-ish target in."""
    if args.freq is not None:
        return args.freq, args.anharm if args.anharm is not None else -200.0
    if not sys.stdin.isatty():
        return 5.0, -200.0                                 # non-interactive default
    _p(f"{C['b']}What qubit would you like to design?{C['x']}")
    try:
        f = input(f"  target frequency (GHz) [5.0]: ").strip()
        a = input(f"  target anharmonicity (MHz, negative) [-200]: ").strip()
    except EOFError:
        return 5.0, -200.0
    freq = float(f) if f else 5.0
    anharm = float(a) if a else -200.0
    return freq, anharm


def _coupler_overrides(args):
    """User-set resonator↔feedline coupler geometry → '<x>um' strings, or None."""
    ov = {}
    if args.coupling_gap is not None:
        ov["coupling_space"] = f"{args.coupling_gap}um"
    if args.coupling_length is not None:
        ov["coupling_length"] = f"{args.coupling_length}um"
    if args.down_length is not None:
        ov["down_length"] = f"{args.down_length}um"
    return ov or None


def _run_multi(args, injected):
    """multi mode: N qubits on a shared feedline. Targets from --freqs."""
    if not args.freqs:
        _p(f"{C['y']}multi mode needs --freqs, e.g. --freqs 5.0,5.5,6.0{C['x']}")
        return
    freqs = [float(x) for x in args.freqs.split(",") if x.strip()]
    cavities = [float(x) for x in args.cavities.split(",")] if args.cavities else None
    anharm = args.anharm if args.anharm is not None else -200.0
    g = args.g if args.g is not None else 70.0
    targets = []
    for i, f in enumerate(freqs):
        cav = cavities[i] if cavities and i < len(cavities) else round(f + 2.5, 2)
        targets.append({"qubit_frequency_GHz": f, "anharmonicity_MHz": anharm,
                        "cavity_frequency_GHz": cav, "g_MHz": g})
    prefix = args.prefix or f"aq_multi_{len(freqs)}q"

    _p(f"\n{C['b']}AgentQ · designer–critic multi-agent{C['x']}")
    _p(f"{C['dim']}judgment backend : {llm.backend_name()}{C['x']}")
    _p(f"{C['dim']}layout mode      : multi ({len(freqs)} qubits, shared feedline){C['x']}")
    _p(f"{C['dim']}targets (GHz)    : {', '.join(str(f) for f in freqs)}{C['x']}")
    _p(f"{C['dim']}spacing          : {args.spacing} um{C['x']}")
    _p(f"{C['dim']}output dir       : {config.OUT_DIR}{C['x']}")

    record = DesignRecord(freq_GHz=freqs[0], anharm_MHz=anharm, prefix=prefix,
                          mode="multi", targets=targets, coupler_spacing_um=args.spacing,
                          coupler_overrides=_coupler_overrides(args))
    orch = Orchestrator(DesignerAgent(faults=injected, source=args.source,
                                    clean_gds=not args.keep_all_layers),
                        max_revisions=args.max_revisions, on_event=on_event)
    record, ok = orch.run(record)

    _p(f"\n{C['b']}▶ Result{C['x']}")
    verdict = f"{C['g']}ALL CHECKPOINTS PASSED{C['x']}" if ok else f"{C['r']}UNRESOLVED{C['x']}"
    _p(f"  {verdict}")
    for i, q in enumerate(record.metrics.get("qubits", [])):
        _p(f"  {C['dim']}Q{i+1}: target {q['target_GHz']:.2f} GHz → LC est "
           f"{q['expected_GHz']:.2f} GHz (Lj {q['Lj_nH']:.2f} nH){C['x']}")
    if ok:
        _p(f"  {C['g']}validated artifacts, ready to ship to HPC:{C['x']}")
        for label, path in (("GDS layout", record.gds_file),
                            ("mask preview", record.preview_png),
                            ("npy mask", record.mask_no_jj),
                            ("Artemis input", record.artemis_input),
                            ("sbatch script", record.slurm_script)):
            if path:
                _p(f"    {label:14s} {os.path.relpath(path, config.OUT_DIR)}")
    _p(f"\n{C['dim']}next: a downstream agent submits the sbatch, collects plotfiles,\n"
       f"      and runs Decision Point 3 (FFT / mode identification).{C['x']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=float, default=None, help="target qubit frequency (GHz)")
    ap.add_argument("--anharm", type=float, default=None, help="target anharmonicity (MHz)")
    ap.add_argument("--prefix", default=None, help="output filename prefix")
    ap.add_argument("--mode", default="qubit",
                    choices=["qubit", "qubit_resonator", "multi"],
                    help="qubit (qubit+feedline) | qubit_resonator (+claw+coupler+resonator) "
                         "| multi (N qubits, shared feedline)")
    ap.add_argument("--cavity", type=float, default=None, help="readout cavity freq (GHz)")
    ap.add_argument("--g", type=float, default=None, help="qubit–cavity coupling g (MHz)")
    ap.add_argument("--freqs", default=None,
                    help="multi: comma-separated qubit frequencies, e.g. 5.0,5.5,6.0")
    ap.add_argument("--cavities", default=None,
                    help="multi: comma-separated cavity frequencies (optional)")
    ap.add_argument("--spacing", type=float, default=2500.0,
                    help="multi: spacing between qubits in um")
    ap.add_argument("--coupling-gap", type=float, default=None,
                    help="resonator↔feedline coupler gap in um (CoupledLineTee coupling_space)")
    ap.add_argument("--coupling-length", type=float, default=None,
                    help="resonator↔feedline coupler overlap length in um (coupling_length)")
    ap.add_argument("--down-length", type=float, default=None,
                    help="coupler down_length in um (drop to the resonator)")
    ap.add_argument("--keep-all-layers", action="store_true",
                    help="keep every GDS layer (default: trim to 1/0, 1/10, 1/11, 20/10)")
    ap.add_argument("--source", default=config.DEFAULT_SOURCE, choices=["auto", "db", "ml"],
                    help="geometry source: auto (DB→ML), db (force SQuADDS), ml (force local model)")
    ap.add_argument("--inject-fault", default="none",
                    help="none | source_in_pml | bad_lj | both")
    ap.add_argument("--max-revisions", type=int, default=config.MAX_REVISIONS)
    args = ap.parse_args()

    # qubit_resonator / multi need the DB (the resonator has no offline surrogate)
    if args.mode in ("qubit_resonator", "multi") and args.source == "ml":
        _p(f"{C['y']}note: {args.mode} needs the resonator from SQuADDS; forcing --source db{C['x']}")
        args.source = "db"

    injected = fault_mod.parse(args.inject_fault)

    if args.mode == "multi":
        return _run_multi(args, injected)

    freq, anharm = _ask_spec(args)
    prefix = args.prefix or f"aq_{args.mode}_{str(freq).replace('.', 'p')}GHz"

    _p(f"\n{C['b']}AgentQ · designer–critic multi-agent{C['x']}")
    _p(f"{C['dim']}judgment backend : {llm.backend_name()}{C['x']}")
    _p(f"{C['dim']}layout mode      : {args.mode}{C['x']}")
    _p(f"{C['dim']}geometry source  : {args.source}{C['x']}")
    tgt = f"f_q={freq} GHz, α={anharm} MHz"
    if args.mode == "qubit_resonator":
        tgt += f", f_cav={args.cavity or 'default'} GHz, g={args.g or 'default'} MHz"
    _p(f"{C['dim']}target           : {tgt}{C['x']}")
    _p(f"{C['dim']}injected fault(s): {sorted(injected) or 'none'}{C['x']}")
    _p(f"{C['dim']}output dir       : {config.OUT_DIR}{C['x']}")

    record = DesignRecord(freq_GHz=freq, anharm_MHz=anharm, prefix=prefix,
                          cavity_GHz=args.cavity, g_MHz=args.g, mode=args.mode,
                          coupler_overrides=_coupler_overrides(args))
    orch = Orchestrator(DesignerAgent(faults=injected, source=args.source,
                                    clean_gds=not args.keep_all_layers),
                        max_revisions=args.max_revisions, on_event=on_event)
    record, ok = orch.run(record)

    _p(f"\n{C['b']}▶ Result{C['x']}")
    verdict = f"{C['g']}ALL CHECKPOINTS PASSED{C['x']}" if ok else f"{C['r']}UNRESOLVED{C['x']}"
    n_rev = sum(1 for h in record.history if "revising" in h)
    _p(f"  {verdict}   ({n_rev} self-correction(s) applied)")
    _p(f"  {C['dim']}geometry from    : {record.source_used}{C['x']}")
    if ok:
        _p(f"  {C['g']}validated artifacts, ready to ship to HPC:{C['x']}")
        for label, path in (("GDS layout", record.gds_file),
                            ("mask preview", record.preview_png),
                            ("npy mask", record.mask_no_jj),
                            ("Artemis input", record.artemis_input),
                            ("sbatch script", record.slurm_script),
                            ("mask metafile", record.metafile)):
            if path:
                _p(f"    {label:14s} {os.path.relpath(path, config.OUT_DIR)}")
    _p(f"\n{C['dim']}next: a downstream agent submits the sbatch, collects plotfiles,\n"
       f"      and runs Decision Point 3 (FFT / mode identification).{C['x']}")


if __name__ == "__main__":
    main()
