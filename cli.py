#!/usr/bin/env python3
"""
One-shot CLI for the qubit pipeline:
    SQuADDS -> qiskit-metal -> GDS -> mask -> Artemis input -> SLURM

Defaults live in defaults.yaml. Common knobs are exposed as flags;
anything else is reachable via --set section.field=value (dotted path).

Examples:
    python cli.py
    python cli.py --freq 6
    python cli.py --freq 4.5 --override cross_length=200um claw_length=180um
    python cli.py --freq 6 --set artemis.max_step=4000000 --set slurm.wall_time=20:00:00
"""
from __future__ import annotations

import argparse
import sys
import time

from qpipe_config import Config, default_prefix
import qpipe_squadds
import qpipe_layout
import qpipe_gds
import qpipe_mask
import qpipe_artemis
import qpipe_slurm


def parse_kv(items, sep="=", label="--set"):
    out = {}
    for it in items or []:
        if sep not in it:
            print(f"  warning: ignoring malformed {label} '{it}' (expected key{sep}value)")
            continue
        k, v = it.split(sep, 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(v: str):
    """Coerce string CLI value to int/float if it looks like one; else keep string."""
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=float, default=None,
                    help="qubit frequency in GHz (overrides target.qubit_frequency_GHz)")
    ap.add_argument("--anharm", type=float, default=None,
                    help="anharmonicity in MHz (overrides target.anharmonicity_MHz)")
    ap.add_argument("--cavity", type=float, default=None,
                    help="cavity frequency in GHz (overrides target.cavity_frequency_GHz)")
    ap.add_argument("--g", type=float, default=None,
                    help="qubit-readout coupling in MHz (overrides target.g_MHz)")
    ap.add_argument("--override", action="extend", nargs="*", default=[],
                    metavar="KEY=VAL",
                    help="qubit geometry overrides applied after SQuADDS query "
                         "(e.g. cross_length=200um claw_length=180um); "
                         "repeatable across multiple --override flags")
    ap.add_argument("--set", action="extend", nargs="*", default=[],
                    metavar="SECTION.FIELD=VAL", dest="set_",
                    help="override any defaults.yaml field via dotted path "
                         "(e.g. artemis.max_step=4000000); "
                         "repeatable across multiple --set flags")
    ap.add_argument("--prefix", type=str, default=None,
                    help="output prefix (default: qubit_<freq>GHz)")
    ap.add_argument("--no-jj", action="store_true",
                    help="disable lumped inductor in Artemis input")
    ap.add_argument("--defaults", type=str, default=None,
                    help="path to alternate defaults YAML (default: ./defaults.yaml)")
    args = ap.parse_args(argv)

    cfg_overrides = parse_kv(args.set_, label="--set")
    if args.freq is not None:
        cfg_overrides["target.qubit_frequency_GHz"] = args.freq
    if args.anharm is not None:
        cfg_overrides["target.anharmonicity_MHz"] = args.anharm
    if args.cavity is not None:
        cfg_overrides["target.cavity_frequency_GHz"] = args.cavity
    if args.g is not None:
        cfg_overrides["target.g_MHz"] = args.g

    cfg = Config.load(
        defaults_path=args.defaults or qpipe_squadds.__file__.replace(
            "qpipe_squadds.py", "defaults.yaml"
        ),
        overrides=cfg_overrides,
    )
    geometry_overrides = parse_kv(args.override, label="--override")
    prefix = args.prefix or default_prefix(cfg)

    t0 = time.time()
    print(f"\n[1/6] SQuADDS query: {cfg.target}")
    dq, dk, Lj = qpipe_squadds.query(cfg, overrides=geometry_overrides)
    print(qpipe_squadds.summarize(dq, dk, Lj))

    print(f"\n[2/6] Build qiskit-metal layout")
    design = qpipe_layout.build(cfg, dq, dk, Lj)
    print(f"  components: {list(design.components.keys())}")

    print(f"\n[3/6] Export GDS")
    gds_file = f"{prefix}.gds"
    qpipe_gds.export(design, gds_file)
    print(f"  -> {gds_file}")

    print(f"\n[4/6] Rasterize mask + metafile + preview")
    res = qpipe_mask.rasterize(gds_file, prefix, cfg, dq, Lj)
    print(f"  -> {res.files['no_jj']}")
    print(f"  -> {res.files['with_jj']}")
    print(f"  -> {res.files['preview']}")
    print(f"  -> {res.files['meta']}")
    jjc = res.meta["junction_location"]["center"]
    print(f"  JJ at index ({jjc['ix']},{jjc['iy']})")

    print(f"\n[4.5] Sanity checks")
    issues = qpipe_mask.sanity_check(res, cfg)
    if issues:
        print(f"  {len(issues)} issue(s):")
        for s in issues:
            print(f"    ! {s}")
    else:
        print(f"  all checks passed")

    print(f"\n[5/6] Generate Artemis input")
    input_file = f"input_{prefix}"
    # Both use_inductor and --no-jj runs reference the no_jj mask. The
    # with_jj mask has a PEC bridge across the JJ gap which would short out
    # the lumped inductor.
    art = qpipe_artemis.write_input(
        res.meta, res.files["no_jj"],
        input_file, cfg, use_inductor=not args.no_jj,
    )
    print(f"  -> {input_file}")
    print(f"  source y = {art['source_y_um']:.0f} um (auto from CPW)")

    print(f"\n[6/6] Generate SLURM script")
    slurm_file = qpipe_slurm.write_slurm(prefix, input_file, cfg)
    print(f"  -> {slurm_file}")

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"Done in {elapsed:.1f}s. Prefix: {prefix}")
    print(f"  geometry: {gds_file}")
    print(f"  masks:    {res.files['no_jj']}, {res.files['with_jj']}")
    print(f"  preview:  {res.files['preview']}")
    print(f"  metafile: {res.files['meta']}")
    print(f"  Artemis:  {input_file}")
    print(f"  SLURM:    {slurm_file}")
    print(f"  Lj={Lj:.2f}nH, expected freq~{res.meta['expected_qubit_freq_GHz']:.2f}GHz at C=80fF")
    print(f"{'═' * 60}\n")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
