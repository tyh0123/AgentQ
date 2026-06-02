#!/usr/bin/env python3
"""
Standalone: take an existing GDS file → npy masks + metafile + preview +
Artemis input + SLURM job script.

Use this when you already have a GDS (from manual design, an older qiskit-metal
run, etc.) and want to feed it through the FDTD pipeline without going through
SQuADDS query / qiskit-metal layout.

Required: the GDS must have a transmon-only topology (cross + claw + vertical
CPW reaching to the chip top edge) with the same GDS layer convention used by
the rest of the pipeline:
    (1, 0)   = chip / ground outline
    (1, 10)  = etched / subtract regions
    (1, 11)  = positive features (CPW center conductor, etc.)
    (20, 10) = JJ bridge (only in the with_jj mask)

Examples:
    python cli_from_gds.py qubit_5p5GHz.gds --lj-nH 7.437
    python cli_from_gds.py mydesign.gds --lj-nH 6.5 --freq 5.0 --prefix qb5
    python cli_from_gds.py qubit.gds --lj-nH 7.5 --no-jj
    python cli_from_gds.py q.gds --lj-nH 7.5 --set artemis.max_step=4000000
"""

import argparse
import os
import sys
import time

from qpipe_config import Config
import qpipe_mask
import qpipe_artemis
import qpipe_slurm


def parse_kv(items, label="--set"):
    out = {}
    for it in items or []:
        if "=" not in it:
            print(f"  warning: ignoring malformed {label} '{it}'")
            continue
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("gds_file", help="path to input GDS file")
    ap.add_argument(
        "--lj-nH", type=float, required=True, dest="lj_nH",
        help="Josephson inductance in nH (used for the Artemis lumped inductor "
             "and recorded in the metafile design_params.Lj_nH)",
    )
    ap.add_argument(
        "--prefix", default=None,
        help="output prefix (default: GDS filename without .gds)",
    )
    ap.add_argument(
        "--freq", type=float, default=None,
        help="qubit frequency in GHz for Artemis excitation "
             "(default from defaults.yaml target.qubit_frequency_GHz)",
    )
    ap.add_argument(
        "--set", nargs="*", action="extend", default=[], dest="set_",
        metavar="SECTION.FIELD=VAL",
        help="override any defaults.yaml field via dotted path "
             "(e.g. artemis.max_step=4000000 slurm.wall_time=20:00:00)",
    )
    ap.add_argument(
        "--no-jj", action="store_true",
        help="disable the lumped inductor in the Artemis input "
             "(uses the no_jj mask instead of with_jj)",
    )
    args = ap.parse_args(argv)

    if not os.path.exists(args.gds_file):
        ap.error(f"GDS not found: {args.gds_file}")

    cfg_overrides = parse_kv(args.set_, label="--set")
    if args.freq is not None:
        cfg_overrides["target.qubit_frequency_GHz"] = args.freq
    cfg = Config.load(overrides=cfg_overrides)

    prefix = args.prefix or os.path.splitext(os.path.basename(args.gds_file))[0]

    # Construct a minimal qubit-options dict. The standalone path doesn't know
    # the SQuADDS design parameters, so we record them as "unknown" placeholders
    # in the metafile. The pipeline only consumes Lj_nH downstream (Artemis).
    dq_stub = {
        "cross_length": ["unknown"],
        "cross_width": ["unknown"],
        "cross_gap": ["unknown"],
        "claw_length": ["unknown"],
        "claw_width": ["unknown"],
        "claw_gap": ["unknown"],
        "ground_spacing": ["unknown"],
    }

    t0 = time.time()
    print(f"\n[1/3] Rasterize GDS → mask + metafile + preview")
    print(f"  input: {args.gds_file}")
    print(f"  prefix: {prefix}")
    res = qpipe_mask.rasterize(args.gds_file, prefix, cfg, dq_stub, args.lj_nH)
    for k, v in res.files.items():
        print(f"  -> {v}")
    jjc = res.meta["junction_location"]["center"]
    print(f"  JJ index ({jjc['ix']},{jjc['iy']})  source-gap detect: "
          f"{res.meta.get('cpw_source_gaps')}")

    issues = qpipe_mask.sanity_check(res, cfg)
    if issues:
        print(f"  Sanity: {len(issues)} issue(s):")
        for s in issues:
            print(f"    ! {s}")
    else:
        print(f"  Sanity: all checks passed")

    print(f"\n[2/3] Generate Artemis input")
    input_file = f"input_{prefix}"
    # Both use_inductor and --no-jj runs reference the no_jj mask. The
    # with_jj mask has a PEC bridge across the JJ gap which would short out
    # the lumped inductor.
    mask_for_artemis = res.files["no_jj"]
    art = qpipe_artemis.write_input(
        res.meta, mask_for_artemis, input_file, cfg,
        use_inductor=not args.no_jj,
    )
    print(f"  -> {input_file}")
    print(f"  source y = {art['source_y_um']:.0f} um (mask_top - pml_margin)")
    print(f"  Lj = {art['Lj_nH']:.3f} nH "
          f"{'(disabled)' if not art['use_inductor'] else ''}")

    print(f"\n[3/3] Generate SLURM script")
    slurm_file = qpipe_slurm.write_slurm(prefix, input_file, cfg)
    print(f"  -> {slurm_file}")

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"Done in {elapsed:.1f}s. Prefix: {prefix}")
    print(f"  masks:    {res.files['no_jj']}, {res.files['with_jj']}")
    print(f"  preview:  {res.files['preview']}")
    print(f"  metafile: {res.files['meta']}")
    print(f"  Artemis:  {input_file}")
    print(f"  SLURM:    {slurm_file}")
    print(f"{'═' * 60}\n")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
