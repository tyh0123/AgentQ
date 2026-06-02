#!/usr/bin/env python3
"""
Standalone: take a mask metafile (from mask_from_gds.py) → write an Artemis
FDTD input file.

Self-contained — does NOT import from qpipe_* / cli.py / mcp pipeline. Only
stdlib (json, argparse).

The metafile is the JSON produced by mask_from_gds.py. It must contain an
`artemis_inputs` section with: n_cellx/y/z, dx/dy/dz_m, prob_hi*_m,
source_x_gaps_um (already widened), source_y_um, and — for use_inductor —
jj_x_lo_um, jj_x_hi_um, jj_y_um.

User must supply --lj-nH (Josephson inductance) unless --no-inductor.
Frequency defaults to 5 GHz unless --freq-GHz is set.

Examples:
    python artemis_input_from_meta.py q.json --lj-nH 7.437 --freq-GHz 5.5
    python artemis_input_from_meta.py q.json --lj-nH 6.5 --output input_q5
    python artemis_input_from_meta.py q.json --no-inductor --freq-GHz 7.0
    python artemis_input_from_meta.py q.json --lj-nH 7.4 --max-step 4000000

Output: the Artemis input file at the given --output path (default:
input_<metafile-basename-stripped-of-_mask_meta>).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


# ────────────────────────────────────────────────────────────────────
#  Defaults for Artemis FDTD physics + numerics
# ────────────────────────────────────────────────────────────────────
DEFAULT_MAX_STEP = 8_000_000
DEFAULT_CFL = 0.8
DEFAULT_EPS_R_SI = 11.7
DEFAULT_H_SI_UM = 100.0           # silicon substrate thickness in μm
DEFAULT_SIGMA = 1.0e11            # metal conductivity (S/m)
DEFAULT_NPY_K_INDEX = 20          # which z-slice of mu npy to read
DEFAULT_PLT_INTERVALS = 1000      # plotfile cadence (steps)
DEFAULT_DIAG_SLICE_Z_UM = (100.0, 102.0)
DEFAULT_FREQ_GHZ = 5.0


# ────────────────────────────────────────────────────────────────────
#  Core
# ────────────────────────────────────────────────────────────────────

def _fmt_int_or_float(x: float) -> str:
    """Format whole-number floats as ints (matches verified Artemis style)."""
    return f"{int(x)}" if float(x).is_integer() else f"{x}"


def _require(d: dict, key: str, context: str):
    if key not in d:
        raise KeyError(f"{context} is missing required key '{key}'")
    return d[key]


def write_artemis_input(
    meta_path: str,
    output_path: str,
    lj_nH: Optional[float],
    freq_GHz: float = DEFAULT_FREQ_GHZ,
    use_inductor: bool = True,
    mask_npy_path: Optional[str] = None,
    max_step: int = DEFAULT_MAX_STEP,
    cfl: float = DEFAULT_CFL,
    eps_r_si: float = DEFAULT_EPS_R_SI,
    h_si_um: float = DEFAULT_H_SI_UM,
    sigma_value: float = DEFAULT_SIGMA,
    npy_k_index: int = DEFAULT_NPY_K_INDEX,
    plt_intervals: int = DEFAULT_PLT_INTERVALS,
    diag_slice_z_um: tuple = DEFAULT_DIAG_SLICE_Z_UM,
) -> dict:
    """Render the Artemis input. Returns the parameters actually used."""
    with open(meta_path) as f:
        meta = json.load(f)
    art = _require(meta, "artemis_inputs",
                   f"metafile {meta_path}")

    nx = int(_require(art, "n_cellx", "artemis_inputs"))
    ny = int(_require(art, "n_celly", "artemis_inputs"))
    nz = int(_require(art, "n_cellz", "artemis_inputs"))
    Lx_um = float(_require(art, "Lx_um", "artemis_inputs"))
    Ly_um = float(_require(art, "Ly_um", "artemis_inputs"))
    Lz_um = float(_require(art, "Lz_um", "artemis_inputs"))
    dx_um = float(_require(art, "dx_um", "artemis_inputs"))

    # Mask npy: always default to no_jj. The with_jj mask has a PEC bridge
    # across the JJ gap, which shorts out the lumped inductor. So both
    # use_inductor and no-inductor runs default to no_jj; override via
    # --mask-npy if you really want with_jj (e.g. for a galvanic-short sanity
    # check without the inductor).
    if mask_npy_path is None:
        mask_npy_path = _require(art, "mask_npy_no_jj", "artemis_inputs")

    # Source x straddles the two CPW air gaps (already widened in metafile)
    src_gaps = _require(art, "source_x_gaps_um", "artemis_inputs")
    if not src_gaps or len(src_gaps) < 2:
        raise ValueError(
            f"artemis_inputs.source_x_gaps_um in {meta_path} has fewer than "
            f"2 entries — rerun mask_from_gds.py on a design with a CPW."
        )
    (gap1_lo, gap1_hi), (gap2_lo, gap2_hi) = src_gaps[0], src_gaps[1]

    source_y_um = float(_require(art, "source_y_um", "artemis_inputs"))
    source_y_str = _fmt_int_or_float(source_y_um)

    # Inductor section. Artemis lumped inductor is summed in SERIES along the
    # current direction (x here, since inductor_x_function is for E_x). So the
    # value in the function is per-cell inductance: L_per_cell = L_total / N,
    # where N = number of cells inside the JJ x-region.
    inductor_section = ""
    n_jj_x_cells = None
    L_per_cell_H = None
    if use_inductor:
        if lj_nH is None:
            raise ValueError("--lj-nH is required when use_inductor=True")
        if not all(k in art for k in ("jj_x_lo_um", "jj_x_hi_um", "jj_y_um")):
            raise ValueError(
                f"Metafile has no JJ info (jj_x_lo_um/jj_x_hi_um/jj_y_um). "
                f"Either the GDS had no bridge layer, or use --no-inductor."
            )
        Lj_total_H = lj_nH * 1e-9
        jj_x_lo = float(art["jj_x_lo_um"])
        jj_x_hi = float(art["jj_x_hi_um"])
        jj_cy = float(art["jj_y_um"])
        n_jj_x_cells = max(1, round((jj_x_hi - jj_x_lo) / dx_um))
        L_per_cell_H = Lj_total_H / n_jj_x_cells
        inductor_section = (
            "algo.use_lumped_inductor = 1\n"
            f"# L_per_cell = L_total ({lj_nH} nH) / N_x_cells ({n_jj_x_cells})\n"
            f"inductor.inductor_x_function(x,y,z) = \"{L_per_cell_H:.6e} "
            "* (z > h_si - ddz) * (z < h_si + ddz) "
            f"* (x > {jj_x_lo}e-6 + ddx) * (x < {jj_x_hi}e-6 - ddx) "
            f"* (y > {jj_cy}e-6 - ddy) "
            f"* (y < {jj_cy}e-6 + ddy)\"\n"
            "inductor.inductor_y_function(x,y,z) = \"0.\"\n"
            "inductor.inductor_z_function(x,y,z) = \"0.\"\n"
        )
        jj_x_range = (jj_x_lo, jj_x_hi)
    else:
        jj_x_range = None

    freq_hz = freq_GHz * 1e9
    z_lo, z_hi = diag_slice_z_um

    template = f"""# Auto-generated Artemis input — from {os.path.basename(meta_path)}
max_step = {max_step}

amr.n_cell = n_cellx n_celly n_cellz
amr.max_grid_size = max_grid_sizex max_grid_sizey max_grid_sizez
amr.blocking_factor = blocking_factor
amr.refine_grid_layout = 1
geometry.dims = 3
geometry.prob_lo = prob_lox prob_loy prob_loz
geometry.prob_hi = prob_hix prob_hiy prob_hiz
amr.max_level = 0
boundary.field_lo = pml pml pml
boundary.field_hi = pml pml pml

warpx.verbose = 1
warpx.cfl = {cfl}
algo.em_solver_medium = macroscopic
algo.macroscopic_sigma_method = laxwendroff

macroscopic.sigma_function(x,y,z) = "sigma_0"
macroscopic.sigma_npy_file = "{mask_npy_path}"
macroscopic.sigma_npy_value = {sigma_value:.1e}
macroscopic.mu_npy_file = "{mask_npy_path}"
macroscopic.npy_k_index = {npy_k_index}
macroscopic.mu_npy_value = 1.25663707e-06
algo.use_PEC_mask = 1
macroscopic.epsilon_function(x,y,z) = "eps_0 + eps_0 * (eps_r_si - 1.) * (z <= h_si)"
my_constants.mu_0 = 1.25663706212e-06
macroscopic.mu_function(x,y,z) = "mu_0"

my_constants.n_cellx = {nx}
my_constants.n_celly = {ny}
my_constants.n_cellz = {nz}
my_constants.max_grid_sizex = {nx}
my_constants.max_grid_sizey = {ny // 2}
my_constants.max_grid_sizez = {nz // 2}
my_constants.blocking_factor = 2

my_constants.prob_lox = 0.
my_constants.prob_loy = 0.
my_constants.prob_loz = 0.
my_constants.prob_hix = {_fmt_int_or_float(Lx_um)}.e-6
my_constants.prob_hiy = {_fmt_int_or_float(Ly_um)}.e-6
my_constants.prob_hiz = {_fmt_int_or_float(Lz_um)}.e-6

my_constants.Lx = prob_hix - prob_lox
my_constants.Ly = prob_hiy - prob_loy
my_constants.Lz = prob_hiz - prob_loz

my_constants.sigma_0 = 0.0
my_constants.eps_0 = 8.8541878128e-12
my_constants.eps_r_si = {eps_r_si}
my_constants.mu_0 = 1.25663706212e-06
my_constants.h_si = {_fmt_int_or_float(h_si_um)}.e-6
my_constants.pi = 3.14159265358979
my_constants.freq = {freq_hz:.1e}
my_constants.TP = 1./freq
my_constants.dx = Lx / n_cellx
my_constants.dy = Ly / n_celly
my_constants.dz = Lz / n_cellz
my_constants.ddx = dx/1.e6
my_constants.ddy = dy/1.e6
my_constants.ddz = dz/1.e6
my_constants.flag_none = 0
my_constants.flag_hs = 1
my_constants.flag_ss = 2

{inductor_section}

warpx.E_excitation_on_grid_style = parse_E_excitation_grid_function
warpx.Ey_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ex_excitation_flag_function(x,y,z) = "flag_ss * ( (x >= {gap1_lo}e-6 + ddx) * (x < {gap1_hi}e-6 - ddx) + (x >= {gap2_lo}e-6 + ddx) * (x <= {gap2_hi}e-6 - ddx)) * (z >= h_si - ddz) * (z <= h_si + ddz) * (y > {source_y_str}e-6 - ddy) * (y < {source_y_str}e-6 + ddy)"
warpx.Ez_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ey_excitation_grid_function(x,y,z,t) = "0."
warpx.Ex_excitation_grid_function(x,y,z,t) = "exp(-(t-3*TP)**2/(2*TP**2))*sin(2*pi*freq*t) * ( (x >= {gap1_lo}e-6 + ddx) * (x < {gap1_hi}e-6 - ddx) - (x >= {gap2_lo}e-6 + ddx) * (x <= {gap2_hi}e-6 - ddx)) * (z >= h_si - ddz) * (z <= h_si + ddz) * (y > {source_y_str}e-6 - ddy) * (y < {source_y_str}e-6 + ddy)"
warpx.Ez_excitation_grid_function(x,y,z,t) = "0."

diagnostics.diags_names = plt
plt.intervals = {plt_intervals}
plt.fields_to_plot = Ex Ey Ez Bx By Bz mu sigma
plt.diag_type = Full
plt.file_min_digits = 7
plt.diag_lo = 0. 0. {_fmt_int_or_float(z_lo)}.e-6
plt.diag_hi = {nx - 0.5}e-6 {ny - 0.5}e-6 {_fmt_int_or_float(z_hi)}.e-6
"""

    with open(output_path, "w") as f:
        f.write(template)

    return {
        "output_file": output_path,
        "metafile": meta_path,
        "mask_npy": mask_npy_path,
        "freq_GHz": freq_GHz,
        "source_y_um": source_y_um,
        "source_x_gaps_um": src_gaps,
        "jj_x_range_um": jj_x_range,
        "Lj_total_nH": lj_nH,
        "n_jj_x_cells": n_jj_x_cells,
        "L_per_cell_nH": L_per_cell_H * 1e9 if L_per_cell_H is not None else None,
        "use_inductor": use_inductor,
        "grid": (nx, ny, nz),
        "domain_um": (Lx_um, Ly_um, Lz_um),
    }


# ────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("metafile",
                    help="path to mask metafile JSON (from mask_from_gds.py)")
    ap.add_argument("--output", "-o", default=None,
                    help="output Artemis input file path "
                         "(default: input_<basename-of-metafile>)")
    ap.add_argument("--lj-nH", type=float, default=None, dest="lj_nH",
                    help="Josephson inductance in nH (required unless "
                         "--no-inductor)")
    ap.add_argument("--freq-GHz", type=float, default=DEFAULT_FREQ_GHZ,
                    dest="freq_GHz",
                    help=f"qubit excitation frequency in GHz "
                         f"(default {DEFAULT_FREQ_GHZ})")
    ap.add_argument("--no-inductor", action="store_true",
                    help="disable the lumped inductor (no JJ in simulation; "
                         "uses the no_jj mask)")
    ap.add_argument("--mask-npy", type=str, default=None,
                    help="override which npy mask the input references "
                         "(default: from metafile, with_jj or no_jj per "
                         "--no-inductor)")

    ap.add_argument("--max-step", type=int, default=DEFAULT_MAX_STEP,
                    dest="max_step",
                    help=f"max time steps (default {DEFAULT_MAX_STEP})")
    ap.add_argument("--cfl", type=float, default=DEFAULT_CFL,
                    help=f"CFL number (default {DEFAULT_CFL})")
    ap.add_argument("--eps-r-si", type=float, default=DEFAULT_EPS_R_SI,
                    dest="eps_r_si",
                    help=f"silicon dielectric constant (default "
                         f"{DEFAULT_EPS_R_SI})")
    ap.add_argument("--h-si-um", type=float, default=DEFAULT_H_SI_UM,
                    dest="h_si_um",
                    help=f"silicon substrate thickness in μm (default "
                         f"{DEFAULT_H_SI_UM})")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                    dest="sigma_value",
                    help=f"metal conductivity in S/m (default "
                         f"{DEFAULT_SIGMA:.1e})")
    ap.add_argument("--npy-k-index", type=int, default=DEFAULT_NPY_K_INDEX,
                    dest="npy_k_index",
                    help=f"z-slice index of mu npy (default "
                         f"{DEFAULT_NPY_K_INDEX})")
    ap.add_argument("--plt-intervals", type=int, default=DEFAULT_PLT_INTERVALS,
                    dest="plt_intervals",
                    help=f"plotfile output cadence in steps (default "
                         f"{DEFAULT_PLT_INTERVALS})")
    ap.add_argument("--diag-slice-z-um", nargs=2, type=float,
                    default=list(DEFAULT_DIAG_SLICE_Z_UM),
                    dest="diag_slice_z_um",
                    metavar=("Z_LO", "Z_HI"),
                    help=f"diagnostic slice z range in μm "
                         f"(default {DEFAULT_DIAG_SLICE_Z_UM[0]} "
                         f"{DEFAULT_DIAG_SLICE_Z_UM[1]})")

    args = ap.parse_args(argv)

    if not os.path.exists(args.metafile):
        ap.error(f"metafile not found: {args.metafile}")

    use_inductor = not args.no_inductor
    if use_inductor and args.lj_nH is None:
        ap.error("--lj-nH is required unless --no-inductor is given")

    # Default output path: strip "_mask_meta.json" or ".json" → input_<base>
    output_path = args.output
    if output_path is None:
        base = os.path.basename(args.metafile)
        for suf in ("_mask_meta.json", "_meta.json", ".json"):
            if base.endswith(suf):
                base = base[:-len(suf)]
                break
        output_path = f"input_{base}"

    result = write_artemis_input(
        meta_path=args.metafile,
        output_path=output_path,
        lj_nH=args.lj_nH,
        freq_GHz=args.freq_GHz,
        use_inductor=use_inductor,
        mask_npy_path=args.mask_npy,
        max_step=args.max_step,
        cfl=args.cfl,
        eps_r_si=args.eps_r_si,
        h_si_um=args.h_si_um,
        sigma_value=args.sigma_value,
        npy_k_index=args.npy_k_index,
        plt_intervals=args.plt_intervals,
        diag_slice_z_um=tuple(args.diag_slice_z_um),
    )

    print(f"\nWrote Artemis input → {result['output_file']}")
    print(f"  metafile:    {result['metafile']}")
    print(f"  mask npy:    {result['mask_npy']}")
    print(f"  grid:        {result['grid'][0]} × {result['grid'][1]} × "
          f"{result['grid'][2]} cells")
    print(f"  domain:      {result['domain_um'][0]:.1f} × "
          f"{result['domain_um'][1]:.1f} × {result['domain_um'][2]:.1f} μm")
    print(f"  freq:        {result['freq_GHz']:.3f} GHz")
    print(f"  source y:    {result['source_y_um']:.1f} μm")
    print(f"  source x:    "
          f"[{result['source_x_gaps_um'][0][0]:.1f}, "
          f"{result['source_x_gaps_um'][0][1]:.1f}] μm  and  "
          f"[{result['source_x_gaps_um'][1][0]:.1f}, "
          f"{result['source_x_gaps_um'][1][1]:.1f}] μm")
    if result["use_inductor"]:
        print(f"  Lj_total:    {result['Lj_total_nH']} nH  "
              f"(x ∈ [{result['jj_x_range_um'][0]}, "
              f"{result['jj_x_range_um'][1]}] μm)")
        print(f"  N_x_cells:   {result['n_jj_x_cells']}  "
              f"(series cells along x in JJ region)")
        print(f"  L_per_cell:  {result['L_per_cell_nH']:.4f} nH  "
              f"= {result['Lj_total_nH']} / {result['n_jj_x_cells']}  "
              f"→ written into inductor_x_function")
    else:
        print(f"  Lj:          (disabled — --no-inductor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
