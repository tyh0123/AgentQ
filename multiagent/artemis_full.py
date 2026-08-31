"""
Artemis FDTD input for the HORIZONTAL-feedline topologies (qubit_resonator, multi).

qpipe_artemis assumes the qubit-only *vertical* CPW: an Ex source at a fixed y,
straddling two x-gaps. The qubit_resonator / multi feedline runs in x, so the
readout drive is transposed — an **Ey source at a fixed x** (input-port drive,
near wb_in), straddling the feedline's two y-gaps. The lumped Josephson inductor
is unchanged (each qubit's JJ bridge is along x, like validation); multi simply
sums one inductor term per qubit.

Everything else (grid, materials, diagnostics) mirrors qpipe_artemis so the two
writers stay in sync.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from qpipe_config import Config


# ──────────────────────────────────────────────────────────────
#  Feedline gap detection (transpose of qpipe_mask._detect_cpw_gaps)
# ──────────────────────────────────────────────────────────────
def _zero_runs(col: np.ndarray):
    runs, in_air, s = [], False, None
    for i, v in enumerate(col):
        if v == 0:
            if not in_air:
                s, in_air = i, True
        elif in_air:
            runs.append((s, i)); in_air = False
    if in_air:
        runs.append((s, len(col)))
    return runs


def detect_feedline_gaps(mask: np.ndarray, source_x: int, res_um: float,
                         center_y: int = None):
    """At column x=source_x, return the feedline's two y-gaps that flank the
    feedline centre conductor. With `center_y` (feedline cell y) the two gaps
    bracketing it are used — needed when qubits sit on BOTH sides of the
    feedline; otherwise the two topmost air runs are used. None if not found."""
    runs = _zero_runs(mask[source_x, :].astype(int))
    if len(runs) < 2:
        return None
    if center_y is None:
        g_lower, g_upper = sorted(sorted(runs, key=lambda r: -r[0])[:2], key=lambda r: r[0])
    else:
        below = [r for r in runs if (r[0] + r[1]) / 2 < center_y]
        above = [r for r in runs if (r[0] + r[1]) / 2 >= center_y]
        if not below or not above:
            return None
        g_lower = max(below, key=lambda r: (r[0] + r[1]) / 2)
        g_upper = min(above, key=lambda r: (r[0] + r[1]) / 2)
    return {
        "gap1_lo_um": g_lower[0] * res_um, "gap1_hi_um": g_lower[1] * res_um,
        "gap2_lo_um": g_upper[0] * res_um, "gap2_hi_um": g_upper[1] * res_um,
        "center_y_um": (g_lower[1] + g_upper[0]) / 2.0 * res_um,
    }


def pick_source_x(mask: np.ndarray, cfg: Config, center_y: int = None):
    """Choose a source x column: input-port side (near wb_in / left), clear of
    the left PML, landing on the feedline. Returns (source_x_cell, gaps)."""
    nx = mask.shape[0]
    res = cfg.mask.resolution_um
    pml_cells = int(cfg.artemis.pml_margin_um / res)
    for frac in (0.15, 0.18, 0.12, 0.20, 0.25):
        sx = max(int(nx * frac), pml_cells + 20)
        if sx >= nx:
            continue
        gaps = detect_feedline_gaps(mask, sx, res, center_y)
        if gaps:
            return sx, gaps
    return None, None


# ──────────────────────────────────────────────────────────────
#  Input writer
# ──────────────────────────────────────────────────────────────
def write_input(mask_info: dict, mask_npy: str, output_file: str, cfg: Config,
                jj_list: List[dict], source_x_cell: int, gaps: dict,
                freq_GHz: Optional[float] = None) -> dict:
    """Write the Artemis input with an Ey input-port source + one lumped inductor
    per qubit. `mask_info` needs n_cellx / n_celly / resolution_um. `jj_list`
    items: {x_lo_um, x_hi_um, cy_um, Lj_nH}."""
    A = cfg.artemis
    nx, ny = mask_info["n_cellx"], mask_info["n_celly"]
    res = float(mask_info["resolution_um"])
    freq_GHz = freq_GHz if freq_GHz is not None else cfg.target.qubit_frequency_GHz
    freq_hz = freq_GHz * 1e9
    source_x_um = source_x_cell * res

    # Ey source straddling the two feedline y-gaps at x = source_x (differential).
    g1lo, g1hi = gaps["gap1_lo_um"], gaps["gap1_hi_um"]
    g2lo, g2hi = gaps["gap2_lo_um"], gaps["gap2_hi_um"]

    # One lumped inductor per qubit JJ (each bridge is along x, like validation).
    terms = []
    for jj in jj_list:
        n_x = max(1, round((jj["x_hi_um"] - jj["x_lo_um"]) / res))
        L_per_cell = (jj["Lj_nH"] * 1e-9) / n_x
        cy = jj["cy_um"]
        terms.append(
            f"{L_per_cell:.6e} * (z > h_si - ddz) * (z < h_si + ddz) "
            f"* (x > {jj['x_lo_um']}e-6 + ddx) * (x < {jj['x_hi_um']}e-6 - ddx) "
            f"* (y > {cy}e-6 - ddy) * (y < {cy}e-6 + ddy)")
    inductor_expr = " + ".join(terms)
    inductor_section = (
        "algo.use_lumped_inductor = 1\n"
        f"# {len(jj_list)} lumped Lj term(s), one per qubit JJ (bridge along x)\n"
        f"inductor.inductor_x_function(x,y,z) = \"{inductor_expr}\"\n"
        "inductor.inductor_y_function(x,y,z) = \"0.\"\n"
        "inductor.inductor_z_function(x,y,z) = \"0.\"\n"
    )

    y_gate = (f"( (y >= {g1lo}e-6 + ddy) * (y < {g1hi}e-6 - ddy) "
              f"+ (y >= {g2lo}e-6 + ddy) * (y <= {g2hi}e-6 - ddy) )")
    y_drive = (f"( (y >= {g1lo}e-6 + ddy) * (y < {g1hi}e-6 - ddy) "
               f"- (y >= {g2lo}e-6 + ddy) * (y <= {g2hi}e-6 - ddy) )")
    x_gate = f"(x > {source_x_um}e-6 - ddx) * (x < {source_x_um}e-6 + ddx)"

    template = f"""# Auto-generated Artemis input (horizontal feedline, input-port Ey drive)
# source: Ey at x={source_x_um:.0f}um across feedline y-gaps [{g1lo:.0f},{g1hi:.0f}] & [{g2lo:.0f},{g2hi:.0f}]um
# NOTE: source placement is an input-port readout drive; validate before large runs.
max_step = {A.max_step}

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
warpx.cfl = {A.cfl}
algo.em_solver_medium = macroscopic
algo.macroscopic_sigma_method = laxwendroff

macroscopic.sigma_function(x,y,z) = "sigma_0"
macroscopic.sigma_npy_file = "{mask_npy}"
macroscopic.sigma_npy_value = {A.sigma_value:.1e}
macroscopic.mu_npy_file = "{mask_npy}"
macroscopic.npy_k_index = {A.npy_k_index}
macroscopic.mu_npy_value = 1.25663707e-06
algo.use_PEC_mask = 1
macroscopic.epsilon_function(x,y,z) = "eps_0 + eps_0 * (eps_r_si - 1.) * (z <= h_si)"
my_constants.mu_0 = 1.25663706212e-06
macroscopic.mu_function(x,y,z) = "mu_0"

my_constants.n_cellx = {nx}
my_constants.n_celly = {ny}
my_constants.n_cellz = {A.n_cellz}
my_constants.max_grid_sizex = {nx}
my_constants.max_grid_sizey = {ny // 2}
my_constants.max_grid_sizez = {A.n_cellz // 2}
my_constants.blocking_factor = 2

my_constants.prob_lox = 0.
my_constants.prob_loy = 0.
my_constants.prob_loz = 0.
my_constants.prob_hix = {nx}.e-6
my_constants.prob_hiy = {ny}.e-6
my_constants.prob_hiz = {A.prob_hiz_um}.e-6

my_constants.Lx = prob_hix - prob_lox
my_constants.Ly = prob_hiy - prob_loy
my_constants.Lz = prob_hiz - prob_loz

my_constants.sigma_0 = 0.0
my_constants.eps_0 = 8.8541878128e-12
my_constants.eps_r_si = {A.eps_r_si}
my_constants.mu_0 = 1.25663706212e-06
my_constants.h_si = {A.h_si_um}.e-6
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
warpx.Ex_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ey_excitation_flag_function(x,y,z) = "flag_ss * {y_gate} * (z >= h_si - ddz) * (z <= h_si + ddz) * {x_gate}"
warpx.Ez_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ex_excitation_grid_function(x,y,z,t) = "0."
warpx.Ey_excitation_grid_function(x,y,z,t) = "exp(-(t-3*TP)**2/(2*TP**2))*sin(2*pi*freq*t) * {y_drive} * (z >= h_si - ddz) * (z <= h_si + ddz) * {x_gate}"
warpx.Ez_excitation_grid_function(x,y,z,t) = "0."

diagnostics.diags_names = plt
plt.intervals = {A.plt_intervals}
plt.fields_to_plot = Ex Ey Ez Bx By Bz mu sigma
plt.diag_type = Full
plt.file_min_digits = 7
plt.diag_lo = 0. 0. {A.diag_slice_z_um[0]}.e-6
plt.diag_hi = {nx - 0.5}e-6 {ny - 0.5}e-6 {A.diag_slice_z_um[1]}.e-6
"""
    with open(output_file, "w") as f:
        f.write(template)

    return {
        "output_file": output_file,
        "freq_GHz": freq_GHz,
        "source_x_um": source_x_um,
        "feedline_center_y_um": gaps["center_y_um"],
        "n_inductors": len(jj_list),
        "grid": (nx, ny, A.n_cellz),
    }
