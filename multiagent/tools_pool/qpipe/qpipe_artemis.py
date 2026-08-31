"""
Generate Artemis FDTD input file from a mask metafile + config.
"""
from __future__ import annotations

from typing import Optional

from qpipe_config import Config


def auto_source_y(meta: dict, cfg: Config) -> float:
    """Source y = mask_top - PML margin.

    This assumes the CPW reaches the mask top (the layout-constrained topology
    for transmon-only mode). The source sits one PML-margin below the boundary,
    cleanly inside the CPW.
    """
    ny = meta["mask_info"]["n_celly"]
    return float(ny - cfg.artemis.pml_margin_um)


def write_input(meta: dict, mask_npy: str, output_file: str,
                cfg: Config, freq_GHz: Optional[float] = None,
                source_y_um: Optional[float] = None,
                use_inductor: bool = True) -> dict:
    """Render the Artemis input. Returns the parameters actually used."""
    A = cfg.artemis
    mi = meta["mask_info"]
    nx = mi["n_cellx"]
    ny = mi["n_celly"]
    jj = meta["junction_location"]
    Lj_H = meta["design_params"]["Lj_nH"] * 1e-9

    freq_GHz = freq_GHz if freq_GHz is not None else cfg.target.qubit_frequency_GHz
    source_y = source_y_um if source_y_um is not None else auto_source_y(meta, cfg)
    # Format as int when whole-number — matches verified Artemis input style
    source_y_str = f"{int(source_y)}" if float(source_y).is_integer() else f"{source_y}"

    jj_x_lo = jj["start"]["x_um"]
    jj_x_hi = jj["end"]["x_um"]
    jj_cx = jj["center"]["x_um"]
    jj_cy = jj["center"]["y_um"]
    dx_um = float(mi["resolution_um"])

    # JJ model: gap model bridges a single lumped Lj across the carved gap cell on
    # the solid-electrode with_jj mask; legacy distributes Lj over the bridge span
    # on the open-gap no_jj mask.
    gap_model = cfg.artemis.jj_gap_model
    mask_file = mi["mask_with_jj"] if gap_model else mask_npy

    # Source x must straddle the CPW air gaps (NOT the JJ x). The mask scanner
    # in qpipe_mask recorded the actual gap bounds; we widen by ±0.5 um per
    # the verified Artemis input so the source comfortably covers each gap.
    src = meta.get("cpw_source_gaps")
    if not src:
        raise RuntimeError(
            "Metafile is missing cpw_source_gaps. Regenerate the mask with "
            "the current qpipe_mask.py (which auto-detects the CPW gaps)."
        )
    widen = A.cpw_source_widening_um
    gap1_lo = src["gap1_lo_um"] - widen
    gap1_hi = src["gap1_hi_um"] + widen
    gap2_lo = src["gap2_lo_um"] - widen
    gap2_hi = src["gap2_hi_um"] + widen

    freq_hz = freq_GHz * 1e9

    inductor_section = ""
    n_jj_x_cells = None
    L_per_cell_H = None
    if use_inductor:
        # Artemis lumped inductor is summed in SERIES along the current
        # direction (x here). So inductor_x_function returns per-cell L:
        #   L_per_cell = L_total / N_x_cells
        if gap_model:
            # Bridge only the carved gap (jj_gap_cells wide) centered on the JJ.
            n_jj_x_cells = max(1, int(cfg.mask.jj_gap_cells))
            ind_x_lo = jj_cx - n_jj_x_cells * dx_um / 2.0
            ind_x_hi = jj_cx + n_jj_x_cells * dx_um / 2.0
            comment = (f"# single lumped Lj across the {n_jj_x_cells}-cell JJ gap "
                       f"at x={jj_cx}um (L_total {Lj_H*1e9} nH)")
        else:
            n_jj_x_cells = max(1, round((jj_x_hi - jj_x_lo) / dx_um))
            ind_x_lo = jj_x_lo
            ind_x_hi = jj_x_hi
            comment = f"# L_per_cell = L_total ({Lj_H*1e9} nH) / N_x_cells ({n_jj_x_cells})"
        L_per_cell_H = Lj_H / n_jj_x_cells
        # y is a single cell line at jj_cy, guarded by ddy on each side
        # (matches the verified Artemis input).
        inductor_section = (
            "algo.use_lumped_inductor = 1\n"
            f"{comment}\n"
            f"inductor.inductor_x_function(x,y,z) = \"{L_per_cell_H:.6e} "
            "* (z > h_si - ddz) * (z < h_si + ddz) "
            f"* (x > {ind_x_lo}e-6 + ddx) * (x < {ind_x_hi}e-6 - ddx) "
            f"* (y > {jj_cy}e-6 - ddy) "
            f"* (y < {jj_cy}e-6 + ddy)\"\n"
            "inductor.inductor_y_function(x,y,z) = \"0.\"\n"
            "inductor.inductor_z_function(x,y,z) = \"0.\"\n"
        )

    template = f"""# Auto-generated Artemis input
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
macroscopic.sigma_npy_file = "{mask_file}"
macroscopic.sigma_npy_value = {A.sigma_value:.1e}
macroscopic.mu_npy_file = "{mask_file}"
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
warpx.Ey_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ex_excitation_flag_function(x,y,z) = "flag_ss * ( (x >= {gap1_lo}e-6 + ddx) * (x < {gap1_hi}e-6 - ddx) + (x >= {gap2_lo}e-6 + ddx) * (x <= {gap2_hi}e-6 - ddx)) * (z >= h_si - ddz) * (z <= h_si + ddz) * (y > {source_y_str}e-6 - ddy) * (y < {source_y_str}e-6 + ddy)"
warpx.Ez_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ey_excitation_grid_function(x,y,z,t) = "0."
warpx.Ex_excitation_grid_function(x,y,z,t) = "exp(-(t-3*TP)**2/(2*TP**2))*sin(2*pi*freq*t) * ( (x >= {gap1_lo}e-6 + ddx) * (x < {gap1_hi}e-6 - ddx) - (x >= {gap2_lo}e-6 + ddx) * (x <= {gap2_hi}e-6 - ddx)) * (z >= h_si - ddz) * (z <= h_si + ddz) * (y > {source_y_str}e-6 - ddy) * (y < {source_y_str}e-6 + ddy)"
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
        "source_y_um": source_y,
        "jj_x_range_um": (jj_x_lo, jj_x_hi),
        "Lj_nH": Lj_H * 1e9,
        "n_jj_x_cells": n_jj_x_cells,
        "L_per_cell_nH": (L_per_cell_H * 1e9) if L_per_cell_H is not None else None,
        "use_inductor": use_inductor,
        "jj_gap_model": gap_model,
        "mask_file": mask_file,
        "grid": (nx, ny, A.n_cellz),
    }
