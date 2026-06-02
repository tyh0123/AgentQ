#!/usr/bin/env python3
"""
Standalone: GDS → npy masks + preview PNG + metafile JSON.

Self-contained — does NOT import from qpipe_* / cli.py / mcp pipeline. Safe to
copy into another project. Only deps: numpy, gdspy, scikit-image, matplotlib.

Includes all the bugfixes from the agentic pipeline:
  - C-contiguous npy output (Artemis reads C-order, not Python default F-order
    that .T views produce)
  - round() instead of int() for cell-count math (avoids floating-point
    off-by-one like 1099 vs 1100)
  - pad_cells > 0 guard on the trim slice (avoids `arr[:, :-0]` wiping the mask)
  - CPW air-gap auto-detection for downstream Artemis source-x placement

Physical grid:
    Specify cell sizes (dx, dy, dz) and physical extents (Lx, Ly, Lz)
    directly. Cell counts are derived: n_cellx = round(Lx/dx), etc. This
    avoids confusion where "target_nx=840" silently meant different physical
    sizes for different resolutions, and supports anisotropic dx ≠ dy.

Default layer convention (qiskit-metal / SQuADDS style):
    (1, 0)   = chip outline / ground plane
    (1, 10)  = etched / subtract regions
    (1, 11)  = positive features (CPW center conductor, etc.)
    (20, 10) = JJ bridge (only in the with_jj mask)

Outputs (for input X.gds → prefix X):
    X_mask_no_jj.npy        float64 (nx, ny, 1), C-contiguous
    X_mask_with_jj.npy      float64 (nx, ny, 1), C-contiguous
    X_mask_preview.png      side-by-side visualization
    X_mask_meta.json        shape, JJ location, per-spec bbox, CPW source
                            gaps, and a full `artemis_inputs` section with
                            ncells/dx/dy/dz/prob_hi/source_y for FDTD input

Usage:
    python mask_from_gds.py qubit.gds
    python mask_from_gds.py qubit.gds --dx 1.0 --dy 1.0 --Lx-um 840 --Ly-um 1100
    python mask_from_gds.py qubit.gds --dx 2.0 --dy 1.0 --Lx-um 840   # anisotropic
    python mask_from_gds.py qubit.gds --specs-no-jj 1,0 1,10 1,11 --specs-with-jj 1,0 1,10 1,11 20,10
    python mask_from_gds.py qubit.gds --pad-cut 0 --trim-margin 200 --no-preview
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import gdspy
import numpy as np
from skimage.draw import polygon as sk_polygon


# ────────────────────────────────────────────────────────────────────
#  Rasterize
# ────────────────────────────────────────────────────────────────────

def rasterize(poly_dict, specs, x_min, x_range, y_min, y_range, nx, ny):
    """Fill mask cells inside any polygon from the requested (layer, datatype)
    specs. Returns a (ny, nx) uint8 array — caller transposes if desired."""
    m = np.zeros((ny, nx), dtype=np.uint8)
    for s in specs:
        if s not in poly_dict:
            continue
        for poly in poly_dict[s]:
            xi = (poly[:, 0] - x_min) / (x_range + 1e-12) * (nx - 1)
            yi = (poly[:, 1] - y_min) / (y_range + 1e-12) * (ny - 1)
            rr, cc = sk_polygon(yi, xi)
            rr = np.clip(rr.astype(int), 0, ny - 1)
            cc = np.clip(cc.astype(int), 0, nx - 1)
            m[rr, cc] = 1
    return m


def bbox(poly_dict, specs):
    """Return (x_min, y_min, x_range, y_range, scale). `scale` is 1000 if the
    GDS units look like mm, else 1 (already in um)."""
    all_polys = []
    for s in specs:
        if s in poly_dict:
            all_polys.extend(poly_dict[s])
    if not all_polys:
        raise RuntimeError(f"No polygons found for any of specs {specs}")
    coords = np.vstack(all_polys)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    scale = 1000.0 if (x_range < 100 and y_range < 100) else 1.0
    return x_min, y_min, x_range, y_range, scale


def autocrop_bounds(m, margin_x, margin_y):
    """Find the active-region bbox (variance > 0.001) and add cell margins
    around it. margin_x and margin_y are in CELLS (caller converts μm → cells
    using the appropriate dx/dy). Returns (xs, xe, ys, ye)."""
    nx, ny = m.shape
    rv = np.array([m[i, :].var() for i in range(nx)])
    ax = np.where(rv > 0.001)[0]
    xs = max(0, ax[0] - margin_x) if len(ax) else 0
    xe = min(nx, ax[-1] + margin_x) if len(ax) else nx
    cv = np.array([m[:, j].var() for j in range(ny)])
    ay = np.where(cv > 0.001)[0]
    ys = max(0, ay[0] - margin_y) if len(ay) else 0
    ye = min(ny, ay[-1] + margin_y) if len(ay) else ny
    return xs, xe, ys, ye


def per_spec_info(poly_dict, specs, x_min, x_range, y_min, y_range,
                  nx, ny, xs, xe, ys, ye, pad_cells, target_nx, target_ny,
                  dx_um, dy_um):
    """For each spec separately, rasterize and report its bbox in the final
    (cropped + pad_cut + target_nx + target_ny) mask coordinates. Uses dx_um
    for x → μm and dy_um for y → μm (supports anisotropic resolution)."""
    info = {}
    for s in specs:
        if s not in poly_dict:
            continue
        sm = rasterize(poly_dict, [s], x_min, x_range, y_min, y_range, nx, ny).T
        sm = sm[xs:xe, ys:ye]
        if pad_cells > 0 and sm.shape[1] > pad_cells:
            sm = sm[:, :-pad_cells]
        # x crop/pad
        if target_nx and sm.shape[0] != target_nx:
            if sm.shape[0] > target_nx:
                trim = (sm.shape[0] - target_nx) // 2
                sm = sm[trim:trim + target_nx, :]
            else:
                pad_total = target_nx - sm.shape[0]
                pl = pad_total // 2
                pr = pad_total - pl
                sm = np.pad(sm, ((pl, pr), (0, 0)),
                            mode="constant", constant_values=0)
        # y crop/pad (bottom only)
        if target_ny and sm.shape[1] != target_ny:
            if sm.shape[1] > target_ny:
                sm = sm[:, sm.shape[1] - target_ny:]
            else:
                pad = target_ny - sm.shape[1]
                sm = np.pad(sm, ((0, 0), (pad, 0)),
                            mode="constant", constant_values=0)
        loc = np.where(sm > 0)
        if len(loc[0]) > 0:
            info[f"layer_{s[0]}_dt_{s[1]}"] = {
                "ix_range": [int(loc[0].min()), int(loc[0].max())],
                "iy_range": [int(loc[1].min()), int(loc[1].max())],
                "x_range_um": [float(loc[0].min() * dx_um),
                               float(loc[0].max() * dx_um)],
                "y_range_um": [float(loc[1].min() * dy_um),
                               float(loc[1].max() * dy_um)],
                "pixel_count": int(sm.sum()),
            }
    return info


# ────────────────────────────────────────────────────────────────────
#  CPW gap detection (for Artemis source-x placement downstream)
# ────────────────────────────────────────────────────────────────────

def detect_cpw_gaps(m_no, nx, ny, spec_info, dx_um, dy_um,
                    cpw_layer_key="layer_1_dt_11"):
    """Find the two air gaps flanking the CPW center conductor, sampled at
    the mid-y of the CPW. Returns a dict with gap1/gap2 bounds in um. Uses
    dx_um for x-axis cell → μm and dy_um for the sample-y annotation."""
    cpw = spec_info.get(cpw_layer_key)
    if cpw is None:
        return None
    iy_lo, iy_hi = cpw["iy_range"]
    sample_y = (iy_lo + iy_hi) // 2
    if sample_y < 0 or sample_y >= ny:
        return None

    row = m_no[:, sample_y].astype(np.int64)
    cx = nx // 2
    search = 100
    lo = max(0, cx - search)
    hi = min(nx, cx + search)
    runs = []
    in_air = False
    start = None
    for i in range(lo, hi):
        if row[i] == 0:
            if not in_air:
                start = i
                in_air = True
        elif in_air:
            runs.append((start, i))
            in_air = False
    if in_air:
        runs.append((start, hi))

    if len(runs) < 2:
        return None
    runs.sort(key=lambda r: abs((r[0] + r[1]) / 2 - cx))
    g1, g2 = sorted(runs[:2], key=lambda r: r[0])
    return {
        "sample_y_um": float(sample_y * dy_um),
        "gap1_lo_um": float(g1[0] * dx_um),
        "gap1_hi_um": float(g1[1] * dx_um),
        "gap2_lo_um": float(g2[0] * dx_um),
        "gap2_hi_um": float(g2[1] * dx_um),
    }


# ────────────────────────────────────────────────────────────────────
#  Preview PNG
# ────────────────────────────────────────────────────────────────────

def write_preview(m_no, m_jj, jj_cx, jj_cy, out_png,
                  cpw_gaps=None, source_y=None, pml_margin_um=None,
                  dx_um=1.0, dy_um=1.0):
    """Side-by-side mask preview, annotated with JJ, CPW air gaps, source y,
    and PML region. cpw_gaps dict from detect_cpw_gaps(). source_y is the
    Artemis excitation y in μm. pml_margin_um draws a hatched band on the +y
    boundary. dx_um/dy_um are used to convert μm-valued annotations back to
    cell indices for plotting (image axes are in cells).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    nx, ny = m_no.shape
    Lx_um = nx * dx_um
    Ly_um = ny * dy_um

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    title_suffix = (f"{nx} × {ny} cells @ {dx_um} × {dy_um} μm/cell  "
                    f"({Lx_um:.0f} × {Ly_um:.0f} μm)")
    axes[0].imshow(m_no.T, cmap="viridis", origin="lower", aspect="auto")
    axes[0].set_title(f"Without JJ bridge — {title_suffix}")
    axes[1].imshow(m_jj.T, cmap="viridis", origin="lower", aspect="auto")
    axes[1].set_title(f"With JJ bridge — {title_suffix}")

    for ax in axes:
        # JJ marker
        ax.plot(jj_cx, jj_cy, "r+", markersize=18, markeredgewidth=2.5)
        ax.annotate(f"JJ ({jj_cx},{jj_cy})", (jj_cx, jj_cy),
                    color="red", fontsize=10,
                    xytext=(12, 12), textcoords="offset points")

        # CPW air gap shading — red rectangles spanning full y (x-axis: use dx)
        if cpw_gaps:
            for gl, gh in [(cpw_gaps["gap1_lo_um"], cpw_gaps["gap1_hi_um"]),
                           (cpw_gaps["gap2_lo_um"], cpw_gaps["gap2_hi_um"])]:
                glc = gl / dx_um
                ghc = gh / dx_um
                ax.add_patch(mpatches.Rectangle(
                    (glc, 0), ghc - glc, ny,
                    facecolor="red", alpha=0.15, edgecolor="red",
                    linewidth=0.5, label="_nolegend_",
                ))

        # Source y line (y-axis: use dy)
        if source_y is not None:
            sy_cell = source_y / dy_um
            ax.axhline(sy_cell, color="orange", linewidth=1.5, linestyle="--",
                       label=f"Artemis source y = {source_y:.0f} μm")
            ax.annotate(f"source y={source_y:.0f}", (nx * 0.02, sy_cell + 8),
                        color="orange", fontsize=9)

        # PML region shading along +y boundary (y-axis: use dy)
        if pml_margin_um is not None and pml_margin_um > 0:
            pml_cells = pml_margin_um / dy_um
            ax.add_patch(mpatches.Rectangle(
                (0, ny - pml_cells), nx, pml_cells,
                facecolor="white", alpha=0.0, edgecolor="black",
                hatch="//", linewidth=0.5, label="_nolegend_",
            ))
            ax.annotate(f"PML ({pml_margin_um:.0f} μm)",
                        (nx * 0.78, ny - pml_cells / 2),
                        color="black", fontsize=9, ha="center", va="center",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        ax.set_xlabel(f"x (cells, dx={dx_um} μm)")
        ax.set_ylabel(f"y (cells, dy={dy_um} μm)")
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        if source_y is not None:
            ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────────
#  Main pipeline
# ────────────────────────────────────────────────────────────────────

def gds_to_mask(
    gds_file: str,
    prefix: str,
    specs_no_jj: List[Tuple[int, int]],
    specs_with_jj: List[Tuple[int, int]],
    dx_um: float = 1.0,
    dy_um: float = 1.0,
    dz_um: float = 5.0,
    Lx_um: Optional[float] = 840.0,
    Ly_um: Optional[float] = None,
    Lz_um: float = 200.0,
    pad_cut_um: float = 0.0,
    trim_margin_um: float = 200.0,
    write_preview_png: bool = True,
    pml_margin_um: float = 100.0,
) -> dict:
    """Rasterize GDS → two npy masks + metafile + optional PNG.

    Grid: cell sizes dx_um, dy_um, dz_um (μm); physical extents Lx_um, Ly_um,
    Lz_um (μm). Cell counts derived: n_cellx = round(Lx_um/dx_um), etc. Set
    Lx_um=None or Ly_um=None to autocrop in that axis. Lz/dz only affect the
    metafile (z is not rasterized — masks are 2D extruded by Artemis).
    """
    # Derived cell counts (only x and y matter for rasterization)
    target_nx = round(Lx_um / dx_um) if Lx_um else None
    target_ny = round(Ly_um / dy_um) if Ly_um else None
    n_cellz = max(1, round(Lz_um / dz_um))

    # Load GDS
    gdsii = gdspy.GdsLibrary()
    gdsii.read_gds(gds_file)
    cell = gdsii.top_level()[0]
    poly_dict = cell.get_polygons(by_spec=True)

    # Initial raster grid sized so dx_physical = dx_um and dy_physical = dy_um
    # (independently — supports anisotropic resolution without distortion)
    x_min, y_min, x_range, y_range, scale = bbox(poly_dict, specs_with_jj)
    nx = round(x_range * scale / dx_um)
    ny = round(y_range * scale / dy_um)

    # Rasterize both masks
    m_no = rasterize(poly_dict, specs_no_jj,
                     x_min, x_range, y_min, y_range, nx, ny).T
    m_jj = rasterize(poly_dict, specs_with_jj,
                     x_min, x_range, y_min, y_range, nx, ny).T

    # Autocrop with per-axis margins (trim_margin_um is in physical μm,
    # convert to cells differently per axis since dx may != dy)
    margin_x = int(trim_margin_um / dx_um)
    margin_y = int(trim_margin_um / dy_um)
    xs, xe, ys, ye = autocrop_bounds(m_jj, margin_x, margin_y)
    m_no = m_no[xs:xe, ys:ye]
    m_jj = m_jj[xs:xe, ys:ye]

    # Pad cut from top — guard against pad_cells == 0 making `:-0` strip everything
    pad_cells = int(pad_cut_um / dy_um)
    if pad_cells > 0 and m_no.shape[1] > pad_cells:
        m_no = m_no[:, :-pad_cells]
        m_jj = m_jj[:, :-pad_cells]

    # Force x dimension via center crop or pad with ground
    if target_nx and m_no.shape[0] != target_nx:
        if m_no.shape[0] > target_nx:
            trim = (m_no.shape[0] - target_nx) // 2
            m_no = m_no[trim:trim + target_nx, :]
            m_jj = m_jj[trim:trim + target_nx, :]
        else:
            pad_total = target_nx - m_no.shape[0]
            pad_lo = pad_total // 2
            pad_hi = pad_total - pad_lo
            m_no = np.pad(m_no, ((pad_lo, pad_hi), (0, 0)),
                          mode="constant", constant_values=1)
            m_jj = np.pad(m_jj, ((pad_lo, pad_hi), (0, 0)),
                          mode="constant", constant_values=1)

    # Force y dimension. The CPW reaches the top of the mask (FDTD topology
    # constraint), so we only modify the bottom: crop from bottom if too tall,
    # pad ground at the bottom if too short.
    y_pad_lo = 0
    if target_ny and m_no.shape[1] != target_ny:
        if m_no.shape[1] > target_ny:
            trim = m_no.shape[1] - target_ny
            m_no = m_no[:, trim:]
            m_jj = m_jj[:, trim:]
            y_pad_lo = -trim  # negative => bottom rows were removed
        else:
            pad = target_ny - m_no.shape[1]
            m_no = np.pad(m_no, ((0, 0), (pad, 0)),
                          mode="constant", constant_values=1)
            m_jj = np.pad(m_jj, ((0, 0), (pad, 0)),
                          mode="constant", constant_values=1)
            y_pad_lo = pad  # positive => bottom was padded with ground

    nx_f, ny_f = m_no.shape

    # Junction location from bridge diff (explicit int64 — uint8 reduction bug)
    bridge = m_jj.astype(np.int64) - m_no.astype(np.int64)
    locs = np.where(bridge > 0)
    has_jj = len(locs[0]) > 0
    if has_jj:
        jj_ix_min, jj_ix_max = int(locs[0].min()), int(locs[0].max())
        jj_iy_min, jj_iy_max = int(locs[1].min()), int(locs[1].max())
        jj_ix_center = (jj_ix_min + jj_ix_max) // 2
        jj_iy_center = (jj_iy_min + jj_iy_max) // 2
    else:
        jj_ix_center, jj_iy_center = nx_f // 4, ny_f // 3
        jj_ix_min, jj_ix_max = jj_ix_center - 1, jj_ix_center + 1
        jj_iy_min, jj_iy_max = jj_iy_center - 1, jj_iy_center + 1

    # Per-spec bbox info and CPW gaps
    spec_info = per_spec_info(
        poly_dict, specs_with_jj, x_min, x_range, y_min, y_range,
        nx, ny, xs, xe, ys, ye, pad_cells, target_nx, target_ny, dx_um, dy_um
    )
    cpw_source_gaps = detect_cpw_gaps(m_no, nx_f, ny_f, spec_info,
                                      dx_um, dy_um)

    # Save npy — C-contiguous, float64 (Artemis convention)
    files = {
        "no_jj": f"{prefix}_mask_no_jj.npy",
        "with_jj": f"{prefix}_mask_with_jj.npy",
        "meta": f"{prefix}_mask_meta.json",
    }
    np.save(files["no_jj"],
            np.ascontiguousarray(m_no[:, :, np.newaxis].astype(np.float64)))
    np.save(files["with_jj"],
            np.ascontiguousarray(m_jj[:, :, np.newaxis].astype(np.float64)))

    # Artemis source y = mask_top - PML margin (the layout-constrained position)
    source_y_um = ny_f * dy_um - pml_margin_um

    if write_preview_png:
        files["preview"] = f"{prefix}_mask_preview.png"
        write_preview(
            m_no, m_jj, jj_ix_center, jj_iy_center, files["preview"],
            cpw_gaps=cpw_source_gaps,
            source_y=source_y_um,
            pml_margin_um=pml_margin_um,
            dx_um=dx_um, dy_um=dy_um,
        )

    # Source x ranges (in μm) — derived from CPW gap detection, widened by
    # half a cell on each side so the source flag straddles the air comfortably
    source_x_gaps_um = None
    if cpw_source_gaps:
        w = 0.5 * dx_um
        source_x_gaps_um = [
            [cpw_source_gaps["gap1_lo_um"] - w,
             cpw_source_gaps["gap1_hi_um"] + w],
            [cpw_source_gaps["gap2_lo_um"] - w,
             cpw_source_gaps["gap2_hi_um"] + w],
        ]

    # Artemis input block — everything an FDTD input file needs
    artemis_inputs = {
        "n_cellx": int(nx_f),
        "n_celly": int(ny_f),
        "n_cellz": int(n_cellz),
        "dx_um": float(dx_um),
        "dy_um": float(dy_um),
        "dz_um": float(dz_um),
        "dx_m": float(dx_um) * 1e-6,
        "dy_m": float(dy_um) * 1e-6,
        "dz_m": float(dz_um) * 1e-6,
        "prob_lox_m": 0.0,
        "prob_loy_m": 0.0,
        "prob_loz_m": 0.0,
        "prob_hix_m": nx_f * dx_um * 1e-6,
        "prob_hiy_m": ny_f * dy_um * 1e-6,
        "prob_hiz_m": n_cellz * dz_um * 1e-6,
        "Lx_um": float(nx_f * dx_um),
        "Ly_um": float(ny_f * dy_um),
        "Lz_um": float(n_cellz * dz_um),
        "pml_margin_um": float(pml_margin_um),
        "source_y_um": float(source_y_um),
        "source_x_gaps_um": source_x_gaps_um,
        "mask_npy_no_jj": files["no_jj"],
        "mask_npy_with_jj": files["with_jj"],
    }
    if has_jj:
        artemis_inputs["jj_x_lo_um"] = float(jj_ix_min * dx_um)
        artemis_inputs["jj_x_hi_um"] = float(jj_ix_max * dx_um)
        artemis_inputs["jj_y_um"] = float(jj_iy_center * dy_um)

    # Metafile
    meta = {
        "source_gds": os.path.abspath(gds_file),
        "mask_info": {
            "mask_no_jj": files["no_jj"],
            "mask_with_jj": files["with_jj"],
            "preview_png": files.get("preview"),
            "specs_no_jj": [[s[0], s[1]] for s in specs_no_jj],
            "specs_with_jj": [[s[0], s[1]] for s in specs_with_jj],
            "n_cellx": int(nx_f),
            "n_celly": int(ny_f),
            "n_cellz": int(n_cellz),
            "dx_um": float(dx_um),
            "dy_um": float(dy_um),
            "dz_um": float(dz_um),
            "Lx_um": float(nx_f * dx_um),
            "Ly_um": float(ny_f * dy_um),
            "Lz_um": float(n_cellz * dz_um),
            "metal_fraction_no_jj": float(m_no.sum() / m_no.size),
            "metal_fraction_with_jj": float(m_jj.sum() / m_jj.size),
        },
        "junction_location": {
            "present": bool(has_jj),
            "start": {"ix": jj_ix_min, "iy": jj_iy_min,
                      "x_um": float(jj_ix_min * dx_um),
                      "y_um": float(jj_iy_min * dy_um)},
            "end": {"ix": jj_ix_max, "iy": jj_iy_max,
                    "x_um": float(jj_ix_max * dx_um),
                    "y_um": float(jj_iy_max * dy_um)},
            "center": {"ix": jj_ix_center, "iy": jj_iy_center,
                       "x_um": float(jj_ix_center * dx_um),
                       "y_um": float(jj_iy_center * dy_um)},
        },
        "mask_feature_locations": spec_info,
        "cpw_source_gaps": cpw_source_gaps,
        "artemis_inputs": artemis_inputs,
        "trim_applied": {
            "x_start": int(xs), "x_end": int(xe),
            "y_start": int(ys), "y_end_before_pad_cut": int(ye),
            "pad_cut_um": pad_cut_um,
        },
    }
    with open(files["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    return {"files": files, "meta": meta,
            "mask_no_jj": m_no, "mask_with_jj": m_jj}


# ────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────

def parse_spec(s: str) -> Tuple[int, int]:
    """Parse 'layer,datatype' (e.g. '1,10') into (1, 10). Also accepts '1/10'."""
    parts = s.replace("/", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Spec must be 'layer,datatype' (got {s!r})"
        )
    return (int(parts[0]), int(parts[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("gds_file", help="path to input GDS")
    ap.add_argument("--prefix", default=None,
                    help="output prefix (default: GDS filename without .gds)")
    ap.add_argument("--specs-no-jj", nargs="*", type=parse_spec,
                    default=[(1, 0), (1, 10), (1, 11)],
                    help="layer specs to rasterize for the no-JJ mask "
                         "(e.g. 1,0 1,10 1,11)")
    ap.add_argument("--specs-with-jj", nargs="*", type=parse_spec,
                    default=[(1, 0), (1, 10), (1, 11), (20, 10)],
                    help="layer specs to rasterize for the with-JJ mask")
    ap.add_argument("--dx", type=float, default=1.0,
                    help="cell size in x (μm/cell, default 1.0)")
    ap.add_argument("--dy", type=float, default=1.0,
                    help="cell size in y (μm/cell, default 1.0)")
    ap.add_argument("--dz", type=float, default=5.0,
                    help="cell size in z (μm/cell, default 5.0) — metafile only "
                         "since masks are 2D extruded by Artemis")
    ap.add_argument("--Lx-um", type=float, default=840.0, dest="Lx_um",
                    help="physical mask x extent in μm (default 840). "
                         "n_cellx = round(Lx/dx). Set 0 to autocrop in x.")
    ap.add_argument("--Ly-um", type=float, default=0.0, dest="Ly_um",
                    help="physical mask y extent in μm (default 0 = autocrop). "
                         "n_celly = round(Ly/dy). Crop/pad at bottom only so "
                         "CPW stays at the top.")
    ap.add_argument("--Lz-um", type=float, default=200.0, dest="Lz_um",
                    help="physical simulation z extent in μm (default 200). "
                         "n_cellz = round(Lz/dz). Metafile only.")
    ap.add_argument("--pad-cut", type=float, default=0.0,
                    help="trim this many μm off the top of the mask "
                         "(default 0; use e.g. 500 to strip a wirebond pad)")
    ap.add_argument("--trim-margin", type=float, default=200.0,
                    help="autocrop margin around the active region in μm "
                         "(default 200)")
    ap.add_argument("--pml-margin", type=float, default=100.0,
                    help="PML margin in μm (drawn on PNG and used to compute "
                         "Artemis source y = mask_top - pml_margin)")
    ap.add_argument("--no-preview", action="store_true",
                    help="skip writing the preview PNG")
    args = ap.parse_args(argv)

    if not os.path.exists(args.gds_file):
        ap.error(f"GDS not found: {args.gds_file}")

    prefix = args.prefix or os.path.splitext(os.path.basename(args.gds_file))[0]
    Lx_um = args.Lx_um if args.Lx_um > 0 else None
    Ly_um = args.Ly_um if args.Ly_um > 0 else None
    target_nx = round(Lx_um / args.dx) if Lx_um else None
    target_ny = round(Ly_um / args.dy) if Ly_um else None
    n_cellz = max(1, round(args.Lz_um / args.dz))

    t0 = time.time()
    print(f"\nRasterizing {args.gds_file}")
    print(f"  specs_no_jj   = {args.specs_no_jj}")
    print(f"  specs_with_jj = {args.specs_with_jj}")
    print(f"  dx, dy, dz    = {args.dx}, {args.dy}, {args.dz} μm/cell")
    print(f"  Lx, Ly, Lz    = {Lx_um}, {Ly_um}, {args.Lz_um} μm  "
          f"→ n_cellx, n_celly, n_cellz = {target_nx}, {target_ny}, {n_cellz}")
    print(f"  pad_cut       = {args.pad_cut} μm")
    print(f"  trim_margin   = {args.trim_margin} μm")
    print(f"  pml_margin    = {args.pml_margin} μm")

    result = gds_to_mask(
        gds_file=args.gds_file,
        prefix=prefix,
        specs_no_jj=args.specs_no_jj,
        specs_with_jj=args.specs_with_jj,
        dx_um=args.dx,
        dy_um=args.dy,
        dz_um=args.dz,
        Lx_um=Lx_um,
        Ly_um=Ly_um,
        Lz_um=args.Lz_um,
        pad_cut_um=args.pad_cut,
        trim_margin_um=args.trim_margin,
        pml_margin_um=args.pml_margin,
        write_preview_png=not args.no_preview,
    )

    info = result["meta"]["mask_info"]
    art = result["meta"]["artemis_inputs"]
    jl = result["meta"]["junction_location"]
    jjc = jl["center"]
    print(f"\nOutputs (prefix='{prefix}'):")
    for k, v in result["files"].items():
        print(f"  {k:8s} -> {v}")
    print(f"\nMask shape: {info['n_cellx']} × {info['n_celly']} × {info['n_cellz']} "
          f"cells @ {info['dx_um']} × {info['dy_um']} × {info['dz_um']} μm/cell")
    print(f"Domain:     {info['Lx_um']:.1f} × {info['Ly_um']:.1f} × {info['Lz_um']:.1f} μm")
    print(f"Metal fraction (no_jj):   {info['metal_fraction_no_jj']:.3f}")
    print(f"Metal fraction (with_jj): {info['metal_fraction_with_jj']:.3f}")
    if jl["present"]:
        print(f"JJ index: ({jjc['ix']}, {jjc['iy']}) "
              f"-> ({jjc['x_um']:.1f} μm, {jjc['y_um']:.1f} μm)")
    else:
        print("JJ: not detected (no bridge layer or empty diff)")
    if result["meta"]["cpw_source_gaps"]:
        g = result["meta"]["cpw_source_gaps"]
        print(f"CPW air gaps (for Artemis source x):")
        print(f"  gap1: [{g['gap1_lo_um']}, {g['gap1_hi_um']}] μm")
        print(f"  gap2: [{g['gap2_lo_um']}, {g['gap2_hi_um']}] μm")
    print(f"Artemis source y: {art['source_y_um']:.1f} μm "
          f"(mask_top {info['Ly_um']:.0f} μm − PML {art['pml_margin_um']:.0f} μm)")
    print(f"\nDone in {time.time() - t0:.1f}s.\n")

    # Verify C-contig (Artemis requirement)
    m_check = np.load(result["files"]["with_jj"])
    print(f"Sanity: with_jj mask is C-contiguous = {m_check.flags['C_CONTIGUOUS']} "
          f"(must be True for Artemis)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
