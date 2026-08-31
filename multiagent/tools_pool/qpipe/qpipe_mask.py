"""
Rasterize a GDS into:
  - {prefix}_mask_no_jj.npy
  - {prefix}_mask_with_jj.npy
  - {prefix}_mask_preview.png
  - {prefix}_mask_meta.json

Plus sanity checks against the mask.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gdspy
import numpy as np
from skimage.draw import polygon as sk_polygon

from qpipe_config import Config


# ──────────────────────────────────────────────
#  Result container
# ──────────────────────────────────────────────

@dataclass
class MaskResult:
    mask_no_jj: np.ndarray   # uint8 (nx, ny)
    mask_with_jj: np.ndarray
    meta: dict
    files: dict              # {"no_jj": str, "with_jj": str, "preview": str, "meta": str}


# ──────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────

def rasterize(gds_file: str, prefix: str, cfg: Config,
              dq: dict, Lj_nH: float) -> MaskResult:
    M = cfg.mask
    gdsii = gdspy.GdsLibrary()
    gdsii.read_gds(gds_file)
    cell = gdsii.top_level()[0]
    poly_dict = cell.get_polygons(by_spec=True)

    x_min, y_min, x_range, y_range, scale = _bbox(poly_dict, M.specs_with_jj)
    nx = round(x_range * scale / M.resolution_um)
    ny = round(y_range * scale / M.resolution_um)

    m_no = _rasterize(poly_dict, M.specs_no_jj,
                      x_min, x_range, y_min, y_range, nx, ny).T
    m_jj = _rasterize(poly_dict, M.specs_with_jj,
                      x_min, x_range, y_min, y_range, nx, ny).T

    xs, xe, ys, ye = _autocrop_bounds(m_jj, int(M.trim_margin_um / M.resolution_um))
    m_no = m_no[xs:xe, ys:ye]
    m_jj = m_jj[xs:xe, ys:ye]

    pad_cells = int(M.pad_cut_um / M.resolution_um)
    if pad_cells > 0 and m_no.shape[1] > pad_cells:
        m_no = m_no[:, :-pad_cells]
        m_jj = m_jj[:, :-pad_cells]

    if M.target_nx and m_no.shape[0] > M.target_nx:
        trim = (m_no.shape[0] - M.target_nx) // 2
        m_no = m_no[trim:trim + M.target_nx, :]
        m_jj = m_jj[trim:trim + M.target_nx, :]

    nx_f, ny_f = m_no.shape

    # Junction location — the JJ-only metal (present in with_jj, absent in no_jj:
    # the bridge layer 20 + junction-window layer 60).
    bridge = m_jj.astype(np.int64) - m_no.astype(np.int64)
    locs = np.where(bridge > 0)
    if len(locs[0]) > 0:
        jj_ix_min, jj_ix_max = int(locs[0].min()), int(locs[0].max())
        jj_iy_min, jj_iy_max = int(locs[1].min()), int(locs[1].max())
        # Carve a `jj_gap_cells`-wide air gap through the JJ center, splitting the
        # bridge into two electrodes so a lumped Lj bridges exactly one cell in
        # FDTD. Without this the sub-cell (0.14um) Dolan gap rasterizes solid and
        # the junction is shorted. The gap severs the bridge's long axis.
        if M.jj_gap_cells > 0:
            m_jj = _carve_jj_gap(m_jj, jj_ix_min, jj_ix_max,
                                 jj_iy_min, jj_iy_max, M.jj_gap_cells)
        jj_ix_center = (jj_ix_min + jj_ix_max) // 2
        jj_iy_center = (jj_iy_min + jj_iy_max) // 2
    else:
        jj_ix_center, jj_iy_center = nx_f // 4, ny_f // 3
        jj_ix_min, jj_ix_max = jj_ix_center - 1, jj_ix_center + 1
        jj_iy_min, jj_iy_max = jj_iy_center - 1, jj_iy_center + 1

    spec_info = _per_spec_info(
        poly_dict, M.specs_with_jj, x_min, x_range, y_min, y_range,
        nx, ny, xs, xe, ys, ye, pad_cells, M.target_nx, M.resolution_um
    )

    # Detect actual CPW air-gap positions in the mask near the top of the CPW.
    # We scan a few rows midway through the CPW range and confirm both gaps are
    # present and consistent. These bounds drive the Artemis source placement —
    # the source must sit *inside* the CPW air gaps, not on the ground plane.
    cpw_source_gaps = _detect_cpw_gaps(m_no, nx_f, ny_f, spec_info, M.resolution_um)

    files = {
        "no_jj": f"{prefix}_mask_no_jj.npy",
        "with_jj": f"{prefix}_mask_with_jj.npy",
        "preview": f"{prefix}_mask_preview.png",
        "meta": f"{prefix}_mask_meta.json",
    }
    # Save as C-contiguous (Artemis reads bytes in C-order — the .T above made
    # the arrays F-contig views, so we must explicitly copy here).
    np.save(files["no_jj"], np.ascontiguousarray(m_no[:, :, np.newaxis].astype(np.float64)))
    np.save(files["with_jj"], np.ascontiguousarray(m_jj[:, :, np.newaxis].astype(np.float64)))

    _write_preview(m_no, m_jj, jj_ix_center, jj_iy_center, files["preview"])

    expected_f_GHz = 1 / (2 * np.pi * np.sqrt(Lj_nH * 1e-9 * 80e-15)) / 1e9
    meta = {
        "target_params": {
            "qubit_frequency_GHz": cfg.target.qubit_frequency_GHz,
            "anharmonicity_MHz": cfg.target.anharmonicity_MHz,
            "cavity_frequency_GHz": cfg.target.cavity_frequency_GHz,
            "g_MHz": cfg.target.g_MHz,
        },
        "design_params": {
            "cross_length_um": _um(dq["cross_length"][0]),
            "cross_width_um": _um(dq["cross_width"][0]),
            "cross_gap_um": _um(dq["cross_gap"][0]),
            "claw_length_um": _um(dq["claw_length"][0]),
            "claw_width_um": _um(dq["claw_width"][0]),
            "claw_gap_um": _um(dq["claw_gap"][0]),
            "ground_spacing_um": _um(dq["ground_spacing"][0]),
            "Lj_nH": float(Lj_nH),
        },
        "mask_info": {
            "mask_no_jj": files["no_jj"],
            "mask_with_jj": files["with_jj"],
            "preview_png": files["preview"],
            "specs_no_jj": [[s[0], s[1]] for s in M.specs_no_jj],
            "specs_with_jj": [[s[0], s[1]] for s in M.specs_with_jj],
            "n_cellx": int(nx_f),
            "n_celly": int(ny_f),
            "n_cellz": 1,
            "resolution_um": M.resolution_um,
            "prob_hix_m": nx_f * M.resolution_um * 1e-6,
            "prob_hiy_m": ny_f * M.resolution_um * 1e-6,
        },
        "junction_location": {
            "start": {
                "ix": jj_ix_min, "iy": jj_iy_min,
                "x_um": float(jj_ix_min * M.resolution_um),
                "y_um": float(jj_iy_min * M.resolution_um),
            },
            "end": {
                "ix": jj_ix_max, "iy": jj_iy_max,
                "x_um": float(jj_ix_max * M.resolution_um),
                "y_um": float(jj_iy_max * M.resolution_um),
            },
            "center": {
                "ix": jj_ix_center, "iy": jj_iy_center,
                "x_um": float(jj_ix_center * M.resolution_um),
                "y_um": float(jj_iy_center * M.resolution_um),
            },
        },
        "mask_feature_locations": spec_info,
        "cpw_source_gaps": cpw_source_gaps,
        "expected_qubit_freq_GHz": float(expected_f_GHz),
        "note": "expected_qubit_freq_GHz assumes C=80fF; actual C depends on geometry",
    }
    with open(files["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    return MaskResult(mask_no_jj=m_no, mask_with_jj=m_jj, meta=meta, files=files)


# ──────────────────────────────────────────────
#  Sanity checks
# ──────────────────────────────────────────────

def sanity_check(result: MaskResult, cfg: Config) -> List[str]:
    issues: List[str] = []
    S = cfg.sanity
    m_no = result.mask_no_jj
    m_jj = result.mask_with_jj
    meta = result.meta
    jj = meta["junction_location"]
    nx, ny = m_no.shape

    # JJ pixels in bridge match metafile center
    bridge = m_jj.astype(np.int64) - m_no.astype(np.int64)
    locs = np.where(bridge > 0)
    if len(locs[0]) == 0:
        issues.append("JJ bridge has zero pixels in with_jj mask")
    else:
        cx = (int(locs[0].min()) + int(locs[0].max())) // 2
        cy = (int(locs[1].min()) + int(locs[1].max())) // 2
        if (abs(cx - jj["center"]["ix"]) > S.jj_pos_tol_cells
                or abs(cy - jj["center"]["iy"]) > S.jj_pos_tol_cells):
            issues.append(
                f"JJ center mismatch: bridge at ({cx},{cy}) "
                f"vs metafile ({jj['center']['ix']},{jj['center']['iy']})"
            )

    # CPW reaches top of mask
    cpw_key = f"layer_{cfg.mask.specs_with_jj[2][0]}_dt_{cfg.mask.specs_with_jj[2][1]}"
    cpw_info = meta["mask_feature_locations"].get(cpw_key)
    if cpw_info is None:
        issues.append(f"CPW layer {cpw_key} absent from mask")
    elif cpw_info["iy_range"][1] < ny - S.cpw_top_gap_cells:
        issues.append(
            f"CPW does not reach top: y_max={cpw_info['iy_range'][1]} "
            f"(expected within {S.cpw_top_gap_cells} of ny={ny})"
        )

    # Cross y-symmetry around JJ center y
    jc_y = jj["center"]["iy"]
    half = 100
    above = int(m_no[:, jc_y:min(jc_y + half, ny)].astype(np.int64).sum())
    below = int(m_no[:, max(0, jc_y - half):jc_y].astype(np.int64).sum())
    if above + below > 0:
        asym = abs(above - below) / (above + below)
        if asym > S.asymmetry_threshold:
            issues.append(
                f"Cross y-asymmetry around JJ: {asym:.2f} "
                f"(above={above}, below={below})"
            )

    # Metal fraction sanity
    frac = float(m_no.sum() / m_no.size)
    if frac < S.metal_fraction_min or frac > S.metal_fraction_max:
        issues.append(
            f"Metal fraction unusual: {frac:.3f} "
            f"(expected {S.metal_fraction_min}-{S.metal_fraction_max})"
        )
    return issues


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _bbox(poly_dict, specs):
    all_polys = []
    for s in specs:
        if s in poly_dict:
            all_polys.extend(poly_dict[s])
    coords = np.vstack(all_polys)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    scale = 1000.0 if (x_range < 100 and y_range < 100) else 1.0
    return x_min, y_min, x_range, y_range, scale


def _rasterize(poly_dict, specs, x_min, x_range, y_min, y_range, nx, ny):
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


def _carve_jj_gap(m_jj, ix0, ix1, iy0, iy1, gap_cells):
    """Normalize the JJ region to two solid electrodes separated by a clean
    `gap_cells`-wide air gap (the lumped-Lj location).

    The physical Dolan gap is sub-cell (~0.14um) and rasterizes as a ragged
    near-short, so we (1) fill the JJ footprint solid to give two clean
    electrodes, then (2) cut a single slit across the footprint's *long* axis.
    Returns a new array; input is not modified.
    """
    m = m_jj.copy()
    m[ix0:ix1 + 1, iy0:iy1 + 1] = 1          # solidify bridge footprint
    x_ext = ix1 - ix0 + 1
    y_ext = iy1 - iy0 + 1
    if x_ext >= y_ext:                       # leads split along x -> vertical slit
        cx = (ix0 + ix1) // 2
        lo = cx - gap_cells // 2
        m[lo:lo + gap_cells, iy0:iy1 + 1] = 0
    else:                                    # leads split along y -> horizontal slit
        cy = (iy0 + iy1) // 2
        lo = cy - gap_cells // 2
        m[ix0:ix1 + 1, lo:lo + gap_cells] = 0
    return m


def _autocrop_bounds(m_jj, margin):
    nx, ny = m_jj.shape
    rv = np.array([m_jj[i, :].var() for i in range(nx)])
    ax = np.where(rv > 0.001)[0]
    xs = max(0, ax[0] - margin) if len(ax) else 0
    xe = min(nx, ax[-1] + margin) if len(ax) else nx
    cv = np.array([m_jj[:, j].var() for j in range(ny)])
    ay = np.where(cv > 0.001)[0]
    ys = max(0, ay[0] - margin) if len(ay) else 0
    ye = min(ny, ay[-1] + margin) if len(ay) else ny
    return xs, xe, ys, ye


def _per_spec_info(poly_dict, specs, x_min, x_range, y_min, y_range,
                   nx, ny, xs, xe, ys, ye, pad_cells, target_nx, res_um):
    info = {}
    for s in specs:
        if s not in poly_dict:
            continue
        sm = _rasterize(poly_dict, [s],
                        x_min, x_range, y_min, y_range, nx, ny).T
        sm = sm[xs:xe, ys:ye]
        if pad_cells > 0 and sm.shape[1] > pad_cells:
            sm = sm[:, :-pad_cells]
        if target_nx and sm.shape[0] > target_nx:
            trim = (sm.shape[0] - target_nx) // 2
            sm = sm[trim:trim + target_nx, :]
        sl = np.where(sm > 0)
        if len(sl[0]) > 0:
            info[f"layer_{s[0]}_dt_{s[1]}"] = {
                "ix_range": [int(sl[0].min()), int(sl[0].max())],
                "iy_range": [int(sl[1].min()), int(sl[1].max())],
                "x_range_um": [float(sl[0].min() * res_um),
                               float(sl[0].max() * res_um)],
                "y_range_um": [float(sl[1].min() * res_um),
                               float(sl[1].max() * res_um)],
                "pixel_count": int(sm.sum()),
            }
    return info


def _write_preview(m_no, m_jj, jj_cx, jj_cy, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(m_no.T, cmap="viridis", origin="lower", aspect="auto")
    axes[0].set_title("Without JJ bridge (lumped inductor)")
    axes[1].imshow(m_jj.T, cmap="viridis", origin="lower", aspect="auto")
    axes[1].set_title("With JJ electrodes + 1-cell gap (layers 20/60)")
    for ax in axes:
        ax.plot(jj_cx, jj_cy, "r+", markersize=15, markeredgewidth=2)
        ax.annotate(f"JJ ({jj_cx},{jj_cy})", (jj_cx, jj_cy),
                    color="red", fontsize=10,
                    xytext=(10, 10), textcoords="offset points")
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _um(s: str) -> float:
    """Parse a qiskit-metal-style '200um' / '0.2mm' / '5.1um' string to a float
    in micrometers. Returns 0.0 if the value isn't parseable (e.g. 'unknown',
    used by the from-GDS standalone path where geometry isn't known)."""
    try:
        return float(str(s).replace("um", "").replace("mm", ""))
    except ValueError:
        return 0.0


def _detect_cpw_gaps(m_no, nx, ny, spec_info, res_um):
    """Scan the mask near the top of the CPW to find the two air-gap bounds
    that flank the center conductor. Returns a dict with gap1/gap2 bounds in
    um, ready for use in Artemis source-x placement. None if not detected.

    The CPW (layer 1, datatype 11) is expected to span a y range; we sample a
    row in the upper half of that range (representative of where the source
    would sit). We then find the two air-gap runs nearest the mask x center.
    """
    cpw = spec_info.get("layer_1_dt_11")
    if cpw is None:
        return None
    iy_lo, iy_hi = cpw["iy_range"]
    sample_y = (iy_lo + iy_hi) // 2  # middle of CPW
    if sample_y < 0 or sample_y >= ny:
        return None

    row = m_no[:, sample_y].astype(np.int64)
    cx = nx // 2

    # Find runs of zeros (air) within ±100 cells of mask center
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
            runs.append((start, i))  # half-open [start, i)
            in_air = False
    if in_air:
        runs.append((start, hi))

    if len(runs) < 2:
        return None
    # Sort by distance to center, take the two closest
    runs.sort(key=lambda r: abs((r[0] + r[1]) / 2 - cx))
    g1, g2 = sorted(runs[:2], key=lambda r: r[0])

    return {
        "sample_y_um": float(sample_y * res_um),
        "gap1_lo_um": float(g1[0] * res_um),
        "gap1_hi_um": float(g1[1] * res_um),
        "gap2_lo_um": float(g2[0] * res_um),
        "gap2_hi_um": float(g2[1] * res_um),
    }
