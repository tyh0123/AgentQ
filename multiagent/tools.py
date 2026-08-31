"""
Deterministic tool layer (the "MCP-tool" analog).

Stateless, function-like operations that accept structured inputs and return
structured outputs *without* reasoning. They drive the project's **real** design
pipeline end to end — the same code the `qubit-pipeline` MCP server exposes:

    resolve_geometry   SQuADDS DB query  → local inverse-ML fallback   (qpipe_squadds / mlp_inverse)
    build_gds          qiskit-metal layout → GDS                        (qpipe_layout + qpipe_gds)
    rasterize_mask     GDS → npy mask (no_jj / with_jj) + preview PNG    (qpipe_mask)
    write_artemis      metafile + mask → Artemis FDTD input             (qpipe_artemis)
    write_slurm        Artemis input → Perlmutter sbatch                (qpipe_slurm)

Every artifact this system emits — GDS, npy masks, preview PNG, Artemis input,
sbatch — is the genuine file, locally testable before anything reaches the HPC.

The design STOPS at a validated Artemis input + sbatch. Running on the cluster,
collecting plotfiles, and post-processing (Decision Point 3) are a downstream
agent's job — see README.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Dict, Optional

import numpy as np

import config
import mlp_inverse
from blackboard import DesignRecord

# Real pipeline config loader (light; heavy modules imported lazily below).
from qpipe_config import Config


# SQuADDS prints a few harmless-but-alarming lines to stdout during a query
# (a per-row parse warning, a timing line, the HF unauth notice). A successful
# query still prints `source: squadds_db (...)`, which we keep.
_SQUADDS_NOISE = ("Error processing row", "Time taken to add the coupled H params",
                  "You are sending unauthenticated")


class _LineFilter(io.TextIOBase):
    def __init__(self, real):
        self._real, self._buf = real, ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(n in line for n in _SQUADDS_NOISE):
                self._real.write(line + "\n")
        return len(s)

    def flush(self):
        self._real.flush()


@contextlib.contextmanager
def _quiet_squadds():
    """Drop the known SQuADDS noise lines on stdout AND stderr (the HF
    unauthenticated notice goes to stderr). Unless QMETAL_VERBOSE."""
    if os.environ.get("QMETAL_VERBOSE"):
        yield
        return
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _LineFilter(old_out), _LineFilter(old_err)
    try:
        yield
    finally:
        for filt, real in ((sys.stdout, old_out), (sys.stderr, old_err)):
            if getattr(filt, "_buf", ""):          # flush any trailing partial line
                real.write(filt._buf)
        sys.stdout, sys.stderr = old_out, old_err


# ────────────────────────────────────────────────────────────
#  cwd / config helpers
# ────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _in_out_dir():
    """Run a block with cwd == OUT_DIR so the real pipeline's relative-path
    writes (GDS, npy, png, input, sbatch) all land in out/."""
    os.makedirs(config.OUT_DIR, exist_ok=True)
    prev = os.getcwd()
    os.chdir(config.OUT_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


def cfg_for(record: DesignRecord, extra: dict = None) -> Config:
    """Load defaults.yaml with this record's target injected — every qpipe
    stage reads cfg.target, so the target must flow in through config.
    `extra` adds dotted-path overrides (e.g. slurm.nodes from an HPC fix)."""
    ov = {
        "target.qubit_frequency_GHz": record.freq_GHz,
        "target.anharmonicity_MHz":   record.anharm_MHz,
    }
    if record.cavity_GHz is not None:
        ov["target.cavity_frequency_GHz"] = record.cavity_GHz
    if record.g_MHz is not None:
        ov["target.g_MHz"] = record.g_MHz
    if extra:
        ov.update(extra)
    return Config.load(overrides=ov)


# ────────────────────────────────────────────────────────────
#  Tool 1 — geometry: SQuADDS DB query, local-ML fallback
# ────────────────────────────────────────────────────────────
def resolve_geometry(cfg: Config, source: str) -> Dict[str, object]:
    """Return (dq, dk, Lj_nH, source_used) shaped exactly like the real
    qpipe_squadds.query output, so the downstream layout is source-agnostic.

    source:
      "db"   — real SQuADDS DB (with its own remote-ML fallback on a far match)
      "ml"   — local offline inverse model (reproducible; no network)
      "auto" — try the DB, fall back to the local model on any failure
    """
    if source == "ml":
        return _local_ml_geometry(cfg)

    if source == "db":
        dq, dk, Lj = _squadds_db_geometry(cfg)
        return {"dq": dq, "dk": dk, "Lj_nH": float(Lj), "source_used": "squadds_db"}

    # auto: DB first, local ML on any failure (offline, DB miss, import error).
    try:
        dq, dk, Lj = _squadds_db_geometry(cfg)
        return {"dq": dq, "dk": dk, "Lj_nH": float(Lj), "source_used": "squadds_db"}
    except Exception as e:                                  # noqa: BLE001
        print(f"  [tools] SQuADDS DB path unavailable ({type(e).__name__}: {e}); "
              f"falling back to the local inverse model")
        return _local_ml_geometry(cfg)


def _squadds_db_geometry(cfg: Config):
    """Real SQuADDS query (heavy import kept local to this path)."""
    import qpipe_squadds
    with _quiet_squadds():
        return qpipe_squadds.query(cfg)


def _local_ml_geometry(cfg: Config) -> Dict[str, object]:
    """Local trained inverse model → dq/dk shaped like the DB output.

    The model predicts cross_length / claw_length / ground_spacing; the other
    four qubit-geometry fields + the CPW trace come from cfg fallbacks (same
    contract the remote ML fallback uses). L_J is the analytic transmon value.
    """
    geom = mlp_inverse.predict(cfg.target.qubit_frequency_GHz,
                               cfg.target.anharmonicity_MHz)
    fb = cfg.squadds.fallback_qubit_geometry
    fbc = cfg.squadds.fallback_cpw
    dq = {
        "cross_length":   [f"{geom['cross_length_um']:.3f}um"],
        "claw_length":    [f"{geom['claw_length_um']:.3f}um"],
        "ground_spacing": [f"{geom['ground_spacing_um']:.3f}um"],
        "cross_width":    [fb.cross_width],
        "cross_gap":      [fb.cross_gap],
        "claw_width":     [fb.claw_width],
        "claw_gap":       [fb.claw_gap],
    }
    dk = {"second_width": [fbc.second_width], "second_gap": [fbc.second_gap]}
    return {"dq": dq, "dk": dk, "Lj_nH": float(geom["Lj_nH"]),
            "source_used": "local_inverse_ml"}


# ────────────────────────────────────────────────────────────
#  Tool 1b — SQuADDS query WITH resonator (dc), for full / multi
# ────────────────────────────────────────────────────────────
def query_full(cfg: Config, targets: list, coupler_overrides: dict = None) -> list:
    """Query SQuADDS for full geometry including the readout resonator (dc).

    `targets` is a list of target dicts (qubit_frequency_GHz, anharmonicity_MHz,
    cavity_frequency_GHz, g_MHz) — one for `full`, N for `multi`. Returns a list
    of {dq, dc, dk, Lj_nH}. Requires the DB: the resonator meander length has no
    offline ML surrogate.

    `coupler_overrides` (optional) sets the feedline↔resonator coupler geometry
    the DB does not provide: coupling_space (gap), coupling_length (overlap),
    down_length — each an "<x>um" string. Unset keys keep the defaults.
    """
    from squadds import Analyzer, SQuADDS_DB
    from squadds.core.utils import convert_numpy
    sq = cfg.squadds
    with _quiet_squadds():
        db = SQuADDS_DB()
        db.select_system(["cavity_claw", "qubit"])
        db.select_qubit(sq.qubit_type)
        db.select_cavity_claw(sq.cavity_type)
        db.select_resonator_type(sq.resonator_type)
        db.create_system_df()

    # Walk more than the validation pool for a row with usable qubit +
    # resonator fields. NOTE: this DB's cavity_claw/RouteMeander/quarter rows
    # carry NO feedline-coupler geometry (coupler_options are all null), so the
    # coupler + feedline are filled from defaults — see _fill_coupler_defaults.
    num_top = max(sq.num_top, 60)

    out = []
    for tgt in targets:
        az = Analyzer(db)
        with _quiet_squadds():
            results = az.find_closest(tgt, num_top=num_top, metric=sq.metric)
            geo = _pick_usable_full_row(az, results, sq, coupler_overrides)
        if geo is None:
            raise RuntimeError(
                f"no SQuADDS row within top-{num_top} of {tgt} has complete "
                f"qubit + resonator geometry")
        out.append(geo)
    return out


# Qubit + resonator fields the DB actually provides for these rows.
_REQ_RESONATOR = ("total_length", "trace_width", "trace_gap")
_REQ_QUBIT = ("cross_length", "cross_width", "cross_gap",
              "claw_length", "claw_width", "claw_gap", "ground_spacing")

# The DB gives no feedline coupler for these rows; a sensible readout CLT.
_DEFAULT_COUPLER = {"coupling_length": "200um", "coupling_space": "5um",
                    "down_length": "100um"}


def _fill_coupler_defaults(dc: dict, sq, overrides: dict = None) -> dict:
    """The DB's coupler_options are null for these rows. Build a usable
    CoupledLineTee: geometry defaults + trace widths tied to the resonator
    (second_*) and the feedline (prime_* from the CPW fallback).

    `overrides` may set coupling_space (the resonator↔feedline gap),
    coupling_length (the overlap), and down_length — the user-facing knobs.
    """
    out = dict(_DEFAULT_COUPLER)
    for k in ("coupling_space", "coupling_length", "down_length"):
        if overrides and overrides.get(k) is not None:
            out[k] = overrides[k]
    out["second_width"] = dc["trace_width"][0]      # resonator side
    out["second_gap"] = dc["trace_gap"][0]
    out["prime_width"] = sq.fallback_cpw.second_width   # feedline side
    out["prime_gap"] = sq.fallback_cpw.second_gap
    return {k: [v] for k, v in out.items()}


def _pick_usable_full_row(az, results, sq, coupler_overrides=None):
    """Return the first candidate with populated qubit + resonator geometry
    (shaped {dq, dc, dk, Lj_nH}), the coupler filled from defaults/overrides.
    None if none qualify."""
    from squadds.core.utils import convert_numpy

    def _ok(d, keys):
        return all(d.get(k) and d[k][0] is not None for k in keys)

    for i in range(len(results)):
        one = results.iloc[[i]]
        try:
            dq = convert_numpy(az.get_qubit_options(one))
            dc = convert_numpy(az.get_cpw_options(one))
            Lj = convert_numpy(az.get_Ljs(one))
        except Exception:                                    # noqa: BLE001
            continue
        if _ok(dq, _REQ_QUBIT) and _ok(dc, _REQ_RESONATOR):
            dk = _fill_coupler_defaults(dc, sq, coupler_overrides)
            return {"dq": dq, "dc": dc, "dk": dk, "Lj_nH": float(Lj[0])}
    return None


# ────────────────────────────────────────────────────────────
#  Tool 2 — qiskit-metal layout → GDS  (real)
# ────────────────────────────────────────────────────────────
def build_gds(cfg: Config, dq: dict, dk: dict, Lj_nH: float, prefix: str) -> str:
    """Build the validation layout (qubit + CPW feedline + JJ bridge) → GDS.
    Returns the absolute GDS path in OUT_DIR."""
    import qpipe_layout
    import qpipe_gds
    gds_name = f"{prefix}.gds"
    with _in_out_dir():
        design = qpipe_layout.build(cfg, dq, dk, Lj_nH)
        qpipe_gds.export(design, gds_name)
    return os.path.join(config.OUT_DIR, gds_name)


def build_gds_full(cfg: Config, geo: dict, prefix: str) -> str:
    """Build the full layout (qubit + coupler + meander resonator + feedline +
    JJ) → GDS. Returns the absolute GDS path in OUT_DIR."""
    import layouts
    import qpipe_gds
    gds_name = f"{prefix}.gds"
    with _in_out_dir():
        design = layouts.build_full(cfg, geo["dq"], geo["dc"], geo["dk"], geo["Lj_nH"])
        qpipe_gds.export(design, gds_name)
    return os.path.join(config.OUT_DIR, gds_name)


def build_gds_multi(cfg: Config, geos: list, prefix: str,
                    coupler_spacing_um: float = 2500.0):
    """Build N unit cells on a shared feedline → GDS. Returns (gds_path, layout_info)."""
    import layouts
    import qpipe_gds
    gds_name = f"{prefix}.gds"
    with _in_out_dir():
        design, info = layouts.build_multi(cfg, geos, coupler_spacing_um)
        qpipe_gds.export(design, gds_name)
    return os.path.join(config.OUT_DIR, gds_name), info


# The four layers a clean fab-style GDS keeps: ground plane, etch, CPW, JJ bridge.
CLEAN_GDS_SPECS = frozenset({(1, 0), (1, 10), (1, 11), (20, 10)})


def clean_gds_layers(gds_path: str, keep=CLEAN_GDS_SPECS) -> dict:
    """Rewrite `gds_path` keeping only the requested (layer, datatype) specs —
    strips cheese holes (1,101), marker frames (1,99/1,100/1,102) and the JJ
    window (60,10). Paths are flattened to polygons. Returns per-spec counts.

    Call AFTER rasterization so the mask is built from the full geometry; only
    the on-disk GDS artifact is trimmed.
    """
    import gdspy
    src = gdspy.GdsLibrary()
    src.read_gds(gds_path)
    top = src.top_level()[0]
    by_spec = top.get_polygons(by_spec=True)

    out = gdspy.GdsLibrary()
    cell = out.new_cell(top.name)
    kept = {}
    for (layer, dt), polys in by_spec.items():
        if (layer, dt) not in keep:
            continue
        for poly in polys:
            cell.add(gdspy.Polygon(poly, layer=layer, datatype=dt))
        kept[(layer, dt)] = len(polys)
    out.write_gds(gds_path)
    return kept


def rasterize_multi(cfg: Config, gds_file: str, prefix: str,
                    layout_info: dict) -> Dict[str, object]:
    """Rasterize a multi-qubit GDS → npy ground-plane mask + preview PNG, and
    locate each qubit's JJ from the (20,10) bridge pixels (one consistent
    transform for the mask and the JJ detection). Returns paths + per-qubit JJ
    list (cell→um) for the lumped inductors."""
    import gdspy
    import qpipe_mask
    M = cfg.mask
    struct = [(1, 0), (1, 10), (1, 11)]
    res = M.resolution_um

    with _in_out_dir():
        lib = gdspy.GdsLibrary()
        lib.read_gds(os.path.basename(gds_file))
        poly = lib.top_level()[0].get_polygons(by_spec=True)

        x_min, y_min, x_range, y_range, scale = qpipe_mask._bbox(poly, struct)
        nx = round(x_range * scale / res)
        ny = round(y_range * scale / res)
        m_struct = qpipe_mask._rasterize(poly, struct, x_min, x_range, y_min, y_range, nx, ny).T
        m_jj = qpipe_mask._rasterize(poly, [(20, 10)], x_min, x_range, y_min, y_range, nx, ny).T

        xs, xe, ys, ye = qpipe_mask._autocrop_bounds(m_struct, int(M.trim_margin_um / res))
        m_struct, m_jj = m_struct[xs:xe, ys:ye], m_jj[xs:xe, ys:ye]
        nx_f, ny_f = m_struct.shape

        # feedline cell y (same transform), so the source detector knows where
        # the feedline is now that qubits sit on both sides of it.
        feed_raw = layout_info["feedline_y_um"] / scale
        feedline_y_cell = int(round((feed_raw - y_min) / y_range * (ny - 1))) - ys
        feedline_y_cell = max(0, min(ny_f - 1, feedline_y_cell))

        npy_name = f"{prefix}_mask.npy"
        np.save(npy_name, np.ascontiguousarray(m_struct[:, :, np.newaxis].astype(np.float64)))
        jj_list = _cluster_jjs(m_jj, res, layout_info)
        preview = f"{prefix}_mask_preview.png"
        _multi_preview(m_struct, jj_list, res, preview)

    return {
        "mask_npy":       os.path.join(config.OUT_DIR, npy_name),
        "preview_png":    os.path.join(config.OUT_DIR, preview),
        "n_cellx":        int(nx_f),
        "n_celly":        int(ny_f),
        "metal_fraction": float((m_struct > 0).mean()),
        "jj_list":        jj_list,
        "feedline_y_cell": int(feedline_y_cell),
    }


def _cluster_jjs(m_jj, res_um, layout_info):
    """Split the JJ-bridge pixels into per-qubit clusters by x, returning each
    as {x_lo_um, x_hi_um, cy_um, Lj_nH}. Lj comes from layout_info order (left→right)."""
    xs_idx, ys_idx = np.where(m_jj > 0)
    ljs = [q["Lj_nH"] for q in sorted(layout_info["qubit_positions"], key=lambda q: q["x_um"])]
    if len(xs_idx) == 0:
        return []
    order = np.argsort(xs_idx)
    xs_sorted, ys_sorted = xs_idx[order], ys_idx[order]
    # break into clusters wherever consecutive x jump by > 100 cells
    splits = np.where(np.diff(xs_sorted) > 100)[0] + 1
    groups = np.split(np.arange(len(xs_sorted)), splits)
    out = []
    for i, g in enumerate(groups):
        gx, gy = xs_sorted[g], ys_sorted[g]
        lj = ljs[i] if i < len(ljs) else ljs[-1]
        out.append({"x_lo_um": float(gx.min() * res_um), "x_hi_um": float(gx.max() * res_um),
                    "cy_um": float(gy.mean() * res_um), "Lj_nH": float(lj)})
    return out


def _multi_preview(m_struct, jj_list, res_um, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.imshow(m_struct.T, cmap="viridis", origin="lower", aspect="auto")
    for i, jj in enumerate(jj_list):
        ax.plot(jj["x_lo_um"] / res_um, jj["cy_um"] / res_um, "r+", markersize=14, markeredgewidth=2)
        ax.annotate(f"Q{i+1} (Lj {jj['Lj_nH']:.1f})", (jj["x_lo_um"] / res_um, jj["cy_um"] / res_um),
                    color="red", fontsize=9, xytext=(8, 8), textcoords="offset points")
    ax.set_title(f"Multi-qubit ground-plane mask ({m_struct.shape[0]}x{m_struct.shape[1]}), {len(jj_list)} JJ")
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()


# ────────────────────────────────────────────────────────────
#  Tool 3 — GDS → npy masks + preview PNG + metafile  (real)
# ────────────────────────────────────────────────────────────
def rasterize_mask(cfg: Config, gds_file: str, prefix: str,
                   dq: dict, Lj_nH: float) -> Dict[str, object]:
    """Rasterize the GDS into the with/without-JJ npy masks, a preview PNG, and
    a metafile compatible with generate_artemis_input. Also runs the project's
    real sanity checks. Returns paths + the metrics the critics inspect."""
    import qpipe_mask
    with _in_out_dir():
        res = qpipe_mask.rasterize(os.path.basename(gds_file), prefix, cfg, dq, Lj_nH)
        issues = qpipe_mask.sanity_check(res, cfg)

    def _abs(name):
        return os.path.join(config.OUT_DIR, name)

    mi = res.meta["mask_info"]
    return {
        "metafile":       _abs(res.files["meta"]),
        "mask_no_jj":     _abs(res.files["no_jj"]),
        "mask_with_jj":   _abs(res.files["with_jj"]),
        "preview_png":    _abs(res.files["preview"]),
        "expected_freq_GHz": float(res.meta["expected_qubit_freq_GHz"]),
        "metal_fraction": float((res.mask_no_jj > 0).mean()),
        "n_celly":        int(mi["n_celly"]),
        "n_cellx":        int(mi["n_cellx"]),
        "sanity_issues":  list(issues),
    }


# ────────────────────────────────────────────────────────────
#  Tool 4 — Artemis FDTD input  (real qpipe_artemis)
# ────────────────────────────────────────────────────────────
def write_artemis(cfg: Config, metafile: str, mask_npy: str, prefix: str,
                  freq_GHz: float, source_y_um: Optional[float] = None) -> Dict[str, object]:
    """Render a real Artemis input file. `source_y_um=None` auto-places it from
    the CPW; a value forces it (used by the fault demo). Written into OUT_DIR."""
    import json
    import qpipe_artemis
    with open(metafile) as f:
        meta = json.load(f)
    with _in_out_dir():
        art = qpipe_artemis.write_input(
            meta, os.path.basename(mask_npy), f"input_{prefix}", cfg,
            freq_GHz=freq_GHz, source_y_um=source_y_um, use_inductor=True,
        )
    art["output_file"] = os.path.join(config.OUT_DIR, os.path.basename(art["output_file"]))
    return art


# ────────────────────────────────────────────────────────────
#  Tool 4b — Artemis input for horizontal feedline (qubit_resonator / multi)
# ────────────────────────────────────────────────────────────
def write_artemis_horizontal(cfg: Config, mask_info: dict, mask_npy: str,
                             prefix: str, jj_list: list, freq_GHz: float,
                             feedline_y_cell: int = None) -> Dict[str, object]:
    """Detect the feedline y-gaps + input-port source x, then write an Ey-drive
    Artemis input with one lumped inductor per qubit JJ. `feedline_y_cell` locates
    the feedline when qubits sit on both sides of it. Into OUT_DIR."""
    import artemis_full
    with _in_out_dir():
        mask = np.load(os.path.basename(mask_npy))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        source_x, gaps = artemis_full.pick_source_x(mask, cfg, feedline_y_cell)
        if gaps is None:
            raise RuntimeError("could not detect the feedline gaps for the source")
        art = artemis_full.write_input(mask_info, os.path.basename(mask_npy),
                                       f"input_{prefix}", cfg, jj_list,
                                       source_x, gaps, freq_GHz)
    art["output_file"] = os.path.join(config.OUT_DIR, os.path.basename(art["output_file"]))
    return art


# ────────────────────────────────────────────────────────────
#  Tool 4c — Design Rule Check on the GDS  (real qpipe_drc / KLayout)
# ────────────────────────────────────────────────────────────
def run_drc(cfg: Config, gds_file: str) -> Dict[str, object]:
    """Run the project's KLayout DRC (min width / spacing / gap / area, JJ
    width) on the exported GDS. Returns a structured report; degrades to a
    'skipped' report if klayout is unavailable."""
    try:
        import qpipe_drc
    except ImportError as e:                                 # pragma: no cover
        return {"skipped": True, "reason": f"klayout unavailable: {e}"}
    with _in_out_dir():
        try:
            res = qpipe_drc.run_drc(os.path.basename(gds_file), cfg)
        except ImportError as e:
            return {"skipped": True, "reason": f"klayout unavailable: {e}"}
    rep = qpipe_drc.to_report_dict(res)
    rep["skipped"] = False
    return rep


# ────────────────────────────────────────────────────────────
#  Tool 5 — SLURM submission script  (real qpipe_slurm)
# ────────────────────────────────────────────────────────────
def write_slurm(cfg: Config, prefix: str, artemis_input: str) -> str:
    """Generate the Perlmutter sbatch script. Written into OUT_DIR."""
    import qpipe_slurm
    with _in_out_dir():
        out = qpipe_slurm.write_slurm(prefix, os.path.basename(artemis_input), cfg)
    return os.path.join(config.OUT_DIR, out)
