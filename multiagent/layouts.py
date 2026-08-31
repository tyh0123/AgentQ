"""
Layout builders beyond the single-qubit validation cell.

`qpipe_layout.build` (reused by the validation path) makes a single qubit + a
straight CPW feedline. The richer topologies the user's project already prototypes
as standalone scripts are refactored here into importable builders, so the
multi-agent designer can drive them and the critics can gate their artifacts:

    build_full   — one qubit + claw + CoupledLineTee coupler + RouteMeander
                   readout resonator + launchpad feedline + JJ bridge.
                   (from squadds_qmetal_test_together.py, with the JJ that the
                    standalone script imports but never places — needed for the
                    lumped-inductor mask.)
    build_multi  — N such unit cells sharing one feedline.
                   (from multi_qubit_pipeline.py)

Geometry (dq / dc / dk / Lj) comes from SQuADDS — see tools.resolve_geometry_full.
The resonator meander length (dc.total_length) has no offline ML surrogate, so
these modes require the DB.

GDS export reuses qpipe_gds.export; mask rasterization reuses qpipe_mask.rasterize
(the JJ bridge makes the single-qubit detection path apply to each qubit).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import geopandas as gpd

import qiskit_metal as metal
from qiskit_metal import Dict as MDict
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight

pd.DataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)
gpd.GeoDataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)

from squadds.components.jjs import JjDolan

from qpipe_config import Config


# ──────────────────────────────────────────────────────────────
#  JJ placement — south arm of a TransmonCross (shared by full/multi)
# ──────────────────────────────────────────────────────────────
def _place_jj(design, name, dq, cx_um, cy_um, orientation_deg, J):
    """Add a JjDolan bridge at the qubit's south cross arm, matching the
    multi_qubit_pipeline placement (orientation-rotated offset)."""
    cross_length_mm = float(str(dq["cross_length"][0]).replace("um", "")) / 1000
    cross_gap_mm = float(str(dq["cross_gap"][0]).replace("um", "")) / 1000
    jjy = -cross_gap_mm / 2 - cross_length_mm
    jjxprime = -np.sin(np.radians(orientation_deg)) * jjy
    jjyprime = np.cos(np.radians(orientation_deg)) * jjy
    JjDolan(design, name, options=MDict(
        pos_x=f"{cx_um}um", pos_y=f"{cy_um}um",
        orientation=str(orientation_deg),
        bridge_length=f"{(cross_gap_mm - 2 * J.jj_pad_mm) * 1000}um",
        JJ_width=J.jj_width, bridge_layer=J.bridge_layer, JJ_layer=J.jj_layer,
    ))
    jj = design.components[name]
    jj.options.pos_x = f"{cx_um / 1000 + jjxprime}mm"
    jj.options.pos_y = f"{cy_um / 1000 + jjyprime}mm"


def _unit_cell(design, idx, dq, dc, dk, Lj_nH, cx_um, feed_y_um, qubit_y_um,
               qubit_ori, coupler_ori, L, J):
    """One qubit + coupler + meander resonator unit cell (shared by full/multi).
    `qubit_ori` / `coupler_ori` place the cell below ("-90"/"0") or above
    ("90"/"180") the feedline. `idx` labels the components."""
    CoupledLineTee(design, f"cplr_{idx}", options=MDict(
        coupling_length=dk["coupling_length"][0], coupling_space=dk["coupling_space"][0],
        down_length=dk["down_length"][0], open_termination=False, orientation=coupler_ori,
        prime_gap=dk["prime_gap"][0], prime_width=dk["prime_width"][0],
        second_gap=dk["second_gap"][0], second_width=dk["second_width"][0],
        pos_x=f"{cx_um}um", pos_y=f"{feed_y_um}um", layer=str(L.metal_layer),
    ))

    TransmonCross(design, f"qubit_{idx}", options=MDict(
        hfss_inductance=str(Lj_nH * 1e-9), q3d_inductance=str(Lj_nH * 1e-9),
        chip="main",
        connection_pads=MDict(readout=MDict(
            claw_cpw_length=L.claw_cpw_length, claw_cpw_width=L.claw_cpw_width,
            claw_gap=dq["claw_gap"][0], claw_length=dq["claw_length"][0],
            claw_width=dq["claw_width"][0], connector_location="0",
            connector_type="0", ground_spacing=dq["ground_spacing"][0],
        )),
        cross_gap=dq["cross_gap"][0], cross_length=dq["cross_length"][0],
        cross_width=dq["cross_width"][0], layer=str(L.metal_layer),
        orientation=qubit_ori, pos_x=f"{cx_um}um", pos_y=f"{qubit_y_um}um",
    ))

    _place_jj(design, f"jj_{idx}", dq, cx_um, qubit_y_um, float(qubit_ori), J)

    RouteMeander(design, f"meander_{idx}", options=MDict(
        fillet="49.9um",
        lead=MDict(end_straight="50um", start_straight="100um"),
        meander=MDict(asymmetry="0um", spacing="100um"),
        pin_inputs=MDict(
            start_pin=MDict(component=f"cplr_{idx}", pin="second_end"),
            end_pin=MDict(component=f"qubit_{idx}", pin="readout"),
        ),
        total_length=dc["total_length"][0], trace_gap=dc["trace_gap"][0],
        trace_width=dc["trace_width"][0], layer=str(L.metal_layer),
    ))


def _finalize(design, L):
    for name, comp in design.components.items():
        if not name.startswith("jj"):
            comp.options.layer = str(L.metal_layer)
    design.rebuild()
    _fit_chip(design)


def _fit_chip(design, margin_um: float = 500.0):
    """Grow the chip so the (1,0) ground plane encloses every structure with a
    margin. The default chip size is fixed and too small once a resonator +
    feedline are present, leaving them outside the ground plane. Chip is centred
    at the origin, so size = 2·(max |coord|) + 2·margin."""
    import pandas as pd

    xs, ys = [], []
    for tbl in design.qgeometry.tables.values():
        if "geometry" not in getattr(tbl, "columns", []):
            continue
        for geom in tbl["geometry"].dropna():
            minx, miny, maxx, maxy = geom.bounds          # mm
            xs += [minx, maxx]
            ys += [miny, maxy]
    if not xs:
        return
    m = margin_um / 1000.0                                # um → mm
    half_x = max(abs(min(xs)), abs(max(xs))) + m
    half_y = max(abs(min(ys)), abs(max(ys))) + m
    design.chips.main.size.size_x = f"{2 * half_x * 1000:.1f}um"
    design.chips.main.size.size_y = f"{2 * half_y * 1000:.1f}um"
    design.rebuild()


# ──────────────────────────────────────────────────────────────
#  full — one qubit + coupler + meander resonator + feedline
# ──────────────────────────────────────────────────────────────
def build_full(cfg: Config, dq: dict, dc: dict, dk: dict, Lj_nH: float):
    """Single readout unit cell on a horizontal launchpad feedline."""
    L, J = cfg.layout, cfg.junction
    feed_y, qubit_y = 800.0, -800.0
    pad_x = 2000.0

    design = metal.designs.design_planar.DesignPlanar()
    design.overwrite_enabled = True
    design._chips.main.size.size_x = f"{L.chip_x_um}um"
    design._chips.main.size.size_y = f"{L.chip_y_um}um"

    LaunchpadWirebond(design, "wb_in", options=MDict(
        pos_x=f"{-pad_x}um", pos_y=f"{feed_y}um", orientation="0",
        pad_width="160um", pad_length="200um", tapper_height="200um",
        trace_width=dk["prime_width"][0], trace_gap=dk["prime_gap"][0], layer=str(L.metal_layer)))
    LaunchpadWirebond(design, "wb_out", options=MDict(
        pos_x=f"{pad_x}um", pos_y=f"{feed_y}um", orientation="180",
        pad_width="160um", pad_length="200um", tapper_height="200um",
        trace_width=dk["prime_width"][0], trace_gap=dk["prime_gap"][0], layer=str(L.metal_layer)))

    _unit_cell(design, 1, dq, dc, dk, Lj_nH, 0.0, feed_y, qubit_y, "-90", "0", L, J)

    RouteStraight(design, "feed_in", options=MDict(
        pin_inputs=MDict(start_pin=MDict(component="wb_in", pin="tie"),
                         end_pin=MDict(component="cplr_1", pin="prime_start")),
        trace_width=dk["prime_width"][0], trace_gap=dk["prime_gap"][0], layer=str(L.metal_layer)))
    RouteStraight(design, "feed_out", options=MDict(
        pin_inputs=MDict(start_pin=MDict(component="cplr_1", pin="prime_end"),
                         end_pin=MDict(component="wb_out", pin="tie")),
        trace_width=dk["prime_width"][0], trace_gap=dk["prime_gap"][0], layer=str(L.metal_layer)))

    _finalize(design, L)
    return design


# ──────────────────────────────────────────────────────────────
#  multi — N unit cells sharing one feedline
# ──────────────────────────────────────────────────────────────
def build_multi(cfg: Config, geos: list, coupler_spacing_um: float = 2500.0):
    """N readout unit cells on one shared launchpad feedline.

    geos: list of per-qubit {dq, dc, dk, Lj_nH}. Returns (design, layout_info).
    """
    L, J = cfg.layout, cfg.junction
    n = len(geos)
    feed_y, qubit_offset = 800.0, 1600.0     # qubits sit ±qubit_offset from the feedline

    total_span = (n - 1) * coupler_spacing_um
    coupler_xs = [int(-total_span / 2 + i * coupler_spacing_um) for i in range(n)]

    design = metal.designs.design_planar.DesignPlanar()
    design.overwrite_enabled = True
    design._chips.main.size.size_x = f"{L.chip_x_um}um"
    design._chips.main.size.size_y = f"{L.chip_y_um}um"

    dk0 = geos[0]["dk"]
    pad_margin = 1500
    lp_l, lp_r = coupler_xs[0] - pad_margin, coupler_xs[-1] + pad_margin
    LaunchpadWirebond(design, "wb_in", options=MDict(
        pos_x=f"{lp_l}um", pos_y=f"{feed_y}um", orientation="0",
        pad_width="160um", pad_length="200um", tapper_height="200um",
        trace_width=dk0["prime_width"][0], trace_gap=dk0["prime_gap"][0], layer=str(L.metal_layer)))
    LaunchpadWirebond(design, "wb_out", options=MDict(
        pos_x=f"{lp_r}um", pos_y=f"{feed_y}um", orientation="180",
        pad_width="160um", pad_length="200um", tapper_height="200um",
        trace_width=dk0["prime_width"][0], trace_gap=dk0["prime_gap"][0], layer=str(L.metal_layer)))

    # Alternate qubits below / above the feedline. A "180" coupler swaps its
    # prime_start/prime_end, so track each coupler's (left, right) feedline pins.
    def _lr(ori):
        return ("prime_start", "prime_end") if ori == "0" else ("prime_end", "prime_start")

    qubit_positions, cori = [], []
    for i, geo in enumerate(geos):
        cx = coupler_xs[i]
        if i % 2 == 0:                                    # below the feedline
            qy, qori, co = feed_y - qubit_offset, "-90", "0"
        else:                                            # above the feedline
            qy, qori, co = feed_y + qubit_offset, "90", "180"
        cori.append(co)
        qubit_positions.append({"x_um": cx, "y_um": qy, "Lj_nH": float(geo["Lj_nH"]),
                                "side": "below" if i % 2 == 0 else "above"})
        _unit_cell(design, i + 1, geo["dq"], geo["dc"], geo["dk"], geo["Lj_nH"],
                   float(cx), feed_y, qy, qori, co, L, J)

    RouteStraight(design, "feed_seg_0", options=MDict(
        pin_inputs=MDict(start_pin=MDict(component="wb_in", pin="tie"),
                         end_pin=MDict(component="cplr_1", pin=_lr(cori[0])[0])),
        trace_width=dk0["prime_width"][0], trace_gap=dk0["prime_gap"][0], layer=str(L.metal_layer)))
    for i in range(n - 1):
        RouteStraight(design, f"feed_seg_{i+1}", options=MDict(
            pin_inputs=MDict(start_pin=MDict(component=f"cplr_{i+1}", pin=_lr(cori[i])[1]),
                             end_pin=MDict(component=f"cplr_{i+2}", pin=_lr(cori[i+1])[0])),
            trace_width=dk0["prime_width"][0], trace_gap=dk0["prime_gap"][0], layer=str(L.metal_layer)))
    RouteStraight(design, f"feed_seg_{n}", options=MDict(
        pin_inputs=MDict(start_pin=MDict(component=f"cplr_{n}", pin=_lr(cori[n-1])[1]),
                         end_pin=MDict(component="wb_out", pin="tie")),
        trace_width=dk0["prime_width"][0], trace_gap=dk0["prime_gap"][0], layer=str(L.metal_layer)))

    _finalize(design, L)
    layout_info = {
        "n_qubits": n, "feedline_y_um": feed_y,
        "coupler_spacing_um": coupler_spacing_um, "coupler_xs_um": coupler_xs,
        "qubit_positions": qubit_positions,
    }
    return design, layout_info
