"""
qiskit-metal layout for a single qubit + straight CPW feedline + JJ bridge.

All placement / geometry comes from cfg.layout and cfg.junction.
SQuADDS-derived dimensions (cross_length, claw_*, ground_spacing, second_width,
second_gap) come from the dq/dk dicts returned by qpipe_squadds.query.
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
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight

pd.DataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)
gpd.GeoDataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)

from squadds.components.jjs import JjDolan

from qpipe_config import Config


def build(cfg: Config, dq: dict, dk: dict, Lj_nH: float):
    """Construct a qiskit-metal DesignPlanar with qubit + CPW + JJ.

    Returns the DesignPlanar object, ready for GDS export.
    """
    L = cfg.layout
    J = cfg.junction

    design = metal.designs.design_planar.DesignPlanar()
    design.chips.main.size.size_x = f"{L.chip_x_um}um"
    design.chips.main.size.size_y = f"{L.chip_y_um}um"

    TransmonCross(design, "qubit", options=MDict(
        cross_length=dq["cross_length"][0],
        cross_width=dq["cross_width"][0],
        cross_gap=dq["cross_gap"][0],
        orientation=L.qubit_orientation,
        pos_x=f"{L.qubit_x_um}um",
        pos_y=f"{L.qubit_y_um}um",
        layer=str(L.metal_layer),
        connection_pads=MDict(readout=MDict(
            claw_cpw_length=L.claw_cpw_length,
            claw_cpw_width=L.claw_cpw_width,
            claw_gap=dq["claw_gap"][0],
            claw_length=dq["claw_length"][0],
            claw_width=dq["claw_width"][0],
            connector_location="0",
            connector_type="0",
            ground_spacing=dq["ground_spacing"][0],
        )),
        hfss_inductance=str(Lj_nH * 1e-9),
        q3d_inductance=str(Lj_nH * 1e-9),
    ))

    OpenToGround(design, "cpw_top", options=MDict(
        pos_x=f"{L.qubit_x_um}um",
        pos_y=f"{L.qubit_y_um + L.cpw_end_y_offset_um}um",
        orientation=L.cpw_end_orientation,
        width=dk["second_width"][0],
        gap=dk["second_gap"][0],
        termination_gap=L.cpw_termination_gap,
        layer=str(L.metal_layer),
    ))

    RouteStraight(design, "cpw_feed", options=MDict(
        pin_inputs=MDict(
            start_pin=MDict(component="cpw_top", pin="open"),
            end_pin=MDict(component="qubit", pin="readout"),
        ),
        trace_width=dk["second_width"][0],
        trace_gap=dk["second_gap"][0],
        layer=str(L.metal_layer),
    ))

    # JJ bridge sits at the cross arm that orientation maps to "south".
    cross_length_mm = _um_to_mm(dq["cross_length"][0])
    cross_gap_mm = _um_to_mm(dq["cross_gap"][0])
    jjy = -cross_gap_mm / 2 - cross_length_mm
    ori_deg = float(L.qubit_orientation)
    jjxprime = -np.sin(np.radians(ori_deg)) * jjy
    jjyprime = np.cos(np.radians(ori_deg)) * jjy

    JjDolan(design, "jj", options=MDict(
        pos_x=f"{L.qubit_x_um}um",
        pos_y=f"{L.qubit_y_um}um",
        orientation=str(ori_deg),
        bridge_length=f"{(cross_gap_mm - 2 * J.jj_pad_mm) * 1000}um",
        JJ_width=J.jj_width,
        bridge_layer=J.bridge_layer,
        JJ_layer=J.jj_layer,
    ))
    jj_comp = design.components["jj"]
    jj_comp.options.pos_x = f"{L.qubit_x_um / 1000 + jjxprime}mm"
    jj_comp.options.pos_y = f"{L.qubit_y_um / 1000 + jjyprime}mm"

    for name, comp in design.components.items():
        if not name.startswith("jj"):
            comp.options.layer = str(L.metal_layer)
    design.rebuild()
    return design


def _um_to_mm(s: str) -> float:
    """Convert a qiskit-metal-style '200um' / '0.2mm' string to mm."""
    s = s.strip()
    if s.endswith("um"):
        return float(s[:-2]) / 1000
    if s.endswith("mm"):
        return float(s[:-2])
    return float(s) / 1000  # assume um
