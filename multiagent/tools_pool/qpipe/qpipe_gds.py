"""
GDS export with the qiskit-metal renderer monkey-patches we need to dodge
known qiskit-metal bugs (None geometry rows, NaN layer columns, non-polygon
items in the q_subtract tables).
"""
from __future__ import annotations

import gdstk
import numpy as np
import pandas as pd

import qiskit_metal.renderers.renderer_gds.gds_renderer as gds_mod


def export(design, gds_file: str) -> None:
    valid_geom = (gdstk.Polygon, gdstk.FlexPath, gdstk.RobustPath)

    o_qg = gds_mod.QGDSRenderer._qgeometry_to_gds
    o_pm = gds_mod.QGDSRenderer._positive_mask
    o_hf = gds_mod.QGDSRenderer._handle_q_subtract_false

    def p_qg(self, e):
        if hasattr(e, "geometry") and e.geometry is None:
            return None
        if hasattr(e, "layer") and pd.isna(e.layer):
            e = e.copy()
            e["layer"] = 1
        return o_qg(self, e)

    def p_pm(self, lib, top, gc, cn, cl, prec):
        for key in ("q_subtract_true", "q_subtract_false"):
            items = self.chip_info[cn][cl].get(key, [])
            if items is not None:
                self.chip_info[cn][cl][key] = [
                    i for i in items if isinstance(i, valid_geom)
                ]
        return o_pm(self, lib, top, gc, cn, cl, prec)

    def p_hf(self, cn, cl, gc):
        items = self.chip_info[cn][cl].get("q_subtract_false", [])
        if items is not None:
            self.chip_info[cn][cl]["q_subtract_false"] = [
                i for i in items if isinstance(i, valid_geom)
            ]
        return o_hf(self, cn, cl, gc)

    gds_mod.QGDSRenderer._qgeometry_to_gds = p_qg
    gds_mod.QGDSRenderer._positive_mask = p_pm
    gds_mod.QGDSRenderer._handle_q_subtract_false = p_hf

    try:
        for name in design.qgeometry.tables:
            df = design.qgeometry.tables[name]
            df = df[df["geometry"].notnull()].copy()
            if "layer" not in df.columns:
                df["layer"] = 1
            df["layer"] = (
                df["layer"].fillna(1).replace([np.inf, -np.inf], 1).astype(int)
            )
            design.qgeometry.tables[name] = df

        g = design.renderers.gds
        g.options["tolerance"] = "0.0001"
        g.options["precision"] = "0.000000001"
        g.options["short_segments_to_not_fillet"] = "True"
        g.options["fabricate"] = "False"
        g.export_to_gds(gds_file)
    finally:
        gds_mod.QGDSRenderer._qgeometry_to_gds = o_qg
        gds_mod.QGDSRenderer._positive_mask = o_pm
        gds_mod.QGDSRenderer._handle_q_subtract_false = o_hf
