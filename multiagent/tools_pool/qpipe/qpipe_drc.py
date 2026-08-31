"""
Design Rule Check (DRC) on the exported GDS using the KLayout polygon engine.

Runs per-layer geometric checks against fab rules from cfg.drc and returns a
structured list of violations (with coordinates in um). Optionally writes a
marker-layer GDS so violations can be inspected visually in KLayout.

Rules are declared in cfg.drc.rules (the `drc.rules:` list in defaults.yaml) —
each names a check type and its layers/threshold, so adding or retuning a rule
is a YAML edit, not a code change. Supported rule types:

  width      — min feature width on each layer            (min_um)
  spacing    — min same-layer spacing on each layer       (min_um)
  area       — min polygon area on each layer             (min_um2)
  separation — min distance between layer_a and layer_b   (min_um)
  enclosing  — layer_a must enclose layer_b by ≥ min_um   (min_um)
  overlap    — layer_a and layer_b must not overlap

Legacy flat configs (min_metal_width_um / width_layers / …) are synthesized
into the equivalent rules by qpipe_config, so pre-`rules:` YAMLs keep working.

Requires the `klayout` Python module:  pip install klayout

This runs on GDS polygons (dbu / nanometre precision), so it catches
sub-micron rules the rasterized-mask sanity_check cannot. It is complementary
to qpipe_mask.sanity_check (which checks the simulation mask topology).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from qpipe_config import Config


# ──────────────────────────────────────────────
#  Result containers
# ──────────────────────────────────────────────

@dataclass
class DrcViolation:
    rule: str                       # "min_metal_width", "min_spacing", ...
    layer: Tuple[int, int]          # (layer, datatype)
    limit_um: float                 # the rule threshold
    value_um: Optional[float]       # measured value (None if not measurable)
    location_um: Tuple[float, float]

    def __str__(self) -> str:
        v = f"{self.value_um:.3f}um" if self.value_um is not None else "?"
        return (f"[{self.rule}] layer {self.layer}: {v} < {self.limit_um}um "
                f"@ ({self.location_um[0]:.1f}, {self.location_um[1]:.1f})um")


@dataclass
class DrcResult:
    violations: List[DrcViolation] = field(default_factory=list)
    marker_gds: Optional[str] = None
    checked_layers: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0


# ──────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────

def run_drc(gds_file: str, cfg: Config,
            marker_gds: Optional[str] = None) -> DrcResult:
    """Run DRC on `gds_file`. Returns a DrcResult.

    marker_gds: path to write violation markers to. If None and
    cfg.drc.write_marker_gds is True, defaults to '<gds_stem>_drc_markers.gds'.
    """
    import klayout.db as kdb

    D = cfg.drc
    ly = kdb.Layout()
    ly.read(gds_file)
    dbu = ly.dbu                      # micrometres per database unit
    top = ly.top_cell()

    # qiskit-metal designs in mm but its GDS export keeps a um/nm UNITS
    # header, so the whole chip reads ~1000x too small. Detect that the same
    # way qpipe_mask._bbox does (chip span < 100 "um") and scale the
    # effective um-per-dbu so thresholds and reported values are real um.
    bb = top.bbox()
    span_um = max(bb.width(), bb.height()) * dbu
    if 0 < span_um < 100.0:
        dbu *= 1000.0

    def to_dbu(um: float) -> int:
        return int(round(um / dbu))

    violations: List[DrcViolation] = []
    marker_regions: List["kdb.Region"] = []
    checked: List[Tuple[int, int]] = []

    def _region(spec: Tuple[int, int]):
        li = ly.find_layer(spec[0], spec[1])
        if li is None:
            return None
        reg = kdb.Region(top.begin_shapes_rec(li))
        reg.merge()
        checked.append(spec)
        return reg

    # ── run every enabled rule from cfg.drc.rules ──
    for rule in D.rules:
        if not rule.enabled:
            continue

        if rule.type in ("width", "spacing"):
            for spec in rule.layers:
                reg = _region(spec)
                if reg is None:
                    continue
                check = reg.width_check if rule.type == "width" else reg.space_check
                _collect_edge_pairs(
                    check(to_dbu(rule.min_um)),
                    rule.name, spec, rule.min_um, dbu,
                    violations, marker_regions,
                )

        elif rule.type == "area":
            # min area: isolated polygons (post-merge) strictly below the
            # limit. NB: KLayout's Region.with_area(min, max) overload-resolves
            # ambiguously in Python (the max arg gets read as the `inverse`
            # flag), so filter the merged polygons explicitly instead.
            for spec in rule.layers:
                reg = _region(spec)
                if reg is None:
                    continue
                small = kdb.Region()
                for p in reg.each():
                    if p.area() * dbu * dbu < rule.min_um2:
                        small.insert(p)
                if not small.is_empty():
                    _collect_polygons(
                        small, rule.name, spec, rule.min_um2, dbu,
                        violations, marker_regions, area=True,
                    )

        elif rule.type in ("separation", "enclosing"):
            reg_a, reg_b = _region(rule.layer_a), _region(rule.layer_b)
            if reg_a is None or reg_b is None:
                continue
            check = (reg_a.separation_check if rule.type == "separation"
                     else reg_a.enclosing_check)
            _collect_edge_pairs(
                check(reg_b, to_dbu(rule.min_um)),
                rule.name, rule.layer_a, rule.min_um, dbu,
                violations, marker_regions,
            )

        elif rule.type == "overlap":
            reg_a, reg_b = _region(rule.layer_a), _region(rule.layer_b)
            if reg_a is None or reg_b is None:
                continue
            inter = reg_a & reg_b
            if not inter.is_empty():
                _collect_polygons(
                    inter, rule.name, rule.layer_a, 0.0, dbu,
                    violations, marker_regions,
                )

        else:
            raise ValueError(
                f"unknown DRC rule type '{rule.type}' (rule '{rule.name}')")

    # ── marker GDS ──
    out_marker = None
    if D.write_marker_gds and violations:
        out_marker = marker_gds or _default_marker_path(gds_file)
        # markers keep the source file's raw dbu so they overlay the GDS as-is
        _write_markers(out_marker, marker_regions, ly.dbu, D.marker_layer)

    # de-dup checked layers, preserve order
    seen = set()
    checked_unique = [s for s in checked if not (s in seen or seen.add(s))]

    return DrcResult(violations=violations, marker_gds=out_marker,
                     checked_layers=checked_unique)


# ──────────────────────────────────────────────
#  Collectors
# ──────────────────────────────────────────────

def _collect_edge_pairs(edge_pairs, rule, spec, limit_um, dbu,
                        violations, marker_regions):
    """Turn a KLayout EdgePairs result into DrcViolation records."""
    import klayout.db as kdb
    n = 0
    for ep in edge_pairs.each():
        loc = _edge_pair_center_um(ep, dbu)
        val = _edge_pair_distance_um(ep, dbu)
        violations.append(DrcViolation(rule, tuple(spec), limit_um, val, loc))
        n += 1
    if n:
        # markers: convert edge pairs to small polygons for visual overlay
        marker_regions.append(edge_pairs.polygons())


def _collect_polygons(region, rule, spec, limit_um, dbu,
                      violations, marker_regions, area=False):
    n = 0
    for poly in region.each():
        bbox = poly.bbox()
        cx = (bbox.left + bbox.right) / 2.0 * dbu
        cy = (bbox.bottom + bbox.top) / 2.0 * dbu
        val = (poly.area() * dbu * dbu) if area else None
        violations.append(DrcViolation(rule, tuple(spec), limit_um, val, (cx, cy)))
        n += 1
    if n:
        marker_regions.append(region)


# ──────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────

def _edge_pair_center_um(ep, dbu) -> Tuple[float, float]:
    e1, e2 = ep.first, ep.second
    cx = (e1.x1 + e1.x2 + e2.x1 + e2.x2) / 4.0 * dbu
    cy = (e1.y1 + e1.y2 + e2.y1 + e2.y2) / 4.0 * dbu
    return (cx, cy)


def _edge_pair_distance_um(ep, dbu) -> float:
    """Approximate the violating dimension: distance between the two edges'
    midpoints (good enough for reporting width/space/gap magnitude)."""
    e1, e2 = ep.first, ep.second
    m1x, m1y = (e1.x1 + e1.x2) / 2.0, (e1.y1 + e1.y2) / 2.0
    m2x, m2y = (e2.x1 + e2.x2) / 2.0, (e2.y1 + e2.y2) / 2.0
    return math.hypot(m1x - m2x, m1y - m2y) * dbu


def _write_markers(path, marker_regions, dbu, marker_layer):
    import klayout.db as kdb
    out = kdb.Layout()
    out.dbu = dbu
    cell = out.create_cell("DRC_MARKERS")
    li = out.layer(marker_layer, 0)
    for reg in marker_regions:
        cell.shapes(li).insert(reg)
    out.write(path)


def _default_marker_path(gds_file: str) -> str:
    import os
    stem = os.path.splitext(gds_file)[0]
    return f"{stem}_drc_markers.gds"


# ──────────────────────────────────────────────
#  Reporting
# ──────────────────────────────────────────────

def summarize(res: DrcResult) -> str:
    if res.passed:
        return f"  DRC PASSED ({len(res.checked_layers)} layer(s) checked)"
    by_rule: dict = {}
    for v in res.violations:
        by_rule[v.rule] = by_rule.get(v.rule, 0) + 1
    lines = [f"  DRC FAILED: {len(res.violations)} violation(s)"]
    for rule, n in sorted(by_rule.items()):
        lines.append(f"    {rule}: {n}")
    for v in res.violations[:10]:
        lines.append(f"    {v}")
    if len(res.violations) > 10:
        lines.append(f"    ... and {len(res.violations) - 10} more")
    if res.marker_gds:
        lines.append(f"  marker GDS: {res.marker_gds}")
    return "\n".join(lines)


def to_report_dict(res: DrcResult) -> dict:
    """JSON-serializable summary for the MCP tool / metafile."""
    return {
        "passed": res.passed,
        "n_violations": len(res.violations),
        "checked_layers": [list(s) for s in res.checked_layers],
        "marker_gds": res.marker_gds,
        "violations": [
            {
                "rule": v.rule,
                "layer": list(v.layer),
                "limit_um": v.limit_um,
                "value_um": v.value_um,
                "location_um": list(v.location_um),
            }
            for v in res.violations
        ],
    }
