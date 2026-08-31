"""
Config dataclasses + YAML loader for the qubit pipeline.

Usage:
    cfg = Config.load()                              # defaults.yaml
    cfg = Config.load(overrides={"target.qubit_frequency_GHz": 6.0})
    cfg.target.qubit_frequency_GHz                   # typed access
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, List, Optional, Tuple

import yaml


DEFAULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defaults.yaml")


# ──────────────────────────────────────────────
#  Section dataclasses
# ──────────────────────────────────────────────

@dataclass
class TargetParams:
    qubit_frequency_GHz: float
    anharmonicity_MHz: float
    cavity_frequency_GHz: float
    g_MHz: float


@dataclass
class FallbackCpw:
    second_width: str
    second_gap: str


@dataclass
class FallbackQubitGeometry:
    cross_width: str
    cross_gap: str
    claw_width: str
    claw_gap: str


@dataclass
class MlFallbackConfig:
    enabled: bool
    freq_GHz_threshold: float
    anharm_MHz_threshold: float
    endpoint: str
    model_id: str
    timeout_s: float


@dataclass
class SquaddsConfig:
    qubit_type: str
    cavity_type: str
    resonator_type: str
    metric: str
    num_top: int
    fallback_cpw: FallbackCpw
    fallback_qubit_geometry: FallbackQubitGeometry
    ml_fallback: MlFallbackConfig


@dataclass
class LayoutConfig:
    qubit_x_um: float
    qubit_y_um: float
    chip_x_um: float
    chip_y_um: float
    cpw_end_y_offset_um: float        # y distance from qubit to top of CPW (OpenToGround position)
    cpw_end_orientation: str          # orientation of the OpenToGround termination
    cpw_termination_gap: str          # OpenToGround termination_gap (small etch at end)
    qubit_orientation: str
    claw_cpw_length: str
    claw_cpw_width: str
    metal_layer: int


@dataclass
class JunctionConfig:
    jj_width: str
    bridge_layer: int
    jj_layer: int
    jj_pad_mm: float


@dataclass
class MaskConfig:
    resolution_um: float
    target_nx: Optional[int]
    pad_cut_um: float
    trim_margin_um: float
    specs_no_jj: List[Tuple[int, int]]
    specs_with_jj: List[Tuple[int, int]]
    jj_gap_cells: int = 1


@dataclass
class SanityConfig:
    jj_pos_tol_cells: int
    cpw_top_gap_cells: int
    asymmetry_threshold: float
    metal_fraction_min: float
    metal_fraction_max: float


@dataclass
class DrcRule:
    """One declarative DRC rule. `type` picks the KLayout check:

      width      — min feature width on each of `layers`         (min_um)
      spacing    — min same-layer spacing on each of `layers`    (min_um)
      area       — min polygon area on each of `layers`          (min_um2)
      separation — min distance between `layer_a` and `layer_b`  (min_um)
      enclosing  — `layer_a` must enclose `layer_b` by ≥ min_um  (min_um)
      overlap    — `layer_a` and `layer_b` must not overlap      (no threshold)
    """
    name: str
    type: str
    min_um: float = 0.0
    min_um2: float = 0.0
    layers: List[Tuple[int, int]] = field(default_factory=list)
    layer_a: Optional[Tuple[int, int]] = None
    layer_b: Optional[Tuple[int, int]] = None
    enabled: bool = True


@dataclass
class DrcConfig:
    enabled: bool
    rules: List[DrcRule] = field(default_factory=list)
    fail_on_violation: bool = False
    write_marker_gds: bool = True
    marker_layer: int = 1000


@dataclass
class ArtemisConfig:
    max_step: int
    n_cellz: int
    cfl: float
    pml_margin_um: float
    h_si_um: float
    eps_r_si: float
    prob_hiz_um: float
    plt_intervals: int
    diag_slice_z_um: List[float]
    npy_k_index: int
    sigma_value: float
    cpw_source_widening_um: float    # extend source x past each CPW gap by this much on each side
    jj_gap_model: bool = False       # bridge a single gap cell on the with_jj mask vs. distribute Lj


@dataclass
class SlurmConfig:
    nodes: int
    gpus: int
    queue: str
    account: str
    wall_time: str
    binary: str
    job_name_prefix: str


@dataclass
class Config:
    target: TargetParams
    squadds: SquaddsConfig
    layout: LayoutConfig
    junction: JunctionConfig
    mask: MaskConfig
    sanity: SanityConfig
    drc: DrcConfig
    artemis: ArtemisConfig
    slurm: SlurmConfig

    # ──────────────────────────────────────────
    #  Loading / overriding
    # ──────────────────────────────────────────

    @classmethod
    def load(cls, defaults_path: str = DEFAULTS_PATH,
             overrides: Optional[dict] = None) -> "Config":
        """Load defaults.yaml and apply optional dotted-key overrides.

        overrides: dict mapping dotted paths to values, e.g.
            {"target.qubit_frequency_GHz": 6.0,
             "artemis.max_step": 4000000}
        """
        with open(defaults_path) as f:
            data = yaml.safe_load(f)

        if overrides:
            data = _apply_overrides(data, overrides)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        sq = data["squadds"]
        return cls(
            target=TargetParams(**data["target"]),
            squadds=SquaddsConfig(
                qubit_type=sq["qubit_type"],
                cavity_type=sq["cavity_type"],
                resonator_type=sq["resonator_type"],
                metric=sq["metric"],
                num_top=sq["num_top"],
                fallback_cpw=FallbackCpw(**sq["fallback_cpw"]),
                fallback_qubit_geometry=FallbackQubitGeometry(**sq["fallback_qubit_geometry"]),
                ml_fallback=MlFallbackConfig(**sq["ml_fallback"]),
            ),
            layout=LayoutConfig(**data["layout"]),
            junction=JunctionConfig(**data["junction"]),
            mask=MaskConfig(
                resolution_um=data["mask"]["resolution_um"],
                target_nx=data["mask"]["target_nx"],
                pad_cut_um=data["mask"]["pad_cut_um"],
                trim_margin_um=data["mask"]["trim_margin_um"],
                specs_no_jj=[tuple(s) for s in data["mask"]["specs_no_jj"]],
                specs_with_jj=[tuple(s) for s in data["mask"]["specs_with_jj"]],
                jj_gap_cells=data["mask"].get("jj_gap_cells", 1),
            ),
            sanity=SanityConfig(**data["sanity"]),
            drc=cls._drc_from_dict(data.get("drc")),
            artemis=ArtemisConfig(**data["artemis"]),
            slurm=SlurmConfig(**data["slurm"]),
        )

    @staticmethod
    def _drc_from_dict(d: Optional[dict]) -> DrcConfig:
        """Build DrcConfig. Missing section → disabled default (keeps configs
        that predate the DRC feature working). A `rules:` list is the modular
        format; legacy flat keys (min_metal_width_um, width_layers, …) are
        synthesized into the equivalent rules so old configs keep working."""
        if not d:
            return DrcConfig(enabled=False)
        if "rules" in d:
            rules = [_drc_rule_from_dict(r) for r in d["rules"] or []]
        else:
            rules = _drc_rules_from_legacy(d)
        return DrcConfig(
            enabled=d["enabled"],
            rules=rules,
            fail_on_violation=d.get("fail_on_violation", False),
            write_marker_gds=d.get("write_marker_gds", True),
            marker_layer=d.get("marker_layer", 1000),
        )

    def to_dict(self) -> dict:
        return _asdict_recursive(self)


# ──────────────────────────────────────────────
#  DRC rule parsing
# ──────────────────────────────────────────────

def _drc_rule_from_dict(r: dict) -> DrcRule:
    return DrcRule(
        name=r["name"],
        type=r["type"],
        min_um=r.get("min_um", 0.0),
        min_um2=r.get("min_um2", 0.0),
        layers=[tuple(s) for s in r.get("layers", [])],
        layer_a=tuple(r["layer_a"]) if r.get("layer_a") else None,
        layer_b=tuple(r["layer_b"]) if r.get("layer_b") else None,
        enabled=r.get("enabled", True),
    )


def _drc_rules_from_legacy(d: dict) -> List[DrcRule]:
    """Synthesize the classic five rules from the pre-`rules:` flat keys."""
    _pairs = lambda key: [tuple(s) for s in d.get(key, [])]
    width, spacing = _pairs("width_layers"), _pairs("spacing_layers")
    gap, jj = _pairs("gap_layers"), _pairs("jj_layers")
    return [
        DrcRule("min_metal_width", "width",   min_um=d["min_metal_width_um"], layers=width),
        DrcRule("min_area",        "area",    min_um2=d["min_area_um2"],      layers=width),
        DrcRule("min_spacing",     "spacing", min_um=d["min_spacing_um"],     layers=spacing),
        DrcRule("min_gap",         "width",   min_um=d["min_gap_um"],         layers=gap),
        DrcRule("jj_min_width",    "width",   min_um=d["jj_min_width_um"],    layers=jj,
                enabled=bool(jj)),
    ]


# ──────────────────────────────────────────────
#  Override helpers
# ──────────────────────────────────────────────

def _apply_overrides(data: dict, overrides: dict) -> dict:
    """Apply dotted-key overrides to a nested dict. Returns a new copy."""
    out = copy.deepcopy(data)
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = out
        for k in keys[:-1]:
            if k not in node:
                raise KeyError(f"Override path '{dotted}' invalid at '{k}'")
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"Override path '{dotted}' invalid at '{keys[-1]}'")
        node[keys[-1]] = value
    return out


def _asdict_recursive(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict_recursive(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict_recursive(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _asdict_recursive(v) for k, v in obj.items()}
    return obj


# ──────────────────────────────────────────────
#  Output filename helper
# ──────────────────────────────────────────────

def default_prefix(cfg: Config) -> str:
    f = cfg.target.qubit_frequency_GHz
    return f"qubit_{str(f).replace('.', 'p')}GHz"
