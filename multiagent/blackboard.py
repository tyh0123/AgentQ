"""
The shared blackboard: one DesignRecord flows between the designer and the
critics. Agents read and write it; the orchestrator logs every transition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DesignRecord:
    # target spec (from the user / natural language)
    freq_GHz: float
    anharm_MHz: float
    prefix: str
    cavity_GHz: Optional[float] = None    # readout resonator target (qubit_resonator/multi)
    g_MHz: Optional[float] = None         # qubit–cavity coupling target (qubit_resonator/multi)
    mode: str = "qubit"                   # qubit | qubit_resonator | multi
    targets: Optional[List[Dict[str, Any]]] = None   # multi: per-qubit target dicts
    coupler_spacing_um: float = 2500.0    # multi: spacing between qubits
    coupler_overrides: Optional[Dict[str, str]] = None   # resonator↔feedline gap/overlap/down

    # produced by the designer (deterministic tool layer)
    geometry: Optional[Dict[str, float]] = None       # cross/claw/ground + Lj + source
    source_used: Optional[str] = None                 # "squadds_db" | "local_inverse_ml"
    dq: Optional[Dict[str, Any]] = None               # SQuADDS-shaped qubit options
    dk: Optional[Dict[str, Any]] = None               # SQuADDS-shaped coupler/CPW options
    gds_file: Optional[str] = None                    # exported GDS path
    layout_info: Optional[Dict[str, Any]] = None      # multi: positions / spacing
    metafile: Optional[str] = None                    # mask metafile path
    mask_no_jj: Optional[str] = None
    mask_with_jj: Optional[str] = None
    preview_png: Optional[str] = None                 # mask visualization
    artemis_input: Optional[str] = None
    slurm_script: Optional[str] = None

    # measured facts the critics inspect
    metrics: Dict[str, Any] = field(default_factory=dict)

    # audit trail: every critic verdict + every applied fix
    history: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.history.append(msg)
