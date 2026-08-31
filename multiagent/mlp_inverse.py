"""
Local inverse-design ML: (qubit_frequency_GHz, anharmonicity_MHz) -> geometry + L_J.

This is the project's own trained inverse model, run *locally* (torch only, no
network) via the deployment bundle's standalone PyTorch inference. It replaces
the remote HF-Space call so the whole pipeline — and the poster demo — runs
offline and reproducibly.

  - geometry  : the trained MLP  (Linear(2->64) -> LeakyReLU -> Linear(64->3)),
                weights `inverse_model.pt`, trained with a surrogate-defined loss.
  - L_J       : closed-form transmon inversion (harmonic approximation).
"""
from __future__ import annotations

import math
from typing import Dict

import config  # noqa: F401  (side effect: puts ML_DIR / PYPROJECT_DIR on sys.path)

# SI constants for the analytic L_J.
_H    = 6.62607015e-34
_E    = 1.602176634e-19
_PHI0 = _H / (2 * _E)


def analytic_Lj_nH(freq_GHz: float, anharm_MHz: float) -> float:
    """Standard transmon inversion:  E_C≈|α|,  hf=√(8 E_J E_C)-E_C,  L_J=Φ₀²/(4π²E_J)."""
    E_C  = abs(anharm_MHz) * 1e6 * _H
    hf_q = freq_GHz * 1e9 * _H
    E_J  = (hf_q + E_C) ** 2 / (8.0 * E_C)
    L_J  = _PHI0 ** 2 / (4.0 * math.pi ** 2 * E_J)
    return L_J * 1e9


def predict(freq_GHz: float, anharm_MHz: float) -> Dict[str, float]:
    """Run the local inverse model. Returns geometry in µm plus L_J in nH.

    Raises RuntimeError if the trained model / torch aren't available so callers
    can decide how to degrade — there is no silent fabrication of geometry.
    """
    try:
        import predict_pytorch as pp  # from ML_DIR
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(f"local inverse model unavailable: {e}") from e

    geom_m = pp.predict([{
        "qubit_frequency_GHz": float(freq_GHz),
        "anharmonicity_MHz":   float(anharm_MHz),
    }])[0]

    return {
        "cross_length_um":    geom_m["design_options.cross_length"] * 1e6,
        "claw_length_um":     geom_m["design_options.connection_pads.readout.claw_length"] * 1e6,
        "ground_spacing_um":  geom_m["design_options.connection_pads.readout.ground_spacing"] * 1e6,
        "Lj_nH":              analytic_Lj_nH(freq_GHz, anharm_MHz),
        "source":             "local_mlp_inverse + analytic_Lj",
    }
