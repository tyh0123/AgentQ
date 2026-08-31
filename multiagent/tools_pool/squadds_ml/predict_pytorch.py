"""Standalone PyTorch inference for transmon_cross_hamiltonian_inverse.

One-time setup (needs Keras + TensorFlow installed):
    python predict_pytorch.py --convert

Inference (only needs torch + scikit-learn + joblib + numpy):
    python predict_pytorch.py --qubit-freq 4.85 --anharmonicity -205.0
    python predict_pytorch.py --input '{"qubit_frequency_GHz": 4.85, "anharmonicity_MHz": -205.0}'
    python predict_pytorch.py --input examples/predict_transmon_hamiltonian.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "transmon_cross_hamiltonian_inverse"
KERAS_PATH = ART / "model" / "best_inverse_model_surrogate_defined_loss.keras"
TORCH_PATH = ART / "model" / "inverse_model.pt"
X_SCALER = "scalers/scaler_X_{name}.save"
Y_SCALER = "scalers/scaler_y_{name}_one_hot_encoding.save"

INPUT_NAMES = ["qubit_frequency_GHz", "anharmonicity_MHz"]
OUTPUT_NAMES = [
    "design_options.connection_pads.readout.claw_length",
    "design_options.connection_pads.readout.ground_spacing",
    "design_options.cross_length",
]
OUTPUT_UNITS = {name: "m" for name in OUTPUT_NAMES}


class InverseModel(nn.Module):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.fc0 = nn.Linear(len(INPUT_NAMES), 64)
        self.act = nn.LeakyReLU(alpha)
        self.out = nn.Linear(64, len(OUTPUT_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.fc0(x)))


def convert_from_keras() -> None:
    from keras.models import load_model

    km = load_model(KERAS_PATH, compile=False)
    w0, b0 = km.get_layer("fc0").get_weights()
    w1, b1 = km.get_layer("qiskit_output").get_weights()
    alpha = float(km.get_layer("leaky_relu0").negative_slope)

    model = InverseModel(alpha)
    with torch.no_grad():
        # Keras Dense stores W as (in, out); torch Linear expects (out, in).
        model.fc0.weight.copy_(torch.from_numpy(w0.T).float())
        model.fc0.bias.copy_(torch.from_numpy(b0).float())
        model.out.weight.copy_(torch.from_numpy(w1.T).float())
        model.out.bias.copy_(torch.from_numpy(b1).float())

    torch.save({"alpha": alpha, "state_dict": model.state_dict()}, TORCH_PATH)
    print(f"Saved PyTorch checkpoint -> {TORCH_PATH}")


def load_torch_model() -> InverseModel:
    if not TORCH_PATH.exists():
        sys.exit(
            f"PyTorch checkpoint not found at {TORCH_PATH}.\n"
            "Run `python predict_pytorch.py --convert` first."
        )
    ckpt = torch.load(TORCH_PATH, map_location="cpu", weights_only=True)
    model = InverseModel(ckpt["alpha"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def scale_inputs(rows: List[dict]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(INPUT_NAMES)), dtype=np.float32)
    for col, name in enumerate(INPUT_NAMES):
        scaler = joblib.load(ART / X_SCALER.format(name=name))
        column = np.array([[row[name]] for row in rows], dtype=np.float64)
        matrix[:, col] = scaler.transform(column).flatten()
    return matrix


def unscale_outputs(scaled: np.ndarray) -> np.ndarray:
    result = scaled.copy()
    for col, name in enumerate(OUTPUT_NAMES):
        scaler = joblib.load(ART / Y_SCALER.format(name=name))
        result[:, col] = scaler.inverse_transform(scaled[:, col : col + 1]).flatten()
    return result


def predict(rows: List[dict]) -> List[dict]:
    for i, row in enumerate(rows):
        missing = [n for n in INPUT_NAMES if n not in row]
        extra = [n for n in row if n not in INPUT_NAMES]
        if missing or extra:
            sys.exit(
                f"Row {i}: expected inputs {INPUT_NAMES}. "
                f"Missing: {missing or 'none'}. Extra: {extra or 'none'}."
            )

    model = load_torch_model()
    scaled_in = scale_inputs(rows)
    with torch.no_grad():
        scaled_out = model(torch.from_numpy(scaled_in)).numpy()
    unscaled = unscale_outputs(scaled_out)
    return [
        {name: float(unscaled[i, j]) for j, name in enumerate(OUTPUT_NAMES)}
        for i in range(len(rows))
    ]


def _parse_input(value: str) -> List[dict]:
    path = Path(value)
    payload = json.loads(path.read_text() if path.exists() else value)
    if isinstance(payload, dict) and "inputs" in payload:
        payload = payload["inputs"]
    return payload if isinstance(payload, list) else [payload]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--convert",
        action="store_true",
        help="Convert Keras checkpoint -> PyTorch (.pt) and exit.",
    )
    ap.add_argument(
        "--input",
        help="JSON string, path to a JSON file, or omit to use --qubit-freq/--anharmonicity.",
    )
    ap.add_argument("--qubit-freq", type=float, help="qubit_frequency_GHz")
    ap.add_argument("--anharmonicity", type=float, help="anharmonicity_MHz")
    args = ap.parse_args()

    if args.convert:
        convert_from_keras()
        return

    if args.input:
        rows = _parse_input(args.input)
    elif args.qubit_freq is not None and args.anharmonicity is not None:
        rows = [
            {
                "qubit_frequency_GHz": args.qubit_freq,
                "anharmonicity_MHz": args.anharmonicity,
            }
        ]
    else:
        ap.error("Provide --input or both --qubit-freq and --anharmonicity.")

    preds = predict(rows)
    print(
        json.dumps(
            {
                "predictions": preds,
                "input_order": INPUT_NAMES,
                "output_order": OUTPUT_NAMES,
                "output_units": OUTPUT_UNITS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
