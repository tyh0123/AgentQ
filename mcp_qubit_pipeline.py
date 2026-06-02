#!/usr/bin/env python3
"""
MCP Server for Superconducting Qubit Design Pipeline.

Tools:
  1. query_squadds       - Query SQuADDS DB for design parameters
  2. build_layout        - Build qiskit-metal layout (validation or full)
  3. export_gds          - Export design to GDS
  4. generate_mask       - GDS → npy mask + metafile
  5. generate_artemis_input - Generate Artemis FDTD input file
  6. estimate_hpc_resources - Estimate GPU count and wall time
  7. extract_timeseries  - Extract field time series from plotfiles

Usage:
  python mcp_qubit_pipeline.py
"""

import os
import sys
import json
import subprocess
import numpy as np
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Working directory for all outputs
WORK_DIR = os.environ.get("QUBIT_PIPELINE_WORKDIR",
                           os.path.dirname(os.path.abspath(__file__)))

# Python interpreter (use mag_torch conda env)
PYTHON = "/opt/anaconda3/envs/mag_torch/bin/python"

server = Server("qubit-pipeline")


# ══════════════════════════════════════════════════════════════════
#  Helper: run a Python snippet in a subprocess (isolates Qt/GUI deps)
# ══════════════════════════════════════════════════════════════════

def run_python(code: str, timeout: int = 300) -> str:
    """Run Python code in subprocess, return stdout+stderr."""
    env = os.environ.copy()
    env["QT_API"] = "pyqt6"
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True, text=True, timeout=timeout,
        cwd=WORK_DIR, env=env,
    )
    output = result.stdout
    if result.stderr:
        output += "\n[STDERR]\n" + result.stderr
    if result.returncode != 0:
        output = f"[EXIT CODE {result.returncode}]\n" + output
    return output


# ══════════════════════════════════════════════════════════════════
#  Tool definitions
# ══════════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="design_qubit",
            description=(
                "One-shot transmon design pipeline (single qubit + straight CPW feedline). "
                "Runs SQuADDS query → qiskit-metal layout → GDS → npy mask → preview PNG → "
                "sanity checks → Artemis FDTD input → SLURM submission script. "
                "All parameters come from defaults.yaml; pass `overrides` for SQuADDS qubit "
                "geometry tweaks and `config_overrides` for any other field via dotted path. "
                "Returns paths to all generated files and the sanity-check report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "qubit_frequency_GHz": {"type": "number"},
                    "anharmonicity_MHz": {"type": "number"},
                    "cavity_frequency_GHz": {"type": "number"},
                    "g_MHz": {"type": "number"},
                    "overrides": {
                        "type": "object",
                        "description": (
                            "SQuADDS qubit geometry overrides applied after the query "
                            '(string values). E.g. {"cross_length": "200um", "claw_length": "180um"}'
                        ),
                    },
                    "config_overrides": {
                        "type": "object",
                        "description": (
                            "Any defaults.yaml field via dotted path. "
                            'E.g. {"artemis.max_step": 4000000, "slurm.wall_time": "20:00:00"}'
                        ),
                    },
                    "prefix": {"type": "string", "description": "Output filename prefix (default: qubit_<freq>GHz)"},
                    "skip_artemis_slurm": {
                        "type": "boolean", "default": False,
                        "description": "If true, skip Artemis input + SLURM generation (design only)",
                    },
                },
                "required": ["qubit_frequency_GHz"],
            },
        ),
        Tool(
            name="generate_slurm",
            description="Generate a Perlmutter SLURM job script for an existing Artemis input. Defaults from defaults.yaml slurm section.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prefix": {"type": "string", "description": "Output prefix for SLURM script (perl_<prefix>.run)"},
                    "input_file": {"type": "string", "description": "Artemis input filename (basename used in srun command)"},
                    "config_overrides": {
                        "type": "object",
                        "description": 'Dotted-path overrides, e.g. {"slurm.wall_time": "20:00:00", "slurm.queue": "regular"}',
                    },
                },
                "required": ["prefix", "input_file"],
            },
        ),
        Tool(
            name="query_squadds",
            description="Query SQuADDS database for qubit design parameters. Supports single qubit (pass individual params) or multiple qubits (pass target_params_list). Returns cross dimensions, coupler params, CPW params, and Josephson inductance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "qubit_frequency_GHz": {"type": "number", "description": "Target qubit frequency in GHz (single qubit)"},
                    "anharmonicity_MHz": {"type": "number", "description": "Target anharmonicity in MHz (single qubit)"},
                    "cavity_frequency_GHz": {"type": "number", "description": "Target cavity frequency in GHz (single qubit)"},
                    "g_MHz": {"type": "number", "description": "Target coupling strength in MHz (single qubit)"},
                    "target_params_list": {
                        "type": "array",
                        "description": "List of target param dicts for multi-qubit. Each has qubit_frequency_GHz, anharmonicity_MHz, cavity_frequency_GHz, g_MHz",
                        "items": {"type": "object"},
                    },
                },
            },
        ),
        Tool(
            name="build_layout",
            description="Build qiskit-metal layout and export GDS. modes: 'validation' (single qubit + feedline), 'full' (single qubit + coupler + meander + feedline), 'multi' (N qubits + shared feedline), 'meander_only' (N meanders + couplers + feedline, NO qubits — terminates in OpenToGround).",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["validation", "full", "multi", "meander_only"], "description": "Layout mode"},
                    "design_params_file": {"type": "string", "description": "Path to design_params.json from query_squadds"},
                    "output_gds": {"type": "string", "description": "Output GDS filename", "default": "design.gds"},
                    "coupler_spacing_um": {"type": "number", "description": "Spacing between elements in um", "default": 2500},
                    "chip_size_x_um": {"type": "number", "description": "Chip x size in um", "default": 12000},
                    "chip_size_y_um": {"type": "number", "description": "Chip y size in um", "default": 6000},
                    "resolution_um": {"type": "number", "description": "Mask resolution in um/cell", "default": 1.0},
                    "output_prefix": {"type": "string", "description": "Output file prefix (meander_only mode)"},
                },
                "required": ["mode", "design_params_file"],
            },
        ),
        Tool(
            name="generate_mask",
            description="Convert GDS to npy mask files (with and without JJ bridge) and generate metafile with junction coordinates and grid parameters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gds_file": {"type": "string", "description": "Input GDS file path"},
                    "output_prefix": {"type": "string", "description": "Output prefix for npy/json files"},
                    "resolution_um": {"type": "number", "description": "Grid resolution in um/cell", "default": 1.0},
                    "target_nx": {"type": "integer", "description": "Force x dimension to this value (optional)"},
                    "pad_cut_um": {"type": "number", "description": "Cut this many um from top (remove launchpad)", "default": 500},
                    "specs_no_jj": {
                        "type": "array", "items": {"type": "array", "items": {"type": "integer"}},
                        "description": "Layer specs without JJ, e.g. [[1,0],[1,10],[1,11]]",
                    },
                    "specs_with_jj": {
                        "type": "array", "items": {"type": "array", "items": {"type": "integer"}},
                        "description": "Layer specs with JJ, e.g. [[1,0],[1,10],[1,11],[20,10]]",
                    },
                },
                "required": ["gds_file", "output_prefix"],
            },
        ),
        Tool(
            name="generate_artemis_input",
            description="Generate Artemis FDTD input file from metafile. Sets up grid, materials, source, inductor, and diagnostics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "metafile": {"type": "string", "description": "Path to mask metafile JSON"},
                    "output_file": {"type": "string", "description": "Output Artemis input filename"},
                    "mask_npy": {"type": "string", "description": "Mask npy file to use"},
                    "freq_GHz": {"type": "number", "description": "Excitation frequency in GHz", "default": 5.0},
                    "max_step": {"type": "integer", "description": "Max simulation steps", "default": 2000000},
                    "use_inductor": {"type": "boolean", "description": "Enable lumped inductor", "default": True},
                    "source_y_um": {"type": "number", "description": "Source y position in um (should be far from PML)"},
                },
                "required": ["metafile", "output_file", "mask_npy"],
            },
        ),
        Tool(
            name="estimate_hpc_resources",
            description="Estimate HPC resources (GPU count, wall time, memory) for an Artemis simulation based on grid size and step count.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_cellx": {"type": "integer"},
                    "n_celly": {"type": "integer"},
                    "n_cellz": {"type": "integer", "default": 40},
                    "max_step": {"type": "integer", "default": 2000000},
                    "gpu_type": {"type": "string", "enum": ["A100_80GB", "V100_32GB", "H100_80GB"], "default": "A100_80GB"},
                },
                "required": ["n_cellx", "n_celly"],
            },
        ),
        Tool(
            name="extract_timeseries",
            description="Extract field time series at probe points from Artemis plotfiles. Returns CSV files and FFT plots.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plot_dir": {"type": "string", "description": "Directory containing plotfiles"},
                    "field": {"type": "string", "description": "Field component", "default": "Ex"},
                    "probe1_x_um": {"type": "number", "description": "Probe 1 x in um"},
                    "probe1_y_um": {"type": "number", "description": "Probe 1 y in um"},
                    "probe2_x_um": {"type": "number", "description": "Probe 2 x in um"},
                    "probe2_y_um": {"type": "number", "description": "Probe 2 y in um"},
                    "probe_z_um": {"type": "number", "description": "Probe z in um", "default": 101.0},
                    "save_dir": {"type": "string", "description": "Output directory for CSV/plots"},
                    "resume": {"type": "boolean", "description": "Resume from existing CSV", "default": False},
                },
                "required": ["plot_dir", "probe1_x_um", "probe1_y_um", "probe2_x_um", "probe2_y_um"],
            },
        ),
        Tool(
            name="analyze_fft",
            description="Analyze FFT from existing CSV time series files. Returns peak frequencies and normalized spectrum.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_files": {"type": "array", "items": {"type": "string"}, "description": "CSV file paths"},
                    "npoints": {"type": "integer", "description": "Number of points to use (-1=all)", "default": -1},
                    "skip": {"type": "integer", "description": "Skip first N points", "default": 0},
                    "fft_max_GHz": {"type": "number", "description": "Max frequency for plot", "default": 20.0},
                    "save_dir": {"type": "string", "description": "Output directory"},
                },
                "required": ["csv_files"],
            },
        ),
    ]


# ══════════════════════════════════════════════════════════════════
#  Tool implementations
# ══════════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:

    # ── 0. design_qubit (one-shot pipeline) ──
    if name == "design_qubit":
        cfg_overrides = dict(arguments.get("config_overrides") or {})
        for k_arg, k_cfg in (
            ("qubit_frequency_GHz", "target.qubit_frequency_GHz"),
            ("anharmonicity_MHz", "target.anharmonicity_MHz"),
            ("cavity_frequency_GHz", "target.cavity_frequency_GHz"),
            ("g_MHz", "target.g_MHz"),
        ):
            if arguments.get(k_arg) is not None:
                cfg_overrides[k_cfg] = arguments[k_arg]
        overrides = arguments.get("overrides") or {}
        prefix = arguments.get("prefix")
        skip_artemis = bool(arguments.get("skip_artemis_slurm", False))

        code = f"""
import sys, json
sys.path.insert(0, {repr(WORK_DIR)})
from qpipe_config import Config, default_prefix
import qpipe_squadds, qpipe_layout, qpipe_gds, qpipe_mask, qpipe_artemis, qpipe_slurm

cfg = Config.load(overrides={json.dumps(cfg_overrides)})
overrides = {json.dumps(overrides)}
prefix = {json.dumps(prefix)} or default_prefix(cfg)
skip_artemis = {skip_artemis}

print(f"target: freq={{cfg.target.qubit_frequency_GHz}}GHz, anharm={{cfg.target.anharmonicity_MHz}}MHz, cavity={{cfg.target.cavity_frequency_GHz}}GHz, g={{cfg.target.g_MHz}}MHz")
print(f"prefix: {{prefix}}")
print()
print("[1/6] SQuADDS query")
dq, dk, Lj = qpipe_squadds.query(cfg, overrides=overrides)
print(qpipe_squadds.summarize(dq, dk, Lj))

print()
print("[2/6] Build qiskit-metal layout")
design = qpipe_layout.build(cfg, dq, dk, Lj)
print(f"  components: {{list(design.components.keys())}}")

print()
print("[3/6] Export GDS")
gds_file = f"{{prefix}}.gds"
qpipe_gds.export(design, gds_file)
print(f"  -> {{gds_file}}")

print()
print("[4/6] Rasterize mask + preview + metafile")
res = qpipe_mask.rasterize(gds_file, prefix, cfg, dq, Lj)
for k, v in res.files.items():
    print(f"  -> {{v}}")
jjc = res.meta['junction_location']['center']
print(f"  JJ index ({{jjc['ix']}},{{jjc['iy']}}) -> ({{jjc['x_um']:.1f}}um, {{jjc['y_um']:.1f}}um)")
print(f"  expected freq (C=80fF): {{res.meta['expected_qubit_freq_GHz']:.2f}} GHz")

print()
print("[4.5] Sanity checks")
issues = qpipe_mask.sanity_check(res, cfg)
if issues:
    print(f"  {{len(issues)}} issue(s):")
    for i in issues:
        print(f"    ! {{i}}")
else:
    print("  all checks passed")

if not skip_artemis:
    print()
    print("[5/6] Generate Artemis input")
    input_file = f"input_{{prefix}}"
    art = qpipe_artemis.write_input(res.meta, res.files['no_jj'], input_file, cfg)
    print(f"  -> {{input_file}}")
    print(f"  source y = {{art['source_y_um']:.0f}} um (auto from CPW)")
    print(f"  Lj = {{art['Lj_nH']:.2f}} nH, grid {{art['grid']}}")

    print()
    print("[6/6] Generate SLURM script")
    slurm_file = qpipe_slurm.write_slurm(prefix, input_file, cfg)
    print(f"  -> {{slurm_file}}")

summary = {{
    "prefix": prefix,
    "files": dict(res.files),
    "junction_location": res.meta['junction_location'],
    "expected_qubit_freq_GHz": res.meta['expected_qubit_freq_GHz'],
    "design_params": res.meta['design_params'],
    "sanity_issues": issues,
}}
if not skip_artemis:
    summary['artemis_input'] = input_file
    summary['slurm_script'] = slurm_file
    summary['source_y_um'] = art['source_y_um']
print()
print("SUMMARY_JSON:" + json.dumps(summary))
"""
        result = run_python(code, timeout=180)
        return [TextContent(type="text", text=result)]

    # ── 0b. generate_slurm ──
    elif name == "generate_slurm":
        prefix = arguments["prefix"]
        input_file = arguments["input_file"]
        cfg_overrides = arguments.get("config_overrides") or {}

        code = f"""
import sys, json
sys.path.insert(0, {repr(WORK_DIR)})
from qpipe_config import Config
import qpipe_slurm

cfg = Config.load(overrides={json.dumps(cfg_overrides)})
out = qpipe_slurm.write_slurm({json.dumps(prefix)}, {json.dumps(input_file)}, cfg)
print(f"-> {{out}}")
with open(out) as f:
    print(f.read())
"""
        result = run_python(code, timeout=60)
        return [TextContent(type="text", text=result)]

    # ── 1. query_squadds ──
    if name == "query_squadds":
        # Support single or multi-qubit query
        target_list = arguments.get("target_params_list", None)
        if target_list is None:
            target_list = [{
                "qubit_frequency_GHz": arguments["qubit_frequency_GHz"],
                "anharmonicity_MHz": arguments["anharmonicity_MHz"],
                "cavity_frequency_GHz": arguments["cavity_frequency_GHz"],
                "g_MHz": arguments["g_MHz"],
            }]

        code = f"""
import os, json
os.environ["QT_API"] = "pyqt6"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from squadds import Analyzer, SQuADDS_DB
from squadds.core.utils import convert_numpy

db = SQuADDS_DB()
db.select_system(["cavity_claw", "qubit"])
db.select_qubit("TransmonCross")
db.select_cavity_claw("RouteMeander")
db.select_resonator_type("quarter")
db.create_system_df()

target_list = {json.dumps(target_list)}
all_designs = []

for i, target in enumerate(target_list):
    analyzer = Analyzer(db)
    results = analyzer.find_closest(target, num_top=1, metric="Euclidean")
    dq = convert_numpy(analyzer.get_qubit_options(results))
    dc = convert_numpy(analyzer.get_cpw_options(results))
    dk = convert_numpy(analyzer.get_coupler_options(results))
    LJs = convert_numpy(analyzer.get_Ljs(results))
    all_designs.append({{
        "target_params": target,
        "qubit_options": dq,
        "cpw_options": dc,
        "coupler_options": dk,
        "Lj_nH": LJs,
    }})
    print(f"Qubit {{i+1}}: Lj={{LJs[0]:.2f}} nH, cross_length={{dq['cross_length'][0]}}")

output = {{
    "n_qubits": len(target_list),
    "designs": all_designs,
}}

outpath = os.path.join({repr(WORK_DIR)}, "design_params.json")
with open(outpath, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\\nSaved {{len(target_list)}} design(s) to: {{outpath}}")
"""
        result = run_python(code, timeout=180)
        return [TextContent(type="text", text=result)]

    # ── 2. build_layout ──
    elif name == "build_layout":
        mode = arguments["mode"]
        params_file = arguments["design_params_file"]
        output_gds = arguments.get("output_gds", "design.gds")
        coupler_spacing = arguments.get("coupler_spacing_um", 2500)
        chip_x = arguments.get("chip_size_x_um", 12000)
        chip_y = arguments.get("chip_size_y_um", 6000)
        resolution = arguments.get("resolution_um", 1.0)
        output_prefix = arguments.get("output_prefix", "meander_only")

        if mode == "validation":
            script = os.path.join(WORK_DIR, "single_qubit_validation.py")
            cmd_list = [PYTHON, script]
        elif mode == "full":
            script = os.path.join(WORK_DIR, "squadds_qmetal_test_together.py")
            cmd_list = [PYTHON, script]
        elif mode == "multi":
            script = os.path.join(WORK_DIR, "multi_qubit_pipeline.py")
            cmd_list = [
                PYTHON, script,
                "--design-params", params_file,
                "--coupler-spacing", str(coupler_spacing),
                "--chip-size-x", str(chip_x),
                "--chip-size-y", str(chip_y),
                "--resolution", str(resolution),
            ]
        elif mode == "meander_only":
            script = os.path.join(WORK_DIR, "meander_only_pipeline.py")
            cmd_list = [
                PYTHON, script,
                "--design-params", params_file,
                "--coupler-spacing", str(coupler_spacing),
                "--chip-size-x", str(chip_x),
                "--chip-size-y", str(chip_y),
                "--resolution", str(resolution),
                "--output-prefix", output_prefix,
            ]
        else:
            return [TextContent(type="text", text=f"Unknown mode: {mode}")]

        cmd_repr = repr(cmd_list)
        code = f"""
import subprocess, sys
result = subprocess.run(
    {cmd_repr},
    capture_output=True, text=True, timeout=300,
    env={{**__import__('os').environ, "QT_API": "pyqt6", "QT_QPA_PLATFORM": "offscreen"}},
    cwd={repr(WORK_DIR)},
)
print(result.stdout)
if result.stderr:
    print("[STDERR]", result.stderr[-500:])
print(f"Exit code: {{result.returncode}}")
"""
        result = run_python(code, timeout=360)
        return [TextContent(type="text", text=result)]

    # ── 3. generate_mask ──
    elif name == "generate_mask":
        gds_file = arguments["gds_file"]
        output_prefix = arguments["output_prefix"]
        resolution = arguments.get("resolution_um", 1.0)
        target_nx = arguments.get("target_nx", None)
        pad_cut = arguments.get("pad_cut_um", 500)
        specs_no_jj = arguments.get("specs_no_jj", [[1,0],[1,10],[1,11]])
        specs_with_jj = arguments.get("specs_with_jj", [[1,0],[1,10],[1,11],[20,10]])

        code = f"""
import numpy as np
import gdspy
import json
from skimage.draw import polygon as sk_polygon

gds_file = {repr(gds_file)}
output_prefix = {repr(output_prefix)}
resolution = {resolution}
target_nx = {target_nx}
pad_cut = {pad_cut}
specs_no_jj = [tuple(s) for s in {specs_no_jj}]
specs_with_jj = [tuple(s) for s in {specs_with_jj}]

gdsii = gdspy.GdsLibrary()
gdsii.read_gds(gds_file)
cell = gdsii.top_level()[0]
poly_dict = cell.get_polygons(by_spec=True)

print("GDS specs found:")
for spec in sorted(poly_dict.keys()):
    print(f"  {{spec}}: {{len(poly_dict[spec])}} polygons")

# Bounding box from larger set
all_polys = []
for spec in specs_with_jj:
    if spec in poly_dict:
        all_polys.extend(poly_dict[spec])

all_coords = np.vstack(all_polys)
x_min, y_min = all_coords.min(axis=0)
x_max, y_max = all_coords.max(axis=0)
x_range = x_max - x_min
y_range = y_max - y_min
scale = 1000.0 if x_range < 100 and y_range < 100 else 1.0

chip_x_um = x_range * scale
chip_y_um = y_range * scale
n_cellx = int(chip_x_um / resolution)
n_celly = int(chip_y_um / resolution)

def rasterize(specs):
    mask = np.zeros((n_celly, n_cellx), dtype=np.uint8)
    for spec in specs:
        if spec not in poly_dict:
            continue
        for poly in poly_dict[spec]:
            x_idx = (poly[:, 0] - x_min) / (x_range + 1e-12) * (n_cellx - 1)
            y_idx = (poly[:, 1] - y_min) / (y_range + 1e-12) * (n_celly - 1)
            rr, cc = sk_polygon(y_idx, x_idx)
            rr = np.clip(rr.astype(int), 0, n_celly - 1)
            cc = np.clip(cc.astype(int), 0, n_cellx - 1)
            mask[rr, cc] = 1
    return mask.T

mask_no_jj = rasterize(specs_no_jj)
mask_with_jj = rasterize(specs_with_jj)

# Trim
margin = int(500 / resolution)
row_var = np.array([mask_with_jj[i, :].var() for i in range(mask_with_jj.shape[0])])
active_x = np.where(row_var > 0.001)[0]
x_start = max(0, active_x[0] - margin) if len(active_x) > 0 else 0
x_end = min(n_cellx, active_x[-1] + margin) if len(active_x) > 0 else n_cellx
col_var = np.array([mask_with_jj[:, j].var() for j in range(mask_with_jj.shape[1])])
active_y = np.where(col_var > 0.001)[0]
y_start = max(0, active_y[0] - margin) if len(active_y) > 0 else 0
y_end = min(n_celly, active_y[-1] + margin) if len(active_y) > 0 else n_celly

mask_no_jj = mask_no_jj[x_start:x_end, y_start:y_end]
mask_with_jj = mask_with_jj[x_start:x_end, y_start:y_end]

# Pad cut
pad_cut_cells = int(pad_cut / resolution)
if mask_no_jj.shape[1] > pad_cut_cells:
    mask_no_jj = mask_no_jj[:, :-pad_cut_cells]
    mask_with_jj = mask_with_jj[:, :-pad_cut_cells]

# Force target_nx
if target_nx and mask_no_jj.shape[0] > target_nx:
    trim = (mask_no_jj.shape[0] - target_nx) // 2
    mask_no_jj = mask_no_jj[trim:trim+target_nx, :]
    mask_with_jj = mask_with_jj[trim:trim+target_nx, :]

nx, ny = mask_no_jj.shape

# Junction location
bridge = mask_with_jj.astype(int) - mask_no_jj.astype(int)
locs = np.where(bridge > 0)
if len(locs[0]) > 0:
    jj = dict(
        start=dict(ix=int(locs[0].min()), iy=int(locs[1].min()),
                   x_um=float(locs[0].min()*resolution), y_um=float(locs[1].min()*resolution)),
        end=dict(ix=int(locs[0].max()), iy=int(locs[1].max()),
                 x_um=float(locs[0].max()*resolution), y_um=float(locs[1].max()*resolution)),
        center=dict(ix=int((locs[0].min()+locs[0].max())//2), iy=int((locs[1].min()+locs[1].max())//2),
                    x_um=float((locs[0].min()+locs[0].max())/2*resolution),
                    y_um=float((locs[1].min()+locs[1].max())/2*resolution)),
    )
else:
    jj = dict(start=dict(ix=0,iy=0), end=dict(ix=0,iy=0), center=dict(ix=nx//2,iy=ny//3))
    print("Warning: JJ bridge not resolved")

# Save
f_no = output_prefix + "_no_jj.npy"
f_with = output_prefix + "_with_jj.npy"
np.save(f_no, np.repeat(mask_no_jj[:,:,np.newaxis], 1, axis=2).astype(np.float64))
np.save(f_with, np.repeat(mask_with_jj[:,:,np.newaxis], 1, axis=2).astype(np.float64))

# Preview PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jj_cx = jj["center"].get("ix", nx//2)
jj_cy = jj["center"].get("iy", ny//3)

# Downsample for preview if mask is large
ds = max(1, max(nx, ny) // 2000)
if ds > 1:
    preview_no = mask_no_jj[::ds, ::ds]
    preview_with = mask_with_jj[::ds, ::ds]
    jj_cx_ds = jj_cx // ds
    jj_cy_ds = jj_cy // ds
    print(f"Preview downsampled {{ds}}x: {{preview_no.shape}}")
else:
    preview_no = mask_no_jj
    preview_with = mask_with_jj
    jj_cx_ds = jj_cx
    jj_cy_ds = jj_cy

# Show GAP regions (1-mask) so circuit structure is visible
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
axes[0].imshow((1-preview_no).T, cmap='hot', origin='lower', aspect='auto')
axes[0].set_title(f'Without JJ - GAP regions ({{nx}}x{{ny}})')
axes[0].set_xlabel('x (cells)')
axes[0].set_ylabel('y (cells)')
axes[0].plot(jj_cx_ds, jj_cy_ds, 'c+', markersize=15, markeredgewidth=2)
axes[0].annotate(f'JJ ({{jj_cx}},{{jj_cy}})', (jj_cx_ds, jj_cy_ds), color='cyan',
                 fontsize=10, xytext=(10, 10), textcoords='offset points')

axes[1].imshow((1-preview_with).T, cmap='hot', origin='lower', aspect='auto')
axes[1].set_title(f'With JJ bridge - GAP regions ({{nx}}x{{ny}})')
axes[1].set_xlabel('x (cells)')
axes[1].set_ylabel('y (cells)')
axes[1].plot(jj_cx_ds, jj_cy_ds, 'c+', markersize=15, markeredgewidth=2)

plt.tight_layout()
preview_path = output_prefix + "_preview.png"
try:
    plt.savefig(preview_path, dpi=150)
    import os as _os
    if _os.path.exists(preview_path):
        print(f"Preview PNG saved: {{preview_path}} ({{_os.path.getsize(preview_path)}} bytes)")
    else:
        print(f"WARNING: preview PNG not created at {{preview_path}}")
except Exception as _e:
    print(f"WARNING: Failed to save preview PNG: {{_e}}")
    preview_path = None
plt.close()

meta = dict(
    mask_no_jj=f_no, mask_with_jj=f_with,
    preview_png=preview_path,
    n_cellx=nx, n_celly=ny, resolution_um=resolution,
    prob_hix_m=nx*resolution*1e-6, prob_hiy_m=ny*resolution*1e-6,
    metal_fraction_no_jj=float(mask_no_jj.sum()/mask_no_jj.size),
    junction_location=jj,
)
meta_path = output_prefix + "_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))
print(f"\\nSaved: {{f_no}}, {{f_with}}, {{preview_path}}, {{meta_path}}")
"""
        result = run_python(code)
        return [TextContent(type="text", text=result)]

    # ── 4. generate_artemis_input ──
    elif name == "generate_artemis_input":
        metafile = arguments["metafile"]
        output_file = arguments["output_file"]
        mask_npy = arguments["mask_npy"]
        freq = arguments.get("freq_GHz", None)
        max_step = arguments.get("max_step", None)
        use_inductor = arguments.get("use_inductor", True)
        source_y = arguments.get("source_y_um", None)
        cfg_overrides = arguments.get("config_overrides") or {}
        if max_step is not None:
            cfg_overrides["artemis.max_step"] = max_step

        code = f"""
import sys, json
sys.path.insert(0, {repr(WORK_DIR)})
from qpipe_config import Config
import qpipe_artemis

with open({repr(metafile)}) as f:
    meta = json.load(f)
cfg = Config.load(overrides={json.dumps(cfg_overrides)})
art = qpipe_artemis.write_input(
    meta, {repr(mask_npy)}, {repr(output_file)}, cfg,
    freq_GHz={freq if freq is not None else 'None'},
    source_y_um={source_y if source_y is not None else 'None'},
    use_inductor={use_inductor},
)
print(f"Artemis input written: {{art['output_file']}}")
print(f"  freq: {{art['freq_GHz']}} GHz")
print(f"  source y: {{art['source_y_um']:.0f}} um")
print(f"  grid: {{art['grid']}}")
print(f"  Lj: {{art['Lj_nH']:.2f}} nH")
print(f"  inductor: {{'enabled' if art['use_inductor'] else 'disabled'}}")
"""
        result = run_python(code, timeout=60)
        return [TextContent(type="text", text=result)]

    # ── 5. estimate_hpc_resources ──
    elif name == "estimate_hpc_resources":
        nx = arguments["n_cellx"]
        ny = arguments["n_celly"]
        nz = arguments.get("n_cellz", 40)
        max_step = arguments.get("max_step", 2000000)
        gpu_type = arguments.get("gpu_type", "A100_80GB")

        total_cells = nx * ny * nz
        n_fields = 20  # E, B, sigma, mu, etc.
        bytes_per_cell = 8 * n_fields * 2  # double, 2x for old/new

        gpu_mem = {"A100_80GB": 80e9, "V100_32GB": 32e9, "H100_80GB": 80e9}[gpu_type]
        cells_per_gpu = int(gpu_mem / bytes_per_cell * 0.7)  # 70% utilization

        min_gpus = max(1, int(np.ceil(total_cells / cells_per_gpu)))

        # Time estimate
        dx = 1e-6  # assume 1um
        dt = 0.8 * dx / (3e8 * np.sqrt(3))
        sim_time_s = max_step * dt
        # ~1us per cell per step on A100
        wall_time_s = total_cells * max_step * 1e-6 / max(min_gpus, 1)

        result = json.dumps({
            "total_cells": total_cells,
            "memory_GB": total_cells * bytes_per_cell / 1e9,
            "gpu_type": gpu_type,
            "gpu_memory_GB": gpu_mem / 1e9,
            "min_gpus": min_gpus,
            "recommended_gpus": min_gpus * 2,  # headroom
            "dt_s": dt,
            "sim_time_ns": sim_time_s * 1e9,
            "estimated_wall_time_hours": wall_time_s / 3600,
            "max_step": max_step,
            "suggested_max_grid_size": [nx, max(ny // min_gpus, 100), nz],
        }, indent=2)

        return [TextContent(type="text", text=result)]

    # ── 6. extract_timeseries ──
    elif name == "extract_timeseries":
        script = os.path.join(WORK_DIR, "extract_point_timeseries.py")
        plot_dir = arguments["plot_dir"]
        field = arguments.get("field", "Ex")
        p1x = arguments["probe1_x_um"]
        p1y = arguments["probe1_y_um"]
        p2x = arguments["probe2_x_um"]
        p2y = arguments["probe2_y_um"]
        pz = arguments.get("probe_z_um", 101.0)
        save_dir = arguments.get("save_dir", WORK_DIR)
        resume = arguments.get("resume", False)

        cmd = [
            PYTHON, script,
            "--plot_dir", plot_dir,
            "--field", field,
            "--probe1_x", str(p1x * 1e-6),
            "--probe1_y", str(p1y * 1e-6),
            "--probe1_z", str(pz * 1e-6),
            "--probe2_x", str(p2x * 1e-6),
            "--probe2_y", str(p2y * 1e-6),
            "--probe2_z", str(pz * 1e-6),
            "--save_dir", save_dir,
            "--use_index",
        ]
        if resume:
            cmd.append("--resume")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=WORK_DIR)
            result = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr else "")
        except subprocess.TimeoutExpired:
            result = "Timeout: extraction took > 10 minutes"

        return [TextContent(type="text", text=result)]

    # ── 7. analyze_fft ──
    elif name == "analyze_fft":
        script = os.path.join(WORK_DIR, "plot_timeseries.py")
        csv_files = arguments["csv_files"]
        npoints = arguments.get("npoints", -1)
        skip = arguments.get("skip", 0)
        fft_max = arguments.get("fft_max_GHz", 20.0)
        save_dir = arguments.get("save_dir", WORK_DIR)

        cmd = [
            PYTHON, script,
            "--csv", *csv_files,
            "--npoints", str(npoints),
            "--skip", str(skip),
            "--fft_max", str(fft_max),
            "--save_dir", save_dir,
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=WORK_DIR)
            result = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr else "")
        except subprocess.TimeoutExpired:
            result = "Timeout: FFT analysis took > 2 minutes"

        return [TextContent(type="text", text=result)]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
