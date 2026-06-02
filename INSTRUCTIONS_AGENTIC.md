# Agentic qubit-pipeline — setup instructions

This is the natural-language → Claude Code → MCP server → qpipe Python
pipeline. You speak ("design me a 5.5 GHz transmon"), Claude calls the
`design_qubit` MCP tool, which runs the SQuADDS → qiskit-metal → GDS → npy
mask → Artemis input → SLURM script pipeline end-to-end.

## 1. Files to copy into the new folder

```
qubit_pipeline_agentic/
├── INSTRUCTIONS_AGENTIC.md         ← this file
├── environment_agentic.yml         ← conda env spec
├── requirements_agentic.txt        ← pip alternative
├── defaults.yaml                   ← all default parameters (override via CLI)
│
├── qpipe_config.py                 ← config loader + dataclasses
├── qpipe_squadds.py                ← SQuADDS query (null-coupler fallback)
├── qpipe_layout.py                 ← qiskit-metal layout
├── qpipe_gds.py                    ← GDS export
├── qpipe_mask.py                   ← npy mask rasterization
├── qpipe_artemis.py                ← Artemis FDTD input writer
├── qpipe_slurm.py                  ← Perlmutter SLURM script writer
│
├── cli.py                          ← agentic CLI (full SQuADDS → SLURM)
├── cli_from_gds.py                 ← CLI starting from an existing GDS
├── mcp_qubit_pipeline.py           ← Claude Code MCP server
│
├── .mcp.json                       ← MCP server registration (edit paths!)
└── .claude/
    ├── settings.local.json         ← permissions (edit paths!)
    └── commands/
        ├── design-qubit.md         ← /design-qubit slash command
        ├── setup-sim.md            ← /setup-sim slash command
        ├── run-pipeline.md         ← /run-pipeline slash command
        └── analyze-sim.md          ← /analyze-sim slash command
```

12 Python files + 1 YAML + 6 Claude/MCP files = **19 files** total.

## 2. Install the conda environment

```bash
cd qubit_pipeline_agentic
conda env create -f environment_agentic.yml
conda activate qubit-pipeline
```

This installs everything from a single channel mix (conda-forge for the
native deps, pip for `gdspy`/`qiskit-metal`/`SQuADDS`/`mcp`).

Alternative pure-pip route (slower, less reliable on macOS due to geopandas
native deps):

```bash
pip install -r requirements_agentic.txt
```

## 3. Edit absolute paths in `.mcp.json` and `.claude/settings.local.json`

Both files currently point at the old location. After moving, update:

**`.mcp.json`**
```json
{
  "mcpServers": {
    "qubit-pipeline": {
      "command": "/opt/anaconda3/envs/qubit-pipeline/bin/python",   ← new env path
      "args": ["/absolute/path/to/qubit_pipeline_agentic/mcp_qubit_pipeline.py"],
      "env": {
        "QT_API": "pyqt6",
        "QT_QPA_PLATFORM": "offscreen",
        "QUBIT_PIPELINE_WORKDIR": "/absolute/path/to/qubit_pipeline_agentic"
      }
    }
  }
}
```

To find the conda env python path:
```bash
conda activate qubit-pipeline
which python    # use this in `command`
```

**`.claude/settings.local.json`** — the `permissions.allow` list has paths
hardcoded. Open it and replace `/Users/ytang4/research/Qmetal_test/pythonProject1`
and `/opt/anaconda3/envs/mag_torch` with your new paths. (Or just delete
the file and accept permission prompts the first time around — Claude Code
will re-grow the allow-list as you approve calls.)

## 4. Smoke-test the MCP server

```bash
conda activate qubit-pipeline
QT_API=pyqt6 QT_QPA_PLATFORM=offscreen python mcp_qubit_pipeline.py --help 2>&1 | head -5
```

If it prints without import errors, the env is good. (The script is a stdio
MCP server, so it won't print anything useful — the goal is just to confirm
imports work.)

## 5. Launch Claude Code in the new folder

```bash
cd qubit_pipeline_agentic
claude
```

Claude Code auto-discovers `.mcp.json` and `.claude/commands/*.md`. The
first time you run a slash command Claude Code may ask to approve the MCP
server — say yes.

## 6. Try it

```
> design me a 5.5 GHz transmon
```

Claude invokes `/design-qubit` → `design_qubit` MCP tool → runs everything,
returns paths to GDS / npy masks / preview PNG / Artemis input / SLURM
script.

Other things you can say:
- "design me a 4.5 GHz transmon with anharmonicity -210 MHz"
- "now change claw_length to 180um"
- "bump max_step to 4 million and use the regular queue"
- "skip the Artemis part, I just want the GDS"

## What the pipeline does NOT include (yet)

- Multi-qubit + coupler designs (the older `query_squadds` + `build_layout`
  MCP tools work for these but aren't validated for this demo)
- Postprocessing (FFT, matrix pencil, beat-fit) — see `postprocess/`
  package in the original folder if you need that. Not part of the
  design-only agentic workflow.

## Common gotchas

- **MCP server stale**: if you edit any `qpipe_*.py` or `mcp_qubit_pipeline.py`,
  restart Claude Code so MCP picks up the new code. Quick way: `/exit` then
  re-run `claude`.
- **`gdspy` missing**: it's pip-only — make sure `pip install gdspy` ran
  inside the conda env (not the system Python).
- **Qt complaints on headless servers**: the `QT_QPA_PLATFORM=offscreen`
  env var in `.mcp.json` handles this for qiskit-metal's hidden GUI calls.
- **SQuADDS network timeout**: SQuADDS hits a HuggingFace dataset on first
  use and caches it. Run `python -c "from squadds import SQuADDS_DB; SQuADDS_DB()"`
  once with a good connection to seed the cache before the demo.
