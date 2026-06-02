# Standalone GDS → Mask → Artemis pipeline

Two self-contained Python scripts:

1. **`mask_from_gds.py`** — rasterize a GDS into two npy masks (with / without
   JJ bridge), an annotated preview PNG, and a metafile JSON that records
   everything Artemis needs (cell counts, dx/dy/dz, domain extents, JJ
   location, CPW source-gap x-bounds, source y).

2. **`artemis_input_from_meta.py`** — read that metafile JSON and write an
   Artemis FDTD input file. Uses ONLY the Python stdlib (no numpy needed).

Both scripts are completely independent of the agentic `qpipe_*` /
`mcp_qubit_pipeline` workflow — you can drop this folder anywhere and use it.

---

## Setup

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate qubit-pipeline-standalone
```

### Option B — pip

```bash
pip install -r requirements.txt
```

Either route gets you: numpy, gdspy, scikit-image, matplotlib.
(`artemis_input_from_meta.py` doesn't need any of these — only stdlib.)

---

## Step 1: GDS → mask + metafile

```bash
python mask_from_gds.py qubit.gds
```

Default settings produce a 840×1100 cell mask at 1 μm/cell, autodetect JJ +
CPW source gaps, and write four files (using prefix = GDS basename):

```
qubit_mask_no_jj.npy        # float64 (nx, ny, 1), C-contiguous
qubit_mask_with_jj.npy      # same + JJ bridge
qubit_mask_preview.png      # annotated visualization
qubit_mask_meta.json        # all geometry info Artemis needs
```

### Common options

```bash
# Custom resolution + domain
python mask_from_gds.py qubit.gds --dx 1.0 --dy 1.0 --Lx-um 840 --Ly-um 1100

# Anisotropic resolution
python mask_from_gds.py qubit.gds --dx 2.0 --dy 1.0 --Lx-um 840 --Ly-um 1100

# Custom GDS layers (default: (1,0)=ground, (1,10)=etched, (1,11)=positive,
#                            (20,10)=JJ bridge)
python mask_from_gds.py mydesign.gds \
    --specs-no-jj 2,0 2,10 \
    --specs-with-jj 2,0 2,10 30,5

# Other tuning
python mask_from_gds.py qubit.gds \
    --pad-cut 500          # trim 500 μm off the top (e.g. wirebond pad)
    --trim-margin 200      # autocrop margin
    --pml-margin 100       # PML thickness (drawn on PNG + used for source y)
    --no-preview           # skip the preview PNG
```

`--help` lists everything.

---

## Step 2: metafile → Artemis input file

```bash
python artemis_input_from_meta.py qubit_mask_meta.json \
    --lj-nH 7.437 --freq-GHz 5.5
```

Writes `input_qubit` (filename = `input_<metafile-base>`). Override with
`--output my_input`.

### Common options

```bash
# Resonator-only (no JJ in sim)
python artemis_input_from_meta.py q.json --no-inductor --freq-GHz 7.0

# Tune sim parameters
python artemis_input_from_meta.py q.json --lj-nH 6.5 --freq-GHz 5.0 \
    --max-step 4000000 \
    --cfl 0.7 \
    --plt-intervals 2000

# Change physics constants
python artemis_input_from_meta.py q.json --lj-nH 7.0 \
    --eps-r-si 11.4 \
    --h-si-um 200 \
    --sigma 5.0e10
```

`--help` lists every default.

### How L_j is applied

Artemis uses a lumped inductor that sums in **series along x** (the current
direction). So `inductor_x_function` returns inductance **per cell**, not
total. The script automatically computes:

```
n_jj_x_cells  = round((jj_x_hi - jj_x_lo) / dx)
L_per_cell    = L_total / n_jj_x_cells
```

and prints the calculation:

```
Lj_total:    7.437 nH  (x ∈ [191.0, 220.0] μm)
N_x_cells:   29
L_per_cell:  0.2564 nH  = 7.437 / 29  → written into inductor_x_function
```

### Which mask gets referenced

Both inductor-on and inductor-off runs reference the **`no_jj`** mask by
default. The `with_jj` mask has a PEC bridge across the JJ gap which would
short out the lumped inductor — only useful for a galvanic-short sanity
check. Override with `--mask-npy <path>` if you need the other one.

---

## End-to-end example

```bash
conda activate qubit-pipeline-standalone

# Generate mask
python mask_from_gds.py qubit_5p5GHz.gds

# Generate Artemis input
python artemis_input_from_meta.py qubit_5p5GHz_mask_meta.json \
    --lj-nH 7.437 --freq-GHz 5.5

ls -1
# → qubit_5p5GHz.gds
#   qubit_5p5GHz_mask_no_jj.npy
#   qubit_5p5GHz_mask_with_jj.npy
#   qubit_5p5GHz_mask_preview.png
#   qubit_5p5GHz_mask_meta.json
#   input_qubit_5p5GHz
```

Now submit `input_qubit_5p5GHz` to Artemis (e.g. via your usual SLURM
script on Perlmutter). Two of the output files Artemis will need to find
relative to its working directory:

- `qubit_5p5GHz_mask_no_jj.npy` (the sigma + mu mask)
- `input_qubit_5p5GHz` (the input file itself)

Put both in the same dir as the Artemis run.

---

## GDS layer convention

The defaults assume the qiskit-metal / SQuADDS convention:

| `(layer, datatype)` | meaning |
|---|---|
| `(1, 0)`   | chip outline / ground plane |
| `(1, 10)`  | etched / subtract regions |
| `(1, 11)`  | positive features (CPW center conductor, claw, cross arms) |
| `(20, 10)` | JJ bridge (only in the `with_jj` mask) |

If your GDS uses different layers, pass `--specs-no-jj` and
`--specs-with-jj` to override.

---

## File inventory

```
qubit_pipeline_standalone/
├── README.md                      ← this file
├── environment.yml                ← conda env spec
├── requirements.txt               ← pip alternative
├── mask_from_gds.py               ← step 1: GDS → mask + metafile
└── artemis_input_from_meta.py     ← step 2: metafile → Artemis input
```

5 files total.
