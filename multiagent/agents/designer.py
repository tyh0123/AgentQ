"""
DesignerAgent — the producer.

Turns a target spec into the full artifact chain via the deterministic tool
layer, running the project's **real** pipeline:

    geometry (SQuADDS DB → local inverse-ML fallback)
      → qiskit-metal layout → GDS
      → rasterized npy masks + preview PNG + metafile
      → Artemis FDTD input
      → Perlmutter sbatch

It can also `revise` an artifact when a critic hands back a concrete fix.

For the fail→detect→recover demo it accepts a set of injected faults so it
produces a subtly-wrong artifact on purpose.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Set

import config
import tools
from blackboard import DesignRecord
from .base import Agent


class DesignerAgent(Agent):
    name = "designer"

    def __init__(self, faults: Set[str] | None = None, source: str = config.DEFAULT_SOURCE,
                 clean_gds: bool = True):
        super().__init__()
        self.faults = faults or set()
        self.source = source
        self.clean_gds = clean_gds

    # ── build the whole chain from scratch ──
    def design(self, record: DesignRecord) -> DesignRecord:
        cfg = tools.cfg_for(record)

        if record.mode == "multi":                           # N qubits on a shared feedline
            self._design_multi(record, cfg)
            return record

        # 1+2) geometry (SQuADDS / ML) → qiskit-metal layout → GDS, by mode
        lj = self._geometry_and_gds(record, cfg)

        # 3) GDS → npy masks + preview PNG + metafile (real) + sanity checks
        self._rasterize(record, cfg, lj)
        self._maybe_clean_gds(record)
        self._run_drc(record, cfg)

        # 4+5) Artemis FDTD input + sbatch (source dispatched on the topology)
        self._emit_artemis_slurm(record, cfg)
        return record

    # ── Artemis input + sbatch, source dispatched on the layout topology ──
    def _emit_artemis_slurm(self, record: DesignRecord, cfg) -> None:
        if record.mode == "qubit":
            self._write_artemis(record, cfg)                 # vertical CPW source
        else:
            self._write_artemis_horizontal(record, cfg)      # horizontal feedline source
        record.slurm_script = tools.write_slurm(cfg, record.prefix, record.artemis_input)
        record.log(f"{self.name}: wrote sbatch → {os.path.basename(record.slurm_script)}")

    # ── horizontal feedline: Ey input-port source + per-qubit lumped inductor ──
    def _write_artemis_horizontal(self, record: DesignRecord, cfg) -> None:
        import json
        if record.mode == "qubit_resonator":
            with open(record.metafile) as f:
                meta = json.load(f)
            jj = meta["junction_location"]
            jj_list = [{"x_lo_um": min(jj["start"]["x_um"], jj["end"]["x_um"]),
                        "x_hi_um": max(jj["start"]["x_um"], jj["end"]["x_um"]),
                        "cy_um": jj["center"]["y_um"],
                        "Lj_nH": meta["design_params"]["Lj_nH"]}]
            mask_info = meta["mask_info"]
            feedline_y_cell = None
        else:  # multi
            jj_list = record.metrics["jj_list"]
            mask_info = record.metrics["mask_info"]
            feedline_y_cell = record.metrics.get("feedline_y_cell")
        art = tools.write_artemis_horizontal(cfg, mask_info, record.mask_no_jj,
                                             record.prefix, jj_list, record.freq_GHz,
                                             feedline_y_cell)
        record.artemis_input = art["output_file"]
        record.metrics["source_x_um"] = art["source_x_um"]
        record.log(f"{self.name}: wrote Artemis input → {os.path.basename(record.artemis_input)} "
                   f"(Ey input-port source x={art['source_x_um']:.0f}µm, "
                   f"{art['n_inductors']} lumped Lj, grid={art['grid']})")

    # ── geometry + GDS, dispatched on the layout mode ──
    def _geometry_and_gds(self, record: DesignRecord, cfg) -> float:
        if record.mode == "qubit_resonator":
            target = {"qubit_frequency_GHz": record.freq_GHz,
                      "anharmonicity_MHz": record.anharm_MHz,
                      "cavity_frequency_GHz": cfg.target.cavity_frequency_GHz,
                      "g_MHz": cfg.target.g_MHz}
            geo = tools.query_full(cfg, [target], record.coupler_overrides)[0]
            geo["source_used"] = "squadds_db"
            record.dq, record.dk = geo["dq"], geo["dk"]
            lj = self._apply_lj(record, geo["Lj_nH"], geo["source_used"])
            record.log(f"{self.name}: qubit+resonator geometry via squadds_db "
                       f"(cross={geo['dq']['cross_length'][0]}, "
                       f"resonator={geo['dc']['total_length'][0]}, Lj={lj:.2f}nH, "
                       f"coupler gap={geo['dk']['coupling_space'][0]}, "
                       f"overlap={geo['dk']['coupling_length'][0]})")
            record.gds_file = tools.build_gds_full(cfg, {**geo, "Lj_nH": lj}, record.prefix)
            record.log(f"{self.name}: built qubit+resonator layout "
                       f"(qubit+claw+coupler+resonator+feedline) "
                       f"→ GDS {os.path.basename(record.gds_file)}")
            return lj

        # qubit only: SQuADDS DB query, local inverse-ML on a miss/offline
        geo = tools.resolve_geometry(cfg, self.source)
        record.dq, record.dk = geo["dq"], geo["dk"]
        lj = self._apply_lj(record, geo["Lj_nH"], geo["source_used"])
        record.log(f"{self.name}: geometry via {geo['source_used']} "
                   f"(cross={record.dq['cross_length'][0]}, Lj={lj:.2f}nH)")
        record.gds_file = tools.build_gds(cfg, record.dq, record.dk, lj, record.prefix)
        record.log(f"{self.name}: built qubit-only layout → GDS "
                   f"{os.path.basename(record.gds_file)}")
        return lj

    # ── multi-qubit: N unit cells on a shared feedline ──
    def _design_multi(self, record: DesignRecord, cfg) -> None:
        geos = tools.query_full(cfg, record.targets, record.coupler_overrides)
        record.source_used = "squadds_db"
        record.geometry = {"source": "squadds_db", "n_qubits": len(geos)}
        lj_str = ", ".join(f"{g['Lj_nH']:.2f}" for g in geos)
        record.log(f"{self.name}: queried {len(geos)} qubits from squadds_db "
                   f"(Lj = {lj_str} nH)")

        record.gds_file, record.layout_info = tools.build_gds_multi(
            cfg, geos, record.prefix, record.coupler_spacing_um)
        record.log(f"{self.name}: built {len(geos)}-qubit layout "
                   f"(shared feedline) → GDS {os.path.basename(record.gds_file)}")

        mask = tools.rasterize_multi(cfg, record.gds_file, record.prefix, record.layout_info)
        record.mask_no_jj = mask["mask_npy"]
        record.preview_png = mask["preview_png"]

        # per-qubit LC frequency estimate from each Lj (C=80fF), for DP1
        qubits = []
        for g, tgt in zip(geos, record.targets):
            lj = g["Lj_nH"]
            f_exp = 1.0 / (2 * math.pi * math.sqrt(lj * 1e-9 * config.C_SHUNT_F)) / 1e9
            qubits.append({"target_GHz": tgt["qubit_frequency_GHz"],
                           "Lj_nH": lj, "expected_GHz": f_exp})
        record.metrics.update({
            "qubits": qubits, "metal_fraction": mask["metal_fraction"],
            "n_cellx": mask["n_cellx"], "n_celly": mask["n_celly"],
            "jj_list": mask["jj_list"], "feedline_y_cell": mask["feedline_y_cell"],
            "mask_info": {"n_cellx": mask["n_cellx"], "n_celly": mask["n_celly"],
                          "resolution_um": cfg.mask.resolution_um},
        })
        record.log(f"{self.name}: rasterized multi-qubit mask → "
                   f"{os.path.basename(record.mask_no_jj)} + preview "
                   f"{os.path.basename(record.preview_png)} "
                   f"(metal={mask['metal_fraction']:.2%}, {len(mask['jj_list'])} JJ located)")
        self._maybe_clean_gds(record)
        self._run_drc(record, cfg)

        # Artemis input (Ey input-port source + N lumped inductors) + sbatch
        self._emit_artemis_slurm(record, cfg)

    # ── KLayout DRC on the final GDS, feeding Decision Point 1b ──
    def _run_drc(self, record: DesignRecord, cfg) -> None:
        rep = tools.run_drc(cfg, record.gds_file)
        record.metrics["drc"] = rep
        if rep.get("skipped"):
            record.log(f"{self.name}: DRC skipped — {rep.get('reason')}")
            return
        n = rep.get("n_violations", len(rep.get("violations", [])))
        record.log(f"{self.name}: ran KLayout DRC on "
                   f"{os.path.basename(record.gds_file)} → "
                   f"{'clean' if rep.get('passed', n == 0) else f'{n} violation(s)'}")

    # ── trim the on-disk GDS to the clean fab layers (after rasterization) ──
    def _maybe_clean_gds(self, record: DesignRecord) -> None:
        if not (self.clean_gds and record.gds_file):
            return
        kept = tools.clean_gds_layers(record.gds_file)
        specs = ", ".join(f"{l}/{d}" for (l, d) in sorted(kept))
        record.log(f"{self.name}: cleaned GDS → layers {specs}")

    # ── record the geometry + optionally inject the bad-Lj fault ──
    def _apply_lj(self, record: DesignRecord, lj_true: float, source_used: str) -> float:
        record.source_used = source_used
        lj = lj_true
        if "bad_lj" in self.faults:                          # FAULT: mis-set Lj
            lj *= 10.0
            record.log(f"{self.name}: [fault injected] Lj set to {lj:.2f} nH (10x too high)")
        record.geometry = {"source": source_used, "Lj_nH_true": lj_true, "Lj_nH_used": lj,
                           "cross_length": record.dq["cross_length"][0],
                           "claw_length": record.dq["claw_length"][0],
                           "ground_spacing": record.dq["ground_spacing"][0]}
        return lj

    # ── apply a critic's fix and regenerate the affected artifacts ──
    def revise(self, record: DesignRecord, fix: Dict[str, Any]) -> DesignRecord:
        cfg = tools.cfg_for(record)
        kind = fix["kind"]

        if kind == "set_source_y":
            record.log(f"{self.name}: revising — set source_y = {fix['source_y_um']} "
                       f"(re-generating Artemis input)")
            self._write_artemis(record, cfg, source_y_um=fix["source_y_um"])
            record.slurm_script = tools.write_slurm(cfg, record.prefix, record.artemis_input)

        elif kind == "recompute_lj":
            self.faults.discard("bad_lj")                  # stop re-injecting the fault
            lj = record.geometry["Lj_nH_true"]             # the correct analytic value
            record.geometry["Lj_nH_used"] = lj
            record.log(f"{self.name}: revising — restore Lj = {lj:.2f} nH and re-rasterize "
                       f"(GDS geometry unchanged; Lj only re-seeds the metafile + Artemis)")
            self._rasterize(record, cfg, lj)              # reuse the existing GDS
            self._emit_artemis_slurm(record, cfg)
        else:
            raise ValueError(f"unknown fix kind: {kind}")
        return record

    # ── internal: rasterize the GDS into masks + preview + metafile ──
    def _rasterize(self, record: DesignRecord, cfg, lj: float) -> None:
        mask = tools.rasterize_mask(cfg, record.gds_file, record.prefix, record.dq, lj)
        record.metafile     = mask["metafile"]
        record.mask_no_jj   = mask["mask_no_jj"]
        record.mask_with_jj = mask["mask_with_jj"]
        record.preview_png  = mask["preview_png"]
        record.metrics.update({
            "expected_freq_GHz": mask["expected_freq_GHz"],
            "metal_fraction":    mask["metal_fraction"],
            "n_celly":           mask["n_celly"],
            "n_cellx":           mask["n_cellx"],
            "sanity_issues":     mask["sanity_issues"],
        })
        record.log(f"{self.name}: rasterized mask → {os.path.basename(record.metafile)} "
                   f"+ preview {os.path.basename(record.preview_png)} "
                   f"(expected f≈{mask['expected_freq_GHz']:.2f}GHz, "
                   f"metal={mask['metal_fraction']:.2%}, "
                   f"{len(mask['sanity_issues'])} sanity issue(s))")

    # ── internal: write the Artemis input, honouring the source_in_pml fault ──
    def _write_artemis(self, record: DesignRecord, cfg, source_y_um=None) -> None:
        if source_y_um is None and "source_in_pml" in self.faults:
            ny = record.metrics["n_celly"]
            source_y_um = ny - 5                            # FAULT: 5 cells from boundary
            record.log(f"{self.name}: [fault injected] source_y = {source_y_um} "
                       f"(only 5 cells from the top PML)")
        art = tools.write_artemis(
            cfg, record.metafile, record.mask_no_jj, record.prefix,
            record.freq_GHz, source_y_um=source_y_um,
        )
        record.artemis_input = art["output_file"]
        record.metrics["source_y_um"] = art["source_y_um"]
        record.log(f"{self.name}: wrote Artemis input → {os.path.basename(record.artemis_input)} "
                   f"(source_y={art['source_y_um']:.0f}µm, grid={art['grid']})")
