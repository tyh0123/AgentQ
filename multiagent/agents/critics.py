"""
Critic agents — independent reviewers at AgentQ's decision points.

Each critic inspects the designer's real artifacts and returns a Verdict. When
Claude is reachable it makes the *judgment*; the concrete *fix* is always
computed here from physics. This is the multi-agent core: a producer and
independent critics, not one monolithic agent.

  MaskCritic       — Decision Point 1: mask + lumped-Lj frequency sanity
  DrcCritic        — Decision Point 1b: KLayout fab design rules on the GDS
  SimConfigCritic  — Decision Point 2: Artemis source placement vs the PML

Decision Point 3 (spectral result interpretation) runs *after* the FDTD job and
is out of scope here — it belongs to the downstream run/collect/post-process
agent.
"""
from __future__ import annotations

import config
import math
import llm
from blackboard import DesignRecord
from .base import Agent, Verdict


class MaskCritic(Agent):
    """DP1 — does the rasterized mask + lumped inductor reproduce the target?"""
    name = "mask-critic"

    def review(self, record: DesignRecord) -> Verdict:
        if record.metrics.get("qubits"):
            return self._review_multi(record)

        f_exp = record.metrics["expected_freq_GHz"]
        f_tgt = record.freq_GHz
        rel = abs(f_exp - f_tgt) / f_tgt
        frac = record.metrics["metal_fraction"]

        sanity = record.metrics.get("sanity_issues", [])

        issues, fix = [], None
        if rel > config.FREQ_TOL_PCT:
            issues.append(
                f"LC frequency estimate {f_exp:.2f} GHz is {rel:.0%} off the "
                f"{f_tgt:.2f} GHz target (tol {config.FREQ_TOL_PCT:.0%}) — Lj is likely mis-set")
            fix = {"kind": "recompute_lj"}
        if not (config.METAL_FRACTION_MIN <= frac <= config.METAL_FRACTION_MAX):
            issues.append(f"metal fraction {frac:.2%} outside sane band")

        facts = (f"geometry source = {record.geometry.get('source')}; "
                 f"target f_q = {f_tgt:.3f} GHz; expected LC f = {f_exp:.3f} GHz "
                 f"({rel:.1%} off); metal fraction = {frac:.3f}; "
                 f"Lj = {record.geometry['Lj_nH_used']:.2f} nH; "
                 f"rasterizer sanity checks: "
                 f"{'; '.join(sanity) if sanity else 'all passed'}.")
        return self._verdict(facts, issues, fix)

    def _review_multi(self, record: DesignRecord) -> Verdict:
        """DP1 for the N-qubit chip: each qubit's lumped-Lj estimate must land
        near its own target, and the shared-feedline mask must be sane."""
        qubits = record.metrics["qubits"]
        frac = record.metrics["metal_fraction"]
        issues = []
        for i, q in enumerate(qubits):
            rel = abs(q["expected_GHz"] - q["target_GHz"]) / q["target_GHz"]
            if rel > config.FREQ_TOL_PCT:
                issues.append(
                    f"Q{i+1}: LC estimate {q['expected_GHz']:.2f} GHz is {rel:.0%} off "
                    f"the {q['target_GHz']:.2f} GHz target (Lj={q['Lj_nH']:.2f} nH)")
        if not (config.METAL_FRACTION_MIN <= frac <= config.METAL_FRACTION_MAX):
            issues.append(f"metal fraction {frac:.2%} outside sane band")

        per_q = "; ".join(
            f"Q{i+1} target {q['target_GHz']:.2f}→est {q['expected_GHz']:.2f} GHz "
            f"(Lj {q['Lj_nH']:.2f})" for i, q in enumerate(qubits))
        facts = (f"{len(qubits)}-qubit shared-feedline chip; metal fraction {frac:.3f}; "
                 f"per qubit: {per_q}.")
        return self._verdict(facts, issues, None)

    def _verdict(self, facts, issues, fix):
        if llm.available():
            v = llm.judge(
                "You are a superconducting-qubit mask reviewer. Given the rasterized "
                "mask facts, decide if the lumped-Lj design will reproduce the target "
                "qubit frequency and whether the mask is physically sane. Report issues "
                "concisely.", facts)
            return Verdict(self.name, v["ok"], v["issues"], v["rationale"],
                           fix if not v["ok"] else None, llm.backend_name())
        return Verdict(self.name, not issues, issues,
                       "mask + Lj reproduce the target within tolerance" if not issues else "",
                       fix, "rules")


class DrcCritic(Agent):
    """Manufacturability gate — KLayout design-rule check on the exported GDS
    (min metal width / spacing / etched gap / area, JJ sub-micron width)."""
    name = "drc-critic"

    def review(self, record: DesignRecord) -> Verdict:
        rep = record.metrics.get("drc")
        if rep is None or rep.get("skipped"):
            reason = (rep or {}).get("reason", "no DRC report")
            return Verdict(self.name, True, [], f"skipped — {reason}", None, "rules")

        n = rep.get("n_violations", len(rep.get("violations", [])))
        issues = []
        if not rep.get("passed", n == 0):
            by_rule = {}
            for v in rep.get("violations", []):
                by_rule[v.get("rule", "?")] = by_rule.get(v.get("rule", "?"), 0) + 1
            det = ", ".join(f"{k}×{c}" for k, c in sorted(by_rule.items()))
            issues.append(f"GDS violates fab design rules: {n} violation(s) ({det})")

        facts = (f"KLayout DRC on {record.prefix}.gds: "
                 f"{'PASSED, 0 violations' if not issues else issues[0]}.")
        if llm.available():
            v = llm.judge(
                "You are a fabrication design-rule reviewer for superconducting "
                "circuits. Decide whether this GDS is manufacturable as reported.",
                facts)
            return Verdict(self.name, v["ok"], v["issues"], v["rationale"], None,
                           llm.backend_name())
        return Verdict(self.name, not issues, issues,
                       "GDS passes all fab design rules" if not issues else "",
                       None, "rules")


class HpcCritic(Agent):
    """Resource / load-balance gate — will the requested GPUs be used sanely?
    Checks memory-per-GPU against the device limit and work-per-GPU against a
    communication-bound floor before the sbatch ships to the cluster."""
    name = "hpc-critic"

    BYTES_PER_CELL = 320            # 8 B × ~20 fields × 2 (old/new)
    GPU_MEM_B = 80e9                # Perlmutter A100 80GB
    USABLE = 0.7                    # usable fraction of device memory
    MIN_CELLS_PER_GPU = 5e5         # below this, ranks are communication-bound

    def review(self, record: DesignRecord) -> Verdict:
        h = record.metrics.get("hpc")
        if h is None:
            return Verdict(self.name, True, [], "skipped — no HPC job emitted",
                           None, "rules")
        cells = h["n_cellx"] * h["n_celly"] * h["n_cellz"]
        gpus = h["gpus"]
        mem_per_gpu = cells * self.BYTES_PER_CELL / gpus
        cells_per_gpu = cells / gpus
        limit = self.GPU_MEM_B * self.USABLE

        issues, fix = [], None
        if mem_per_gpu > limit:
            need = math.ceil(cells * self.BYTES_PER_CELL / limit)
            issues.append(
                f"grid needs {mem_per_gpu/1e9:.0f} GB/GPU on {gpus} GPUs "
                f"(> {limit/1e9:.0f} GB usable) — job would OOM on the cluster")
            fix = {"kind": "set_slurm_resources", "gpus": need,
                   "nodes": math.ceil(need / 4)}
        elif cells_per_gpu < self.MIN_CELLS_PER_GPU:
            need = max(1, math.ceil(cells / 5e6))
            issues.append(
                f"only {cells_per_gpu:,.0f} cells/GPU across {gpus} GPUs — "
                f"ranks are communication-bound; GPUs mostly idle")
            fix = {"kind": "set_slurm_resources", "gpus": need,
                   "nodes": math.ceil(need / 4)}

        facts = (f"grid {h['n_cellx']}×{h['n_celly']}×{h['n_cellz']} = {cells:.2e} cells; "
                 f"{h['nodes']} nodes / {gpus} GPUs; {mem_per_gpu/1e9:.1f} GB/GPU "
                 f"(limit {limit/1e9:.0f} GB); {cells_per_gpu:.2e} cells/GPU.")
        if llm.available():
            v = llm.judge(
                "You are an HPC job reviewer for distributed FDTD runs. Decide "
                "whether this domain decomposition is memory-safe and load-balanced "
                "across the requested GPUs.", facts)
            return Verdict(self.name, v["ok"], v["issues"], v["rationale"],
                           fix if not v["ok"] else None, llm.backend_name())
        return Verdict(self.name, not issues, issues,
                       f"memory + load balance sane ({mem_per_gpu/1e9:.1f} GB/GPU, "
                       f"{cells_per_gpu:.1e} cells/GPU)" if not issues else "",
                       fix, "rules")


class SimConfigCritic(Agent):
    """DP2 — is the Artemis excitation source clear of the PML boundary?"""
    name = "simconfig-critic"

    def _review_horizontal(self, record: DesignRecord) -> Verdict:
        """Input-port Ey source at a fixed x on a horizontal feedline: it must sit
        clear of both the left and right x-PML."""
        nx = record.metrics["n_cellx"]
        sx = record.metrics["source_x_um"]
        clearance = min(sx, nx - sx)                        # cells to nearest x boundary
        issues, fix = [], None
        if clearance < config.MIN_PML_CLEARANCE_CELLS:
            issues.append(
                f"input-port source at x={sx:.0f} is only {clearance:.0f} cells from an "
                f"x boundary (need ≥ {config.MIN_PML_CLEARANCE_CELLS}); it sits in the PML")
        facts = (f"grid n_cellx = {nx}; source_x = {sx:.0f}; "
                 f"clearance from nearest x boundary = {clearance:.0f} cells; "
                 f"minimum required = {config.MIN_PML_CLEARANCE_CELLS} cells.")
        if llm.available():
            v = llm.judge(
                "You are an FDTD simulation-setup reviewer. Decide whether the Artemis "
                "input-port excitation on the horizontal feedline is safely inside the "
                "domain, clear of the PML absorbing boundary.", facts)
            return Verdict(self.name, v["ok"], v["issues"], v["rationale"], fix, llm.backend_name())
        return Verdict(self.name, not issues, issues,
                       "input-port source is clear of the PML" if not issues else "",
                       fix, "rules")

    def review(self, record: DesignRecord) -> Verdict:
        if record.metrics.get("source_x_um") is not None:
            return self._review_horizontal(record)      # qubit_resonator / multi
        if record.metrics.get("source_y_um") is None:
            return Verdict(self.name, True, [],
                           f"skipped — {record.mode} mode has no Artemis input", None, "rules")
        ny = record.metrics["n_celly"]
        source_y = record.metrics["source_y_um"]
        clearance = ny - source_y                          # cells from the top boundary

        issues, fix = [], None
        if clearance < config.MIN_PML_CLEARANCE_CELLS:
            issues.append(
                f"excitation source at y={source_y:.0f} is only {clearance:.0f} cells "
                f"from the top boundary (need ≥ {config.MIN_PML_CLEARANCE_CELLS}); it sits "
                f"inside the PML and the run would be wasted")
            # Recover to one full PML margin (100 cells) below the boundary.
            fix = {"kind": "set_source_y", "source_y_um": float(ny - 100)}

        facts = (f"grid n_celly = {ny}; source_y = {source_y:.0f}; "
                 f"clearance from top boundary = {clearance:.0f} cells; "
                 f"minimum required = {config.MIN_PML_CLEARANCE_CELLS} cells.")
        if llm.available():
            v = llm.judge(
                "You are an FDTD simulation-setup reviewer. Decide whether the Artemis "
                "excitation source is safely inside the domain, clear of the PML absorbing "
                "boundary. A source in the PML wastes the whole GPU run.", facts)
            return Verdict(self.name, v["ok"], v["issues"], v["rationale"],
                           fix if not v["ok"] else None, llm.backend_name())
        return Verdict(self.name, not issues, issues,
                       "source is safely clear of the PML" if not issues else "",
                       fix, "rules")
