"""
Orchestrator — the coordinator.

Runs the designer, then gates each decision point with its critic. On a FAIL it
hands the critic's concrete fix back to the designer, which revises, and the
critic re-reviews — the self-correcting loop that is the poster's headline.
It never runs the FDTD job; it stops at a validated Artemis input + sbatch.
"""
from __future__ import annotations

from typing import List, Tuple

import config
from blackboard import DesignRecord
from .base import Verdict
from .critics import DrcCritic, MaskCritic, SimConfigCritic
from .designer import DesignerAgent


class Orchestrator:
    def __init__(self, designer: DesignerAgent, max_revisions: int = config.MAX_REVISIONS,
                 on_event=None):
        self.designer = designer
        self.max_revisions = max_revisions
        # decision-point label -> critic
        self.checkpoints = [("Decision Point 1 · mask structure", MaskCritic()),
                            ("Decision Point 1b · fab DRC",        DrcCritic()),
                            ("Decision Point 2 · sim config",      SimConfigCritic())]
        self._emit = on_event or (lambda *_: None)

    def run(self, record: DesignRecord) -> Tuple[DesignRecord, bool]:
        self._emit("phase", "Designer builds the artifact chain")
        self.designer.design(record)

        all_ok = True
        for label, critic in self.checkpoints:
            ok = self._gate(record, label, critic)
            all_ok = all_ok and ok
        return record, all_ok

    def _gate(self, record: DesignRecord, label: str, critic) -> bool:
        self._emit("phase", label)
        for attempt in range(self.max_revisions + 1):
            v = critic.review(record)
            self._emit("verdict", v)
            record.log(v.summary())
            if v.ok:
                return True
            if attempt == self.max_revisions or not v.fix:
                record.log(f"orchestrator: {critic.name} unresolved after "
                           f"{attempt} revision(s)")
                return False
            self._emit("revise", v.fix)
            self.designer.revise(record, v.fix)
        return False
