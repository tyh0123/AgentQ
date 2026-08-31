"""
Deliberate fault injectors for the fail -> detect -> recover demo.

These make the *designer* produce a subtly-wrong artifact, exactly the kind of
config error that would otherwise waste GPU-days on the HPC run. Each fault is
caught by a specific critic at a specific decision point:

    "source_in_pml"  -> SimConfigCritic (Decision Point 2)
    "bad_lj"         -> MaskCritic       (Decision Point 1)
"""
from __future__ import annotations

from typing import Set

VALID = {"source_in_pml", "bad_lj"}


def parse(spec: str) -> Set[str]:
    if not spec or spec == "none":
        return set()
    if spec == "both":
        return set(VALID)
    faults = {f.strip() for f in spec.split(",") if f.strip()}
    bad = faults - VALID
    if bad:
        raise ValueError(f"unknown fault(s) {bad}; valid: {sorted(VALID)} / none / both")
    return faults
