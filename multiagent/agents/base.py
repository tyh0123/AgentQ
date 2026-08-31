"""Shared agent primitives: the Verdict a critic returns, and the Agent base."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Verdict:
    critic: str                                   # which critic
    ok: bool                                       # passed?
    issues: List[str] = field(default_factory=list)
    rationale: str = ""
    fix: Optional[Dict[str, Any]] = None           # concrete corrective action for the designer
    backend: str = "rules"                         # "Claude (...)" or "rules"

    def summary(self) -> str:
        tag = "PASS" if self.ok else "FAIL"
        head = f"[{self.critic}/{tag} · {self.backend}]"
        if self.ok:
            return f"{head} {self.rationale}".rstrip()
        return f"{head} " + "; ".join(self.issues)


class Agent:
    """Base class: a named participant in the session."""

    name = "agent"

    def __init__(self, name: Optional[str] = None):
        if name:
            self.name = name
