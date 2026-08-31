"""
SQuADDS database query with retry-on-null-coupler fallback.

The SQuADDS database has some rows where coupler fields parse to null.
A naive find_closest(num_top=1) can land on such a row and break downstream
launchpad / meander construction. This module walks the top-N candidates
and returns the first usable design.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from squadds import Analyzer, SQuADDS_DB

from qpipe_config import Config, SquaddsConfig


# Fields the layout actually needs. If any of these is None for a candidate row,
# we skip it. Coupler "second_*" entries fall back to defaults (validation mode
# has no real coupler), so they're not required.
REQUIRED_QUBIT_FIELDS = (
    "cross_length", "cross_width", "cross_gap",
    "claw_length", "claw_width", "claw_gap",
    "ground_spacing",
)


def query(cfg: Config, overrides: Optional[Dict[str, str]] = None) -> Tuple[dict, dict, float]:
    """Query SQuADDS for the closest match to cfg.target.

    Workflow:
      1. Run find_closest against the DB.
      2. If the best usable row is within distance thresholds → use DB row.
      3. Otherwise, if cfg.squadds.ml_fallback.enabled → call the ML inverse
         model + analytic L_J formula. Coupler/cavity info is lost on this path.

    Returns (dq, dk, Lj_nH) shaped identically regardless of which source ran.
    The source ("squadds_db" or "ml_fallback") is printed to stdout.
    """
    overrides = overrides or {}
    sq = cfg.squadds

    db = SQuADDS_DB()
    db.select_system(["cavity_claw", "qubit"])
    db.select_qubit(sq.qubit_type)
    db.select_cavity_claw(sq.cavity_type)
    db.select_resonator_type(sq.resonator_type)
    db.create_system_df()

    analyzer = Analyzer(db)
    target = {
        "qubit_frequency_GHz": cfg.target.qubit_frequency_GHz,
        "anharmonicity_MHz": cfg.target.anharmonicity_MHz,
        "cavity_frequency_GHz": cfg.target.cavity_frequency_GHz,
        "g_MHz": cfg.target.g_MHz,
    }
    results = analyzer.find_closest(target, num_top=sq.num_top, metric=sq.metric)

    # Walk candidates to find one with usable qubit fields
    chosen_idx = _pick_usable_row(analyzer, results)
    db_failure_reason = None
    if chosen_idx is None:
        db_failure_reason = (
            f"None of the top-{sq.num_top} SQuADDS matches has complete qubit geometry"
        )

    # Distance check against thresholds
    if chosen_idx is not None:
        one = results.iloc[[chosen_idx]]
        dfreq = float(abs(one["qubit_frequency_GHz"].iloc[0] - cfg.target.qubit_frequency_GHz))
        danharm = float(abs(one["anharmonicity_MHz"].iloc[0] - cfg.target.anharmonicity_MHz))
        too_far = (
            dfreq > sq.ml_fallback.freq_GHz_threshold
            or danharm > sq.ml_fallback.anharm_MHz_threshold
        )
        if too_far:
            db_failure_reason = (
                f"DB best match is too far from target "
                f"(Δfreq={dfreq:.3f} GHz, Δanharm={danharm:.2f} MHz; "
                f"thresholds {sq.ml_fallback.freq_GHz_threshold} GHz / "
                f"{sq.ml_fallback.anharm_MHz_threshold} MHz)"
            )

    if db_failure_reason is not None:
        if not sq.ml_fallback.enabled:
            raise RuntimeError(
                db_failure_reason + ". ML fallback is disabled. "
                "Enable it via squadds.ml_fallback.enabled or relax thresholds."
            )
        # Local import keeps httpx out of the DB-only path.
        from qpipe_ml_api import query_ml_fallback

        print(f"  source: ml_fallback ({db_failure_reason})")
        print("  note: cavity coupling (g_MHz, cavity_frequency_GHz) not modeled by ML")
        dq, dk, Lj = query_ml_fallback(cfg)
        # Apply user overrides on top of ML output (same semantics as DB path)
        applied = []
        for key, val in overrides.items():
            if key in dq:
                dq[key] = [val]
                applied.append(f"{key}={val}")
            else:
                print(f"  warning: override key '{key}' not in qubit options (ignored)")
        if applied:
            print(f"  overrides applied: {', '.join(applied)}")
        return dq, dk, Lj

    print(
        f"  source: squadds_db "
        f"(Δfreq={dfreq:.3f} GHz, Δanharm={danharm:.2f} MHz)"
    )
    one = results.iloc[[chosen_idx]]
    dq = analyzer.get_qubit_options(one)
    dk = analyzer.get_coupler_options(one)
    LJs = analyzer.get_Ljs(one)

    # Coupler fallback: validation mode doesn't need a real coupler, but the
    # launchpad still wants trace_width/gap. Use defaults from config.
    if dk.get("second_width", [None])[0] is None:
        dk["second_width"] = [sq.fallback_cpw.second_width]
    if dk.get("second_gap", [None])[0] is None:
        dk["second_gap"] = [sq.fallback_cpw.second_gap]

    # User overrides on top of SQuADDS values
    applied = []
    for key, val in overrides.items():
        if key in dq:
            dq[key] = [val]
            applied.append(f"{key}={val}")
        else:
            print(f"  warning: override key '{key}' not in qubit options (ignored)")
    if applied:
        print(f"  overrides applied: {', '.join(applied)}")

    return dq, dk, float(LJs[0])


def _pick_usable_row(analyzer, results) -> Optional[int]:
    """Return the index in `results` of the first row whose qubit fields are
    all populated. Returns None if no row qualifies."""
    for i in range(len(results)):
        one = results.iloc[[i]]
        try:
            dq = analyzer.get_qubit_options(one)
        except Exception as e:
            print(f"  skip row {i}: get_qubit_options failed ({e})")
            continue
        missing = [k for k in REQUIRED_QUBIT_FIELDS
                   if not dq.get(k) or dq[k][0] is None]
        if missing:
            print(f"  skip row {i}: missing {missing}")
            continue
        return i
    return None


def summarize(dq: dict, dk: dict, Lj: float) -> str:
    """Human-readable one-block summary of the chosen design."""
    lines = [f"  Lj = {Lj:.3f} nH"]
    for k in REQUIRED_QUBIT_FIELDS:
        lines.append(f"  {k} = {dq[k][0]}")
    lines.append(f"  cpw_trace_width = {dk['second_width'][0]}")
    lines.append(f"  cpw_trace_gap = {dk['second_gap'][0]}")
    return "\n".join(lines)
