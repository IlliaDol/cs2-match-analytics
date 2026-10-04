"""Causal analysis utilities for roster-change studies."""

from .estimators import balance_table, bootstrap_did, did_estimate, event_study_table
from .roster import (
    build_event_windows,
    build_paired_event_panel,
    build_team_match_panel,
    nearest_control_matches,
)

__all__ = [
    "balance_table",
    "bootstrap_did",
    "build_event_windows",
    "build_paired_event_panel",
    "build_team_match_panel",
    "did_estimate",
    "event_study_table",
    "nearest_control_matches",
]
