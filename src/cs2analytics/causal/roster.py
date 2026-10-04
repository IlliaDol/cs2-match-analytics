"""Roster-change cohorts for causal, not predictive, analysis.

The functions in this module never use an outcome to define treatment. A roster
change is detected only by comparing a team's current known five-player set with
its previous known set. Unknown lineups remain unknown; they are never treated as
"no change".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RosterColumns:
    """Column naming contract for the two sides of a match."""

    player_template: str = "{side}_player{i}_id"

    def names(self, side: str) -> list[str]:
        return [self.player_template.format(side=side, i=i) for i in range(1, 6)]


def _lineup(row: pd.Series, side: str, columns: RosterColumns) -> frozenset[str] | None:
    names = columns.names(side)
    if not all(name in row.index for name in names):
        return None
    values = [row[name] for name in names]
    if any(pd.isna(value) for value in values):
        return None
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized) or len(set(normalized)) != 5:
        return None
    return frozenset(normalized)


def _round_diff(row: pd.Series, side: str) -> float:
    """Return a team-oriented series score difference when available."""
    if side == "team1":
        left, right = "t1_series_score", "t2_series_score"
    else:
        left, right = "t2_series_score", "t1_series_score"
    if left not in row.index or right not in row.index:
        return float("nan")
    values = pd.to_numeric(pd.Series([row[left], row[right]]), errors="coerce")
    if values.isna().any():
        return float("nan")
    return float(values.iloc[0] - values.iloc[1])


def build_team_match_panel(
    matches: pd.DataFrame,
    *,
    columns: RosterColumns | None = None,
) -> pd.DataFrame:
    """Convert one-row-per-series data into one-row-per-team-per-series data.

    Required columns: ``match_id``, ``datetime``, ``team1``, ``team2``, and
    ``winner``. Lineup columns are optional, but without them no roster event can
    be identified. The returned panel includes pre-treatment history counters:

    ``pre_stable_matches`` counts consecutive prior known matches with the same
    lineup; ``roster_changed`` is true only when both current and previous lineups
    are known and differ; ``change_count`` is the number of replaced players.
    """
    required = {"match_id", "datetime", "team1", "team2", "winner"}
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"missing columns for roster panel: {missing}")

    lineup_columns = columns or RosterColumns()
    source = matches.copy()
    source["datetime"] = pd.to_datetime(source["datetime"], utc=True, errors="coerce")
    if source["datetime"].isna().any():
        raise ValueError("roster panel requires parseable datetimes")
    source = source.sort_values(["datetime", "match_id"], kind="mergesort").reset_index(drop=True)

    previous: dict[str, frozenset[str]] = {}
    stable_run: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    for row in source.to_dict("records"):
        row_series = pd.Series(row)
        for side in ("team1", "team2"):
            team = str(row[side])
            current = _lineup(row_series, side, lineup_columns)
            prior = previous.get(team)
            changed = current is not None and prior is not None and current != prior
            if current is None:
                run_before = stable_run.get(team, 0)
            else:
                run_before = stable_run.get(team, 0)
            if changed:
                change_count = len(current - prior)  # type: ignore[operator]
                stable_before = run_before
                stable_run[team] = 0
            elif current is not None and prior is not None:
                change_count = 0
                stable_before = run_before
                stable_run[team] = run_before + 1
            elif current is not None:
                change_count = np.nan
                stable_before = run_before
                stable_run[team] = 0
            else:
                change_count = np.nan
                stable_before = run_before

            if current is not None:
                previous[team] = current
            else:
                # A gap in lineup observability breaks continuity. Do not let a
                # later known lineup be compared against stale information.
                previous.pop(team, None)
                stable_run[team] = 0

            rows.append(
                {
                    "match_id": row["match_id"],
                    "datetime": row["datetime"],
                    "team": team,
                    "opponent": str(row["team2" if side == "team1" else "team1"]),
                    "side": side,
                    "winner": row["winner"],
                    "win": float(row["winner"] == team),
                    "round_diff": _round_diff(row_series, side),
                    "tier": row.get("tier", np.nan),
                    "is_bo1": row.get("games_played", np.nan) == 1,
                    "roster": current,
                    "previous_roster": prior,
                    "roster_known": current is not None,
                    "roster_changed": bool(changed),
                    "change_count": change_count,
                    "pre_stable_matches": int(stable_before),
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values(["team", "datetime", "match_id"], kind="mergesort").reset_index(
        drop=True
    )


def build_event_windows(
    team_panel: pd.DataFrame,
    *,
    pre_window: int = 5,
    post_window: int = 5,
    min_pre_stable: int = 3,
    max_post_changes: int = 0,
) -> pd.DataFrame:
    """Create treated event windows aligned by team-relative match index.

    Event time zero is the first series with the changed lineup. Events require
    enough preceding stable matches and enough subsequent matches. By default a
    second roster change inside the post window excludes the event, avoiding a
    contaminated treatment window. The function returns treated windows only;
    controls are selected separately by :func:`nearest_control_matches`.
    """
    if pre_window < 1 or post_window < 1:
        raise ValueError("pre_window and post_window must be positive")
    required = {"team", "datetime", "roster_changed", "pre_stable_matches"}
    missing = sorted(required - set(team_panel.columns))
    if missing:
        raise ValueError(f"missing columns for event windows: {missing}")

    panel = team_panel.sort_values(["team", "datetime", "match_id"], kind="mergesort").copy()
    rows: list[pd.DataFrame] = []
    event_counter = 0
    for team, history in panel.groupby("team", sort=False):
        history = history.reset_index(drop=True)
        change_indices = history.index[
            history["roster_changed"].astype(bool)
            & (history["pre_stable_matches"] >= min_pre_stable)
        ].tolist()
        for index in change_indices:
            start = index - pre_window
            stop = index + post_window
            if start < 0 or stop >= len(history):
                continue
            window = history.iloc[start : stop + 1].copy()
            if int(window.iloc[pre_window + 1 :]["roster_changed"].sum()) > max_post_changes:
                continue
            event_counter += 1
            window["event_id"] = f"event-{event_counter:05d}"
            window["relative_match"] = np.arange(-pre_window, post_window + 1)
            window["treated"] = 1
            window["post"] = (window["relative_match"] >= 0).astype(int)
            window["event_team"] = team
            rows.append(window)

    if not rows:
        return pd.DataFrame(
            columns=[*panel.columns, "event_id", "relative_match", "treated", "post", "event_team"]
        )
    return pd.concat(rows, ignore_index=True)


def _candidate_score(treated: pd.Series, control: pd.Series, covariates: list[str]) -> float:
    distances: list[float] = []
    for column in covariates:
        a = pd.to_numeric(pd.Series([treated[column]]), errors="coerce").iloc[0]
        b = pd.to_numeric(pd.Series([control[column]]), errors="coerce").iloc[0]
        if pd.isna(a) or pd.isna(b):
            return float("inf")
        distances.append(float(a - b))
    return float(np.sqrt(np.mean(np.square(distances)))) if distances else 0.0


def nearest_control_matches(
    team_panel: pd.DataFrame,
    treated_events: pd.DataFrame,
    *,
    covariates: Iterable[str] = ("pre_stable_matches",),
    max_date_days: int = 90,
    require_same_tier: bool = True,
    pre_window: int = 5,
    post_window: int = 5,
) -> pd.DataFrame:
    """Match each treated event to one deterministic untreated event point.

    Candidate controls must have no roster change at the candidate point, a known
    lineup, and the same tier when both datasets expose a tier. Matching is
    greedy and without replacement; it is a transparent baseline, not a claim
    that matching solves unobserved confounding. A candidate is eligible only if
    its complete pre/post window contains no roster change.
    """
    needed = {"event_id", "team", "datetime", "relative_match"}
    if not needed.issubset(treated_events.columns):
        raise ValueError(f"treated_events missing {sorted(needed - set(treated_events.columns))}")
    if pre_window < 1 or post_window < 1:
        raise ValueError("pre_window and post_window must be positive")
    covariate_names = list(covariates)
    all_panel = team_panel.sort_values(
        ["team", "datetime", "match_id"], kind="mergesort"
    ).copy()
    eligible_controls: set[tuple[object, object]] = set()
    for team, history in all_panel.groupby("team", sort=False):
        history = history.reset_index(drop=True)
        for position, candidate in history.iterrows():
            start = position - pre_window
            stop = position + post_window
            if start < 0 or stop >= len(history):
                continue
            window = history.iloc[start : stop + 1]
            if (
                candidate["roster_known"]
                and not bool(candidate["roster_changed"])
                and not window["roster_changed"].astype(bool).any()
            ):
                eligible_controls.add((team, candidate["match_id"]))
    base = all_panel.loc[
        [
            (row["team"], row["match_id"]) in eligible_controls
            for _, row in all_panel.iterrows()
        ]
    ]
    used: set[tuple[str, object]] = set()
    result: list[dict[str, object]] = []

    for _event_index, event in treated_events[treated_events["relative_match"] == 0].iterrows():
        treated_time = pd.Timestamp(event["datetime"])
        candidates = base[base["team"] != event["team"]].copy()
        candidates["date_distance"] = (candidates["datetime"] - treated_time).abs().dt.days
        candidates = candidates[candidates["date_distance"] <= max_date_days]
        if (
            require_same_tier
            and "tier" in candidates
            and "tier" in event.index
            and not pd.isna(event["tier"])
        ):
            candidates = candidates[candidates["tier"] == event["tier"]]
        candidates = candidates[
            ~candidates.apply(
                lambda candidate: (candidate["team"], candidate["match_id"]) in used,
                axis=1,
            )
        ]
        if candidates.empty:
            continue
        candidates = candidates.copy()
        treated_event = event.copy()
        candidates["distance"] = [
            _candidate_score(treated_event, candidate, covariate_names)
            for _, candidate in candidates.iterrows()
        ]
        candidates = candidates.sort_values(
            ["distance", "date_distance", "team", "match_id"], kind="mergesort"
        )
        candidate = candidates.iloc[0]
        if not np.isfinite(float(candidate["distance"])):
            continue
        used.add((candidate["team"], candidate["match_id"]))
        result.append(
            {
                "event_id": event["event_id"],
                "treated_team": event["team"],
                "control_team": candidate["team"],
                "treated_match_id": event["match_id"],
                "control_match_id": candidate["match_id"],
                "treated_datetime": event["datetime"],
                "control_datetime": candidate["datetime"],
                "distance": float(candidate["distance"]),
            }
        )
    return pd.DataFrame(result)


def build_paired_event_panel(
    team_panel: pd.DataFrame,
    treated_windows: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    pre_window: int = 5,
    post_window: int = 5,
) -> pd.DataFrame:
    """Align matched control windows to the treated event-relative timeline."""
    if matches.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for _, pair in matches.iterrows():
        event_id = pair["event_id"]
        treated = treated_windows[treated_windows["event_id"] == event_id].copy()
        if treated.empty:
            continue
        control = team_panel[team_panel["team"] == pair["control_team"]].copy()
        control["date_distance"] = (control["datetime"] - pair["control_datetime"]).abs().dt.days
        control = control.sort_values(["datetime", "match_id"], kind="mergesort")
        anchor = control.index[control["match_id"] == pair["control_match_id"]]
        if len(anchor) != 1:
            continue
        position = control.index.get_loc(anchor[0])
        start, stop = position - pre_window, position + post_window
        if start < 0 or stop >= len(control):
            continue
        control_window = control.iloc[start : stop + 1].copy()
        if len(control_window) != pre_window + post_window + 1:
            continue
        control_window["event_id"] = event_id
        control_window["relative_match"] = np.arange(-pre_window, post_window + 1)
        control_window["treated"] = 0
        control_window["post"] = (control_window["relative_match"] >= 0).astype(int)
        control_window["event_team"] = treated["event_team"].iloc[0]
        rows.extend([treated, control_window])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
