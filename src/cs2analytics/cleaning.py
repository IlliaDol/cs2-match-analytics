"""M6 §2 — cleaning functions: the decision log as code.

Pure functions, no silent mutation (all return copies, drops are logged).
Each function documents its "what does this cost me" trade-off.
"""

from __future__ import annotations

import pandas as pd


def deduplicate_series(df: pd.DataFrame) -> pd.DataFrame:
    """Unique match_id (keep first), returns a copy; logs how many were dropped.

    Cost: duplicate rows (same match re-reported) would double-count matches in
    win-rate aggregates; keeping the first is safe because the loader guarantees
    identical content across duplicates.
    """
    out = df.copy()
    n_before = len(out)
    out = out.drop_duplicates(subset="match_id", keep="first")
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"[cleaning] deduplicate_series: dropped {n_dropped} duplicate match_id rows")
    return out.reset_index(drop=True)


def normalize_team_names(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Canonicalize team-name columns via a lowercase-lookup mapping.

    Only names present as keys in `mapping` (lowercased) are rewritten; unknown
    names pass through untouched. Cost: an over-aggressive mapping could merge
    two genuinely different teams; the mapping is therefore kept small and
    manual (org renames only), never auto-generated.
    """
    out = df.copy()
    lowered = {k.strip().lower(): v for k, v in mapping.items()}
    for col in ("team1", "team2", "winner"):
        if col in out.columns:
            out[col] = (
                out[col]
                .map(lambda name: lowered.get(str(name).strip().lower(), name))
                .astype(out[col].dtype)
            )
    return out


def flag_forfeits(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_forfeit: True where the series ended without real play.

    Signal: games_played >= 1 but both series scores are 0 (a 'played' match
    with zero maps won by either side), or missing series scores entirely.
    Cost of dropping them: lose walkover/forfeit signal; cost of keeping:
    a 0-0 row has no meaningful margin/round info. Default stance: FLAG, let
    the model layer decide.
    """
    out = df.copy()
    s1 = pd.to_numeric(out.get("t1_series_score"), errors="coerce")
    s2 = pd.to_numeric(out.get("t2_series_score"), errors="coerce")
    gp = pd.to_numeric(out.get("games_played"), errors="coerce")
    out["is_forfeit"] = (s1.fillna(0) == 0) & (s2.fillna(0) == 0) & (gp.fillna(0) >= 1)
    return out


def winsorize_round_scores(df: pd.DataFrame, lo: int = 0, hi: int = 25) -> pd.DataFrame:
    """Clip series score columns into [lo, hi]; row count and winner untouched.

    Cost: clipping corrupts genuinely extreme-but-real scores (none exist in
    this dataset: series scores max at 3) but protects downstream features from
    absurd outliers if the score columns are misused as round counts.
    """
    out = df.copy()
    for col in ("t1_series_score", "t2_series_score"):
        if col in out.columns:
            numeric = pd.to_numeric(out[col], errors="coerce")
            out[col] = numeric.clip(lower=lo, upper=hi).astype("Int64")
    return out
