"""M12 §3 — Streamlit demo: two teams, a format, the model's probability.

Run from repo root:  .venv/Scripts/python.exe -m streamlit run src/cs2analytics/serve/dashboard.py
Ships with artifacts only (model.pkl + features.json + elo_ratings.json, git-ignored);
the match data itself is not shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "artifacts"
FIG = REPO / "outputs" / "fig_calibration.png"


@st.cache_resource
def load_everything() -> dict:
    model = joblib.load(ARTIFACTS / "model.pkl")
    meta = json.loads((ARTIFACTS / "features.json").read_text(encoding="utf-8"))
    elo = json.loads((ARTIFACTS / "elo_ratings.json").read_text(encoding="utf-8"))
    teams = sorted(elo.keys())
    return {"model": model, "meta": meta, "elo": elo, "teams": teams}


def main() -> None:
    st.set_page_config(page_title="CS2 Match Analytics", page_icon="🎯", layout="wide")
    st.title("CS2 Match Analytics — win probability demo")
    st.caption(
        "Calibrated logistic model on Elo gap + form/rest/h2h features. "
        "Unknown teams fall back to a neutral 1500 rating."
    )

    art = load_everything()
    teams = art["teams"]

    col1, col2, col3 = st.columns([2, 2, 1])
    default_t1 = teams.index("Natus Vincere") if "Natus Vincere" in teams else 0
    default_t2 = teams.index("FaZe Clan") if "FaZe Clan" in teams else 1
    with col1:
        team1 = st.selectbox("Team A", teams, index=default_t1)
    with col2:
        team2 = st.selectbox("Team B", teams, index=default_t2)
    with col3:
        best_of = st.selectbox("Format", (1, 3, 5), index=1)

    if st.button("Predict", type="primary") or True:
        if team1 == team2:
            st.warning("Pick two different teams.")
        else:
            elo = art["elo"]
            meta = art["meta"]
            e1, e2 = elo.get(team1, 1500.0), elo.get(team2, 1500.0)
            row = {}
            for name in meta["feature_names"]:
                if name == "elo_diff":
                    row[name] = e1 - e2
                elif name == "is_bo1":
                    row[name] = 1.0 if best_of == 1 else 0.0
                else:
                    row[name] = 0.0
            X = np.array([[row[n] for n in meta["feature_names"]]])
            X_swap = X.copy()
            X_swap[0, meta["feature_names"].index("elo_diff")] *= -1.0
            p_raw = float(art["model"].predict_proba(X)[0, 1])
            p_swap = float(art["model"].predict_proba(X_swap)[0, 1])
            p = (p_raw + (1.0 - p_swap)) / 2.0

            c1, c2 = st.columns(2)
            c1.metric(f"{team1} (Elo {e1:.0f})", f"{p:.1%}")
            c2.metric(f"{team2} (Elo {e2:.0f})", f"{1 - p:.1%}")
            st.progress(p if p >= 0.5 else 1 - p, text=f"model version: {meta['model_version']}")

    st.divider()
    st.subheader("Calibration (the money chart)")
    if FIG.exists():
        st.image(str(FIG), use_container_width=True)
        st.caption(
            "Predicted probability vs observed win rate on the 2026 time-split test set. "
            "The diagonal is perfect calibration."
        )
    else:
        st.info("Run the M7 notebook first to generate outputs/fig_calibration.png")

    if st.checkbox("Show model comparison table"):
        comp_path = REPO / "outputs" / "m8_model_comparison.csv"
        if comp_path.exists():
            st.dataframe(pd.read_csv(comp_path))
        else:
            st.info("Run the M8 notebook first.")


if __name__ == "__main__":
    main()
