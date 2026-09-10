# M12 BUILD SPEC — Portfolio capstone: ship it  · `DA DS`

Nothing new gets built here. **This module is packaging, writing, and the stranger test.**
Everything already exists; M12 makes a recruiter understand it in 90 seconds.

---

## §1 `README.md` — the rewrite (yours). Required structure, in this order:

1. **One-sentence pitch** (max 2 lines):
   *"A from-scratch probabilistic model of professional CS2 match outcomes — Elo baseline,
   calibrated logistic/GBM, validated walk-forward — built as a walkthrough of the whole
   data-science stack."*
2. **The money chart inline** (`![calibration](outputs/fig_calibration.png)`) — ABOVE the fold.
3. **Headline results table** — copy `outputs/m7_model_comparison.csv` (incl. the `constant_0.5`
   baseline; showing the baseline is what makes the numbers credible).
4. **Method, 5 bullets max:** data → Elo (time-ordered replay) → features → model →
   calibration. One line each, no equations here.
5. **Honest limitations** — 4 bullets minimum, from your own notebooks:
   the odds comparison is v1-subset only; Bo1s are a different regime; roster/standin
   features not yet used; tier-3 forfeit noise.
6. **Reproducibility** — exact commands to go from clone to `outputs/fig_calibration.png`,
   including `pip install -e ".[dev]"` and the M4/M6/M7 scripts in order.
7. **Data & ethics** — one short paragraph: Kaggle source, git-ignored, gambling-adjacent
   use = calibration study not betting advice.

## §2 `reports/method_note.md` — the 1-page method note (yours)
Sections: Problem · Data (with the quirks that mattered) · Method · Results · Limitations ·
What I'd do next. **Exactly one page** (≤ 600 words). No equations longer than a single line.

## §3 `src/cs2analytics/serve/dashboard.py` — Streamlit demo (yours)
`pip install streamlit`. Minimal: two team dropdowns (from `outputs/series_clean.csv`),
a best-of selector, a "predict" button hitting your local model, and the calibration chart
rendered below. ~80 lines. Deploy to HF Spaces (free) or a Render static/web service; put
the URL in the README badge row.

## §4 `docs/presentation.md` — the walkthrough script (yours)
Word-for-word, 3 minutes, the doc you read when someone says "tell me about this project".
Structure: problem → one quirk that shows you touched real data → the money chart →
the honest limitation → what you'd do next. Write it as SPOKEN text.

## §5 Tests (`tests/test_m12_readme.py`)
- README contains all 7 section headings in order (parse with `re` on `^## `)
- README embeds the calibration PNG (`fig_calibration.png` appears in an image tag)
- README has a results table containing `constant_0.5`
- `reports/method_note.md` exists and is 300–700 words
- `docs/presentation.md` exists and is ≥ 300 words
- README links to no placeholder text: no `TODO`, `lorem`, `<insert`

## §6 Final housekeeping
- Pin the repo on GitHub; set topics: `cs2`, `elo`, `calibration`, `machine-learning`,
  `sports-analytics`, `duckdb`, `mlops`.
- `git tag v1.0.0 && git push --tags`.
- Add the CI badge and (if deployed) the demo URL at the very top of the README.

## Checkpoint — the stranger test (do this literally)
Send the repo link to someone who does not know the project. Ask two questions:
1. "What does this model predict, and how well?"
2. "What's the most suspicious thing about it?"
If they can't answer #1 in 60 seconds, the README failed — fix #1 and #3, not the model.
Their answer to #2 is free feedback for your Limitations section.

## Appendix — hints
- Section heading check: `[l for l in readme.splitlines() if l.startswith("## ")]`.
- Keep the results table as a Markdown copy — don't link the CSV for the headline numbers
  (recruiters don't download files).
- If the Streamlit app needs the DuckDB file too, ship only what's needed: model.pkl +
  features.json + elo_ratings.json, and say the data isn't shipped.
