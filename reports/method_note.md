# Method Note — CS2 Match Analytics

**Problem.** Predict the winner of a professional CS2 series *before it is played*, as a
probability — not a guess. The deliverable is a calibrated model with honest validation,
plus the ratings machinery (Elo, Bayesian) that makes the numbers interpretable.

**Data.** 9,920 professional series (2023–2026) across three competitive tiers, from a
public Kaggle dump. The raw file looks simple and is not: the series-row score columns
are sorted by *team column*, not by winner; the series-row `team1_win` flag is broken
(94% zeros) and contradicts both the map-level evidence and two externally verifiable
Major finals (NAVI 2-1 FaZe, Spirit 2-1 FaZe), which pinned the correct reading; ~20% of
"best-of-three" rows are actually single-map series; forfeits leave 0-0 map rows. Every
quirk is documented in DATA.md and pinned by contract tests. Two corrupt rows were
dropped (9,923 → 9,920).

**Method.** Four layers, each tested: (1) an Elo engine written from scratch, with the
update rule derived as one gradient step on logistic log-loss and replayed walk-forward
(pre-match ratings only); (2) a feature store of strictly pre-match signals — Elo gap,
rolling 5-series form, rest days, head-to-head share, format flag — guarded so any
post-outcome column raises a LeakageError; (3) models on a fixed time split: logistic
regression, gradient boosting, a team-embedding network, and isotonic calibration; and
(4) evaluation: reliability diagrams, expected calibration error, Brier decomposition,
and a leakage demo that shows what random splits would have lied about.

**Results.** On the 2026 hold-out (1,811 series): logistic with features 0.6464 logloss
(61.9% accuracy), from-scratch Elo 0.6472, GBM 0.6511, the embedding net 0.6632, versus
0.6931 for always-0.5. Logistic beats Elo only slightly — because Elo already is a
logistic model on its own feature — and the neural model loses outright: at this sample
size the signal is nearly linear in rating difference. The K-sweep peaks exactly at the
conventional K=32, and Bayesian ratings agree with Elo's ranking while adding credible
intervals (Vitality top at ~1966, matching their real 2025 form).

**Limitations.** No usable odds data, so this is a calibration study against outcomes,
not a market-efficiency test. Bo1s are a different regime that the format flag only
partly captures. Roster and stand-in effects are unmodeled despite being present in the
data. Tier-3 forfeit noise leaks through in ~1.5% of map rows.

**What I'd do next.** Roster-stability features from the lineup columns; per-regime
calibration (Bo1/Bo3, tier); a small odds dataset for the model-vs-market chart; and
weekly refits triggered by the PSI drift report that already ships with the serving API.