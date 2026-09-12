# Presentation — the 3-minute walkthrough script

Say this, in this order, when someone asks "tell me about this project".

---

So — this project predicts professional Counter-Strike 2 match outcomes as probabilities,
before the match is played. Not "who wins" — a number, like 0.62, that's meant to be
honest about its own uncertainty.

Why build it at all? Because it's a complete data-science stack in miniature: a real,
messy dataset; a rating model derived from first principles; proper walk-forward
validation; and a calibration story. Every step has contract tests that fail loudly if
you do it wrong.

The data part is where the fun starts. The public dataset looks simple — one row per
series, a winner column, two score columns. But the score columns are sorted by team
column position, not by winner, and the winner flag on those rows is broken — it says
one thing 94% of the time and the opposite the rest. If you derive winners the intuitive
way, you get half the outcomes silently wrong. I caught it because the baseline
accuracy came out at 13% — statistically impossible — and proved the correct reading
three independent ways, including checking two real Major finals against the data. That
bug fix is honestly the thing I'm most proud of here, because it's the difference
between a model that runs and a model that's right.

The modeling itself: an Elo engine written from scratch — and I mean from scratch: the
update rule is derived as one gradient step on logistic loss, so Elo is literally
stochastic gradient descent with a fixed learning rate. I sweep that rate and the
conventional value of 32 is exactly where the validation curve bottoms out, which still
surprises me. Then features — strictly pre-match only: form over the last five series,
rest days, head-to-head record, format. There's a leakage guard in the code, and any
post-outcome column you try to sneak in raises an exception. Then logistic regression
against gradient boosting against a small neural net with learned team embeddings.

Here's the money chart — the reliability diagram. Predictions on the x-axis, observed
win rates on the y. The diagonal is perfect. All three models hug it, which is the
actual deliverable: the probabilities are calibrated, not just ranked. The headline
number: 0.646 log loss versus 0.693 for the do-nothing baseline. And the honest part:
the neural net lost to plain logistic regression. At ten thousand series, the signal is
basically linear in rating difference, and the embeddings just added variance. That's a
finding, not a failure.

The most suspicious thing? There's no bookmaker odds in the comparison — the free odds
data that exists is too thin to trust. So this is a calibration study against outcomes,
not a market test, and I say so in the README instead of pretending otherwise.

What came next is actually already done. Three things, each one closing a hole a
reviewer would spot:

One — the roster signal. The dataset ships full five-player lineups per map, and my
first version ignored them. Adding roster-stability and stand-in flags to the logistic
model took log loss from 0.6464 to 0.6377 — the single biggest feature win in the repo,
and it was sitting in the data the whole time unused. That's the honest version of
"the model got better because I fed it information I already had."

Two — calibration wasn't as clean as the aggregate chart claimed. Splitting the
reliability diagram by format and tier shows the aggregate number was hiding a tier-3
problem: expected calibration error of 0.126 in tier 3 versus 0.02 to 0.03 everywhere
else. The model is over-confident exactly where the data is noisiest, and saying that
out loud is the point.

Three — one test split is one anecdote. So there's now a rolling-origin backtest across
six monthly cutoffs: mean log loss 0.6476 with a 95% confidence band from 0.628 to
0.667. The headline 0.6377 was real but it sat near the optimistic edge; the June
window alone, with only 200 series, degrades to 0.683. Small test sets are fragile,
and the backtest proves it instead of hiding it.

Plus a map-level model underneath the series-level one — per-map win probability from
rating gap + map win-rate + map name, lifted to best-of-three and best-of-five — which
is the structure that lets the next version make map vetoes meaningful. The model-vs-
market comparison is scaffolded with the de-vigging math and the metrics; it just needs
odds that aren't too thin to trust. Everything keeps shipping as a FastAPI endpoint
with PSI monitoring, and the deploy + nightly-data-rerun configs are in the repo.