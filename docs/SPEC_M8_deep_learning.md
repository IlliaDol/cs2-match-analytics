# M8 BUILD SPEC — Deep Learning: team embeddings  · `DS MLE` · [HE]

**Install:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU-only;
your AMD iGPU is not usable for torch on Windows — CPU is fine at this data size).

---

## §1 `src/cs2analytics/models/net.py` (yours)

```python
class TeamEmbeddingNet(nn.Module):
    def __init__(self, n_teams: int, emb_dim: int = 16, hidden: int = 32): ...
    def forward(self, team1_idx, team2_idx, numeric) -> torch.Tensor   # logits, shape (B,)
```
Architecture: `emb1, emb2 = E[i1], E[i2]`; input to MLP = `[emb1 - emb2, emb1 * emb2,
numeric]`; MLP = Linear(→hidden) → ReLU → Dropout(0.2) → Linear(→1). Output = logit
(feed to `BCEWithLogitsLoss`).

**Contract (tested, CI runs):** forward returns shape `(B,)` logits (NOT probabilities);
with `n_teams=5, emb_dim=4` the parameter count is exactly
`5*4 + (4*2 + n_num)*hidden + hidden + hidden + 1`; embedding gradients flow
(`loss.backward()` → `E.weight.grad is not None`).

## §2 `src/cs2analytics/models/train_dl.py` (yours)

```python
def train_model(...) -> dict   # {"model": ..., "history": [...], "logloss": float, ...}
```
- Categorical encoding: build `team_to_idx` from TRAIN ONLY (test teams unseen → index 0
  reserved as `<unk>`). State this in a comment: it is the neural version of the leakage law.
- Optimiser Adam lr=1e-3, batch 256, ≤ 30 epochs, early stop on train loss plateau.
- Fixed seeds (`torch.manual_seed(42)`, numpy seed) — determinism is a repo rule.

## §3 `notebooks/04_embedding_experiments.ipynb` (yours)

1. Train; plot train/val loss curves in-line.
2. **Extend the comparison table** → write `outputs/m8_model_comparison.csv`: copy M7's
   table and append a `dl_embedding` row with the SAME test split and metrics.
3. Embedding sanity: cosine similarity between the top-2 rivals' learned embeddings vs
   their actual head-to-head record. Report both numbers, even if they disagree.
4. Conclusion paragraph, honest either way: "embeddings helped / didn't help, because …".
   If the MLP loses to LR's 0.66 logloss, that is a FINDING, not a failure.

## §4 Tests (`tests/test_m8_net.py`, CI-runs the model-shape tests; artifacts data-gated)
- forward shape and dtype
- parameter-count formula exact
- gradients reach the embedding table
- logits are NOT bounded to [0,1] (a sigmoid-in-forward bug would hide the loss function)
- `outputs/m8_model_comparison.csv` exists with a `dl_embedding` row (data-gated)

## Checkpoint
1. Trace backprop for ONE weight in a 2-2-1 net by hand with actual numbers.
2. Why do embeddings beat one-hot here? What is the emb_dim tradeoff?
3. Why is dropout questionable when teams repeat within a batch (temporal correlation in
   the shuffled sampler)? What would you do instead?

## Appendix — hints
- Param count check: `sum(p.numel() for p in model.parameters())`.
- If `E.weight.grad` is None: you probably used `E[i1]` with integer tensors of wrong dtype
  (needs `torch.long`).
- Unseen-team index 0 must be excluded from the embedding-gradient update? No — simplest is
  to let index 0 be a real "unknown" vector; it only trains if train has unknown teams.
