"""M8 tests — team-embedding net shapes/gradients. CI skips when torch is absent."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).parent.parent

try:
    from cs2analytics.models.net import TeamEmbeddingNet

    _M8_READY = True
    _M8_REASON = ""
except ModuleNotFoundError as _e:
    _M8_READY = False
    _M8_REASON = f"M8 net not implemented yet ({_e.name}) — the human's build"

pytestmark = [pytest.mark.skipif(not _M8_READY, reason=_M8_REASON)]

N_TEAMS, EMB_DIM, HIDDEN, N_NUM = 7, 4, 3, 2


@pytest.fixture()
def net():
    torch.manual_seed(0)
    return TeamEmbeddingNet(n_teams=N_TEAMS, emb_dim=EMB_DIM, hidden=HIDDEN)


def _batch(b=5):
    return (
        torch.randint(0, N_TEAMS, (b,), dtype=torch.long),
        torch.randint(0, N_TEAMS, (b,), dtype=torch.long),
        torch.randn(b, N_NUM),
    )


def test_forward_returns_logits_batch_shape(net):
    out = net(*_batch())
    assert out.shape == (5,)
    assert out.dtype == torch.float32


def test_forward_is_not_bounded_to_probabilities(net):
    """A sigmoid inside forward would silently break BCEWithLogitsLoss."""
    i1, i2, num = _batch(64)
    with torch.no_grad():
        out = net(i1, i2, num)
    assert (out < 0).any() or (out > 1).any(), "output looks like a probability — remove sigmoid"


def test_parameter_count_formula(net):
    emb_params = N_TEAMS * EMB_DIM
    mlp_in = EMB_DIM * 2 + N_NUM
    mlp_params = mlp_in * HIDDEN + HIDDEN + HIDDEN + 1
    expected = emb_params + mlp_params
    actual = sum(p.numel() for p in net.parameters())
    assert actual == expected, f"expected {expected} params, got {actual}"


def test_embedding_gradients_flow(net):
    i1, i2, num = _batch()
    logits = net(i1, i2, num)
    loss = logits.mean()
    loss.backward()
    emb = net.embedding.weight
    assert emb.grad is not None
    assert float(emb.grad.abs().sum()) > 0.0


def test_determinism_with_fixed_seed():
    torch.manual_seed(42)
    a = TeamEmbeddingNet(n_teams=N_TEAMS, emb_dim=EMB_DIM, hidden=HIDDEN)
    torch.manual_seed(42)
    b = TeamEmbeddingNet(n_teams=N_TEAMS, emb_dim=EMB_DIM, hidden=HIDDEN)
    assert all(torch.allclose(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True))


def test_dl_row_in_comparison_table():
    path = REPO / "outputs" / "m8_model_comparison.csv"
    if not path.exists():
        pytest.skip("outputs/m8_model_comparison.csv not produced yet (M8 §3)")
    import pandas as pd

    df = pd.read_csv(path)
    assert "dl_embedding" in set(df["model"])
    row = df[df["model"] == "dl_embedding"].iloc[0]
    assert 0 < float(row["logloss"]) < 0.6932
