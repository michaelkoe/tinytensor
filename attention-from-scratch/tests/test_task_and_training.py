"""
Task-shape tests and an end-to-end training test.
"""

import numpy as np
import pytest

from tinytensor import TinyGPT, Adam, set_seed, exact_match
from tinytensor.data import make_batch, make_dataset


def test_dataset_shape_and_causality_of_targets():
    L, V = 8, 16
    x, y = make_dataset(50, L, V, seed=3)
    assert x.shape == (50, 2 * L) and y.shape == (50, 2 * L)
    # first half ignored (pad), second half is the reversal of the first copy
    assert (y[:, :L] == 0).all()
    inp_first = x[:, :L]
    np.testing.assert_array_equal(y[:, L:], inp_first[:, ::-1])
    # the model sees the sequence twice
    np.testing.assert_array_equal(x[:, L:], inp_first)


def test_pad_targets_are_ignored_in_loss():
    """Loss must average over scored positions only, and match manual computation."""
    logits = np.random.randn(4, 6, 5)
    targets = np.zeros((4, 6), dtype=np.int64)      # first 6 positions all pad
    targets[:, 3:] = np.random.randint(1, 5, size=(4, 3))
    from tinytensor.engine import Tensor
    got = float(Tensor(logits).cross_entropy(targets, ignore_index=0).data)
    rows, cols = np.nonzero(targets)
    shifted = logits - logits.max(-1, keepdims=True)
    nll = -shifted[rows, cols, targets[rows, cols]] + np.log(np.exp(shifted).sum(-1))[rows, cols]
    assert abs(got - nll.mean()) < 1e-10


def test_exact_match_ignores_pad_positions():
    pred = np.array([[5, 5, 4, 2]])
    target = np.array([[0, 0, 4, 2]])               # first two positions are pad
    assert exact_match(pred, target, ignore_value=0) == 1.0
    assert exact_match(pred, np.array([[0, 0, 4, 3]]), ignore_value=0) == 0.0


@pytest.mark.slow
def test_model_learns_reversal_end_to_end():
    """Tiny config, short training: the full pipeline (engine -> model -> loss ->
    Adam) must actually solve the task. ~15 s."""
    set_seed(7)
    rng = np.random.default_rng(7)
    L, V = 4, 10
    model = TinyGPT(vocab_size=V, dim=32, n_heads=2, n_layers=2, block_size=2 * L)
    opt = Adam(model.parameters(), lr=3e-3)

    for step in range(300):
        x, y = make_batch(rng, 32, L, V)
        loss = model.loss(x, y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ex, ey = make_dataset(256, L, V, seed=12345)
    pred = model(ex).data.argmax(-1)
    assert exact_match(pred, ey, ignore_value=0) > 0.9, (
        "end-to-end training failed to learn reversal — check engine gradients or task setup"
    )
