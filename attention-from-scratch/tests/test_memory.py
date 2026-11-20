"""
Memory regression tests.

Context (2026-11-20): the first working version leaked ~9 MB per training step and
was OOM-killed at batch 64. Two separate reference cycles were responsible:

  1. Every op's backward rule was a closure that captured its own output tensor,
     giving each node a self-cycle (out -> _backward -> closure -> out) that the
     generational GC reclaimed only sporadically. Fix: backward rules now take the
     output tensor as an argument, so nothing points back at itself.
  2. The topological sort was a recursive closure that referenced itself for the
     recursive call, so Python captured `build` in its own closure cell — and the
     captured `topo` list dragged every graph tensor into that cycle. Fix: the
     toposort is iterative, with no closure and no recursion-depth ceiling.

These tests pin the behaviour: repeated training steps must not accumulate live
graph tensors, even with no explicit garbage collection.
"""

import gc

import numpy as np

from tinytensor import Adam, TinyGPT, Tensor, set_seed
from tinytensor.data import make_batch


def live_tensors():
    return sum(1 for obj in gc.get_objects() if isinstance(obj, Tensor))


def test_no_tensor_growth_over_training_steps():
    set_seed(0)
    rng = np.random.default_rng(0)
    model = TinyGPT(vocab_size=16, dim=32, n_heads=2, n_layers=2, block_size=8)
    opt = Adam(model.parameters(), lr=1e-3)

    # warm-up so lazy allocations (Adam state, etc.) settle
    for _ in range(3):
        x, y = make_batch(rng, 8, 4, 16)
        loss = model.loss(x, y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        del loss
    gc.collect()
    baseline = live_tensors()

    for _ in range(40):
        x, y = make_batch(rng, 8, 4, 16)
        loss = model.loss(x, y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        del loss
    gc.collect()
    assert live_tensors() == baseline, (
        f"graph tensors leaked: {baseline} -> {live_tensors()}. "
        "See the module docstring: backward rules must not capture their output."
    )


def test_forward_only_does_not_accumulate():
    """Inference without backward must also free graphs (they contain the same
    closure edges)."""
    set_seed(0)
    rng = np.random.default_rng(0)
    model = TinyGPT(vocab_size=16, dim=32, n_heads=2, n_layers=1, block_size=8)

    for _ in range(3):
        x, _ = make_batch(rng, 8, 4, 16)
        out = model(x)
        del out
    gc.collect()
    baseline = live_tensors()

    for _ in range(20):
        x, _ = make_batch(rng, 8, 4, 16)
        out = model(x)
        del out
    gc.collect()
    assert live_tensors() == baseline, "forward-only passes are accumulating graph tensors"


def test_deep_graph_does_not_hit_recursion_limit():
    """The iterative toposort must handle graphs deeper than Python's recursion
    limit (the recursive version crashed at ~1000 chained ops)."""
    depth = 1200
    x = Tensor(np.ones(1), requires_grad=True)
    out = x
    for _ in range(depth):
        out = (out * 1.0001 + 0.5).tanh()
    root = out.sum()
    root.backward()
    assert x.grad is not None and np.isfinite(x.grad).all(), "deep graph failed to backprop"
