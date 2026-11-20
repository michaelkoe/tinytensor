"""
data.py — the sequence-reversal task.

Why the input is repeated
-------------------------
A causal (decoder-only) model cannot reverse a sequence it is reading
position-by-position: the token that belongs at output position i sits at input
position L-1-i, which the causal mask hides for every i < L/2. We hit exactly this
in development — the model learned to answer the second half of the output and
plateaued at 50% token accuracy with the first half constant, which is the
signature of the mask, not of undertraining.

The standard fix (the same trick used in copy/induction-head benchmarks) is to
show the sequence twice:

    input  = [x1 .. xL, x1 .. xL]
    target = [ ignored,  xL .. x1 ]        (positions 0..L-1 carry pad=0, ignored in the loss)

The model reads the first copy and writes the reversal over the second copy. Now
every answer position L+i can causally see x[L-1-i] in the first copy, and the
task requires genuine order-sensitive use of attention: a bag-of-tokens model
sees the identical multiset for every permutation, so it is structurally capped
at chance.

Tokens are integers in [1, vocab_size). 0 is reserved as pad/ignore.
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_dataset", "make_batch"]


def _pair(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = x.shape[1]
    inp = np.concatenate([x, x], axis=1)
    tgt = np.concatenate([np.zeros_like(x), x[:, ::-1]], axis=1)
    return inp, tgt


def make_dataset(n_examples: int, seq_len: int, vocab_size: int, seed: int = 0):
    """Returns (inputs, targets), each of shape (n_examples, 2*seq_len)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, vocab_size, size=(n_examples, seq_len), dtype=np.int64)
    return _pair(x)


def make_batch(rng: np.random.Generator, batch_size: int, seq_len: int, vocab_size: int):
    x = rng.integers(1, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    return _pair(x)
