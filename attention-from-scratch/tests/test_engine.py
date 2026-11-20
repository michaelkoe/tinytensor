"""
Gradient checks for every op in the engine.

The standard finite-difference test: for a scalar-valued f, the analytic gradient
dy/dx should match (f(x+h) - f(x-h)) / 2h for small h. float64 + central
differences + h=1e-6 gives ~7 significant digits on smooth functions, so a
relative error under 1e-6 is a real pass and not a generous one.
"""

import numpy as np
import pytest

from tinytensor.engine import Tensor
from tinytensor.nn import Linear, LayerNorm, MLP, CausalSelfAttention, Embedding


def numeric_grad(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar function f at array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + h
        f_plus = f(x)
        x[i] = orig - h
        f_minus = f(x)
        x[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


def check_op(fn, *arrays, rtol=1e-6, atol=1e-8):
    """fn: (*Tensors) -> scalar Tensor. Verifies every input gradient."""
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*tensors)
    out.backward()
    analytic = [t.grad.copy() for t in tensors]

    def scalar_fn(*arrays):
        ts = [Tensor(a) for a in arrays]
        return float(out_of(fn(*ts)))

    for i, a in enumerate(arrays):
        num = numeric_grad(lambda x, idx=i: scalar_fn(*[
            x if j == idx else arrays[j].copy() for j in range(len(arrays))
        ]), a.copy())
        np.testing.assert_allclose(analytic[i], num, rtol=rtol, atol=atol,
                                   err_msg=f"grad mismatch for input {i}")


def out_of(t: Tensor) -> np.ndarray:
    return t.data


# ------------------------------------------------------------------ primitives

def test_add_and_mul():
    a = np.random.randn(3, 4)
    b = np.random.randn(3, 4)
    check_op(lambda x, y: ((x + y) * x * y).sum(), a, b)


def test_broadcast_add():
    a = np.random.randn(3, 4)
    b = np.random.randn(4)
    check_op(lambda x, y: ((x + y) ** 2).sum(), a, b)


def test_matmul():
    a = np.random.randn(3, 5)
    b = np.random.randn(5, 2)
    check_op(lambda x, y: (x @ y).sum(), a, b)


def test_batched_matmul():
    a = np.random.randn(2, 3, 4, 5)
    b = np.random.randn(2, 3, 5, 6)
    check_op(lambda x, y: (x @ y).sum(), a, b)


def test_div_and_pow():
    a = np.random.rand(3, 4) + 1.0
    b = np.random.rand(3, 4) + 1.0
    check_op(lambda x, y: (x / y + (y ** 0.5)).sum(), a, b)


def test_exp_log_tanh_relu():
    a = np.random.randn(4, 4)
    check_op(lambda x: (x.exp() + (x.exp() + 1.0).log() + x.tanh() + x.relu()).sum(), a)


def test_reshape_and_transpose():
    a = np.random.randn(2, 3, 4)
    check_op(lambda x: (x.reshape(6, 4).transpose(1, 0) ** 2).sum(), a)
    check_op(lambda x: (x.transpose(2, 0, 1) ** 2).sum(), a)


def test_sum_mean_keepdims():
    a = np.random.randn(3, 4, 5)
    check_op(lambda x: (x.sum(axis=1, keepdims=True) * x.mean(axis=-1, keepdims=True)).sum(), a)


# ---------------------------------------------------------------------- softmax

def test_softmax_grad():
    a = np.random.randn(3, 5) * 2.0
    check_op(lambda x: (x.softmax(-1) ** 2).sum(), a)


def test_softmax_is_stable_and_sums_to_one():
    x = Tensor(np.array([[1000.0, 1001.0, 999.0]]))
    p = x.softmax(-1).data
    assert np.isfinite(p).all(), "softmax overflowed on large logits"
    np.testing.assert_allclose(p.sum(-1), 1.0, rtol=1e-12)
    # softmax([1000, 1001, 999]) == softmax([-1, 0, -2])
    expected = np.exp([[-1.0, 0.0, -2.0]])
    expected = expected / expected.sum(-1, keepdims=True)
    np.testing.assert_allclose(p, expected, rtol=1e-12)


def test_log_softmax_matches_naive():
    a = np.random.randn(4, 7)
    got = Tensor(a).log_softmax(-1).data
    naive = a - np.log(np.exp(a).sum(-1, keepdims=True))
    np.testing.assert_allclose(got, naive, rtol=1e-12, atol=1e-12)


def test_cross_entropy_matches_reference():
    logits = np.random.randn(5, 6)
    targets = np.random.randint(0, 6, size=5)
    got = float(Tensor(logits).cross_entropy(targets).data)
    shifted = logits - logits.max(-1, keepdims=True)
    ref = float(np.mean(-shifted[np.arange(5), targets] + np.log(np.exp(shifted).sum(-1))))
    assert abs(got - ref) < 1e-12


# ----------------------------------------------------------------------- layers

def test_linear_grad():
    layer = Linear(4, 3)
    x = np.random.randn(5, 4)

    def fn(xt):
        return (layer(xt) ** 2).sum()

    t = Tensor(x.copy(), requires_grad=True)
    fn(t).backward()
    num = numeric_grad(lambda arr: float(fn(Tensor(arr)).data), x.copy())
    np.testing.assert_allclose(t.grad, num, rtol=1e-6, atol=1e-8)


def test_layernorm_grad_and_properties():
    ln = LayerNorm(6)
    x = np.random.randn(4, 6) * 5.0 + 2.0
    t = Tensor(x.copy(), requires_grad=True)
    (ln(t) ** 2).sum().backward()
    num = numeric_grad(lambda arr: float((ln(Tensor(arr)) ** 2).sum().data), x.copy())
    np.testing.assert_allclose(t.grad, num, rtol=1e-6, atol=1e-8)

    out = ln(Tensor(x)).data
    np.testing.assert_allclose(out.mean(-1), np.zeros(4), atol=1e-10)
    np.testing.assert_allclose(out.std(-1), np.ones(4), atol=1e-4)


def test_embedding_grad():
    emb = Embedding(7, 3)
    idx = np.array([[1, 5], [2, 2]])
    emb(idx)
    (emb(idx) ** 2).sum().backward()
    # gradient wrt the embedding table, by finite differences on the table
    W = emb.W.data.copy()
    num = np.zeros_like(W)
    h = 1e-6
    def f(Warr):
        old = emb.W.data
        emb.W.data = Warr
        val = float((emb(idx).data ** 2).sum())
        emb.W.data = old
        return val
    for i in np.ndindex(W.shape):
        Wp = W.copy(); Wp[i] += h
        Wm = W.copy(); Wm[i] -= h
        num[i] = (f(Wp) - f(Wm)) / (2 * h)
    np.testing.assert_allclose(emb.W.grad, num, rtol=1e-6, atol=1e-8)


def test_mlp_grad():
    mlp = MLP(4, hidden_mult=2)
    x = np.random.randn(3, 5, 4)
    t = Tensor(x.copy(), requires_grad=True)
    (mlp(t) ** 2).sum().backward()
    num = numeric_grad(lambda arr: float((mlp(Tensor(arr)) ** 2).sum().data), x.copy())
    np.testing.assert_allclose(t.grad, num, rtol=1e-6, atol=1e-8)


def test_attention_grad():
    attn = CausalSelfAttention(dim=8, n_heads=2, block_size=4)
    x = np.random.randn(2, 4, 8)
    t = Tensor(x.copy(), requires_grad=True)
    (attn(t) ** 2).sum().backward()
    num = numeric_grad(lambda arr: float((attn(Tensor(arr)) ** 2).sum().data), x.copy())
    np.testing.assert_allclose(t.grad, num, rtol=1e-5, atol=1e-7)


# ------------------------------------------------------------------- behaviour

def test_gradients_accumulate_on_fanout():
    a = Tensor(np.array([3.0]), requires_grad=True)
    b = Tensor(np.array([4.0]), requires_grad=True)
    (a * b + a * b).sum().backward()      # d/da = 2b = 8, d/db = 2a = 6
    np.testing.assert_allclose(a.grad, [8.0])
    np.testing.assert_allclose(b.grad, [6.0])


def test_backward_resets_stale_grads():
    a = Tensor(np.array([2.0]), requires_grad=True)
    (a * 3.0).sum().backward()
    (a * 3.0).sum().backward()
    np.testing.assert_allclose(a.grad, [3.0], err_msg="grads must not accumulate across backward() calls")


def test_attention_is_causal():
    """Changing a future token must not change any earlier position's output."""
    attn = CausalSelfAttention(dim=8, n_heads=2, block_size=6)
    x = np.random.randn(1, 6, 8)
    base = attn(Tensor(x)).data
    x2 = x.copy()
    x2[0, 5, :] += 10.0                       # perturb the last position only
    perturbed = attn(Tensor(x2)).data
    np.testing.assert_allclose(base[0, :5], perturbed[0, :5], atol=1e-10,
                               err_msg="information leaked backwards in time")
    assert not np.allclose(base[0, 5], perturbed[0, 5])
