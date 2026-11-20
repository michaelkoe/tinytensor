"""
engine.py — a small reverse-mode automatic differentiation engine over NumPy arrays.

Every operation records how to compute its local gradient; Tensor.backward() walks
the recorded graph in reverse topological order and accumulates gradients.

Design notes
------------
* float64 throughout. This is deliberate: the point of the repo is numerical
  correctness, and float64 makes finite-difference gradient checks meaningful.
* Gradients are accumulated (added) into ``.grad``, never overwritten. This is what
  makes fan-out graphs (one tensor used by several ops) correct.
* Broadcasting is handled in the backward pass by summing the incoming gradient
  back down to each operand's original shape (``_unbroadcast``).
* Memory: a naive implementation leaks. Each op's backward rule is a closure, and a
  closure that captures its own output tensor forms a reference cycle the
  generational GC reclaims only sporadically — that leaked ~9 MB per training step
  here and OOM-killed the process at batch 64. Two rules prevent it, both matching
  how PyTorch's autograd engine is built:
    1. backward rules receive their output tensor as an argument (``_bw(out)``)
       instead of capturing it, so a node never points back at itself;
    2. the topological sort is iterative with an explicit stack. Besides avoiding
       the closure, it removes Python's recursion-depth ceiling: the recursive
       version failed for graphs deeper than ~1000 nodes.
* Only the operations needed by the model are implemented. That is a feature, not a
  gap: a small surface area is a small surface area to get wrong, and every op here
  is verified against finite differences in ``tests/test_engine.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Tensor", "no_grad"]


def _as_tensor(x) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A NumPy array that remembers how it was computed."""

    __slots__ = ("data", "requires_grad", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad: bool = False, _children=(), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = _children
        self._op = _op
        self._backward = None  # set by ops: a rule taking (out) as its argument

    # ---------------------------------------------------------------- helpers
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def item(self):
        return float(self.data.reshape(-1)[0]) if self.data.size == 1 else self.data.item()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    def _acc(self, g: np.ndarray):
        """Accumulate gradient into this tensor."""
        if not self.requires_grad:
            return
        g = _unbroadcast(np.asarray(g, dtype=np.float64), self.data.shape)
        self.grad = g if self.grad is None else self.grad + g

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------ arithmetic
    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other), "add")

        def _bw(out):
            self._acc(out.grad)
            other._acc(out.grad)

        out._backward = _bw
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __rsub__(self, other):
        return _as_tensor(other) + (-self)

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other), "mul")

        def _bw(out):
            self._acc(out.grad * other.data)
            other._acc(out.grad * self.data)

        out._backward = _bw
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad,
                     (self, other), "div")

        def _bw(out):
            self._acc(out.grad / other.data)
            other._acc(-out.grad * self.data / (other.data ** 2))

        out._backward = _bw
        return out

    def __rtruediv__(self, other):
        return _as_tensor(other) / self

    def __pow__(self, exponent: float):
        out = Tensor(self.data ** exponent, self.requires_grad, (self,), f"pow{exponent}")

        def _bw(out):
            self._acc(out.grad * exponent * (self.data ** (exponent - 1)))

        out._backward = _bw
        return out

    def __matmul__(self, other):
        """Batched matrix multiply over the last two axes."""
        other = _as_tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     (self, other), "matmul")

        def _bw(out):
            g = out.grad
            if self.requires_grad:
                self._acc(g @ np.swapaxes(other.data, -1, -2))
            if other.requires_grad:
                other._acc(np.swapaxes(self.data, -1, -2) @ g)

        out._backward = _bw
        return out

    # -------------------------------------------------------------- unary ops
    def exp(self):
        out = Tensor(np.exp(self.data), self.requires_grad, (self,), "exp")

        def _bw(out):
            self._acc(out.grad * out.data)

        out._backward = _bw
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _bw(out):
            self._acc(out.grad / self.data)

        out._backward = _bw
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _bw(out):
            self._acc(out.grad * (1.0 - t ** 2))

        out._backward = _bw
        return out

    def relu(self):
        mask = (self.data > 0).astype(np.float64)
        out = Tensor(self.data * mask, self.requires_grad, (self,), "relu")

        def _bw(out):
            self._acc(out.grad * mask)

        out._backward = _bw
        return out

    # --------------------------------------------------------- shape ops
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old_shape = self.data.shape
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _bw(out):
            self._acc(out.grad.reshape(old_shape))

        out._backward = _bw
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        inverse = tuple(int(i) for i in np.argsort(axes))
        out = Tensor(np.transpose(self.data, axes), self.requires_grad, (self,), "transpose")

        def _bw(out):
            self._acc(np.transpose(out.grad, inverse))

        out._backward = _bw
        return out

    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "sum")

        def _bw(out):
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.ones_like(self.data) * g)

        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims: bool = False):
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # --------------------------------------------------------- softmax family
    def softmax(self, axis: int = -1):
        """
        Numerically stable softmax.

        The max-shift is treated as a constant, which is mathematically exact:
        softmax(x - c) == softmax(x) for any constant c, so d/dx is unchanged and
        the shift introduces no spurious gradient path. Without the shift, exp()
        overflows to inf for logits above ~709 in float64.
        """
        shifted = self - Tensor(self.data.max(axis=axis, keepdims=True))
        e = shifted.exp()
        return e / e.sum(axis=axis, keepdims=True)

    def log_softmax(self, axis: int = -1):
        shifted = self - Tensor(self.data.max(axis=axis, keepdims=True))
        return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()

    def cross_entropy(self, targets: np.ndarray, ignore_index: int = -1):
        """
        Mean token-level cross entropy. ``targets`` is an int array whose shape is
        ``logits.shape[:-1]``. Implemented as -(log_softmax * onehot).sum(), which
        keeps every backward pass inside the verified op set above.
        """
        logp = self.log_softmax(-1)
        onehot = one_hot(targets, self.data.shape[-1])
        nll = -(logp * Tensor(onehot)).sum(axis=-1)
        if ignore_index >= 0:
            mask = (targets != ignore_index).astype(np.float64)
            return (nll * Tensor(mask)).sum() * (1.0 / max(mask.sum(), 1.0))
        return nll.mean()

    # ------------------------------------------------------------- backward
    def backward(self, grad: np.ndarray | None = None, retain_graph: bool = False):
        """
        Reverse-mode pass over the graph ending at this tensor.

        The topological order is computed with an iterative DFS (explicit stack,
        post-order), so this works for arbitrarily deep graphs without hitting
        Python's recursion limit.

        The graph is freed afterwards (``retain_graph=False``), matching PyTorch's
        default: freeing ``_prev``/``_backward`` drops the closures and their
        references to parents, so intermediates die by reference count the moment
        the root goes out of scope. Pass ``retain_graph=True`` to backward through
        the same graph more than once.
        """
        # iterative post-order DFS: inputs first, self last. Nodes are marked seen
        # when EXPANDED, not when pushed — marking at push time lets a child's
        # subtree finish before an ancestor that shares it (e.g. the softmax
        # diamond where exp output feeds both div and sum), which would emit the
        # child before its parent and produce a wrong topological order.
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                stack.append((parent, False))

        for t in topo:
            t.grad = None
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)

        for t in reversed(topo):
            # A node reached only through a constant branch never receives gradient
            # (requires_grad=False), so there is nothing for its backward to do.
            if t._backward is None or t.grad is None:
                continue
            t._backward(t)

        if not retain_graph:
            for t in topo:
                t._prev = ()
                t._backward = None


def one_hot(idx: np.ndarray, n_classes: int) -> np.ndarray:
    idx = np.asarray(idx).astype(np.int64)
    out = np.zeros(idx.shape + (n_classes,), dtype=np.float64)
    np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
    return out


class no_grad:
    """Placeholder for API symmetry; this engine only builds graphs for ops on
    tensors whose ``requires_grad`` is True, so this is a no-op context."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
