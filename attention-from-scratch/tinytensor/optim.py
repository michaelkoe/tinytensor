"""optim.py — SGD (+momentum) and Adam, plus gradient-norm clipping."""

from __future__ import annotations

import numpy as np

from .engine import Tensor

__all__ = ["SGD", "Adam", "clip_grad_norm_"]


class Optimizer:
    def __init__(self, params, lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1.0, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.buf = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.buf[i] = self.momentum * self.buf[i] + p.grad
            p.data -= self.lr * self.buf[i]


class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm_(params, max_norm: float) -> float:
    """Clip by global L2 norm, in place. Returns the pre-clip norm."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float((p.grad ** 2).sum())
    norm = total ** 0.5
    if max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return norm
