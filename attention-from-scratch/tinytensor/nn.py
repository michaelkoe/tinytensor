"""
nn.py — neural network layers built on the autograd engine.

Everything here is written out by hand rather than imported, so that every
gradient in the model is either verified by finite differences in tests/ or is a
composition of ops that are.

Architecture is a pre-LayerNorm decoder-only transformer (GPT-style) with causal
self-attention, sized for CPU NumPy in float64.
"""

from __future__ import annotations

import numpy as np

from .engine import Tensor, one_hot

__all__ = [
    "Module", "Linear", "LayerNorm", "Embedding", "MLP",
    "CausalSelfAttention", "TransformerBlock", "TinyGPT",
]


class Module:
    def parameters(self):
        out = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    out.append(value)
            elif isinstance(value, Module):
                out.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        out.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        out.append(item)
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def num_params(self) -> int:
        return sum(p.data.size for p in self.parameters())


def _param(*shape, scale: float = 1.0) -> Tensor:
    return Tensor(np.random.randn(*shape) * scale, requires_grad=True)


class Linear(Module):
    """y = x @ W + b, with LeCun-normal initialisation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        std = scale = 1.0 / np.sqrt(in_features)
        self.W = _param(in_features, out_features, scale=std)
        self.b = Tensor(np.zeros(out_features), requires_grad=bias)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class LayerNorm(Module):
    """
    Normalises over the last axis, then applies a learned affine transform.

    Var is computed with a biased (1/N) estimate, matching the reference
    implementation. eps sits inside the sqrt.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        std = (var + self.eps) ** 0.5
        return (xc / std) * self.gamma + self.beta


class Embedding(Module):
    """
    Lookup implemented as one_hot(idx) @ W.

    A real implementation would index directly. The one-hot form keeps the whole
    graph inside the verified op set, at the cost of memory proportional to vocab —
    acceptable at this scale and worth saying out loud rather than hiding.
    """

    def __init__(self, n_tokens: int, dim: int, std: float = 0.02):
        self.n_tokens = n_tokens
        self.W = _param(n_tokens, dim, scale=std)

    def forward(self, idx: np.ndarray) -> Tensor:
        return Tensor(one_hot(idx, self.n_tokens)) @ self.W


class MLP(Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        hidden = dim * hidden_mult
        self.fc1 = Linear(dim, hidden)
        self.fc2 = Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).tanh())


class CausalSelfAttention(Module):
    """
    Multi-head causal self-attention.

    Q, K and V use three separate projections rather than one fused QKV matmul.
    The fused version is faster and is what production code does; the split version
    keeps the graph readable, which matters more here.

    Masking is additive: a large negative constant is added above the diagonal
    before the softmax, so future positions get ~0 weight. An additive constant is
    differentiable-safe (its gradient is zero and it carries no parameters), which
    is why it does not need its own backward rule.
    """

    def __init__(self, dim: int, n_heads: int, block_size: int):
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.Wq = Linear(dim, dim)
        self.Wk = Linear(dim, dim)
        self.Wv = Linear(dim, dim)
        self.proj = Linear(dim, dim)
        # causal mask: 0 on and below the diagonal, -1e30 above it
        future = np.triu(np.ones((block_size, block_size)), k=1)
        self._mask = np.where(future > 0, -1e30, 0.0)

    def _split_heads(self, t: Tensor, batch: int, seq: int) -> Tensor:
        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        return t.reshape(batch, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, dim = x.shape
        q = self._split_heads(self.Wq(x), batch, seq)
        k = self._split_heads(self.Wk(x), batch, seq)
        v = self._split_heads(self.Wv(x), batch, seq)

        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale          # (B,H,T,T)
        att = att + self._mask[:seq, :seq]                        # additive causal mask
        weights = att.softmax(axis=-1)

        out = weights @ v                                          # (B,H,T,D)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, dim)   # merge heads
        return self.proj(out)


class TransformerBlock(Module):
    """Pre-LayerNorm block: x = x + attn(LN(x)); x = x + mlp(LN(x))."""

    def __init__(self, dim: int, n_heads: int, block_size: int, hidden_mult: int = 4):
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, block_size)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim, hidden_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(Module):
    """Decoder-only transformer: token + position embeddings, N blocks, LM head."""

    def __init__(self, vocab_size: int, dim: int = 64, n_heads: int = 4,
                 n_layers: int = 2, block_size: int = 16):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(block_size, dim)
        self.blocks = [TransformerBlock(dim, n_heads, block_size) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, vocab_size)

    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx)
        batch, seq = idx.shape
        assert seq <= self.block_size, f"sequence {seq} exceeds block_size {self.block_size}"
        positions = np.tile(np.arange(seq), (batch, 1))
        x = self.tok_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def loss(self, idx: np.ndarray, targets: np.ndarray,
             ignore_index: int = -1) -> Tensor:
        return self.forward(idx).cross_entropy(targets, ignore_index=ignore_index)

    def predict(self, idx: np.ndarray) -> np.ndarray:
        return self.forward(idx).data.argmax(axis=-1)
