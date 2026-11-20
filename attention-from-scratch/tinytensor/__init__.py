"""
tinytensor — a decoder-only transformer built on a hand-written NumPy autograd engine.

Quick start
-----------
    from tinytensor import TinyGPT, Adam
    model = TinyGPT(vocab_size=32, dim=64, n_heads=4, n_layers=2, block_size=16)
    opt = Adam(model.parameters(), lr=3e-3)
    loss = model.loss(x, y); loss.backward(); opt.step(); opt.zero_grad()
"""

from .engine import Tensor, one_hot
from .nn import (
    Module, Linear, LayerNorm, Embedding, MLP,
    CausalSelfAttention, TransformerBlock, TinyGPT,
)
from .optim import SGD, Adam, clip_grad_norm_
from .utils import set_seed, token_accuracy, exact_match
from .data import make_dataset, make_batch

__version__ = "0.1.0"

__all__ = [
    "Tensor", "one_hot",
    "Module", "Linear", "LayerNorm", "Embedding", "MLP",
    "CausalSelfAttention", "TransformerBlock", "TinyGPT",
    "SGD", "Adam", "clip_grad_norm_",
    "set_seed", "token_accuracy", "exact_match",
    "make_dataset", "make_batch",
]
