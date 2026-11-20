"""utils.py — seeding, metrics, and small reporting helpers."""

from __future__ import annotations

import json
import time

import numpy as np

__all__ = ["set_seed", "token_accuracy", "exact_match", "Timer", "save_json"]


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def token_accuracy(pred: np.ndarray, target: np.ndarray, ignore_value=None) -> float:
    """Fraction of positions correct. With ``ignore_value``, ignored positions
    (e.g. pads) are excluded from both numerator and denominator."""
    if ignore_value is None:
        return float((pred == target).mean())
    mask = target != ignore_value
    return float(((pred == target) & mask).sum() / max(int(mask.sum()), 1))


def exact_match(pred: np.ndarray, target: np.ndarray, ignore_value=None) -> float:
    """Fraction of sequences where every scored position is correct. With
    ``ignore_value``, ignored positions count as correct (they are not part of
    the task), so this is exact match over the scored region."""
    if ignore_value is None:
        return float((pred == target).all(axis=-1).mean())
    scored = np.where(target == ignore_value, pred, target)
    return float((pred == scored).all(axis=-1).mean())


class Timer:
    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


def save_json(path: str, payload) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
