#!/usr/bin/env python3
"""Plot the learning curves produced by train_reverse.py.

    python scripts/plot_curve.py --hist results/history.json --hist results/history_1layer.json \
        --out results/learning_curve.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hist", action="append", default=[],
                   help="path to a history.json (repeatable)")
    p.add_argument("--labels", action="append", default=[])
    p.add_argument("--out", type=str, default="results/learning_curve.png")
    args = p.parse_args()
    if not args.hist:
        args.hist = ["results/history.json"]
        args.labels = ["TinyGPT (2 layers)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for path, label in zip(args.hist, args.labels or [None] * len(args.hist)):
        h = json.load(open(path))
        steps = h["steps"]
        name = label or f"{h['config'].get('layers', '?')} layer(s)"
        axes[0].plot(steps, h["eval_loss"], marker=".", label=name)
        axes[1].plot(steps, [100 * v for v in h["eval_exact_match"]], marker=".", label=name)

    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("held-out cross-entropy")
    axes[0].set_title("eval loss")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("exact match (%)")
    axes[1].set_title("held-out exact match")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning to reverse a sequence — tinytensor (NumPy autograd transformer)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
