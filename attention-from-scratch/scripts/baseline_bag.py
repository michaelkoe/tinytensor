#!/usr/bin/env python3
"""
Baseline: a bag-of-tokens model on the reversal task.

This model sees only the multiset of input tokens (sum of one-hots) — identical
for every permutation of the input — and must still produce per-position
predictions. Without order, the only exploitable signal is token frequency: the
optimal position-free strategy is to predict the most frequent bag token at every
position, which scores E[max_v count(v)] / L ≈ 15% token accuracy here and can
never produce a correct full sequence.

This is the repo's central ablation: same engine, same loss, same data — the only
difference is that order information is removed by construction. Measured result:
15.1% token accuracy (the theoretical ceiling), 0.0% exact match — versus 100%
exact match for the transformer. The gap is what attention buys.

    python scripts/baseline_bag.py --steps 300
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinytensor import Adam, Module, Linear, Tensor, clip_grad_norm_, set_seed  # noqa: E402
from tinytensor.data import make_batch, make_dataset  # noqa: E402
from tinytensor.engine import one_hot  # noqa: E402
from tinytensor.utils import exact_match, save_json, token_accuracy  # noqa: E402

PAD = 0


class BagOfTokens(Module):
    """Position-free predictor: per-position logits from the token multiset only."""

    def __init__(self, vocab_size: int, seq_len: int, hidden: int = 256):
        self.seq_len = seq_len
        self.fc1 = Linear(vocab_size, hidden)
        self.fc2 = Linear(hidden, 2 * seq_len * vocab_size)
        self.vocab_size = vocab_size

    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx)
        bag = Tensor(one_hot(idx, self.vocab_size)).sum(axis=1)        # (B, V) — order-free
        h = self.fc1(bag).tanh()
        return self.fc2(h).reshape(idx.shape[0], 2 * self.seq_len, self.vocab_size)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--out", type=str, default="results/baseline_bag.json")
    return p.parse_args()


def evaluate(model, x, y):
    losses, n = [], x.shape[0]
    correct_tok, total_tok, exact = 0, 0, 0.0
    for start in range(0, n, 256):
        xb, yb = x[start:start + 256], y[start:start + 256]
        logits = model(xb)
        losses.append(float(logits.cross_entropy(yb, ignore_index=PAD).data))
        pred = logits.data.argmax(-1)
        correct_tok += int(((pred == yb) & (yb != PAD)).sum())
        total_tok += int((yb != PAD).sum())
        exact += exact_match(pred, yb, ignore_value=PAD) * xb.shape[0]
    return {"loss": float(np.mean(losses)), "token_acc": correct_tok / max(total_tok, 1),
            "exact_match": exact / n}


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    L = args.seq_len

    train_x, train_y = make_dataset(20_000, L, args.vocab, seed=args.seed + 1)
    eval_x, eval_y = make_dataset(args.eval_n, L, args.vocab, seed=args.seed + 99_999)

    model = BagOfTokens(args.vocab, L)
    opt = Adam(model.parameters(), lr=args.lr)
    print(f"bag-of-tokens baseline: {model.num_params():,} params "
          f"(input is the token multiset — order-free by construction)")

    for step in range(1, args.steps + 1):
        xb, yb = make_batch(rng, args.batch_size, L, args.vocab)
        logits = model(xb)
        loss = logits.cross_entropy(yb, ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            m = evaluate(model, eval_x, eval_y)
            print(f"step {step:5d} | train_loss {float(loss.data):.4f} | "
                  f"eval_loss {m['loss']:.4f} | tok_acc {m['token_acc']:.3f} | "
                  f"exact {m['exact_match']:.3f}", flush=True)

    final = evaluate(model, eval_x, eval_y)
    save_json(args.out, {"config": vars(args), "final": final,
                         "params": model.num_params()})
    print(f"\nFINAL  token acc {final['token_acc']:.1%} (position-free ceiling "
          f"~E[max count]/{L} ≈ 15%) | exact-match {final['exact_match']:.1%}")
    print("Frequency is the only signal left without order; the transformer's")
    print("attention path is what carries the order information.")


if __name__ == "__main__":
    main()
