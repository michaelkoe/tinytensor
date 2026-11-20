#!/usr/bin/env python3
"""
Train TinyGPT on the sequence-reversal task.

The model sees each sequence twice: it reads the first copy and must emit the
reversal over the second copy (see tinytensor/data.py for why). Only the second
half is scored.

    python scripts/train_reverse.py --steps 2000
    python scripts/train_reverse.py --steps 100 --seq-len 8     # quick smoke run

Held-out evaluation uses a *different* RNG seed from training, so the reported
numbers are on unseen sequences. History is written to results/history.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinytensor import TinyGPT, Adam, clip_grad_norm_, set_seed  # noqa: E402
from tinytensor.data import make_batch, make_dataset  # noqa: E402
from tinytensor.utils import exact_match, save_json, token_accuracy  # noqa: E402

PAD = 0  # targets are 0 on the ignored first half, 1..V-1 on the scored second half


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=16, help="logical sequence length; the model sees 2x this")
    p.add_argument("--vocab", type=int, default=32, help="tokens are 1..vocab-1")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--out", type=str, default="results/history.json")
    return p.parse_args()


def evaluate(model, x, y, seq_len, batch_size=256):
    """Held-out metrics. Only the second half (targets != PAD) is scored."""
    losses, n = [], x.shape[0]
    correct_tok, total_tok, exact = 0, 0, 0.0
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(xb)
        losses.append(float(logits.cross_entropy(yb, ignore_index=PAD).data))
        pred = logits.data.argmax(-1)
        correct_tok += int(((pred == yb) & (yb != PAD)).sum())
        total_tok += int((yb != PAD).sum())
        exact += exact_match(pred, yb, ignore_value=PAD) * xb.shape[0]
    return {
        "loss": float(np.mean(losses)),
        "token_acc": correct_tok / max(total_tok, 1),
        "exact_match": exact / n,
    }


def show_samples(model, x, y, seq_len, k=3):
    L = seq_len
    logits = model(x[:k])
    pred = logits.data.argmax(-1)
    for i in range(k):
        src = "".join(map(str, x[i, :L] % 10))
        want = "".join(map(str, y[i, L:] % 10))
        got = "".join(map(str, pred[i, L:] % 10))
        print(f"  input   {src}")
        print(f"  target  {want}   (reversal, scored)")
        print(f"  pred    {got}   {'OK' if got == want else ''}\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    L = args.seq_len

    train_x, train_y = make_dataset(20_000, L, args.vocab, seed=args.seed + 1)
    eval_x, eval_y = make_dataset(args.eval_n, L, args.vocab, seed=args.seed + 99_999)

    model = TinyGPT(vocab_size=args.vocab, dim=args.dim, n_heads=args.heads,
                    n_layers=args.layers, block_size=2 * L)
    opt = Adam(model.parameters(), lr=args.lr)

    print(f"model: {model.num_params():,} params | "
          f"{args.layers} layers x {args.heads} heads | dim {args.dim} | context {2 * L}")
    print(f"task: read a {L}-token sequence, emit its reversal over the repeated copy "
          f"(vocab {args.vocab - 1})")
    print(f"baseline: chance token accuracy on the scored half = {1 / (args.vocab - 1):.4f}\n")

    history = {"config": vars(args), "steps": [], "train_loss": [],
               "eval_loss": [], "eval_token_acc": [], "eval_exact_match": [], "grad_norm": []}

    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb, yb = make_batch(rng, args.batch_size, L, args.vocab)
        loss = model.loss(xb, yb, ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        gnorm = clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            metrics = evaluate(model, eval_x, eval_y, L)
            history["steps"].append(step)
            history["train_loss"].append(float(loss.data))
            history["eval_loss"].append(metrics["loss"])
            history["eval_token_acc"].append(metrics["token_acc"])
            history["eval_exact_match"].append(metrics["exact_match"])
            history["grad_norm"].append(gnorm)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train_loss {float(loss.data):.4f} | "
                  f"eval_loss {metrics['loss']:.4f} | "
                  f"tok_acc {metrics['token_acc']:.3f} | "
                  f"exact {metrics['exact_match']:.3f} | "
                  f"|g| {gnorm:6.2f} | {elapsed:6.1f}s", flush=True)

    final = evaluate(model, eval_x, eval_y, L)
    history["final"] = final
    history["wall_clock_sec"] = time.time() - t0
    history["params"] = model.num_params()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_json(args.out, history)

    print("\n" + "=" * 68)
    print(f"FINAL  loss {final['loss']:.4f} | token acc {final['token_acc']:.1%} | "
          f"exact-match {final['exact_match']:.1%}")
    print(f"chance token accuracy: {1 / (args.vocab - 1):.1%}")
    print(f"{model.num_params():,} params, {time.time() - t0:.1f}s wall clock (float64 NumPy, CPU)")
    print("=" * 68 + "\n")

    show_samples(model, eval_x, eval_y, L)


if __name__ == "__main__":
    main()
