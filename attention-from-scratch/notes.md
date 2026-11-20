# Build log — tinytensor

Dated notes on decisions, failures and fixes. Kept deliberately raw: this is the
record I will use to answer "walk me through how you built it" truthfully, and
the fix history is also encoded as regression tests.

---

## 2026-11-20 — design choices before writing code

**Goal:** one small, real, measured repo demonstrating *Evaluation* and
*Data & Feature Engineering* fundamentals from the ML-engineer listing, at a
depth that survives 2–3 layers of interview probing. Chose a from-scratch
autograd engine + transformer over a PyTorch tutorial clone because:

- no GPU is available, so correctness (float64 + finite-difference gradient
  checks) is a stronger differentiator than throughput;
- every line must be explainable — a hand-written engine leaves nowhere to hide;
- the task (sequence reversal) has an exact metric and a provable ablation.

**Decisions:**
1. float64 everywhere — finite differences at h=1e-6 then agree with analytic
   gradients to ~7 digits, so tests can be strict (rtol 1e-6), not generous.
2. Gradients accumulate into `.grad` (never overwrite) — fan-out correctness.
3. Pre-LN GPT block; separate Q/K/V projections instead of fused QKV — slower,
   but the graph stays readable, which is the point here.
4. Softmax via max-shift with the shift treated as a constant: exact, and it
   carries no gradient path. (Checked against overflow at logits ≈ 1000.)

## 2026-11-20 — engine built, tests green, then training OOM-killed

Engine + model + training script done; 20 gradient/behavioural tests passing.
First real training run: **the process was OOM-killed at ~step 40, batch 64.**

Measurement (RSS via `/proc/self/status`): start 36 MB → 602 MB @ step 10 →
1596 MB @ step 40 — roughly **35 MB/step leaked**. This is a 105K-param model;
the parameters are ~1 MB. The *graph*, not the model, was leaking.

## 2026-11-20 — leak diagnosis

- `gc.collect()` between steps held RSS flat → the objects were in cycles, not
  genuinely referenced.
- `gc.get_objects()` sweep: live graph tensors grew ~179/step; after
  `loss = None; gc.collect()` they all vanished — so the cycle chain was intact
  per-step.
- Referrer-chain walk (`gc.get_referrers`) on a leaked attention intermediate
  ended at a **function: `Tensor.backward.<locals>.build`** — the recursive
  toposort closure.

**Two cycles, both invisible in the loss curve:**
1. Every op's `_backward` closure captured its own output tensor:
   `out → _backward → closure → out`. A self-cycle per node, per step.
2. `build()` referenced itself for recursion, so Python captured `build` in its
   own closure cell; that cell also referenced the enclosing scope, whose `topo`
   list held every tensor in the graph. One cycle per backward call, containing
   the whole graph.

**Fixes:**
- backward rules take the output as an argument: `_bw(out)`, never capture it.
- toposort rewritten iterative (explicit stack, post-order). Side benefit: no
  recursion limit — the recursive version failed at ~1000 chained ops; the
  iterative version handles 1200 (now a test).

**Verification:** 0 live `Tensor` objects after 50 engine-only steps and 60 full
training steps, with `gc` *observed but never forced* during the loop. Encoded
as `tests/test_memory.py`.

## 2026-11-20 — the rewrite broke toposort ordering, caught by tests

First iterative version marked nodes "seen" **at push time**. That is wrong for
shared subgraphs: a child pushed by two parents (e.g. the softmax diamond where
`exp` output feeds both `div` and `sum`) could be completed via one parent and
then skipped for the other, emitting it before an ancestor — wrong topological
order. Three finite-difference tests failed instantly (softmax, LayerNorm,
attention).

**Fix:** mark seen at expansion time; skip stale pushes. All 20 tests green.
Lesson, twice over: **gradient checks are not decoration** — they caught a bug
that produced *plausible* numbers.

## 2026-11-20 — the 51.4% plateau (task-design error)

Full run converged — to **51.4% token accuracy, 0% exact match**, with every
sample correct on the second half of the output and constant on the first half:

```
in    7936519893500075
want  5700053989156397
got   2222222289156397      ← second half perfect, first half garbage
```

This is not undertraining: **output position i can never see input position
L−1−i for i < L/2 under the causal mask.** The information is mathematically
unavailable. The model had learned exactly the half it could, and the loss curve
looked like clean convergence. A naive "train longer / bigger model" response
would have shipped a broken project with a tidy graph.

**Fix (standard in copy/induction benchmarks):** show the sequence twice —
read the first copy, write the reversal over the second — and score only the
second half (`ignore_index=0` in cross-entropy).

**Result:** 100% exact match on 512 held-out sequences by step 30 (2-layer;
1-layer also 100%). The failure and fix are documented in `data.py` and the
README, because explaining *why* the task is shaped this way is precisely the
kind of question an interviewer asks.

## 2026-11-20 — ablation: position-free baseline

Bag-of-tokens MLP (input = sum of one-hots ⇒ order-blind by construction):
**15.1% token accuracy, 0% exact match** after 300 steps. Initially expected
chance (~3.2%); the measured 15% is actually the *theoretical* ceiling — with no
order information the best strategy is predicting the most frequent bag token at
every position, scoring E[max_v count(v)]/L ≈ 15%. The baseline is at its ceiling,
which makes the ablation sharper: 0% vs 100% exact match, attributable to exactly
one variable — access to order.

## Follow-ups (planned, in priority order)

1. Per-position accuracy breakdown + attention-pattern figure (which heads align
   position i → L−1−i?) — visual proof the attention path does the work.
2. Length generalisation: train at L=16, evaluate L∈{16,24,32} — known failure
   mode of position embeddings; would strengthen the "evaluation" angle.
3. AdamW + dropout for a more complete optimiser surface.
4. float32/BLAS benchmark column to quantify the float64 cost honestly.
5. Generalise `data.py` into a small task suite (reverse / copy / sort) with a
   shared eval harness — the seed of a real evaluation library.
