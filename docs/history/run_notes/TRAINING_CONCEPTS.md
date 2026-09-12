# Training Concepts — a walkthrough grounded in run3a

A self-contained explainer of how MuLAN training works, built up from the
concrete numbers in `results/run3a_prostt5_L7/` (ProstT5 layer-7, no
augmentation) and generalized into transferable deep-learning concepts.

Reference run: **run3a** — ΔΔG regression, `LightAttModel` trained from scratch,
769/157/174 seed-42 split, batch 32, 50 epochs, LR 5e-4, `ReduceLROnPlateau`,
early-stopping patience 10, MPS. Final **test** PCC 0.686 (NEGATIVE vs the 0.740
last-layer baseline).

---

## 1. Reading the training metrics

Two log line types appear once per epoch (`logging_strategy="epoch"`,
`eval_strategy="epoch"`):

**Eval line** (validation set, N=157, forward-only — no weight update):

| Metric | Meaning |
|---|---|
| `eval_loss` | Validation MSE. `loss ≈ rmse²` (confirms the loss is MSE). |
| `eval_mae` | Mean Absolute Error — avg miss in kcal/mol. |
| `eval_rmse` | Root Mean Squared Error — penalizes large errors; RMSE ≫ MAE ⇒ heavy-tailed errors. |
| `eval_pcc` | Pearson (linear) correlation of preds vs truth. |
| `eval_scc` | Spearman (rank) correlation. ~0 means no useful ranking. |
| `eval_runtime / *_per_second` | Eval throughput. |

**Train line** (during training):

| Metric | Meaning |
|---|---|
| `loss` | Epoch-average **training** loss across all 25 steps of that epoch. |
| `grad_norm` | Magnitude of the gradient. Very large ⇒ instability. |
| `learning_rate` | Current LR (changes when the scheduler fires). |

Early instability in run3a: epoch-1 train loss 195.6 (grad_norm 735), epoch-8
grad_norm 270,514. Caused by raw mid-layer ProstT5 activations having large
scale; recovers once LR halves. Not overfitting — see §6.

---

## 2. Data partitioning

Three-way split, fixed `set_seed(42)` (hardcoded in `scripts/train.py`), identical
across run1–run4 so only the embedding varies:

| Split | Rows | Purpose |
|---|---|---|
| Train | 769 | gradient updates |
| Validation (`eval_*`) | 157 | per-epoch monitoring, LR schedule, early stopping, best-checkpoint selection |
| Test | 174 | held out; scored once at the end |

Note: the log's **1444 unique sequence ids** is the *embedding cache* (every
distinct WT+mutant sequence embedded once, 1024-d), **not** the number of
training rows (1100 mutation records total).

How the numbers tie together:
- steps/epoch = `ceil(769 / 32)` = **25** → ×50 = **1250 total steps** (the `0/1250` bar)
- eval steps = `ceil(157 / 32)` = **5**;  `422.9 samples/s × 0.371 s ≈ 157` ✓
- test rows = `test_predictions.tsv` line count = 174 ✓

---

## 3. Where the hyperparameters live (and how they were chosen)

Three layers, with different override-ability:

1. **Model architecture → JSON** (`models/config/lightatt_default_config.json`,
   defaults in `mulan/config.py`): `hidden_size 64`, `kernel_sizes [1,5,9]`,
   dropout 0.1, etc. Change by editing/pointing to a different JSON.

2. **Training hyperparameters → CLI args** (`CustomisableTrainingArguments` in
   `mulan/train_utils.py`), parsed by HF `HfArgumentParser`:

   | Arg | Default | run3a |
   |---|---|---|
   | `num_epochs` | 30 | **50** |
   | `batch_size` | 8 | **32** |
   | `learning_rate` | 5e-4 | 5e-4 |
   | `early_stopping_patience` | None | **10** |

3. **Hardcoded in `scripts/train.py`** (need a code edit): `set_seed(42)`,
   AdamW `weight_decay=0.01`, `ReduceLROnPlateau(mode="min", factor=0.5, patience=5)`,
   **MSE loss** (`MulanTrainer.compute_loss`), `metric_for_best_model="loss"`,
   `logging/eval/save_strategy="epoch"`, `save_total_limit=2`,
   `load_best_model_at_end`.

**Provenance:** This repo is a fork of upstream MuLAN (added MPS support + the
ProstT5/Ankh layer & augmentation experiments). The optimizer/scheduler/loss/arch
defaults are **inherited from the published MuLAN baseline**, kept fixed so the
layer/augmentation experiments stay comparable. The per-run knobs (50 epochs,
batch 32, patience 10) were chosen for these runs and held identical across
run1–run4. Caveat: there is **no gradient-clipping** knob exposed
(`max_grad_norm` isn't wired in), which is why the early instability isn't
damped — addressing it (e.g. normalizing cached mid-layer features) requires
editing `train.py`.

---

## 4. Steps vs epochs — and when backprop / weight updates happen

Hierarchy:

```
sample (1 mutation row)
   ↓ ×32  (batch_size)
batch  = 1 STEP = 1 weight update      ← backprop + optimizer.step() HERE
   ↓ ×25  (ceil(769/32))
epoch  = one full pass over all 769 train samples
   ↓ ×50
run    = 1250 total steps
```

Per **step** the trainer does:
1. **Forward** — push the batch of 32 embeddings through the model → 32 predictions.
2. **Loss** — `MSE(preds, labels)` averaged over the 32 → one scalar.
3. **Backprop** — `loss.backward()`: gradient of that scalar w.r.t. every weight
   (chain rule, backward through the net; ~one forward pass of cost).
4. **Optimizer step** — `optimizer.step()`: AdamW updates every weight. **← the actual update.**
5. **Zero grads** for the next batch.

Key points:
- Weights update **once per step**, not per sample and not per epoch. run3a
  updates weights **1250 times** (25× per epoch). The 32 samples in a batch are
  averaged into **one** gradient → **one** update.
- The per-epoch `'loss'` log line is the **average of that epoch's 25 updates**,
  which already happened — you see one line but 25 backprop+update cycles occurred.
- Cadence of the moving parts:
  - **AdamW** updates every **step** (1250×).
  - **`ReduceLROnPlateau`** checks the metric every **epoch**; after `patience=5`
    epochs without improvement, halves LR. This produced 5e-4 → 2.5e-4 (~ep 12)
    → 1.25e-4 (~ep 46).
  - **Early stopping** also counts in **epochs**.
- Trade-off: larger batch ⇒ fewer, smoother updates/epoch (needs more epochs or
  higher LR); smaller batch ⇒ more, noisier updates/epoch (noise can regularize).

---

## 5. The general AI lesson — the universal training loop

Every deep-learning model (a 64-unit `LightAttModel` or a frontier LLM) trains
with the same loop; only scale, loss, and parallelism change.

```
initialize weights θ
for each epoch:
    for each mini-batch in training_data:          # one STEP
        predictions = forward(batch, θ)
        loss        = L(predictions, targets)
        gradient    = backprop(loss)               # compute direction
        θ           = optimizer.step(θ, gradient)  # UPDATE weights
    metric = evaluate(validation_data)             # forward only, no update
    scheduler.step(metric)                          # maybe lower LR
    if no improvement for `patience` epochs: stop   # early stopping
report performance on the untouched test set
```

Concepts introduced:
- **Loss function** — single number measuring wrongness. Regression → MSE/MAE
  (run3a); classification → cross-entropy; language modeling → cross-entropy over vocab.
- **Gradient** (∂L/∂θ) — direction of steepest *increase*; step the opposite way.
- **Backpropagation** — the efficient algorithm that *computes* the gradient
  (chain rule backward). Backprop ≠ learning; it's the gradient computation the update consumes.
- **Learning rate** — step size. Too big → diverge (run3a's early blow-ups);
  too small → slow. The most important single hyperparameter.
- **Mini-batch / SGD** — compromise between full-batch (accurate, slow) and
  per-sample (fast, noisy). Defines "step ≠ epoch."
- **Optimizer (AdamW)** — adds momentum (running avg of gradients) + per-weight
  adaptive scaling + weight decay (regularization). Acts once per step.
- **LR schedule** — change LR over time (run3a: `ReduceLROnPlateau`).
- **Generalization** — the only thing that matters: performance on data never
  trained on.

Across GPT/Claude/vision/`LightAttModel`, what changes is only: **scale**
(params, steps — frontier runs do millions of steps on thousands of GPUs),
the **loss**, **parallelism** (a step may be split across devices, gradients
averaged before one update), and **schedule/optimizer details** (warmup, cosine
decay, gradient clipping).

---

## 6. Early stopping & overfitting metrics

### What's watched
Early stopping watches a **held-out** metric, never training loss. run3a:

```python
metric_for_best_model="loss"   # → eval_loss on the 157-sample validation set
load_best_model_at_end=True
EarlyStoppingCallback(patience=10)
```

| Concept | Meaning | run3a |
|---|---|---|
| Monitored metric | held-out number defining "better" | `eval_loss` |
| Mode (min/max) | lower-better (loss) vs higher-better (PCC) | `min` |
| `min_delta` | improvement that *counts* (ignore wiggles) | 0 |
| `patience` | epochs allowed without improvement before stopping | 10 |
| Best checkpoint | snapshot at best metric, restored at end | `load_best_model_at_end` |

### Algorithm (per epoch)
```
if eval_loss < best_loss − min_delta:
    best_loss = eval_loss; save "best" checkpoint; counter = 0
else:
    counter += 1
    if counter >= patience (10): STOP
at end: reload "best" checkpoint, not the last one
```
Buys you (1) a stopping rule and (2) checkpoint selection — deploy the best-eval
epoch, not whatever epoch you stopped on.

**Did it fire in run3a?** No. All 50 epochs / 1250 steps completed and `eval_loss`
was still at its minimum (2.94) at epoch 50 — validation never went 10 epochs
without improving. The model was still learning when the epoch budget ran out
(a hint more epochs might help slightly).

### Overfitting = train-vs-validation divergence
Diagnosed by comparing two curves, not one number:
- **Training loss** keeps falling (can memorize).
- **Validation loss** falls while learning generalizable structure, then **turns
  up** when it starts memorizing noise. The "elbow" = best generalization.
  Early stopping = "stop at the elbow"; patience avoids overreacting to a blip.

run3a diagnostic:

| Epoch | Train loss | `eval_loss` | `eval_pcc` |
|---|---|---|---|
| 1 | 195.6 | 8.32 | 0.29 |
| 11 | 19.2 | 6.97 | 0.42 |
| 12 | 18.8 | 6.11 | 0.51 |
| 34 | ~3.6 | 4.17 | 0.69 |
| 49 | ~2.9 | 2.93 | 0.79 |
| 50 | — | 2.94 | 0.80 |

Healthy, **non-overfit** run: validation loss still descending at epoch 50 (no
elbow), and train (~2.9) ≈ val (~2.94) — tiny gap. (The huge *early* gap 196 vs 8
was instability, not overfitting; it closed.) run3a's problem isn't overfitting —
its best achievable generalization (test PCC 0.686) is simply **below** the
0.740 baseline: layer-7 features are weaker, not memorized.

### Caveat: metrics can disagree
Early stopping optimizes whatever metric you name. Epoch 9 in run3a:
`eval_loss 7.16` (improving) but `eval_pcc 0.35` (dropped) while `eval_scc 0.48`
(rose). Selecting on `loss` may not pick the best-correlation checkpoint. If
ranking is the goal, set `metric_for_best_model="pcc"` (mode `max`) — possibly a
different epoch.

### Two-set vs three-set
Early stopping *tunes to* the validation set, so eval numbers are optimistic —
hence the separate **test** set scored once. run3a: validation PCC peaked at
**0.80**, true **test** PCC was **0.686**. Overfitting to the train set is
caught by the train–val gap; overfitting to the val set (from tuning pressure)
is caught only by the untouched test set.

The tying rule:
- **Train metric** → did it *learn* the data?
- **Validation metric** → *when to stop / which checkpoint / which hyperparameters*
- **Test metric** → *how good is it really* (touched once, used for no decision)

---

## 7. Cross-validation (CV10) — why one split isn't enough

Everything above (run3a) uses a **single holdout split**: one fixed
769/157/174 partition (`num_folds=1`). That gives **one** test number, and that
number depends on *which* 174 rows happened to land in the test set. With only
~1100 mutation records, a lucky or unlucky split can move the test PCC by several
hundredths — so a single split has **high variance** and isn't a fair point of
comparison against the published baseline.

The paper's headline numbers (Ankh-large **PCC 0.868 / RMSE 1.185**) are *not* a
single split — they are a **10-fold cross-validation (CV10) average**. That is
the apples-to-apples target every run in `results/` is implicitly measured
against, and why those docs keep flagging "single holdout, higher variance."

### What CV10 does
`mulan.data.split_data(mutated_complexes_file, num_folds=10, random_state=42)`
partitions the data into **10 equal folds**, then trains **10 separate models**,
rotating which fold is held out:

```
fold_index = shuffled labels 0..9, one per row   (random_state=42 → reproducible)
for test_fold in 0..9:                            # 10 independent trainings
    test  = rows where fold_index == test_fold              # 1 fold  (~10%)
    val   = rows where fold_index == (test_fold - 1) % 10   # 1 fold  (~10%, the previous one)
    train = the remaining 8 folds                            # 8 folds (~80%)
```

So per fold the split is **≈8/1/1** (train/val/test), e.g. ~880/110/110 for the
~1100-record set, vs. the single-split 769/157/174. The validation fold is just
the **previous** fold in the rotation, so train/val/test stay disjoint within
each fold. `split_data` writes `output_dir/fold_0/ … fold_9/`, each containing
its own `*_train.tsv` / `*_val.tsv` / `*_test.tsv`.

### The key property: every row is tested exactly once
Across the 10 folds, each fold serves as the test set **exactly once**, so every
data point gets a held-out prediction from a model that never trained on it. The
reported metric is the **average of the 10 fold test scores** (often with their
spread). That average:
- **uses all the data for evaluation** (not just one 15% slice), and
- **averages out split luck** → a far more stable, trustworthy estimate than any
  single holdout.

That's the whole point: CV10 trades **10× compute** for a **low-variance** number.

### Why these runs used a single split anyway
A single split was a deliberate cost trade-off for *screening* layer/augmentation
ideas: one training instead of ten. The catch is the comparison — a single-split
PCC (e.g. run3a's 0.686, or run4a's 0.757) **cannot** be directly ranked against
the paper's 0.868 CV10 average, because the gap could be real *or* just split
variance. To make any result here a publishable claim, the standing
recommendation in the results docs is to **re-run the winning configuration under
CV10** for a variance-controlled number. Budget roughly **~5.5 h per PLM** for all
10 folds (≈10× a single ~30–48 min run).

### The general lesson
Cross-validation is the standard answer to "**is this number real or did I get
lucky with the split?**" — most relevant when data is **scarce** (every row is
precious) and when you're **comparing configurations** whose true gap may be
smaller than split noise. The cost is linear in the number of folds, so it's used
for final/reported numbers, not for every quick experiment. (Frontier-scale
models skip k-fold entirely: with massive datasets a single held-out split is
already low-variance, and 10× the training cost is prohibitive.)
