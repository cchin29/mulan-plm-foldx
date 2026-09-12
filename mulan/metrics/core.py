"""The scoring metrics, as a library.

These were defined inside ``experiments/rescore_perstructure/rescore.py``, a script. Every
scorer imported it by inserting its directory onto ``sys.path`` — which worked, but meant the
metrics could not be imported without also importing a module that resolves result-tree paths at
import time and defaults to the *leaky* 10-fold tier.

The function bodies are unchanged. They are deliberately numpy-only, with no scipy dependency,
and every ranking is tie-aware (average ranks) and deterministic (stable sorts), so a rerun on
the same predictions reproduces the same number exactly.

The headline metric of this project is :func:`per_structure` at ``T=10``: group test predictions
by complex, correlate within each complex having at least ten mutations, then average. A pooled
correlation over all test mutations is dominated by *between*-complex variance, so a model that
ranks complexes correctly but ranks mutations within a complex at chance still scores well.
Per-structure correlation asks the question a protein engineer actually has.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_ranks", "pearson", "spearman", "rmse", "mae", "rmse_corr", "mae_corr",
    "auroc", "precision_recall_at_k", "per_structure", "pdb_of",
    "STRONG", "TOPK", "PS_THRESHOLDS",
]

#: |ddG| at or above which a mutation counts as a strong destabilizer, for AUROC.
STRONG = 2.0
#: Cutoff for precision/recall at k.
TOPK = 50
#: Minimum mutations per complex for the per-structure metric. 10 is the headline, matching the
#: convention the published frontier reports; 5 is carried as a secondary, more inclusive read.
PS_THRESHOLDS = (10, 5)


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Average (fractional) ranks, 1..n, with ties assigned the mean of their positions."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")   # stable
    ranks = np.empty(n, dtype=float)
    sx = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        # positions i..j-1 (0-based) -> ranks i+1..j (1-based); mean rank for the tie block
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def degenerate(a: np.ndarray) -> bool:
    """True if `a` carries no usable ordering, so a correlation against it is meaningless.

    NOT ``np.std(a) == 0``. For an exactly-constant NONZERO array that test is often False:
    summing n copies of the value accumulates rounding, so the computed mean is off by an ulp and
    np.std returns ~1e-15 rather than 0. The constant array then reaches the correlation, which
    returns nan, and one nan poisons the mean over every complex in the tier -- silently, and
    without incrementing the degenerate count that would have disclosed it. It never surfaced
    while the only constant predictions were exactly 0.0, whose mean IS exact; a predictor whose
    within-complex constant is a nonzero intercept is the case that breaks it.

    Compare spread against scale instead, which is convention-free.
    """
    a = np.asarray(a, dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        return True
    return bool(np.ptp(a) <= 1e-12 * max(1.0, float(np.max(np.abs(a)))))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or degenerate(a) or degenerate(b):
        return float("nan")
    am, bm = a - a.mean(), b - b.mean()
    denom = np.sqrt((am * am).sum() * (bm * bm).sum())
    if denom == 0:
        return float("nan")
    return float((am * bm).sum() / denom)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman = Pearson on average-rank-transformed values (mean ranks for ties).

    Non-finite input yields NaN. `average_ranks` sorts NaN to the end and hands back a finite
    rank, so without this guard a single NaN prediction produced a confident-looking
    correlation — one NaN in a 10-mutation complex returned a perfect 1.0 while Pearson
    correctly returned NaN. Spearman is this project's headline metric, so it was the one that
    hid the problem.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    return pearson(average_ranks(a), average_ranks(b))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    d = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    d = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return float(np.mean(np.abs(d)))


def _linear_corrected(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """OLS fit true ~ a*pred + b, return the corrected prediction a*pred + b.

    Matches the RDE-PPI reference `overall_rmse_mae` (LinearRegression(pred -> true)) with a
    dependency-free closed form. Degenerate (zero-variance pred) -> constant mean(true).
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    vp = np.var(pred)
    if vp == 0:
        return np.full_like(true, true.mean())
    a = np.cov(pred, true, bias=True)[0, 1] / vp
    b = true.mean() - a * pred.mean()
    return a * pred + b


def rmse_corr(pred: np.ndarray, true: np.ndarray) -> float:
    """RMSE of the OLS-rescaled prediction (RDE/leaderboard convention: scale+offset removed)."""
    pc = _linear_corrected(pred, true)
    d = pc - np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def mae_corr(pred: np.ndarray, true: np.ndarray) -> float:
    """MAE of the OLS-rescaled prediction (RDE/leaderboard convention: scale+offset removed)."""
    pc = _linear_corrected(pred, true)
    return float(np.mean(np.abs(pc - np.asarray(true, dtype=float))))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the rank / Mann-Whitney identity, tie-aware (mean ranks).

        AUROC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    Larger score => predicted positive. Returns NaN if either class is empty.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    # A NaN score ranks top, and `astype(bool)` turns NaN into True — so a label column with
    # missing values silently became all-positive. Reject non-finite input instead.
    if not np.all(np.isfinite(scores)):
        return float("nan")
    if labels.dtype.kind == "f" and not np.all(np.isfinite(labels)):
        return float("nan")
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = average_ranks(scores)
    r_pos = r[labels].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_at_k(pred: np.ndarray, true: np.ndarray, k: int, thr: float):
    """Rank by pred desc (stable, ties broken by original index), take top-k.

    precision = fraction of the top-k that are truly (true >= thr).
    recall    = of all (true >= thr), fraction captured in the top-k.
    Deterministic: stable argsort on -pred keeps original order within ties.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    pos = true >= thr
    n_pos = int(pos.sum())
    order = np.argsort(-pred, kind="mergesort")   # stable => deterministic tie-break
    k_eff = min(k, pred.size)
    topk = order[:k_eff]
    hits = int(pos[topk].sum())
    precision = hits / k_eff if k_eff > 0 else float("nan")
    recall = hits / n_pos if n_pos > 0 else float("nan")
    return precision, recall, n_pos


def per_structure(pred: np.ndarray, true: np.ndarray, pdb: np.ndarray, T: int):
    """Mean within-complex Pearson & Spearman over complexes with n_mut >= T.

    Skips complexes where true or pred has zero variance (degenerate correlation) and
    counts them. Returns (mean_pearson, mean_spearman, n_qualifying, n_skipped_degenerate).
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    pdb = np.asarray(pdb)
    pearsons, spearmans = [], []
    n_qual = 0
    n_degen = 0
    for key in np.unique(pdb):
        m = pdb == key
        if int(m.sum()) < T:
            continue
        n_qual += 1
        p, t = pred[m], true[m]
        if degenerate(p) or degenerate(t):
            n_degen += 1
            continue
        pearsons.append(pearson(p, t))
        spearmans.append(spearman(p, t))
    mp = float(np.mean(pearsons)) if pearsons else float("nan")
    ms = float(np.mean(spearmans)) if spearmans else float("nan")
    return mp, ms, n_qual, n_degen


def _pdb_of(chain1_id: str) -> str:
    return chain1_id.split("_")[0]


def pdb_of(chain1_id: str) -> str:
    """The complex key for grouping: everything before the first underscore."""
    return _pdb_of(chain1_id)
