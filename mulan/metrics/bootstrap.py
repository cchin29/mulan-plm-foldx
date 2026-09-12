"""Cluster bootstrap over complexes — one implementation.

There were three. The original lived in ``experiments/rescore_perstructure/bootstrap_ci.py`` and
hardcoded an absolute path to one machine's filesystem at module scope, so it could not be
imported anywhere. ``experiments/retrain_split/evaluate.py`` therefore re-implemented it — its
docstring says so explicitly — and ``score_multipoint.py`` then imported *that* copy.

Resampling is over **complexes, not mutations**. Mutations within a complex are strongly
dependent: they share an interface, a structure, and a measurement protocol. Treating them as
independent samples produces confidence intervals that are far too narrow, which matters here
because the headline metric is itself a per-complex average and the number of qualifying
complexes is small — tens, not thousands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import per_structure


@dataclass
class BootstrapResult:
    point: float
    """The statistic on the observed data."""
    lower: float
    upper: float
    percentile: float
    n_clusters: int
    n_resamples: int

    def __str__(self) -> str:
        return (f"{self.point:.4f} "
                f"[{self.lower:.4f}, {self.upper:.4f}] "
                f"{self.percentile:.0f}% CI over {self.n_clusters} complexes")


def cluster_bootstrap(
    statistic: Callable[..., float],
    groups: Sequence,
    n_resamples: int = 10000,
    percentile: float = 95.0,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap ``statistic`` by resampling whole groups with replacement.

    ``statistic`` receives ``(row_indices, group_labels)`` — the labels for those rows, **not**
    the original ones. That second argument is load-bearing: a cluster drawn twice must count
    twice.

    It previously received only indices and re-derived groups from the data, so a duplicated
    cluster collapsed back into a single group with double the rows. Per-complex correlation is
    invariant to exact duplication, so the duplicate contributed weight 1 instead of 2, making
    each resample a random *subset* of ~63.2% of clusters rather than a bootstrap over n. The
    resulting intervals were ~24% too narrow — anti-conservative — and a per-complex minimum
    like ``T>=10`` was applied to the inflated row count, so a 6-mutation complex drawn twice
    qualified. Relabelling each draw fixes both.

    Resamples producing a non-finite statistic are dropped, and the interval is reported over
    what remains.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}

    point = statistic(np.arange(len(groups)), groups)
    rng = np.random.default_rng(seed)
    draws: List[float] = []
    for _ in range(n_resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        # Distinct label per draw, so a cluster picked twice is two clusters downstream.
        labels = np.concatenate([
            np.full(len(index_by_group[g]), f"{g}#{j}") for j, g in enumerate(picked)])
        value = statistic(idx, labels)
        if np.isfinite(value):
            draws.append(value)

    if not draws:
        nan = float("nan")
        return BootstrapResult(point, nan, nan, percentile, len(unique), 0)

    tail = (100.0 - percentile) / 2.0
    lower, upper = np.percentile(draws, [tail, 100.0 - tail])
    return BootstrapResult(point, float(lower), float(upper), percentile,
                           len(unique), len(draws))


def per_structure_bootstrap(
    pred: np.ndarray,
    true: np.ndarray,
    pdb: np.ndarray,
    threshold: int = 10,
    n_resamples: int = 10000,
    percentile: float = 95.0,
    seed: int = 0,
) -> BootstrapResult:
    """Confidence interval for per-structure Spearman at ``threshold``."""
    pred, true, pdb = np.asarray(pred, float), np.asarray(true, float), np.asarray(pdb)

    def statistic(idx: np.ndarray, labels=None) -> float:
        g = pdb[idx] if labels is None else labels
        _, spearman_mean, _, _ = per_structure(pred[idx], true[idx], g, threshold)
        return spearman_mean

    return cluster_bootstrap(statistic, pdb, n_resamples, percentile, seed)


def paired_delta(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    pdb: np.ndarray,
    threshold: int = 10,
    n_resamples: int = 10000,
    percentile: float = 95.0,
    seed: int = 0,
) -> BootstrapResult:
    """Interval for ``metric(a) - metric(b)`` on the **same** complexes.

    Paired, because the two arms are evaluated on identical rows. An unpaired comparison of two
    intervals is the wrong test and is much less sensitive: most of the variance is
    between-complex difficulty, which cancels in the difference.
    """
    pred_a, pred_b = np.asarray(pred_a, float), np.asarray(pred_b, float)
    true, pdb = np.asarray(true, float), np.asarray(pdb)

    def statistic(idx: np.ndarray, labels=None) -> float:
        g = np.asarray(pdb[idx] if labels is None else labels)
        # Restrict both arms to the complexes BOTH can score. `per_structure` drops
        # zero-variance complexes independently per arm, so differencing its two means was a
        # difference over different complex sets -- which reintroduces exactly the
        # between-complex difficulty variance a paired contrast exists to cancel.
        keep = _mutually_scorable(pred_a[idx], pred_b[idx], true[idx], g, threshold)
        if not keep.any():
            return float("nan")
        _, sa, _, _ = per_structure(pred_a[idx][keep], true[idx][keep], g[keep], threshold)
        _, sb, _, _ = per_structure(pred_b[idx][keep], true[idx][keep], g[keep], threshold)
        return sa - sb

    return cluster_bootstrap(statistic, pdb, n_resamples, percentile, seed)


def _mutually_scorable(pred_a, pred_b, true, groups, threshold) -> np.ndarray:
    """Row mask for complexes that BOTH arms can score: size >= threshold and non-degenerate
    in the truth and in each arm's predictions."""
    keep = np.zeros(len(groups), dtype=bool)
    for key in np.unique(groups):
        m = groups == key
        if int(m.sum()) < threshold:
            continue
        if np.std(true[m]) == 0 or np.std(pred_a[m]) == 0 or np.std(pred_b[m]) == 0:
            continue
        if not (np.all(np.isfinite(pred_a[m])) and np.all(np.isfinite(pred_b[m]))
                and np.all(np.isfinite(true[m]))):
            continue
        keep |= m
    return keep


def sign_test(deltas: Sequence[float]) -> Tuple[int, int, float]:
    """Two-sided sign test over per-fold deltas: ``(n_positive, n_total, p)``.

    Used for the "does this arm beat that one across folds" question, where the number of folds
    is small enough that a distributional assumption is not worth making.
    """
    from math import comb

    finite = [d for d in deltas if np.isfinite(d) and d != 0]
    n = len(finite)
    if n == 0:
        return 0, 0, float("nan")
    k = sum(1 for d in finite if d > 0)
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / (2 ** n)
    return k, n, min(1.0, 2 * tail)
