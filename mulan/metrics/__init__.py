"""Scoring metrics for ΔΔG prediction.

    from mulan import metrics

    mp, ms, n_qual, n_degen = metrics.per_structure(pred, true, pdb, T=10)
    ci = metrics.per_structure_bootstrap(pred, true, pdb, threshold=10)

The headline is per-structure Spearman at T>=10 -- see `docs/SPLITS_AND_METRICS.md` for why,
and for the three checks to make before comparing any two numbers across papers.
"""

from .bootstrap import (
    BootstrapResult,
    cluster_bootstrap,
    paired_delta,
    per_structure_bootstrap,
    sign_test,
)
from .tiers import DEFAULT_TIER, Tier, describe, get_tier, list_tiers
from .core import (
    PS_THRESHOLDS,
    STRONG,
    TOPK,
    auroc,
    average_ranks,
    mae,
    mae_corr,
    pdb_of,
    pearson,
    per_structure,
    precision_recall_at_k,
    rmse,
    rmse_corr,
    spearman,
)

__all__ = [
    "PS_THRESHOLDS", "STRONG", "TOPK",
    "auroc", "average_ranks", "mae", "mae_corr", "pdb_of", "pearson", "per_structure",
    "precision_recall_at_k", "rmse", "rmse_corr", "spearman",
    "BootstrapResult", "cluster_bootstrap", "paired_delta", "per_structure_bootstrap",
    "sign_test",
    "DEFAULT_TIER", "Tier", "describe", "get_tier", "list_tiers",
]
