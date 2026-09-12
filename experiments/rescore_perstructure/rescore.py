#!/usr/bin/env python3
"""Re-score existing balanced-splitter OOF predictions under frontier-standard metrics.

Implements docs/history/RESCORE_PERSTRUCTURE_SPEC.md: per-structure Pearson/Spearman,
overall Pearson/Spearman/RMSE/MAE, AUROC (destabilizer sign + strong destabilizer), and
top-50 precision/recall — pooled over the 10 disjoint balanced test folds (1100 OOF rows).

SCOPE CAVEAT (repeated everywhere by design): this changes the METRIC, not the SPLIT. The
predictions come from the leaky per-mutation 10-fold CV (same complex in train and test), so
the per-structure numbers are an UPPER BOUND, not a generalization figure. Leakage-controlled by-complex
(RDE 3-fold) / CATH-clustered numbers still require retraining (a separate task).

Metric core is dependency-free (numpy only) so it can be reused unchanged on future RDE-split
predictions. scipy is used ONLY under --selftest to cross-validate the hand-rolled metrics.

Run:  experiments/rescore_perstructure/.venv/bin/python experiments/rescore_perstructure/rescore.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The metric definitions now live in `mulan.metrics`, so they can be imported without importing
# this script (which resolves result-tree paths at import time and defaults to the leaky tier).
# Re-exported here unchanged so every existing caller keeps working.
from mulan.metrics import (  # noqa: E402,F401
    average_ranks, pearson, spearman, rmse, mae, rmse_corr, mae_corr,
    auroc, precision_recall_at_k, per_structure,
)

# ---------------------------------------------------------------------------
# Paths (relative to repo root). Repo root = three levels up from this file:
#   <root>/experiments/rescore_perstructure/rescore.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "scratch/results/embedding_sweep_balanced"
INC = ROOT / "scratch/incoming_esmc6b_20260714"
OUT = ROOT / "experiments/rescore_perstructure"

# The evaluation tier. Defaults to the CATH-superfamily hold-out -- the strictest protocol and
# the one comparable to published tables.
#
# This used to default to the LEAKY 10-fold splits, which meant the metric core's default
# behaviour was the one protocol no claim rests on, and a scorer that forgot to override scored
# against the wrong truth while reporting a perfectly normal-looking number. The leaky tier is
# still reachable -- reproducing the upstream paper needs it -- but only by naming it:
#
#     rescore.use_tier("legacy_cv10")
#
# Scorers that assign to TRUTH_TMPL / N_FOLDS directly keep working unchanged.
from mulan.metrics import get_tier  # noqa: E402

_TIER = get_tier()                                   # DEFAULT_TIER == "cath"
TRUTH_TMPL = _TIER.truth_template(ROOT)
N_FOLDS = _TIER.n_folds


def use_tier(name=None, test_file=None):
    """Point the module at an evaluation tier by name; returns the resolved Tier.

    `name=None` selects the default. Legacy tiers must be named explicitly.
    """
    global _TIER, TRUTH_TMPL, N_FOLDS
    _TIER = get_tier(name)
    TRUTH_TMPL = _TIER.truth_template(ROOT, test_file)
    N_FOLDS = _TIER.n_folds
    return _TIER


def current_tier():
    """The tier in effect, or None if the globals were reassigned directly.

    The four scorers set TRUTH_TMPL / N_FOLDS themselves rather than calling use_tier(). That
    still works, but it means this function must not claim a tier it no longer describes -- a
    diagnostic that says "cath" while scoring against a 3-fold clustered split is worse than no
    diagnostic at all.
    """
    if _TIER is None:
        return None
    if str(TRUTH_TMPL) != _TIER.truth_template(ROOT) or N_FOLDS != _TIER.n_folds:
        return None
    return _TIER

STRONG = 2.0          # |ddg| threshold for "strong destabilizer"
TOPK = 50             # precision@k / recall@k
PS_THRESHOLDS = (10, 5)   # per-structure min mutations: T=10 primary, T=5 secondary

# The 6 PLMs that make up each consensus family (fixed membership; all-6-required).
CONSENSUS_PLMS = ("esm2", "esmc600m", "prostt5", "saprot", "ankh", "esmc6b")


def _plm_path(plm: str, family: str) -> Path:
    """Path template for a single-PLM arm. family in {base, aug, foldxmlp, foldx_scalar}."""
    if plm == "esmc6b":
        if family == "base":
            return INC / "results/base/S1102/fold_{f}/training_run/test_predictions.tsv"
        if family == "aug":
            return INC / "results/aug/S1102/fold_{f}/training_run/test_predictions.tsv"
        if family == "foldx_scalar":
            # Stage-1 scalar Interaction-Energy arm: the already-trained balanced scalar preds
            # (sibling of foldx_mlp). Pure scoring — no training. Fills the leaky rung's scalar column.
            return INC / "data/S1102/foldx/cv10_esmc6b_balanced/foldx/fold_{f}/training_run/test_predictions.tsv"
        return INC / "data/S1102/foldx/cv10_esmc6b_balanced/foldx_mlp/fold_{f}/training_run/test_predictions.tsv"
    if plm == "ankh" and family == "base":
        # NOTE: ankh base lives under cv10_ankh_converge, NOT {RES}/ankh (which is empty).
        return ROOT / "scratch/results/cv10_ankh_converge/fold_{f}/training_run/test_predictions.tsv"
    suffix = plm if family == "base" else f"{plm}_{family}"
    return RES / f"{suffix}/fold_{{f}}/training_run/test_predictions.tsv"


# arm-id -> path template with a literal "{f}" fold placeholder. Consensus arms have no
# template (value None) and are assembled from their constituent PLM arms afterwards.
def build_arm_templates() -> dict[str, Path | None]:
    arms: dict[str, Path | None] = {}
    for plm in ("esm2", "esmc600m", "prostt5", "saprot", "ankh", "esmc6b"):
        for family in ("base", "aug", "foldxmlp"):
            arms[f"{plm}_{family}"] = _plm_path(plm, family)
    # esmc6b-only Stage-1 scalar FoldX arm (leaky-rung scalar fill; the other PLMs have no scalar
    # preds, so this arm exists for esmc6b alone). Arm id matches the frontier `<tag>_foldx_scalar`.
    arms["esmc6b_foldx_scalar"] = _plm_path("esmc6b", "foldx_scalar")
    arms["mint_complex"] = ROOT / "scratch/results/mint_a2/mint/fold_{f}/training_run/test_predictions.tsv"
    arms["mint_mono"] = ROOT / "scratch/results/mint_mono/mint_mono/fold_{f}/training_run/test_predictions.tsv"
    for family in ("base", "aug", "foldxmlp"):
        arms[f"consensus_{family}"] = None
    return arms


# ---------------------------------------------------------------------------
# Metric core — dependency-free (numpy only). Do NOT import scipy here.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _pdb_of(chain1_id: str) -> str:
    return chain1_id.split("_")[0]


def load_truth_folds():
    """Return list (len N_FOLDS) of dicts: (pdb, mutation) -> true_ddg.

    Asserts the (pdb, mutation) join key is unique within each fold (it drops chain2_id;
    a duplicate would make the truth<->pred join many-to-one and silently corrupt an arm).
    """
    folds = []
    for f in range(N_FOLDS):
        path = Path(str(TRUTH_TMPL).format(f=f))
        d: dict[tuple[str, str], float] = {}
        with open(path) as fh:
            for ln, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                cols = line.split("\t")
                pdb = _pdb_of(cols[0])
                mut = cols[2]
                key = (pdb, mut)
                if key in d:
                    raise ValueError(
                        f"Non-unique join key {key} in {path} (line {ln}); "
                        "(pdb, mutation) must be unique within a fold."
                    )
                d[key] = float(cols[3])
        folds.append(d)
    return folds


def load_pred_file(path: Path) -> dict[tuple[str, str], float]:
    """Read a 4-col pred TSV -> (pdb, mutation) -> pred. Missing file -> None."""
    if not path.exists():
        return None
    d: dict[tuple[str, str], float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            d[(_pdb_of(cols[0]), cols[2])] = float(cols[3])
    return d


def assemble_arm(pred_folds, truth_folds):
    """Join per-fold pred dicts to truth on (pdb, mutation), pool across folds.

    Returns (true, pred, pdb) numpy arrays and a list of warnings.
    pred_folds: list (len N_FOLDS) of pred dicts (or None where a fold is missing).
    """
    trues, preds, pdbs, warns = [], [], [], []
    for f in range(N_FOLDS):
        pf = pred_folds[f]
        if pf is None:
            warns.append(f"fold {f}: predictions missing")
            continue
        tf = truth_folds[f]
        for key, tv in tf.items():
            if key in pf:
                trues.append(tv)
                preds.append(pf[key])
                pdbs.append(key[0])
        matched = sum(1 for key in tf if key in pf)
        if matched != len(tf):
            warns.append(f"fold {f}: {matched}/{len(tf)} truth rows matched")
    return (np.array(trues), np.array(preds), np.array(pdbs, dtype=object), warns)


def load_single_plm_folds(plm: str, family: str):
    """List (len N_FOLDS) of pred dicts for one PLM arm (None where missing)."""
    tmpl = _plm_path(plm, family)
    return [load_pred_file(Path(str(tmpl).format(f=f))) for f in range(N_FOLDS)]


def build_consensus_folds(family: str, plm_folds_cache):
    """Per-fold consensus pred dicts: mean over exactly the 6 PLMs, all-6-required per row.

    A (pdb, mutation) row is emitted only where all 6 constituent PLM arms have a prediction
    for that fold. Averaging over "whatever is available" is intentionally NOT done.
    """
    out = []
    for f in range(N_FOLDS):
        dicts = [plm_folds_cache[(plm, family)][f] for plm in CONSENSUS_PLMS]
        if any(d is None for d in dicts):
            out.append(None)
            continue
        keys = set(dicts[0])
        for d in dicts[1:]:
            keys &= set(d)
        out.append({k: float(np.mean([d[k] for d in dicts])) for k in keys})
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
METRIC_FIELDS = [
    "arm", "n", "pearson", "spearman", "rmse", "mae", "rmse_corr", "mae_corr",
    "ps_pearson_T10", "ps_spearman_T10", "ps_ncomplex_T10",
    "ps_pearson_T5", "ps_spearman_T5", "ps_ncomplex_T5",
    "auroc_destab", "auroc_strong", "prec_at50", "recall_at50",
]


def score_arm(arm: str, true, pred, pdb):
    row = {"arm": arm, "n": int(true.size)}
    if true.size == 0:
        for k in METRIC_FIELDS:
            row.setdefault(k, float("nan"))
        return row, {"n_strong": 0, "ps_degen_T10": 0, "ps_degen_T5": 0}
    row["pearson"] = pearson(pred, true)
    row["spearman"] = spearman(pred, true)
    row["rmse"] = rmse(pred, true)
    row["mae"] = mae(pred, true)
    row["rmse_corr"] = rmse_corr(pred, true)
    row["mae_corr"] = mae_corr(pred, true)
    degen = {}
    for T in PS_THRESHOLDS:
        mp, ms, nq, ndegen = per_structure(pred, true, pdb, T)
        row[f"ps_pearson_T{T}"] = mp
        row[f"ps_spearman_T{T}"] = ms
        row[f"ps_ncomplex_T{T}"] = nq
        degen[f"ps_degen_T{T}"] = ndegen
    row["auroc_destab"] = auroc(true > 0, pred)
    row["auroc_strong"] = auroc(true >= STRONG, pred)
    prec, rec, n_strong = precision_recall_at_k(pred, true, TOPK, STRONG)
    row["prec_at50"] = prec
    row["recall_at50"] = rec
    aux = {"n_strong": n_strong, **degen}
    return row, aux


def fmt(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Selftest — cross-validate hand-rolled metrics against scipy/sklearn (venv only).
# ---------------------------------------------------------------------------
def selftest():
    from scipy import stats
    from sklearn import metrics as skm
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 200))
        a = rng.normal(size=n).round(2)   # induce ties
        b = rng.normal(size=n).round(2)
        assert abs(pearson(a, b) - stats.pearsonr(a, b)[0]) < 1e-9
        assert abs(spearman(a, b) - stats.spearmanr(a, b)[0]) < 1e-9
        y = (rng.normal(size=n) > 0).astype(int)
        if 0 < y.sum() < n:
            assert abs(auroc(y.astype(bool), b) - skm.roc_auc_score(y, b)) < 1e-9
    print("selftest: pearson/spearman/auroc match scipy+sklearn to <1e-9  ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="cross-validate hand-rolled metrics against scipy/sklearn, then exit")
    ap.add_argument("--tier", default=None,
                    help="evaluation tier to score against (see --list-tiers). This script's "
                         "consensus-arm machinery is built for the S1102 10-fold set, so it "
                         "normally wants --tier legacy_cv10 -- which must be said out loud.")
    ap.add_argument("--list-tiers", action="store_true", help="list tiers and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    from mulan.metrics import describe, list_tiers as _lt
    if args.list_tiers:
        for name in _lt(include_legacy=True):
            print(describe(name))
        return

    if args.tier is None:
        ap.error(
            "no --tier given. This used to fall back to the leaky 10-fold splits silently. "
            f"Pass one of: {', '.join(_lt(include_legacy=True))}. "
            "For the upstream-paper reproduction, that is --tier legacy_cv10."
        )
    tier = use_tier(args.tier)
    print(f"[tier] {describe(tier.name)}", file=sys.stderr)

    log = []

    def logline(msg):
        log.append(msg)
        print(msg, file=sys.stderr)

    # ---- truth + sanity (SPEC §6) ----
    truth_folds = load_truth_folds()
    pooled = {}
    for d in truth_folds:
        pooled.update(d)
    n_total = sum(len(d) for d in truth_folds)
    complexes = {}
    for d in truth_folds:
        for (pdb, _mut) in d:
            complexes[pdb] = complexes.get(pdb, 0) + 1
    top3 = sorted(complexes.values(), reverse=True)[:3]
    logline(f"[truth] pooled rows={n_total} over {len(complexes)} complexes; top3 sizes={top3}")
    assert n_total == 1100, f"expected 1100 pooled truth rows, got {n_total}"
    assert len(complexes) == 110, f"expected 110 complexes, got {len(complexes)}"
    assert top3 == [177, 177, 176], f"expected top3 [177,177,176], got {top3}"

    arm_templates = build_arm_templates()

    # Load every single-PLM arm's per-fold pred dicts once (reused for consensus).
    plm_folds_cache = {}
    for plm in CONSENSUS_PLMS:
        for family in ("base", "aug", "foldxmlp"):
            plm_folds_cache[(plm, family)] = load_single_plm_folds(plm, family)

    # ---- assemble every arm's pooled (true, pred, pdb) ----
    arm_data = {}
    for arm, tmpl in arm_templates.items():
        if tmpl is None:
            continue   # consensus assembled below
        pred_folds = [load_pred_file(Path(str(tmpl).format(f=f))) for f in range(N_FOLDS)]
        if all(pf is None for pf in pred_folds):
            logline(f"[skip] {arm}: no prediction files found ({tmpl})")
            continue
        arm_data[arm] = assemble_arm(pred_folds, truth_folds)
    for family in ("base", "aug", "foldxmlp"):
        cf = build_consensus_folds(family, plm_folds_cache)
        if all(c is None for c in cf):
            logline(f"[skip] consensus_{family}: constituent PLM arms unavailable")
            continue
        arm_data[f"consensus_{family}"] = assemble_arm(cf, truth_folds)

    # ---- score ----
    rows = []
    aux_by_arm = {}
    for arm, (true, pred, pdb, warns) in arm_data.items():
        for w in warns:
            logline(f"[warn] {arm}: {w}")
        if true.size != 1100:
            logline(f"[warn] {arm}: n={true.size} != 1100 (coverage gap)")
        row, aux = score_arm(arm, true, pred, pdb)
        rows.append(row)
        aux_by_arm[arm] = aux
        # SPEC §3: log how many qualifying complexes were skipped as degenerate (zero variance).
        for T in PS_THRESHOLDS:
            nd = aux.get(f"ps_degen_T{T}", 0)
            if nd:
                logline(f"[degen] {arm}: {nd} complex(es) skipped at T={T} "
                        "(zero variance in true or pred)")
        # per-structure < overall spearman is a flag, not a gate
        if not np.isnan(row["spearman"]) and not np.isnan(row["ps_spearman_T10"]):
            if row["ps_spearman_T10"] >= row["spearman"]:
                logline(f"[flag] {arm}: ps_spearman_T10 ({row['ps_spearman_T10']:.4f}) "
                        f">= overall spearman ({row['spearman']:.4f}) — unexpected")

    # ---- cross-checks (SPEC §6) ----
    def _overall_pcc(arm):
        for r in rows:
            if r["arm"] == arm:
                return r["pearson"]
        return None
    for arm, expect in (("esmc6b_base", 0.886), ("consensus_base", 0.870)):
        got = _overall_pcc(arm)
        if got is None:
            logline(f"[xcheck] {arm}: absent — cannot verify (expected ~{expect})")
        elif abs(got - expect) > 0.01:
            logline(f"[xcheck][WARN] {arm} pearson={got:.4f}, expected ~{expect} "
                    "(join may be wrong)")
        else:
            logline(f"[xcheck] {arm} pearson={got:.4f} ~= {expect}  ✓")

    # ---- write results.csv ----
    rows.sort(key=lambda r: (-9 if np.isnan(r["ps_spearman_T10"]) else r["ps_spearman_T10"]),
              reverse=True)
    csv_path = OUT / "results.csv"
    with open(csv_path, "w") as fh:
        fh.write(",".join(METRIC_FIELDS) + "\n")
        for r in rows:
            fh.write(",".join(fmt(r[k]) for k in METRIC_FIELDS) + "\n")
    logline(f"[out] wrote {csv_path} ({len(rows)} arms)")

    write_summary(rows, aux_by_arm, complexes, log)
    logline(f"[out] wrote {OUT / 'SUMMARY.md'}")


def write_summary(rows, aux_by_arm, complexes, log):
    # All full-coverage arms share the same truth pool, so n_strong is identical; max is robust
    # to any degenerate/empty arm that would report 0.
    n_strong_pool = max((a["n_strong"] for a in aux_by_arm.values()), default=0)
    lines = []
    lines.append("# Per-structure + AUROC re-score of balanced-splitter OOF predictions")
    lines.append("")
    lines.append("_Generated by `rescore.py`. Sorted by per-structure Spearman (T=10) desc._")
    lines.append("")
    lines.append("> **⚠ Scope caveat.** This changes the **metric, not the split.** Predictions are from "
                 "the **leaky per-mutation 10-fold CV** (same complex in train and test), so the "
                 "per-structure numbers are an **upper bound**, not a generalization figure. The leakage-controlled "
                 "by-complex (RDE 3-fold) and CATH-clustered numbers still require retraining "
                 "(separate task). Read every number below through that caveat.")
    lines.append("")
    lines.append(f"Pooled OOF: **{sum(complexes.values())}** rows over **{len(complexes)}** complexes; "
                 f"**{n_strong_pool}** rows are strong destabilizers (true ≥ {STRONG:.1f}).")
    lines.append("")
    lines.append("_Metric math validated against the **RDE-PPI reference** "
                 "(`../reference_ddg_splits/RDE-PPI/rde/utils/skempi.py`): `crosscheck_reference.py` "
                 "reproduces its `per_complex_corr`/`overall_correlations`/`overall_auroc`/"
                 "`overall_rmse_mae` to <1e-9 across all arms. `rmse_corr`/`mae_corr` are the reference's "
                 "OLS-rescaled convention; `rmse`/`mae` are raw error._")
    lines.append("")

    # table
    cols = ["arm", "n", "pearson", "spearman",
            "ps_pearson_T10", "ps_spearman_T10", "ps_ncomplex_T10",
            "ps_spearman_T5", "auroc_destab", "auroc_strong", "prec_at50", "recall_at50"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines.append("")

    # written read
    lines.append("## Read")
    lines.append("")
    ranked = [r for r in rows if not np.isnan(r["ps_spearman_T10"])]
    top = ranked[0]["arm"] if ranked else "—"
    lines.append(f"- **Top arm by per-structure Spearman (T=10):** `{top}`.")
    lines.append("- **Overall → per-structure shift** (leakage/leverage proxy): the 3 mega-scanned "
                 "complexes (177/177/176 muts, ~48% of rows) carry most of the overall correlation; the "
                 "`spearman` → `ps_spearman_T10` drop per arm is how much of each arm's score is that "
                 "leverage rather than genuine within-complex ranking.")
    lines.append("- **Arm-ranking vs `S1102_CROSSFOLD_MISPREDICT.md`** (pooled-PCC ranking: FoldX > aug; "
                 "MINT-complex bulk lift; 6B on top): compare the `pearson` column ordering here against "
                 "that write-up; note any re-ordering once per-structure ranking is applied.")
    lines.append("- **Design use-case (flag strong destabilizers):** `auroc_strong` and `prec_at50` are "
                 "robust to magnitude shrinkage — an arm can lose on RMSE yet still rank strong "
                 "destabilizers to the top. Read these independently of the correlation columns.")
    lines.append("- **Expected direction:** per-structure Spearman < overall Spearman for essentially "
                 "every arm (removing the mega-complex leverage). Any violation is logged as a `[flag]`.")
    lines.append("")
    lines.append("## Run log")
    lines.append("")
    lines.append("```")
    lines.extend(log)
    lines.append("```")

    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
