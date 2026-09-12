#!/usr/bin/env python3
"""Cross-check rescore.py's hand-rolled metrics against the RDE-PPI reference implementation.

Reference: ../reference_ddg_splits/RDE-PPI/rde/utils/skempi.py (luost26/RDE-PPI @ 58887deb) — the
canonical SKEMPI-v2 ddG benchmark whose split+metrics are reused by Prompt-DDG/DiffAffinity/GearBind/
PPIformer/etc. See ../reference_ddg_splits/README.md ("Two uses" #1: metric parity).

What this does: for every arm rescore.py scores, build the reference's dataframe
(columns complex/ddG/ddG_pred) on the SAME pooled OOF rows and complex key, run the REFERENCE's own
metric functions (extracted verbatim from their source — not retyped), and assert they agree with our
hand-rolled numbers. This isolates metric-math parity from the split/grouping question.

Two known-by-design differences it reports rather than asserts:
  1. overall_rmse_mae: the reference reports RMSE/MAE of a LinearRegression(true~pred)-CORRECTED
     prediction (scale/offset removed), NOT raw error. We print both so the gap is explicit.
  2. complex key: the reference uses the full #Pdb (e.g. 1A22_A_B); we use the 4-letter code
     (1A22). For math parity we feed the reference OUR key so grouping is identical on both sides;
     the tighter-key regrouping is a split-parity question (future work), not a metric-math one.

Run:  experiments/rescore_perstructure/.venv/bin/python experiments/rescore_perstructure/crosscheck_reference.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

import rescore

# This crosscheck is against the S1102 10-fold reference, so it must name that tier —
# it previously inherited it as the module default.
rescore.use_tier("legacy_cv10")  # same directory

REF = (Path(__file__).resolve().parents[2].parent
       / "reference_ddg_splits/RDE-PPI/rde/utils/skempi.py")
REF_FUNCS = ("per_complex_corr", "overall_correlations", "overall_auroc", "overall_rmse_mae")
TOL = 1e-9


def load_reference_functions():
    """Exec the reference metric functions verbatim (their module top-imports torch/rde which we
    don't have, so we AST-extract only the 4 pure pandas/numpy/sklearn functions)."""
    src = REF.read_text()
    tree = ast.parse(src)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in REF_FUNCS:
            segments.append(ast.get_source_segment(src, node))
    ns = {"pd": pd, "np": np, "roc_auc_score": roc_auc_score,
          "LinearRegression": LinearRegression}
    exec("\n\n".join(segments), ns)
    missing = [f for f in REF_FUNCS if f not in ns]
    if missing:
        raise RuntimeError(f"failed to extract reference functions: {missing}")
    return {f: ns[f] for f in REF_FUNCS}


def build_arm_data():
    """Reuse rescore.py's loaders to assemble every arm's pooled (true, pred, pdb)."""
    truth_folds = rescore.load_truth_folds()
    arm_templates = rescore.build_arm_templates()
    plm_cache = {}
    for plm in rescore.CONSENSUS_PLMS:
        for fam in ("base", "aug", "foldxmlp"):
            plm_cache[(plm, fam)] = rescore.load_single_plm_folds(plm, fam)

    data = {}
    for arm, tmpl in arm_templates.items():
        if tmpl is None:
            continue
        pred_folds = [rescore.load_pred_file(Path(str(tmpl).format(f=f)))
                      for f in range(rescore.N_FOLDS)]
        if all(pf is None for pf in pred_folds):
            continue
        data[arm] = rescore.assemble_arm(pred_folds, truth_folds)
    for fam in ("base", "aug", "foldxmlp"):
        cf = rescore.build_consensus_folds(fam, plm_cache)
        if not all(c is None for c in cf):
            data[f"consensus_{fam}"] = rescore.assemble_arm(cf, truth_folds)
    return data


def approx(a, b, tol=TOL):
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= tol


def main():
    ref = load_reference_functions()
    print(f"[ref] extracted {', '.join(REF_FUNCS)} from {REF}")
    arm_data = build_arm_data()
    print(f"[data] {len(arm_data)} arms assembled\n")

    n_checks = 0
    n_fail = 0
    rmse_rows = []

    header = (f"{'arm':<20} {'overall_pcc':>12} {'overall_spr':>12} "
              f"{'auroc':>8} {'ps_pcc_T10':>11} {'ps_spr_T10':>11}")
    print(header)
    print("-" * len(header))

    for arm, (true, pred, pdb, _warns) in sorted(arm_data.items()):
        df = pd.DataFrame({"complex": [str(x) for x in pdb],
                           "ddG": true.astype(float),
                           "ddG_pred": pred.astype(float)})

        # --- reference numbers ---
        ref_overall = ref["overall_correlations"](df)
        ref_auroc = ref["overall_auroc"](df)["auroc"]
        ref_ps_pcc, ref_ps_spr = ref["per_complex_corr"](df, limit=10)
        ref_rmse_mae = ref["overall_rmse_mae"](df)   # regression-CORRECTED

        # --- our hand-rolled numbers ---
        my_pcc = rescore.pearson(pred, true)
        my_spr = rescore.spearman(pred, true)
        my_auroc = rescore.auroc(true > 0, pred)
        my_ps_pcc, my_ps_spr, _, _ = rescore.per_structure(pred, true, pdb, 10)
        my_rmse = rescore.rmse(pred, true)                 # RAW (no reference counterpart)
        my_mae = rescore.mae(pred, true)
        my_rmse_corr = rescore.rmse_corr(pred, true)       # OLS-corrected == reference definition
        my_mae_corr = rescore.mae_corr(pred, true)

        checks = [
            ("overall_pearson", my_pcc, ref_overall["overall_pearson"]),
            ("overall_spearman", my_spr, ref_overall["overall_spearman"]),
            ("auroc_destab", my_auroc, ref_auroc),
            ("ps_pearson_T10", my_ps_pcc, ref_ps_pcc),
            ("ps_spearman_T10", my_ps_spr, ref_ps_spr),
            ("rmse_corr", my_rmse_corr, ref_rmse_mae["rmse"]),
            ("mae_corr", my_mae_corr, ref_rmse_mae["mae"]),
        ]
        arm_ok = True
        for name, mine, theirs in checks:
            n_checks += 1
            if not approx(float(mine), float(theirs)):
                n_fail += 1
                arm_ok = False
                print(f"  [FAIL] {arm} {name}: ours={mine:.10f} ref={theirs:.10f} "
                      f"(|Δ|={abs(mine-theirs):.2e})")
        flag = " " if arm_ok else "✗"
        print(f"{flag}{arm:<19} {my_pcc:>12.6f} {my_spr:>12.6f} {my_auroc:>8.4f} "
              f"{my_ps_pcc:>11.6f} {my_ps_spr:>11.6f}")

        rmse_rows.append((arm, my_rmse, my_mae,
                          float(ref_rmse_mae["rmse"]), float(ref_rmse_mae["mae"])))

    print()
    print("=" * 72)
    print("RMSE / MAE — raw vs OLS-corrected (rmse_corr/mae_corr are the RDE/leaderboard convention,")
    print("asserted equal to the reference above; raw_* is our extra honest-error column, no ref):")
    print(f"{'arm':<20} {'raw_rmse':>10} {'corr_rmse':>10} {'raw_mae':>10} {'corr_mae':>10}")
    print("-" * 64)
    for arm, r_rmse, r_mae, ref_rmse, ref_mae in sorted(rmse_rows):
        print(f"{arm:<20} {r_rmse:>10.4f} {ref_rmse:>10.4f} {r_mae:>10.4f} {ref_mae:>10.4f}")

    print()
    print("=" * 72)
    print(f"PARITY: {n_checks - n_fail}/{n_checks} metric checks agree to <{TOL:g} "
          f"across {len(arm_data)} arms "
          f"(pearson/spearman/auroc/per-structure/rmse_corr/mae_corr).")
    if n_fail:
        print(f"*** {n_fail} MISMATCH(ES) — hand-rolled metrics diverge from RDE reference. ***")
        sys.exit(1)
    print("All hand-rolled correlation/AUROC/per-structure metrics reproduce the RDE-PPI reference. ✓")


if __name__ == "__main__":
    main()
