#!/usr/bin/env python3
"""Evaluate the full-SKEMPI single-point, sequence-only base runs (docs/history/PLAN_FULL_SKEMPI.md §6).

THIN wrapper over the frontier-metric core in experiments/rescore_perstructure/rescore.py
(imported, not re-implemented), pointed at the homology-clustered 3-fold split:

  predictions : scratch/results/full_skempi/<tag>_base/fold_{0,1,2}/training_run/test_predictions.tsv
  truth       : scratch/splits_skempi_full_clustered_id60_kfold/fold_{f}/skempi_sp_test.tsv (4-col)

Reuses unchanged: _pdb_of / load_truth_folds / load_pred_file / assemble_arm / score_arm /
METRIC_FIELDS / fmt / spearman (rescore globals N_FOLDS + TRUTH_TMPL re-pointed to the 3-fold
full-SKEMPI clustered layout before the loaders run).

Outputs to experiments/full_skempi_seqonly/:
  results_clustered.csv   -- one row per PLM base arm (rescore column set)
  SUMMARY.md              -- honest sequence-only table + per-fold per-structure read, vs the
                             ProtBFF ~0.51 P / 0.48 S honest-split anchor.

Run:  .venv/bin/python experiments/full_skempi_seqonly/evaluate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESCORE_DIR = ROOT / "experiments" / "rescore_perstructure"
if str(RESCORE_DIR) not in sys.path:
    sys.path.insert(0, str(RESCORE_DIR))

import rescore  # noqa: E402
from rescore import (  # noqa: E402
    METRIC_FIELDS, assemble_arm, fmt, load_pred_file, load_truth_folds, score_arm, spearman,
)

N_FOLDS = 3
TAGS = ("ankh", "esm2", "saprot", "prostt5",
        "esmc600m", "ankh3_large", "ankh3_xl", "saprot13b")
PRED_TMPL = ROOT / "scratch/results/full_skempi/{tag}_base/fold_{f}/training_run/test_predictions.tsv"
TRUTH_TMPL = ROOT / "scratch/splits_skempi_full_clustered_id60_kfold/fold_{f}/skempi_sp_test.tsv"
OUT = HERE
# ProtBFF honest family-split anchor (sequence+FoldX ceiling); sequence-only peers ~0.40-0.50 P.
ANCHOR = "ProtBFF honest-split anchor ~0.51 Pearson / 0.48 Spearman (structure+FoldX ceiling)"


def main():
    rescore.N_FOLDS = N_FOLDS
    rescore.TRUTH_TMPL = TRUTH_TMPL
    truth = load_truth_folds()
    n_truth = sum(len(t) for t in truth)
    print(f"[truth] {n_truth} rows over {N_FOLDS} folds "
          f"({[len(t) for t in truth]}); {len({k[0] for t in truth for k in t})} complexes")

    rows, perfold = [], {}
    for tag in TAGS:
        pred_folds = [load_pred_file(Path(str(PRED_TMPL).format(tag=tag, f=f))) for f in range(N_FOLDS)]
        if all(p is None for p in pred_folds):
            print(f"[skip] {tag}_base: no prediction files yet")
            continue
        true, pred, pdb, warns = assemble_arm(pred_folds, truth)
        for w in warns:
            print(f"    [{tag}] {w}")
        row = score_arm(f"{tag}_base", true, pred, pdb)
        rows.append(row)
        # per-fold pooled Spearman (fold_0 test = protease mega-family; report separately)
        pf = []
        for f in range(N_FOLDS):
            if pred_folds[f] is None:
                pf.append(float("nan")); continue
            tf = truth[f]
            keys = [k for k in tf if k in pred_folds[f]]
            t = np.array([tf[k] for k in keys]); p = np.array([pred_folds[f][k] for k in keys])
            pf.append(spearman(p, t) if len(keys) > 2 else float("nan"))
        perfold[tag] = pf
        print(f"[score] {tag}_base: n={len(true)} rows, {len(set(pdb))} complexes")

    if not rows:
        print("[out] nothing to score yet (no predictions).")
        return

    csv_path = OUT / "results_clustered.csv"
    with open(csv_path, "w") as fh:
        fh.write(",".join(METRIC_FIELDS) + "\n")
        for r in rows:
            fh.write(",".join(fmt(r[k]) for k in METRIC_FIELDS) + "\n")

    _write_summary(rows, perfold, n_truth, truth)
    print(f"[out] wrote {csv_path}, {OUT/'SUMMARY.md'}")


def _write_summary(rows, perfold, n_truth, truth):
    rows = sorted(rows, key=lambda r: -(r["ps_spearman_T10"] if r["ps_spearman_T10"] == r["ps_spearman_T10"] else -9))
    L = ["# Full SKEMPI v2 — single-point, sequence-only base (homology-clustered ≤60% id)", ""]
    L += [f"_RDE-curated single-point set ({n_truth} muts, per-fold {[len(t) for t in truth]}); "
          f"MuLAN base head per PLM on the 3-fold mmseqs≤60%-id clustered split. Sorted by "
          f"per-structure Spearman (T≥10)._", ""]
    L += [f"> **Honest generalization number.** Whole sequence families held out. Anchor: {ANCHOR}. "
          "MuLAN base here is sequence-only (its fair peer group is ESM-1v / MSA-T / Tranception / "
          "MINT, not the structure+FoldX SOTA).", ""]
    L += ["## Per-structure + overall metrics", ""]
    hdr = ["arm", "n", "pearson", "spearman", "rmse", "ps_pearson_T10", "ps_spearman_T10",
           "ps_ncomplex_T10", "auroc_destab", "auroc_strong", "prec_at50"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        L.append("| " + " | ".join(fmt(r[k]) if k != "arm" else r[k] for k in hdr) + " |")
    L += ["", "## Per-fold per-structure Spearman (fold_0 test = protease-inhibitor mega-family)", ""]
    L += ["| arm | fold_0 | fold_1 | fold_2 |", "|---|---|---|---|"]
    for tag, pf in perfold.items():
        L.append(f"| {tag}_base | " + " | ".join(f"{x:.3f}" if x == x else "—" for x in pf) + " |")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
