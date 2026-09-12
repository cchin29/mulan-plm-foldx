#!/usr/bin/env python3
"""Per-structure re-score of the full-SKEMPI MULTI-POINT runs (clustered + by-complex).

THIN wrapper over the frontier-metric core in `experiments/rescore_perstructure/rescore.py`
(imported, not re-implemented) — the same pattern as `experiments/retrain_split/evaluate.py`,
re-pointed at the multi-point pred/truth layout:

  * 3 folds,
  * predictions at
    scratch/results/full_skempi_mp_<split>/<tag>_<arm>/fold_{0,1,2}/training_run/test_predictions.tsv
    arms {base, foldx (12-term MLP), foldx_scalar}; tags = whatever has run,
  * truth (per fold) preferring the 16-col foldxdec split, falling back to the 4-col base split:
    scratch/foldx_skempi_full/splits_mp_<split>_mp_foldxdec/fold_{f}/skempi_mp_test.tsv
    scratch/splits_skempi_full_mp_<split>.../fold_{f}/skempi_mp_test.tsv

The per-structure grouping key is `_pdb_of(chain1_id) = chain1_id.split("_")[0]`, which for the
multi-point labels (e.g. `1A22.A.B_A`) yields the dot-joined `#Pdb` grouping `1A22.A.B` — exactly
the complex a fold is held out by. So no metric/loader change is needed; only the paths + N_FOLDS.

Reuses unchanged from rescore.py: metrics, load_pred_file / load_truth_folds / assemble_arm /
score_arm, METRIC_FIELDS, fmt. Reuses evaluate.py's cluster-bootstrap (bootstrap_report).
Partial folds are fine — arms with predictions are scored, missing ones skipped (re-run as folds land).

Outputs (experiments/full_skempi_seqonly/, results_matrix.py --ps consumable):
  results_mp_<split>.csv    — one row per arm, rescore results.csv schema
  bootstrap_mp_<split>.txt  — per-arm CI + paired foldx-base contrasts
  SUMMARY_mp_<split>.md     — per-structure Spearman table + FoldX Δ-lift

Run:  ./.venv/bin/python experiments/full_skempi_seqonly/score_multipoint.py            # both splits
      ./.venv/bin/python experiments/full_skempi_seqonly/score_multipoint.py --split clustered
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent               # experiments/full_skempi_seqonly
ROOT = HERE.parents[1]
RESCORE_DIR = ROOT / "experiments" / "rescore_perstructure"
RETRAIN_DIR = ROOT / "experiments" / "retrain_split"
for _p in (RESCORE_DIR, RETRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rescore  # noqa: E402  (metric core)
from rescore import (  # noqa: E402
    METRIC_FIELDS, assemble_arm, fmt, load_pred_file, load_truth_folds, score_arm,
)
from evaluate import bootstrap_report  # noqa: E402  (cluster-bootstrap + paired contrasts)

N_FOLDS = 3
TAGS = ("ankh", "saprot", "prostt5", "esmc6b", "esmc600m", "esm2",
        "ankh3_large", "ankh3_xl", "aido")  # score whatever has run (GPU-box ankh3/aido land via snapshot)
ARMS = ("base", "foldx", "foldx_scalar")

SPLITS = {
    "clustered": {
        "pred": ROOT / "scratch/results/full_skempi_mp_clustered/{tag}_{arm}/fold_{f}/training_run/test_predictions.tsv",
        "truth_dec": ROOT / "scratch/foldx_skempi_full/splits_mp_clustered_mp_foldxdec/fold_{f}/skempi_mp_test.tsv",
        "truth_base": ROOT / "scratch/splits_skempi_full_mp_clustered_id60_kfold/fold_{f}/skempi_mp_test.tsv",
        "label": "multi-point homology-clustered (mmseqs <=60% id, 3-fold)",
    },
    "bycomplex": {
        "pred": ROOT / "scratch/results/full_skempi_mp_bycomplex/{tag}_{arm}/fold_{f}/training_run/test_predictions.tsv",
        "truth_dec": ROOT / "scratch/foldx_skempi_full/splits_mp_bycomplex_mp_foldxdec/fold_{f}/skempi_mp_test.tsv",
        "truth_base": ROOT / "scratch/splits_skempi_full_mp_bycomplex_seed42/fold_{f}/skempi_mp_test.tsv",
        "label": "multi-point by-complex (whole #Pdb grouping held out, 3-fold)",
    },
}


def choose_truth(cfg, log):
    for key, name in (("truth_dec", "foldxdec 16-col"), ("truth_base", "base 4-col")):
        tmpl = cfg[key]
        if all(Path(str(tmpl).format(f=f)).exists() for f in range(N_FOLDS)):
            log(f"[truth] using {name}: {tmpl}")
            return tmpl
    return None


def score_split(split, log):
    cfg = SPLITS[split]
    log(f"\n===== scoring MP {split} — {cfg['label']} =====")
    truth_tmpl = choose_truth(cfg, log)
    if truth_tmpl is None:
        log(f"[skip] {split}: no complete {N_FOLDS}-fold truth split found.")
        return
    rescore.N_FOLDS = N_FOLDS
    rescore.TRUTH_TMPL = truth_tmpl
    truth_folds = load_truth_folds()
    complexes = {}
    for d in truth_folds:
        for (pdb, _mut) in d:
            complexes[pdb] = complexes.get(pdb, 0) + 1
    log(f"[truth] pooled {sum(len(d) for d in truth_folds)} rows over {len(complexes)} complexes")

    arm_data, rows = {}, []
    for tag in TAGS:
        for arm in ARMS:
            arm_id = f"{tag}_{arm}"
            pred_folds = [load_pred_file(Path(str(cfg["pred"]).format(tag=tag, arm=arm, f=f)))
                          for f in range(N_FOLDS)]
            if all(pf is None for pf in pred_folds):
                continue
            n_present = sum(pf is not None for pf in pred_folds)
            if n_present < N_FOLDS:
                log(f"[partial] {arm_id}: {n_present}/{N_FOLDS} folds (still training?)")
            arm_data[arm_id] = assemble_arm(pred_folds, truth_folds)

    if not arm_data:
        log(f"[skip] {split}: no predictions yet.")
        return

    for arm_id, (true, pred, pdb, warns) in arm_data.items():
        for w in warns:
            log(f"[warn] {arm_id}: {w}")
        log(f"[score] {arm_id}: n={true.size} rows, {len(np.unique(pdb))} complexes")
        row, _aux = score_arm(arm_id, true, pred, pdb)
        rows.append(row)

    boot_lines = bootstrap_report(arm_data, log)
    _write(split, cfg, rows, complexes, boot_lines, log)


def _write(split, cfg, rows, complexes, boot_lines, log):
    rows.sort(key=lambda r: (r.get("ps_spearman_T10") if not np.isnan(r.get("ps_spearman_T10", float("nan"))) else -9),
              reverse=True)
    csv_path = HERE / f"results_mp_{split}.csv"
    with open(csv_path, "w") as fh:
        fh.write(",".join(METRIC_FIELDS) + "\n")
        for r in rows:
            fh.write(",".join(fmt(r.get(k, float("nan"))) for k in METRIC_FIELDS) + "\n")
    (HERE / f"bootstrap_mp_{split}.txt").write_text("\n".join(boot_lines) + "\n" if boot_lines else "")

    by = {r["arm"]: r for r in rows}

    def ps(arm):
        return by.get(arm, {}).get("ps_spearman_T10", float("nan"))

    L = [f"# {cfg['label']} — per-structure re-score", "",
         "_Generated by `experiments/full_skempi_seqonly/score_multipoint.py` (reuses the "
         "`rescore_perstructure` metric core). Sorted by per-structure Spearman (T=10) desc._", "",
         f"Pooled test rows: **{sum(complexes.values())}** over **{len(complexes)}** complexes.", "",
         "## FoldX per-structure Spearman lift over base (T>=10)", "",
         "| PLM | base | fx_scalar | fx_mlp | Δscalar | Δmlp |", "|---|---|---|---|---|---|"]
    for tag in TAGS:
        b, s, m = ps(f"{tag}_base"), ps(f"{tag}_foldx_scalar"), ps(f"{tag}_foldx")
        if np.isnan(b) and np.isnan(s) and np.isnan(m):
            continue
        ds = s - b if not (np.isnan(s) or np.isnan(b)) else float("nan")
        dm = m - b if not (np.isnan(m) or np.isnan(b)) else float("nan")
        L.append("| " + " | ".join(fmt(x) for x in (tag, b, s, m, ds, dm)) + " |")
    L += ["", "## Full metrics", "",
          "| " + " | ".join(METRIC_FIELDS) + " |", "|" + "|".join(["---"] * len(METRIC_FIELDS)) + "|"]
    for r in rows:
        L.append("| " + " | ".join(fmt(r.get(k, float("nan"))) for k in METRIC_FIELDS) + " |")
    L += ["", "## Cluster-bootstrap CIs + paired contrasts", "", "```"] + \
         (boot_lines if boot_lines else ["(bootstrap skipped)"]) + ["```", ""]
    (HERE / f"SUMMARY_mp_{split}.md").write_text("\n".join(L) + "\n")
    log(f"[out] wrote {csv_path.name}, bootstrap_mp_{split}.txt, SUMMARY_mp_{split}.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLITS) + ["both"], default="both")
    a = ap.parse_args()

    def log(msg):
        print(msg, file=sys.stderr)

    targets = list(SPLITS) if a.split == "both" else [a.split]
    for sp in targets:
        try:
            score_split(sp, log)
        except Exception as e:  # keep going so one bad/partial split can't block the other
            log(f"[error] {sp}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
