#!/usr/bin/env python3
"""Per-structure re-score of the CATH-superfamily hold-out retrain, broken out single/multiple/all
to match USP-ddG Table 1's mutation-category rows.

THIN wrapper over the `experiments/rescore_perstructure/rescore.py` metric core (same pattern as
score_multipoint.py), re-pointed at the single-fold CATH layout:
  * 1 fold (num_folds=1; CATH is a fixed train/test partition, not CV),
  * predictions:
    scratch/results/full_skempi_cath/<tag>_<arm>/fold_0/training_run/test_predictions.tsv
    arms {base, foldx (12-term MLP), foldx_scalar}; tags = whatever has run,
  * truth = the CATH test split; BREAKOUT by swapping the truth file (rescore matches preds->truth
    by (pdb,mut), so a single-only / multiple-only truth yields that subset's metrics for free):
        all      -> splits_skempi_full_cath_kfold/fold_0/skempi_all_test.tsv (or the 16-col foldxdec)
        single   -> splits_skempi_full_cath_kfold/test_single.tsv
        multiple -> splits_skempi_full_cath_kfold/test_multiple.tsv

Outputs (experiments/full_skempi_seqonly/):
  results_cath.csv / results_cath_single.csv / results_cath_multiple.csv   (rescore results schema)
  SUMMARY_cath.md    — per-arm single/multiple/all per-structure Spearman + overall + AUROC

Then drop into the frontier leaderboard:
  .venv/bin/python experiments/retrain_split/format_usp_row.py results_cath.csv --all --split cath --mut all

Run:  .venv/bin/python experiments/full_skempi_seqonly/score_cath.py
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "experiments/rescore_perstructure", ROOT / "experiments/retrain_split"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rescore  # noqa: E402
from rescore import METRIC_FIELDS, assemble_arm, fmt, load_pred_file, load_truth_folds, score_arm  # noqa: E402

TAGS = ("ankh", "esm2", "saprot", "prostt5", "esmc6b", "esmc600m", "ankh3_large", "ankh3_xl", "aido")
ARMS = ("base", "foldx", "foldx_scalar")
PRED = ROOT / "scratch/results/full_skempi_cath/{tag}_{arm}/fold_0/training_run/test_predictions.tsv"
SPLIT = ROOT / "scratch/splits_skempi_full_cath_kfold"
# truth templates (no {f} needed — single fold; .format(f=0) leaves them unchanged)
TRUTH = {
    "all": SPLIT / "fold_0/skempi_all_test.tsv",
    "single": SPLIT / "test_single.tsv",
    "multiple": SPLIT / "test_multiple.tsv",
}
# Tier -> (results dir under scratch/results, output CSV stem). `--tier cath_aug` scores the
# Tier-1-augmented FoldX arms against the SAME TRUTH as the plain tier: augmentation adds
# reverse-mutation and identity rows to *train* only, and the augmented split's test and val TSVs
# are byte-identical to the unaugmented ones. Scoring both against one truth file is what makes the
# aug-vs-plain delta a paired comparison rather than two numbers that merely ought to be comparable.
# The aug tier has no `base` arm, so that row is simply absent from its CSV.
TIERS = {
    "cath": ("full_skempi_cath", "results_cath"),
    "cath_aug": ("full_skempi_cath_aug", "results_cath_aug"),
}


def score_subset(subset, log):
    truth_path = TRUTH[subset]
    if not truth_path.exists():
        log(f"[skip] {subset}: truth {truth_path} missing")
        return []
    rescore.N_FOLDS = 1
    rescore.TRUTH_TMPL = str(truth_path)
    truth_folds = load_truth_folds()
    n_truth = sum(len(d) for d in truth_folds)
    complexes = {pdb for d in truth_folds for (pdb, _m) in d}
    log(f"[{subset}] truth {n_truth} rows / {len(complexes)} complexes")

    rows = []
    for tag in TAGS:
        for arm in ARMS:
            arm_id = f"{tag}_{arm}"
            pred_folds = [load_pred_file(Path(str(PRED).format(tag=tag, arm=arm, f=0)))]
            if pred_folds[0] is None:
                continue
            true, pred, pdb, warns = assemble_arm(pred_folds, truth_folds)
            for w in warns:
                log(f"[warn] {arm_id}/{subset}: {w}")
            row, _ = score_arm(arm_id, true, pred, pdb)
            rows.append(row)
    rows.sort(key=lambda r: (r.get("ps_spearman_T10") if not np.isnan(r.get("ps_spearman_T10", float("nan"))) else -9),
              reverse=True)
    return rows


def write_csv(path, rows):
    with open(path, "w") as fh:
        fh.write(",".join(METRIC_FIELDS) + "\n")
        for r in rows:
            fh.write(",".join(fmt(r.get(k, float("nan"))) for k in METRIC_FIELDS) + "\n")


def main():
    global PRED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIERS), default="cath",
                    help="cath (default) or cath_aug — the Tier-1-augmented FoldX arms, scored "
                         "against the same truth (see TIERS)")
    args = ap.parse_args()
    resdir, stem = TIERS[args.tier]
    PRED = ROOT / f"scratch/results/{resdir}/{{tag}}_{{arm}}/fold_0/training_run/test_predictions.tsv"

    def log(m):
        print(m, file=sys.stderr)

    log(f"===== scoring CATH-superfamily hold-out [{args.tier}] (single / multiple / all) =====")
    by_subset = {s: score_subset(s, log) for s in ("all", "single", "multiple")}
    if not any(by_subset.values()):
        log(f"[skip] no {args.tier} predictions under scratch/results/{resdir} — run the retrain "
            f"first (see docs/history/runbooks/RUN_CATH_LINUX.md).")
        return

    write_csv(HERE / f"{stem}.csv", by_subset["all"])
    write_csv(HERE / f"{stem}_single.csv", by_subset["single"])
    write_csv(HERE / f"{stem}_multiple.csv", by_subset["multiple"])

    # SUMMARY: per-arm single/multiple/all, USP-ddG Table-1 style
    idx = {s: {r["arm"]: r for r in by_subset[s]} for s in by_subset}
    arms = sorted({r["arm"] for r in by_subset["all"]},
                  key=lambda a: -(idx["all"].get(a, {}).get("ps_spearman_T10", -9)))
    L = ["# CATH-superfamily hold-out (USP-ddG's exact 813-mut split) — MuLAN retrain", "",
         "_Generated by `score_cath.py`. Literal frontier test set (shipped `cath_fold` labels, "
         "per-complex join). Compare to USP-ddG Table 1 / CATH-ddG directly._", "",
         "## Per-structure Spearman (T≥10) by mutation category", "",
         "| arm | all | single | multiple | overall-P (all) | AUROC (all) |",
         "|---|---|---|---|---|---|"]

    def g(s, a, k):
        return idx[s].get(a, {}).get(k, float("nan"))

    for a in arms:
        L.append("| " + " | ".join([a,
                  fmt(g("all", a, "ps_spearman_T10")), fmt(g("single", a, "ps_spearman_T10")),
                  fmt(g("multiple", a, "ps_spearman_T10")), fmt(g("all", a, "pearson")),
                  fmt(g("all", a, "auroc_destab"))]) + " |")
    L += ["", f"_Rows: {stem}{{,_single,_multiple}}.csv. Frontier row: "
          f"`format_usp_row.py {stem}.csv --all --split cath --mut all`._", ""]
    summary = HERE / (f"SUMMARY_{args.tier}.md" if args.tier != "cath" else "SUMMARY_cath.md")
    summary.write_text("\n".join(L) + "\n")
    log(f"[out] wrote {stem}{{,_single,_multiple}}.csv + {summary.name}")


if __name__ == "__main__":
    main()
