#!/usr/bin/env python3
"""Score EVERY PLM's SINGLE-point full-SKEMPI run against the two leakage-controlled
SP truths: homology-clustered id60 (headline) and by-complex. Generalized sibling of
score_sp_ankh3.py / score_{bycomplex_sp,full_skempi}_esmc6b.py — instead of hardcoding
one PLM it auto-discovers every `{plm}_{arm}` dir present in the SP result trees, so a
newly-landed arm (prostt5 / saprot / esm2 / ankh v1 / ankh3 / esmc6b) is picked up with
no code change. Same rescore.py core (assemble_arm / score_arm / per_structure).

Conclusion metric = per-structure Spearman T>=10 (ppS). 3-fold leakage-free pooled join.

Usage:  ./.venv/bin/python experiments/full_skempi_seqonly/score_sp_all.py
            [clustered|bycomplex|bycomplex_all|clustered_all|clustered_all_aug|both|all]
            [--csv PATH]
        (default: both = the two SP tiers. The combined single+multi tiers and the augmented tier
         are NOT in `both` — name one explicitly, or use `all` for every configured tier.
         CSV -> scratch/sp_perstructure_ppS.csv)

Moved here from scratch/ on 2026-07-25 and put under version control — it had been living in a
gitignored dir, so every edit travelled only by tarball snapshot. Sibling of score_cath.py /
score_multipoint.py, which score the other rungs of the same ladder.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# Portable ROOT: this file lives at <repo>/experiments/full_skempi_seqonly/, so parents[2] is the
# mulan repo root on either box (Mac or Linux) — resolved from __file__, never hardcoded. Same idiom as score_cath.py.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "rescore_perstructure"))
import rescore
from rescore import (assemble_arm, load_pred_file, load_truth_folds, score_arm, spearman,
                     METRIC_FIELDS)

N_FOLDS = 3
rescore.N_FOLDS = N_FOLDS
ARMS = ["base", "foldx", "foldx_scalar"]

# Canonical display order; any PLM discovered but not listed here is appended (sorted).
PLM_ORDER = ["esmc6b", "ankh3_xl", "ankh3_large", "ankh", "prostt5", "saprot", "esm2"]

CFG = {
    "clustered": dict(
        label="CLUSTERED-SP (homology id60, headline)",
        truth=ROOT / "scratch/splits_skempi_full_clustered_id60_kfold/fold_{f}/skempi_sp_test.tsv",
        resdir=ROOT / "scratch/results/full_skempi",
        # build_ps-compatible rung CSV (arm=<tag>_<suffix>) so results_matrix.py --ps picks it up.
        rung=ROOT / "experiments/full_skempi_seqonly/results_sp_clustered.csv",
    ),
    "bycomplex": dict(
        label="BYCOMPLEX-SP",
        truth=ROOT / "scratch/splits_skempi_full_bycomplex_seed42/fold_{f}/skempi_sp_test.tsv",
        resdir=ROOT / "scratch/results/full_skempi_bycomplex",
        rung=ROOT / "experiments/full_skempi_seqonly/results_sp_bycomplex.csv",
    ),
    # COMBINED single+multi by-complex — the frontier-matched protocol (RDE-Network / DiffAffinity /
    # Prompt-DDG / BA-DDG all train ONE model on single+multi with a whole-PDB hold-out). Not an SP
    # tier despite this file's name, but the truth/pred join is identical (rescore matches by
    # (pdb, mut)), so it rides the same CFG machinery. `both` still means the two SP tiers only —
    # invoke this one explicitly. Truth basename is skempi_all_*, cf. config_skempi_full_bycomplex_all.sh.
    "bycomplex_all": dict(
        label="BYCOMPLEX-ALL (combined single+multi, frontier protocol)",
        truth=ROOT / "scratch/splits_skempi_full_bycomplex_all_seed42/fold_{f}/skempi_all_test.tsv",
        resdir=ROOT / "scratch/results/full_skempi_bycomplex_all",
        rung=ROOT / "experiments/full_skempi_seqonly/results_bycomplex_all.csv",
    ),
    # COMBINED single+multi, homology-clustered (mmseqs id60) — the clustered analog of
    # bycomplex_all. Its results tree has been complete for every backbone since 2026-08-04 but no
    # scorer covered it, so the tier had no ppS number at all while its by-complex twin did.
    "clustered_all": dict(
        label="CLUSTERED-ALL (combined single+multi, homology id60)",
        truth=ROOT / "scratch/splits_skempi_full_clustered_all_id60_kfold/fold_{f}/skempi_all_test.tsv",
        resdir=ROOT / "scratch/results/full_skempi_clustered_all",
        rung=ROOT / "experiments/full_skempi_seqonly/results_clustered_all.csv",
    ),
    # Tier-1 augmentation (reverse-mutation + identity rows, FoldX antisymmetric) on top of the
    # FoldX arms. Deliberately the SAME truth as `clustered_all`: augmentation touches train only —
    # the aug split's test and val TSVs are byte-identical to the unaugmented tier's, verified per
    # fold. Sharing the truth path is what makes the aug-vs-plain delta a paired comparison on one
    # test set rather than two runs scored against two files that merely ought to agree.
    #
    # Only `foldx` and `foldx_scalar` resolve here; `base` is reported as skipped until the
    # augmented four-column split exists. That split is not a build — the 4-column base split is
    # the 5-column FoldX split with its last column removed (byte-identical across train/val/test
    # on every fold), so it is `cut -f1-4` plus an ES_SPLIT_DIR line in the aug config.
    "clustered_all_aug": dict(
        label="CLUSTERED-ALL + Tier-1 aug (FoldX arms; aug is train-only)",
        truth=ROOT / "scratch/splits_skempi_full_clustered_all_id60_kfold/fold_{f}/skempi_all_test.tsv",
        resdir=ROOT / "scratch/results/full_skempi_clustered_all_aug",
        rung=ROOT / "experiments/full_skempi_seqonly/results_clustered_all_aug.csv",
    ),
    # Aug on the frontier-matched protocol. Wired ahead of the runs: until
    # scratch/results/full_skempi_bycomplex_all_aug exists this scores as "(none found)" and writes
    # an empty rung, which is the intended behaviour. Truth is bycomplex_all's, for the same
    # train-only reason as the clustered pair above.
    "bycomplex_all_aug": dict(
        label="BYCOMPLEX-ALL + Tier-1 aug (FoldX arms; aug is train-only)",
        truth=ROOT / "scratch/splits_skempi_full_bycomplex_all_seed42/fold_{f}/skempi_all_test.tsv",
        resdir=ROOT / "scratch/results/full_skempi_bycomplex_all_aug",
        rung=ROOT / "experiments/full_skempi_seqonly/results_bycomplex_all_aug.csv",
    ),
}
PRED = "{plm}_{arm}/fold_{f}/training_run/test_predictions.tsv"


# Fold-averaged correlation: score each fold's test set on its own, then average the per-fold
# values. Distinct from BOTH of the other two metrics here — `spearman` pools every fold into one
# vector before correlating, and `ps_spearman_T10` correlates within each structure. The frontier's
# CD-HIT ≤60% column is reported fold-averaged, so this is the only form of the metric that is
# apples-to-apples with it; pooling instead lets a fold with a wider ΔΔG range dominate the rank
# vector, and the two can differ by more than the model gaps being compared.
FOLD_FIELDS = ["spearman_foldavg", "spearman_foldavg_sd", "pearson_foldavg",
               "pearson_foldavg_sd", "n_folds_scored"]

HDR = ["plm", "arm", "n", "pearson", "spearman", "ps_pearson_T10", "ps_spearman_T10",
       "ps_ncomplex_T10", "auroc_destab", "prec_at50"]
FOLD_HDR = ["plm", "arm", "n_folds_scored", "spearman_foldavg", "spearman_foldavg_sd",
            "spearman", "pearson_foldavg", "pearson"]
CSV_COLS = ["split", "plm", "arm", "n", "pearson", "spearman", "spearman_foldavg",
            "ps_spearman_T10", "ps_ncomplex_T10", "auroc_destab", "delta_ppS_vs_base"]


def discover_plms(resdir: Path) -> list[str]:
    """Group `{plm}_{arm}` subdirs back into PLM names. Longest arm suffix first so
    `prostt5_foldx_scalar` -> prostt5 (not prostt5_scalar)."""
    if not resdir.is_dir():
        return []
    found = set()
    for d in resdir.iterdir():
        if not d.is_dir():
            continue
        for arm in sorted(ARMS, key=len, reverse=True):
            if d.name.endswith(f"_{arm}"):
                found.add(d.name[: -len(arm) - 1])
                break
    ordered = [p for p in PLM_ORDER if p in found]
    ordered += sorted(found - set(ordered))
    return ordered


def fold_average(pred_folds, truth_folds) -> dict:
    """Per-fold Spearman/Pearson, averaged over the folds that scored.

    The join is `assemble_arm`'s, restricted to one fold at a time, so a row that is dropped from
    the pooled metric for want of a prediction is dropped here too and the two are computed off
    the same matched set. Folds are weighted equally regardless of size — that is what makes it
    a fold average and not a slower pooling; it is also why the SD travels with the mean, since
    three folds average a spread the single pooled number cannot show.
    """
    sps, prs = [], []
    for f in range(N_FOLDS):
        pf = pred_folds[f]
        if pf is None:
            continue
        tf = truth_folds[f]
        pairs = [(tv, pf[key]) for key, tv in tf.items() if key in pf]
        if len(pairs) < 2:
            continue
        true = np.array([p[0] for p in pairs])
        pred = np.array([p[1] for p in pairs])
        sp, pr = spearman(pred, true), rescore.pearson(pred, true)
        if sp == sp:
            sps.append(sp)
        if pr == pr:
            prs.append(pr)
    nan = float("nan")
    return {
        "spearman_foldavg": float(np.mean(sps)) if sps else nan,
        # ddof=0: these three folds ARE the partition, not a sample drawn from folds.
        "spearman_foldavg_sd": float(np.std(sps)) if len(sps) > 1 else nan,
        "pearson_foldavg": float(np.mean(prs)) if prs else nan,
        "pearson_foldavg_sd": float(np.std(prs)) if len(prs) > 1 else nan,
        "n_folds_scored": len(sps),
    }


def score_split(name: str, csv_rows: list):
    c = CFG[name]
    rescore.TRUTH_TMPL = c["truth"]
    truth = load_truth_folds()
    plms = discover_plms(c["resdir"])
    print(f"\n{'='*92}\n# {c['label']}  (single-point; 3-fold pooled, leakage-free)\n{'='*92}")
    print(f"[truth] {sum(len(t) for t in truth)} rows, per-fold {[len(t) for t in truth]}, "
          f"{len({k[0] for t in truth for k in t})} complexes")
    print(f"[plms ] {', '.join(plms) if plms else '(none found)'}")

    table = []  # (plm, {arm: row})
    for plm in plms:
        arm_rows = {}
        for arm in ARMS:
            files = [c["resdir"] / PRED.format(plm=plm, arm=arm, f=f) for f in range(N_FOLDS)]
            missing = [f.parent.parent.name for f in files if not f.exists()]
            if missing:
                if arm == "base" or arm in ("foldx", "foldx_scalar"):
                    print(f"  [skip] {plm}_{arm}: missing fold preds {missing}")
                continue
            pred_folds = [load_pred_file(f) for f in files]
            true, pred, pdb, warns = assemble_arm(pred_folds, truth)
            for w in warns:
                print(f"  [{plm}_{arm}] {w}")
            row, _ = score_arm(f"{plm}_{arm}", true, pred, pdb)
            row.update(fold_average(pred_folds, truth))
            arm_rows[arm] = row
        if arm_rows:
            table.append((plm, arm_rows))

    # Rung CSV in rescore's METRIC_FIELDS schema (arm = "<tag>_<suffix>") so results_matrix.py's
    # build_ps ingests this SP rung exactly like the CATH/MP rungs — one source of truth.
    def _cell(v):
        if isinstance(v, float):
            return "" if v != v else f"{v:.4f}"     # NaN -> blank
        return str(v)
    fields = METRIC_FIELDS + FOLD_FIELDS   # FOLD_FIELDS appended, never interleaved, so a reader
                                           # pinned to METRIC_FIELDS by position is unaffected
    with open(c["rung"], "w") as fh:
        fh.write(",".join(fields) + "\n")
        for plm, arm_rows in table:
            for arm in ARMS:
                if arm in arm_rows:
                    r = arm_rows[arm]
                    fh.write(",".join(_cell(r.get(k, "")) for k in fields) + "\n")
    print(f"[rung ] wrote {c['rung'].relative_to(ROOT)}")

    # Pooled + per-structure table
    print("\n## Pooled + per-structure metrics")
    print("  " + "  ".join(f"{h:>15}" if h not in ("plm", "arm") else f"{h:<14}" for h in HDR))
    for plm, arm_rows in table:
        for arm in ARMS:
            if arm not in arm_rows:
                continue
            r = arm_rows[arm]
            cells = []
            for h in HDR:
                if h == "plm":
                    cells.append(f"{plm:<14}")
                elif h == "arm":
                    cells.append(f"{arm:<14}")
                else:
                    v = r[h]
                    cells.append(f"{v:>15.4f}" if isinstance(v, float) else f"{v:>15}")
            print("  " + "  ".join(cells))

    # Fold-averaged table — the frontier-comparable form on the clustered tier. `spearman` is
    # printed beside it so the pooled/fold-avg gap is visible on the same line.
    print("\n## Fold-averaged vs pooled correlation")
    print("  " + "  ".join(f"{h:>18}" if h not in ("plm", "arm") else f"{h:<14}" for h in FOLD_HDR))
    for plm, arm_rows in table:
        for arm in ARMS:
            if arm not in arm_rows:
                continue
            r = arm_rows[arm]
            cells = []
            for h in FOLD_HDR:
                if h == "plm":
                    cells.append(f"{plm:<14}")
                elif h == "arm":
                    cells.append(f"{arm:<14}")
                else:
                    v = r[h]
                    cells.append(f"{v:>18.4f}" if isinstance(v, float) else f"{v:>18}")
            print("  " + "  ".join(cells))

    # Gate A + CSV rows
    print("\n## Gate A — FoldX per-structure Spearman lift over base (T>=10)")
    for plm, arm_rows in table:
        b = arm_rows["base"]["ps_spearman_T10"] if "base" in arm_rows else float("nan")
        for arm in ARMS:
            if arm not in arm_rows:
                continue
            r = arm_rows[arm]
            d = (r["ps_spearman_T10"] - b) if arm != "base" else float("nan")
            if arm != "base":
                print(f"  {plm:12s} {arm:14s} ppS = {r['ps_spearman_T10']:.4f}   Δ vs base = {d:+.4f}")
            csv_rows.append([name, plm, arm, r["n"], f"{r['pearson']:.4f}", f"{r['spearman']:.4f}",
                             f"{r['spearman_foldavg']:.4f}",
                             f"{r['ps_spearman_T10']:.4f}", r["ps_ncomplex_T10"],
                             f"{r['auroc_destab']:.4f}", "" if arm == "base" else f"{d:+.4f}"])


def main():
    args = sys.argv[1:]
    csv_path = ROOT / "scratch/sp_perstructure_ppS.csv"
    if "--csv" in args:
        i = args.index("--csv")
        csv_path = Path(args[i + 1]); del args[i:i + 2]
    which = args[0] if args else "both"
    if which == "both":            # the two SP tiers, unchanged default
        splits = ["clustered", "bycomplex"]
    elif which == "all":           # every configured tier, in CFG order
        splits = list(CFG)
    else:
        splits = [which]
    unknown = [s for s in splits if s not in CFG]
    if unknown:
        sys.exit(f"unknown tier(s) {unknown}; configured: {', '.join(CFG)}")

    csv_rows = []
    for name in splits:
        score_split(name, csv_rows)

    with open(csv_path, "w") as fh:
        fh.write(",".join(CSV_COLS) + "\n")
        for r in csv_rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[wrote] {csv_path}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
