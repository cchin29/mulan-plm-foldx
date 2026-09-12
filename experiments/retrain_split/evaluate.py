#!/usr/bin/env python3
"""Evaluate the by-complex (leakage-controlled) retrain predictions.

THIN wrapper over the frontier-metric core in
`experiments/rescore_perstructure/rescore.py` (imported, not re-implemented).
Parameterized for the by-complex retrain:

  * 3 folds (not 10),
  * predictions at scratch/results/retrain_bycomplex/<tag>_<arm>/fold_{0,1,2}/training_run/test_predictions.tsv
    arms {base, foldx}; tags {ankh, prostt5, saprot, esmc600m, esm2},
  * truth (per by-complex test fold) at
    scratch/foldx_s1102/splits_bycomplex_foldxdec/fold_{f}/S1102_filtered_test.tsv (16-col)
    or scratch/splits_bycomplex_seed42/fold_{f}/S1102_filtered_test.tsv (4-col) as a fallback.

Reuses, unchanged, from rescore.py: the metric functions (pearson/spearman/rmse/
mae/rmse_corr/mae_corr/auroc/precision_recall_at_k/per_structure), the loaders
(_pdb_of / load_pred_file / load_truth_folds / assemble_arm), score_arm,
METRIC_FIELDS and fmt. The rescore module globals N_FOLDS and TRUTH_TMPL are
re-pointed to the 3-fold by-complex layout before the loaders are called.

Cluster-bootstrap CIs + paired (foldx-base) contrasts mirror
`experiments/rescore_perstructure/bootstrap_ci.py` (that script is a hard-coded,
top-level-executable module with a Linux ROOT, so it cannot be imported here; its
algorithm -- per-complex Spearman over complexes with >=T muts, shared resample
indices for paired deltas, percentile CI, P(delta>0) -- is replicated using
rescore.spearman for the correlations).

Outputs (written to experiments/retrain_split/, never touching rescore_perstructure/):
  * results_bycomplex.csv   -- one row per arm, same columns as rescore results.csv
  * bootstrap_bycomplex.txt -- per-arm CI table + paired foldx-base contrasts
  * SUMMARY.md              -- headline leaky -> by-complex comparison (Gate A) + tables

Run:  ./.venv/bin/python experiments/retrain_split/evaluate.py
Metric-core self-test still lives in rescore.py:  ... rescore.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Import the rescore metric core (add its dir to sys.path so this works whether
# evaluate.py is run as a script or imported as experiments.retrain_split.evaluate).
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # <root>/experiments/retrain_split -> <root>
RESCORE_DIR = ROOT / "experiments" / "rescore_perstructure"
if str(RESCORE_DIR) not in sys.path:
    sys.path.insert(0, str(RESCORE_DIR))

import rescore  # noqa: E402  (metric core)
from rescore import (  # noqa: E402
    METRIC_FIELDS,
    PS_THRESHOLDS,
    STRONG,
    assemble_arm,
    fmt,
    load_pred_file,
    load_truth_folds,
    score_arm,
    spearman,
)

# --------------------------------------------------------------------------
# By-complex configuration (this is the ONLY thing that differs from rescore.py).
# --------------------------------------------------------------------------
N_FOLDS = 3
TAGS = ("ankh", "prostt5", "saprot", "esmc600m", "esmc6b", "esm2", "ankh3_large", "ankh3_xl", "mint")
ARMS = ("base", "foldx", "foldx_scalar")

# Two split ladders share this harness (leaky -> by-complex -> clustered). --split selects one;
# all path templates + output basenames are swapped from here (module globals mutated in main()).
SPLITS = {
    "bycomplex": {
        "pred": ROOT / "scratch/results/retrain_bycomplex/{tag}_{arm}/fold_{f}/training_run/test_predictions.tsv",
        "foldxdec": ROOT / "scratch/foldx_s1102/splits_bycomplex_foldxdec/fold_{f}/S1102_filtered_test.tsv",
        "seed42": ROOT / "scratch/splits_bycomplex_seed42/fold_{f}/S1102_filtered_test.tsv",
        "suffix": "bycomplex",
        "label": "by-complex (whole PDB held out)",
    },
    "clustered": {
        "pred": ROOT / "scratch/results/retrain_clustered/{tag}_{arm}/fold_{f}/training_run/test_predictions.tsv",
        "foldxdec": ROOT / "scratch/foldx_s1102/splits_clustered_id60_foldxdec/fold_{f}/S1102_filtered_test.tsv",
        "seed42": ROOT / "scratch/splits_clustered_id60_kfold/fold_{f}/S1102_filtered_test.tsv",
        "suffix": "clustered",
        "label": "homology-clustered (whole family held out, mmseqs <=60% id)",
    },
}
# active templates -- rebound in main() from the chosen --split.
PRED_TMPL = SPLITS["bycomplex"]["pred"]
TRUTH_FOLDXDEC = SPLITS["bycomplex"]["foldxdec"]
TRUTH_SEED42 = SPLITS["bycomplex"]["seed42"]
OUT_SUFFIX = "bycomplex"
SPLIT_LABEL = SPLITS["bycomplex"]["label"]

OUT = HERE
LEAKY_CSV = RESCORE_DIR / "results.csv"   # existing balanced-splitter baseline (read-only)

# by-complex arm-id <-> leaky arm-id in rescore results.csv (retrain "foldx" = phase2 FoldX-MLP,
# labelled "foldxmlp" in the leaky re-score). "foldx_scalar" (phase1 single-scalar) has NO leaky
# counterpart (the leaky re-score never ran a scalar arm) -> None means "by-complex only".
LEAKY_ARM = {"base": "base", "foldx": "foldxmlp", "foldx_scalar": None}
# Per-tag override for the leaky base arm-id in results.csv when it isn't "<tag>_base":
# MINT's leaky lineage is `mint_complex`; ankh3_large/xl have NO leaky counterpart (-> NaN).
LEAKY_BASE_ID = {"mint": "mint_complex"}

T_BOOT = 10          # per-structure min-count threshold for the bootstrap (matches bootstrap_ci.py)
B_BOOT = 10000       # bootstrap resamples
BOOT_SEED = 0

TS = None  # scope caveat header timestamp filled in main()


# --------------------------------------------------------------------------
# Truth selection: prefer the 16-col foldxdec split (SPEC layout), fall back to
# the 4-col seed42 split. Both carry identical (pdb, mutation) -> true_ddg.
# --------------------------------------------------------------------------
def choose_truth_template(log):
    for tmpl, name in ((TRUTH_FOLDXDEC, "foldxdec 16-col"), (TRUTH_SEED42, "seed42 4-col")):
        if all(Path(str(tmpl).format(f=f)).exists() for f in range(N_FOLDS)):
            log(f"[truth] using {name}: {tmpl}")
            return tmpl
    raise FileNotFoundError(
        "No complete by-complex truth split found (need all 3 folds of either "
        f"{TRUTH_FOLDXDEC} or {TRUTH_SEED42})."
    )


# --------------------------------------------------------------------------
# Per-complex Spearman map for one arm (over complexes with >= T muts), reusing
# rescore.spearman. Mirrors bootstrap_ci.py's `percx`.
# --------------------------------------------------------------------------
def per_complex_spearman(true, pred, pdb, T):
    out = {}
    for c in np.unique(pdb):
        m = pdb == c
        if int(m.sum()) < T:
            continue
        s = spearman(true[m], pred[m])
        if not np.isnan(s):
            out[str(c)] = s
    return out


def bootstrap_report(arm_data, log):
    """Cluster-bootstrap CIs over test complexes + paired foldx-base contrasts.

    arm_data: {arm_id: (true, pred, pdb, warns)}. Returns list[str] report lines.
    Mirrors bootstrap_ci.py: intersection of qualifying complexes across all present
    arms, shared resample indices -> paired, percentile 95% CI, P(delta>0).
    """
    lines = []

    def w(s):
        lines.append(s)

    percx = {a: per_complex_spearman(t, p, pdb, T_BOOT) for a, (t, p, pdb, _) in arm_data.items()}
    percx = {a: d for a, d in percx.items() if d}            # drop arms with no qualifying complex
    if not percx:
        w(f"[bootstrap] no arm has a complex with >= {T_BOOT} muts (partial folds?) -- skipped.")
        for s in lines:
            log(s)
        return lines

    common = sorted(set.intersection(*[set(d) for d in percx.values()]))
    w(f"==== per-structure Spearman (T>={T_BOOT}) with 95% cluster-bootstrap CI ====")
    w(f"qualifying complexes per arm: "
      f"{ {a: len(d) for a, d in sorted(percx.items())} }")
    w(f"complexes common across all present arms: {len(common)}")
    if len(common) < 2:
        w("[bootstrap] < 2 common complexes -> CIs/contrasts not meaningful yet (partial folds).")
        for s in lines:
            log(s)
        return lines

    M = {a: np.array([percx[a][c] for c in common]) for a in percx}
    nC = len(common)
    rng = np.random.default_rng(BOOT_SEED)
    BOOT = rng.integers(0, nC, size=(B_BOOT, nC))   # shared resample -> paired contrasts

    def ci(vec):
        bs = vec[BOOT].mean(axis=1)
        return vec.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    w("")
    w(f"{'arm':>18} {'mean':>7}  [{'2.5%':>6},{'97.5%':>6}]")
    for a in sorted(percx, key=lambda a: -M[a].mean()):
        m, lo, hi = ci(M[a])
        w(f"{a:>18} {m:7.3f}  [{lo:6.3f},{hi:6.3f}]")

    def paired(a, b):
        d = M[a] - M[b]
        bs = d[BOOT].mean(axis=1)
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), float((bs > 0).mean())

    w("")
    w("==== paired contrasts (delta = foldx_variant - base, shared complex resample) ====")
    w(f"{'contrast':>30} {'dmean':>7}  [{'2.5%':>6},{'97.5%':>6}]  P(>0)")
    any_contrast = False
    for tag in TAGS:
        b = f"{tag}_base"
        for fx in ("foldx", "foldx_scalar"):
            a = f"{tag}_{fx}"
            if a in M and b in M:
                any_contrast = True
                dm, lo, hi, pg = paired(a, b)
                w(f"{a + ' - ' + b:>30} {dm:7.3f}  [{lo:6.3f},{hi:6.3f}]  {pg:.3f}")
    if not any_contrast:
        w("(no tag has BOTH a base and a foldx/foldx_scalar arm complete yet -- no paired contrasts)")

    for s in lines:
        log(s)
    return lines


# --------------------------------------------------------------------------
# Leaky baseline (existing rescore results.csv) -> {arm_id: {field: value}}.
# --------------------------------------------------------------------------
def _load_results_csv(path):
    """Parse a rescore-style results CSV -> {arm_id: {field: value}} (floats, NaN on blanks)."""
    if not Path(path).exists():
        return {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split(",")
        rows = {}
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            vals = line.split(",")
            d = {}
            for k, v in zip(header, vals):
                if k == "arm":
                    d[k] = v
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = float("nan")
            rows[d["arm"]] = d
    return rows


def load_leaky(log):
    rows = _load_results_csv(LEAKY_CSV)
    if not rows:
        log(f"[leaky] {LEAKY_CSV} not found -- headline comparison will show leaky as NaN.")
    else:
        log(f"[leaky] loaded {len(rows)} arms from {LEAKY_CSV}")
    return rows


def _get(d, arm, field):
    return d.get(arm, {}).get(field, float("nan"))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    import argparse
    global PRED_TMPL, TRUTH_FOLDXDEC, TRUTH_SEED42, OUT_SUFFIX, SPLIT_LABEL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=list(SPLITS), default="bycomplex",
                    help="which split ladder to score (default: bycomplex)")
    a = ap.parse_args()
    cfg = SPLITS[a.split]
    PRED_TMPL, TRUTH_FOLDXDEC, TRUTH_SEED42 = cfg["pred"], cfg["foldxdec"], cfg["seed42"]
    OUT_SUFFIX, SPLIT_LABEL = cfg["suffix"], cfg["label"]

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg, file=sys.stderr)

    log(f"[split] scoring '{a.split}' -- {SPLIT_LABEL}")
    # Re-point the rescore module globals at the chosen 3-fold layout, then
    # reuse its truth loader / assembler unchanged.
    truth_tmpl = choose_truth_template(log)
    rescore.N_FOLDS = N_FOLDS
    rescore.TRUTH_TMPL = truth_tmpl
    truth_folds = load_truth_folds()

    n_total = sum(len(d) for d in truth_folds)
    complexes = {}
    for d in truth_folds:
        for (pdb, _mut) in d:
            complexes[pdb] = complexes.get(pdb, 0) + 1
    log(f"[truth] pooled rows={n_total} over {len(complexes)} complexes "
        f"(full by-complex set = 1100 over 110 when all 3 folds present)")

    # ---- assemble every present arm ----
    arm_data = {}
    for tag in TAGS:
        for arm in ARMS:
            arm_id = f"{tag}_{arm}"
            pred_folds = [
                load_pred_file(Path(str(PRED_TMPL).format(tag=tag, arm=arm, f=f)))
                for f in range(N_FOLDS)
            ]
            if all(pf is None for pf in pred_folds):
                log(f"[skip] {arm_id}: no prediction files found "
                    f"({PRED_TMPL.name} under {tag}_{arm})")
                continue
            n_present = sum(pf is not None for pf in pred_folds)
            if n_present < N_FOLDS:
                log(f"[partial] {arm_id}: {n_present}/{N_FOLDS} folds present (still training?)")
            arm_data[arm_id] = assemble_arm(pred_folds, truth_folds)

    if not arm_data:
        log("[done] no completed folds yet -- no arms to score. "
            "Re-run once retrain writes test_predictions.tsv.")
        _write_outputs([], {}, {}, complexes, [], log_lines, n_present_any=False)
        return

    # ---- score each arm (reusing rescore.score_arm) ----
    rows = []
    aux_by_arm = {}
    for arm_id, (true, pred, pdb, warns) in arm_data.items():
        for wmsg in warns:
            log(f"[warn] {arm_id}: {wmsg}")
        log(f"[score] {arm_id}: n={true.size} rows, "
            f"{len(np.unique(pdb))} complexes")
        row, aux = score_arm(arm_id, true, pred, pdb)
        rows.append(row)
        aux_by_arm[arm_id] = aux

    # ---- bootstrap CIs + paired contrasts ----
    boot_lines = bootstrap_report(arm_data, log)

    # ---- leaky baseline for the headline comparison ----
    leaky = load_leaky(log)

    _write_outputs(rows, aux_by_arm, leaky, complexes, boot_lines, log_lines,
                   n_present_any=True)
    log(f"[out] wrote {OUT/f'results_{OUT_SUFFIX}.csv'}, {OUT/f'bootstrap_{OUT_SUFFIX}.txt'}, "
        f"{OUT/_summary_name()}")


def _summary_name():
    return "SUMMARY.md" if OUT_SUFFIX == "bycomplex" else f"SUMMARY_{OUT_SUFFIX}.md"


def _write_outputs(rows, aux_by_arm, leaky, complexes, boot_lines, log_lines, n_present_any):
    # results_bycomplex.csv (same columns as rescore results.csv)
    rows_sorted = sorted(
        rows,
        key=lambda r: (-9 if np.isnan(r.get("ps_spearman_T10", float("nan")))
                       else r["ps_spearman_T10"]),
        reverse=True,
    )
    csv_path = OUT / f"results_{OUT_SUFFIX}.csv"
    with open(csv_path, "w") as fh:
        fh.write(",".join(METRIC_FIELDS) + "\n")
        for r in rows_sorted:
            fh.write(",".join(fmt(r.get(k, float("nan"))) for k in METRIC_FIELDS) + "\n")

    (OUT / f"bootstrap_{OUT_SUFFIX}.txt").write_text("\n".join(boot_lines) + "\n" if boot_lines else "")

    _write_summary(rows_sorted, leaky, complexes, boot_lines, log_lines, n_present_any)


def _write_summary(rows, leaky, complexes, boot_lines, log_lines, n_present_any):
    L = []
    L.append(f"# {SPLIT_LABEL} retrain -- per-structure re-score")
    L.append("")
    L.append("_Generated by `experiments/retrain_split/evaluate.py` (thin wrapper over the "
             "`rescore_perstructure` metric core). Sorted by per-structure Spearman (T=10) desc._")
    L.append("")
    L.append(f"> **Leakage-controlled generalization number ({SPLIT_LABEL}).** Predictions come from the "
             f"**3-fold** retrain on this split, vs the leaky per-mutation 10-fold baseline in "
             "`../rescore_perstructure/`. The FoldX-base per-structure delta as the split gets "
             "stricter (leaky -> by-complex -> clustered) is **Gate A** of "
             "`docs/history/PLAN_RETRAIN_BYCOMPLEX.md`.")
    L.append("")
    pooled = sum(complexes.values())
    L.append(f"Pooled test rows so far: **{pooled}** over **{len(complexes)}** complexes "
             f"(complete run = 1100 over 110).")
    L.append("")

    if not n_present_any:
        L.append("## Status")
        L.append("")
        L.append("**No completed folds yet** -- the retrain has not written any "
                 "`test_predictions.tsv`. Re-run this script once folds complete.")
        L.append("")
        _append_runlog(L, log_lines)
        (OUT / _summary_name()).write_text("\n".join(L) + "\n")
        return

    # ---- Gate A headline: FoldX-base per-structure delta as the split gets stricter ----
    # Ladder: leaky (10-fold, upper bound) -> by-complex -> clustered. The by-complex run is the
    # middle rung; for the clustered summary we read its results CSV so the ladder shows all three.
    this = "bycx" if OUT_SUFFIX == "bycomplex" else "clust"
    prior = _load_results_csv(OUT / "results_bycomplex.csv") if OUT_SUFFIX == "clustered" else None
    L.append(f"## Gate A -- FoldX-base per-structure delta: leaky -> ... -> {this}")
    L.append("")
    L.append("Per-structure Spearman (T=10). `leaky` from `../rescore_perstructure/results.csv` "
             "(balanced 10-fold, upper bound). delta = foldx(mlp) - base within each split; the "
             "**central result is whether that delta HOLDS/WIDENS as the split gets stricter** "
             "(Gate A). `ddelta = delta_" + this + " - delta_leaky`.")
    L.append("")
    hdr = ["PLM", "base leaky", "base " + this, "foldx leaky", "foldx " + this,
           "delta leaky", "delta " + this, "ddelta", "scalar " + this, "dscalar " + this]
    if prior is not None:
        hdr[2:2] = ["base bycx"]          # insert the by-complex middle rung for reference
        hdr.insert(6, "foldx bycx")
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join(["---"] * len(hdr)) + "|")
    by = {r["arm"]: r for r in rows}

    def _ps(d, arm):
        return d.get(arm, {}).get("ps_spearman_T10", float("nan"))

    for tag in TAGS:
        b_this = _ps(by, f"{tag}_base")
        f_this = _ps(by, f"{tag}_foldx")
        s_this = _ps(by, f"{tag}_foldx_scalar")
        b_leak = _get(leaky, LEAKY_BASE_ID.get(tag, f"{tag}_{LEAKY_ARM['base']}"), "ps_spearman_T10")
        f_leak = _get(leaky, f"{tag}_{LEAKY_ARM['foldx']}", "ps_spearman_T10")
        d_leak = f_leak - b_leak
        d_this = f_this - b_this
        dd = d_this - d_leak
        ds_this = s_this - b_this          # scalar has no leaky counterpart (phase1 not re-scored)
        if np.isnan(b_this) and np.isnan(f_this) and np.isnan(s_this):
            continue
        vals = [tag, b_leak, b_this, f_leak, f_this, d_leak, d_this, dd, s_this, ds_this]
        if prior is not None:
            vals[2:2] = [_ps(prior, f"{tag}_base")]
            vals.insert(6, _ps(prior, f"{tag}_foldx"))
        L.append("| " + " | ".join(fmt(x) for x in vals) + " |")
    L.append("")
    L.append("_(NaN = that arm/fold set not complete yet; re-run as folds finish. "
             "A positive `ddelta` = FoldX-MLP's per-structure advantage **widens** as leakage is "
             "removed = pre-registered Gate-A win. `scalar` = phase1 single-Interaction-Energy arm "
             "(no leaky counterpart); compare `dscalar` vs `delta` for the plan's "
             "scalar-vs-MLP-on-small-sets question.)_")
    L.append("")

    # ---- full metric table ----
    L.append(f"## Full metrics ({this})")
    L.append("")
    cols = ["arm", "n", "pearson", "spearman",
            "ps_pearson_T10", "ps_spearman_T10", "ps_ncomplex_T10",
            "ps_spearman_T5", "auroc_destab", "auroc_strong", "prec_at50", "recall_at50"]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        L.append("| " + " | ".join(fmt(r.get(c, float("nan"))) for c in cols) + " |")
    L.append("")

    # ---- Gate B/C read ----
    L.append("## Gate B / Gate C read")
    L.append("")
    L.append(f"- **Gate B (how far base rho falls):** compare the `base {this}` column above against "
             "`base leaky`. The leaky per-structure base rho was ~0.32-0.49 (already an upper bound); "
             "the stricter-split value is the leakage-controlled generalization figure.")
    L.append(f"- **Gate C (arm ordering out-of-distribution):** does FoldX > base survive? Read the "
             f"paired-contrast P(>0) in `bootstrap_{OUT_SUFFIX}.txt`.")
    L.append("")

    # ---- bootstrap block ----
    L.append("## Cluster-bootstrap CIs + paired contrasts")
    L.append("")
    L.append("_Cluster bootstrap over test complexes (per-structure Spearman T>=10), "
             "shared resample indices for paired foldx-base deltas -- mirrors "
             "`../rescore_perstructure/bootstrap_ci.py`._")
    L.append("")
    L.append("```")
    L.extend(boot_lines if boot_lines else ["(bootstrap skipped -- see run log)"])
    L.append("```")
    L.append("")

    _append_runlog(L, log_lines)
    (OUT / _summary_name()).write_text("\n".join(L) + "\n")


def _append_runlog(L, log_lines):
    L.append("## Run log")
    L.append("")
    L.append("```")
    L.extend(log_lines)
    L.append("```")


if __name__ == "__main__":
    main()
