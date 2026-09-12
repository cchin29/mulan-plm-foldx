#!/usr/bin/env python3
"""Status + data matrix across {PLM} x {arm} x {evaluation regime}.

Walks every result tree, reads `test_pcc` from `<dir>/fold_*/training_run/all_results.json`
(the same key aggregate_folds.py uses), and prints a per-regime matrix plus a tidy long CSV
(scripts_plots/results_matrix.csv). Re-run any time to pull fresh data/progress as folds land —
a cell fills the moment its folds exist. Stdlib only.

Arms (normalized — dir naming differs by regime, this collapses it):
  base       plain lightatt (no scores)
  aug        Tier-1 augmentation (reverse+identity)
  fx_scalar  +FoldX Stage-1 scalar Interaction-Energy   (foldx_s1102: `foldx`; frontier: `_foldx_scalar`)
  fx_mlp     +FoldX Stage-2 12-term MLP                  (foldx_s1102: `foldx_mlp`; frontier: `_foldx`)

Regimes:
  S1102-CV 50ep     cv10_* splits, 50 epoch / patience 10          (10 folds)
  S1102-CV 300ep    embedding_sweep[_balanced], 300 ep / p30       (10 folds)
  Benchmarks 50ep   S1131 / S2003 / S4169 CV                       (10 folds)
  Honest ladder     TWO SEPARATE datasets — never mixed:            (3 folds)
   — S1102           retrain_{bycomplex,clustered}  (S1102_filtered, 1100 muts; the EARLY runs)
   — full-SKEMPI     full_skempi (clustered-SP), full_skempi_bycomplex (bycomplex-SP),
                     full_skempi_mp_{clustered,bycomplex} (multi-point)
  CATH-superfamily  USP-ddG's exact 813-mut CATH hold-out          (1 fold)
                    full-SKEMPI; --ps breaks out single/multiple/all

Every CSV carries test-set counts (pooled: mean_ntest/total_ntest; --ps: n_muts/n_complex_T10)
so an S1102 row (~366/fold, ~1100 total) can never be mistaken for a full-SKEMPI row (thousands).

Metric = test_pcc (pooled Pearson) — uniform + present in every all_results.json. NOTE the
Frontier headline metric is per-structure Spearman (see experiments/*/SUMMARY*.md +
results_*.csv); this tool tracks pooled pcc as the common done/progress signal.

    python scripts_plots/results_matrix.py              # all regimes (pooled test_pcc)
    python scripts_plots/results_matrix.py frontier     # filter by regime-name substring
    python scripts_plots/results_matrix.py --csv-only    # just (re)write the pooled CSV, no print
    python scripts_plots/results_matrix.py --ps          # per-structure Spearman honest ladder
                                                         #   -> results_matrix_ps.csv + Δ FoldX lift
    python scripts_plots/results_matrix.py --res         # runtime (min/fold) + inferred platform

Runtime (train_runtime) is real, from all_results.json. Platform is INFERRED by provenance
(device isn't recorded). Max memory is NOT captured (HF skip_memory_metrics default True) — the
pooled CSV carries mean_runtime_s + platform columns; memory would need a re-run to record.
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
R = os.path.join(ROOT, "scratch", "results")
FX = os.path.join(ROOT, "scratch", "foldx_s1102")
GPU = os.path.join(ROOT, "scratch", "gpu_results", "data", "benchmarks")

# tag -> display name (order = matrix row order). Mirrors aggregate_folds.TAG2NAME + the
# structure-split-only tags (mint) so a model reads identically across tools.
MODELS = [
    ("ankh", "Ankh-large"), ("prostt5", "ProstT5"), ("saprot", "SaProt"),
    ("saprot13b", "SaProt-1.3B"), ("esm2", "ESM2-3B"), ("esm3", "ESM3-1.4B"),
    ("esmc600m", "ESM-C 600M"), ("esmc6b", "ESM-C 6B"),
    ("ankh3_large", "Ankh3-large"), ("ankh3_xl", "Ankh3-xl"),
    ("aido", "AIDO-16B"), ("mint", "MINT"),
]
ARMS = ["base", "aug", "fx_scalar", "fx_mlp"]


def _count_test_rows(run_dir):
    """Exact test-set size for a fold = lines in test_predictions.tsv.

    No header subtraction: the writer emits no header row. Every line of all 2395 of these files
    is data (`1A22_A\\t1A22_B\\tIA168A\\t0.4773407`). Deducting one made every count in the CSV
    short by one per fold, which showed as S1102 rows reading 109/1090 against the 110/1100 the
    regime label itself names, and the honest ladder reading 1097 under a heading that says 1100.
    """
    p = os.path.join(run_dir, "test_predictions.tsv")
    try:
        with open(p) as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def read_folds(dirpath):
    """{fold: (test_pcc, train_runtime_s, n_test)} for one arm dir (first-wins on dup fold ids).

    n_test = exact test rows from test_predictions.tsv — the sanity check that distinguishes
    S1102 (~366/fold) from full-SKEMPI (thousands/fold) so the two ladders can't be confused.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "fold_*", "training_run", "all_results.json"))):
        try:
            fold = int(os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            with open(f) as fh:
                j = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        v = j.get("test_pcc")
        if v is not None and fold not in out:
            out[fold] = (v, j.get("train_runtime"), _count_test_rows(os.path.dirname(f)))
    return out


# Platform is not recorded in all_results.json/logs; infer by provenance. esmc6b ran only on the
# GPU (CUDA) box; anything under gpu_results/incoming = GPU; else the Mac (MPS). Marked '*' = inferred.
def platform_of(path, tag):
    if not path:
        return ""
    if tag == "esmc6b" or "gpu_results" in path or "incoming_esmc6b" in path:
        return "CUDA*"
    return "MPS*"


def cell(candidates, tag=""):
    """First candidate dir with folds → (n, mean_pcc, std_pcc, mean_runtime_s, ntest, platform, path).

    ntest = (mean test rows/fold, total test rows across folds) — dataset-size sanity check.
    """
    for d in candidates:
        fs = read_folds(d)
        if fs:
            pccs = [p for p, _, _ in fs.values()]
            rts = [r for _, r, _ in fs.values() if r is not None]
            nts = [n for _, _, n in fs.values() if n is not None]
            std = st.pstdev(pccs) if len(pccs) > 1 else 0.0
            mrt = st.mean(rts) if rts else None
            ntest = (round(st.mean(nts)), sum(nts)) if nts else (None, None)
            return len(pccs), st.mean(pccs), std, mrt, ntest, platform_of(d, tag), d
    return 0, None, None, None, (None, None), "", None


# ---- regime path rules: (tag, dataset) -> {arm: [candidate dirs, first non-empty wins]} ----
def s1102_50(tag, ds):
    tagless = ["/cv10"] if tag == "ankh" else []  # foldx_s1102/cv10 (tagless) = ankh reference
    # `_nlu` fallback: the two Ankh3 backbones were run under BOTH prefix conventions, so their
    # dirs carry the prefix in the name (cv10_ankh3_large_nlu, and cv10_ankh3_large_s2s alongside).
    # [NLU] is the convention embed_sequence defaults to and the one RESULTS.md quotes; the bare
    # cv10_ankh3_* these rules looked for never existed, which read as four empty cells against
    # four fully-scored 10-fold runs. Only ankh3 has such a dir, so the fallback is inert elsewhere
    # -- and it deliberately does not fall back to `_s2s`, which is a different experiment.
    return {
        "base": [f"{R}/cv10_{tag}", f"{R}/cv10_{tag}_nlu"],
        "aug": [f"{R}/cv10_aug_{tag}", f"{R}/cv10_{tag}_aug", f"{R}/cv10_{tag}_aug_nlu"],
        "fx_scalar": [f"{FX}/cv10_{tag}/foldx"] + [f"{FX}{t}/foldx" for t in tagless],
        "fx_mlp": [f"{FX}/cv10_{tag}/foldx_mlp"] + [f"{FX}{t}/foldx_mlp" for t in tagless],
    }


def s1102_300(tag, ds):
    return {  # balanced preferred, rng.integers fallback; foldxmlp only in the balanced sweep
        "base": [f"{R}/embedding_sweep_balanced/{tag}", f"{R}/embedding_sweep/{tag}"],
        "aug": [f"{R}/embedding_sweep_balanced/{tag}_aug", f"{R}/embedding_sweep/{tag}_aug"],
        # balanced 300ep FoldX sweep. The GPU box writes the same runs under a different layout
        # (s1102_<tag>_balanced/{foldx,foldx_mlp}); verified identical for aido, and it is the ONLY
        # home of the esmc6b FoldX arms.
        "fx_scalar": [f"{R}/embedding_sweep_balanced/{tag}_foldxscalar", f"{R}/s1102_{tag}_balanced/foldx"],
        "fx_mlp": [f"{R}/embedding_sweep_balanced/{tag}_foldxmlp", f"{R}/s1102_{tag}_balanced/foldx_mlp"],
    }


def bench_50(tag, ds):
    b = f"{GPU}/{ds}/foldx/cv10_{tag}_bench"
    return {
        "base": [f"{R}/bench_{ds}_{tag}"],
        "aug": [f"{R}/bench_{ds}_{tag}_aug"],
        "fx_scalar": [f"{b}/foldx"],
        "fx_mlp": [f"{b}/foldx_mlp"],
    }


# IMPORTANT: two DISTINCT datasets — keep them in separate regimes, never mix.
#   * S1102 honest ladder  = the EARLY runs on S1102_filtered (1100 muts): retrain_{bycomplex,clustered}.
#   * full-SKEMPI honest ladder = the frontier runs on all of SKEMPI (single-point SP + multi-point MP):
#       full_skempi (clustered-SP), full_skempi_bycomplex (bycomplex-SP), full_skempi_mp_{clustered,bycomplex}.
#   * CATH (full-SKEMPI) is its own single-fold regime below.
# `retrain_*` (S1102, ~1100 muts) and `full_skempi*` (full SKEMPI) are NOT the same test — a model's
# S1102-clustered row is not comparable to its full-SKEMPI-clustered row. Arm dir naming is shared:
# `_foldx` = 12-term MLP, `_foldx_scalar` = scalar.
S1102_LADDER_DIR = {  # S1102_filtered (1100 muts) — early honest-ladder runs
    "S1102 by-complex": "retrain_bycomplex",
    "S1102 clustered": "retrain_clustered",
}
FULLSK_LADDER_DIR = {  # full SKEMPI — single-point (SP) + multi-point (MP) + combined
    "fullSK clustered-SP": "full_skempi",
    "fullSK bycomplex-SP": "full_skempi_bycomplex",
    "fullSK clustered-MP": "full_skempi_mp_clustered",
    "fullSK bycomplex-MP": "full_skempi_mp_bycomplex",
    # COMBINED single+multi, whole-PDB hold-out = the frontier's own protocol (RDE-Network /
    # DiffAffinity / Prompt-DDG / BA-DDG train ONE model on single+multi). NOT the SP and MP rungs
    # pooled — those are two separately trained models. 5801 muts / 337 complexes, 3 folds.
    "fullSK bycomplex-ALL": "full_skempi_bycomplex_all",
}


def _ladder(dirmap):
    def rule(tag, ds):
        d = f"{R}/{dirmap[ds]}"
        return {"base": [f"{d}/{tag}_base"], "aug": [],
                "fx_scalar": [f"{d}/{tag}_foldx_scalar"], "fx_mlp": [f"{d}/{tag}_foldx"]}
    return rule


def cath(tag, ds):
    d = f"{R}/full_skempi_cath"  # single-fold (num_folds=1): USP-ddG's exact CATH-superfamily hold-out
    return {  # full SKEMPI; arm naming: `_foldx` = 12-term MLP, `_foldx_scalar` = scalar
        "base": [f"{d}/{tag}_base"], "aug": [],
        "fx_scalar": [f"{d}/{tag}_foldx_scalar"], "fx_mlp": [f"{d}/{tag}_foldx"],
    }


REGIMES = [
    ("S1102-CV 50ep/p10", 10, ["S1102"], s1102_50),
    ("S1102-CV 300ep/p30", 10, ["S1102"], s1102_300),
    ("Benchmarks 50ep/p10", 10, ["S1131", "S2003", "S4169"], bench_50),
    ("Honest ladder — S1102 (1100 muts)", 3, list(S1102_LADDER_DIR), _ladder(S1102_LADDER_DIR)),
    ("Honest ladder — full-SKEMPI", 3, list(FULLSK_LADDER_DIR), _ladder(FULLSK_LADDER_DIR)),
    ("CATH-superfamily hold-out (full-SKEMPI)", 1, ["CATH"], cath),  # USP-ddG's exact 813-mut split
]


def fmt(n, mean, std, exp):
    if n == 0:
        return "·"
    if n < exp:
        return f"{mean:.3f}[{n}/{exp}]"     # partial / in-progress
    return f"{mean:.3f}±{std:.3f}"           # complete


# ---- per-structure Spearman view (the Frontier headline metric) --------------------------
# Pre-scored CSVs (schema: arm,n,pearson,spearman,...,ps_spearman_T10,ps_ncomplex_T10,...). Node
# labels carry the DATASET explicitly so the S1102 (1100-mut) rungs are never confused with the
# full-SKEMPI rungs — a "S1102 clustered" per-structure number is NOT comparable to a
# "fullSK ... " one (different, much larger test set; check the n / ncomplex columns in the CSV).
# Arm key = "<tag>_<suffix>": base / foldx (=MLP) / foldx_scalar.
PS_LADDER = [
    # --- S1102_filtered (1100 muts) — early honest ladder ---
    ("S1102 leaky (per-mut CV)", os.path.join(ROOT, "experiments/rescore_perstructure/results.csv")),
    ("S1102 by-complex", os.path.join(ROOT, "experiments/retrain_split/results_bycomplex.csv")),
    ("S1102 clustered", os.path.join(ROOT, "experiments/retrain_split/results_clustered.csv")),
    # --- full SKEMPI — single-point (SP) honest splits (scored by
    #     experiments/full_skempi_seqonly/score_sp_all.py) ---
    ("fullSK clustered-SP", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_sp_clustered.csv")),
    ("fullSK bycomplex-SP", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_sp_bycomplex.csv")),
    # --- full SKEMPI — COMBINED single+multi by-complex (the frontier's own protocol; scored by
    #     `score_sp_all.py bycomplex_all`, whose truth is skempi_all_test.tsv) ---
    ("fullSK bycomplex-ALL", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_bycomplex_all.csv")),
    # --- full SKEMPI — COMBINED single+multi, homology-clustered (`score_sp_all.py clustered_all`).
    #     The clustered analog of bycomplex-ALL, and the tier the augmented arms sit on. ---
    ("fullSK clustered-ALL", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_clustered_all.csv")),
    # --- Tier-1 augmentation on the clustered-ALL FoldX arms. Same test set as the row above, so
    #     the two are directly differenced; `base` is absent here until the aug base split exists. ---
    ("fullSK clustered-ALL+aug", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_clustered_all_aug.csv")),
    # --- Tier-1 augmentation on the CATH FoldX arms (`score_cath.py --tier cath_aug`), single fold. ---
    ("fullSK CATH-all+aug", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_cath_aug.csv")),
    # --- full SKEMPI — multi-point (MP) honest splits ---
    ("fullSK MP-clustered", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_mp_clustered.csv")),
    ("fullSK MP-bycomplex", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_mp_bycomplex.csv")),
    # --- full SKEMPI — CATH-superfamily hold-out (USP-ddG's exact 813-mut split), single/multiple/all ---
    ("fullSK CATH-all", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_cath.csv")),
    ("fullSK CATH-single", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_cath_single.csv")),
    ("fullSK CATH-multiple", os.path.join(ROOT, "experiments/full_skempi_seqonly/results_cath_multiple.csv")),
]
# The fx_mlp arm suffix differs across the ladder CSVs: the leaky rescore.py CSV names it
# `<tag>_foldxmlp`, while the by-complex/clustered/MP CSVs use `<tag>_foldx`. Accept both so the
# leaky rung's MLP column stops rendering as `·`.
PS_ARM_SUFFIX = {"base": ("base",), "fx_scalar": ("foldx_scalar",), "fx_mlp": ("foldx", "foldxmlp")}


def _ps_row(d, tag, col):
    """First present row-dict across the accepted suffixes for this column (or None)."""
    for suf in PS_ARM_SUFFIX[col]:
        row = d.get(f"{tag}_{suf}")
        if row is not None:
            return row
    return None


def _ps_num(row, field):
    """float(row[field]) or None."""
    if row is None:
        return None
    try:
        return float(row[field]) if row.get(field, "") != "" else None
    except (ValueError, KeyError):
        return None


def _ps_get(d, tag, col):
    """First non-None ps_spearman_T10 across the accepted suffixes for this column."""
    return _ps_num(_ps_row(d, tag, col), "ps_spearman_T10")


def read_ps_csv(path):
    """{arm_key: {field: value}} from a scored per-structure CSV, or {} if absent. Keeps the whole
    row so the ps output can carry n (mutations scored) + ps_ncomplex_T10 (structures) as the
    dataset-size sanity check alongside ps_spearman_T10."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split(",")
        if "arm" not in header:
            return {}
        ai = header.index("arm")
        for ln in fh:
            p = ln.rstrip("\n").split(",")
            if len(p) > ai:
                out[p[ai]] = dict(zip(header, p))
    return out


def build_ps():
    """Print the per-structure Spearman (T>=10) honest ladder + write results_matrix_ps.csv."""
    cols = ["base", "fx_scalar", "fx_mlp"]
    # `spearman_foldavg` rides along so the fold-averaged panel is data-driven like the rest.
    # A rung CSV written before score_sp_all.py grew the column leaves it blank, which readers
    # must treat as "not scored", never as zero.
    csv_rows = ["split,model,arm,ps_spearman_T10,n_muts,n_complex_T10,delta_vs_base,"
                "spearman_foldavg,spearman_foldavg_sd"]
    blocks = []
    for node, path in PS_LADDER:
        d = read_ps_csv(path)
        head = f"per-structure Spearman T>=10  |  {node}   ({'found' if d else 'NOT SCORED YET'})"
        lines = [head, "-" * len(head),
                 f"{'model':<13}" + "".join(f"{c:>12}" for c in cols)
                 + f"{'Δscalar':>10}{'Δmlp':>8}{'n':>8}{'nCplx':>7}"]
        for tag, name in MODELS:
            rows = {c: _ps_row(d, tag, c) for c in cols}
            vals = {c: _ps_num(rows[c], "ps_spearman_T10") for c in cols}
            if all(v is None for v in vals.values()):
                continue
            base = vals["base"]
            cells = "".join(f"{(f'{vals[c]:.3f}' if vals[c] is not None else '·'):>12}" for c in cols)
            dsc = (vals["fx_scalar"] - base) if (vals["fx_scalar"] is not None and base is not None) else None
            dm = (vals["fx_mlp"] - base) if (vals["fx_mlp"] is not None and base is not None) else None
            # n (mutations) + nCplx (structures T>=10): dataset-size sanity check per row
            nrow = rows["base"] or rows["fx_scalar"] or rows["fx_mlp"]
            nmut, ncpx = _ps_num(nrow, "n"), _ps_num(nrow, "ps_ncomplex_T10")
            cells += f"{(f'{dsc:+.3f}' if dsc is not None else '·'):>10}{(f'{dm:+.3f}' if dm is not None else '·'):>8}"
            cells += f"{(f'{int(nmut)}' if nmut is not None else '·'):>8}{(f'{int(ncpx)}' if ncpx is not None else '·'):>7}"
            lines.append(f"{name:<13}{cells}")
            for c in cols:
                if vals[c] is not None:
                    dv = f"{vals[c]-base:+.4f}" if (base is not None and c != "base") else ""
                    nm, nc = _ps_num(rows[c], "n"), _ps_num(rows[c], "ps_ncomplex_T10")
                    nm = f"{int(nm)}" if nm is not None else ""
                    nc = f"{int(nc)}" if nc is not None else ""
                    fa = _ps_num(rows[c], "spearman_foldavg")
                    fasd = _ps_num(rows[c], "spearman_foldavg_sd")
                    fa = f"{fa:.4f}" if fa is not None else ""
                    fasd = f"{fasd:.4f}" if fasd is not None else ""
                    csv_rows.append(f"{node},{name},{c},{vals[c]:.4f},{nm},{nc},{dv},{fa},{fasd}")
        blocks.append("\n".join(lines))
    out_csv = os.path.join(HERE, "results_matrix_ps.csv")
    with open(out_csv, "w") as fh:
        fh.write("\n".join(csv_rows) + "\n")
    print("\n\n".join(blocks))
    print("\nΔscalar/Δmlp = FoldX per-structure lift over base. Story: lift should hold/widen "
          "as the split\ngets stricter (leaky -> by-complex -> clustered); scalar vs MLP is the "
          "key contrast.")
    print(f"\n[wrote] {out_csv}  ({len(csv_rows) - 1} rows)")


def main():
    args = [a for a in sys.argv[1:]]
    if "--ps" in args:
        build_ps()
        return
    csv_only = "--csv-only" in args
    filt = next((a.lower() for a in args if not a.startswith("-")), None)

    res = "--res" in args
    csv_rows = ["regime,dataset,model,arm,n_folds,expected,mean_ntest,total_ntest,"
                "mean_pcc,std_pcc,mean_runtime_s,platform,status"]
    blocks = []
    tot = {"done": 0, "partial": 0, "empty": 0}

    for rname, exp, datasets, rule in REGIMES:
        if filt and filt not in rname.lower():
            continue
        for ds in datasets:
            metric = "mean min/fold + platform" if res else "test_pcc pooled"
            header = f"{rname}  |  {ds}   ({metric}; expected {exp} folds)"
            lines = [header, "-" * len(header)]
            colw = 15
            lines.append(f"{'model':<13}" + "".join(f"{a:>{colw}}" for a in ARMS) +
                         (f"{'plat*':>8}" if res else ""))
            any_data = False
            for tag, name in MODELS:
                paths = rule(tag, ds)
                cells, plat = [], ""
                for arm in ARMS:
                    n, mean, std, mrt, ntest, pf, chosen = cell(paths.get(arm, []), tag)
                    if res:
                        cells.append(f"{mrt/60:.1f}[{n}/{exp}]" if (mrt and n < exp)
                                     else (f"{mrt/60:.1f}" if mrt else "·"))
                    else:
                        cells.append(fmt(n, mean, std, exp))
                    if pf:
                        plat = pf
                    status = "done" if n >= exp and n > 0 else ("partial" if n > 0 else "empty")
                    if n > 0:
                        any_data = True
                    if paths.get(arm):  # only emit CSV rows for cells that CAN exist in this regime
                        tot[status] += 1
                        mp = f"{mean:.6f}" if mean is not None else ""
                        sp = f"{std:.6f}" if std is not None else ""
                        rt = f"{mrt:.1f}" if mrt is not None else ""
                        mnt = f"{ntest[0]}" if ntest[0] is not None else ""
                        tnt = f"{ntest[1]}" if ntest[1] is not None else ""
                        csv_rows.append(f"{rname},{ds},{name},{arm},{n},{exp},{mnt},{tnt},{mp},{sp},"
                                        f"{rt},{pf},{status}")
                row = f"{name:<13}" + "".join(f"{c:>{colw}}" for c in cells)
                lines.append(row + (f"{plat:>8}" if res else ""))
            if any_data or not filt:
                blocks.append("\n".join(lines))

    # Only (re)write the full CSV on an UNFILTERED run — a positional filter yields a partial
    # matrix, and silently clobbering results_matrix.csv with a subset would defeat the whole
    # point of the file as a coverage sanity check. Filtered runs are print-only.
    out_csv = os.path.join(HERE, "results_matrix.csv")
    wrote_csv = filt is None
    if wrote_csv:
        # Refuse to overwrite the committed CSV with an all-empty one. `scratch/` is gitignored, so
        # in any clone -- including the publication tree -- every candidate path resolves to nothing
        # and this would rewrite 528 rows to status=empty and exit 0, destroying the only record of
        # what was run. The header invites exactly that ("re-run any time"). Same posture as
        # foldx_gap_analysis.py, which exits rather than emit a matrix it has no data for.
        if not any(int(r.split(",")[4]) for r in csv_rows[1:]):  # [1:] skips the header row
            sys.exit(f"refusing to write {out_csv}: no folds found under {R} -- scratch/ is "
                     "gitignored and absent in a fresh clone. The committed CSV is the record of "
                     "runs that happened; regenerate it only on a tree that holds them.")
        with open(out_csv, "w") as fh:
            fh.write("\n".join(csv_rows) + "\n")

    if not csv_only:
        print("\n\n".join(blocks))
        if res:
            print("\nCells = mean train minutes/fold  ([n/N] if partial).  plat* = platform, "
                  "INFERRED by provenance\n(esmc6b/gpu_results = CUDA GPU box; else Mac MPS) — "
                  "device is not recorded in all_results.json.")
            print("Max GPU/host memory is NOT captured (HF skip_memory_metrics defaults True); to "
                  "record it\non future runs pass --skip_memory_metrics False to mulan-train.")
        else:
            print("\nLegend:  m±s = complete (mean±pstdev)   m[n/N] = partial (in progress)   · = none")
            print("Add --res for runtime (min/fold) + inferred platform;  --ps for per-structure Spearman.")
        print(f"Cells (excl. N/A arms):  done={tot['done']}  partial={tot['partial']}  empty={tot['empty']}")
    if wrote_csv:
        print(f"\n[wrote] {out_csv}  ({len(csv_rows) - 1} rows)")
    else:
        print(f"\n[skip] filtered run ({filt!r}) — CSV not rewritten (would be partial). "
              f"Run with no filter to refresh {os.path.basename(out_csv)}.")


if __name__ == "__main__":
    main()
