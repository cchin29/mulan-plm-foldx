#!/usr/bin/env python3
"""Compute the MuLAN data blocks for the SKEMPI benchmark matrix, and write them
into `experiments/benchmark_matrix.html` and `experiments/BENCHMARK_MATRIX.md`.

WHAT IT EMITS
-------------
For each embedding x arm it computes, in one place, every metric the matrix shows:
  * full-SKEMPI  : byCplx per-structure Spearman (T>=10),
                   CD-HIT<=60% clustered **pooled** Pearson & Spearman,
                   CD-HIT<=60% clustered **fold-averaged** Spearman,
                   CATH-superfamily per-PPI Spearman.
  * S1102 ladder : leaky CV10 pooled Pearson + per-structure Spearman,
                   by-complex per-structure Spearman,
                   CD-HIT<=60% clustered **pooled** Pearson & Spearman.
  * AUROC        : sign-of-effect, per embedding x arm — CATH from results_cath.csv,
                   by-complex all-muts pooled from the SP + MP prediction runs.
It also prints the collapsed "best" values and the ProtBFF (full-SKEMPI, CD-HIT<=60%,
pooled) comparison.

WHY IT NOW WRITES INSTEAD OF PRINTING FOR A HUMAN TO PASTE
----------------------------------------------------------
Until 2026-08-06 this script printed "ready-to-paste JS array literals" and a person
carried them into the HTML by hand and mirrored them into the Markdown by hand. One
computation, three copies, two manual hops. When `--verify` was first run against the
files as they stood, it found 38 stale published cells — all of them in the Markdown,
none in the HTML:

  * `fullsk` and `ladder` were clean in both files. The paste discipline had held for
    the two blocks that get re-pasted whenever a fold lands.
  * `aurM` had not. After the FoldX coverage fix (87.8% -> 99.2%) the AUROC block was
    re-pasted into the HTML and never mirrored into the Markdown: 36 cells in the
    by-embedding x arm table (every FoldX-arm CATH and by-complex value, plus three
    ESM-C 600M by-complex cells still published as em-dashes), the <details> caption's
    best-by-complex figure (0.723 -> 0.760), and the headline `MuLAN + FoldX (ESM-C 6B)`
    by-complex cell (0.710 -> 0.732).

That second bullet is the reason the paste step had to go rather than merely be done
more carefully. The drift was invisible from inside either file — both were internally
consistent — and scripts_plots/plot_ppS_scaling.py reads the AUROC numbers back out of
the Markdown through comparators.py, so panel 4b had been drawing the stale values for
as long as they sat there. A block that is *usually* re-pasted is a block whose staleness
nobody looks for.

A number that exists in three places has two chances to be updated in only one. So
the paste step is gone: `--write` patches both files in place, and `--verify` fails
loudly if either has drifted from what the predictions actually say.

METRIC NOTES (why these and not others)
  * per-structure Spearman (T>=10) = RDE `per_complex_corr`; the by-complex / CATH
    frontier comparators report this.
  * pooled Pearson/Spearman on the CD-HIT<=60% clustered split = the ONLY
    metric+split that is directly comparable to ProtBFF (which reports pooled only).
    MuLAN clusters by **sequence identity** (mmseqs/CD-HIT <=60%), NOT iDist<=0.03
    interface dedup (PPIformer/PPIRef) — different criterion, kept separate.
  * pooled = correlation over the union of all fold predictions (each mutation
    scored once, in its held-out fold), matching how ProtBFF pools its CV.

USAGE
-----
    python3 experiments/gen_benchmark_matrix_data.py            # print all blocks + summary
    python3 experiments/gen_benchmark_matrix_data.py --check    # just the ProtBFF comparison line
    python3 experiments/gen_benchmark_matrix_data.py --verify   # exit 1 if HTML or MD has drifted
    python3 experiments/gen_benchmark_matrix_data.py --write    # patch HTML + MD in place

`--write` owns the table cells, the <details> caption and the headline AUROC row. It does
NOT own the § footnote sentence under the AUROC companion table — that paragraph is checked
(its three numbers must appear in it) and reported for a hand edit, because re-wrapping human
prose is a worse trade than retyping three digits. `--write` exits 1 while such a finding
stands, so a stale sentence cannot be mistaken for a clean run.

Run from anywhere; it chdirs to the repo root. Needs numpy only. After new folds land:
`--write`, then re-run the figure scripts that read the matrix (plot_ppS_scaling.py,
plot_ppS_generalization.py, plot_fishbone.py — all of which go through comparators.py).
`--verify` is the thing to run in a pre-commit hook or before building the deck.

INPUTS (paths relative to repo root)
  predictions : scratch/results/full_skempi{,_bycomplex,_mp_bycomplex}/<arm>/fold_{0,1,2}/training_run/test_predictions.tsv
  truth       : scratch/splits_skempi_full_clustered_id60_kfold/fold_{f}/skempi_sp_test.tsv
                scratch/splits_skempi_full_bycomplex_seed42/fold_{f}/skempi_sp_test.tsv
                scratch/splits_skempi_full_mp_bycomplex_seed42/fold_{f}/skempi_mp_test.tsv
                (only the clustered directory carries the `_kfold` suffix — do not glob for it)
  S1102       : scripts_plots/results_matrix_ps.csv (leaky/by-complex psS, clustered fold-avg S),
                experiments/retrain_split/results_clustered.csv (S1102 clustered pooled P/S),
                experiments/rescore_perstructure/results.csv (leaky CV10 pooled Pearson)
  CATH        : experiments/full_skempi_seqonly/results_cath.csv
ProtBFF anchor (full SKEMPI, CD-HIT 60%, pooled): Pearson 0.514 / Spearman 0.477 (bioRxiv 2025-12-23).
"""
import numpy as np, os, csv, sys, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts_plots"))
import mdtable   # the table locate/parse rule, shared with comparators.py

FOLDS = [0, 1, 2]
PROTBFF = (0.514, 0.477)   # (pooled Pearson, pooled Spearman), full SKEMPI, CD-HIT 60%
HTML = "experiments/benchmark_matrix.html"
MD = "experiments/BENCHMARK_MATRIX.md"

# ---- display order + arm-name maps across the various result files -----------
DISP = [("ESM-C 6B", "ESM-C 6B", "esmc6b"), ("ESM-C 600M", "ESM-C 600M", "esmc600m"),
        ("SaProt", "SaProt", "saprot"), ("ProstT5", "ProstT5", "prostt5"),
        ("ESM2-3B", "ESM2-3B", "esm2"), ("Ankh-large", "Ankh-large", "ankh"),
        ("Ankh3-large", "Ankh3-large", "ankh3_large"), ("Ankh3-xl", "Ankh3-xl", "ankh3_xl"),
        ("AIDO-16B", "AIDO-16B", "aido")]
ARMS = [("base", "base", "base", "base"),                  # (label, ps-matrix arm, pred/clustered arm, rescore arm)
        ("+ FoldX scalar", "fx_scalar", "foldx_scalar", "foldx_scalar"),
        ("+ FoldX MLP", "fx_mlp", "foldx", "foldxmlp")]
HEAD = "esmc6b_foldx_scalar"  # headline MuLAN arm


def _avg_ranks(a):
    """Ranks with ties averaged — Spearman's actual definition, and rescore.average_ranks'.

    `argsort(argsort(x))` was used here until 2026-08-05. That assigns ORDINAL ranks, breaking ties
    by array position, and experimental ΔΔG is full of ties (repeated round numbers), so tied rows
    got an arbitrary order that correlated with nothing. It biased every Spearman in this file low
    by ~0.003 against the rung CSVs — small, uniform, and invisible without a cross-check.
    """
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spear(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(_avg_ranks(x), _avg_ranks(y))[0, 1])

def auroc(scores, labels):
    # AUROC (Mann-Whitney) for sign-of-effect: label = ddG>0 (destabilizing). Matches MuLAN `auroc_destab`
    # and the frontier's "AUROC" (RDE/USP definition: mutations classified by sign of ddG).
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return None
    ranks = np.empty(len(s)); ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

# Frontier AUROC (sign-of-effect), transcribed from primary tables — for the companion AUROC table.
FRONTIER_AUROC = {  # method: (CATH all-muts, by-complex all-muts)
    "USP-ddG": (0.802, None), "CATH-ddG": (0.781, 0.776), "flex-ddG": (0.764, None),
    "FoldX": (0.754, 0.658), "RDE-Network": (0.745, 0.745), "BA-DDG": (0.732, 0.773),
    "DiffAffinity": (0.625, 0.744), "Prompt-DDG": (None, 0.757), "ProMIM": (None, 0.760)}
# sources: CATH col = USP-ddG Table 1; by-complex col = BA-DDG Table 1 / CATH-ddG Suppl. S2.

# The working tree keeps inputs under scratch/ with a training_run/ level inside each fold; the
# published tree drops scratch/, moves splits to data/splits/, and flattens training_run/ away.
# Resolve rather than believe: pointed at the layout it does not expect, this script computed null
# for every cell and reported 257 spurious findings against a tree that was correct.

def _first(*cands):
    for c in cands:
        if c and os.path.exists(os.path.join(ROOT, c)):
            return os.path.join(ROOT, c)
    return None


def _resolve_base(base):
    alt = None
    if base.startswith("scratch/results/"):
        alt = base.replace("scratch/results/", "results/", 1)
    elif base.startswith("scratch/splits_"):
        alt = base.replace("scratch/", "data/splits/", 1)
    return _first(base, alt)


def load_truth(base, fn="skempi_sp_test.tsv"):
    d = {}
    root = _resolve_base(base)
    if root is None: return d
    for f in FOLDS:
        p = f"{root}/fold_{f}/{fn}"
        if not os.path.exists(p): continue
        for ln in open(p):
            c = ln.rstrip("\n").split("\t")
            if len(c) >= 4:
                try: d[(c[0], c[1], c[2])] = float(c[3])
                except ValueError: pass
    return d

def load_pred(resdir, arm):
    pred = {}
    root = _resolve_base(resdir)
    if root is None: return None
    for f in FOLDS:
        p = _first(f"{root}/{arm}/fold_{f}/training_run/test_predictions.tsv",
                   f"{root}/{arm}/fold_{f}/test_predictions.tsv")
        if p is None: return None
        for ln in open(p):
            c = ln.rstrip("\n").split("\t")
            if len(c) >= 4:
                try: pred[(c[0], c[1], c[2])] = float(c[3])
                except ValueError: pass
    return pred

def pooled(resdir, truth, arm):
    pr = load_pred(resdir, arm)
    if pr is None: return None, None
    xy = [(v, truth[k]) for k, v in pr.items() if k in truth]
    if len(xy) < 3: return None, None
    x = [a for a, _ in xy]; y = [b for _, b in xy]
    return float(np.corrcoef(x, y)[0, 1]), spear(x, y)

def perstruct(resdir, truth, arm, T=10):
    pr = load_pred(resdir, arm)
    if pr is None: return None
    g = {}
    for k, v in pr.items():
        # Group by PDB — `chain1_id.split("_")[0]`, the RDE `per_complex_corr` key that every
        # frontier comparator in this matrix is measured under, and the same key rescore.py uses.
        # This grouped by the full chain pair (k[0], k[1]) until 2026-08-05, which splits one PDB
        # into a group per chain-pair annotation and then drops the pieces that fall under T=10.
        # It biased every byCplx/CATH cell here low by ~0.005 against the rung CSVs.
        if k in truth: g.setdefault(k[0].split("_")[0], []).append((v, truth[k]))
    vs = [spear([a for a, _ in l], [b for _, b in l]) for l in g.values() if len(l) >= T]
    vs = [v for v in vs if not np.isnan(v)]
    return float(np.mean(vs)) if vs else None

def _none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)

def fmt(x): return "null" if _none(x) is None else f"{_none(x):.3f}"


# =============================================================================== compute
#
# Everything numeric happens here, once. Nothing below prints, and nothing above
# writes — so the three renderings (stdout, HTML, Markdown) are three views of one
# result rather than three transcriptions of it.


class Block:
    """One computed data block and the three places it is published.

    `cols` is the canonical column set. `html_cols` and `md_cols` are orderings of
    it, because the HTML array and the Markdown table happen to lay the same numbers
    out differently — which column goes where is presentation, and belongs next to
    the thing being presented, not in the computation.
    """

    def __init__(self, key, rows, cols, html_cols, md_cols, html_array, md_header):
        self.key, self.rows, self.cols = key, rows, cols
        self.html_cols, self.md_cols = html_cols, md_cols
        self.html_array, self.md_header = html_array, md_header

    def best(self, col):
        vals = [r[2][col] for r in self.rows if r[2].get(col) is not None]
        return max(vals) if vals else None


def compute():
    """Every number the matrix publishes about MuLAN. Returns {block key: Block} + extras."""
    truth_clust = load_truth("scratch/splits_skempi_full_clustered_id60_kfold")
    truth_bycpx = load_truth("scratch/splits_skempi_full_bycomplex_seed42")
    truth_mp = load_truth("scratch/splits_skempi_full_mp_bycomplex_seed42", "skempi_mp_test.tsv")
    cath = {r["arm"]: r for r in csv.DictReader(open("experiments/full_skempi_seqonly/results_cath.csv"))}
    cath_au = {r["arm"]: float(r["auroc_destab"]) for r in
               csv.DictReader(open("experiments/full_skempi_seqonly/results_cath.csv")) if r.get("auroc_destab")}
    s1102c = {r["arm"]: r for r in csv.DictReader(open("experiments/retrain_split/results_clustered.csv"))}
    resc = {r["arm"]: r for r in csv.DictReader(open("experiments/rescore_perstructure/results.csv"))}
    psrows = list(csv.DictReader(open("scripts_plots/results_matrix_ps.csv")))

    def psv(split, model, arm, col="ps_spearman_T10"):
        for r in psrows:
            if r["split"] == split and r["model"] == model and r["arm"] == arm and r.get(col):
                return float(r[col])
        return None

    # ---- full-SKEMPI -------------------------------------------------------
    fullsk, best = [], {"fs_cd_s": (-9, ""), "fs_cd_p": (-9, "")}
    for dn, mdl, key in DISP:
        for lab, psa, prda, _ in ARMS:
            arm = f"{key}_{prda}"
            bp = perstruct("scratch/results/full_skempi_bycomplex", truth_bycpx, arm)
            cp, cs = pooled("scratch/results/full_skempi", truth_clust, arm)
            ca = cath.get(arm, {}).get("ps_spearman_T10")
            ca = float(ca) if ca else None
            if cs is not None and cs > best["fs_cd_s"][0]: best["fs_cd_s"] = (cs, f"{dn} {lab}")
            if cp is not None and cp > best["fs_cd_p"][0]: best["fs_cd_p"] = (cp, f"{dn} {lab}")
            fullsk.append((dn, lab, {
                "byCplx": _none(bp), "cath": _none(ca),
                "cdhitS_foldavg": psv("fullSK clustered-SP", mdl, psa, "spearman_foldavg"),
                "cdhitS_pooled": _none(cs), "cdhitP_pooled": _none(cp)}))

    # ---- S1102 ladder ------------------------------------------------------
    RESC = {"base": "base", "foldx_scalar": "foldx_scalar", "foldx": "foldxmlp"}
    ladder = []
    for dn, mdl, key in DISP:
        for lab, psa, prda, _ in ARMS:
            ck, rk = f"{key}_{prda}", f"{key}_{RESC[prda]}"
            row = {"cv10P": float(resc[rk]["pearson"]) if rk in resc else None,
                   "cv10psS": psv("S1102 leaky (per-mut CV)", mdl, psa),
                   "byCplx": psv("S1102 by-complex", mdl, psa),
                   "cdhitS_pooled": float(s1102c[ck]["spearman"]) if ck in s1102c else None,
                   "cdhitP_pooled": float(s1102c[ck]["pearson"]) if ck in s1102c else None}
            if all(row[c] is None for c in ("cv10P", "cv10psS", "byCplx", "cdhitS_pooled")):
                continue
            ladder.append((dn, lab, row))
    if "mint_base" in s1102c:
        ladder.append(("MINT (multi-chain)", "base", {
            "cv10P": float(resc["mint_complex"]["pearson"]),
            "cv10psS": psv("S1102 leaky (per-mut CV)", "MINT", "base"),
            "byCplx": psv("S1102 by-complex", "MINT", "base"),
            "cdhitS_pooled": float(s1102c["mint_base"]["spearman"]),
            "cdhitP_pooled": float(s1102c["mint_base"]["pearson"])}))

    # ---- AUROC (sign-of-effect) -------------------------------------------
    # by-complex AUROC — pool the separate single-point and multi-point runs so the
    # composition matches the frontier's all-muts column (two MuLAN models vs their one).
    aurM, bc_all, bc_best = [], {}, (-9, "")
    bc_parts = {}   # arm -> (single-point AUROC, multi-point AUROC); the § footnote cites these
    for dn, mdl, key in DISP:
        for lab, psa, prda, _ in ARMS:
            arm = f"{key}_{prda}"
            psp = load_pred("scratch/results/full_skempi_bycomplex", arm)
            pmp = load_pred("scratch/results/full_skempi_mp_bycomplex", arm)
            sp_xy = [(v, truth_bycpx[k]) for k, v in psp.items() if k in truth_bycpx] if psp else []
            mp_xy = [(v, truth_mp[k]) for k, v in pmp.items() if k in truth_mp] if pmp else []
            xy = sp_xy + mp_xy
            bc_parts[arm] = tuple(auroc([v for v, _ in p], [t > 0 for _, t in p]) if p else None
                                  for p in (sp_xy, mp_xy))
            au = auroc([v for v, _ in xy], [t > 0 for _, t in xy]) if xy else None
            if au is not None:
                bc_all[arm] = au
                if au > bc_best[0]: bc_best = (au, f"{dn} {lab}")
            ca = cath_au.get(arm)
            if ca is None and au is None: continue
            aurM.append((dn, lab, {"cath": _none(ca), "byCplx_all": _none(au)}))

    blocks = {
        "fullsk": Block(
            "fullsk", fullsk,
            ["byCplx", "cath", "cdhitS_foldavg", "cdhitS_pooled", "cdhitP_pooled"],
            html_cols=["byCplx", "cdhitS_pooled", "cdhitP_pooled", "cath"],
            md_cols=["byCplx", "cath", "cdhitS_foldavg", "cdhitS_pooled", "cdhitP_pooled"],
            html_array="fullsk",
            md_header=["| Embedding | arm | byCplx (ps-Sp)"]),
        "ladder": Block(
            "ladder", ladder,
            ["cv10P", "cv10psS", "byCplx", "cdhitS_pooled", "cdhitP_pooled"],
            html_cols=["cv10P", "cv10psS", "byCplx", "cdhitS_pooled", "cdhitP_pooled"],
            md_cols=["cv10P", "cv10psS", "byCplx", "cdhitS_pooled", "cdhitP_pooled"],
            html_array="ladder",
            md_header=["| Embedding | arm | CV10 (P)"]),
        "aurM": Block(
            "aurM", aurM,
            ["cath", "byCplx_all"],
            html_cols=["cath", "byCplx_all"],
            md_cols=["cath", "byCplx_all"],
            html_array="aurM",
            md_header=["| Embedding | arm | CATH | by-complex (all-muts)"]),
    }
    return dict(blocks=blocks, best=best, cath_au=cath_au, bc_all=bc_all, bc_best=bc_best,
                bc_parts=bc_parts)


# =============================================================================== stdout


def render_stdout(D, check=False):
    """The original human-readable dump. Unchanged, so a diff against an old run is meaningful."""
    out = []
    B = D["blocks"]
    if not check:
        out.append("// ===== FULL-SKEMPI expanded: [emb, arm, byCplx_psS, cdhit_pooledS, cdhit_pooledP, cath_psS] =====")
        for dn, lab, r in B["fullsk"].rows:
            vals = ",".join(fmt(r[c]) for c in ["byCplx", "cdhitS_pooled", "cdhitP_pooled", "cath"])
            out.append(f'  ["{dn if lab=="base" else ""}","{lab}",{vals}],')

        out.append("\n// ===== S1102 ladder expanded: [emb, arm, cv10_pooledP, cv10_psS, byCplx_psS, cdhit_pooledS, cdhit_pooledP] =====")
        for dn, lab, r in B["ladder"].rows:
            vals = ",".join(fmt(r[c]) for c in B["ladder"].html_cols)
            out.append(f'  ["{dn if lab=="base" else ""}","{lab}",{vals}],')

        cath_au, bc_all, bc_best = D["cath_au"], D["bc_all"], D["bc_best"]
        out.append("\n// ===== AUROC (sign-of-effect) companion — MuLAN vs frontier =====")
        out.append(f"//   MuLAN+FoldX (ESM-C 6B) : CATH all-muts {cath_au.get(HEAD,float('nan')):.3f}  |  "
                   f"by-complex ALL-muts {bc_all.get(HEAD,float('nan')):.3f} (SP+MP pooled)")
        out.append(f"//   MuLAN best CATH AUROC = {max(cath_au.values()):.3f} ; best by-complex all-muts AUROC = {bc_best[0]:.3f} ({bc_best[1]})")
        out.append("//   Frontier (CATH all-muts / by-complex all-muts):")
        for m, (c, b) in FRONTIER_AUROC.items():
            out.append(f"//     {m:13} CATH {c if c is not None else '—'} | byCplx {b if b is not None else '—'}")
        out.append("//   NB: MuLAN by-complex is single-point; frontier by-complex is all-muts (multi-point lifts AUROC) — cite CATH.")
        out.append("//   --- per-arm AUROC (paste into aurM in benchmark_matrix.html): [emb, arm, cath, bycplx_all] ---")
        for dn, lab, r in B["aurM"].rows:
            fa = lambda x: "null" if x is None else f"{x:.3f}"
            out.append(f'//     ["{dn if lab=="base" else ""}","{lab}",{fa(r["cath"])},{fa(r["byCplx_all"])}],')

    s, sa = D["best"]["fs_cd_s"]; p, pa = D["best"]["fs_cd_p"]
    out.append("\n// ===== full-SKEMPI CD-HIT<=60% pooled — MuLAN vs ProtBFF (the ONLY dataset+criterion+metric match) =====")
    out.append(f"//   MuLAN best pooled Spearman = {s:.3f}  ({sa})")
    out.append(f"//   MuLAN best pooled Pearson  = {p:.3f}  ({pa})")
    # Verdict derived from the numbers, not asserted. It read "MuLAN is BELOW ProtBFF" as a fixed
    # string until 2026-08-05, and kept printing that under a Spearman of 0.509 against ProtBFF's
    # 0.477 — the sentence outlived the 87.8% -> 99.2% FoldX coverage fix that overturned it.
    def _verdict(mulan, ref, name):
        rel = "ABOVE" if mulan > ref else ("BELOW" if mulan < ref else "LEVEL WITH")
        return f"{name} {mulan:.3f} vs {ref:.3f} -> MuLAN {rel}"
    out.append(f"//   ProtBFF (full SKEMPI, CD-HIT 60%) pooled Pearson {PROTBFF[0]} / Spearman {PROTBFF[1]}")
    if s < -1 or p < -1:
        # `best` is still its -9 sentinel: no arm scored, because this tree has no scratch/results.
        # Say so rather than comparing -9 to 0.477 and printing a confident, false "MuLAN BELOW".
        out.append("//   no MuLAN arm scored — scratch/results/ is absent, so there is nothing to compare.")
        out.append("//   Run this in a tree that carries the prediction dirs; the published numbers are")
        out.append("//   already committed in BENCHMARK_MATRIX.md and scripts_plots/results_matrix_ps.csv.")
        return "\n".join(out)
    out.append(f"//   {_verdict(p, PROTBFF[0], 'Pearson ')}")
    out.append(f"//   {_verdict(s, PROTBFF[1], 'Spearman')}")
    out.append("//   NB pooled is one of two readings of ProtBFF's column; the fold-averaged form is in")
    out.append("//   results_matrix_ps.csv (spearman_foldavg). Both now clear 0.477 — see BENCHMARK_MATRIX.md.")
    return "\n".join(out)


# =============================================================================== HTML


def html_lines(block):
    """The JS array body, three arms to a line, embedding named only on its first arm —
    matching how the file was already laid out by hand."""
    lines, buf, cur = [], [], None
    for dn, lab, r in block.rows:
        if dn != cur and buf:
            lines.append("  " + "".join(buf))
            buf = []
        vals = ",".join(fmt(r[c]) for c in block.html_cols)
        buf.append(f'["{dn if dn != cur else ""}","{lab}",{vals}],')
        cur = dn
    if buf:
        lines.append("  " + "".join(buf))
    return lines


def html_span(text, array):
    """(start, end) character offsets of `array`'s row body in the HTML.

    fullsk/ladder are `const X={...  sub:[ <rows> ]};`; aurM is `const aurM=[ <rows> ];`.
    Anchored on the declaration, not on a line number.
    """
    if array == "aurM":
        m = re.search(r"(const aurM=\[\n)(.*?)(\];)", text, re.S)
    else:
        m = re.search(r"(const %s=\{.*?\n sub:\[\n)(.*?)(\n \]\};)" % array, text, re.S)
    if not m:
        sys.exit(f"FATAL: could not locate the {array!r} array in {HTML}. "
                 f"Its declaration has changed shape; this script cannot patch it safely.")
    return m.start(2), m.end(2)


def html_read(text, array):
    """Parse an array body back into [(emb, arm, [str values])] with embeddings filled forward."""
    lo, hi = html_span(text, array)
    rows, cur = [], None
    for m in re.finditer(r'\["([^"]*)","([^"]*)",([^\]]*)\]', text[lo:hi]):
        emb = m.group(1) or cur
        cur = emb
        rows.append((emb, m.group(2), [v.strip() for v in m.group(3).split(",")]))
    return rows


def html_write(text, block):
    lo, hi = html_span(text, block.html_array)
    body = "\n".join(html_lines(block))
    if block.html_array == "aurM":
        body = body.rstrip(",")          # aurM closes on its last element: `...]];`
    return text[:lo] + body + text[hi:]


# =============================================================================== Markdown


def md_lines(block, keycol=("Embedding", "arm")):
    """Markdown body rows. Best-in-column is bolded — the matrix's own convention, and
    derived here rather than left where a human has to remember to move the asterisks."""
    bests = {c: block.best(c) for c in block.md_cols}
    out = []
    for dn, lab, r in block.rows:
        cells = [mdtable.fmt(r[c], bold=(r[c] is not None and r[c] == bests[c]))
                 for c in block.md_cols]
        out.append("| " + " | ".join([dn, lab] + cells) + " |")
    return out


def md_write(lines, block):
    _, lo, hi = mdtable.locate(MD, lines, block.key, block.md_header)
    return lines[:lo] + md_lines(block) + lines[hi:]


# =============================================================================== verify


def _cmp(published, computed):
    """Agreement is agreement *as published* — same three-decimal spelling, or both absent.

    Comparing the raw float against the printed cell with a tolerance looks more
    careful but is wrong at the boundary: a computed 0.2295 prints as 0.229 and sits
    exactly 5e-4 away from it, so a tolerance test flags a cell that is in fact
    correctly published. The matrix publishes three decimals; three decimals is the
    thing to check.
    """
    def spell(x):
        if isinstance(x, str):
            x = x.replace("*", "").strip()
            if x in ("null", mdtable.EMDASH, ""):
                return None
            x = float(x)
        x = _none(x)
        return None if x is None else f"{x:.3f}"
    return spell(published) == spell(computed)


def verify(D):
    """Compare every published cell against what the predictions actually say.

    Returns (machine-fixable findings, prose findings). The split matters: `--write`
    can close the first list on its own, and must hand the second back to a human.
    """
    bad = []
    text = open(HTML).read()
    lines = mdtable.read(MD)

    for block in D["blocks"].values():
        want = {(dn, lab): r for dn, lab, r in block.rows}

        # ---- HTML
        got = html_read(text, block.html_array)
        if len(got) != len(block.rows):
            bad.append(f"{HTML}  {block.html_array}: {len(got)} rows published, {len(block.rows)} computed")
        for emb, arm, vals in got:
            w = want.get((emb, arm))
            if w is None:
                bad.append(f"{HTML}  {block.html_array}: row ({emb}, {arm}) is published but not computed")
                continue
            for col, v in zip(block.html_cols, vals):
                if not _cmp(v, w[col]):
                    bad.append(f"{HTML}  {block.html_array} [{emb} · {arm}] {col}: "
                               f"published {v}, computed {fmt(w[col])}")

        # ---- Markdown
        t = mdtable.parse(MD, lines, block.key, block.md_header, keycols=2)
        for (emb, arm), r in t.rows.items():
            w = want.get((emb, arm))
            if w is None:
                bad.append(f"{MD}:{t.linenos[(emb, arm)]}  row ({emb}, {arm}) is published but not computed")
                continue
            for col, mdcol in zip(block.md_cols, t.cols):
                if not _cmp(r[mdcol], w[col]):
                    bad.append(f"{MD}:{t.linenos[(emb, arm)]}  [{emb} · {arm}] {mdcol}: "
                               f"published {mdtable.fmt(r[mdcol])}, computed {fmt(w[col])}")
        missing = set(want) - set(t.rows)
        for k in sorted(missing):
            bad.append(f"{MD}  {block.key}: computed row {k} is not published")

    # ---- derived captions and headline cells that live in prose
    for msg in _verify_derived(D, lines):
        bad.append(msg)
    return bad, _verify_prose(D, lines)


def _derived(D):
    """The numbers that appear in the Markdown's prose/captions but are computed here."""
    cath_au, bc_all, bc_best = D["cath_au"], D["bc_all"], D["bc_best"]
    aur = D["blocks"]["aurM"]
    bestC = aur.best("cath")
    bestC_who = next(f"{dn} {lab}" for dn, lab, r in aur.rows if r["cath"] == bestC)
    sp, mp = D["bc_parts"].get(HEAD, (None, None))
    return {
        "head_cath": cath_au.get(HEAD),
        "head_bycplx": bc_all.get(HEAD),
        "head_bycplx_sp": sp, "head_bycplx_mp": mp,
        "best_cath": bestC, "best_cath_who": bestC_who,
        "best_bycplx": bc_best[0], "best_bycplx_who": bc_best[1],
    }


AUR_CAPTION = ("<details><summary><b>MuLAN AUROC by embedding × arm</b> "
               "(best CATH <b>{best_cath:.3f}</b> = {best_cath_who}; "
               "best by-complex all-muts <b>{best_bycplx:.3f}</b> = {best_bycplx_who})</summary>")


def _verify_derived(D, lines):
    d = _derived(D)
    out = []
    want = AUR_CAPTION.format(**d)
    hits = [i for i, l in enumerate(lines) if l.startswith("<details><summary><b>MuLAN AUROC by embedding")]
    if len(hits) != 1:
        out.append(f"{MD}  expected exactly 1 'MuLAN AUROC by embedding' <details> caption, found {len(hits)}")
    elif lines[hits[0]] != want:
        out.append(f"{MD}:{hits[0]+1}  AUROC <details> caption is stale\n"
                   f"      published: {lines[hits[0]]}\n      computed : {want}")

    # The headline MuLAN row in the AUROC companion table.
    t = mdtable.parse(MD, lines, "AUROC companion", ["| Method ", "CATH (all-muts)", "by-complex"])
    lbl = t.row("MuLAN + FoldX")
    for col, key in [("CATH (all-muts)", "head_cath"), ("by-complex", "head_bycplx")]:
        if not _cmp(t.rows[lbl][col], d[key]):
            out.append(f"{MD}:{t.linenos[lbl]}  [{lbl}] {col}: published "
                       f"{mdtable.fmt(t.rows[lbl][col])}, computed {fmt(d[key])}")
    return out


# The § footnote under the AUROC companion table is a hand-written sentence that happens
# to carry three computed numbers. It is deliberately NOT machine-rewritten: re-wrapping a
# human paragraph is a worse trade than asking a human to retype three digits. But it is
# machine-*checked*, because "prose we forgot to update" is exactly how the AUROC block
# went stale in the first place — the numbers there had no owner. `--write` fixes what it
# can and then names these so the hand edit is a five-second job, not an archaeology one.
PROSE_CHECKS = [
    ("§ footnote under the AUROC companion table", "§ MuLAN by-complex", [
        ("all-muts by-complex AUROC (ESM-C 6B + FoldX scalar)", "head_bycplx"),
        ("its single-point component", "head_bycplx_sp"),
        ("its multi-point component", "head_bycplx_mp"),
    ]),
]


def _verify_prose(D, lines):
    """Findings in hand-authored sentences. Reported separately: --write cannot fix these."""
    d = _derived(D)
    out = []
    for name, anchor, wanted in PROSE_CHECKS:
        hits = [i for i, l in enumerate(lines) if l.startswith(anchor)]
        if len(hits) != 1:
            out.append(f"{MD}  expected exactly 1 paragraph starting {anchor!r} ({name}), "
                       f"found {len(hits)}")
            continue
        start = hits[0]
        end = start
        while end < len(lines) and lines[end].strip():
            end += 1
        para = " ".join(lines[start:end])
        for what, key in wanted:
            v = d[key]
            if v is None:
                continue
            if f"{v:.3f}" not in para:
                out.append(f"{MD}:{start+1}  {name}: {what} is {v:.3f}, which does not appear "
                           f"in the paragraph. Rewrite the sentence by hand.")
    return out


def _write_derived(lines, D):
    d = _derived(D)
    lines = list(lines)
    for i, l in enumerate(lines):
        if l.startswith("<details><summary><b>MuLAN AUROC by embedding"):
            lines[i] = AUR_CAPTION.format(**d)
    t = mdtable.parse(MD, lines, "AUROC companion", ["| Method ", "CATH (all-muts)", "by-complex"])
    lbl = t.row("MuLAN + FoldX")
    i = t.linenos[lbl] - 1
    cs = mdtable.cells(lines[i])
    cs[1] = f"**{d['head_cath']:.3f}**"
    cs[2] = f"{d['head_bycplx']:.3f}§"
    lines[i] = "| " + " | ".join(cs) + " |"
    return lines


# =============================================================================== main


def main():
    argv = set(sys.argv[1:])
    unknown = argv - {"--check", "--verify", "--write"}
    if unknown:
        sys.exit(f"FATAL: unknown option(s) {sorted(unknown)}. See --help in the module docstring.")
    D = compute()

    if "--verify" in argv or "--write" in argv:
        bad, prose = verify(D)

    if "--write" in argv:
        if not bad:
            print(f"Nothing to write — {HTML} and {MD} already match the predictions.")
        else:
            text = open(HTML).read()
            for block in D["blocks"].values():
                text = html_write(text, block)
            open(HTML, "w").write(text)
            lines = mdtable.read(MD)
            for block in D["blocks"].values():
                lines = md_write(lines, block)
            lines = _write_derived(lines, D)
            open(MD, "w").write("\n".join(lines) + "\n")
            left, prose = verify(compute())
            print(f"Wrote {HTML} and {MD} — {len(bad)} stale cell(s) corrected.")
            if left:
                print("\nFATAL: still disagreeing after the write:")
                for m in left:
                    print("  " + m)
                sys.exit(1)
            print("Re-verified clean. Re-run the figure scripts: plot_ppS_scaling.py, "
                  "plot_ppS_generalization.py, plot_fishbone.py.")
        if prose:
            print(f"\n{len(prose)} hand-written sentence(s) still carry a stale number "
                  f"(--write does not rewrite prose):")
            for m in prose:
                print("  " + m)
            sys.exit(1)
        return

    if "--verify" in argv:
        if bad or prose:
            print(f"FATAL: {len(bad) + len(prose)} published value(s) disagree with the predictions.\n")
            for m in bad:
                print("  " + m)
            if prose:
                print("  --- hand-written prose (--write cannot fix these) ---")
                for m in prose:
                    print("  " + m)
            if bad:
                print(f"\nRun `python3 {os.path.relpath(__file__, ROOT)} --write` to correct the "
                      f"{len(bad)} cell finding(s)" + ("; the prose one(s) need a hand edit."
                                                       if prose else "."))
            else:
                print("\nNothing for --write to do — these are hand-written sentences. Edit them.")
            sys.exit(1)
        n = sum(len(b.rows) * len(b.cols) for b in D["blocks"].values())
        print(f"OK — every published MuLAN cell in {HTML} and {MD} matches the predictions "
              f"({n} cells across {len(D['blocks'])} blocks), and the prose numbers agree too.")
        return

    print(render_stdout(D, check="--check" in argv))


if __name__ == "__main__":
    main()
