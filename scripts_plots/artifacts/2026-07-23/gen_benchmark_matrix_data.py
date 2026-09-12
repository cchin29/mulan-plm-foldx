#!/usr/bin/env python3
"""Regenerate the MuLAN data arrays for the SKEMPI benchmark matrix
(`experiments/benchmark_matrix.html` / `BENCHMARK_MATRIX.md`).

WHAT IT EMITS
-------------
For each embedding x arm it computes, in one place, every metric the matrix shows:
  * full-SKEMPI  : byCplx per-structure Spearman (T>=10),
                   CD-HIT<=60% clustered **pooled** Pearson & Spearman,
                   CATH-superfamily per-PPI Spearman.
  * S1102 ladder : leaky CV10 pooled Pearson + per-structure Spearman,
                   by-complex per-structure Spearman,
                   CD-HIT<=60% clustered **pooled** Pearson & Spearman.
It prints ready-to-paste JS array literals for the two expanded tables, plus the
collapsed "best" values, the ProtBFF (full-SKEMPI, CD-HIT<=60%, pooled) comparison,
and the **AUROC companion table** (sign-of-effect: MuLAN by-complex computed from
predictions + CATH from results_cath.csv, vs FRONTIER_AUROC constants).

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
    python3 experiments/gen_benchmark_matrix_data.py            # print all arrays + summary
    python3 experiments/gen_benchmark_matrix_data.py --check    # just the ProtBFF comparison line

Run from the mulan repo root. Needs numpy only. Re-run after new folds land, then
paste the FULL-SKEMPI / S1102 blocks into the `fullsk`/`ladder` data in
benchmark_matrix.html (and mirror into BENCHMARK_MATRIX.md).

INPUTS (paths relative to repo root)
  predictions : scratch/results/full_skempi{,_bycomplex}/<arm>/fold_{0,1,2}/training_run/test_predictions.tsv
  truth       : scratch/splits_skempi_full_{clustered_id60,bycomplex_seed42}_kfold*/fold_{f}/skempi_sp_test.tsv
  S1102       : scripts_plots/results_matrix_ps.csv (leaky/by-complex psS),
                experiments/retrain_split/results_clustered.csv (S1102 clustered pooled P/S),
                experiments/rescore_perstructure/results.csv (leaky CV10 pooled Pearson)
  CATH        : experiments/full_skempi_seqonly/results_cath.csv
ProtBFF anchor (full SKEMPI, CD-HIT 60%, pooled): Pearson 0.514 / Spearman 0.477 (bioRxiv 2025-12-23).
"""
import numpy as np, os, csv, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FOLDS = [0, 1, 2]
PROTBFF = (0.514, 0.477)   # (pooled Pearson, pooled Spearman), full SKEMPI, CD-HIT 60%

# ---- display order + arm-name maps across the various result files -----------
DISP = [("ESM-C 6B", "ESM-C 6B", "esmc6b"), ("ESM-C 600M", "ESM-C 600M", "esmc600m"),
        ("SaProt", "SaProt", "saprot"), ("ProstT5", "ProstT5", "prostt5"),
        ("ESM2-3B", "ESM2-3B", "esm2"), ("Ankh-large", "Ankh-large", "ankh"),
        ("Ankh3-large", "Ankh3-large", "ankh3_large"), ("Ankh3-xl", "Ankh3-xl", "ankh3_xl"),
        ("AIDO-16B", "AIDO-16B", "aido")]
ARMS = [("base", "base", "base", "base"),                  # (label, ps-matrix arm, pred/clustered arm, rescore arm)
        ("+ FoldX scalar", "fx_scalar", "foldx_scalar", "foldx_scalar"),
        ("+ FoldX MLP", "fx_mlp", "foldx", "foldxmlp")]

def spear(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

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

def load_truth(base, fn="skempi_sp_test.tsv"):
    d = {}
    for f in FOLDS:
        p = f"{base}/fold_{f}/{fn}"
        if not os.path.exists(p): continue
        for ln in open(p):
            c = ln.rstrip("\n").split("\t")
            if len(c) >= 4:
                try: d[(c[0], c[1], c[2])] = float(c[3])
                except ValueError: pass
    return d

def load_pred(resdir, arm):
    pred = {}
    for f in FOLDS:
        p = f"{resdir}/{arm}/fold_{f}/training_run/test_predictions.tsv"
        if not os.path.exists(p): return None
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
        if k in truth: g.setdefault((k[0], k[1]), []).append((v, truth[k]))
    vs = [spear([a for a, _ in l], [b for _, b in l]) for l in g.values() if len(l) >= T]
    vs = [v for v in vs if not np.isnan(v)]
    return float(np.mean(vs)) if vs else None

def fmt(x): return "null" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

# ---- load static CSVs --------------------------------------------------------
truth_clust = load_truth("scratch/splits_skempi_full_clustered_id60_kfold")
truth_bycpx = load_truth("scratch/splits_skempi_full_bycomplex_seed42")
cath  = {r["arm"]: r for r in csv.DictReader(open("experiments/full_skempi_seqonly/results_cath.csv"))}
s1102c = {r["arm"]: r for r in csv.DictReader(open("experiments/retrain_split/results_clustered.csv"))}
resc  = {r["arm"]: r for r in csv.DictReader(open("experiments/rescore_perstructure/results.csv"))}
psrows = list(csv.DictReader(open("scripts_plots/results_matrix_ps.csv")))
def psv(split, model, arm):
    for r in psrows:
        if r["split"] == split and r["model"] == model and r["arm"] == arm and r["ps_spearman_T10"]:
            return float(r["ps_spearman_T10"])
    return None

def main():
    check = "--check" in sys.argv
    # collapsed-best trackers
    best = {"fs_cd_s": (-9, ""), "fs_cd_p": (-9, "")}
    if not check:
        print("// ===== FULL-SKEMPI expanded: [emb, arm, byCplx_psS, cdhit_pooledS, cdhit_pooledP, cath_psS] =====")
    for dn, mdl, key in DISP:
        for lab, psa, prda, _ in ARMS:
            arm = f"{key}_{prda}"
            bp = perstruct("scratch/results/full_skempi_bycomplex", truth_bycpx, arm)
            cp, cs = pooled("scratch/results/full_skempi", truth_clust, arm)
            ca = cath.get(arm, {}).get("ps_spearman_T10")
            ca = float(ca) if ca else None
            if cs is not None and cs > best["fs_cd_s"][0]: best["fs_cd_s"] = (cs, f"{dn} {lab}")
            if cp is not None and cp > best["fs_cd_p"][0]: best["fs_cd_p"] = (cp, f"{dn} {lab}")
            if not check:
                print(f'  ["{dn if lab=="base" else ""}","{lab}",{fmt(bp)},{fmt(cs)},{fmt(cp)},{fmt(ca)}],')

    if not check:
        print("\n// ===== S1102 ladder expanded: [emb, arm, cv10_pooledP, cv10_psS, byCplx_psS, cdhit_pooledS, cdhit_pooledP] =====")
        RESC = {"base": "base", "foldx_scalar": "foldx_scalar", "foldx": "foldxmlp"}
        for dn, mdl, key in DISP:
            for lab, psa, prda, _ in ARMS:
                cv10ps = psv("S1102 leaky (per-mut CV)", mdl, psa)
                bcps   = psv("S1102 by-complex", mdl, psa)
                ck = f"{key}_{prda}"
                cS = float(s1102c[ck]["spearman"]) if ck in s1102c else None
                cP = float(s1102c[ck]["pearson"])  if ck in s1102c else None
                rk = f"{key}_{RESC[prda]}"
                cv10P = float(resc[rk]["pearson"]) if rk in resc else None
                if cv10ps is None and bcps is None and cS is None and cv10P is None: continue
                print(f'  ["{dn if lab=="base" else ""}","{lab}",{fmt(cv10P)},{fmt(cv10ps)},{fmt(bcps)},{fmt(cS)},{fmt(cP)}],')
        if "mint_base" in s1102c:
            print(f'  ["MINT (multi-chain)","base",{fmt(float(resc["mint_complex"]["pearson"]))},'
                  f'{fmt(psv("S1102 leaky (per-mut CV)","MINT","base"))},{fmt(psv("S1102 by-complex","MINT","base"))},'
                  f'{fmt(float(s1102c["mint_base"]["spearman"]))},{fmt(float(s1102c["mint_base"]["pearson"]))}],')

    # ---- AUROC companion table (sign-of-effect) ----
    # MuLAN by-complex AUROC: computed from full_skempi_bycomplex predictions (single-point).
    # MuLAN CATH AUROC: read auroc_destab from results_cath.csv (all-muts).
    cath_au = {r["arm"]: float(r["auroc_destab"]) for r in
               csv.DictReader(open("experiments/full_skempi_seqonly/results_cath.csv")) if r.get("auroc_destab")}
    # by-complex AUROC — single-point, multi-point, and ALL-muts (pool the two separate runs -> matches
    # the frontier's all-muts composition, but uses MuLAN's SP + MP models, not one).
    truth_mp = load_truth("scratch/splits_skempi_full_mp_bycomplex_seed42", "skempi_mp_test.tsv")
    bc_best = (-9, ""); bc_all = {}
    for dn, mdl, key in DISP:
        for lab, psa, prda, _ in ARMS:
            arm = f"{key}_{prda}"
            psp = load_pred("scratch/results/full_skempi_bycomplex", arm)
            pmp = load_pred("scratch/results/full_skempi_mp_bycomplex", arm)
            xy = []
            if psp: xy += [(v, truth_bycpx[k]) for k, v in psp.items() if k in truth_bycpx]
            if pmp: xy += [(v, truth_mp[k])    for k, v in pmp.items() if k in truth_mp]
            au = auroc([v for v, _ in xy], [t > 0 for _, t in xy]) if xy else None
            if au is not None:
                bc_all[arm] = au
                if au > bc_best[0]: bc_best = (au, f"{dn} {lab}")
    HEAD = "esmc6b_foldx_scalar"  # headline MuLAN arm
    if not check:
        print("\n// ===== AUROC (sign-of-effect) companion — MuLAN vs frontier =====")
        print(f"//   MuLAN+FoldX (ESM-C 6B) : CATH all-muts {cath_au.get(HEAD,float('nan')):.3f}  |  "
              f"by-complex ALL-muts {bc_all.get(HEAD,float('nan')):.3f} (SP+MP pooled)")
        print(f"//   MuLAN best CATH AUROC = {max(cath_au.values()):.3f} ; best by-complex all-muts AUROC = {bc_best[0]:.3f} ({bc_best[1]})")
        print("//   Frontier (CATH all-muts / by-complex all-muts):")
        for m, (c, b) in FRONTIER_AUROC.items():
            print(f"//     {m:13} CATH {c if c is not None else '—'} | byCplx {b if b is not None else '—'}")
        print("//   NB: MuLAN by-complex is single-point; frontier by-complex is all-muts (multi-point lifts AUROC) — cite CATH.")
        # per-arm AUROC for the expandable AUROC table: [emb, arm, CATH auroc_destab, by-complex all-muts]
        print("//   --- per-arm AUROC (paste into aurM in benchmark_matrix.html): [emb, arm, cath, bycplx_all] ---")
        for dn, mdl, key in DISP:
            for lab, psa, prda, _ in ARMS:
                arm = f"{key}_{prda}"
                ca = cath_au.get(arm)
                bc = bc_all.get(arm)
                fa = lambda x: "null" if x is None else f"{x:.3f}"
                if ca is None and bc is None: continue
                print(f'//     ["{dn if lab=="base" else ""}","{lab}",{fa(ca)},{fa(bc)}],')

    # ---- headline comparison ----
    s, sa = best["fs_cd_s"]; p, pa = best["fs_cd_p"]
    print("\n// ===== full-SKEMPI CD-HIT<=60% pooled — MuLAN vs ProtBFF (the ONLY dataset+criterion+metric match) =====")
    print(f"//   MuLAN best pooled Spearman = {s:.3f}  ({sa})")
    print(f"//   MuLAN best pooled Pearson  = {p:.3f}  ({pa})")
    print(f"//   ProtBFF (full SKEMPI, CD-HIT 60%) pooled Pearson {PROTBFF[0]} / Spearman {PROTBFF[1]}  -> MuLAN is BELOW ProtBFF")

if __name__ == "__main__":
    main()
