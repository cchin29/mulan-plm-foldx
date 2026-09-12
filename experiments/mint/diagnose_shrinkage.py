"""Diagnose WHY the MINT (A2) arm does not de-shrink the extreme electrostatic tail.

The A2 hypothesis: monomer PLM embeddings zero the non-canceling partner term
a_wt⊙(b_mut−b_wt) because the partner chain's embedding is mutation-independent (b_mut≡b_wt).
MINT makes the partner chain mutation-dependent, so b_mut≠b_wt and the term revives. The probe
(probe_partner_sensitivity.py) confirmed b_mut≠b_wt at a few rows. This script asks the follow-up
the per-fold PCC alone can't answer: is that revived partner signal actually LARGE enough, and does
it correlate with where MINT beats the monomer base — or does the head still regress the tail to 0?

It is self-contained and auto-detects completed folds, so it can be re-run unchanged as more MINT
folds land (partial now, full at 10CV). All quantities are computed directly from:
  - MINT preds:     scratch/results/mint_a2/mint/fold_*/training_run/test_predictions.tsv
  - truth+folds:    scratch/foldx_s1102/splits_balanced_foldxdec/fold_*/S1102_filtered_test.tsv
  - base consensus: the 6 balanced base PLMs (esm2 esmc600m prostt5 saprot ankh esmc6b)
  - partner shift:  scratch/embeddings_mint/{wt,mut}/*.pt  (δ = mean per-residue L2 of Δembedding)

Run (needs torch → main venv):  ./.venv/bin/python experiments/mint/diagnose_shrinkage.py
"""
import os, glob
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (experiments/mint/..)
J = lambda *p: os.path.join(ROOT, *p)
MINT_RES = J("scratch/results/mint_a2/mint")
TRUTH = J("scratch/foldx_s1102/splits_balanced_foldxdec")
EMB = J("scratch/embeddings_mint")
NFOLD = 10

# balanced base arms: 5 in embedding_sweep_balanced + esmc6b in the incoming drop
BASE_DIRS = {
    "esm2": J("scratch/results/embedding_sweep_balanced/esm2"),
    "esmc600m": J("scratch/results/embedding_sweep_balanced/esmc600m"),
    "prostt5": J("scratch/results/embedding_sweep_balanced/prostt5"),
    "saprot": J("scratch/results/embedding_sweep_balanced/saprot"),
    "ankh": J("scratch/results/embedding_sweep_balanced/ankh"),
    "esmc6b": J("scratch/incoming_esmc6b_20260714/results/base/S1102"),
}
NAMED = {"1MAH", "2O3B", "1BRS", "2PCC", "1JTG"}


def pcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() and b.std() else float("nan")


def slope(true, pred):
    """OLS slope of pred on true; <1 means the arm shrinks extremes toward the mean."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    if len(true) < 3 or true.std() == 0:
        return float("nan")
    return float(np.polyfit(true, pred, 1)[0])


def read_preds(path):
    """(s1id, s2id, mut, pred) rows -> {(pdb, mut): pred}."""
    d = {}
    if not os.path.exists(path):
        return d
    for ln in open(path):
        c = ln.rstrip("\n").split("\t")
        if len(c) < 4:
            continue
        d[(c[0].split("_")[0], c[2])] = float(c[3])
    return d


def load_truth():
    t = {}
    for f in range(NFOLD):
        for ln in open(f"{TRUTH}/fold_{f}/S1102_filtered_test.tsv"):
            c = ln.rstrip("\n").split("\t")
            if len(c) < 4:
                continue
            t[(c[0].split("_")[0], c[2])] = {"true": float(c[3]), "fold": f, "s1": c[0], "s2": c[1], "mut": c[2]}
    return t


def load_mint():
    preds, done = {}, []
    for f in range(NFOLD):
        p = f"{MINT_RES}/fold_{f}/training_run/test_predictions.tsv"
        if os.path.exists(p):
            done.append(f)
            preds.update(read_preds(p))
    return preds, done


def base_consensus():
    """Mean over the 6 balanced base PLMs, per (pdb, mut). Reports coverage."""
    per = {}
    for name, d in BASE_DIRS.items():
        rows = {}
        for p in glob.glob(f"{d}/fold_*/training_run/test_predictions.tsv"):
            rows.update(read_preds(p))
        per[name] = rows
    cons, n_plm = {}, {}
    keys = set().union(*[set(r) for r in per.values()])
    for k in keys:
        vs = [per[name][k] for name in per if k in per[name]]
        cons[k] = float(np.mean(vs))
        n_plm[k] = len(vs)
    return cons, per, n_plm


def partner_shift(s1, s2, mut):
    """δ_partner, δ_mut = mean per-residue L2 norm of the partner / mutated chain's Δembedding
    between wt and mut MINT bundles. Chain from mut[1] (A→chain '1', B→chain '2')."""
    wt = torch.load(f"{EMB}/wt/{s1}__{s2}.pt", map_location="cpu")
    mt = torch.load(f"{EMB}/mut/{s1}__{s2}__{mut}.pt", map_location="cpu")
    mut_ch = "1" if mut[1] == "A" else "2"
    par_ch = "2" if mut_ch == "1" else "1"
    dmut = (mt[mut_ch] - wt[mut_ch]).norm(dim=-1).mean().item()
    dpar = (mt[par_ch] - wt[par_ch]).norm(dim=-1).mean().item()
    return dpar, dmut


def main():
    truth = load_truth()
    mint, done = load_mint()
    base, per, n_plm = base_consensus()
    print(f"MINT completed folds: {done}  ({len(done)}/{NFOLD})")
    keys = [k for k in truth if k in mint]
    print(f"rows: truth={len(truth)}  mint-aligned={len(keys)}  base-consensus={len(base)}")
    plm_ok = sum(1 for k in keys if n_plm.get(k, 0) == 6)
    print(f"base consensus with all 6 PLMs on aligned rows: {plm_ok}/{len(keys)}")

    tr = np.array([truth[k]["true"] for k in keys])
    mp = np.array([mint[k] for k in keys])
    bp = np.array([base.get(k, np.nan) for k in keys])
    bok = ~np.isnan(bp)

    print("\n==== overall (completed folds) ====")
    print(f"  MINT   PCC={pcc(tr, mp):.4f}  MAE={np.mean(np.abs(tr-mp)):.3f}  slope(pred~true)={slope(tr, mp):.3f}")
    print(f"  base   PCC={pcc(tr[bok], bp[bok]):.4f}  MAE={np.mean(np.abs(tr[bok]-bp[bok])):.3f}  slope(pred~true)={slope(tr[bok], bp[bok]):.3f}")
    print("  (slope<1 = shrinks extremes toward the mean; higher slope = less shrinkage)")

    print("\n==== tail shrinkage by |true| threshold ====")
    print(f"  {'|true|>=':>9} {'n':>4} {'MINT_MAE':>9} {'base_MAE':>9} {'MINT_slope':>11} {'base_slope':>11} {'MINT_meanPred':>14} {'true_mean':>10}")
    for thr in (0, 2, 3, 4):
        m = (np.abs(tr) >= thr) & bok
        if m.sum() < 3:
            continue
        print(f"  {thr:>9} {int(m.sum()):>4} {np.mean(np.abs(tr[m]-mp[m])):>9.3f} {np.mean(np.abs(tr[m]-bp[m])):>9.3f} "
              f"{slope(tr[m], mp[m]):>11.3f} {slope(tr[m], bp[m]):>11.3f} {mp[m].mean():>14.3f} {tr[m].mean():>10.3f}")

    # ---- causal: does the revived partner term explain where MINT beats base? ----
    # Per the parallel-session feedback, the GLOBAL corr dilutes the signal: most of the 440 rows are
    # non-interface, where the partner barely moves. The fair test is the same correlation restricted
    # to the charged/interface subset — where A2 is supposed to act. Collect per-row records, then
    # slice. Interface proxy = rows whose partner actually moved (top-quartile δ_partner); charged =
    # electrostatic residue (D/E/K/R/H) as either wt or mut of the point mutation.
    CHARGED = set("DEKRH")
    rec = []  # dict per row with everything the subsets need
    for k in keys:
        info = truth[k]
        try:
            dp, dm = partner_shift(info["s1"], info["s2"], info["mut"])
        except FileNotFoundError:
            continue
        mut = info["mut"]  # e.g. "IA168A": wt=mut[0], chain=mut[1], newaa=mut[-1]
        rec.append({
            "dpar": dp, "dmut": dm, "true": info["true"],
            "impr": (abs(info["true"] - base[k]) - abs(info["true"] - mint[k])) if k in base else np.nan,
            "mint_ae": abs(info["true"] - mint[k]),
            "base_ae": abs(info["true"] - base[k]) if k in base else np.nan,
            "charged": (mut[0] in CHARGED) or (mut[-1] in CHARGED),
        })
    dpar = np.array([r["dpar"] for r in rec])
    dmut = np.array([r["dmut"] for r in rec])
    impr = np.array([r["impr"] for r in rec])
    tvals = np.array([r["true"] for r in rec])
    charged = np.array([r["charged"] for r in rec])
    mint_ae = np.array([r["mint_ae"] for r in rec])
    base_ae = np.array([r["base_ae"] for r in rec])
    imok = ~np.isnan(impr)
    hi = dpar >= np.quantile(dpar, 0.75)  # "partner actually moved" = top-quartile δ_partner
    print("\n==== revived partner-term magnitude (global) ====")
    print(f"  δ_partner: median={np.median(dpar):.4f}  δ_mutated: median={np.median(dmut):.4f}  "
          f"ratio partner/mutated median={np.median(dpar/dmut):.3f}")
    print(f"  corr(δ_partner, |true|) = {pcc(dpar, np.abs(tvals)):+.3f}")

    print("\n==== corr(δ_partner, MINT-over-base improvement) by subset ====")
    print("  (feedback: global average dilutes any tail effect; restrict to where the partner moves)")
    print(f"  {'subset':<32} {'n':>4} {'corr(δpar,impr)':>16} {'medΔpar':>9} {'MINT_MAE':>9} {'base_MAE':>9}")

    def report(name, mask):
        m = mask & imok
        if m.sum() < 3:
            print(f"  {name:<32} {int(mask.sum()):>4}  (too few for corr)")
            return
        print(f"  {name:<32} {int(m.sum()):>4} {pcc(dpar[m], impr[m]):>16.3f} "
              f"{np.median(dpar[m]):>9.4f} {np.mean(mint_ae[m]):>9.3f} {np.mean(base_ae[m]):>9.3f}")

    report("all rows", np.ones(len(rec), bool))
    report("charged mutation (D/E/K/R/H)", charged)
    report("interface (top-25% δ_partner)", hi)
    report("charged AND interface", charged & hi)
    report("tail |true|>=2", np.abs(tvals) >= 2)
    report("charged AND |true|>=2", charged & (np.abs(tvals) >= 2))
    report("interface AND |true|>=2", hi & (np.abs(tvals) >= 2))
    print("  (if A2 helped through this head, corr would go strongly + on the charged/interface tail)")

    # named electrostatic hotspots present so far
    print("\n==== named electrostatic hotspots (completed folds) ====")
    print(f"  {'pdb':>5} {'mut':>8} {'fold':>4} {'true':>7} {'base':>7} {'MINT':>7} {'δ_par':>7} {'δ_mut':>7}")
    for k in sorted(keys):
        if k[0] not in NAMED:
            continue
        info = truth[k]
        try:
            dp, dm = partner_shift(info["s1"], info["s2"], info["mut"])
        except FileNotFoundError:
            dp = dm = float("nan")
        b = base.get(k, float("nan"))
        print(f"  {k[0]:>5} {k[1]:>8} {info['fold']:>4} {info['true']:>7.2f} {b:>7.2f} {mint[k]:>7.2f} {dp:>7.4f} {dm:>7.4f}")


if __name__ == "__main__":
    main()
