"""Analyze the MINT (A2) arm against the doc's base/aug/FoldX arms on the balanced splitter.

MINT trains on experiments/embedding_sweep/splits/balanced_seed42, whose per-fold test membership is
identical to scratch/foldx_s1102/splits_balanced_foldxdec (verified 110/110 every fold), so the
pooled OOF PCC and the worst-40 tail are directly comparable to scripts_plots/hotspots_table.json.

Reports: MINT pooled OOF PCC (all 1100) and per-fold PCC; MINT vs base/aug/FoldX on the worst-40
hotspots (from hotspots_table.json) and on the named electrostatic hotspots the whole exercise
targets (1MAH WA276R, 2O3B DB75E/N, 1BRS RA81Q).

Run:  ./.venv/bin/python experiments/mint/analyze_mint.py
"""
import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (experiments/mint/..)
MINT_RES = os.path.join(ROOT, "scratch/results/mint_a2/mint")
TRUTH = os.path.join(ROOT, "scratch/foldx_s1102/splits_balanced_foldxdec")
HOTS = os.path.join(ROOT, "scripts_plots/hotspots_table.json")
NFOLD = 10


def pcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() and b.std() else float("nan")


def load_truth():
    t = {}
    for f in range(NFOLD):
        p = f"{TRUTH}/fold_{f}/S1102_filtered_test.tsv"
        for ln in open(p):
            c = ln.rstrip("\n").split("\t")
            if len(c) < 4:
                continue
            pdb = c[0].split("_")[0]
            t[(pdb, c[2])] = {"true": float(c[3]), "fold": f}
    return t


def load_mint():
    preds, missing = {}, 0
    for f in range(NFOLD):
        p = f"{MINT_RES}/fold_{f}/training_run/test_predictions.tsv"
        if not os.path.exists(p):
            missing += 1
            continue
        for ln in open(p):
            c = ln.rstrip("\n").split("\t")
            if len(c) < 4:
                continue
            preds[(c[0].split("_")[0], c[2])] = float(c[3])
    return preds, missing


def main():
    truth = load_truth()
    mint, missing = load_mint()
    if missing:
        print(f"[!] {missing}/{NFOLD} MINT folds missing — CV not finished yet ({MINT_RES}).")
    if not mint:
        print("No MINT predictions found. Run the CV first (config_mint.sh).")
        return
    keys = [k for k in truth if k in mint]
    print(f"MINT predictions aligned to truth: {len(keys)}/{len(truth)}")

    print("\n==== MINT pooled OOF PCC ====")
    print(f"  all {len(keys)}: PCC = {pcc([truth[k]['true'] for k in keys], [mint[k] for k in keys]):.4f}")

    print("\n==== MINT per-fold PCC (vs doc base/aug/FoldX 6-PLM mean) ====")
    doc = json.load(open(HOTS))["fold_pcc"] if os.path.exists(HOTS) else {}
    print(f"  {'fold':>4} {'MINT':>7} | {'base':>7} {'aug':>7} {'FoldX':>7}  (doc 6-PLM mean)")
    for f in range(NFOLD):
        fk = [k for k in keys if truth[k]["fold"] == f]
        m = pcc([truth[k]["true"] for k in fk], [mint[k] for k in fk])
        d = doc.get(str(f), {})
        print(f"  {f:>4} {m:7.3f} | {d.get('base', float('nan')):7.3f} {d.get('aug', float('nan')):7.3f} {d.get('foldx', float('nan')):7.3f}")

    if os.path.exists(HOTS):
        top = json.load(open(HOTS))["top40"]
        rows = [r for r in top if (r["pdb"], r["mut"]) in mint]
        print(f"\n==== worst-40 hotspots: MINT vs base/aug/FoldX consensus ({len(rows)}/40 aligned) ====")
        ae = lambda arm: np.mean([abs(r["true"] - (mint[(r['pdb'], r['mut'])] if arm == 'mint' else r[arm])) for r in rows])
        print(f"  MAE  base={ae('base'):.2f}  aug={ae('aug'):.2f}  FoldX={ae('foldx'):.2f}  MINT={ae('mint'):.2f}")
        win = np.mean([abs(r["true"] - mint[(r['pdb'], r['mut'])]) < abs(r["true"] - r["base"]) for r in rows])
        print(f"  MINT beats base on {win*100:.0f}% of the worst-40")

        print("\n==== named electrostatic hotspots (the A2 target) ====")
        print(f"  {'pdb':>5} {'mut':>7} {'true':>7} {'base':>6} {'aug':>6} {'FoldX':>6} {'MINT':>6}")
        for r in top:
            k = (r["pdb"], r["mut"])
            if r["pdb"] in ("1MAH", "2O3B", "1BRS", "2PCC", "1JTG") and k in mint:
                print(f"  {r['pdb']:>5} {r['mut']:>7} {r['true']:7.2f} {r['base']:6.2f} {r['aug']:6.2f} {r['foldx']:6.2f} {mint[k]:6.2f}")


if __name__ == "__main__":
    main()
