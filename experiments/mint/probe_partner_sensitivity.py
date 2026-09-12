"""Probe gate (docs/history/PLAN_MINT_A2.md §4): does a single-residue mutation actually move the PARTNER
chain's MINT embedding? If not, A2's non-canceling term is ~0 and the arm can't beat monomer
through the bilinear head — stop before spending CV compute.

For each hotspot row we measure the shift of the chain whose SEQUENCE did NOT change (the partner),
induced purely by the mutation in the other chain via cross-chain attention:

    partner is the non-mutated chain c ;  shift = || enc(c | mut_other) - enc(c | wt_other) ||

We report the absolute norm, the norm relative to the WT partner embedding, and the mean per-residue
cosine drift, plus (as a monomer-baseline contrast) the shift of the MUTATED chain — which is large
by construction and NOT the signal of interest.

Run (after gen_mint_emb.py has produced the bundles):
    ./.venv-mint/bin/python experiments/mint/probe_partner_sensitivity.py
Env: MINT_EMB (default scratch/mint_probe/emb), MINT_PROBE_TBL (default scratch/mint_probe/hotspots.tsv).
"""
import os, torch

EMB = os.environ.get("MINT_EMB", "scratch/mint_probe/emb")
TBL = os.environ.get("MINT_PROBE_TBL", "scratch/mint_probe/hotspots.tsv")


def wt_key(s1, s2):
    return f"{s1}__{s2}"


def mut_key(s1, s2, mutations):
    return f"{s1}__{s2}__" + "-".join(mutations)


def load(kind, name):
    return torch.load(f"{EMB}/{kind}/{name}.pt", weights_only=True)


def drift(a, b):
    """(abs L2 of difference, relative to ||a||, mean per-residue cosine distance)."""
    d = (b - a).norm().item()
    rel = d / (a.norm().item() + 1e-9)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return d, rel, float((1 - cos).mean())


def main():
    print(f"{'row':<26} {'mutchain':>8} {'PARTNER Δabs':>12} {'PARTNER Δrel':>12} {'part cosΔ':>10} | {'mutchain Δrel':>12}")
    print("-" * 96)
    rows = []
    for line in open(TBL):
        p = line.split()
        if len(p) < 3:
            continue
        s1, s2, muts = p[0], p[1], tuple(p[2].split(","))
        wt = load("wt", wt_key(s1, s2))
        mut = load("mut", mut_key(s1, s2, muts))
        # which chain carries the mutation? (single-chain rows in S1102)
        chains = set(m[1] for m in muts)
        mut_ch = "1" if chains == {"A"} else "2" if chains == {"B"} else "both"
        partner = "2" if mut_ch == "1" else "1"
        pabs, prel, pcos = drift(wt[partner], mut[partner])            # partner (unchanged seq) shift
        _, mrel, _ = drift(wt[mut_ch], mut[mut_ch]) if mut_ch in ("1", "2") else (0, 0, 0)  # mutated chain shift
        tag = f"{s1.split('_')[0]} {muts[0]}"
        print(f"{tag:<26} {mut_ch:>8} {pabs:12.3f} {prel:12.4f} {pcos:10.4f} | {mrel:12.4f}")
        rows.append(prel)
    mean_rel = sum(rows) / len(rows)
    print("-" * 96)
    print(f"mean PARTNER relative shift over {len(rows)} hotspots: {mean_rel:.4f}")
    # Gate: the partner term must be materially non-zero for A2 to have a mechanism.
    verdict = "GO — partner context is materially perturbed; the non-canceling term is alive." if mean_rel > 5e-3 \
        else "MARGINAL/NO-GO — partner barely moves; route via B1 bypass or reconsider (PLAN §6) before CV."
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
