"""Layer-wise linear probe: which PLM hidden layer best encodes S1102 ΔΔG?

For each hidden state of a protein language model (input embeddings + every encoder
layer), build a per-mutation feature from the *mutated chain* and fit a linear
RidgeCV probe (5-fold CV) to predict ΔΔG. Report held-out PCC/RMSE per layer.

This is a fast proxy for MuLAN's wt<->mut Light-Attention head, so absolute numbers
are lower than full MuLAN training -- the **ranking across layers** is the signal.

PLM-agnostic: preprocessing mirrors `mulan.utils.embed_sequence` (ProstT5 gets the
`<AA2fold>` amino-acid prefix; ProtTrans models get residue spacing; others as-is).

Usage (from repo root, mulan venv active):
    python experiments/layer_probe.py --plm prostt5 \
        --data scratch/results/run1_s1102_ankh/S1102_filtered.tsv \
        --fasta scratch/results/run1_s1102_ankh/wt_sequences.fasta \
        --features-out scratch/probe_features_prostt5.npz \
        --md-out experiments/results/probe_prostt5.md

Reproduces the run referenced in experiments/LAYER_PROBE.md.
"""
import argparse
import re
import time
from collections import defaultdict

import numpy as np
import os

# MPS: route ops without a Metal kernel to CPU rather than erroring; must precede `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

from mulan.utils import load_pretrained_plm, parse_fasta, get_device


def embed_all_layers(model, tok, sequence):
    """Per-residue hidden states for every layer, aligned to `sequence`.
    Shape [n_layers+1, L, d]. Same preprocessing/stripping as
    mulan.utils.embed_sequence, but with output_hidden_states=True."""
    name = tok.name_or_path
    seq = re.sub(r"[UZOB]", "X", sequence.upper())
    is_prostt5 = "ProstT5" in name
    if is_prostt5:
        seq = "<AA2fold> " + " ".join(seq)
    elif "Rostlab/prot" in name:
        seq = " ".join(seq)
    inp = tok(seq, return_tensors="pt", add_special_tokens=True,
              return_special_tokens_mask=True).to(model.device)
    with torch.inference_mode():
        out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                    output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0]      # [n, T, d]
    keep = ~inp["special_tokens_mask"].bool()[0]          # strips </s> (+ ProstT5 prefix below)
    hs = hs[:, keep, :]
    if is_prostt5:
        hs = hs[:, 1:, :]                                  # drop <AA2fold> prefix position
    return hs.float().cpu()


def load_records(data, fasta):
    wt = parse_fasta(fasta)
    records = []   # (wt_chain_seq, pos0, mut_aa, ddG, mut_str)
    skipped = 0
    with open(data) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            s1, s2, muts, score = parts[0], parts[1], parts[2], float(parts[3])
            mut_list = muts.split(",")
            if len(mut_list) != 1:                         # S1102 is single mutations
                skipped += 1
                continue
            m = mut_list[0]
            chain, pos, mut_aa = m[1], int(m[2:-1]), m[-1]
            wt_label = s1 if chain == "A" else s2
            if wt_label not in wt:
                skipped += 1
                continue
            records.append((wt[wt_label], pos - 1, mut_aa, score, m))
    return records, skipped


def probe(X, y, seed=42):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        mdl = RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0, 10000.0])
        mdl.fit(sc.transform(X[tr]), y[tr])
        preds[te] = mdl.predict(sc.transform(X[te]))
    pcc = pearsonr(y, preds)[0]
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    return pcc, rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plm", required=True, help="PLM name in mulan (e.g. prostt5, ankh)")
    ap.add_argument("--data", required=True, help="S1102_filtered.tsv")
    ap.add_argument("--fasta", required=True, help="wild-type sequences FASTA")
    ap.add_argument("--features-out", required=True, help="npz path for cached features")
    ap.add_argument("--md-out", required=True, help="markdown results path")
    ap.add_argument("--limit", type=int, default=0, help="probe only first N mutations (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    records, skipped = load_records(args.data, args.fasta)
    if args.limit:
        records = records[: args.limit]
    print(f"[{args.plm}] mutations: {len(records)} (skipped {skipped})", flush=True)

    model, tok = load_pretrained_plm(args.plm, device=get_device())
    n_layers = model.config.num_layers + 1
    d_model = model.config.d_model
    print(f"[{args.plm}] {model.config.num_layers} layers, d_model {d_model}", flush=True)

    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        groups[rec[0]].append(idx)

    N = len(records)
    site = np.zeros((N, n_layers, d_model), dtype=np.float32)
    pool = np.zeros((N, n_layers, d_model), dtype=np.float32)
    y = np.array([r[3] for r in records], dtype=np.float64)

    done = 0
    for wtseq, idxs in groups.items():
        hs_wt = embed_all_layers(model, tok, wtseq)
        wt_mean = hs_wt.mean(dim=1)
        for i in idxs:
            _, pos0, mut_aa, _, _ = records[i]
            mutseq = wtseq[:pos0] + mut_aa + wtseq[pos0 + 1:]
            hs_mut = embed_all_layers(model, tok, mutseq)
            site[i] = (hs_mut[:, pos0, :] - hs_wt[:, pos0, :]).numpy()
            pool[i] = (hs_mut.mean(dim=1) - wt_mean).numpy()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{N} embedded ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{args.plm}] embedding done in {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(args.features_out, site=site, pool=pool, y=y)
    print(f"saved features -> {args.features_out}", flush=True)

    rows = []
    for L in range(n_layers):
        ps, rs = probe(site[:, L, :], y)
        pp, rp = probe(pool[:, L, :], y)
        rows.append((L, ps, rs, pp, rp))
        print(f"layer {L:2d}: site PCC={ps:.3f} RMSE={rs:.3f} | pool PCC={pp:.3f} RMSE={rp:.3f}",
              flush=True)

    best_site = max(rows, key=lambda r: r[1])
    best_pool = max(rows, key=lambda r: r[3])
    last = n_layers - 1

    with open(args.md_out, "w") as f:
        f.write(f"# {args.plm} layer-wise probe for S1102 ΔΔG\n\n")
        f.write(f"N = {N} single mutations. Linear RidgeCV probe (5-fold CV), per layer.\n")
        f.write("Features from the mutated chain: **site delta** (`mut-wt` at the mutated "
                "residue) and **pool delta** (mean over residues). "
                f"Layer {last} = `last_hidden_state` (the default MuLAN uses).\n\n")
        f.write("| Layer | site PCC | site RMSE | pool PCC | pool RMSE |\n")
        f.write("|---|---|---|---|---|\n")
        for L, ps, rs, pp, rp in rows:
            mark = " ⬅ default layer" if L == last else ""
            f.write(f"| {L} | {ps:.3f} | {rs:.3f} | {pp:.3f} | {rp:.3f} |{mark}\n")
        f.write(f"\n**Best site-delta layer:** {best_site[0]} (PCC {best_site[1]:.3f})\n\n")
        f.write(f"**Best pool-delta layer:** {best_pool[0]} (PCC {best_pool[3]:.3f})\n\n")
        f.write(f"**Final layer ({last}):** site PCC {rows[last][1]:.3f}, "
                f"pool PCC {rows[last][3]:.3f}\n")
    print(f"wrote {args.md_out}", flush=True)
    print(f"[{args.plm}] BEST site layer {best_site[0]} PCC {best_site[1]:.3f} | "
          f"BEST pool layer {best_pool[0]} PCC {best_pool[3]:.3f}", flush=True)


if __name__ == "__main__":
    main()
