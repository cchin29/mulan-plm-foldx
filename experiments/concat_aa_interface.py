"""Append the binding-interface indicator as a trailing channel on AA embeddings (C3).

For every sequence id MuLAN derives from the table (wt + mutant), stack the AA-mode
embedding (1024, run2 cache) with the per-residue interface indicator (1, from
gen_interface.py) -> [L, 1025]. MuLAN's encoder (config.interface_bias) splits the trailing
channel off and uses it as a learnable attention bias. A mutant id reuses its WT chain's
interface vector (interface is a structural property of the complex), while its AA tensor is
mutation-specific.

Usage (from repo root, mulan venv):
    python experiments/concat_aa_interface.py \
        --tables   scratch/results/run1_s1102_ankh/S1102_filtered.tsv \
        --fasta    scratch/results/run1_s1102_ankh/wt_sequences.fasta \
        --aa-dir   scratch/embeddings_prostt5 \
        --iface-dir scratch/iface_prostt5 \
        --out-dir  scratch/emb_prostt5_aaiface
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_layer_embeddings import build_id_seqs   # noqa: E402

from mulan.utils import parse_fasta              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--aa-dir", required=True)
    ap.add_argument("--iface-dir", required=True, help="per-chain [L] interface vectors")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    wt_labels = set(parse_fasta(args.fasta))
    ids = build_id_seqs(args.tables, args.fasta)
    os.makedirs(args.out_dir, exist_ok=True)

    made, skipped, mism = 0, [], []
    for id_ in tqdm(ids, desc="concat AA+interface"):
        out = os.path.join(args.out_dir, f"{id_}.pt")
        if os.path.exists(out):
            continue
        parent = id_ if id_ in wt_labels else id_.rsplit("_", 1)[0]
        aa_p = os.path.join(args.aa_dir, f"{id_}.pt")
        if_p = os.path.join(args.iface_dir, f"{parent}.pt")
        if not (os.path.exists(aa_p) and os.path.exists(if_p)):
            skipped.append(id_); continue
        aa = torch.load(aa_p, weights_only=True)               # [L, 1024]
        iface = torch.load(if_p, weights_only=True)            # [L]
        if aa.shape[0] != iface.shape[0]:
            mism.append((id_, aa.shape[0], iface.shape[0])); continue
        emb = torch.cat([aa, iface.unsqueeze(-1)], dim=-1).contiguous()  # [L, 1025]
        torch.save(emb, out)
        made += 1

    print(f"[concat] wrote {made} embeddings [L,1025] -> {args.out_dir}; total ids {len(ids)}",
          flush=True)
    if skipped:
        print(f"[concat] SKIPPED missing {len(skipped)}: {skipped[:10]}", flush=True)
    if mism:
        print(f"[concat] LENGTH MISMATCH {len(mism)}: {mism[:10]}", flush=True)


if __name__ == "__main__":
    main()
