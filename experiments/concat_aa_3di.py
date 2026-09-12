"""Concatenate AA-mode and 3Di-mode ProstT5 embeddings -> [L, 2048] (run6c).

For every sequence id MuLAN derives from the S1102 table (wt + mutant chains), stack the
AA-mode embedding (1024, from the run2 cache) with the structure-mode 3Di embedding
(1024, from gen_struct_embeddings.py) along the feature axis. The AA channel carries the
mutation identity; the 3Di channel carries the (WT) structural context — so a **mutant id
reuses its WT chain's 3Di tensor** (single-point mutation => backbone ~unchanged), while
its AA tensor is mutation-specific. LazyConv1d adapts to the 2048 width with no model
change. This is the direct test of H-struct (PLAN_PROSTT5_STRUCTURE_v2.md, run6c).

Usage (from repo root, mulan venv):
    python experiments/concat_aa_3di.py \
        --tables  scratch/results/run1_s1102_ankh/S1102_filtered.tsv \
        --fasta   scratch/results/run1_s1102_ankh/wt_sequences.fasta \
        --aa-dir     scratch/embeddings_prostt5 \
        --struct-dir scratch/emb_prostt5_3di \
        --out-dir    scratch/emb_prostt5_aa3di
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
    ap.add_argument("--fasta", required=True, help="AA WT fasta (to identify WT vs mutant labels)")
    ap.add_argument("--aa-dir", required=True, help="AA-mode [L,1024] cache (run2)")
    ap.add_argument("--struct-dir", required=True, help="3Di-mode [L,1024] cache (WT chains)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    wt_labels = set(parse_fasta(args.fasta))           # e.g. {1A22_A, 1A22_B, ...}
    ids = build_id_seqs(args.tables, args.fasta)       # wt + mutant ids -> sequences
    os.makedirs(args.out_dir, exist_ok=True)

    made, skipped, mism = 0, [], []
    for id_ in tqdm(ids, desc="concat AA+3Di"):
        out = os.path.join(args.out_dir, f"{id_}.pt")
        if os.path.exists(out):
            continue
        parent = id_ if id_ in wt_labels else id_.rsplit("_", 1)[0]   # mutant -> WT chain
        aa_p = os.path.join(args.aa_dir, f"{id_}.pt")
        st_p = os.path.join(args.struct_dir, f"{parent}.pt")
        if not (os.path.exists(aa_p) and os.path.exists(st_p)):
            skipped.append(id_); continue
        aa = torch.load(aa_p, weights_only=True)
        st = torch.load(st_p, weights_only=True)
        if aa.shape[0] != st.shape[0]:
            mism.append((id_, aa.shape[0], st.shape[0])); continue
        torch.save(torch.cat([aa, st], dim=-1).contiguous(), out)     # [L, 2048]
        made += 1

    print(f"[concat] wrote {made} embeddings [L,2048] -> {args.out_dir}; total ids {len(ids)}",
          flush=True)
    if skipped:
        print(f"[concat] SKIPPED missing {len(skipped)}: {skipped[:10]}", flush=True)
    if mism:
        print(f"[concat] LENGTH MISMATCH {len(mism)}: {mism[:10]}", flush=True)


if __name__ == "__main__":
    main()
