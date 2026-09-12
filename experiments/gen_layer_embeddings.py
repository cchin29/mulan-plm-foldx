"""Generate MuLAN-format per-residue embeddings from chosen PLM layer(s).

By default MuLAN embeds with the PLM's `last_hidden_state`. This tool writes
embeddings taken from an arbitrary hidden layer, or the concatenation of several
layers, so a model can be trained on them with `mulan-train` (the cache format is
identical: one `<id>.pt` of shape [L, D] per sequence; LazyConv1d adapts to D).

It enumerates the exact sequence ids MuLAN derives from a data table + FASTA
(wild-type chains + mutant chains, same labels as mulan.data.MulanDataset), so a
subsequent `mulan-train` over the same table finds every embedding cached and
never falls back to last-layer generation.

Usage (from repo root):
    python experiments/gen_layer_embeddings.py --plm prostt5 --layers 7 \
        --tables scratch/results/run1_s1102_ankh/S1102_filtered.tsv \
        --fasta  scratch/results/run1_s1102_ankh/wt_sequences.fasta \
        --out-dir scratch/emb_prostt5_L7

    # concat of two layers -> 2*d dim
    python experiments/gen_layer_embeddings.py --plm prostt5 --layers 7,10 ...

    # all hidden states stacked -> [L, n_layers, d], for the learned scalar-mix (run6a).
    # The layer axis is residue-second so MuLAN's pad_sequence collator pads along L
    # and the LayerMix module collapses the layer axis before the encoder.
    python experiments/gen_layer_embeddings.py --plm prostt5 --all-layers ...
"""
import argparse
import os
import sys
import time

# MPS: route ops without a Metal kernel to CPU rather than erroring; must precede `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_probe import embed_all_layers                      # noqa: E402

from mulan.utils import load_pretrained_plm, parse_fasta, parse_mutations, get_device  # noqa: E402


def build_id_seqs(tables, fasta_file):
    """Replicate MuLAN's id->sequence enumeration (wt + mutant chain labels)."""
    seqs = dict(parse_fasta(fasta_file))
    ids = {}
    for tbl in tables:
        with open(tbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                s1, s2, muts = parts[0], parts[1], parts[2]
                mutations = tuple(muts.split(","))
                seq1, seq2 = seqs[s1], seqs[s2]
                mut_seq1, mut_seq2 = parse_mutations(mutations, seq1, seq2)
                mut1_label = f"{s1}_{'-'.join([m for m in mutations if m[1] == 'A'])}"
                mut2_label = f"{s2}_{'-'.join([m for m in mutations if m[1] == 'B'])}"
                ids[s1] = seq1
                ids[s2] = seq2
                ids[mut1_label] = mut_seq1
                ids[mut2_label] = mut_seq2
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plm", required=True)
    ap.add_argument("--layers", help="comma-separated layer indices, e.g. 7 or 7,10")
    ap.add_argument("--all-layers", action="store_true",
                    help="stack every hidden state -> [L, n_layers, d] (for the learned scalar-mix)")
    ap.add_argument("--tables", nargs="+", required=True, help="data table(s) to cover (train/val/test)")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.all_layers == bool(args.layers):
        ap.error("provide exactly one of --layers or --all-layers")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    os.makedirs(args.out_dir, exist_ok=True)
    ids = build_id_seqs(args.tables, args.fasta)
    print(f"[{args.plm}] layers={layers} -> {len(ids)} unique sequence ids", flush=True)

    model, tok = load_pretrained_plm(args.plm, device=get_device())
    n_hidden = model.config.num_layers + 1
    if layers is not None:
        for L in layers:
            if not (0 <= L < n_hidden):
                raise ValueError(f"layer {L} out of range 0..{n_hidden - 1} for {args.plm}")

    t0 = time.time()
    made = 0
    for id_, seq in tqdm(ids.items(), desc="embedding"):
        out = os.path.join(args.out_dir, f"{id_}.pt")
        if os.path.exists(out):
            continue
        hs = embed_all_layers(model, tok, seq)                 # [n_hidden, L, d]
        if args.all_layers:
            emb = hs.permute(1, 0, 2).contiguous()             # [L, n_hidden, d]
        else:
            emb = torch.cat([hs[L] for L in layers], dim=-1)    # [L, d*k]
        torch.save(emb.cpu(), out)
        made += 1
    out_dim = f"[L, {n_hidden}, {model.config.d_model}]" if args.all_layers \
        else f"dim {model.config.d_model * len(layers)}"
    print(f"[{args.plm}] wrote {made} new embeddings ({out_dim}) "
          f"to {args.out_dir} in {time.time()-t0:.0f}s; total ids {len(ids)}", flush=True)


if __name__ == "__main__":
    main()
