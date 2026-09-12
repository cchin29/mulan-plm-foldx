#!/usr/bin/env python3
"""Tier-1: MuLAN attention -> interface AUROC on S1102, per PLM backbone.

For a trained MuLAN head, extract per-residue LightAtt attention on each S1102
wild-type chain and score it against the chain's interface labels (residues in
cross-chain contact). This is the head-to-head that answers the question:
"do the tested PLMs improve interaction-site prediction over Ankh-large (0.757)?"

Validation gate: run --tag ankh first and confirm we recover the paper's Fig S1
average AUROC ~= 0.757 on our labels before trusting the other backbones. The
paper number is the AVERAGE over sequences (macro), so that column is the anchor.

Attention recipe (paper Methods L316, identical to scripts/extract_attentions.py):
    model.encoder(embedding).attention -> mean over (3 heads x 64 filters) -> MinMax.
Mirrors the released extraction; per-chain MinMax [0,1].

Labels: scratch/interface_masks/{c1}__{c2}.pt is a bool [L1,L2] cross-chain heavy-
atom-8A contact mask. Per-residue label for chain c1 = mask.any(axis=1), for c2 =
mask.any(axis=0). (Paper Fig S1 used INTBuilder; our masks are heavy-atom 8A -- the
ankh validation checks the two agree closely enough to trust the comparison.)

Sequences: experiments/embedding_sweep/data/wt_sequences.fasta (chain id -> seq).
Head checkpoint: scratch/results/embedding_sweep_balanced/{tag}/fold_{fold}/training_run/model.ckpt
(one fold used as the trained encoder; attention is a ddG-trained encoder property,
weakly sensitive to fold membership -- --fold selects which, default 0).

Usage:  ./.venv/bin/python experiments/attention_interface/score_s1102.py --tag ankh
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MASK_DIR = ROOT / "scratch" / "interface_masks"
WT_FASTA = ROOT / "experiments" / "embedding_sweep" / "data" / "wt_sequences.fasta"
CKPT = "scratch/results/embedding_sweep_balanced/{tag}/fold_{fold}/training_run/model.ckpt"

# dir tag -> load_pretrained_plm key (constants.PLM_ENCODERS). HF-loadable in .venv;
# esmc600m/esmc6b/esm3 need their own SDK/venv or cached embeddings (Tier-1b).
TAG2PLM = {
    "ankh": "ankh", "esm2": "esm", "saprot": "saprot", "saprot13b": "saprot_1.3b",
    "prostt5": "prostt5", "ankh3_large": "ankh3_large", "ankh3_xl": "ankh3_xl",
}


def parse_fasta(p: Path) -> dict[str, str]:
    seqs, name = {}, None
    for line in p.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            seqs[name] = ""
        elif name is not None:
            seqs[name] += line.strip()
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ankh", help="backbone dir tag (checkpoint + output name)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--emb-dir", default=None,
                    help="if set, feed cached per-chain {chain}.pt embeddings instead of live-"
                         "embedding via the PLM (for structure/SDK PLMs: saprot, esmc, esm3).")
    ap.add_argument("--mask-dir", default=str(MASK_DIR),
                    help="interface-mask dir (default S1102; full-SKEMPI = scratch/interface_masks_skempi_full)")
    ap.add_argument("--wt-fasta", default=str(WT_FASTA), help="WT-chain FASTA matching the masks")
    ap.add_argument("--ckpt", default=None,
                    help="explicit checkpoint path; overrides the embedding_sweep_balanced/{tag}/fold_{fold} pattern")
    ap.add_argument("--suffix", default="", help="output-name suffix, e.g. _skempi_full")
    args = ap.parse_args()
    if args.emb_dir is None and args.tag not in TAG2PLM:
        raise SystemExit(f"--tag {args.tag} needs --emb-dir (not HF-loadable); "
                         f"live-embed tags: {list(TAG2PLM)}")

    import mulan  # noqa: E402  (import after arg parse so --help is instant)
    from mulan.modules import LightAttModel

    device = mulan.get_device()
    src = f"cached:{args.emb_dir}" if args.emb_dir else f"live:{TAG2PLM[args.tag]}"
    print(f"device={device}  tag={args.tag}  emb={src}", flush=True)

    wt = parse_fasta(Path(args.wt_fasta))
    masks = sorted(Path(args.mask_dir).glob("*.pt"))
    print(f"{len(wt)} WT chains, {len(masks)} interface masks", flush=True)

    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / CKPT.format(tag=args.tag, fold=args.fold)
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # from_pretrained arch-drift warn
        head = LightAttModel.from_pretrained(str(ckpt_path), device=device)
    head.eval()

    if args.emb_dir:
        emb_dir = ROOT / args.emb_dir
        @torch.inference_mode()
        def embed(chain: str, seq: str) -> torch.Tensor | None:
            p = emb_dir / f"{chain}.pt"
            if not p.exists():
                return None
            e = torch.load(p, map_location=device, weights_only=False)
            return e.unsqueeze(0) if e.dim() == 2 else e   # [L,H] -> [1,L,H]
    else:
        plm_model, plm_tok = mulan.load_pretrained_plm(TAG2PLM[args.tag], device=device)
        @torch.inference_mode()
        def embed(chain: str, seq: str) -> torch.Tensor:
            return mulan.utils.embed_sequence(plm_model, plm_tok, seq)

    @torch.inference_mode()
    def attention_of(chain: str, seq: str) -> np.ndarray | None:
        emb = embed(chain, seq)
        if emb is None:
            return None
        att = head.encoder(emb).attention.squeeze(0).cpu().numpy().mean(axis=(-2, -3))
        return mulan.utils.minmax_scale(att)

    # cache attention per chain (a chain can appear in multiple complexes)
    att_cache: dict[str, np.ndarray | None] = {}
    per_chain, pooled_s, pooled_l = [], [], []
    per_chain_named: dict[str, float] = {}   # chain -> AUROC (for paired bootstrap)
    n_len_mismatch = n_no_label = n_missing = 0

    for i, mp in enumerate(masks):
        c1, c2 = mp.stem.split("__")
        mask = torch.load(mp, map_location="cpu").numpy().astype(bool)  # [L1,L2]
        for chain, lab in ((c1, mask.any(axis=1)), (c2, mask.any(axis=0))):
            if chain not in wt:
                continue
            seq = wt[chain]
            if chain not in att_cache:
                att_cache[chain] = attention_of(chain, seq)
            att = att_cache[chain]
            if att is None:
                n_missing += 1
                continue
            if att.size != lab.size:
                n_len_mismatch += 1
                continue
            npos = int(lab.sum())
            if npos == 0 or npos == lab.size:
                n_no_label += 1        # no ROC defined for a single-class chain
            else:
                a = roc_auc_score(lab, att)
                per_chain.append(a)
                per_chain_named[chain] = a
            pooled_s.append(att)
            pooled_l.append(lab)
        if (i + 1) % 30 == 0:
            print(f"  ...{i+1}/{len(masks)} masks  (chains embedded: {len(att_cache)})", flush=True)

    per_chain = np.asarray(per_chain)
    n_emb = sum(v is not None for v in att_cache.values())
    S, L = np.concatenate(pooled_s), np.concatenate(pooled_l)
    macro = float(per_chain.mean())
    pooled = float(roc_auc_score(L, S))

    print("\n" + "=" * 56)
    print(f"backbone         : {args.tag}  (fold {args.fold})")
    print(f"chains scored    : {per_chain.size} (with both classes) / {n_emb} embedded")
    print(f"missing-emb skip : {n_missing}   len-mismatch skip: {n_len_mismatch}   single-class skip: {n_no_label}")
    print(f"macro AUROC      : {macro:.3f}   (paper Fig S1 Ankh = 0.757)")
    print(f"pooled AUROC     : {pooled:.3f}")
    print(f"per-chain spread : min {per_chain.min():.3f}  med {np.median(per_chain):.3f}  max {per_chain.max():.3f}")
    print("=" * 56)

    base = f"s1102{args.suffix}_{args.tag}"
    out = ROOT / "experiments" / "attention_interface" / f"{base}.csv"
    with open(out, "w") as fh:
        fh.write("tag,fold,n_chains,macro_auroc,pooled_auroc\n")
        fh.write(f"{args.tag},{args.fold},{per_chain.size},{macro:.4f},{pooled:.4f}\n")
    pc = ROOT / "experiments" / "attention_interface" / f"{base}_perchain.csv"
    with open(pc, "w") as fh:
        fh.write("chain,auroc\n")
        for chain, a in sorted(per_chain_named.items()):
            fh.write(f"{chain},{a:.4f}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
