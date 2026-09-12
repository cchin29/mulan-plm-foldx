"""Generate SaProt (650M, SA=AA+3Di) per-residue embeddings for the S1102 ids.

Stage-1 of the SaProt structure test (WT 3Di, no FoldX). For every id MuLAN derives
from the S1102 table (wt + mutant chains), build a **structure-aware (SA) sequence** —
one 2-char token per residue = AA(upper)+3Di(lower) — and embed it with SaProt.

The 3Di channel is the **WT** structure (mini3di, from gen_3di.py); a **mutant id reuses
its parent WT chain's 3Di** (single-point mutation => backbone ~unchanged), exactly like
`concat_aa_3di.py`. Because SaProt fuses AA+3Di per token, the mutated position's SA token
still flips (its AA half changes), so a mutant embedding differs from WT even with WT 3Di.

Two output sets (same id coverage) for a clean structure ablation:
  --mode real     : real WT 3Di in the structure half         -> emb_saprot/
  --mode seqonly  : structure half = '#' (unknown) everywhere  -> emb_saprot_seqonly/
  --mode both     : write both (default; one model load)

Embeddings come from SaProt's final hidden state (EsmForMaskedLM, output_hidden_states),
saved as [L, 1280] .pt — matching the offline-embeddings cache convention.

Usage (from repo root, mulan venv; CPU keeps MPS free for other jobs):
    OMP_NUM_THREADS=6 MULAN=$PWD PYTHONPATH=$PWD python experiments/gen_saprot_emb.py \
        --tables scratch/results/run1_s1102_ankh/S1102_filtered.tsv \
        --fasta  scratch/results/run1_s1102_ankh/wt_sequences.fasta \
        --tdi    scratch/3di/wt_3di.fasta \
        --real-dir    scratch/emb_saprot \
        --seqonly-dir scratch/emb_saprot_seqonly \
        --device cpu --mode both
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_layer_embeddings import build_id_seqs   # noqa: E402

from mulan.constants import PLM_ENCODERS         # noqa: E402
from mulan.utils import parse_fasta              # noqa: E402

MODEL_ID = PLM_ENCODERS["saprot"]  # westlake-repl/SaProt_650M_AF2
UNKNOWN_3DI = "#"                  # SaProt's masked/unknown-structure token
AA_MASK = "#"                     # SaProt's AA-half mask (valid: '#d', '##', etc.)
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def sa_seq(aa: str, tdi: str) -> str:
    """Interleave AA (upper) + 3Di (lower) -> continuous 2-char-token SA string.

    SaProt's SA vocab has ONLY the 20 canonical AAs in the AA-half; a non-canonical
    residue (e.g. 'X') has no valid 2-char token, and two adjacent OOV pairs collapse
    into one token, breaking per-residue alignment. Map any non-canonical AA to the
    AA-mask '#' (a valid half — e.g. '#d'/'##'), the idiomatic "unknown residue"
    encoding. Identical rule in both real/seq-only modes, so the AA channel is the
    same across the ablation and only the structure half differs.
    """
    return "".join((a.upper() if a.upper() in CANONICAL else AA_MASK) + d.lower()
                   for a, d in zip(aa, tdi))


def load_saprot(device, model_id=MODEL_ID):
    from transformers import AutoTokenizer, EsmForMaskedLM
    tok = AutoTokenizer.from_pretrained(model_id)
    # EsmForMaskedLM matches how the checkpoint was saved (model card); the base encoder's
    # final layer == what we want. Loading the LM head avoids the "newly initialized" noise
    # of loading EsmModel and guarantees we use the trained weights.
    model = EsmForMaskedLM.from_pretrained(model_id, output_hidden_states=True)
    model = model.to(device).eval()
    return model, tok


@torch.inference_mode()
def embed_sa(model, tok, sa: str, device):
    enc = tok(sa, return_tensors="pt", add_special_tokens=True,
              return_special_tokens_mask=True).to(device)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    hidden = out.hidden_states[-1]                       # [1, T, 1280]
    per_res = hidden[~enc["special_tokens_mask"].bool()]  # [L, 1280]
    return per_res.contiguous().cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--fasta", required=True, help="AA WT fasta (labels + WT seqs)")
    ap.add_argument("--tdi", required=True, help="WT 3Di fasta (mini3di, gen_3di.py)")
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--seqonly-dir", required=True)
    ap.add_argument("--mode", choices=["real", "seqonly", "both"], default="both")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plm", default="saprot", choices=[k for k in PLM_ENCODERS if k.startswith("saprot")],
                    help="which SaProt checkpoint (constants.PLM_ENCODERS key); e.g. saprot (650M) or saprot_1.3b")
    args = ap.parse_args()

    model_id = PLM_ENCODERS[args.plm]
    do_real = args.mode in ("real", "both")
    do_seq = args.mode in ("seqonly", "both")

    wt_labels = set(parse_fasta(args.fasta))
    ids = build_id_seqs(args.tables, args.fasta)     # wt + mutant ids -> AA seqs
    tdi = parse_fasta(args.tdi)                        # WT label -> 3Di string
    if do_real:
        os.makedirs(args.real_dir, exist_ok=True)
    if do_seq:
        os.makedirs(args.seqonly_dir, exist_ok=True)

    print(f"[{args.plm}] {model_id} on {args.device} | ids={len(ids)} | mode={args.mode}", flush=True)
    model, tok = load_saprot(args.device, model_id)

    made_r = made_s = 0
    skipped_no3di, mism, tok_desync = [], [], []
    for id_ in tqdm(sorted(ids), desc="saprot emb"):
        aa = ids[id_]
        # 3Di is WT-structure (FoldX proved mutant 3Di == WT 3Di, §15), so ANY label —
        # base mutant `PDB_C_mut` OR aug reverse-mutant `PDB_C_mutA_mutB` — takes the 3Di
        # of its base WT chain `PDB_C` = the first two underscore fields.
        parent = "_".join(id_.split("_")[:2])

        # --- real 3Di ---
        if do_real:
            out_r = os.path.join(args.real_dir, f"{id_}.pt")
            if os.path.exists(out_r):
                made_r += 1
            else:
                # Use real WT 3Di when available+aligned; otherwise fall back to the
                # '#' structure-mask so the real set still covers EVERY id (a missing
                # embedding would make CV10 wrongly try to embed it via embed_sequence).
                if parent in tdi and len(tdi[parent]) == len(aa):
                    struct = tdi[parent]
                else:
                    struct = UNKNOWN_3DI * len(aa)
                    (skipped_no3di if parent not in tdi else mism).append(id_)
                emb = embed_sa(model, tok, sa_seq(aa, struct), args.device)
                if emb.shape[0] != len(aa):
                    tok_desync.append((id_, "real", emb.shape[0], len(aa)))
                else:
                    torch.save(emb, out_r)
                    made_r += 1

        # --- seq-only control (structure = '#') ---
        if do_seq:
            out_s = os.path.join(args.seqonly_dir, f"{id_}.pt")
            if os.path.exists(out_s):
                made_s += 1
            else:
                emb = embed_sa(model, tok, sa_seq(aa, UNKNOWN_3DI * len(aa)), args.device)
                if emb.shape[0] != len(aa):
                    tok_desync.append((id_, "seqonly", emb.shape[0], len(aa)))
                else:
                    torch.save(emb, out_s)
                    made_s += 1

    if do_real:
        print(f"[saprot] real:    {made_r} embeddings -> {args.real_dir}", flush=True)
    if do_seq:
        print(f"[saprot] seqonly: {made_s} embeddings -> {args.seqonly_dir}", flush=True)
    if skipped_no3di:
        print(f"[saprot] real '#'-mask fallback (no WT 3Di) {len(skipped_no3di)}: {skipped_no3di[:10]}", flush=True)
    if mism:
        print(f"[saprot] real '#'-mask fallback (len mismatch) {len(mism)}: {mism[:10]}", flush=True)
    if tok_desync:
        print(f"[saprot] TOK DESYNC (skipped) {len(tok_desync)}: {tok_desync[:10]}", flush=True)


if __name__ == "__main__":
    main()
