"""ProstT5 structure-mode (3Di) per-residue embeddings for run6b/6c.

Runs ProstT5 in **fold mode** (`<fold2AA>` prefix on a lowercase 3Di token string) over
the 3Di sequences produced by `gen_3di.py`, writing one `[L, 1024]` `.pt` per label —
the structure-channel counterpart of the AA-mode cache. Stripping mirrors
`mulan.utils.embed_sequence`: drop special tokens, then drop the mode-prefix position, so
the saved tensor is exactly per-residue and parallel to the AA embedding.

Only WT chain labels are embedded; mutant ids reuse their WT chain's 3Di tensor at
concat time (single-point mutation => backbone unchanged).

Usage (from repo root, mulan venv):
    python experiments/gen_struct_embeddings.py \
        --three-di scratch/3di/wt_3di.fasta \
        --out-dir  scratch/emb_prostt5_3di
"""
import argparse
import os
import time

# MPS: route ops without a Metal kernel to CPU rather than erroring; must precede `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from tqdm import tqdm

from mulan.utils import load_pretrained_plm, parse_fasta, get_device


def embed_3di(model, tok, s3di):
    """Per-residue fold-mode embedding for one lowercase 3Di string -> [L, 1024]."""
    assert "ProstT5" in tok.name_or_path, "structure mode requires ProstT5"
    seq = "<fold2AA> " + " ".join(s3di.lower())
    inp = tok(seq, return_tensors="pt", add_special_tokens=True,
              return_special_tokens_mask=True).to(model.device)
    with torch.inference_mode():
        out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
    emb = out.last_hidden_state[~inp["special_tokens_mask"].bool()]   # [1+L, d] (prefix kept)
    emb = emb[1:, :]                                                  # drop <fold2AA> position
    return emb.float().cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--three-di", required=True, help="3Di FASTA from gen_3di.py")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    seqs = parse_fasta(args.three_di)
    os.makedirs(args.out_dir, exist_ok=True)
    model, tok = load_pretrained_plm("prostt5", device=get_device())

    t0, made = time.time(), 0
    for label, s3di in tqdm(seqs.items(), desc="3Di embedding"):
        out = os.path.join(args.out_dir, f"{label}.pt")
        if os.path.exists(out):
            continue
        emb = embed_3di(model, tok, s3di)
        if emb.shape[0] != len(s3di):
            raise RuntimeError(f"{label}: emb len {emb.shape[0]} != 3Di len {len(s3di)}")
        torch.save(emb, out)
        made += 1
    print(f"[gen_struct] wrote {made} new 3Di embeddings (dim {model.config.d_model}) "
          f"to {args.out_dir} in {time.time()-t0:.0f}s; total {len(seqs)}", flush=True)


if __name__ == "__main__":
    main()
