"""B1: per-complex pooled WT structure-context vectors for the MuLAN head.

For each (s1, s2) complex in the data table, mean-pool the WT 3Di-mode embeddings of both
chains (from gen_struct_embeddings.py) and concatenate -> a fixed [2*1024] = 2048-d context
vector, saved as `<s1>__<s2>.pt`. MuLAN's head consumes this *outside* the siamese mut-wt
difference (config.struct_context), so the structure enters as complex-level context rather
than a per-mutation delta that the difference would cancel — the design that respects 3Di's
backbone-invariance under single-point mutation (see PROSTT5_STRUCTURE_OPTIONS.md).

Usage (from repo root, mulan venv):
    python experiments/gen_struct_context.py \
        --tables     scratch/results/run1_s1102_ankh/S1102_filtered.tsv \
        --struct-dir scratch/emb_prostt5_3di \
        --out-dir    scratch/struct_ctx_prostt5
"""
import argparse
import os

# MPS: route ops without a Metal kernel to CPU rather than erroring; must precede `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--struct-dir", required=True, help="WT 3Di-mode [L,1024] cache")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # unique complexes (s1, s2) across the table(s)
    complexes = {}
    for tbl in args.tables:
        with open(tbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                complexes[(parts[0], parts[1])] = True

    os.makedirs(args.out_dir, exist_ok=True)
    cache = {}

    def pooled(label):
        if label not in cache:
            emb = torch.load(os.path.join(args.struct_dir, f"{label}.pt"), weights_only=True)
            cache[label] = emb.mean(dim=0)            # [1024]
        return cache[label]

    made, missing = 0, []
    for s1, s2 in tqdm(complexes, desc="struct context"):
        out = os.path.join(args.out_dir, f"{s1}__{s2}.pt")
        if os.path.exists(out):
            continue
        if not (os.path.exists(os.path.join(args.struct_dir, f"{s1}.pt"))
                and os.path.exists(os.path.join(args.struct_dir, f"{s2}.pt"))):
            missing.append((s1, s2)); continue
        ctx = torch.cat([pooled(s1), pooled(s2)], dim=-1).contiguous()   # [2048]
        torch.save(ctx, out)
        made += 1

    print(f"[struct_ctx] wrote {made}/{len(complexes)} complex context vectors [2048] "
          f"-> {args.out_dir}", flush=True)
    if missing:
        print(f"[struct_ctx] MISSING 3Di for {len(missing)}: {missing[:10]}", flush=True)


if __name__ == "__main__":
    main()
