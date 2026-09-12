"""Generate per-residue embeddings for an arbitrary (table, fasta) via MulanDataset.

For standard PLMs that load through `load_pretrained_plm` (ESM2, Ankh, ProstT5, ...).
Generalizes `scratch/gen_plm_emb.py` (which hardcodes the S1102 table/fasta) to any
benchmark or augmentation table. Device follows `get_device()` (MPS default; force CPU
with MULAN_FORCE_CPU=1). Not for SaProt/ESM-C (custom paths).

Usage: python experiments/gen_emb_generic.py <plm> <emb_dir> <table.tsv> <wt.fasta>
"""
import os, sys, time
import torch
from mulan.data import MulanDataset

plm, emb, tbl, wt = sys.argv[1:5]
os.makedirs(emb, exist_ok=True)
t0 = time.time()
ds = MulanDataset.from_table(tbl, wt, emb, plm)   # generates only missing ids
n = len([f for f in os.listdir(emb) if f.endswith(".pt")])

# Coverage assert: every id this dataset references MUST have a .pt on disk, or a
# downstream mulan-train silently regenerates it from the PLM (and for SDK-only PLMs
# can't). A bare "n tensors" print can't catch a gap because the dir may hold ids from
# other tables. Enumerate the dataset's own expected ids (the same union __init__ uses)
# and verify each landed; then spot-check a few tensors are finite [L, dim] with a
# single consistent width. Turns a partial/failed generation into a loud failure.
expected = {i for ids in ds._sequences_ids for i in ids}
missing = sorted(i for i in expected if not os.path.exists(os.path.join(emb, f"{i}.pt")))
assert not missing, (
    f"[{plm}] {len(missing)}/{len(expected)} expected embeddings MISSING after gen "
    f"(e.g. {missing[:3]}) -> {emb}"
)
dims = set()
for i in list(expected)[:8]:
    t = torch.load(os.path.join(emb, f"{i}.pt"), map_location="cpu")
    assert t.ndim == 2 and bool(torch.isfinite(t).all()), \
        f"[{plm}] bad embedding {i!r}: shape={tuple(t.shape)}, finite={bool(torch.isfinite(t).all())}"
    dims.add(int(t.shape[1]))
assert len(dims) == 1, f"[{plm}] inconsistent embedding dims across sample: {sorted(dims)}"
print(f"[{plm}] {len(ds)} rows | {n} tensors | {len(expected)} ids covered | dim {dims.pop()} "
      f"-> {emb} | {time.time()-t0:.0f}s", flush=True)
