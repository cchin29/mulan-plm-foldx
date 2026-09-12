"""Generate ESM C 600M per-residue embeddings for the S1102 set, saved as {id_}.pt
([L, 1152]) into scratch/embeddings_esmc600m/, named EXACTLY as mulan's MulanDataset
enumerates them (so the 10CV sweep loads them read-only with no PLM at train time).

ESM C 600M loads via the `esm` SDK (`ESMC.from_pretrained("esmc_600m")`) — the same
transformers-independent path as the structural_context `esmc_600m` embedder — so this
must run in **`.venv-esmc`** (the SDK isn't in `.venv-structctx`/`.venv`). It runs on MPS
in bf16 (~1.4 s/seq); embeddings are cast to fp32 before saving to match the other runs.

Portable (unlike scratch/esmc6b_gen.py, which hand-builds the 6B from a hardcoded Linux
snapshot). The id/label enumeration and non-canonical-AA substitution are copied verbatim
from esmc6b_gen.py (== mulan.data.MulanDataset._fill_metadata + mulan.utils) so the cached
tensors are identical in naming/preprocessing to the Ankh/ProstT5/ESM C 6B runs.

Usage (from repo root):
    .venv-esmc/bin/python experiments/embedding_sweep/gen_esmc600m_emb.py
Paths are env-overridable (ESMC_EMB / ESMC_WT / ESMC_TBL) to target other tables.
"""
import os, re, sys, time, torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

EMB = os.environ.get("ESMC_EMB", "scratch/embeddings_esmc600m")
WT = os.environ.get("ESMC_WT", "experiments/embedding_sweep/data/wt_sequences.fasta")
TBL = os.environ.get("ESMC_TBL", "experiments/embedding_sweep/data/S1102_filtered.tsv")
DIM = 1152


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def parse_fasta(path):  # == mulan.utils.parse_fasta
    d, key = {}, None
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(">"):
            key = line.strip().split()[0][1:].split("|")[0]
            d[key] = ""
        else:
            d[key] += line.strip().upper()
    return d


def parse_mutations(mutations, seq1, seq2):  # == mulan.utils.parse_mutations
    seq1, seq2 = list(seq1), list(seq2)
    for m in mutations:
        if m[1] == "A":
            seq1[int(m[2:-1]) - 1] = m[-1]
        else:
            seq2[int(m[2:-1]) - 1] = m[-1]
    return "".join(seq1), "".join(seq2)


def enumerate_ids():
    """Return id_ -> sequence for every WT and mutant chain referenced by the table."""
    wt = parse_fasta(WT)
    seqs = dict(wt)
    for line in open(TBL):
        if not line.strip():
            continue
        f = line.split()
        s1l, s2l, muts = f[0], f[1], tuple(f[2].split(","))
        m1, m2 = parse_mutations(muts, wt[s1l], wt[s2l])
        m1l = f"{s1l}_{'-'.join(m for m in muts if m[1] == 'A')}"
        m2l = f"{s2l}_{'-'.join(m for m in muts if m[1] == 'B')}"
        seqs[s1l], seqs[s2l], seqs[m1l], seqs[m2l] = wt[s1l], wt[s2l], m1, m2
    return seqs


def get_device():
    if torch.backends.mps.is_available() and os.environ.get("MULAN_FORCE_CPU") != "1":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def embed_residues(model, seq):
    """Return [L, 1152] final-layer per-residue embedding, BOS/EOS stripped, fp32/CPU."""
    out = model.logits(
        model.encode(ESMProtein(sequence=seq)),
        LogitsConfig(return_embeddings=True),
    )
    return out.embeddings[0, 1:-1, :].float().contiguous().cpu()


def main():
    os.makedirs(EMB, exist_ok=True)
    seqs = enumerate_ids()
    todo = [i for i in seqs if not os.path.exists(os.path.join(EMB, i + ".pt"))]
    log(f"total unique ids {len(seqs)} | already cached {len(seqs)-len(todo)} | to do {len(todo)}")
    if not todo:
        log("nothing to do"); return
    dev = get_device()
    log(f"loading esmc_600m via esm SDK on {dev}")
    model = ESMC.from_pretrained("esmc_600m", device=dev).eval()
    t0 = time.time()
    for n, id_ in enumerate(todo, 1):
        seq = re.sub(r"[UZOB]", "X", seqs[id_].upper())  # == embed_sequence preprocessing
        emb = embed_residues(model, seq)                  # [L, 1152], bos/eos stripped
        assert emb.shape[0] == len(seq) and emb.shape[1] == DIM, (id_, emb.shape, len(seq))
        assert torch.isfinite(emb).all(), f"non-finite embedding for {id_}"
        torch.save(emb, os.path.join(EMB, id_ + ".pt"))
        if n % 50 == 0 or n == len(todo):
            el = time.time() - t0
            log(f"  {n}/{len(todo)} done | {el/n:.2f}s/seq | ETA {(len(todo)-n)*el/n/60:.0f} min")
    log(f"DONE: {len(todo)} embeddings in {(time.time()-t0)/60:.1f} min -> {EMB}")


if __name__ == "__main__":
    main()
