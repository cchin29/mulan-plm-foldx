"""Generate MINT partner-context per-residue embeddings for S1102 (A2 arm).

Unlike the monomer generators, MINT embeds each chain IN CONTEXT of its partner, so the cache is
keyed per (complex, mutation) not per sequence label — this is what preserves the non-canceling
partner term the whole A2 arm depends on (see docs/history/PLAN_MINT_A2.md §2). For a given row:

    WT complex  MINT([wt1, wt2])  -> enc(wt1|wt2), enc(wt2|wt1)   (shared across a complex's rows)
    MUT complex MINT([mut1, mut2]) -> enc(mut1|mut2), enc(mut2|mut1)  (per row)

Cache layout (scratch/embeddings_mint/, gitignored):
    wt/<s1>__<s2>.pt              -> {"1": [La,1280], "2": [Lb,1280]}   memoized per complex
    mut/<s1>__<s2>__<muts>.pt     -> {"1": [La,1280], "2": [Lb,1280]}   one per row

The MuLAN loader's mint_pair path (mulan/data.py) reconstructs [wt1, wt2, mut1, mut2] from these two
bundles. Key helpers are duplicated there and MUST stay in sync — see wt_key/mut_key below.

Runs in .venv-mint (MINT + torch only). All S1102 complexes fit ESM-2's 1024-token window
(max combined = 919), so no truncation. Idempotent (skip-existing). MPS, fp32 output.

Usage (from repo root):
    ./.venv-mint/bin/python experiments/mint/gen_mint_emb.py
Paths env-overridable: MINT_EMB / MINT_WT / MINT_TBL / MINT_CKPT / MINT_CFG / MINT_SRC.
"""
import os, sys, json, argparse, time, torch
from collections import OrderedDict

MINT_SRC = os.environ.get("MINT_SRC", "scratch/mint_src")
CKPT = os.environ.get("MINT_CKPT", "scratch/mint_weights/mint.ckpt")
CFG = os.environ.get("MINT_CFG", f"{MINT_SRC}/data/esm2_t33_650M_UR50D.json")
EMB = os.environ.get("MINT_EMB", "scratch/embeddings_mint")
WT = os.environ.get("MINT_WT", "experiments/embedding_sweep/data/wt_sequences.fasta")
TBL = os.environ.get("MINT_TBL", "experiments/embedding_sweep/data/S1102_filtered.tsv")
DIM = 1280
sys.path.insert(0, MINT_SRC)
from mint.data import Alphabet
from mint.model.esm2 import ESM2

ALPHABET = Alphabet.from_architecture("ESM-1b")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---- cache keys (CANONICAL; mulan/data.py mint_pair path duplicates these — keep in sync) ----
def wt_key(s1, s2):
    return f"{s1}__{s2}"


def mut_key(s1, s2, mutations):
    return f"{s1}__{s2}__" + "-".join(mutations)


# ---- id/sequence parsing (mirrors mulan.utils.parse_fasta / parse_mutations; duplicated because
# this generator runs in .venv-mint, which does not have the mulan package installed) ----
def parse_fasta(path):
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


def parse_mutations(mutations, seq1, seq2):  # behavior matches mulan.utils.parse_mutations
    s1, s2 = list(seq1), list(seq2)
    for m in mutations:
        if m[1] == "A":
            s1[int(m[2:-1]) - 1] = m[-1]
        elif m[1] == "B":
            s2[int(m[2:-1]) - 1] = m[-1]
        else:
            raise ValueError(f"mutation {m!r}: chain must be A or B, got {m[1]!r}")
    return "".join(s1), "".join(s2)


def load_config(path):
    cfg = argparse.Namespace()
    with open(path) as f:
        cfg.__dict__.update(json.load(f))
    return cfg


def build_model(device):
    cfg = load_config(CFG)
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        token_dropout=cfg.token_dropout,
        use_multimer=True,
    )
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = OrderedDict((k.replace("model.", ""), v) for k, v in ckpt["state_dict"].items())
    model.load_state_dict(sd)
    model.eval().to(device)
    return model


def _enc(s):
    return torch.tensor(ALPHABET.encode("<cls>" + s.replace("J", "L") + "<eos>"), dtype=torch.int64)


@torch.no_grad()
def _embed_single(model, seq, device):
    """Per-residue reps for ONE chain forwarded alone (chain_ids all 0, no partner in the window)."""
    tok = _enc(seq).unsqueeze(0).to(device)
    cid = torch.zeros_like(tok)
    out = model(tok, cid, repr_layers=[33])["representations"][33][0]
    mask = (~tok[0].eq(model.cls_idx) & ~tok[0].eq(model.eos_idx) & ~tok[0].eq(model.padding_idx))
    rep = out[mask].float().cpu()
    assert rep.shape[0] == len(seq), f"len mismatch {rep.shape[0]}/{len(seq)}"
    return rep


@torch.no_grad()
def embed_pair(model, seqA, seqB, device, single_chain=False):
    """Per-residue reps for both chains. Default: each in context of the other (partner concatenated).
    single_chain=True: each chain forwarded ALONE through the same MINT backbone — the §8a control that
    holds the backbone fixed and removes partner context, so MINT-mono vs MINT-complex isolates it."""
    if single_chain:
        return {"1": _embed_single(model, seqA, device), "2": _embed_single(model, seqB, device)}
    a, b = _enc(seqA), _enc(seqB)
    tokens = torch.cat([a, b]).unsqueeze(0).to(device)
    cid = torch.cat([torch.zeros(len(a), dtype=torch.int64), torch.ones(len(b), dtype=torch.int64)]).unsqueeze(0).to(device)
    out = model(tokens, cid, repr_layers=[33])["representations"][33][0]  # (T,1280)
    mask = (~tokens[0].eq(model.cls_idx) & ~tokens[0].eq(model.eos_idx) & ~tokens[0].eq(model.padding_idx))
    cid0 = cid[0]
    repA = out[(cid0 == 0) & mask].float().cpu()
    repB = out[(cid0 == 1) & mask].float().cpu()
    assert repA.shape[0] == len(seqA) and repB.shape[0] == len(seqB), \
        f"len mismatch A:{repA.shape[0]}/{len(seqA)} B:{repB.shape[0]}/{len(seqB)}"
    return {"1": repA, "2": repB}


def rows():
    """Yield (s1, s2, mutations_tuple, wt1, wt2, mut1, mut2) for every table row."""
    wt = parse_fasta(WT)
    for line in open(TBL):
        p = line.split()
        if len(p) < 3:
            continue
        s1, s2, muts = p[0], p[1], tuple(p[2].split(","))
        seq1, seq2 = wt[s1], wt[s2]
        mut1, mut2 = parse_mutations(muts, seq1, seq2)
        yield s1, s2, muts, seq1, seq2, mut1, mut2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-chain", action="store_true",
                    help="§8a control: embed each chain alone (no partner context); set MINT_EMB to a "
                         "separate dir (scratch/embeddings_mint_mono) so it does not overwrite the complex cache")
    args = ap.parse_args()
    if args.single_chain and os.path.abspath(EMB) == os.path.abspath("scratch/embeddings_mint"):
        sys.exit("refusing --single-chain into the complex cache scratch/embeddings_mint; "
                 "set MINT_EMB=scratch/embeddings_mint_mono (skip-existing would otherwise no-op).")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(f"{EMB}/wt", exist_ok=True)
    os.makedirs(f"{EMB}/mut", exist_ok=True)
    all_rows = list(rows())
    mode = "single-chain (mono control)" if args.single_chain else "complex (partner context)"
    log(f"device={device}  rows={len(all_rows)}  mode={mode}  EMB={EMB}  loading MINT...")
    model = build_model(device)
    log("MINT loaded.")

    wt_done, mut_done, wt_skip, mut_skip = 0, 0, 0, 0
    t0 = time.time()
    for i, (s1, s2, muts, seq1, seq2, mut1, mut2) in enumerate(all_rows):
        wp = f"{EMB}/wt/{wt_key(s1, s2)}.pt"
        if not os.path.exists(wp):
            torch.save(embed_pair(model, seq1, seq2, device, single_chain=args.single_chain), wp)
            wt_done += 1
        else:
            wt_skip += 1
        mp = f"{EMB}/mut/{mut_key(s1, s2, muts)}.pt"
        if not os.path.exists(mp):
            torch.save(embed_pair(model, mut1, mut2, device, single_chain=args.single_chain), mp)
            mut_done += 1
        else:
            mut_skip += 1
        if (i + 1) % 50 == 0:
            log(f"  {i+1}/{len(all_rows)}  wt(+{wt_done}/skip {wt_skip}) mut(+{mut_done}/skip {mut_skip})  {time.time()-t0:.0f}s")
    log(f"DONE  wt written={wt_done} skipped={wt_skip} ; mut written={mut_done} skipped={mut_skip} ; {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
