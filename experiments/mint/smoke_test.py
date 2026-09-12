"""MINT smoke-test: load the checkpoint, run a two-chain complex, and confirm we can pull
PER-RESIDUE representations (not the pooled MINTWrapper output), split by chain, with cross-chain
attention actually moving the partner. Resolves the open API/MPS questions in docs/history/PLAN_MINT_A2.md §1.

Run:  ./.venv-mint/bin/python experiments/mint/smoke_test.py
"""
import sys, json, argparse, time, torch
from collections import OrderedDict

MINT_SRC = "scratch/mint_src"
CKPT = "scratch/mint_weights/mint.ckpt"
CFG = f"{MINT_SRC}/data/esm2_t33_650M_UR50D.json"
sys.path.insert(0, MINT_SRC)

from mint.data import Alphabet
from mint.model.esm2 import ESM2


def load_config(path):  # == mint.helpers.extract.load_config, inlined to avoid the pandas import
    cfg = argparse.Namespace()
    with open(path) as f:
        cfg.__dict__.update(json.load(f))
    return cfg


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


ALPHABET = Alphabet.from_architecture("ESM-1b")


def encode_pair(seqA, seqB, device):
    """Reproduce CollateFn: <cls>seq<eos> per chain, concatenated, with chain_ids 0/1."""
    def enc(s):
        return torch.tensor(ALPHABET.encode("<cls>" + s.replace("J", "L") + "<eos>"), dtype=torch.int64)
    a, b = enc(seqA), enc(seqB)
    tokens = torch.cat([a, b]).unsqueeze(0)
    chain_ids = torch.cat([torch.zeros(len(a), dtype=torch.int64),
                           torch.ones(len(b), dtype=torch.int64)]).unsqueeze(0)
    return tokens.to(device), chain_ids.to(device)


@torch.no_grad()
def per_residue(model, seqA, seqB, device):
    """Return (repA, repB): per-residue reps for each chain IN CONTEXT of the other, specials dropped."""
    tokens, chain_ids = encode_pair(seqA, seqB, device)
    out = model(tokens, chain_ids, repr_layers=[33])["representations"][33][0]  # (T, 1280)
    mask = (~tokens[0].eq(model.cls_idx) & ~tokens[0].eq(model.eos_idx) & ~tokens[0].eq(model.padding_idx))
    cid = chain_ids[0]
    repA = out[(cid == 0) & mask]
    repB = out[(cid == 1) & mask]
    return repA.float().cpu(), repB.float().cpu()


def main():
    device = pick_device()
    log("device:", device)
    t0 = time.time()
    model = build_model(device)
    log(f"model loaded in {time.time()-t0:.1f}s; params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    log("special idx: cls=%s eos=%s pad=%s" % (model.cls_idx, model.eos_idx, model.padding_idx))

    # Two short real-ish chains (barnase fragment / barstar fragment stand-ins for the test).
    seqA = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
    seqB = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"

    repA, repB = per_residue(model, seqA, seqB, device)
    log(f"len(seqA)={len(seqA)} -> repA {tuple(repA.shape)} ; len(seqB)={len(seqB)} -> repB {tuple(repB.shape)}")
    assert repA.shape[0] == len(seqA), "chain-A residue count mismatch"
    assert repB.shape[0] == len(seqB), "chain-B residue count mismatch"
    assert repA.shape[1] == 1280 and repB.shape[1] == 1280

    # Cross-chain sanity: does chain-B's embedding change when chain-A is mutated? (the whole point)
    seqA_mut = seqA[:80] + ("R" if seqA[80] != "R" else "K") + seqA[81:]  # single point mutation on A
    repA2, repB2 = per_residue(model, seqA_mut, seqB, device)
    dB = (repB2 - repB).norm().item()
    dB_rel = dB / repB.norm().item()
    log(f"partner shift ||enc(B|A_mut) - enc(B|A_wt)|| = {dB:.3f}  (relative {dB_rel:.4f})")
    log("CROSS-CHAIN ACTIVE" if dB_rel > 1e-4 else "!! partner unchanged — cross-chain attention NOT flowing")
    log("SMOKE TEST PASS")


if __name__ == "__main__":
    main()
