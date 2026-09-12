"""Unit test for the MuLAN mint_pair load path (mulan/data.py). Runs against the 4-row probe
cache (scratch/mint_probe/emb). Verifies:
  1. the dataset assembles [wt1, wt2, mut1, mut2] with correct shapes from the wt/mut bundles;
  2. THE KEY PROPERTY: the partner (non-mutated) chain's embedding differs between wt and mut —
     i.e. the non-canceling cross-chain term the monomer cache would have zeroed is preserved;
  3. an incomplete cache raises a clean FileNotFoundError, not a cryptic load error.

Run:  ./.venv/bin/python experiments/mint/test_mint_pair.py
"""
import os, sys, tempfile, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mulan.data import MulanDataset

EMB = "scratch/mint_probe/emb"
TBL = "scratch/mint_probe/hotspots.tsv"
FASTA = "experiments/embedding_sweep/data/wt_sequences.fasta"


def build(emb=EMB):
    return MulanDataset.from_table(TBL, FASTA, emb, plm_model_name="mint", mint_pair=True)


def test_shapes_and_assembly():
    ds = build()
    assert len(ds) == 4, f"expected 4 rows, got {len(ds)}"
    for i in range(len(ds)):
        emb = ds[i]["inputs_embeds"]
        for t in (emb.seq1, emb.seq2, emb.mut_seq1, emb.mut_seq2):
            assert t.ndim == 2 and t.shape[1] == 1280, f"bad emb shape {tuple(t.shape)}"
        # wt1 and mut1 are the same chain -> same length; ditto wt2/mut2
        assert emb.seq1.shape[0] == emb.mut_seq1.shape[0]
        assert emb.seq2.shape[0] == emb.mut_seq2.shape[0]
    print("  ✓ shapes + [wt1,wt2,mut1,mut2] assembly")


def test_partner_term_preserved():
    """For each single-chain row, the NON-mutated chain must differ wt vs mut (cross-chain term)."""
    ds = build()
    for i in range(len(ds)):
        s1, s2, muts = ds.mutated_complexes[i]
        emb = ds[i]["inputs_embeds"]
        chains = set(m[1] for m in muts)
        if chains == {"A"}:            # partner = chain 2
            wt_p, mut_p = emb.seq2, emb.mut_seq2
        else:                          # chain-B mutation, partner = chain 1
            wt_p, mut_p = emb.seq1, emb.mut_seq1
        assert not torch.allclose(wt_p, mut_p), \
            f"row {i} ({s1} {muts}): partner embedding identical wt vs mut — non-canceling term LOST"
        rel = (mut_p - wt_p).norm().item() / wt_p.norm().item()
        print(f"  ✓ {s1.split('_')[0]} {muts[0]}: partner term preserved (rel shift {rel:.4f})")


def test_incomplete_cache_raises():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "wt")); os.makedirs(os.path.join(d, "mut"))
        try:
            build(emb=d)
        except FileNotFoundError as e:
            assert "mint_pair cache incomplete" in str(e)
            print("  ✓ incomplete cache -> clean FileNotFoundError")
            return
    raise AssertionError("expected FileNotFoundError on empty cache")


if __name__ == "__main__":
    os.chdir(ROOT)
    print("test_shapes_and_assembly");     test_shapes_and_assembly()
    print("test_partner_term_preserved");  test_partner_term_preserved()
    print("test_incomplete_cache_raises"); test_incomplete_cache_raises()
    print("ALL MINT_PAIR TESTS PASS")
