#!/usr/bin/env python3
"""WT-identity + convention tests for build_skempi_full.py.

Run:  .venv/bin/python experiments/full_skempi_seqonly/test_build_skempi.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_skempi_full as b  # noqa: E402


def main():
    rows, fasta, drops = b.build(verbose=False)

    # 1. Every emitted row's mutation validates against the correct partner fasta sequence
    #    ('A' -> wt1/seq1, 'B' -> wt2/seq2, exactly as mulan/utils.parse_mutations applies it).
    bad = 0
    for w1, w2, mut, _ in rows:
        wt, chain, pos, mt = mut[0], mut[1], int(mut[2:-1]), mut[-1]
        assert chain in ("A", "B"), f"chain must be A/B, got {mut}"
        seq = fasta[w1] if chain == "A" else fasta[w2]
        if seq[pos - 1] != wt:
            bad += 1
    assert bad == 0, f"{bad} rows fail WT identity against fasta"

    # 2. Unique (pdb, mutation) join key within the whole set (rescore harness requires it;
    #    replicate measurements were averaged, opposite-side muts differ by A/B chain letter).
    keys = [(w1.split("_")[0], mut) for w1, w2, mut, _ in rows]
    assert len(keys) == len(set(keys)), "duplicate (pdb, mutation) join key survived"

    # 3. Coverage is frontier-level (>=99% of curated entries kept before replicate-merge).
    kept_ok = sum(drops[k] for k in drops)
    curated = len(b.curated_singlepoint_entries())
    assert (curated - sum(drops.values())) / curated >= 0.99, "WT-validated coverage below 99%"

    # 4. Both partner sides exercised (fixed g1->A / g2->B, so 'B' mutations must appear).
    sides = {mut[1] for _, _, mut, _ in rows}
    assert sides == {"A", "B"}, f"expected both A and B side mutations, got {sides}"

    # 5. Multi-chain group: a second-chain mutation lands past the first chain's length.
    saw_second_chain = False
    for w1, w2, mut, _ in rows:
        chain = mut[1]
        lab = w1 if chain == "A" else w2
        cid, group = lab.split("_", 1)
        if len(group) >= 2:
            code = cid.split(".")[0]
            ch = b.load_mapping(code)
            first_len = len(ch[group[0]]["seq"]) if group[0] in ch else 0
            if int(mut[2:-1]) > first_len:
                saw_second_chain = True
                break
    assert saw_second_chain, "no multi-chain second-chain mutation exercised"

    # 6. ddG label sanity: 1CSE_E_I LI38S ~ +1.19 (matches S1102 / gate).
    cse = [r for r in rows if r[0].startswith("1CSE.") and abs(r[3] - 1.19) < 0.05]
    assert cse, "1CSE +1.19 ddG row missing"

    print(f"OK  rows={len(rows)}  fasta_labels={len(fasta)}  sides={sides}")
    print(f"    unique join keys: {len(keys)}  |  multi-chain offset exercised: {saw_second_chain}")
    print(f"    1CSE +1.19 rows: {len(cse)}  e.g. {cse[0]}")


if __name__ == "__main__":
    main()
