#!/usr/bin/env python3
"""WT-identity + FoldX-join tests for build_multipoint.py / merge_foldx_multipoint.py.

Run:  ./.venv/bin/python experiments/full_skempi_seqonly/test_multipoint.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_multipoint as bm  # noqa: E402
import merge_foldx_multipoint as mg  # noqa: E402


def main():
    rows, fasta, raw_of, drops = bm.build(verbose=False)

    # 1. Every sub-mutation of every emitted variant validates against the correct partner
    #    fasta sequence ('A'->wt1/seq1, 'B'->wt2/seq2), exactly as parse_mutations applies it.
    bad = 0
    for w1, w2, mut, _ in rows:
        for sm in mut.split(","):
            wt, chain, pos, mt = sm[0], sm[1], int(sm[2:-1]), sm[-1]
            assert chain in ("A", "B"), f"chain must be A/B: {sm}"
            seq = fasta[w1] if chain == "A" else fasta[w2]
            if seq[pos - 1] != wt:
                bad += 1
    assert bad == 0, f"{bad} sub-muts fail WT identity"

    # 2. Every row is multi-point (>=2 sub-muts).
    assert all("," in mut for _, _, mut, _ in rows), "a single-point row leaked into multi_point"

    # 3. Unique (pdb, mutation) join key (rescore harness requires it).
    keys = [(w1.split(".")[0], w1.rsplit("_", 1)[0], mut) for w1, _, mut, _ in rows]
    assert len(keys) == len(set(keys)), "duplicate (complex, mutation) join key"

    # 4. Both partner sides exercised across the set.
    sides = set()
    for _, _, mut, _ in rows:
        sides |= {sm[1] for sm in mut.split(",")}
    assert sides == {"A", "B"}, f"expected both A and B sides, got {sides}"

    # 5. FoldX merge: every non-quarantined row joins to a real FoldX variant; quarantined
    #    rows get an all-zero vector; the FoldX signal tracks truth (Pearson > 0.3).
    fx = mg.load_foldx(mg.RESULTS)
    join = mg.load_join(mg.KEY)
    exclude = mg.load_exclude(mg.EXCL)
    assert exclude, "exclude sidecar empty — run audit_multipoint_foldx.py first"
    tsv_rows = mg.read_tsv(mg.MULTI_TSV)
    cov = quar = unmapped = 0
    ie, ddg = [], []
    for r in tsv_rows:
        if mg.cid_of(r) in exclude:
            quar += 1
            assert mg.foldx_for(r, join, fx, exclude) is None, "quarantined row got a FoldX value"
            continue
        v = mg.foldx_for(r, join, fx, exclude)
        if v is None:
            unmapped += 1
        else:
            cov += 1
            ie.append(v[0]); ddg.append(float(r[3]))
    # non-quarantined coverage must be ~complete (the FoldX run was ok(N/N))
    assert unmapped == 0, f"{unmapped} non-quarantined rows have no FoldX value"
    r_pearson = mg._pearson(ie, ddg)
    assert r_pearson > 0.3, f"FoldX vs ddG Pearson {r_pearson:.3f} too low — join likely broken"

    # 6. Quarantine matches the two known multi-grouping off-groupings.
    assert exclude == {"2C5D.AB.CD", "3SE3.B.C"}, f"unexpected quarantine set: {exclude}"

    print(f"OK  rows={len(rows)}  fasta_labels={len(fasta)}  sides={sides}")
    print(f"    FoldX covered={cov}  quarantined={quar}  unmapped={unmapped}  "
          f"Pearson(FoldX IE, ddG)={r_pearson:.3f}")
    print(f"    unique join keys: {len(keys)}  |  quarantine: {sorted(exclude)}")


if __name__ == "__main__":
    main()
