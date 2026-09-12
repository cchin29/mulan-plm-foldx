#!/usr/bin/env python3
"""Build the complete label-sequence fasta for the CATH split (WT + mutant sequences) so the GPU
box can GENERATE the missing full-SKEMPI embeddings (esmc600m / ankh3_large / ankh3_xl) — those
dirs are empty on the Mac, so nothing can be transferred; the embedder needs the sequences.

Reproduces mulan/data.py exactly:
  * row -> MutatedComplex(col0, col1, tuple(col2.split(",")))                    (data.py:179)
  * mut_seqN = utils.parse_mutations(mutations, seq1, seq2)                      (data.py:213)
  * mut_seq1_label = f"{col0}_{'-'.join(m for m in mutations if m[1]=='A')}"     (data.py:215)
  * mut_seq2_label = f"{col1}_{'-'.join(m for m in mutations if m[1]=='B')}"     (data.py:218)
parse_mutations is inlined verbatim (incl. the non-A/B guard) to avoid importing transformers.

Emits every label MuLAN will look up (WT: col0/col1; mutant: the two derived labels) -> one seq
each. One fasta serves all PLMs (sequence-only). Verified to equal the 6392-label set the retrain
needs (matches the fully-populated saprot dir 1:1).

Run:  .venv/bin/python experiments/full_skempi_seqonly/build_cath_label_fasta.py
Out:  scratch/skempi_full/all_labels_cath.fasta
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "scratch/splits_skempi_full_cath_kfold/fold_0"
WT_FASTA = ROOT / "scratch/skempi_full/wt_sequences_cath_all.fasta"
OUT = ROOT / "scratch/skempi_full/all_labels_cath.fasta"


def parse_mutations(mutations, seq1, seq2):
    """Verbatim from mulan/utils.py — apply <wt><chain A|B><pos><mut> to seq1/seq2 (1-indexed)."""
    seq1, seq2 = list(seq1), list(seq2)
    for m in mutations:
        chain = m[1]
        if chain == "A":
            seq1[int(m[2:-1]) - 1] = m[-1]
        elif chain == "B":
            seq2[int(m[2:-1]) - 1] = m[-1]
        else:
            raise ValueError(f"Unexpected chain code {chain!r} in mutation {m!r}; expected 'A'/'B'.")
    return "".join(seq1), "".join(seq2)


def read_fasta(p):
    d, k = {}, None
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"):
            k = line[1:]
            d[k] = ""
        elif k is not None:
            d[k] += line
    return d


def main():
    seqs = read_fasta(WT_FASTA)                 # WT label -> sequence
    out = dict(seqs)                            # start with all WT labels
    rows = errs = 0
    for f in ("skempi_all_train.tsv", "skempi_all_val.tsv", "skempi_all_test.tsv"):
        for line in open(SPLIT / f):
            line = line.rstrip("\n")
            if not line:
                continue
            rows += 1
            c0, c1, c2 = line.split("\t")[:3]
            muts = tuple(c2.split(","))
            if c0 not in seqs or c1 not in seqs:
                errs += 1
                print(f"[warn] missing WT seq for {c0} or {c1}", file=sys.stderr)
                continue
            try:
                ms1, ms2 = parse_mutations(muts, seqs[c0], seqs[c1])
            except ValueError as e:
                errs += 1
                print(f"[warn] {e}", file=sys.stderr)
                continue
            l1 = f"{c0}_{'-'.join(m for m in muts if m[1] == 'A')}"
            l2 = f"{c1}_{'-'.join(m for m in muts if m[1] == 'B')}"
            out[l1] = ms1
            out[l2] = ms2

    with open(OUT, "w") as fh:
        for k, v in out.items():
            fh.write(f">{k}\n{v}\n")

    # sanity: equals the required-label set (same derivation the coverage check used)?
    print(f"rows {rows} | errors {errs} | distinct labels written {len(out)}  -> {OUT}")
    sap = ROOT / "scratch/embeddings_skempi_full/saprot"
    if sap.is_dir():
        have = {p.stem for p in sap.glob("*.pt")}
        missing_vs_sap = [k for k in out if k not in have]
        print(f"[check] labels also present in populated saprot dir: "
              f"{len(out) - len(missing_vs_sap)}/{len(out)}  (missing {len(missing_vs_sap)})")
        if missing_vs_sap:
            print("  sample not-in-saprot:", missing_vs_sap[:6])


if __name__ == "__main__":
    main()
