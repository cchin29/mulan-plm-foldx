"""Tier-1 data augmentation for MuLAN ΔΔG training tables.

Produces a physically-grounded augmented *training* set (validation/test are left
untouched, so evaluation stays on real forward mutations):

  1. Reverse-mutation (antisymmetry): ΔΔG(wt→mut) = −ΔΔG(mut→wt). For each training
     mutation, add the reverse example with the label negated. Implemented at the
     table level by adding the mutated chain as a named FASTA entry whose label
     matches MuLAN's internal mutant label (so its embedding is already cached),
     and a row that mutates it back to wild type.
  2. Identity anchors: wt→wt has ΔΔG = 0. One per complex, at the first standard
     residue, anchors the regression's zero point.

Usage (from repo root):
    python experiments/augment.py \
        --train scratch/.../splits/S1102_filtered_train.tsv \
        --fasta scratch/.../wt_sequences.fasta \
        --out-train scratch/.../aug_train.tsv \
        --out-fasta scratch/.../aug.fasta
"""
import argparse

from mulan.utils import parse_fasta, dict_to_fasta

STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-fasta", required=True)
    ap.add_argument("--no-identity", action="store_true", help="skip identity (ΔΔG=0) anchors")
    args = ap.parse_args()

    fasta = dict(parse_fasta(args.fasta))
    aug_fasta = dict(fasta)
    rows = []          # original + reverse rows
    n_rev = 0
    skipped_multi = 0

    with open(args.train) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            s1, s2, muts, score = parts[0], parts[1], parts[2], float(parts[3])
            rows.append((s1, s2, muts, f"{score}"))         # keep forward example verbatim
            mut_list = muts.split(",")
            if len(mut_list) != 1:                            # only single mutations reversed here
                skipped_multi += 1
                continue
            m = mut_list[0]
            wt_aa, chain, pos, mut_aa = m[0], m[1], int(m[2:-1]), m[-1]
            rev_m = f"{mut_aa}{chain}{pos}{wt_aa}"
            if chain == "A":
                wtseq = fasta[s1]
                mutseq = wtseq[:pos - 1] + mut_aa + wtseq[pos:]
                new_label = f"{s1}_{m}"                        # == MuLAN's internal mutant label
                aug_fasta[new_label] = mutseq
                rows.append((new_label, s2, rev_m, f"{-score}"))
            else:
                wtseq = fasta[s2]
                mutseq = wtseq[:pos - 1] + mut_aa + wtseq[pos:]
                new_label = f"{s2}_{m}"
                aug_fasta[new_label] = mutseq
                rows.append((s1, new_label, rev_m, f"{-score}"))
            n_rev += 1

    n_ident = 0
    if not args.no_identity:
        seen = set()
        for s1, s2, muts, score in [(r[0], r[1], r[2], r[3]) for r in list(rows)]:
            # only anchor on original complexes (skip the reverse rows' synthetic labels)
            if s1 not in fasta or s2 not in fasta or (s1, s2) in seen:
                continue
            seen.add((s1, s2))
            seq1 = fasta[s1]
            p = next((i for i, a in enumerate(seq1) if a in STANDARD), None)
            if p is None:
                continue
            aa = seq1[p]
            ident_m = f"{aa}A{p + 1}{aa}"                      # wt->wt, no-op mutation
            rows.append((s1, s2, ident_m, "0.0"))
            n_ident += 1

    with open(args.out_train, "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    dict_to_fasta(aug_fasta, args.out_fasta)

    print(f"forward rows kept, +{n_rev} reverse, +{n_ident} identity "
          f"(skipped {skipped_multi} multi-mut). total train rows: {len(rows)}")
    print(f"fasta entries: {len(fasta)} -> {len(aug_fasta)}")
    print(f"wrote {args.out_train} and {args.out_fasta}")


if __name__ == "__main__":
    main()
