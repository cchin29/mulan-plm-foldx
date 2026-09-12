#!/usr/bin/env python3
"""Build the CATH-superfamily hold-out split (USP-ddG / CATH-ddG's exact 813-mut test set) as a
MuLAN split, by joining our full-SKEMPI training rows to USP-ddG's shipped `cath_fold` labels.

USP-ddG ships the CATH-superfamily partition as a static column (`cath_fold` in {train,val}) in
`data/SKEMPI2/skempi_v2.csv` — 813 val rows over 53 complexes. Crucially the partition is by CATH
SUPERFAMILY, so `cath_fold` is CONSTANT per complex (verified: 0 complexes split across train/val).
We therefore assign by COMPLEX (PDB) — exact and robust, no fragile mutation/chain matching (our
build canonicalizes chains E/I->A/B while USP keeps originals). This reproduces the *literal*
frontier hold-out (the 53 val complexes are USP's exact test superfamilies).

Output (mirrors the clustered split layout so run_bycomplex.sh drives it unchanged, num_folds=1):
    data/splits/splits_skempi_full_cath_kfold/fold_0/skempi_all_{train,val,test}.tsv   (4-col)
    data/splits/splits_skempi_full_cath_kfold/test_single.tsv  test_multiple.tsv       (test subsets)
    data/splits/splits_skempi_full_cath_kfold/JOIN_REPORT.txt                          (coverage audit)

val = 10% of train complexes (seeded), held out for early stopping (never overlaps test).

**This split is not redistributed with the repository**, because the partition it joins onto is not
ours -- see NOTICE and data/splits/splits_skempi_full_cath_kfold/README.md. Running this script is
how you obtain it. The `cath_fold` column comes from USP-ddG's own SKEMPI CSV, which you download:

    git clone https://github.com/ak422/USP-ddG /path/to/USP-ddG
    python experiments/full_skempi_seqonly/build_split_cath.py \
        --usp /path/to/USP-ddG/data/SKEMPI2/skempi_v2.csv

The output is byte-reproducible. Check it against the shipped manifest, which is what every CATH
number in docs/RESULTS.md was computed on:

    cd data/splits/splits_skempi_full_cath_kfold && shasum -a 256 -c MANIFEST.sha256
"""
import argparse, csv, random, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
SP = ROOT / "data/skempi_full/single_point.tsv"
MP = ROOT / "data/skempi_full/multi_point.tsv"
OUT = ROOT / "data/splits/splits_skempi_full_cath_kfold"
USP = None  # set from --usp; there is no default, because the file is not ours to ship
VAL_FRAC = 0.10
SEED = 2024


def pdb_of(wt_label):
    return wt_label.split(".")[0].upper()


def load_usp_by_complex():
    """PDB -> cath_fold label (constant per complex; verified 0 splits). Also count raw val rows."""
    fold = {}
    raw_val = 0
    for r in csv.DictReader(open(USP)):
        cf = (r.get("cath_fold") or "").strip().lower()
        if cf not in ("train", "val"):
            continue
        pdb = (r.get("complex") or "").strip().upper().split("_")[0]
        if not pdb:
            continue
        fold[pdb] = cf  # constant per complex
        if cf == "val":
            raw_val += 1
    return fold, raw_val


def load_ours(path):
    """yield (raw_line, pdb, num_muts)."""
    rows = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        # multi-point muts are comma-separated (single_point has one token); count either way
        nm = len([m for m in parts[2].replace("-", ",").split(",") if m.strip()])
        rows.append((line, pdb_of(parts[0]), nm))
    return rows


def main():
    fold_of, raw_val_usp = load_usp_by_complex()
    val_cplx_usp = {p for p, cf in fold_of.items() if cf == "val"}
    train_cplx_usp = {p for p, cf in fold_of.items() if cf == "train"}

    # Fail before writing anything. Plain SKEMPI 2.0 ships a file with the same name and path shape
    # (data/SKEMPI2/skempi_v2.csv) but no cath_fold column, so pointing --usp at it is the obvious
    # mistake; without this guard every row falls through to `unmatched`, five empty files are
    # written over whatever was there, and the only diagnostic is a ZeroDivisionError from the
    # coverage line further down.
    if not fold_of:
        sys.exit(
            f"no cath_fold labels found in {USP}\n"
            "This must be USP-ddG's own SKEMPI CSV, which adds a `cath_fold` column of "
            "train/val labels;\nplain SKEMPI 2.0 has the same filename and no such column. "
            "Clone https://github.com/ak422/USP-ddG\nand point --usp at its "
            "data/SKEMPI2/skempi_v2.csv."
        )

    ours = load_ours(SP) + load_ours(MP)
    train_rows, test_rows, unmatched = [], [], []
    test_cplx, unmatched_cplx = set(), set()
    for line, pdb, nm in ours:
        cf = fold_of.get(pdb)
        if cf == "train":
            train_rows.append((line, nm))
        elif cf == "val":
            test_rows.append((line, nm))
            test_cplx.add(pdb)
        else:
            unmatched.append((line, pdb, nm))
            unmatched_cplx.add(pdb)
    n_train_usp = len(train_cplx_usp)
    n_val_usp = raw_val_usp
    if not test_rows or not train_rows:
        sys.exit(
            f"cath_fold was read ({n_train_usp} train / {len(val_cplx_usp)} val complexes) but "
            f"joined {len(train_rows)} train and {len(test_rows)} test rows against "
            f"{len(ours)} of ours.\nThe join is by PDB code; a zero here means the two sides do "
            "not share an identifier convention. Nothing was written."
        )

    # val carve-out by complex (10%), seeded, from train
    by_cplx = defaultdict(list)
    for line, nm in train_rows:
        by_cplx[pdb_of(line.split("\t")[0])].append((line, nm))
    cplxs = sorted(by_cplx)
    random.Random(SEED).shuffle(cplxs)
    n_val_cplx = max(1, int(len(cplxs) * VAL_FRAC))
    val_cplxs = set(cplxs[:n_val_cplx])
    # Both lists walk `cplxs`, the seeded-shuffle order. Iterating `val_cplxs` directly would order
    # the val rows by set iteration, which varies with PYTHONHASHSEED -- so the same seed produced a
    # byte-different val file run to run, and MANIFEST.sha256 could not be met. The row *set* was
    # never affected, only its order; the runs under results/ predate this fix.
    tr_final = [(l, nm) for c in cplxs if c not in val_cplxs for (l, nm) in by_cplx[c]]
    val_final = [(l, nm) for c in cplxs if c in val_cplxs for (l, nm) in by_cplx[c]]

    fold = OUT / "fold_0"
    fold.mkdir(parents=True, exist_ok=True)

    def write(path, rows):
        path.write_text("\n".join(l for l, _ in rows) + ("\n" if rows else ""))

    write(fold / "skempi_all_train.tsv", tr_final)
    write(fold / "skempi_all_val.tsv", val_final)
    write(fold / "skempi_all_test.tsv", test_rows)
    write(OUT / "test_single.tsv", [(l, nm) for l, nm in test_rows if nm == 1])
    write(OUT / "test_multiple.tsv", [(l, nm) for l, nm in test_rows if nm > 1])

    ts_single = sum(1 for _, nm in test_rows if nm == 1)
    ts_multi = sum(1 for _, nm in test_rows if nm > 1)

    report = [
        "CATH-superfamily hold-out join report (per-complex assignment; cath_fold constant/complex)",
        f"  USP-ddG: {n_train_usp} train complexes / {len(val_cplx_usp)} val complexes / {n_val_usp} raw val rows",
        f"  our rows total: {len(ours)}  (single_point + multi_point)",
        "",
        f"  TRAIN rows: {len(train_rows)}  ->  train {len(tr_final)} / val {len(val_final)} "
        f"(val = {n_val_cplx}/{len(cplxs)} complexes, seed {SEED})",
        f"  TEST rows (=cath val): {len(test_rows)}   [single {ts_single} / multiple {ts_multi}]  "
        f"over {len(test_cplx)}/{len(val_cplx_usp)} val complexes",
        f"     row coverage vs USP's {n_val_usp}: {100*len(test_rows)/n_val_usp:.1f}%  "
        f"(gap = curation + {sorted(val_cplx_usp - test_cplx)} not embedded)",
        f"  DROPPED our rows (PDB not in USP cath_fold): {len(unmatched)}  "
        f"[PDBs: {sorted(unmatched_cplx)[:12]}]",
    ]
    txt = "\n".join(report)
    (OUT / "JOIN_REPORT.txt").write_text(txt + "\n")
    print(txt)
    print(f"\nwrote split -> {OUT}")


def cli():
    global USP, OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--usp", required=True, type=Path,
                    help="path to USP-ddG's data/SKEMPI2/skempi_v2.csv (clone "
                         "https://github.com/ak422/USP-ddG; not redistributed here)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"output split dir (default {OUT})")
    a = ap.parse_args()
    if not a.usp.is_file():
        sys.exit(f"USP-ddG CSV not found: {a.usp}\n"
                 "This file carries the cath_fold column the split is built from and is not\n"
                 "redistributed with this repository. See "
                 "data/splits/splits_skempi_full_cath_kfold/README.md")
    for p in (SP, MP):
        if not p.is_file():
            sys.exit(f"missing shipped input: {p}")
    USP, OUT = a.usp, a.out
    main()


if __name__ == "__main__":
    cli()
