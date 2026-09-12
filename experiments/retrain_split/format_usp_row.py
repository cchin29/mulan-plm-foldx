#!/usr/bin/env python3
"""Format MuLAN retrain results as a USP-ddG Table-1-style row.

Maps the `evaluate.py` results CSV (results_bycomplex.csv / results_clustered.csv, or the
full-SKEMPI equivalent) onto the column layout of USP-ddG Table 1 / CATH-ddG, so a MuLAN row
drops straight into the frontier leaderboard.

USP-ddG Table 1 columns:
    Method | Mutations | Overall PearsonR SpearmanR RMSE MAE AUROC | Per-PPI PearsonR SpearmanR

Column mapping (MuLAN results.csv  ->  USP-ddG Table 1):
    pearson         -> Overall PearsonR
    spearman        -> Overall SpearmanR
    rmse            -> RMSE
    mae             -> MAE
    auroc_destab    -> AUROC        (sign-of-effect / destabilizing classification == USP-ddG's AUROC)
    ps_pearson_T10  -> Per-PPI PearsonR   (per-structure, complexes with >=10 muts)
    ps_spearman_T10 -> Per-PPI SpearmanR

CAVEATS baked into the output:
  * S1102 is SINGLE-point only -> only the "single" mutation row is fillable; "all"/"multiple"
    need the full-SKEMPI seq-only build.
  * MuLAN per-structure uses T>=10 (24 qualifying complexes on S1102). USP-ddG's per-PPI threshold
    may differ -> per-PPI columns are close-but-not-identical definitions; flagged, don't over-read.
  * Only the CLUSTERED / CATH-superfamily split is frontier-comparable. by-complex rows are the
    leaky tier (USP-ddG: 88.7% "easy") -> printed with a [LEAKY] tag, not for the leaderboard.

Usage:
    python3 format_usp_row.py results_bycomplex.csv saprot_foldx_scalar --split bycomplex
    python3 format_usp_row.py results_clustered.csv saprot_foldx_scalar --split clustered --mut single
    python3 format_usp_row.py results_bycomplex.csv --all           # every arm
"""
import argparse, csv, sys, math

MAP = {  # USP-ddG label -> results.csv column
    "Overall PearsonR":  "pearson",
    "Overall SpearmanR": "spearman",
    "RMSE":              "rmse",
    "MAE":               "mae",
    "AUROC":             "auroc_destab",
    "Per-PPI PearsonR":  "ps_pearson_T10",
    "Per-PPI SpearmanR": "ps_spearman_T10",
}
SPLIT_TAG = {
    "bycomplex": "[LEAKY by-complex — NOT frontier-comparable; USP-ddG: 88.7% easy]",
    "clustered": "[CD-HIT <=60% — honest]",
    "cath":      "[CATH-superfamily hold-out — frontier-comparable]",
}

def load(path):
    with open(path) as fh:
        return {r["arm"]: r for r in csv.DictReader(fh)}

def fmt(v):
    try:
        f = float(v)
        return "  TBD " if math.isnan(f) else f"{f:6.4f}"
    except (ValueError, TypeError):
        return "  TBD "

def row(rec, arm, mut):
    cells = " ".join(fmt(rec.get(col, "nan")) for col in MAP.values())
    return f"| MuLAN·{arm:<22} | {mut:<8} | {cells} |"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("arm", nargs="?", help="arm name, e.g. saprot_foldx_scalar")
    ap.add_argument("--split", default="clustered", choices=SPLIT_TAG)
    ap.add_argument("--mut", default="single", choices=["all", "single", "multiple"],
                    help="S1102 is single-only; use all/multiple only on full-SKEMPI results")
    ap.add_argument("--all", action="store_true", help="emit a row for every arm")
    a = ap.parse_args()

    data = load(a.csv)
    if not data:
        print(f"[empty] {a.csv} has no rows — retrain has not written results yet. "
              f"Re-run evaluate.py once folds land.", file=sys.stderr)

    hdr = ("| Method                       | Muts     | "
           + " ".join(f"{k.split()[-1]:>6}" for k in MAP) + " |")
    sub = ("|                              |          | "
           "OvrP   OvrS   RMSE   MAE    AUROC  ppP    ppS    |")
    print(f"\n### MuLAN — USP-ddG Table 1 format  {SPLIT_TAG[a.split]}\n")
    print(hdr); print(sub.replace(" ", " "))
    print("|" + "-"*30 + "|" + "-"*10 + "|" + "-"*54 + "|")

    arms = sorted(data) if a.all else ([a.arm] if a.arm else [])
    if not arms:
        print("| (no arm given and CSV empty — nothing to format)" + " "*44 + "|")
    for arm in arms:
        if arm not in data:
            tbds = " ".join(["  TBD "] * 7)
            print(f"| MuLAN·{arm:<22} | {a.mut:<8} | {tbds} |   <- pending (arm not in CSV yet)")
            continue
        print(row(data[arm], arm, a.mut))
    print("\nOverall P/S = pooled; ppP/ppS = per-structure (T>=10). "
          "Compare ppP/ppS to USP-ddG's Per-PPI columns; Overall to its Overall.\n")

if __name__ == "__main__":
    main()
