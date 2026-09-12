"""Build MuLAN-format tables + a shared WT fasta for the standard SKEMPI single-mutation
benchmarks S1131, S4169 (downloaded from GeoPPI) and S2003 (derived locally).

All three are restricted to **single-chain-per-partner** complexes — MuLAN takes one
sequence per partner, the same restriction build_wt_fasta applied for S1102. Mutations are
converted to MuLAN format `<wt><role><pos><mut>` (role A=partner-1, B=partner-2), and each
mutation's WT residue is validated against the residue extracted from the on-disk SKEMPI
PDB (mismatches dropped). ΔΔG convention follows GeoPPI: ΔΔG = -RT·ln(Kd_mut/Kd_wt).

  S1131 / S4169 : parsed from scratch/benchmarks/raw/{S1131,S4169}.csv (GeoPPI).
  S2003         : single NON-alanine SKEMPI 2.0 mutations (the "~2000 non-Ala singles"
                  subset; see RESULTS), deduplicated (mean ΔΔG over repeat measurements).

Outputs: scratch/benchmarks/<ds>/<ds>.tsv  and shared scratch/benchmarks/wt_sequences.fasta

Usage (repo root, mulan venv):
    python experiments/build_benchmarks.py
"""
import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_wt_fasta import extract_chain_sequence  # noqa: E402

PDB_DIR = "scratch/skempi2/PDBs"
OUT = "scratch/benchmarks"
R = 1.987204e-3  # kcal/mol/K

_seq_cache = {}


def chain_seq(pdbid, chain):
    key = (pdbid, chain)
    if key not in _seq_cache:
        path = os.path.join(PDB_DIR, f"{pdbid}.pdb")
        _seq_cache[key] = extract_chain_sequence(path, chain) if os.path.exists(path) else None
    return _seq_cache[key]


def emit(pdbid, pa, pb, wt, chain, pos, mut, ddg, rows, fasta, stats):
    """Validate one mutation and append a MuLAN row; returns updated counters via stats."""
    if chain == pa:
        role, rseq = "A", chain_seq(pdbid, pa)
    elif chain == pb:
        role, rseq = "B", chain_seq(pdbid, pb)
    else:
        stats["bad_chain"] += 1; return
    seqA, seqB = chain_seq(pdbid, pa), chain_seq(pdbid, pb)
    if seqA is None or seqB is None:
        stats["no_struct"] += 1; return
    if not (1 <= pos <= len(rseq)) or rseq[pos - 1] != wt:
        stats["wt_mismatch"] += 1; return
    s1, s2 = f"{pdbid}_A", f"{pdbid}_B"
    fasta[s1] = seqA; fasta[s2] = seqB
    rows.append((s1, s2, f"{wt}{role}{pos}{mut}", ddg))
    stats["kept"] += 1


def build_geoppi(name, fasta):
    rows, stats = [], defaultdict(int)
    with open(f"{OUT}/raw/{name}.csv") as f:
        for r in csv.DictReader(f):
            pa, pb = r["Partners(A_B)"].split("_")
            if len(pa) != 1 or len(pb) != 1:
                stats["multichain"] += 1; continue
            chain, mut = r["mutation"].split(":")        # e.g. "I", "L38D"
            wt, pos, mu = mut[0], int(mut[1:-1]), mut[-1]
            emit(r["protein"], pa, pb, wt, chain, pos, mu, float(r["DDG"]), rows, fasta, stats)
    return rows, stats


def build_s2003(fasta):
    # single non-alanine SKEMPI 2.0 mutations, single-chain partners, dedup mean ΔΔG
    agg = defaultdict(list)   # (pdbid,pa,pb,cleaned_mut) -> [ddg,...]
    for r in csv.DictReader(open("scratch/skempi_v2.csv", encoding="utf-8"), delimiter=";"):
        m = r["Mutation(s)_cleaned"].split(",")
        if len(m) != 1 or m[0][-1] == "A":
            continue
        parts = r["#Pdb"].split("_")
        if len(parts) != 3 or len(parts[1]) != 1 or len(parts[2]) != 1:
            continue
        kw, km = r["Affinity_wt_parsed"].strip(), r["Affinity_mut_parsed"].strip()
        if not (kw and km):
            continue
        try:
            kw, km = float(kw), float(km)
        except ValueError:
            continue
        t = r["Temperature"].strip()
        T = float("".join(ch for ch in t[:3] if ch.isdigit()) or 298) if t else 298.0
        T = T if T > 100 else 298.0
        ddg = -R * T * math.log(km / kw)
        agg[(parts[0], parts[1], parts[2], m[0])].append(ddg)
    rows, stats = [], defaultdict(int)
    for (pdbid, pa, pb, cm), ddgs in agg.items():
        wt, chain, pos, mut = cm[0], cm[1], int(cm[2:-1]), cm[-1]
        emit(pdbid, pa, pb, wt, chain, pos, mut, sum(ddgs) / len(ddgs), rows, fasta, stats)
    return rows, stats


def main():
    fasta = {}
    datasets = {"S1131": lambda: build_geoppi("S1131", fasta),
                "S4169": lambda: build_geoppi("S4169", fasta),
                "S2003": lambda: build_s2003(fasta)}
    for name, fn in datasets.items():
        rows, stats = fn()
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        with open(f"{OUT}/{name}/{name}.tsv", "w") as f:
            for s1, s2, mut, ddg in rows:
                f.write(f"{s1}\t{s2}\t{mut}\t{ddg:.6f}\n")
        complexes = len({s1.rsplit('_', 1)[0] for s1, _, _, _ in rows})
        print(f"[{name}] kept {len(rows)} muts / {complexes} complexes | "
              f"dropped: {dict(stats)}", flush=True)

    with open(f"{OUT}/wt_sequences.fasta", "w") as f:
        for label, seq in fasta.items():
            f.write(f">{label}\n{seq}\n")
    print(f"[fasta] {len(fasta)} unique WT chains -> {OUT}/wt_sequences.fasta", flush=True)


if __name__ == "__main__":
    main()
