"""Wild-type sequence FASTA for the S1102 benchmark, from SKEMPI 2.0's cleaned PDB
files and chain mapping.

Raw RCSB SEQRES does not match SKEMPI's mutation numbering, so sequences are read from
the ATOM records of SKEMPI's own cleaned structures. The two helpers below define the
residue selection — one CA per residue number, alt locs and duplicates dropped — that
`gen_3di.py` and `gen_interface.py` reuse so their per-residue channels stay parallel to
the sequence written here.

Inputs live under `scratch/`, which is not redistributed; see `docs/REPRODUCE.md`.
"""

import csv
import os
from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "scratch"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def load_chain_combos(csv_path, pdb_ids):
    combos = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            pdb_field = row["#Pdb"]
            parts = pdb_field.split("_")
            pdbid = parts[0]
            if pdbid in pdb_ids and pdbid not in combos:
                if len(parts) == 3 and len(parts[1]) == 1 and len(parts[2]) == 1:
                    combos[pdbid] = (parts[1], parts[2])
    return combos


def extract_chain_sequence(pdb_path, chain_id):
    residues = {}
    seen_ca = set()
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            this_chain = line[21].strip()
            if this_chain != chain_id:
                continue
            resname = line[17:20].strip()
            resseq = line[22:26].strip()
            try:
                resnum = int(resseq)
            except ValueError:
                continue
            if resnum in seen_ca:
                continue  # skip alt locs / duplicates
            seen_ca.add(resnum)
            residues[resnum] = THREE_TO_ONE.get(resname, "X")
    if not residues:
        return None
    max_num = max(residues)
    return "".join(residues.get(i, "X") for i in range(1, max_num + 1))


def main():
    scratch = Path(os.environ.get("MULAN_SCRATCH", SCRATCH))
    pdb_ids = [l.strip() for l in open(scratch / "pdb_ids.txt") if l.strip()]
    combos = load_chain_combos(scratch / "skempi_v2.csv", set(pdb_ids))

    fasta_lines = []
    missing = []
    for pdbid in pdb_ids:
        combo = combos.get(pdbid)
        if combo is None:
            missing.append(pdbid)
            continue
        chain_a, chain_b = combo
        pdb_path = scratch / "skempi2/PDBs" / f"{pdbid}.pdb"
        seq_a = extract_chain_sequence(pdb_path, chain_a)
        seq_b = extract_chain_sequence(pdb_path, chain_b)
        if seq_a is None or seq_b is None:
            missing.append(pdbid)
            continue
        fasta_lines.append(f">{pdbid}_A\n{seq_a}\n")
        fasta_lines.append(f">{pdbid}_B\n{seq_b}\n")

    with open(scratch / "wt_sequences.fasta", "w") as f:
        f.writelines(fasta_lines)

    print(f"Built sequences for {len(pdb_ids) - len(missing)}/{len(pdb_ids)} complexes")
    print("Missing/excluded:", missing)

    # Validation against the known-correct sequence embedded in examples/sample_mut.txt.
    known_seq1 = (
        "FPTIPLSRLFDNAMLRAHRLHQLAFDTYQEFEEAYIPKEQKYSFLQNPQTSLCFSESIPTPSNREETQQKSNLELL"
        "RISLLLIQSWLEPVQFLRSVFANSLVYGASDSNVYDLLKDLEERIQTLMGRLEGQIFKQTYSKFDTDALLKNYGLL"
        "YCFRKDMDKVETFLRIVQCRSVEGSCGF"
    )
    seq_a = extract_chain_sequence(scratch / "skempi2/PDBs/1A22.pdb", combos["1A22"][0])
    match = seq_a == known_seq1
    print(f"Validation against examples/sample_mut.txt 1A22 chain A: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print("known:", known_seq1)
        print("built:", seq_a)


if __name__ == "__main__":
    main()
