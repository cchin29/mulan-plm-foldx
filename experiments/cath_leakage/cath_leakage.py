#!/usr/bin/env python3
"""CATH-superfamily data-leakage exposure per SKEMPI-derived dataset.

Assigns CATH superfamilies (4-level C.A.T.H) to SKEMPI complex chains via SIFTS,
groups complexes into CATH interface families (shared superfamilies on BOTH
partner sides, either orientation), and reports per-dataset CATH-family exposure.

Network required (EBI/SIFTS egress). Stdlib only.
"""
import csv, gzip, io, json, os, sys, urllib.request
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "experiments", "cath_leakage")
os.makedirs(OUTDIR, exist_ok=True)

BENCH = {
    "S1102": os.path.join(ROOT, "examples", "S1102.tsv"),
    "S1131": os.path.join(ROOT, "scratch", "benchmarks", "S1131", "S1131.tsv"),
    "S2003": os.path.join(ROOT, "scratch", "benchmarks", "S2003", "S2003.tsv"),
    "S4169": os.path.join(ROOT, "scratch", "benchmarks", "S4169", "S4169.tsv"),
}
SKEMPI = os.path.join(ROOT, "scratch", "skempi_v2.csv")
SIFTS_CACHE = os.path.join(OUTDIR, "pdb_chain_cath_uniprot.csv.gz")
SIFTS_URLS = [
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_cath_uniprot.csv.gz",
    "https://www.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_cath_uniprot.csv.gz",
]
# SIFTS pdb_chain_cath_uniprot maps (pdb,chain) -> CATH *domain id* (e.g. 101mA00),
# not the dotted C.A.T.H. cath-domain-list gives domain id -> C,A,T,H columns.
CATHLIST_CACHE = os.path.join(OUTDIR, "cath-domain-list.txt")
CATHLIST_URL = ("http://download.cathdb.info/cath/releases/latest-release/"
                "cath-classification-data/cath-domain-list.txt")
ANTIBODY_SF = "2.60.40.10"  # immunoglobulin-like

# --------------------------------------------------------------------------
# 1. Enumerate complexes + mutation counts for the six datasets.
#    complex key = "PDB_chainsA_chainsB". A dataset maps key -> n_mut.
# --------------------------------------------------------------------------
def parse_bench(path):
    """col0=partnerA 'PDB_chain', col1=partnerB 'PDB_chain'; 1 mut per row."""
    counts = defaultdict(int)
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            a, b = line.split("\t")[:2]
            pdb, ca = a.split("_", 1)
            pdb2, cb = b.split("_", 1)
            assert pdb == pdb2, f"partner PDB mismatch: {a} vs {b}"
            counts[f"{pdb}_{ca}_{cb}"] += 1
    return counts

def parse_skempi(path):
    """#Pdb = PDBcode_chainsA_chainsB; Mutation(s)_cleaned comma-count -> single/multi."""
    single, multi = defaultdict(int), defaultdict(int)
    with open(path) as fh:
        rd = csv.DictReader(fh, delimiter=";")
        for row in rd:
            key = row["#Pdb"].strip()
            if not key:
                continue
            muts = row["Mutation(s)_cleaned"].strip()
            n = len([m for m in muts.split(",") if m])
            (single if n == 1 else multi)[key] += 1
    return single, multi

datasets = {k: parse_bench(v) for k, v in BENCH.items()}
sk_single, sk_multi = parse_skempi(SKEMPI)
datasets["SKEMPI-single"] = sk_single
datasets["SKEMPI-multi"] = sk_multi

# key -> (pdb, [chainsA], [chainsB])
all_complexes = {}
for counts in datasets.values():
    for key in counts:
        if key in all_complexes:
            continue
        parts = key.split("_")
        pdb, ca, cb = parts[0], parts[1], parts[2]
        all_complexes[key] = (pdb.lower(), list(ca), list(cb))

print(f"[1] {len(all_complexes)} distinct complexes across 6 datasets", file=sys.stderr)

# --------------------------------------------------------------------------
# 2. Download SIFTS; build (pdb_lower, chain) -> {4-level CATH superfamily}.
# --------------------------------------------------------------------------
def fetch_sifts():
    if os.path.exists(SIFTS_CACHE) and os.path.getsize(SIFTS_CACHE) > 1000:
        print(f"[2] using cached {SIFTS_CACHE}", file=sys.stderr)
        return open(SIFTS_CACHE, "rb").read()
    last = None
    for url in SIFTS_URLS:
        try:
            print(f"[2] downloading {url}", file=sys.stderr)
            req = urllib.request.Request(url, headers={"User-Agent": "cath-leakage/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            with open(SIFTS_CACHE, "wb") as f:
                f.write(data)
            return data
        except Exception as e:  # noqa
            print(f"    failed: {e}", file=sys.stderr)
            last = e
    raise SystemExit(f"[2] all SIFTS URLs failed: {last}")

raw = fetch_sifts()
text = gzip.decompress(raw).decode("utf-8", "replace")
lines = text.splitlines()
# first line may be a '# date' comment; header is the line with PDB/CHAIN/CATH_ID
hdr_i = 0
for i, ln in enumerate(lines[:5]):
    if "PDB" in ln.upper() and "CHAIN" in ln.upper():
        hdr_i = i
        break
reader = csv.DictReader(lines[hdr_i:])
cols = {c.upper(): c for c in reader.fieldnames}
print(f"[2] SIFTS header: {reader.fieldnames}", file=sys.stderr)
pdb_c = cols.get("PDB")
chain_c = cols.get("CHAIN")
cath_c = cols.get("CATH_ID") or cols.get("CATH")
assert pdb_c and chain_c and cath_c, f"missing columns in {reader.fieldnames}"

# domain id (e.g. 101mA00) -> 4-level C.A.T.H superfamily, from cath-domain-list.txt
if not (os.path.exists(CATHLIST_CACHE) and os.path.getsize(CATHLIST_CACHE) > 1000):
    print(f"[2] downloading {CATHLIST_URL}", file=sys.stderr)
    req = urllib.request.Request(CATHLIST_URL, headers={"User-Agent": "cath-leakage/1.0"})
    ctx = None
    try:
        import ssl
        ctx = ssl.create_default_context()
    except Exception:  # noqa
        ctx = None
    with urllib.request.urlopen(req, timeout=180, context=ctx) as r, open(CATHLIST_CACHE, "wb") as f:
        f.write(r.read())
domain_sf = {}
with open(CATHLIST_CACHE) as fh:
    for ln in fh:
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        if len(p) < 5:
            continue
        domain_sf[p[0]] = ".".join(p[1:5])  # C.A.T.H
print(f"[2] cath-domain-list: {len(domain_sf)} domains", file=sys.stderr)

chain_cath = defaultdict(set)
n_unmapped_dom = 0
for row in reader:
    dom = (row[cath_c] or "").strip()
    sf = domain_sf.get(dom)
    if sf:
        chain_cath[(row[pdb_c].strip().lower(), row[chain_c].strip())].add(sf)
    elif dom:
        n_unmapped_dom += 1
print(f"[2] SIFTS: {len(chain_cath)} (pdb,chain) with >=1 CATH superfamily "
      f"({n_unmapped_dom} SIFTS rows had a domain id absent from cath-domain-list)", file=sys.stderr)

# chain coverage over SKEMPI (all complexes here derive from SKEMPI)
all_pairs = set()
for pdb, cas, cbs in all_complexes.values():
    for ch in cas + cbs:
        all_pairs.add((pdb, ch))
covered_pairs = {p for p in all_pairs if p in chain_cath}
global_cov = 100.0 * len(covered_pairs) / len(all_pairs) if all_pairs else 0.0
print(f"[2] global chain coverage: {len(covered_pairs)}/{len(all_pairs)} = {global_cov:.1f}%", file=sys.stderr)

# --------------------------------------------------------------------------
# 3. Per-complex partner superfamilies; build CATH-family graph over ALL complexes.
#    Edge iff shared superfamilies on BOTH sides in either orientation.
# --------------------------------------------------------------------------
comp_sf = {}  # key -> (frozenset A_sf, frozenset B_sf)
for key, (pdb, cas, cbs) in all_complexes.items():
    a_sf = set().union(*[chain_cath.get((pdb, ch), set()) for ch in cas]) if cas else set()
    b_sf = set().union(*[chain_cath.get((pdb, ch), set()) for ch in cbs]) if cbs else set()
    comp_sf[key] = (frozenset(a_sf), frozenset(b_sf))

def related(k1, k2):
    a1, b1 = comp_sf[k1]
    a2, b2 = comp_sf[k2]
    if not a1 or not b1 or not a2 or not b2:
        return False  # unmapped side -> never edges (singleton)
    same = bool(a1 & a2) and bool(b1 & b2)
    swap = bool(a1 & b2) and bool(b1 & a2)
    return same or swap

# union-find over complexes; index candidate pairs by shared superfamily to avoid O(n^2)
keys = list(all_complexes)
parent = {k: k for k in keys}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

sf_to_keys = defaultdict(set)  # superfamily -> complexes that carry it on either side
for k, (a, b) in comp_sf.items():
    for sf in a | b:
        sf_to_keys[sf].add(k)

checked = set()
for sf, group in sf_to_keys.items():
    g = list(group)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            pair = (g[i], g[j]) if g[i] < g[j] else (g[j], g[i])
            if pair in checked:
                continue
            checked.add(pair)
            if related(g[i], g[j]):
                union(g[i], g[j])

# family id = canonical root; unmapped complexes are their own singleton root already
fam_of = {k: find(k) for k in keys}
# relabel roots to compact ids
roots = sorted(set(fam_of.values()))
fam_id = {r: f"fam_{i:04d}" for i, r in enumerate(roots)}
family_id = {k: fam_id[fam_of[k]] for k in keys}

# unmapped complexes (missing CATH on at least one side => cannot join a family)
unmapped = [k for k in keys if not comp_sf[k][0] or not comp_sf[k][1]]
print(f"[3] {len(roots)} CATH interface families; {len(unmapped)} complexes unmapped "
      f"on >=1 side (each its own singleton family)", file=sys.stderr)

# --------------------------------------------------------------------------
# 4/5. Per-dataset exposure + coverage.
# --------------------------------------------------------------------------
per_dataset = {}
for ds, counts in datasets.items():
    n_mut = sum(counts.values())
    ds_keys = list(counts)
    n_cplx = len(ds_keys)
    # family (restricted to this dataset) with >=2 distinct complexes -> exposed
    fam_members = defaultdict(list)
    for k in ds_keys:
        fam_members[family_id[k]].append(k)
    exposed_keys = {k for fam, ks in fam_members.items() if len(ks) >= 2 for k in ks}
    exposed_mut = sum(counts[k] for k in exposed_keys)
    exposure = 100.0 * exposed_mut / n_mut if n_mut else 0.0
    # chain coverage for this dataset
    pairs = set()
    for k in ds_keys:
        pdb, cas, cbs = all_complexes[k]
        for ch in cas + cbs:
            pairs.add((pdb, ch))
    cov = 100.0 * len({p for p in pairs if p in chain_cath}) / len(pairs) if pairs else 0.0
    # antibody domination: families whose superfamilies are dominated by Ig fold
    ab_fams = []
    for fam, ks in fam_members.items():
        if len(ks) < 2:
            continue
        sfs = set()
        for k in ks:
            sfs |= comp_sf[k][0] | comp_sf[k][1]
        if ANTIBODY_SF in sfs:
            ab_fams.append((fam, len(ks)))
    ab_exposed_mut = sum(
        counts[k] for fam, ks in fam_members.items() if len(ks) >= 2
        for k in ks if any(ANTIBODY_SF in (comp_sf[m][0] | comp_sf[m][1]) for m in ks)
    )
    per_dataset[ds] = {
        "n_mut": n_mut,
        "n_cplx": n_cplx,
        "cath_family_exposure_pct": round(exposure, 1),
        "chain_coverage_pct": round(cov, 1),
        "n_exposed_complexes": len(exposed_keys),
        "n_unmapped_complexes": sum(1 for k in ds_keys if k in set(unmapped)),
        "antibody_family_exposed_mut": ab_exposed_mut,
        "antibody_exposed_pct": round(100.0 * ab_exposed_mut / n_mut, 1) if n_mut else 0.0,
        "n_antibody_dominated_families": len(ab_fams),
    }

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
per_complex = {}
for k in keys:
    a, b = comp_sf[k]
    per_complex[k] = {
        "partnerA_sf": sorted(a),
        "partnerB_sf": sorted(b),
        "family_id": family_id[k],
    }

out = {
    "meta": {
        "sifts_source": SIFTS_URLS[0],
        "global_chain_coverage_pct": round(global_cov, 1),
        "n_complexes_total": len(all_complexes),
        "n_cath_families": len(roots),
        "n_unmapped_complexes": len(unmapped),
        "unmapped_policy": "complex missing CATH on >=1 partner side is its own singleton family (cannot be 'exposed')",
        "antibody_superfamily": ANTIBODY_SF,
    },
    "per_complex": per_complex,
    "per_dataset": per_dataset,
}
with open(os.path.join(OUTDIR, "cath_families.json"), "w") as f:
    json.dump(out, f, indent=2)

# table
order = ["S1102", "S1131", "S2003", "S4169", "SKEMPI-single", "SKEMPI-multi"]
print()
print(f"{'dataset':<15}{'n_mut':>8}{'n_cplx':>8}{'CATH-fam expo %':>18}{'chain cov %':>14}")
print("-" * 63)
for ds in order:
    d = per_dataset[ds]
    print(f"{ds:<15}{d['n_mut']:>8}{d['n_cplx']:>8}{d['cath_family_exposure_pct']:>18}{d['chain_coverage_pct']:>14}")
print("-" * 63)
print(f"global chain coverage: {global_cov:.1f}%   |   "
      f"{len(unmapped)} complexes unmapped on >=1 side (singleton families)")
print("\nAntibody (Ig 2.60.40.10) exposure share of each dataset:")
for ds in order:
    d = per_dataset[ds]
    print(f"  {ds:<15} {d['antibody_exposed_pct']:>5}%  "
          f"({d['n_antibody_dominated_families']} Ig-containing multi-complex families)")
print(f"\nwrote {os.path.join(OUTDIR, 'cath_families.json')}")
