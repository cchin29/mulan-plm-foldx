"""Join FoldX ΔΔG onto evaluation splits — the pipeline written once.

``merge_foldx_full_skempi.py``, ``merge_foldx_cath.py`` and ``merge_foldx_multipoint.py`` under
``experiments/full_skempi_seqonly/`` each carry their own copy of the same pipeline and differ
only in configuration; before them, ``experiments/foldx_s1102/merge_foldx.py`` and
``merge_foldx_decomposed.py`` carried it for the S1102 lineage. The three wrote the
FoldX-annotated splits behind every full-SKEMPI tier; the S1102 pair wrote the ones behind
the two ``retrain_*`` tiers and the S1102 and benchmark runs. The three still resolve
working-tree paths, which is why ``docs/REPRODUCE.md`` lists the FoldX-annotated splits as
rebuildable with this module rather than with them; the S1102 pair defaults to ``data/`` and
runs from a clone. They also drifted: more than one carries its own copy of the
twelve-term list, two crashed on a zero-coverage fold where the others wrote silent zeros, and
one accepted a mutation record that lacked eleven of the twelve terms. This module is that
pipeline once, with the drift removed and the behaviours pinned by ``tests/test_foldx.py``.
Tier-1 augmentation is not here: ``experiments/full_skempi_seqonly/merge_foldx_aug.py``
synthesizes the reverse and identity rows on top of a decomposed split those scripts wrote, and
recovers the standardization constants it needs per fold.

The pipeline
------------
1. Load the FoldX store(s) and index them by candidate join key.
2. For each fold: read ``<base>_train.tsv``, join every row, and fit mean/std **on the training
   rows of that fold only** — never on validation or test.
3. Write train/val/test with ``z = clip((x - mean) / std, ±4)``.

Standardization details that are contractual, not incidental:

* the fit uses **only rows that joined**; an unjoined row contributes nothing to the statistics
  and is then written as ``0.0``, which is the post-standardization mean, so it carries no signal;
* ``std`` is the **sample** standard deviation (``n-1``), and a zero ``std`` becomes ``1.0``;
* the scalar arm is *by construction* column 0 of the decomposed fit, since
  ``TERMS[0] == "Interaction Energy"``. Deriving it separately would let the two arms drift.

Because the fit is per fold, changing which rows join nudges every standardized value in that
fold slightly. That is expected, and is not evidence of a join error.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from skempi_foldx import N_TERMS, SCALAR_TERM, TERMS, load_complex

#: Standardized scores are clipped to +/- this. FoldX occasionally returns a very large magnitude
#: for a clashing rotamer; without a clip those rows dominate the training signal through a
#: channel meant to be a hint, not a label.
#:
#: This and MISSING_FILL live here, not in skempi_foldx, because they describe how THIS model
#: encodes the energies as features -- a different consumer would choose differently. Keeping
#: them out of the producer is what lets it stay dependency-free and model-agnostic.
CLIP = 4.0

#: Written when no FoldX result covers a row. Zero is the fold's training mean *after*
#: standardization, so an uncovered row contributes no signal rather than a misleading one.
MISSING_FILL = 0.0

# Column layouts. The number of columns IS the schema -- `mulan.data.MulanDataset.from_table`
# dispatches on it.
BASE_COLUMNS = 4      # seq1, seq2, mutations, ddG
SCALAR_COLUMNS = 5    # ... + z(Interaction Energy)
DECOMPOSED_COLUMNS = BASE_COLUMNS + N_TERMS   # 16

assert TERMS[0] == SCALAR_TERM, "the scalar arm is column 0 of the decomposed fit"

SPLITS = ("train", "val", "test")
FLOAT_FMT = "{:.5f}"


# ============================================================================================
# Store specification
# ============================================================================================

@dataclass
class StoreSpec:
    """One FoldX result directory and how its keys relate to a split's mutation strings."""

    path: Path
    json_field: str = "muts"
    """``muts`` for single-point results, ``variants`` for multi-point. Hard-coding either
    silently yields an empty store, which then produces an all-zero column that looks like a
    valid but poorly-covered arm."""

    keying: str = "raw"
    """``raw`` — the split already uses FoldX's own mutation strings (the S1102 case).
    ``group_remap`` — translate a per-chain residue index into the split's group-offset
    numbering. ``sidecar`` — join through an explicit key table (multi-point)."""

    join_key_tsv: Optional[Path] = None
    require_all_terms: bool = True
    """Admit a mutation only if all twelve terms are present. The one merger that did not
    enforce this could feed the decomposed arm a record missing eleven terms."""

    def __post_init__(self):
        self.path = Path(self.path)
        if self.join_key_tsv:
            self.join_key_tsv = Path(self.join_key_tsv)


@dataclass
class GuardSpec:
    """Refuse to score a row whose interface grouping FoldX did not actually use.

    A FoldX store is keyed by bare PDB code, but a split row names a specific interface
    (``3SE4.B.A`` vs ``3SE4.B.C``). SKEMPI lists some complexes under more than one grouping and
    the compute pipeline kept the first it saw. Without this guard the rows of the *other*
    grouping silently receive a different interface's energies and are counted as covered — a
    wrong value rather than a missing one, which is strictly worse.
    """

    skempi_csv: Optional[Path] = None
    exclude_tsv: Optional[Path] = None
    strict: bool = True
    """Raise when a configured source is missing. The previous implementations failed *open*:
    a missing file disabled the guard silently, coverage went **up**, and the affected rows
    quietly received wrong values."""

    _trusted: Dict[str, Tuple[str, str]] = field(default_factory=dict, init=False)
    _excluded: set = field(default_factory=set, init=False)

    def load(self) -> "GuardSpec":
        if self.skempi_csv:
            path = Path(self.skempi_csv)
            if not path.exists():
                if self.strict:
                    raise FileNotFoundError(
                        f"Grouping guard needs {path}, which is missing. Without it, rows of a "
                        f"secondary interface grouping silently receive another interface's "
                        f"energies. Pass strict=False only if you have verified that is safe."
                    )
            else:
                with open(path, newline="") as fh:
                    reader = csv.reader(fh, delimiter=";")
                    header = next(reader, None) or []
                    col = header.index("#Pdb") if "#Pdb" in header else 0
                    for row in reader:
                        if not row or len(row) <= col or not row[col]:
                            continue
                        parts = row[col].split("_")
                        if len(parts) >= 3:
                            # First-seen grouping wins, mirroring what the compute pipeline did.
                            self._trusted.setdefault(parts[0], (parts[1], parts[2]))
        if self.skempi_csv and self.strict and not self._trusted:
            raise ValueError(
                f"{self.skempi_csv} yielded no trusted chain groupings, so the guard would be "
                f"silently inert -- rows of a secondary interface grouping would receive another "
                f"interface's energies while coverage went UP. Check the file has a '#Pdb' "
                f"column of the form '<CODE>_<g1>_<g2>'. Pass strict=False to proceed anyway."
            )
        if self.exclude_tsv:
            path = Path(self.exclude_tsv)
            if not path.exists():
                if self.strict:
                    raise FileNotFoundError(f"Guard exclude list {path} is missing.")
            else:
                for line in path.read_text().splitlines():
                    if line.strip() and not line.startswith("#"):
                        self._excluded.add(line.split()[0])
        return self

    def denies(self, complex_id: str) -> bool:
        """``complex_id`` is column 0 with the trailing ``_<group>`` removed."""
        if complex_id in self._excluded:
            return True
        parts = complex_id.split(".")
        if self._trusted and len(parts) >= 3:
            trusted = self._trusted.get(parts[0])
            if trusted is not None and (parts[1], parts[2]) != trusted:
                return True
        return False


# ============================================================================================
# Key construction
# ============================================================================================

class KeyMapper:
    """Builds the candidate lookup keys for a FoldX mutation string.

    **Exactly two candidates are produced**, and the order matters:

    1. the group-offset remap (authoritative — written hard, wins collisions);
    2. the raw FoldX key (written with ``setdefault`` — never overwrites).

    A third candidate once existed, reading the residue number as an author sequence number.
    Measured across every tier it recovered **zero** rows the others did not, while landing on
    other real split rows and handing them a different mutation's ΔΔG. Coverage was identical
    with and without it, so the metric being validated could not see the damage; it surfaced
    only because downstream accuracy fell. Do not add a candidate key without first measuring
    its *unique* recovery.
    """

    def __init__(self, mapping_loader: Optional[Callable[[str], dict]] = None):
        self._loader = mapping_loader
        # Instance-scoped, not module-global: two directories in one process previously shared
        # a cache keyed only by PDB code, so the second silently reused the first's mapping.
        self._chains: Dict[str, Optional[dict]] = {}
        self._offsets: Dict[Tuple[str, str], Optional[dict]] = {}

    def _chains_for(self, code: str) -> Optional[dict]:
        if code not in self._chains:
            try:
                self._chains[code] = self._loader(code) if self._loader else None
            except Exception:
                self._chains[code] = None
        return self._chains[code]

    def _offsets_for(self, code: str, group: str) -> Optional[dict]:
        key = (code, group)
        if key not in self._offsets:
            chains = self._chains_for(code)
            if not chains:
                self._offsets[key] = None
            else:
                offsets, running = {}, 0
                for chain in group:
                    if chain not in chains:
                        offsets = None
                        break
                    offsets[chain] = running
                    running += len(chains[chain]["seq"])
                self._offsets[key] = offsets
        return self._offsets[key]

    def group_remap(
        self, mutation: str, groups: Optional[Tuple[str, str]], code: str
    ) -> Optional[str]:
        """Translate ``<wt><author_chain><per_chain_index><mut>`` into the split's numbering."""
        if groups is None or len(mutation) < 4:
            return None
        chain = mutation[1]
        g1, g2 = groups
        if chain in g1:
            side, group = "A", g1
        elif chain in g2:
            side, group = "B", g2
        else:
            return None
        residue = mutation[2:-1]
        if not residue.isdigit():
            return None          # insertion codes are dropped by the split builder too
        offsets = self._offsets_for(code, group)
        if offsets is None:
            return None
        chains = self._chains_for(code)
        n = int(residue)
        if not (chains and 1 <= n <= len(chains[chain]["seq"])):
            return None
        return f"{mutation[0]}{side}{offsets[chain] + n}{mutation[-1]}"


# ============================================================================================
# Loading and joining
# ============================================================================================

def read_split(path: Path, min_fields: int = 3) -> List[List[str]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) < min_fields:
            fields = line.split()
        if len(fields) >= min_fields:
            rows.append(fields)
    return rows


def parse_groups(split_root: Path, fold_glob: str = "fold_*") -> Dict[str, Tuple[str, str]]:
    """``code -> (group1, group2)`` seen anywhere under a split root."""
    groups: Dict[str, Tuple[str, str]] = {}
    for fold in sorted(Path(split_root).glob(fold_glob)):
        for tsv in fold.glob("*.tsv"):
            for row in read_split(tsv):
                parts = row[0].split("_")[0].split(".")
                if len(parts) >= 3:
                    groups[parts[0]] = (parts[1], parts[2])
    return groups


def load_store_indexed(
    spec: StoreSpec,
    groups: Dict[str, Tuple[str, str]],
    mapper: KeyMapper,
) -> Dict[Tuple[str, str], List[float]]:
    """``(code, split_mutation_string) -> term vector``."""
    index: Dict[Tuple[str, str], List[float]] = {}
    sidecar: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    if spec.keying == "sidecar":
        if not spec.join_key_tsv:
            raise ValueError("keying='sidecar' needs join_key_tsv")
        for line in Path(spec.join_key_tsv).read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if len(f) >= 5:
                sidecar[(f[0], f[1], f[2])] = (f[3], f[4])

    raw_pass: List[Tuple[Tuple[str, str], List[float]]] = []
    for path in sorted(Path(spec.path).glob("*.json")):
        code = path.stem
        records, _ = load_complex(path)
        for mutation, entry in records.items():
            present = [t for t in TERMS if t in entry]
            if spec.require_all_terms and len(present) != N_TERMS:
                continue
            if not present:
                continue
            vector = [float(entry[t]) if t in entry else 0.0 for t in TERMS]
            if spec.keying == "group_remap":
                remapped = mapper.group_remap(mutation, groups.get(code), code)
                if remapped:
                    index[(code, remapped)] = vector      # authoritative: hard write
            raw_pass.append(((code, mutation), vector))    # raw: setdefault, second pass
    for key, vector in raw_pass:
        index.setdefault(key, vector)
    return index, sidecar


def _complex_id(label: str) -> str:
    """The ``code.g1.g2`` prefix of column 0 — e.g. ``1AHW.AB.C_AB`` -> ``1AHW.AB.C``.

    Takes everything before the FIRST underscore, not the last. `rsplit` broke on any label
    carrying a further suffix: an augmented split's reverse-mutation labels look like
    ``1CSE.E.I_E_LI38S-GI32Y``, which `rsplit` cut to ``1CSE.E.I_E`` -- so the grouping guard
    compared ``("E", "I_E")`` against ``("E", "I")``, denied every reverse row, and the tier
    reported ~50% coverage that read as normal partial coverage. `_code_of` already used the
    first underscore, so the two helpers disagreed about how to cut the same field.
    """
    return label.split("_", 1)[0]


def _code_of(label: str, sep: str) -> str:
    head = label.split("_")[0]
    return head.split(sep)[0] if sep else head


# ============================================================================================
# Standardization
# ============================================================================================

def fit_standardizer(
    vectors: Sequence[Optional[Sequence[float]]],
    ddof: int = 1,
    zero_std_to: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """Per-term mean and sample std over the rows that joined."""
    # Drop non-finite vectors from the fit as well: one NaN would make the mean and std NaN and
    # poison every row in the fold, not just its own.
    present = [v for v in vectors
               if v is not None and all(math.isfinite(float(x)) for x in v)]
    means, stds = [], []
    for j in range(N_TERMS):
        column = [float(v[j]) for v in present]
        if not column:
            means.append(0.0)
            stds.append(zero_std_to)
            continue
        mean = sum(column) / len(column)
        variance = sum((x - mean) ** 2 for x in column) / max(1, len(column) - ddof)
        means.append(mean)
        stds.append((variance ** 0.5) or zero_std_to)
    return means, stds


def standardize(
    vector: Optional[Sequence[float]],
    means: Sequence[float],
    stds: Sequence[float],
    clip: float = CLIP,
) -> List[float]:
    if vector is None:
        # The training mean after standardization: no signal rather than a misleading one.
        return [MISSING_FILL] * N_TERMS
    out = []
    for j in range(N_TERMS):
        z = (float(vector[j]) - means[j]) / stds[j]
        # `min(clip, nan)` returns clip, so a single non-finite term used to be written as the
        # MAXIMUM-signal value -- and if it reached the fit, the whole column became NaN and
        # every row in train, val and test saturated to +4.0 while coverage reported 100%.
        out.append(MISSING_FILL if not math.isfinite(z) else max(-clip, min(clip, z)))
    return out


# ============================================================================================
# The merge
# ============================================================================================

@dataclass
class MergeReport:
    per_fold: List[Tuple[int, int, int]] = field(default_factory=list)
    """``(fold, covered, total)`` over all splits."""
    denied: Dict[str, int] = field(default_factory=dict)

    @property
    def covered(self) -> int:
        return sum(c for _, c, _ in self.per_fold)

    @property
    def total(self) -> int:
        return sum(t for _, _, t in self.per_fold)

    def summary(self) -> str:
        lines = [f"fold {i}: {c}/{t} ({100 * c / t:.1f}%)" if t else f"fold {i}: empty"
                 for i, c, t in self.per_fold]
        if self.total:
            lines.append(f"TOTAL  {self.covered}/{self.total} "
                         f"({100 * self.covered / self.total:.1f}%)")
        if self.denied:
            lines.append("denied by the grouping guard: "
                         + ", ".join(f"{k}={v}" for k, v in sorted(self.denied.items())))
        return "\n".join(lines)


def merge_foldx(
    src: Path,
    stores: Sequence[StoreSpec],
    out_scalar: Optional[Path] = None,
    out_dec: Optional[Path] = None,
    base_name: Optional[str] = None,
    base_per_fold: bool = False,
    complex_id_sep: str = ".",
    guard: Optional[GuardSpec] = None,
    clip: float = CLIP,
    ddof: int = 1,
    fold_glob: str = "fold_*",
    splits: Sequence[str] = SPLITS,
    mapping_loader: Optional[Callable[[str], dict]] = None,
    float_fmt: str = FLOAT_FMT,
) -> MergeReport:
    """Write the scalar and/or decomposed split files for every fold under ``src``."""
    src = Path(src)
    guard = (guard or GuardSpec(strict=False)).load()
    groups = parse_groups(src, fold_glob) if complex_id_sep == "." else {}
    mapper = KeyMapper(mapping_loader)

    indexes, sidecars = [], []
    for spec in stores:
        index, sidecar = load_store_indexed(spec, groups, mapper)
        indexes.append((spec, index))
        sidecars.append(sidecar)

    report = MergeReport()
    # Pair each source fold with its OWN number. Enumerating a sorted glob made the output fold
    # index the enumeration position: with >=10 folds the lexicographic sort scrambles it
    # (fold_10 sorts before fold_2), a gap shifts every later fold down, and a stray `fold_*`
    # entry displaces everything after it. The channel for fold_k would then be joined to a
    # different fold's rows than every other channel for fold_k -- silently, with perfect shapes
    # and coverage. Audited: no existing split is affected (max 10 folds, no gaps, no strays).
    folds = []
    for path in sorted(Path(src).glob(fold_glob)):
        m = re.fullmatch(r"fold_(\d+)", path.name)
        if not m:
            print(f"[merge] ignoring non-fold entry {path.name!r}")
            continue
        folds.append((int(m.group(1)), path))
    folds.sort(key=lambda kv: kv[0])

    def lookup(row: List[str], count_denials: bool = False) -> Optional[List[float]]:
        # `count_denials` is False during the standardization fit: the fit re-reads the train
        # rows that the write pass also reads, so counting in both inflates the diagnostic by
        # the number of passes. The joined values are unaffected either way -- but a coverage
        # report that overstates denials is exactly the kind of number that gets quoted.
        cid = _complex_id(row[0])
        if guard.denies(cid):
            if count_denials:
                report.denied[cid] = report.denied.get(cid, 0) + 1
            return None
        code = _code_of(row[0], complex_id_sep)
        is_multi = "," in row[2]
        for (spec, index), sidecar in zip(indexes, sidecars):
            if spec.keying == "sidecar":
                if not is_multi:
                    continue
                bridged = sidecar.get((row[0], row[1], row[2]))
                if bridged and bridged in index:
                    return index[bridged]
                continue
            if is_multi and spec.json_field == "muts":
                continue
            hit = index.get((code, row[2]))
            if hit is not None:
                return hit
        return None

    for i, fold in folds:
        base = base_name
        if base is None or base_per_fold:
            trains = list(fold.glob("*_train.tsv"))
            if not trains:
                continue
            base = trains[0].name[: -len("_train.tsv")]

        train_rows = read_split(fold / f"{base}_train.tsv")
        means, stds = fit_standardizer([lookup(r) for r in train_rows], ddof=ddof)

        covered = total = 0
        for split in splits:
            path = fold / f"{base}_{split}.tsv"
            if not path.exists():
                continue
            rows = read_split(path)
            scalar_lines, dec_lines = [], []
            for row in rows:
                vector = lookup(row, count_denials=True)
                total += 1
                covered += vector is not None
                z = standardize(vector, means, stds, clip=clip)
                head = row[:4] if len(row) >= 4 else row
                if out_scalar:
                    scalar_lines.append("\t".join(head + [float_fmt.format(z[0])]))
                if out_dec:
                    dec_lines.append("\t".join(head + [float_fmt.format(x) for x in z]))
            for out_dir, lines in ((out_scalar, scalar_lines), (out_dec, dec_lines)):
                if not out_dir:
                    continue
                target = Path(out_dir) / f"fold_{i}"
                target.mkdir(parents=True, exist_ok=True)
                (target / f"{base}_{split}.tsv").write_text("\n".join(lines) + "\n")
        report.per_fold.append((i, covered, total))

    return report
