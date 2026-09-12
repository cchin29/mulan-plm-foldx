"""Evaluation tiers — which truth to score against.

A "tier" is a protocol applied to a dataset: the CATH-superfamily hold-out over the combined
single- and multi-point set, the clustered hold-out over single-point only, and so on. It fixes
the truth files and the fold count.

Why this exists
---------------
The tier used to be two module-level globals in the scoring script, initialised to the **leaky
10-fold** splits. Every scorer then reassigned them before use. That worked, but it made the
*default* behaviour of the metric core the one protocol no claim in this project rests on — and
a scorer that forgot to reassign would silently score against the wrong truth and report a
number that looked entirely normal.

Here the default is :data:`DEFAULT_TIER`, the CATH-superfamily hold-out: the strictest protocol
and the one that is directly comparable to published tables. The leaky tier still exists, because
reproducing the upstream paper's number requires it — but it must be **named explicitly**. You
cannot arrive at it by omission.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Scored against unless a tier is named. The strictest protocol, and frontier-comparable.
DEFAULT_TIER = "cath"


@dataclass(frozen=True)
class Tier:
    name: str
    protocol: str
    """The protocol in :mod:`mulan.splits` this tier applies."""
    split_dir: str
    basename: str
    n_folds: int
    mutations: str
    """``single``, ``multi`` or ``mixed``."""
    legacy: bool = False
    shipped: bool = True
    """Whether this tier's split data is published with the repository. The leaky S1102 tier is
    deliberately not shipped; ``cath`` joins onto a third-party partition that is not ours to
    redistribute, so it ships as a recipe (see ``build_cmd``) rather than as data. A tier that
    cannot resolve should say so rather than fail at read time."""
    build_cmd: str = ""
    """How to obtain the split when ``shipped`` is False and it is buildable locally."""
    note: str = ""

    per_fold_test_files: bool = True
    """Whether the per-category test files live inside ``fold_<k>/`` or at the split-dir root.
    The CATH tier ships ``test_single.tsv`` / ``test_multiple.tsv`` at the ROOT; inserting
    ``fold_{f}`` for those produced a path that cannot resolve."""

    def truth_template(self, root: Path, test_file: Optional[str] = None) -> str:
        """Path template for the per-fold test file, with a ``{f}`` placeholder.

        ``test_file`` overrides the default test filename — the CATH tier ships
        ``test_single.tsv`` / ``test_multiple.tsv`` alongside the combined set, so a run can be
        broken out by mutation category to match a published table's rows.
        """
        if test_file and not self.per_fold_test_files:
            return str(Path(root) / self.split_dir / test_file)
        name = test_file or f"{self.basename}_test.tsv"
        return str(Path(root) / self.split_dir / "fold_{f}" / name)


_TIERS: Tuple[Tier, ...] = (
    Tier(
        name="cath", protocol="cath",
        split_dir="data/splits/splits_skempi_full_cath_kfold",
        basename="skempi_all", n_folds=1, mutations="mixed", per_fold_test_files=False,
        shipped=False,
        build_cmd="python experiments/full_skempi_seqonly/build_split_cath.py "
                  "--usp <USP-ddG>/data/SKEMPI2/skempi_v2.csv",
        note="The default. USP-ddG's CATH-superfamily partition, joined onto our rows, so a "
             "result is placeable directly in their table. That partition is not ours to "
             "redistribute, so this split is built locally rather than shipped — see "
             "data/splits/splits_skempi_full_cath_kfold/README.md. Single fixed partition — no "
             "fold spread, so do not over-read small per-arm differences.",
    ),
    Tier(
        name="clustered_sp", protocol="clustered_id60",
        split_dir="data/splits/splits_skempi_full_clustered_id60_kfold",
        basename="skempi_sp", n_folds=3, mutations="single",
        note="CD-HIT <=60% families, single-point. The cleanest apples-to-apples comparison "
             "against the one frontier comparator that uses the same construction.",
    ),
    Tier(
        name="clustered_all", protocol="clustered_id60",
        split_dir="data/splits/splits_skempi_full_clustered_all_id60_kfold",
        basename="skempi_all", n_folds=3, mutations="mixed",
        note="CD-HIT <=60% families over the combined single- and multi-point set. Built on a "
             "second machine and shipped here with the 2026-07-31 merge; rebuildable with "
             "experiments/full_skempi_seqonly/build_split_clustered_all.py.",
    ),
    Tier(
        name="bycomplex_sp", protocol="bycomplex",
        split_dir="data/splits/splits_skempi_full_bycomplex_seed42",
        basename="skempi_sp", n_folds=3, mutations="single",
        note="Leakage-controlled but NOT homology-controlled — not comparable to a CATH number.",
    ),
    Tier(
        name="bycomplex_all", protocol="bycomplex",
        split_dir="data/splits/splits_skempi_full_bycomplex_all_seed42",
        basename="skempi_all", n_folds=3, mutations="mixed",
    ),
    Tier(
        name="mp_clustered", protocol="clustered_id60",
        split_dir="data/splits/splits_skempi_full_mp_clustered_id60_kfold",
        basename="skempi_mp", n_folds=3, mutations="multi",
    ),
    Tier(
        name="mp_bycomplex", protocol="bycomplex",
        split_dir="data/splits/splits_skempi_full_mp_bycomplex_seed42",
        basename="skempi_mp", n_folds=3, mutations="multi",
    ),
    Tier(
        name="s1102_clustered", protocol="clustered_id60",
        split_dir="data/splits/splits_clustered_id60_kfold",
        basename="S1102_filtered", n_folds=3, mutations="single",
        note="The early ladder, on the ~1100-mutation S1102 subset.",
    ),
    Tier(
        name="s1102_bycomplex", protocol="bycomplex",
        split_dir="data/splits/splits_bycomplex_seed42",
        basename="S1102_filtered", n_folds=3, mutations="single",
    ),
    Tier(
        name="legacy_cv10", protocol="legacy_cv10",
        split_dir="scratch/foldx_s1102/splits_balanced_foldxdec",
        basename="S1102_filtered", n_folds=10, mutations="single", legacy=True, shipped=False,
        note="LEAKY: random 10-fold over mutations, so the same complex is in train and test. "
             "The upstream paper's protocol. Available only when named explicitly — no claim "
             "in this work rests on it.",
    ),
)

_BY_NAME: Dict[str, Tier] = {t.name: t for t in _TIERS}


def get_tier(name: Optional[str] = None) -> Tier:
    """Look up a tier. ``None`` gives :data:`DEFAULT_TIER`.

    A legacy tier is returned only when asked for by name — which is the point: the previous
    arrangement let you *land* on the leaky tier by not specifying one.
    """
    if name is None:
        name = DEFAULT_TIER
    if name not in _BY_NAME:
        raise KeyError(
            f"Unknown tier {name!r}. Available: {', '.join(list_tiers())}"
            f" (legacy, explicit only: {', '.join(list_tiers(legacy_only=True))})"
        )
    return _BY_NAME[name]


def list_tiers(include_legacy: bool = False, legacy_only: bool = False) -> List[str]:
    if legacy_only:
        return [t.name for t in _TIERS if t.legacy]
    return [t.name for t in _TIERS if include_legacy or not t.legacy]


def describe(name: Optional[str] = None) -> str:
    tier = get_tier(name)
    lines = [f"{tier.name}  ({tier.protocol}, {tier.n_folds} fold(s), {tier.mutations}-point)"]
    if tier.legacy:
        lines.append("  *** LEGACY / LEAKY — not comparable to any published number ***")
    if not tier.shipped:
        lines.append("  *** split data NOT SHIPPED with this repository ***")
        if tier.build_cmd:
            lines.append(f"  build it:  {tier.build_cmd}")
    if tier.note:
        lines.append(f"  {tier.note}")
    return "\n".join(lines)
