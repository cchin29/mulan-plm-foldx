"""The FoldX score channel: consume FoldX ΔΔG as a model feature.

The pipeline that *produces* those energies is no longer here. It was extracted to
[skempi-foldx](https://github.com/cchin29/skempi-foldx), a dependency-free MIT-licensed package
that computes and ships FoldX ΔΔG for SKEMPI 2.0. Two reasons, both worth recording:

* **Licensing.** None of it derived from upstream MuLAN, but sitting in this repository it
  inherited CC BY-NC-SA — non-commercial, non-relicensable. The computed energies (322 complexes
  / 4238 single-point mutations, 152 / 1765 multi-point) cost a commercial FoldX
  licence and CPU-weeks to reproduce, and are useful to anyone doing ΔΔG work regardless of what
  model they train. A licence inherited from an unrelated dependency should not have blocked that.
* **It was already separable.** Zero third-party imports, and nothing from ``mulan``.

What stays here is the half that is genuinely about *this* model: joining those energies onto
MuLAN's per-fold split files, standardizing per fold on training rows only, and writing the
4/5/16-column contract that :class:`mulan.data.MulanDataset` dispatches on. A different consumer
would encode the same energies differently — which is precisely why that code did not travel.

    from mulan import foldx

    foldx.merge_foldx(src=split_dir, stores=[foldx.StoreSpec(path=store_dir)],
                      out_scalar=..., out_dec=...)

The producer's API is re-exported here, so ``from mulan.foldx import load_store`` still works and
a caller need not know where the boundary fell. ``docs/FOLDX.md`` covers the channel; the
pipeline, the determinism doctrine and the result store are documented in skempi-foldx.
"""

# Re-exported from the extracted package, so this namespace remains one place to import from.
from skempi_foldx import (
    INTRACTABLE,
    MODE_AUTHOR,
    MODE_ROLE,
    MODE_VARIANT,
    ComplexResult,
    ConsolidationReport,
    Exclusion,
    FoldxConfig,
    Mutation,
    SkempiComplex,
    audit,
    consolidate,
    coverage,
    default_config,
    exclude_already_computed,
    filter_complexes,
    filter_worklist,
    infer_kind,
    is_excluded,
    load_complex,
    load_skempi,
    load_store,
    map_role_to_author,
    process_complex,
    run_campaign,
    source_label,
    store_kind,
    worklist_from_table,
    worklist_multi_point,
    worklist_single_point,
    write_store,
)
from skempi_foldx.terms import N_TERMS, SCALAR_TERM, TERMS

from .merge import (
    BASE_COLUMNS,
    CLIP,
    DECOMPOSED_COLUMNS,
    MISSING_FILL,
    SCALAR_COLUMNS,
    GuardSpec,
    KeyMapper,
    MergeReport,
    StoreSpec,
    merge_foldx,
)

__all__ = [
    # this repository's own: how the energies become features
    "merge_foldx", "GuardSpec", "KeyMapper", "MergeReport", "StoreSpec",
    "CLIP", "MISSING_FILL", "BASE_COLUMNS", "SCALAR_COLUMNS", "DECOMPOSED_COLUMNS",
    # re-exported from skempi-foldx: how the energies are produced and stored
    "TERMS", "N_TERMS", "SCALAR_TERM",
    "FoldxConfig", "default_config",
    "INTRACTABLE", "Exclusion", "is_excluded", "filter_complexes", "filter_worklist",
    "MODE_AUTHOR", "MODE_ROLE", "MODE_VARIANT",
    "ComplexResult", "process_complex", "run_campaign",
    "worklist_from_table", "worklist_single_point", "worklist_multi_point",
    "exclude_already_computed",
    "Mutation", "SkempiComplex", "load_skempi", "map_role_to_author",
    "ConsolidationReport", "audit", "consolidate", "coverage", "source_label",
    "load_complex", "load_store", "store_kind", "infer_kind", "write_store",
]
