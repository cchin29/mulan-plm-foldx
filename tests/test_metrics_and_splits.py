"""Metrics and protocol registry.

The metric bodies were extracted verbatim from the script that defined them, and
``rescore.py --selftest`` already checks them against scipy and sklearn to 1e-9. What is tested
here is what that selftest does not cover: the per-structure aggregation, the tie and degenerate
handling, and the bootstrap that previously existed in three divergent copies.
"""

from __future__ import annotations

import numpy as np
import pytest

from mulan import metrics, splits


# ============================================================================================
# Ranking and correlation
# ============================================================================================

def test_ties_get_average_ranks():
    assert list(metrics.average_ranks(np.array([1.0, 2.0, 2.0, 3.0]))) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_pearson_on_ranks():
    a = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    b = np.array([10.0, 2.0, 30.0, 4.0, 50.0])
    assert metrics.spearman(a, b) == pytest.approx(
        metrics.pearson(metrics.average_ranks(a), metrics.average_ranks(b)))


def test_spearman_is_monotone_invariant():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.spearman(a, np.exp(a)) == pytest.approx(1.0)
    assert metrics.spearman(a, -np.exp(a)) == pytest.approx(-1.0)


def test_degenerate_input_is_nan_not_zero():
    """A constant vector has no correlation; reporting 0.0 would read as 'no signal measured'
    rather than 'not measurable'."""
    assert np.isnan(metrics.pearson(np.ones(5), np.arange(5.0)))
    assert np.isnan(metrics.spearman(np.array([1.0]), np.array([2.0])))


# ============================================================================================
# AUROC
# ============================================================================================

def test_auroc_perfect_and_inverted():
    labels = np.array([0, 0, 1, 1])
    assert metrics.auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auroc_all_ties_is_one_half():
    assert metrics.auroc(np.array([0, 1, 0, 1]), np.ones(4)) == pytest.approx(0.5)


def test_auroc_single_class_is_nan():
    assert np.isnan(metrics.auroc(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0])))


# ============================================================================================
# Per-structure aggregation — the headline metric
# ============================================================================================

def _blocks(sizes):
    return np.concatenate([[f"C{i}"] * n for i, n in enumerate(sizes)])


def test_complexes_below_threshold_are_excluded():
    pdb = _blocks([12, 3])
    true = np.arange(15.0)
    pred = true.copy()
    _, _, n_qual, _ = metrics.per_structure(pred, true, pdb, T=10)
    assert n_qual == 1                      # only the 12-mutation complex qualifies


def test_per_structure_averages_complexes_not_mutations():
    """A large complex must not dominate a small one; that is the whole point of the metric."""
    pdb = _blocks([10, 100])
    true = np.concatenate([np.arange(10.0), np.arange(100.0)])
    pred = np.concatenate([np.arange(10.0), -np.arange(100.0)])   # +1 then -1
    _, spearman_mean, n_qual, _ = metrics.per_structure(pred, true, pdb, T=10)
    assert n_qual == 2
    assert spearman_mean == pytest.approx(0.0, abs=1e-9)          # mean of +1 and -1


def test_degenerate_complexes_are_counted_not_silently_dropped():
    pdb = _blocks([10, 10])
    true = np.concatenate([np.arange(10.0), np.ones(10)])          # second has no variance
    pred = np.concatenate([np.arange(10.0), np.arange(10.0)])
    _, spearman_mean, n_qual, n_degen = metrics.per_structure(pred, true, pdb, T=10)
    assert (n_qual, n_degen) == (2, 1)
    assert spearman_mean == pytest.approx(1.0)


def test_pooled_and_per_structure_can_disagree_sharply():
    """The reason this project reports per-structure: a model that ranks complexes correctly
    but ranks within a complex at chance still scores well pooled."""
    pdb = _blocks([10, 10])
    true = np.concatenate([np.arange(10.0), 100 + np.arange(10.0)])
    pred = np.concatenate([np.arange(9.0, -1.0, -1), 100 + np.arange(9.0, -1.0, -1)])
    pooled = metrics.spearman(pred, true)
    _, per_struct, _, _ = metrics.per_structure(pred, true, pdb, T=10)
    # The model is exactly BACKWARDS within every complex, and pooled correlation still comes
    # out substantially positive, because half the ranked pairs are cross-complex and those it
    # gets right. Per-structure reports the truth.
    assert pooled == pytest.approx(0.5038, abs=1e-3)
    assert per_struct == pytest.approx(-1.0)


# ============================================================================================
# Bootstrap
# ============================================================================================

def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    pdb = _blocks([12] * 8)
    true = rng.normal(size=len(pdb))
    pred = true + rng.normal(scale=0.5, size=len(pdb))
    result = metrics.per_structure_bootstrap(pred, true, pdb, threshold=10,
                                             n_resamples=200, seed=0)
    assert result.lower <= result.point <= result.upper
    assert result.n_clusters == 8


def test_bootstrap_is_deterministic_given_a_seed():
    pdb = _blocks([12] * 6)
    rng = np.random.default_rng(1)
    true = rng.normal(size=len(pdb)); pred = true + rng.normal(scale=0.5, size=len(pdb))
    kw = dict(threshold=10, n_resamples=100, seed=7)
    a = metrics.per_structure_bootstrap(pred, true, pdb, **kw)
    b = metrics.per_structure_bootstrap(pred, true, pdb, **kw)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_bootstrap_resamples_whole_complexes():
    """Every resample must contain whole complexes, never partial ones."""
    pdb = _blocks([10, 10, 10])
    sizes = []

    def statistic(idx, labels):
        sizes.extend(int((labels == k).sum()) for k in np.unique(labels))
        return 0.0

    metrics.cluster_bootstrap(statistic, pdb, n_resamples=50, seed=0)
    assert sizes and all(s == 10 for s in sizes)


def test_a_complex_drawn_twice_counts_twice():
    """The bug this guards: `per_structure` re-derived groups from the data, so a duplicated
    complex collapsed back into ONE group with double the rows. Per-complex correlation is
    invariant to exact duplication, so the duplicate carried weight 1 instead of 2 — making each
    resample a random ~63% subset rather than a bootstrap over n, and shrinking every interval
    by ~24% in the anti-conservative direction."""
    pdb = _blocks([10, 10, 10])
    counts = []

    def statistic(idx, labels):
        counts.append(len(np.unique(labels)))
        return 0.0

    metrics.cluster_bootstrap(statistic, pdb, n_resamples=200, seed=0)
    assert set(counts) == {3}, (
        f"every resample must yield exactly 3 clusters, got {sorted(set(counts))} — "
        "duplicates are collapsing")


def test_threshold_is_not_inflated_by_duplicate_draws():
    """A 6-mutation complex must never qualify at T>=10 just because it was drawn twice."""
    pdb = _blocks([6, 6, 6])
    true = np.tile(np.arange(6.0), 3)
    pred = true.copy()
    qualifying = []

    def statistic(idx, labels):
        _, _, n_qual, _ = metrics.per_structure(pred[idx], true[idx], labels, T=10)
        qualifying.append(n_qual)
        return 0.0

    metrics.cluster_bootstrap(statistic, pdb, n_resamples=100, seed=0)
    assert set(qualifying) == {0}, "no 6-mutation complex may qualify at T=10"


def test_paired_delta_of_a_model_with_itself_is_zero():
    pdb = _blocks([12] * 5)
    rng = np.random.default_rng(2)
    true = rng.normal(size=len(pdb)); pred = true + rng.normal(scale=0.3, size=len(pdb))
    result = metrics.paired_delta(pred, pred, true, pdb, threshold=10, n_resamples=50)
    assert result.point == pytest.approx(0.0)
    assert result.lower == pytest.approx(0.0) and result.upper == pytest.approx(0.0)


def test_sign_test():
    assert metrics.sign_test([1, 1, 1, 1, 1])[2] == pytest.approx(2 / 32)
    assert metrics.sign_test([1, -1, 1, -1])[2] == pytest.approx(1.0)
    assert metrics.sign_test([0, 0])[1] == 0          # zeros excluded


# ============================================================================================
# Protocol registry
# ============================================================================================

def test_legacy_protocol_is_opt_in():
    assert "legacy_cv10" not in splits.list_protocols()
    assert "legacy_cv10" in splits.list_protocols(include_legacy=True)


def test_every_default_protocol_is_leakage_controlled():
    for name in splits.list_protocols():
        assert splits.get_protocol(name).leakage_controlled


def test_bycomplex_is_flagged_as_not_homology_controlled():
    """The trap this registry exists to prevent: reading a by-complex number as if it were
    comparable to a CATH-split one."""
    assert not splits.get_protocol("bycomplex").homology_controlled
    assert not splits.frontier_comparable("bycomplex")
    assert "not comparable" in splits.comparability_warning("bycomplex").lower()


def test_cath_warns_about_its_single_fold():
    assert splits.get_protocol("cath").folds == 1
    assert "fold spread" in splits.comparability_warning("cath")


def test_clustered_is_the_one_clean_comparison():
    assert splits.comparability_warning("clustered_id60") is None


def test_unknown_protocol_names_the_alternatives():
    with pytest.raises(KeyError, match="Unknown protocol"):
        splits.get_protocol("nope")


# ============================================================================================
# Evaluation tiers — the default must be honest
# ============================================================================================

def test_default_tier_is_cath_not_the_leaky_one():
    """The whole point: you cannot arrive at the leaky protocol by omission."""
    assert metrics.DEFAULT_TIER == "cath"
    assert metrics.get_tier().name == "cath"
    assert metrics.get_tier(None).name == "cath"
    assert not metrics.get_tier().legacy


def test_legacy_tier_requires_naming():
    assert "legacy_cv10" not in metrics.list_tiers()
    assert "legacy_cv10" in metrics.list_tiers(include_legacy=True)
    assert metrics.get_tier("legacy_cv10").legacy
    assert metrics.get_tier("legacy_cv10").n_folds == 10


def test_every_default_tier_maps_to_a_leakage_controlled_protocol():
    for name in metrics.list_tiers():
        protocol = splits.get_protocol(metrics.get_tier(name).protocol)
        assert protocol.leakage_controlled, f"{name} is not leakage-controlled"


def test_tier_truth_template_has_a_fold_placeholder(tmp_path):
    template = metrics.get_tier("clustered_sp").truth_template(tmp_path)
    assert "{f}" in template and template.format(f=0).endswith("fold_0/skempi_sp_test.tsv")


def test_cath_tier_supports_the_mutation_category_breakouts(tmp_path):
    """USP-ddG's table has single/multiple rows; the split ships the matching test files."""
    tier = metrics.get_tier("cath")
    assert tier.truth_template(tmp_path, "test_single.tsv").endswith("test_single.tsv")
    assert tier.truth_template(tmp_path, "test_multiple.tsv").endswith("test_multiple.tsv")


def test_unknown_tier_names_the_alternatives_including_legacy():
    with pytest.raises(KeyError) as exc:
        metrics.get_tier("nope")
    assert "legacy_cv10" in str(exc.value)


def test_describe_flags_the_leaky_tier_loudly():
    assert "LEAKY" in metrics.describe("legacy_cv10")
    assert "LEAKY" not in metrics.describe("cath")


# ============================================================================================
# Early stopping and checkpointing are coupled — and the coupling is not obvious from the flags
# ============================================================================================

def test_patience_without_save_model_would_have_kept_the_wrong_weights():
    """`load_best_model_at_end` needs a save strategy; HuggingFace cannot restore a checkpoint it
    never wrote. So patience + save_model=False stopped training at the right moment and then kept
    the FINAL, over-trained weights — and every test metric came from those, silently."""
    from mulan.train_utils import resolve_checkpointing

    strategy, load_best, warning = resolve_checkpointing(
        save_model=False, early_stopping_patience=30, has_eval_set=True)
    assert strategy == "epoch", "patience must imply checkpointing"
    assert load_best is True, "the best epoch must actually be restored"
    assert warning and "over-trained" in warning


def test_patience_without_an_eval_set_warns_instead_of_pretending():
    """No eval set means no metric to monitor, so patience cannot fire at all. Enabling
    checkpointing would not help; say so rather than silently doing nothing."""
    from mulan.train_utils import resolve_checkpointing

    strategy, load_best, warning = resolve_checkpointing(
        save_model=False, early_stopping_patience=30, has_eval_set=False)
    assert (strategy, load_best) == ("no", False)
    assert warning and "no effect" in warning


def test_explicit_save_model_is_respected_and_silent():
    from mulan.train_utils import resolve_checkpointing
    assert resolve_checkpointing(True, 30, True) == ("epoch", True, None)
    assert resolve_checkpointing(True, None, True) == ("epoch", True, None)


def test_no_patience_no_save_is_unchanged():
    from mulan.train_utils import resolve_checkpointing
    assert resolve_checkpointing(False, None, True) == ("no", False, None)


def test_load_best_is_always_a_bool():
    """It used to be `eval_dataset and save_model`, which returns the DATASET (or None) rather
    than a bool whenever the left operand decides the expression."""
    from mulan.train_utils import resolve_checkpointing
    for save, pat, ev in [(True, None, True), (False, None, False), (True, 5, False)]:
        assert isinstance(resolve_checkpointing(save, pat, ev)[1], bool)


# ============================================================================================
# Regressions from the review pass
# ============================================================================================

def test_spearman_and_auroc_propagate_nan():
    """`average_ranks` sorts NaN to the end and returns a finite rank, so a single NaN
    prediction used to yield a confident correlation — Pearson went NaN, Spearman returned a
    perfect 1.0. Spearman is the headline metric, so it was the one that hid it."""
    a = np.array([1.0, 2.0, 3.0, 4.0, np.nan]); b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isnan(metrics.spearman(a, b))
    assert np.isnan(metrics.pearson(a, b))
    assert np.isnan(metrics.auroc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, np.nan])))
    assert np.isnan(metrics.auroc(np.array([0.0, np.nan, 1.0, 1.0]), np.array([.1, .2, .8, .9])))


def test_per_structure_reports_nan_not_a_number_when_a_complex_has_nan():
    pdb = _blocks([12])
    true = np.arange(12.0); pred = true.copy(); pred[3] = np.nan
    _, spear, n_qual, _ = metrics.per_structure(pred, true, pdb, T=10)
    assert n_qual == 1 and np.isnan(spear)


def test_paired_delta_compares_the_same_complexes_in_both_arms():
    """`per_structure` drops zero-variance complexes independently per arm, so differencing its
    two means was a difference over different complex sets — reintroducing the between-complex
    difficulty variance a paired contrast exists to cancel."""
    pdb = _blocks([12, 12, 12])
    true = np.tile(np.arange(12.0), 3)
    a = true + 0.1
    b = true.copy(); b[:12] = 5.0                    # arm B degenerate on complex 0 only
    result = metrics.paired_delta(a, b, true, pdb, threshold=10, n_resamples=100, seed=0)
    assert np.isfinite(result.point)
    assert result.point == pytest.approx(0.0, abs=1e-9), \
        "both arms must be scored on the two complexes both can score"


def test_every_shipped_tier_resolves_to_real_files():
    """A tier whose split data is absent should say so, not fail at read time.

    A locally-built tier (``shipped=False`` with a ``build_cmd``) still has its per-fold paths
    checked once built. Skipping it outright would let a typo in ``basename`` or ``split_dir`` pass
    the suite for the *default* tier in both states, which is the one place it matters most."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for name in metrics.list_tiers(include_legacy=True):
        tier = metrics.get_tier(name)
        buildable = not tier.shipped and bool(tier.build_cmd)
        if not tier.shipped and not buildable:
            continue
        built = (root / tier.split_dir).is_dir() and any(
            (root / tier.split_dir).glob("fold_*")
        )
        for f in range(tier.n_folds):
            path = Path(tier.truth_template(root).format(f=f))
            assert path.parent.parent == root / tier.split_dir, \
                f"tier {name} resolves outside its split dir: {path}"
            if tier.shipped or built:
                assert path.exists(), f"tier {name} declares a path that does not exist: {path}"


def test_cath_breakout_files_are_at_the_split_root():
    """They live beside fold_0/, not inside it — inserting fold_{f} produced an unresolvable
    path while the test still passed because it only checked the filename suffix.

    The CATH split is built locally rather than shipped (it joins onto a third-party partition),
    so existence is asserted only once it has been built. The path *shape* is checked either way —
    that is the regression this test exists for, and it does not need the files."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    tier = metrics.get_tier("cath")
    built = (root / tier.split_dir / "fold_0").is_dir()
    for name in ("test_single.tsv", "test_multiple.tsv"):
        path = Path(tier.truth_template(root, name))
        assert "fold_" not in path.name
        assert path.parent == root / tier.split_dir, path
        if built:
            assert path.exists(), path


def test_per_structure_returns_pearson_then_spearman():
    """The return order is (mean_pearson, mean_spearman, ...), and nothing else pinned it.

    `docs/REPRODUCE.md` unpacked this as `mean_rho, median_rho, ...` for a release, so the one
    documented route for checking a published number handed back the Pearson where the reader
    expected the headline Spearman -- 0.336 against a reported 0.310 on the first arm.

    Every other fixture in this file is monotone-linear, where Pearson and Spearman agree to
    within rounding, so swapping the two return values leaves the whole suite green. A cubic
    relationship separates them: Spearman is exactly 1 under any monotone map, Pearson is not.
    """
    from mulan import metrics

    true = [float(i) for i in range(1, 13)]
    pred = [t ** 3 for t in true]
    key = ["1ABC"] * len(true)

    mean_pearson, mean_spearman, n_qual, n_degen = metrics.per_structure(pred, true, key, T=10)

    assert mean_spearman == pytest.approx(1.0), "a monotone map must give Spearman 1"
    assert mean_pearson < 0.95, {
        "mean_pearson": mean_pearson,
        "note": "element 0 is the Pearson; if this is ~1.0 the return order has been swapped",
    }
    assert (n_qual, n_degen) == (1, 0)


def test_audit_discovers_folds_in_both_result_layouts(tmp_path):
    """`scripts/audit_local_results.py` finds folds in both layouts, and reports a missing one.

    The working tree writes `fold_<k>/training_run/all_results.json`; the published tree flattens
    that level away. This function has broken twice on the same case and neither break was caught
    by anything: first it globbed only the `training_run/` form and reported all 666 published
    folds as silent crashes, then the fix left an `.exists()` on a value that can be None and
    crashed outright on the one case check #1 exists to detect -- a fold present with no results.

    Four combinations, because the two layouts and the two outcomes are independent.
    """
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_audit", root / "scripts" / "audit_local_results.py")
    audit = importlib.util.module_from_spec(spec)
    sys.modules["_audit"] = audit
    spec.loader.exec_module(audit)

    # Three arm depths, because discover_arms' own docstring names three layouts and a refactor
    # that fixes the depth passes a single-depth fixture while silently losing the other two:
    #   tier/fold_k                   (tier IS the arm, e.g. cv10_ankh)
    #   tier/arm/fold_k               (the common case)
    #   tier/x/arm/fold_k             (e.g. a1_ankh3_large/dense/ankh3_large)
    for depth in (("tierA",), ("tierA", "armX"), ("tierA", "dense", "armX")):
      for nested in (True, False):
        for missing in (True, False):
            tree = tmp_path / f"d{len(depth)}_n{int(nested)}_m{int(missing)}"
            arm = tree.joinpath(*depth)
            for k in range(3):
                fold = arm / f"fold_{k}"
                target = fold / "training_run" if nested else fold
                target.mkdir(parents=True)
                if not (missing and k == 2):
                    (target / "all_results.json").write_text('{"test_pcc": 0.5}')

            found = audit.discover_arms(tree)
            assert arm in found, {"layout depth": len(depth), "discovered": sorted(map(str, found))}
            # A depth beyond the ones a fixed-depth glob union would plausibly cover. The
            # function's own comment is "never glob a layout you believe in; glob the thing you
            # are counting" -- enumerating four depths still believes in four.
            deep = tree.joinpath(*depth, "extra", "deeper")
            (deep / "fold_0").mkdir(parents=True, exist_ok=True)
            (deep / "fold_0" / "all_results.json").write_text('{"test_pcc": 0.5}')
            assert deep in audit.discover_arms(tree), {
                "note": "discovery must not assume a fixed number of path components",
            }
            folds = found[arm]
            assert set(folds) == {"fold_0", "fold_1", "fold_2"}, (nested, missing, folds)
            resolved = [k for k, v in folds.items() if v is not None]
            expected = 2 if missing else 3
            assert len(resolved) == expected, {
                "layout depth": len(depth),
                "nested layout": nested,
                "a fold is missing its results": missing,
                "folds with results found": len(resolved),
                "expected": expected,
            }
            if missing:
                assert folds["fold_2"] is None, "the missing fold must resolve to None, not crash"


def test_a_constant_nonzero_prediction_is_degenerate_not_nan():
    """A within-complex constant prediction must be skipped, not poison the tier mean.

    `np.std(a) == 0` is the obvious test and it is wrong. Summing n copies of a nonzero value
    accumulates rounding, so the computed mean is off by an ulp and np.std returns ~1e-15 rather
    than 0 -- for 13.69616873214543 repeated 35 times, 1.78e-15. The constant array then reaches
    the correlation, which returns nan, and one nan poisons the mean over every complex in the
    tier. Worse, it does so silently: the degenerate counter that would have disclosed it never
    increments, because the complex was never classified as degenerate.

    It stays hidden while the only constant predictions are exactly 0.0, whose mean is exact. A
    predictor whose within-complex constant is a nonzero intercept is what breaks it.
    """
    from mulan.metrics import core

    v, n = 13.69616873214543, 35
    constant = np.full(n, v)
    assert np.std(constant) != 0, "this fixture only bites while np.std of a constant is nonzero"
    assert core.degenerate(constant), "a constant array carries no ordering and must be degenerate"

    pred = list(constant) + list(np.arange(30.0)) + list(np.arange(30.0) * 2)
    true = list(np.arange(n, dtype=float)) + list(np.arange(30.0)) + list(np.arange(30.0) * 3)
    key = ["DEGEN"] * n + ["OK1"] * 30 + ["OK2"] * 30

    mean_pearson, mean_spearman, n_qual, n_degen = core.per_structure(pred, true, key, T=10)

    assert np.isfinite(mean_spearman), "one degenerate complex poisoned the tier mean"
    assert np.isfinite(mean_pearson)
    assert n_degen == 1, {
        "n_degen": n_degen,
        "note": "the constant complex must be counted as degenerate, not silently averaged in",
    }


# ============================================================================================
# Mutation-testing survivors (round 6). Each test below exists because a single-point mutation
# of the code it covers passed the whole suite; the mutation is named so the test can be read
# against it.
# ============================================================================================

def test_per_structure_is_a_mean_not_a_median():
    """Mutant: ``np.mean(spearmans)`` -> ``np.median``. Every earlier fixture had per-complex
    correlations that were symmetric or single-valued, so mean and median coincided. The
    headline is the mean over complexes; three complexes at +1, +1, -1 must give 1/3."""
    pdb = _blocks([10, 10, 10])
    true = np.tile(np.arange(10.0), 3)
    pred = np.concatenate([np.arange(10.0), np.arange(10.0), -np.arange(10.0)])
    mp, ms, n_qual, n_degen = metrics.per_structure(pred, true, pdb, T=10)
    assert (n_qual, n_degen) == (3, 0)
    assert ms == pytest.approx(1 / 3)
    assert mp == pytest.approx(1 / 3)


def test_threshold_argument_is_honoured():
    """Mutant: ``< T`` -> ``< 10``. Every earlier call passed T=10; the shipped ``ps_*_T5``
    columns are computed with T=5 through the same function."""
    pdb = _blocks([7, 12])
    true = np.arange(19.0)
    pred = true.copy()
    assert metrics.per_structure(pred, true, pdb, T=5)[2] == 2
    assert metrics.per_structure(pred, true, pdb, T=10)[2] == 1
    assert metrics.per_structure(pred, true, pdb, T=13)[2] == 0


def test_degenerate_is_a_numerical_test_not_a_practical_one():
    """Mutant: tolerance 1e-12 -> 1e-2. A complex whose spread is under 1% of its scale would
    then be dropped from the headline mean while ``n_degenerate`` reports the drop as
    legitimate. Small real spread is signal; only rounding-level spread is degenerate."""
    from mulan.metrics.core import degenerate
    assert not degenerate(100.0 + np.arange(10) * 0.05)
    assert degenerate(np.full(10, 5.0))
    assert degenerate(5.0 + np.arange(10) * 1e-14)


def test_precision_recall_threshold_is_inclusive():
    """Mutant: ``true >= thr`` -> ``true > thr``. The function's contract is ``>=``, and a
    threshold that is inclusive in one caller and exclusive in another moves the strong-call
    columns on any row that lands exactly on it."""
    precision, recall, n_pos = metrics.precision_recall_at_k(
        pred=np.array([3.0, 2.0, 1.0]), true=np.array([2.0, 2.0, 0.0]), k=2, thr=2.0)
    assert (precision, recall, n_pos) == (1.0, 1.0, 2)


def test_corrected_error_of_an_exact_linear_map_is_zero():
    """Mutant: OLS slope from ``np.cov(bias=True)`` -> ``bias=False``, which mismatches the
    ``np.var`` denominator. The selftest against sklearn lives in ``rescore.py``, not here."""
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    true = 2.0 * pred + 1.0
    assert metrics.rmse_corr(pred, true) == pytest.approx(0.0, abs=1e-12)
    assert metrics.mae_corr(pred, true) == pytest.approx(0.0, abs=1e-12)


def test_exported_constants_and_helpers_are_what_the_scorers_assume():
    """Mutants: ``STRONG`` 2.0 -> 1.0, ``PS_THRESHOLDS`` swapped, ``pdb_of`` taking the last
    underscore segment. The shipped CSVs were produced with these values; a drift here would
    change every AUROC-strong, T5 and per-complex grouping silently."""
    assert metrics.STRONG == 2.0
    assert metrics.PS_THRESHOLDS == (10, 5)
    assert metrics.pdb_of("1AHW.AB.C_AB") == "1AHW.AB.C"
    assert metrics.pdb_of("1CSE.E.I_E_LI38S-GI32Y") == "1CSE.E.I"


def test_paired_delta_sign_is_a_minus_b():
    """Mutant: ``sa - sb`` -> ``sb - sa``. Every earlier paired fixture compared an arm with
    itself or with a degenerate copy, where the sign cannot show."""
    pdb = _blocks([12, 12, 12])
    true = np.tile(np.arange(12.0), 3)
    a, b = true.copy(), -true
    result = metrics.paired_delta(a, b, true, pdb, threshold=10, n_resamples=50, seed=0)
    assert result.point == pytest.approx(2.0)


def test_bootstrap_percentile_is_two_tailed():
    """Mutant: ``tail = (100 - percentile) / 2`` -> ``(100 - percentile)``. At percentile 0
    both tails meet at the median, so lower must equal upper; the mutant reports the interval
    inverted."""
    groups = np.arange(40)
    rng = np.random.default_rng(1)
    values = rng.normal(size=40)
    stat = lambda idx, labels=None: float(values[idx].mean())
    r0 = metrics.cluster_bootstrap(stat, groups, n_resamples=200, percentile=0.0, seed=0)
    assert r0.lower == pytest.approx(r0.upper)
    r95 = metrics.cluster_bootstrap(stat, groups, n_resamples=200, percentile=95.0, seed=0)
    r50 = metrics.cluster_bootstrap(stat, groups, n_resamples=200, percentile=50.0, seed=0)
    assert r95.lower <= r50.lower <= r0.lower <= r0.upper <= r50.upper <= r95.upper


def test_bootstrap_drops_non_finite_resamples_and_says_so():
    """Mutant: the ``np.isfinite(value)`` guard removed. A NaN draw would then poison the
    percentiles; the result must instead report fewer resamples than requested."""
    groups = np.arange(6)
    calls = {"n": 0}

    def stat(idx, labels=None):
        calls["n"] += 1
        return float("nan") if calls["n"] % 3 == 0 else float(len(idx))

    r = metrics.cluster_bootstrap(stat, groups, n_resamples=30, percentile=95.0, seed=0)
    assert r.n_resamples < 30
    assert np.isfinite(r.lower) and np.isfinite(r.upper)


def test_paired_delta_scores_a_complex_at_exactly_the_threshold():
    """Mutant: ``_mutually_scorable`` ``< threshold`` -> ``<=``. A complex with exactly T
    mutations qualifies for the headline; the paired test must not drop it."""
    pdb = _blocks([10, 10, 10])
    true = np.tile(np.arange(10.0), 3)
    a = true + np.linspace(0, 0.1, 30)
    b = true.copy()
    result = metrics.paired_delta(a, b, true, pdb, threshold=10, n_resamples=20, seed=0)
    assert result.point == pytest.approx(0.0)      # both arms rank every complex perfectly


def test_tier_fold_counts_match_the_shipped_directories():
    """Mutants: ``cath`` n_folds 1 -> 3, ``clustered_sp`` 3 -> 2. The existence loop only
    checks ``range(n_folds)``, so an undercount survived; the count must equal the directory
    count, and the CATH tier is a single fixed partition by construction."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert metrics.get_tier("cath").n_folds == 1
    for name in metrics.list_tiers():
        tier = metrics.get_tier(name)
        folds = sorted((root / tier.split_dir).glob("fold_*"))
        if not folds:
            continue                      # built locally, not shipped
        assert len(folds) == tier.n_folds, f"tier {name} declares {tier.n_folds} folds, {len(folds)} ship"


def test_clustered_tiers_map_to_the_homology_controlled_protocol():
    """Mutant: ``clustered_sp`` protocol -> ``bycomplex``, which would print "not
    homology-controlled" beside a CATH-comparable number."""
    for name in ("clustered_sp", "clustered_all", "mp_clustered", "s1102_clustered"):
        assert splits.get_protocol(metrics.get_tier(name).protocol).homology_controlled, name
    for name in ("bycomplex_sp", "bycomplex_all", "mp_bycomplex", "s1102_bycomplex"):
        assert not splits.get_protocol(metrics.get_tier(name).protocol).homology_controlled, name


def test_default_protocol_list_is_exactly_the_leakage_controlled_three():
    """Mutant: ``is_legacy`` keyed on the homology flag, which silently drops by-complex."""
    assert set(splits.list_protocols()) == {"bycomplex", "clustered_id60", "cath"}
    assert set(splits.list_protocols(include_legacy=True)) == {
        "legacy_cv10", "bycomplex", "clustered_id60", "cath"}


def test_shipped_splits_hold_out_whole_complexes():
    """The split invariant itself, over the shipped data rather than over the protocol flags:
    in every fold no held-out unit is in both train and test, and none is in the test set of two
    folds. The unit is the label before the first underscore of column 0 -- the interface
    ``code.g1.g2`` on the full-SKEMPI tiers, the bare PDB code on the S1102 tiers, which carry
    no chain groups. This is what ``README.md`` §Tests promises the suite covers."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent

    def complexes(path):
        return {line.split("\t")[0].split("_")[0] for line in path.read_text().splitlines()
                if line.strip()}

    checked = 0
    for name in metrics.list_tiers():
        tier = metrics.get_tier(name)
        split_dir = root / tier.split_dir
        if not any(split_dir.glob("fold_*")):
            continue
        tests = []
        for f in range(tier.n_folds):
            fold = split_dir / f"fold_{f}"
            train = complexes(fold / f"{tier.basename}_train.tsv")
            test = complexes(fold / f"{tier.basename}_test.tsv")
            assert test, f"{name} fold {f}: empty test set"
            assert not (train & test), f"{name} fold {f}: {sorted(train & test)[:5]} in train and test"
            val = fold / f"{tier.basename}_val.tsv"
            if val.exists():
                assert not (complexes(val) & test), f"{name} fold {f}: val leaks into test"
            tests.append(test)
        for i in range(len(tests)):
            for j in range(i + 1, len(tests)):
                assert not (tests[i] & tests[j]), f"{name}: folds {i} and {j} share test complexes"
        checked += 1
    assert checked == 8, f"expected the eight shipped tiers, checked {checked}"
