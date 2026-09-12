"""FoldX subsystem: the contracts that, if broken, are silent.

The merge is the dangerous part. Its failures do not raise — they produce a correctly-shaped
split file whose numbers are wrong, and the coverage counter that is supposed to catch that
cannot, because coverage counts keys and the damage is in values. So these tests pin the
numerics and the guard, not just the shapes.

The end-to-end proof that the unified merger reproduces the shipped split files byte-identically
lives outside the suite, because it needs the FoldX result store and the SKEMPI table, neither of
which is small enough to fixture. It is recorded in the commit that introduced the module: all 60
S1102 files and all 18 full-SKEMPI files identical, coverage 99.2%, 54 guard denials.
"""

from __future__ import annotations

import json

import pytest

from mulan.foldx import GuardSpec, StoreSpec, load_store, merge_foldx
from mulan.foldx.merge import CLIP, KeyMapper, fit_standardizer, standardize
from skempi_foldx.terms import N_TERMS, SCALAR_TERM, TERMS


# ============================================================================================
# The term contract
# ============================================================================================

def test_scalar_arm_is_column_zero():
    """The scalar and decomposed arms must not be able to drift apart."""
    assert TERMS[0] == SCALAR_TERM
    assert len(TERMS) == N_TERMS == 12


def _vec(x):
    return [x] * N_TERMS


def test_fit_uses_only_rows_that_joined():
    means, stds = fit_standardizer([_vec(1.0), None, _vec(3.0), None])
    assert means[0] == pytest.approx(2.0)          # unjoined rows must not pull the mean to 0
    assert stds[0] == pytest.approx(2.0 ** 0.5)    # sample std (n-1), not population


def test_missing_rows_become_the_training_mean():
    means, stds = fit_standardizer([_vec(1.0), _vec(3.0)])
    assert standardize(None, means, stds) == [0.0] * N_TERMS


def test_zero_variance_does_not_divide_by_zero():
    means, stds = fit_standardizer([_vec(5.0), _vec(5.0)])
    assert stds[0] == 1.0
    assert standardize(_vec(5.0), means, stds)[0] == pytest.approx(0.0)


def test_single_row_fit_is_safe():
    means, stds = fit_standardizer([_vec(2.0)])
    assert means[0] == pytest.approx(2.0) and stds[0] == 1.0


def test_empty_fit_does_not_crash():
    """Two of the six mergers raised ZeroDivisionError here; the others wrote silent zeros."""
    means, stds = fit_standardizer([None, None])
    assert means == [0.0] * N_TERMS and stds == [1.0] * N_TERMS


def test_values_are_clipped():
    means, stds = fit_standardizer([_vec(0.0), _vec(1.0)])
    assert standardize(_vec(1e6), means, stds)[0] == CLIP
    assert standardize(_vec(-1e6), means, stds)[0] == -CLIP


# ============================================================================================
# The grouping guard
# ============================================================================================

def _skempi_csv(tmp_path, rows):
    path = tmp_path / "skempi.csv"
    path.write_text("#Pdb;x;Mutation(s)_cleaned\n" + "\n".join(rows) + "\n")
    return path


def test_guard_denies_a_secondary_interface_grouping(tmp_path):
    """The 3SE4 case: SKEMPI lists one PDB under two groupings, FoldX computed the first.
    Rows of the second must be denied, not handed the first's energies."""
    guard = GuardSpec(skempi_csv=_skempi_csv(tmp_path, ["3SE4_B_C;;XA1A", "3SE4_B_A;;XA1A"])).load()
    assert not guard.denies("3SE4.B.C")     # the grouping FoldX actually used
    assert guard.denies("3SE4.B.A")         # the other one


def test_guard_is_strict_by_default(tmp_path):
    """Previously a missing file disabled the guard silently and coverage went UP."""
    with pytest.raises(FileNotFoundError, match="silently receive"):
        GuardSpec(skempi_csv=tmp_path / "absent.csv").load()
    GuardSpec(skempi_csv=tmp_path / "absent.csv", strict=False).load()   # opt out explicitly


def test_guard_exclude_list(tmp_path):
    path = tmp_path / "exclude.tsv"
    path.write_text("# header\n2C5D.AB.CD\t x\n3SE3.B.C\n")
    guard = GuardSpec(exclude_tsv=path).load()
    assert guard.denies("2C5D.AB.CD") and guard.denies("3SE3.B.C")
    assert not guard.denies("1AHW.AB.C")


# ============================================================================================
# Key mapping
# ============================================================================================

def _mapping(seqs):
    return {ch: {"seq": s, "resseq2idx": {}} for ch, s in seqs.items()}


def test_group_remap_applies_the_chain_offset():
    """A mutation on the second chain of a group is offset by the first chain's length."""
    mapper = KeyMapper(lambda code: _mapping({"H": "A" * 10, "L": "A" * 20, "C": "A" * 5}))
    # group1 = "HL": H occupies 1..10, L occupies 11..30.
    assert mapper.group_remap("YL5A", ("HL", "C"), "1XXX") == "YA15A"
    assert mapper.group_remap("YH5A", ("HL", "C"), "1XXX") == "YA5A"
    assert mapper.group_remap("YC3A", ("HL", "C"), "1XXX") == "YB3A"


def test_group_remap_rejects_insertion_codes_and_out_of_range():
    mapper = KeyMapper(lambda code: _mapping({"H": "A" * 10, "C": "A" * 5}))
    assert mapper.group_remap("YH116AA", ("H", "C"), "1XXX") is None   # insertion code
    assert mapper.group_remap("YH99A", ("H", "C"), "1XXX") is None     # past the chain end
    assert mapper.group_remap("YZ5A", ("H", "C"), "1XXX") is None      # chain in neither group


def test_key_cache_is_instance_scoped():
    """A module-global cache keyed only by PDB code made a second directory silently reuse
    the first one's mapping."""
    a = KeyMapper(lambda code: _mapping({"H": "A" * 10, "C": "A" * 5}))
    b = KeyMapper(lambda code: _mapping({"H": "A" * 50, "C": "A" * 5}))
    assert a.group_remap("YC1A", ("H", "C"), "1XXX") == "YB1A"
    assert b.group_remap("YC1A", ("H", "C"), "1XXX") == "YB1A"
    assert a.group_remap("YH40A", ("H", "C"), "1XXX") is None    # only 10 residues
    assert b.group_remap("YH40A", ("H", "C"), "1XXX") == "YA40A"  # 50 residues


# ============================================================================================
# SKEMPI role -> author mapping
# ============================================================================================

def _write_store(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    for pdb, muts in records.items():
        (directory / f"{pdb}.json").write_text(json.dumps({"muts": muts, "meta": {}}))


def _fixture_split(tmp_path):
    fold = tmp_path / "split" / "fold_0"
    fold.mkdir(parents=True)
    (fold / "d_train.tsv").write_text(
        "1XXX.A.B_A\t1XXX.A.B_B\tMA1A\t1.0\n"
        "1XXX.A.B_A\t1XXX.A.B_B\tMA2A\t2.0\n"
    )
    (fold / "d_test.tsv").write_text("1XXX.A.B_A\t1XXX.A.B_B\tMA3A\t3.0\n")
    return tmp_path / "split"


def test_merge_writes_both_arms_consistently(tmp_path):
    src = _fixture_split(tmp_path)
    store = tmp_path / "store"
    _write_store(store, {"1XXX": {
        "MA1A": {t: 1.0 for t in TERMS},
        "MA2A": {t: 3.0 for t in TERMS},
        "MA3A": {t: 5.0 for t in TERMS},
    }})
    report = merge_foldx(
        src=src, stores=[StoreSpec(path=store, keying="raw")],
        out_scalar=tmp_path / "sc", out_dec=tmp_path / "dec",
        guard=GuardSpec(strict=False),
    )
    assert report.covered == report.total == 3

    scalar = (tmp_path / "sc" / "fold_0" / "d_train.tsv").read_text().splitlines()
    dec = (tmp_path / "dec" / "fold_0" / "d_train.tsv").read_text().splitlines()
    assert len(scalar[0].split("\t")) == 5
    assert len(dec[0].split("\t")) == 4 + N_TERMS
    # The scalar arm must equal column 0 of the decomposed arm, always.
    for s, d in zip(scalar, dec):
        assert s.split("\t")[4] == d.split("\t")[4]


def test_merge_standardizes_on_train_only(tmp_path):
    """The test row is far outside the training range; it must be standardized with the
    training statistics, not with statistics that saw it."""
    src = _fixture_split(tmp_path)
    store = tmp_path / "store"
    _write_store(store, {"1XXX": {
        "MA1A": {t: 1.0 for t in TERMS},
        "MA2A": {t: 3.0 for t in TERMS},
        "MA3A": {t: 100.0 for t in TERMS},
    }})
    merge_foldx(src=src, stores=[StoreSpec(path=store, keying="raw")],
                out_scalar=tmp_path / "sc", guard=GuardSpec(strict=False))
    train = (tmp_path / "sc" / "fold_0" / "d_train.tsv").read_text().splitlines()
    test = (tmp_path / "sc" / "fold_0" / "d_test.tsv").read_text().splitlines()
    # train mean 2, sample std sqrt(2) -> +/-0.70711
    assert float(train[0].split("\t")[4]) == pytest.approx(-0.70711, abs=1e-5)
    assert float(test[0].split("\t")[4]) == CLIP     # way out of range -> clipped


def test_uncovered_row_is_written_as_the_training_mean(tmp_path):
    src = _fixture_split(tmp_path)
    store = tmp_path / "store"
    _write_store(store, {"1XXX": {
        "MA1A": {t: 1.0 for t in TERMS},
        "MA2A": {t: 3.0 for t in TERMS},
    }})   # MA3A absent
    report = merge_foldx(src=src, stores=[StoreSpec(path=store, keying="raw")],
                         out_scalar=tmp_path / "sc", guard=GuardSpec(strict=False))
    assert (report.covered, report.total) == (2, 3)
    test = (tmp_path / "sc" / "fold_0" / "d_test.tsv").read_text().splitlines()
    assert float(test[0].split("\t")[4]) == 0.0


def test_incomplete_term_record_is_rejected(tmp_path):
    """One merger admitted a record carrying only Interaction Energy, which would feed the
    decomposed arm eleven zeros indistinguishable from real values."""
    src = _fixture_split(tmp_path)
    store = tmp_path / "store"
    _write_store(store, {"1XXX": {"MA1A": {"Interaction Energy": 1.0}}})
    report = merge_foldx(src=src, stores=[StoreSpec(path=store, keying="raw")],
                         out_dec=tmp_path / "dec", guard=GuardSpec(strict=False))
    assert report.covered == 0

    report = merge_foldx(src=src,
                         stores=[StoreSpec(path=store, keying="raw", require_all_terms=False)],
                         out_dec=tmp_path / "dec2", guard=GuardSpec(strict=False))
    assert report.covered == 1


def test_output_fold_number_follows_the_source_fold_number(tmp_path):
    """The output index was the enumeration position of a sorted glob. With >=10 folds the
    lexicographic sort scrambles it (fold_10 before fold_2), a gap shifts every later fold down,
    and a stray `fold_*` entry displaces everything after it — joining the FoldX channel for
    fold_k to a different fold's rows, silently, with perfect shapes and coverage."""
    import json
    src = tmp_path / "src"
    for k in (0, 1, 2, 10, 11):                       # non-contiguous AND double-digit
        d = src / f"fold_{k}"; d.mkdir(parents=True)
        (d / "d_train.tsv").write_text(f"1XXX.A.B_A\t1XXX.A.B_B\tMA{k}A\t{k}.0\n")
        (d / "d_test.tsv").write_text(f"1XXX.A.B_A\t1XXX.A.B_B\tMA{k}A\t{k}.0\n")
    (src / "fold_0_backup").mkdir()                   # stray entry
    store = tmp_path / "store"; store.mkdir()
    (store / "1XXX.json").write_text(json.dumps(
        {"muts": {f"MA{k}A": {t: float(k) for t in TERMS} for k in (0, 1, 2, 10, 11)}, "meta": {}}))

    merge_foldx(src=src, stores=[StoreSpec(path=store, keying="raw")],
                out_scalar=tmp_path / "out", guard=GuardSpec(strict=False))
    for k in (0, 1, 2, 10, 11):
        out = tmp_path / "out" / f"fold_{k}" / "d_test.tsv"
        assert out.exists(), f"fold_{k} must be written under its own number"
        assert f"MA{k}A" in out.read_text(), f"fold_{k} holds another fold's rows"
    assert not (tmp_path / "out" / "fold_3").exists(), "no fold may be invented by renumbering"


def test_a_single_nan_term_does_not_saturate_the_fold(tmp_path):
    """`min(clip, nan)` returns clip, so one non-finite term was written as the MAXIMUM-signal
    value — and if it reached the fit, the whole column went NaN and every row in train, val and
    test saturated to +4.0 while coverage reported 100%."""
    means, stds = fit_standardizer([[1.0] * N_TERMS, [3.0] * N_TERMS])
    vector = [1.0] * N_TERMS; vector[4] = float("nan")
    z = standardize(vector, means, stds)
    assert z[4] == 0.0, "a non-finite term must fall back to the training mean, not to +clip"
    assert all(v == pytest.approx(-0.70711, abs=1e-4) for i, v in enumerate(z) if i != 4)


def test_a_nan_row_is_excluded_from_the_fit(tmp_path):
    means, stds = fit_standardizer([[float("nan")] * N_TERMS, [1.0] * N_TERMS, [3.0] * N_TERMS])
    assert means[0] == pytest.approx(2.0) and stds[0] == pytest.approx(2.0 ** 0.5)


def test_complex_id_survives_a_mutation_suffix():
    """Augmented splits label reverse rows `1CSE.E.I_E_LI38S-GI32Y`. `rsplit` cut that to
    `1CSE.E.I_E`, so the guard compared ("E","I_E") against ("E","I") and denied every reverse
    row — a tier reporting ~50% coverage that read as normal partial coverage."""
    from mulan.foldx.merge import _complex_id
    assert _complex_id("1CSE.E.I_E") == "1CSE.E.I"
    assert _complex_id("1CSE.E.I_E_LI38S-GI32Y") == "1CSE.E.I"
    assert _complex_id("1AHW.AB.C_AB") == "1AHW.AB.C"


def test_guard_refuses_a_present_but_empty_grouping_file(tmp_path):
    """The missing-file arm was fixed; the parses-to-nothing arm still failed open."""
    path = tmp_path / "skempi.csv"
    path.write_text("#Pdb;x;Mutation(s)_cleaned\n")          # header only
    with pytest.raises(ValueError, match="no trusted chain groupings"):
        GuardSpec(skempi_csv=path).load()
    GuardSpec(skempi_csv=path, strict=False).load()           # explicit opt-out still allowed


# ------------------------------------------------------------------- intractable exclusions

def test_curation_block_still_tracks_the_upstream_exclusion_registry():
    """Cross-repo invariant. Curation lives here; the exclusion registry now lives in
    skempi-foldx. They must not drift -- an id blocked for either reason must be blocked for
    both, and the split made that a dependency rather than a shared file."""
    import importlib.util
    from pathlib import Path

    from skempi_foldx import INTRACTABLE

    root = Path(__file__).resolve().parents[1]
    for name in ("build_skempi_full", "audit_multipoint_foldx"):
        path = root / "experiments" / "full_skempi_seqonly" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.BLOCK == set(INTRACTABLE), f"{name}.BLOCK has drifted from skempi_foldx"


# ============================================================================================
# Mutation-testing survivors (round 6): the group-remap keying every full-SKEMPI tier uses had
# no end-to-end coverage, and every fixture above carries one value in all twelve terms.
# ============================================================================================

def test_decomposed_terms_are_standardized_per_term():
    """Mutant: ``/ stds[j]`` -> ``/ stds[0]``. With identical values in every term the scalar
    arm still equals column 0 and the mutant is invisible; a rebuilt decomposed split would be
    wrong in eleven of twelve columns."""
    means, stds = fit_standardizer([[1.0] * 11 + [10.0], [3.0] * 11 + [30.0]])
    out = standardize([3.0] * 11 + [30.0], means, stds)
    assert out[11] == pytest.approx(out[0])
    assert out[0] == pytest.approx(0.70711, abs=1e-5)


def _author_chain_split(tmp_path):
    """A split whose groups are author chains H and C, so the store's author-chain keys must be
    remapped through the chain offsets to join."""
    fold = tmp_path / "split" / "fold_0"
    fold.mkdir(parents=True)
    (fold / "d_train.tsv").write_text(
        "1XXX.H.C_H\t1XXX.H.C_C\tMA1A\t1.0\n"
        "1XXX.H.C_H\t1XXX.H.C_C\tMA2A\t2.0\n"
    )
    (fold / "d_test.tsv").write_text("1XXX.H.C_H\t1XXX.H.C_C\tMB1A\t3.0\n")
    return tmp_path / "split"


def _hc_mapping(code):
    return _mapping({"H": "A" * 10, "C": "A" * 5})


def test_group_remap_keying_joins_end_to_end(tmp_path):
    """Mutant: chain groups never parsed from the split. Every remap then returns None, the
    raw-key fallback misses, and the tier reports a plausible partial coverage with shifted
    standardization."""
    src = _author_chain_split(tmp_path)
    store = tmp_path / "store"
    _write_store(store, {"1XXX": {
        "MH1A": {t: 1.0 for t in TERMS},     # chain H, residue 1  -> A1
        "MH2A": {t: 3.0 for t in TERMS},     # chain H, residue 2  -> A2
        "MC1A": {t: 5.0 for t in TERMS},     # chain C, residue 1  -> B1
    }})
    report = merge_foldx(src=src, stores=[StoreSpec(path=store, keying="group_remap")],
                         out_scalar=tmp_path / "sc", guard=GuardSpec(strict=False),
                         mapping_loader=_hc_mapping)
    assert report.covered == report.total == 3


def test_remapped_key_wins_over_a_colliding_raw_key(tmp_path):
    """Mutant: the raw-key pass writes with ``index[key] = vector`` instead of ``setdefault``.
    A raw FoldX key that happens to spell a split key would then overwrite the authoritative
    remapped value with a different mutation's energies."""
    src = _author_chain_split(tmp_path)
    store = tmp_path / "store"
    _write_store(store, {"1XXX": {
        "MH1A": {t: 1.0 for t in TERMS},     # remaps to A1: authoritative
        "MA1A": {t: 9.0 for t in TERMS},     # raw key that collides with the split's A1
        "MH2A": {t: 3.0 for t in TERMS},
    }})
    merge_foldx(src=src, stores=[StoreSpec(path=store, keying="group_remap")],
                out_scalar=tmp_path / "sc", guard=GuardSpec(strict=False),
                mapping_loader=_hc_mapping)
    train = (tmp_path / "sc" / "fold_0" / "d_train.tsv").read_text().splitlines()
    # train values 1 and 3 -> z = -0.70711 for the first row; the colliding 9.0 would give +0.7.
    assert float(train[0].split("\t")[4]) == pytest.approx(-0.70711, abs=1e-5)


def test_multi_point_rows_do_not_join_a_single_point_store(tmp_path):
    """Mutant: the ``is_multi and json_field == "muts"`` skip removed. A multi-point row whose
    comma-joined string happens to be a key in a single-point store would join it."""
    fold = tmp_path / "split" / "fold_0"
    fold.mkdir(parents=True)
    (fold / "d_train.tsv").write_text("1XXX.A.B_A\t1XXX.A.B_B\tMA1A,MA2A\t1.0\n")
    (fold / "d_test.tsv").write_text("1XXX.A.B_A\t1XXX.A.B_B\tMA3A,MA4A\t3.0\n")
    store = tmp_path / "store"
    _write_store(store, {"1XXX": {"MA1A,MA2A": {t: 1.0 for t in TERMS},
                                  "MA3A,MA4A": {t: 2.0 for t in TERMS}}})
    report = merge_foldx(src=tmp_path / "split", stores=[StoreSpec(path=store, keying="raw")],
                         out_scalar=tmp_path / "sc", guard=GuardSpec(strict=False))
    assert report.covered == 0


# =====================================================================================
# The S1102 mergers (experiments/foldx_s1102/), run as a reader would run them
# =====================================================================================

def test_s1102_mergers_run_from_a_clone_on_shipped_inputs(tmp_path):
    """The two S1102 mergers must work with their defaults from a checkout: the S1102 paper split
    under ``experiments/embedding_sweep/splits/paper_seed42`` and the store under
    ``data/foldx/results_sp``. Checks the shape a trainer relies on (5 and 16 columns, the first
    four untouched), full coverage on S1102, and that the channel really is standardized on the
    train fold and clipped."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = root / "experiments" / "embedding_sweep" / "splits" / "paper_seed42"
    for script, tag_dir, width in (("merge_foldx.py", "splits_ankh_foldx", 5),
                                   ("merge_foldx_decomposed.py", "splits_ankh_foldxdec", 16)):
        out = subprocess.run([sys.executable, str(root / "experiments" / "foldx_s1102" / script),
                              "--out-dir", str(tmp_path)], cwd=root, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "TOTAL coverage 11000/11000" in out.stdout, out.stdout
        folds = sorted((tmp_path / tag_dir).glob("fold_*"))
        assert len(folds) == 10
        for fold in folds:
            for split in ("train", "val", "test"):
                merged = (fold / f"S1102_filtered_{split}.tsv").read_text().splitlines()
                source = (src / fold.name / f"S1102_filtered_{split}.tsv").read_text().splitlines()
                assert len(merged) == len(source) > 0
                for m, s_ in zip(merged, source):
                    mf, sf = m.split("\t"), s_.split("\t")
                    assert len(mf) == width and mf[:4] == sf[:4]
                    assert all(-CLIP <= float(v) <= CLIP for v in mf[4:])
    # The channel is the store's Interaction Energy standardized on the train fold's own values
    # (sample std) and clipped -- recomputed here from the store for fold_0, every row, both
    # scripts. Clipping is why the written column does not have unit variance.
    store = {}
    for fold in ("fold_0",):
        rows = {}
        for split in ("train", "val", "test"):
            rows[split] = [l.split("\t") for l in
                           (src / fold / f"S1102_filtered_{split}.tsv").read_text().splitlines()]
        def energy(r):
            pdb = r[0].split("_")[0]
            if pdb not in store:
                store[pdb] = json.loads((root / "data" / "foldx" / "results_sp" / f"{pdb}.json").read_text())["muts"]
            return store[pdb][r[2]][SCALAR_TERM]
        train_e = [energy(r) for r in rows["train"]]
        mean = sum(train_e) / len(train_e)
        std = (sum((e - mean) ** 2 for e in train_e) / (len(train_e) - 1)) ** 0.5
        for tag_dir in ("splits_ankh_foldx", "splits_ankh_foldxdec"):
            for split in ("train", "val", "test"):
                written = [l.split("\t")[4] for l in
                           (tmp_path / tag_dir / fold / f"S1102_filtered_{split}.tsv").read_text().splitlines()]
                for r, w in zip(rows[split], written):
                    z = max(-CLIP, min(CLIP, (energy(r) - mean) / std))
                    assert float(w) == pytest.approx(z, abs=1e-5), (tag_dir, split, r[:3], w, z)
