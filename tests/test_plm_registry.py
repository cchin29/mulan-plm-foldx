"""The registry must reproduce the pre-refactor behaviour exactly.

Tens of gigabytes of embedding caches were produced by the code this replaces. If the new
dispatch preprocesses a sequence differently, or trims a different number of leading tokens, a
regenerated embedding silently disagrees with a cached one — and nothing downstream would notice
until a metric moved for no visible reason.

So the reference implementations below are the *original* inline branches, copied verbatim from
the pre-refactor ``mulan/utils.py``. Every test compares against them rather than against an
expectation written from memory.
"""

from __future__ import annotations

import os
import re

import pytest
import torch

from mulan import plm
from mulan.plm.backends.hf import HFBackend
from mulan.plm.registry import PlmSpec


# ============================================================================================
# Reference: the original implementation, verbatim.
# ============================================================================================

def legacy_preprocess(sequence: str, name_or_path: str) -> str:
    """`embed_sequence`'s original input handling."""
    sequence = sequence.upper()
    sequence = re.sub(r"[UZOB]", "X", sequence)
    is_prostt5 = "ProstT5" in name_or_path
    is_ankh3 = "ankh3" in name_or_path.lower()
    if is_prostt5:
        sequence = "<AA2fold> " + " ".join(sequence)
    elif is_ankh3:
        sequence = os.environ.get("ANKH3_PREFIX", "[NLU]") + sequence
    elif "Rostlab/prot" in name_or_path:
        sequence = " ".join(sequence)
    return sequence


def legacy_trim(embedding: torch.Tensor, name_or_path: str, n_residues: int) -> torch.Tensor:
    """`embed_sequence`'s original post-special-mask trimming."""
    if "ProstT5" in name_or_path:
        return embedding[:, 1:, :]
    if "ankh3" in name_or_path.lower():
        return embedding[:, embedding.shape[1] - n_residues:, :]
    return embedding


def legacy_loader_choice(model_id: str) -> str:
    """`load_pretrained_plm`'s original model-class selection."""
    if "t5" in model_id.lower() or "ankh" in model_id.lower():
        return "T5EncoderModel+AutoTokenizer" if "ankh" in model_id.lower() \
            else "T5EncoderModel+T5Tokenizer"
    if "esmc" in model_id.lower():
        return "AutoModelForMaskedLM+AutoTokenizer"
    return "AutoModel+AutoTokenizer"


# The 14 tags that existed in PLM_ENCODERS before the refactor, with their ids.
LEGACY_ENCODERS = {
    "esm": "facebook/esm2_t36_3B_UR50D",
    "ankh": "ElnaggarLab/ankh-large",
    "esm_35M": "facebook/esm2_t12_35M_UR50D",
    "esm_650M": "facebook/esm2_t33_650M_UR50D",
    "ankh_base": "ElnaggarLab/ankh-base",
    "protbert": "Rostlab/prot_bert",
    "prott5_xl_half": "Rostlab/prot_t5_xl_half_uniref50-enc",
    "prostt5": "Rostlab/ProstT5",
    "esmc_6b": "EvolutionaryScale/esmc-6b-2024-12",
    "aido": "genbio-ai/AIDO.Protein-16B",
    "ankh3_large": "ElnaggarLab/ankh3-large",
    "ankh3_xl": "ElnaggarLab/ankh3-xl",
    "saprot": "westlake-repl/SaProt_650M_AF2",
    "saprot_1.3b": "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI",
}

SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
SEQ_NONCANONICAL = "MKTUAYZIABKQOR"


# ============================================================================================
# Registry integrity
# ============================================================================================

def test_every_legacy_tag_still_resolves():
    for tag, model_id in LEGACY_ENCODERS.items():
        spec = plm.get_spec(tag)
        assert spec.model_id == model_id, f"{tag}: model id drifted"


def test_plm_encoders_view_is_a_superset_of_legacy():
    from mulan import constants

    for tag, model_id in LEGACY_ENCODERS.items():
        assert constants.PLM_ENCODERS[tag] == model_id


def test_backends_added_by_this_refactor_are_registered():
    """The backbones whose generators lived outside the repository."""
    for tag in ("esmc_600m", "esm3_sm_open_v1", "mint"):
        assert tag in plm.PLM_REGISTRY


def test_aliases_resolve_and_do_not_collide():
    assert plm.resolve_tag("saprot13b") == "saprot_1.3b"   # the two registries disagreed
    assert plm.resolve_tag("esmc600m") == "esmc_600m"
    assert plm.resolve_tag("esm3") == "esm3_sm_open_v1"
    assert plm.resolve_tag("ankh_large") == "ankh"
    # An alias must never shadow a canonical tag.
    for tag in plm.PLM_REGISTRY:
        assert plm.resolve_tag(tag) == tag


def test_unknown_tag_names_the_alternatives():
    with pytest.raises(KeyError, match="Unknown PLM"):
        plm.get_spec("definitely-not-a-model")


def test_dims_are_populated():
    for tag in plm.list_plms():
        assert plm.embedding_dim(tag) is not None, f"{tag} has no registered width"


# ============================================================================================
# Equivalence with the original inline branches
# ============================================================================================

@pytest.mark.parametrize("tag", sorted(LEGACY_ENCODERS))
def test_preprocessing_matches_legacy(tag):
    spec = plm.get_spec(tag)
    normalized = plm.normalize_sequence(SEQ)
    assert spec.preprocess(normalized) == legacy_preprocess(SEQ, spec.model_id)


@pytest.mark.parametrize("tag", sorted(LEGACY_ENCODERS))
def test_preprocessing_matches_legacy_on_noncanonical_residues(tag):
    spec = plm.get_spec(tag)
    normalized = plm.normalize_sequence(SEQ_NONCANONICAL)
    assert spec.preprocess(normalized) == legacy_preprocess(SEQ_NONCANONICAL, spec.model_id)


def test_noncanonical_substitution_is_applied():
    assert plm.normalize_sequence("mktUuaZob") == "MKTXXAXXX"


@pytest.mark.parametrize("tag", sorted(LEGACY_ENCODERS))
def test_loader_choice_matches_legacy(tag):
    """The backend field must select the same classes the substring branches did."""
    spec = plm.get_spec(tag)
    expected = legacy_loader_choice(spec.model_id)
    mapping = {
        "hf_t5": "T5EncoderModel+AutoTokenizer" if "ankh" in spec.model_id.lower()
                 else "T5EncoderModel+T5Tokenizer",
        "hf_auto": "AutoModel+AutoTokenizer",
        "hf_esmc": "AutoModelForMaskedLM+AutoTokenizer",
        # These were mis-served by the generic branch before: the AutoModel path cannot load
        # AIDO (needs trust_remote_code) and cannot embed SaProt (needs 3Di input). They now
        # have dedicated backends, which is the intended behaviour change.
        "aido": "AutoModel+AutoTokenizer",
        "saprot": "AutoModel+AutoTokenizer",
        "esmc_6b_raw": "AutoModelForMaskedLM+AutoTokenizer",
    }
    assert mapping[spec.backend] == expected, (
        f"{tag}: backend {spec.backend!r} would select a different loader than the original"
    )


def test_ankh3_prefix_env_override_still_works():
    spec = plm.get_spec("ankh3_large")
    assert spec.resolved_prefix() == "[NLU]"
    os.environ["ANKH3_PREFIX"] = "[S2S]"
    try:
        assert spec.resolved_prefix() == "[S2S]"
        assert spec.preprocess("MKT") == "[S2S]MKT"
    finally:
        del os.environ["ANKH3_PREFIX"]


# ============================================================================================
# End-to-end trimming, against a stub tokenizer/model pair
# ============================================================================================

class StubTokenizer:
    """Tokenizes the way the real ones do for our purposes: one token per residue, plus any
    prefix token, plus a trailing </s> that IS flagged special."""

    def __init__(self, name_or_path, spurious_unk=False):
        self.name_or_path = name_or_path
        self.spurious_unk = spurious_unk

    def _tokens(self, text):
        toks = []
        if text.startswith("<AA2fold> "):
            toks.append("<AA2fold>")
            text = text[len("<AA2fold> "):]
        elif text.startswith("[NLU]") or text.startswith("[S2S]"):
            if self.spurious_unk:
                toks.append("<unk>")   # older tokenizer builds emit this beside the prefix
            toks.append(text[:5])
            text = text[5:]
        toks += text.split(" ") if " " in text else list(text)
        return toks

    def __call__(self, text, **kw):
        toks = self._tokens(text) + ["</s>"]
        n = len(toks)
        mask = [0] * (n - 1) + [1]
        return _Batch({
            "input_ids": torch.arange(n).unsqueeze(0),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
            "special_tokens_mask": torch.tensor(mask).unsqueeze(0),
        })


class _Batch(dict):
    def to(self, device):
        return self


class StubModel:
    """Returns a hidden state whose value encodes the token index, so a trimming error is
    visible as a shifted sequence rather than as a shape that happens to match."""

    device = torch.device("cpu")

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        n = input_ids.shape[1]
        hidden = torch.arange(n, dtype=torch.float32).reshape(1, n, 1).repeat(1, 1, 4)
        return type("Out", (), {"last_hidden_state": hidden})()


def legacy_embed_sequence(model, tokenizer, sequence):
    """The original `embed_sequence`, verbatim, against the stubs."""
    sequence = sequence.upper()
    sequence = re.sub(r"[UZOB]", "X", sequence)
    n_residues = len(sequence)
    is_prostt5 = "ProstT5" in tokenizer.name_or_path
    is_ankh3 = "ankh3" in tokenizer.name_or_path.lower()
    if is_prostt5:
        sequence = "<AA2fold> " + " ".join(sequence)
    elif is_ankh3:
        sequence = os.environ.get("ANKH3_PREFIX", "[NLU]") + sequence
    elif "Rostlab/prot" in tokenizer.name_or_path:
        sequence = " ".join(sequence)
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True,
                       return_special_tokens_mask=True).to(model.device)
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden = outputs.hidden_states[-1]
    embedding = hidden[~inputs["special_tokens_mask"].bool()].unsqueeze(0)
    if is_prostt5:
        embedding = embedding[:, 1:, :]
    elif is_ankh3:
        embedding = embedding[:, embedding.shape[1] - n_residues:, :]
    return embedding


@pytest.mark.parametrize("tag", sorted(t for t in LEGACY_ENCODERS
                                       if plm.get_spec(t).backend.startswith("hf")))
@pytest.mark.parametrize("spurious_unk", [False, True])
def test_embed_matches_legacy_bit_for_bit(tag, spurious_unk):
    spec = plm.get_spec(tag)
    tok = StubTokenizer(spec.model_id, spurious_unk=spurious_unk)
    model = StubModel()

    new = HFBackend(spec, torch.device("cpu"), model, tok).embed(SEQ)
    old = legacy_embed_sequence(model, tok, SEQ)

    assert new.shape == old.shape, f"{tag}: shape differs"
    assert torch.equal(new, old), f"{tag}: values differ"
    assert new.shape[1] == len(SEQ), f"{tag}: expected one position per residue"


def test_legacy_entry_point_still_routes_correctly():
    """`embed_sequence(model, tokenizer, seq)` keeps working and picks up the registry."""
    from mulan.utils import embed_sequence

    spec = plm.get_spec("prostt5")
    tok = StubTokenizer(spec.model_id)
    out = embed_sequence(StubModel(), tok, SEQ)
    assert out.shape[1] == len(SEQ)
    assert torch.equal(out, legacy_embed_sequence(StubModel(), tok, SEQ))


def test_unregistered_model_falls_back_to_legacy_rules():
    """A backbone nobody registered must behave exactly as it did before."""
    from mulan.utils import embed_sequence

    tok = StubTokenizer("SomeLab/ProstT5-finetuned-v2")  # matches the old substring rule
    out = embed_sequence(StubModel(), tok, SEQ)
    assert torch.equal(out, legacy_embed_sequence(StubModel(), tok, SEQ))
    assert out.shape[1] == len(SEQ)


def test_shape_check_catches_a_wrong_trimming_rule():
    """`embed_checked` must fail loudly rather than cache a misaligned tensor."""
    wrong = PlmSpec(tag="wrong", backend="hf_auto", model_id="x/y", dim=4,
                    strip="leading_1")  # no prefix, so this deletes a real residue
    tok = StubTokenizer("x/y")
    with pytest.raises(RuntimeError, match="trimming rule"):
        HFBackend(wrong, torch.device("cpu"), StubModel(), tok).embed_checked(SEQ)


# ============================================================================================
# The id contract
# ============================================================================================

def test_mutant_labels_match_the_dataset_implementation(tmp_path):
    """`plm.ids` must name embeddings exactly as MulanDataset does, or caches miss."""
    from mulan.data import MulanDataset, MutatedComplex

    seqs = {"1ABC_A": "MKTAYIA", "1ABC_B": "QRQISFV"}
    rows = [
        ("1ABC_A", "1ABC_B", ("YA5A",)),
        ("1ABC_A", "1ABC_B", ("YA5A", "SB4G")),
        ("1ABC_A", "1ABC_B", ("SB4G",)),          # no chain-A mutation: trailing underscore
    ]
    complexes = [MutatedComplex(a, b, m) for a, b, m in rows]

    ds = MulanDataset.__new__(MulanDataset)
    ds.sequences = dict(seqs)
    ds._sequences_ids = []
    ds._fill_metadata(complexes)

    for (a, b, muts), ids in zip(rows, ds._sequences_ids):
        _, _, dataset_a, dataset_b = ids
        assert plm.mutant_labels(a, b, muts) == (dataset_a, dataset_b)


def test_enumerate_sequence_ids_covers_the_whole_table(tmp_path):
    fasta = tmp_path / "wt.fasta"
    fasta.write_text(">1ABC_A\nMKTAYIA\n>1ABC_B\nQRQISFV\n")
    table = tmp_path / "muts.tsv"
    table.write_text("1ABC_A\t1ABC_B\tYA5A\t1.2\n1ABC_A\t1ABC_B\tSB4G\t-0.3\n")

    ids = plm.enumerate_sequence_ids(str(table), str(fasta))
    assert set(ids) == {"1ABC_A", "1ABC_B", "1ABC_A_YA5A", "1ABC_B_",
                        "1ABC_A_", "1ABC_B_SB4G"}
    assert ids["1ABC_A_YA5A"] == "MKTAAIA"      # position 5, Y -> A
    assert ids["1ABC_A_"] == ids["1ABC_A"]      # unmutated chain keeps its sequence


def test_missing_ids_reports_what_is_absent(tmp_path):
    (tmp_path / "a.pt").touch()
    assert plm.missing_ids(["a", "b"], str(tmp_path)) == ["b"]
    assert plm.missing_ids(["a"], str(tmp_path / "nope")) == ["a"]


def test_load_hf_checks_the_transformers_version():
    """`load_hf` calls `check_transformers`, which two documents promise fires at runtime.

    It was exported from `mulan.plm`, documented in `docs/CODEBASE_OVERVIEW.md` as "checked at
    runtime by `mulan.plm.check_transformers`, which warns", and listed in `docs/USAGE.md` among
    the warnings a user will see -- while having no call site anywhere in the package. A bare
    `pip install .` resolves a transformers well outside the validated range, so the guard was
    documented as armed on exactly the install that needs it.

    A source grep was not enough: an audit defeated one by wrapping the call in ``if False:``, by
    moving it behind a branch nothing reaches, and by deleting it while leaving the word in the
    docstring -- which ``inspect.getsource`` returns. So this drives ``load_hf`` and asserts on
    the warning, which is the thing a user actually gets.
    """
    import warnings
    import transformers
    from mulan.plm.backends import hf
    from mulan.plm.registry import PLM_REGISTRY, get_spec

    class _Stub:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()
        def to(self, *a, **k):
            return self
        def eval(self):
            return self

    real_version = transformers.__version__
    real_model, real_tok = transformers.AutoModel, transformers.AutoTokenizer
    real_t5, real_t5tok = transformers.T5EncoderModel, transformers.T5Tokenizer
    transformers.AutoModel = _Stub
    transformers.AutoTokenizer = _Stub
    # The T5 branch loads the model through T5EncoderModel (and non-Ankh T5 ids their tokenizer
    # through T5Tokenizer), not through Auto*. Left unstubbed, the test reached the Hub (or a
    # 14 GB local cache) on every probe and silently swallowed the result.
    transformers.T5EncoderModel = _Stub
    transformers.T5Tokenizer = _Stub
    try:
        # One spec per backend that actually reaches load_hf, derived from the registry rather
        # than hardcoded. Driving a single spec let a refactor that moved the check into that
        # branch pass while the others went unguarded -- and the T5 path is the one the warning's
        # own text is about. A hardcoded trio was no better: it named esmc_600m for `hf_esmc`,
        # which is backend `esm_sdk` and never routes here at all. Only `hf_t5` and `hf_auto`
        # reach load_hf: `saprot` resolves to UnavailableBackend in load_backend before it.
        routed = {}
        for tag, cand in PLM_REGISTRY.items():
            routed.setdefault(cand.backend, tag)
        reachable = [routed[b] for b in ("hf_t5", "hf_auto") if b in routed]
        assert len(reachable) == 2, {
            "backends reaching load_hf that the registry can supply": sorted(routed),
            "note": "if a branch gained or lost specs, this test must follow it",
        }
        for tag in reachable:
            spec = get_spec(tag)
            for version, expected in (("4.44.2", 0), ("4.45.0", 1), ("5.1.0", 1), ("4.26.0", 1)):
                transformers.__version__ = version
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        hf.load_hf(spec, "cpu")
                    except Exception:
                        pass          # only the warning is under test, not the load
                hits = [w for w in caught
                        if issubclass(w.category, RuntimeWarning)
                        and "validated range" in str(w.message)]
                assert len(hits) == expected, {
                    "backbone": tag,
                    "backend": spec.backend,
                    "transformers": version,
                    "expected validated-range warnings": expected,
                    "got": len(hits),
                    "note": "load_hf must check on every backend branch, not just the one a "
                            "single-spec test happens to drive",
                }
    finally:
        transformers.__version__ = real_version
        transformers.AutoModel, transformers.AutoTokenizer = real_model, real_tok
        transformers.T5EncoderModel, transformers.T5Tokenizer = real_t5, real_t5tok


def test_embedding_widths_are_pinned():
    """Mutation-testing survivor: ``ankh`` dim 1536 -> 1024 passed the suite. The widths are
    what every cached embedding and every trained head were built against; a wrong width fails
    only at load time on a machine that has the cache."""
    from mulan.plm.registry import PLM_REGISTRY
    expected = {
        "esm": 2560, "esm_650M": 1280, "esm_35M": 480,
        "ankh": 1536, "ankh_base": 768, "ankh3_large": 1536, "ankh3_xl": 2560,
        "prostt5": 1024, "prott5_xl_half": 1024, "protbert": 1024,
        "saprot": 1280, "saprot_1.3b": 1280,
        "esmc_600m": 1152, "esmc_6b": 2560, "esm3_sm_open_v1": 1536,
        "aido": 2304, "mint": 1280,
    }
    assert {k: v.dim for k, v in PLM_REGISTRY.items()} == expected
