"""Constants used in the package."""

import os 


# Single letter, three letter, and full amino acid names.
aa_names = (
    ('A', 'ALA', 'alanine'),
    ('R', 'ARG', 'arginine'),
    ('N', 'ASN', 'asparagine'),
    ('D', 'ASP', 'aspartic acid'),
    ('C', 'CYS', 'cysteine'),
    ('E', 'GLU', 'glutamic acid'),
    ('Q', 'GLN', 'glutamine'),
    ('G', 'GLY', 'glycine'),
    ('H', 'HIS', 'histidine'),
    ('I', 'ILE', 'isoleucine'),
    ('L', 'LEU', 'leucine'),
    ('K', 'LYS', 'lysine'),
    ('M', 'MET', 'methionine'),
    ('F', 'PHE', 'phenylalanine'),
    ('P', 'PRO', 'proline'),
    ('S', 'SER', 'serine'),
    ('T', 'THR', 'threonine'),
    ('W', 'TRP', 'tryptophan'),
    ('Y', 'TYR', 'tyrosine'),
    ('V', 'VAL', 'valine'),
    # Extended AAs
    ('B', 'ASX', 'asparagine or aspartic acid'),
    ('Z', 'GLX', 'glutamine or glutamic acid'),
    ('X', 'XAA', 'Any'),
    ('J', 'XLE', 'Leucine or isoleucine'),
)

# Indices of standard amino acids in `aa_names`.
standard_indices = tuple(range(20))

# Single letter codes of standard amino acids.
standard_aas = tuple(aa_names[i][0] for i in standard_indices)
AAs = tuple(sorted(standard_aas))

# aa_to_idx and idx_to_aa
aa2idx = dict(zip(AAs, standard_indices))
idx2aa = {v: k for k, v in aa2idx.items()}

# dictionaries for aas names conversion
one2three = dict(aa_names[i][:2] for i in standard_indices)
three2one = {v: k for k, v in one2three.items()}


# Models names and paths
_BASE_DIR = os.environ.get("MULAN", os.getcwd())
_DEFAULT_MODELS_DIR = os.path.join(_BASE_DIR, "models/pretrained")
MODELS_DIR = os.environ.get("MULAN_MODELS_PATH", _DEFAULT_MODELS_DIR)
MODELS = {
    "mulan-esm": f"{MODELS_DIR}/mulan_esm.ckpt",
    "mulan-esm-multiple": f"{MODELS_DIR}/mulan_esm_multiple.ckpt",
    "imulan-esm": f"{MODELS_DIR}/imulan_esm.ckpt",
    "mulan-ankh": f"{MODELS_DIR}/mulan_ankh.ckpt",
    "imulan-ankh": f"{MODELS_DIR}/imulan_ankh.ckpt",
    "mulan-ankh-multiple": f"{MODELS_DIR}/mulan_ankh_multiple.ckpt",
}

# PLMs encoders and HuggingFace Hub ids.
#
# Derived from `mulan.plm.registry`, which is the single source of truth for every backbone —
# model id, embedding width, loader, input convention and trimming rule. This mapping is kept
# because existing code (and the `gen_saprot_emb.py` generator) reads it for the hub id, but it
# is now a *view*: add a backbone in the registry, not here.
from mulan.plm.registry import hub_ids as _hub_ids

PLM_ENCODERS = _hub_ids()
