from typing import Dict, List, NamedTuple, Tuple, Optional

import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from mulan import utils


class MutatedComplex(NamedTuple):
    sequence_A: str
    sequence_B: str
    mutations: Tuple[str]


class MutatedComplexEmbeds(NamedTuple):
    seq1: torch.Tensor
    seq2: torch.Tensor
    mut_seq1: torch.Tensor
    mut_seq2: torch.Tensor


class MulanDataset(Dataset):
    """
    Dataset class for handling muttaion data for MuLAN models.
    Args:
        mutated_complexes (List[MutatedComplex]): A list of MutatedComplex objects representing the mutated complexes.
        wt_sequences (Dict[str, str]): A dictionary mapping sequence labels to wild-type sequences.
        embeddings_dir (str): The directory path where the embeddings are stored or will be generated.
        plm_model_name (str, optional): The name of the pre-trained language model to use for generating embeddings. Defaults to None.
        scores (List[float], optional): A list of scores associated with the mutated complexes. Defaults to None.
        zs_scores (List[float], optional): A list of Z-scores associated with the mutated complexes. Defaults to None.
    Attributes:
        sequences (Dict[str, str]): A dictionary mapping sequence labels to sequences, also including mutated sequences.
        embeddings_dir (str): The directory path where the embeddings are stored or will be generated.
        mutated_complexes (List[MutatedComplex]): A list of MutatedComplex objects representing the mutated complexes.
        zs_scores (List[float]): A list of Z-scores associated with the mutated complexes.
        scores (List[float]): A list of scores associated with the mutated complexes.
    """

    def __init__(
        self,
        mutated_complexes: List[MutatedComplex],
        wt_sequences: Dict[str, str],
        embeddings_dir: str,
        plm_model_name: str = None,
        scores: List[float] = None,
        zs_scores: List[float] = None,
        struct_context_dir: str = None,
        mint_pair: bool = False,
        interface_mask_dir: str = None,
    ):

        self.sequences = wt_sequences
        self.embeddings_dir = embeddings_dir
        self.mutated_complexes = mutated_complexes
        self.zs_scores = zs_scores
        self.scores = scores
        self.struct_context_dir = struct_context_dir
        # A1: per-complex interface contact mask dir (keyed s1__s2.pt, bool [L1,L2]); None
        # (default) => no mask emitted and the model's cross-attention stays dormant.
        self.interface_mask_dir = interface_mask_dir
        # A2 (MINT): embeddings are partner-context and keyed per (complex, mutation), not per
        # sequence label. In this mode `embeddings_dir` holds wt/<s1>__<s2>.pt and
        # mut/<s1>__<s2>__<muts>.pt bundles ({"1","2"} per-chain tensors) instead of {id}.pt files
        # — see experiments/mint/gen_mint_emb.py and docs/history/PLAN_MINT_A2.md §2. The cache MUST be complete
        # (no on-the-fly generation; MINT is out-of-band, like ESM C).
        self.mint_pair = mint_pair
        # In-memory cache of deserialized tensors, keyed by absolute file path. Without it, a
        # cached-embedding run re-runs torch.load on every __getitem__ (once per sample per epoch
        # per fold), re-deserializing the same files thousands of times; the unique working set
        # fits in RAM. Cached tensors are read-only downstream (collator/model produce new
        # tensors), so this is numerically identical. Disable with MULAN_EMB_CACHE=0.
        self._use_tensor_cache = os.environ.get("MULAN_EMB_CACHE", "1") != "0"
        self._tensor_cache: Dict[str, torch.Tensor] = {}
        self._sequences_ids = []
        self._fill_metadata(mutated_complexes)

        if self.mint_pair:
            # Out-of-band cache only; verify completeness up front for a clean error.
            self._check_mint_cache_complete()
        else:
            # generate embeddings if not provided
            all_ids = set([id_ for ids in self._sequences_ids for id_ in ids])
            provided_embeddings_ids = (
                [os.path.splitext(file)[0] for file in os.listdir(self.embeddings_dir)]
                if os.path.exists(self.embeddings_dir)
                else []
            )
            missing_ids = all_ids - set(provided_embeddings_ids)
            if missing_ids:
                if not plm_model_name:
                    raise ValueError(
                        "`plm_model_name` must be provided if embeddings were not pre-computed."
                    )
                self._generate_missing_embeddings(plm_model_name, missing_ids)

    def __len__(self):
        return len(self.mutated_complexes)

    def __getitem__(self, index):
        return {
            "data": self.mutated_complexes[index],
            "inputs_embeds": self._load_embeddings(index),
            "zs_scores": (
                torch.tensor(self.zs_scores[index], dtype=torch.float32)
                if self.zs_scores
                else None
            ),
            "labels": (
                torch.tensor(self.scores[index], dtype=torch.float32) if self.scores else None
            ),
            "struct_ctx": self._load_struct_context(index),
            "iface_mask": self._load_interface_mask(index),
        }

    def _load_tensor(self, path):
        """Deserialize a `.pt` tensor, memoizing by path so each file is read from disk once."""
        if not self._use_tensor_cache:
            return torch.load(path, weights_only=True)
        tensor = self._tensor_cache.get(path)
        if tensor is None:
            tensor = torch.load(path, weights_only=True)
            self._tensor_cache[path] = tensor
        return tensor

    def _load_struct_context(self, index):
        """B1: pooled WT structure context for the complex of `index` (keyed s1__s2)."""
        if not self.struct_context_dir:
            return None
        s1, s2, _ = self.mutated_complexes[index]
        return self._load_tensor(os.path.join(self.struct_context_dir, f"{s1}__{s2}.pt"))

    def _load_interface_mask(self, index):
        """A1: per-complex interface contact mask (bool [L1,L2], keyed s1__s2)."""
        if not self.interface_mask_dir:
            return None
        s1, s2, _ = self.mutated_complexes[index]
        return self._load_tensor(os.path.join(self.interface_mask_dir, f"{s1}__{s2}.pt"))

    @classmethod
    def from_table(
        cls,
        mutated_complexes_file: str,
        wt_sequences_file: str,
        embeddings_dir: str,
        plm_model_name: str = None,
        add_zs_scores: bool = False,
        struct_context_dir: str = None,
        mint_pair: bool = False,
        interface_mask_dir: str = None,
    ):
        """
        Create an instance of the class using data from a table file.
        Args:
            mutated_complexes_file (str): The path to the table file containing information about mutated complexes.
                Each line must contain the following columns:
                - Wild-type sequence A label
                - Wild-type sequence B label
                - Mutations in the format '<wt_aa><chain:A,B><position><mut_aa>'. Multiple mutations must be separated by a comma.
                - (Optional) Score associated with the mutated complex, for evaluation.
                - (Optional) zero-shot scores associated with the mutated complex.
            wt_sequences_file (str): The path to the file containing wild-type sequences.
            embeddings_dir (str): The directory where embeddings are stored.
            plm_model_name (str, optional): The name of the PLM model. Defaults to None.
            add_zs_scores (bool, optional): Whether to include zero-shot scores. Defaults to False. If set to true,
                the fourth column in the table file will be interpreted as zero-shot scores.
        Returns:
            An instance of the class with the specified data.
        """
        wt_sequences = utils.parse_fasta(wt_sequences_file)
        # parse table file
        data = pd.read_table(mutated_complexes_file, sep=r"\s+", header=None)
        mutated_complexes = [
            MutatedComplex(row[0], row[1], tuple(row[2].split(",")))
            for row in data.itertuples(index=False)
        ]
        # Column contract (0=complex, 1=wt, 2=mutations):
        #   4 cols + add_zs_scores=False -> col3 = training label (`scores`).
        #   4 cols + add_zs_scores=True  -> col3 = zero-shot score; NO label -> inference-
        #       only (training would hit `labels=None` in compute_loss). Don't pass
        #       add_zs_scores to a 4-col table you intend to train on.
        #   >4 cols                      -> col3 = label, col4.. = zs score(s); the
        #       add_zs_scores flag is ignored here (the layout is unambiguous). This is the
        #       FoldX add_scores path (single scalar = Stage 1, term vector = Stage 2).
        scores, zs_scores = None, None
        if len(data.columns) > 3:
            if add_zs_scores:
                zs_scores = data[3].astype(float).tolist()
            else:
                scores = data[3].astype(float).tolist()
        if len(data.columns) > 4:
            scores = data[3].astype(float).tolist()
            zs_cols = list(range(4, len(data.columns)))
            if len(zs_cols) == 1:
                zs_scores = data[4].astype(float).tolist()  # single scalar (Stage 1)
            else:
                # decomposed FoldX term vector, one list per row (Stage 2)
                zs_scores = data[zs_cols].astype(float).values.tolist()
        return cls(
            mutated_complexes, wt_sequences, embeddings_dir, plm_model_name, scores, zs_scores,
            struct_context_dir, mint_pair, interface_mask_dir,
        )

    def _fill_metadata(self, mutated_complexes):
        for seq1_label, seq2_label, mutations in mutated_complexes:
            seq1 = self.sequences[seq1_label]
            seq2 = self.sequences[seq2_label]
            mut_seq1, mut_seq2 = utils.parse_mutations(mutations, seq1, seq2)
            mut_seq1_label = (
                f"{seq1_label}_{'-'.join([mut for mut in mutations if mut[1] == 'A'])}"
            )
            mut_seq2_label = (
                f"{seq2_label}_{'-'.join([mut for mut in mutations if mut[1] == 'B'])}"
            )
            self.sequences.update({mut_seq1_label: mut_seq1, mut_seq2_label: mut_seq2})
            self._sequences_ids.append((seq1_label, seq2_label, mut_seq1_label, mut_seq2_label))
        return

    def _generate_missing_embeddings(self, plm_model_name, missing_ids):
        """Fill gaps in the embedding cache on the fly.

        Routed through `mulan.plm.load_backend` rather than the transformers-only pair, so a
        backbone loaded by the ESM SDK or by AIDO's custom code works here too. Previously any
        such tag raised "Invalid model_name" and the only way to use it was to pre-populate the
        cache with a standalone script.
        """
        from mulan.plm import load_backend

        backend = load_backend(plm_model_name)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        for id_ in tqdm(missing_ids, desc="Generating embeddings"):
            embedding = backend.embed_checked(self.sequences[id_])
            utils.save_embedding(embedding, self.embeddings_dir, id_)
        return

    def _mint_bundle_paths(self, index):
        """A2 (MINT): the wt/mut partner-context bundle files for row `index`. Key logic MUST match
        experiments/mint/gen_mint_emb.py (wt_key/mut_key)."""
        s1, s2, mutations = self.mutated_complexes[index]
        wt = os.path.join(self.embeddings_dir, "wt", f"{s1}__{s2}.pt")
        mut = os.path.join(self.embeddings_dir, "mut", f"{s1}__{s2}__{'-'.join(mutations)}.pt")
        return wt, mut

    def _check_mint_cache_complete(self):
        missing = []
        for i in range(len(self.mutated_complexes)):
            for p in self._mint_bundle_paths(i):
                if not os.path.exists(p):
                    missing.append(p)
        if missing:
            raise FileNotFoundError(
                f"mint_pair cache incomplete: {len(missing)} bundle(s) missing under "
                f"{self.embeddings_dir} (e.g. {missing[0]}). Generate with "
                f"experiments/mint/gen_mint_emb.py — MINT embeddings are out-of-band and are not "
                f"created on the fly."
            )

    def _load_embeddings(self, index):
        if self.mint_pair:
            # Reassemble [wt1, wt2, mut1, mut2] from the two partner-context bundles. wt1/wt2 are
            # embedded as the WT complex, mut1/mut2 as the mutant complex — so the partner chain's
            # embedding is mutation-dependent (the non-canceling term the monomer cache discards).
            wt_path, mut_path = self._mint_bundle_paths(index)
            wt, mut = self._load_tensor(wt_path), self._load_tensor(mut_path)
            return MutatedComplexEmbeds(wt["1"], wt["2"], mut["1"], mut["2"])
        return MutatedComplexEmbeds(
            *[
                self._load_tensor(os.path.join(self.embeddings_dir, f"{id_}.pt"))
                for id_ in self._sequences_ids[index]
            ]
        )


class MulanDataCollator(object):
    """
    Data collator for MuLAN dataset.
    Args:
        padding_value (float, optional): The padding value to use. Defaults to 0.
    Returns:
        collated_batch: The collated batch of data.
    """

    def __init__(self, padding_value: float = 0.0):
        self.padding_value = padding_value

    def __call__(self, batch):
        return self._collate_fn(batch)

    def _collate_fn(self, batch):
        elem = batch[0]
        if isinstance(elem, dict):
            return {key: self._collate_fn([d[key] for d in batch]) for key in elem}
        if isinstance(elem, MutatedComplexEmbeds):
            return MutatedComplexEmbeds(
                *[
                    pad_sequence(embeds, batch_first=True, padding_value=self.padding_value)
                    for embeds in (zip(*batch))
                ]
            )
        elif isinstance(elem, torch.Tensor) and elem.dim() == 2:
            # A1: variable [L1,L2] interface masks -> pad both dims to the batch max. Pad
            # value 0/False = "no contact", aligned with how inputs_embeds pad each chain to
            # the same batch-max length (so mask dim0/dim1 match h1/h2 in the model).
            max_a = max(t.shape[0] for t in batch)
            max_b = max(t.shape[1] for t in batch)
            out = elem.new_zeros((len(batch), max_a, max_b))
            for i, t in enumerate(batch):
                out[i, : t.shape[0], : t.shape[1]] = t
            return out
        elif elem is None:
            return None
        elif isinstance(elem, MutatedComplex):
            # Metadata only: `compute_loss` pops "data" before the model sees it, so this field
            # just needs to survive collation, not be stacked. Its mutations tuple is length-1 for
            # single-point (default_collate would stack fine) but variable-length for multi-point
            # (default_collate raises "each element ... equal size"). Pass through as a plain list;
            # single-point is unaffected since the field is discarded downstream either way.
            return list(batch)
        else:
            return default_collate(batch)


def split_data(
    mutated_complexes_file: str,
    output_dir: Optional[str] = None,
    add_validation_set: bool = True,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    num_folds: int = 1,
    random_state: int = 42,
    split_method: str = "balanced",
):
    """Split data into train, validation and test sets for training or cross-validation.
    Args:
        mutated_complexes_file (str): The path to the file containing the mutated complexes data.
        The format for the files is the same as the one used in the `MulanDataset.from_table` method.
        output_dir (str, optional): The directory where the split data will be saved. Defaults to None.
        add_validation_set (bool, optional): Whether to include a validation set. Defaults to True.
        validation_size (float, optional): The proportion of data to be allocated to the validation set. Defaults to 0.15.
        test_size (float, optional): The proportion of data to be allocated to the test set. Defaults to 0.15.
        num_folds (int, optional): The number of folds for cross-validation. Defaults to 1.
        random_state (int, optional): The random seed for reproducibility. Defaults to 42.
        split_method (str, optional): Fold-assignment method for cross-validation (num_folds > 2).
            "balanced" (default): equal-size folds via block-assign + shuffle. This is the method
            used for all local runs prior to 2026-07-09; keep it as the default so those splits
            remain reproducible. "random": independent per-sample assignment with
            `rng.integers` — the upstream/paper method (unbalanced folds), used to reproduce the
            published MuLAN cross-validation partition. Ignored for num_folds <= 2.
    Returns:
        Tuple[List[pd.DataFrame], List[pd.DataFrame], Optional[List[pd.DataFrame]]]:
        A tuple containing the train, test, and validation data sets.
        - train_data_all (List[pd.DataFrame]): A list of train data sets for each fold.
        - test_data_all (List[pd.DataFrame]): A list of test data sets for each fold.
        - val_data_all (Optional[List[pd.DataFrame]]): A list of validation data sets for each fold,
          only present if add_validation_set is True.
    Raises:
        ValueError: If num_folds is less than or equal to 0 or if num_folds is 2 and add_validation_set is True.
    """

    def _save_data(data, output_file):
        data.to_csv(output_file, sep="\t", index=False, header=False)

    train_data_all, test_data_all = [], []
    val_data_all = [] if add_validation_set else None
    files_basename = os.path.splitext(os.path.basename(mutated_complexes_file))[0]
    data = pd.read_table(mutated_complexes_file, sep=r"\s+", header=None)
    rng = np.random.default_rng(random_state)
    if num_folds <= 0:
        raise ValueError("`num_folds` must be greater than 0.")
    elif num_folds == 2 and add_validation_set:
        raise ValueError("`num_folds` must be greater than 2 to add a validation set.")
    elif num_folds == 1:
        split_index = rng.choice(
            [0, 1, 2],
            size=len(data),
            p=[test_size, validation_size, 1 - test_size - validation_size],
        )
        test_data = data[split_index == 0]
        if add_validation_set:
            val_data = data[split_index == 1]
            train_data = data[split_index == 2]
            val_data_all.append(val_data)
        else:
            # No validation set: train = everything not held out for test (split 0),
            # i.e. the val-portion (1) and train-portion (2) merged. The previous
            # `(split_index == 1) & (data[split_index == 2])` mixed a bool ndarray with
            # a DataFrame row-subset and raised / misaligned.
            train_data = data[split_index != 0]
        train_data_all.append(train_data)
        test_data_all.append(test_data)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            _save_data(train_data, os.path.join(output_dir, f"{files_basename}_train.tsv"))
            _save_data(test_data, os.path.join(output_dir, f"{files_basename}_test.tsv"))
            if add_validation_set:
                _save_data(val_data, os.path.join(output_dir, f"{files_basename}_val.tsv"))
    else:
        if split_method == "random":
            # Upstream/paper method (first-commit data.py): independent per-sample fold
            # assignment. Produces unbalanced folds; reproduces the published MuLAN CV partition
            # at random_state=42.
            fold_index = rng.integers(low=0, high=num_folds, size=len(data))
        elif split_method == "balanced":
            # This fork's default: equal-size folds via block-assign + shuffle. Kept as the
            # default so pre-2026-07-09 splits remain reproducible.
            fold_index = np.array(
                [i for i in range(num_folds) for _ in range(len(data) // num_folds)]
            )
            if len(fold_index) < len(data):
                fold_index = np.concatenate(
                    [fold_index, rng.choice(range(num_folds), size=len(data) - len(fold_index))]
                )
            rng.shuffle(fold_index)
        else:
            raise ValueError(
                f"Unknown split_method {split_method!r}; use 'balanced' or 'random'."
            )
        for test_fold_index in range(num_folds):
            test_data = data[fold_index == test_fold_index]
            if add_validation_set:
                val_fold_index = (test_fold_index - 1) % num_folds
                val_data = data[fold_index == val_fold_index]
                val_data_all.append(val_data)
                train_data = data[(fold_index != test_fold_index) & (fold_index != val_fold_index)]
            else:
                train_data = data[fold_index != test_fold_index]
            train_data_all.append(train_data)
            test_data_all.append(test_data)
            if output_dir:
                os.makedirs(os.path.join(output_dir, f"fold_{test_fold_index}"), exist_ok=True)
                _save_data(
                    train_data,
                    os.path.join(
                        output_dir, f"fold_{test_fold_index}", f"{files_basename}_train.tsv"
                    ),
                )
                _save_data(
                    test_data,
                    os.path.join(
                        output_dir, f"fold_{test_fold_index}", f"{files_basename}_test.tsv"
                    ),
                )
                if add_validation_set:
                    _save_data(
                        val_data,
                        os.path.join(
                            output_dir, f"fold_{test_fold_index}", f"{files_basename}_val.tsv"
                        ),
                    )
    return train_data_all, test_data_all, val_data_all
