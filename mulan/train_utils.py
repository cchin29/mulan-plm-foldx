from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from transformers import Trainer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr, spearmanr

from mulan.data import MutatedComplexEmbeds, MutatedComplex


def _metric_spearmanr(y_true, y_pred):
    return spearmanr(y_true, y_pred, nan_policy="omit")[0]


def _metric_pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


_DEFAULT_METRICS = {
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
    "pcc": _metric_pearsonr,
    "scc": _metric_spearmanr,
}


def default_compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    res = {}
    for name, metric in _DEFAULT_METRICS.items():
        res[name] = metric(labels, predictions)
    return res


@dataclass
class DatasetArguments:
    train_data: str = field(
        metadata={
            "help": (
                "Training data in TSV format. Must contain columns for sequence_A, sequence_B,"
                " mutations (separated with comma if multiple), score and (optionally) zero-shot"
                " score."
            )
        }
    )
    train_fasta_file: str = field(
        metadata={
            "help": (
                "Fasta file containing wild-type sequences for training data. Identifiers must"
                " match the training data and, if present, the evaluation data."
            )
        }
    )
    embeddings_dir: str = field(
        metadata={
            "help": (
                "Directory containing pre-computed embeddings in PT format, or where new"
                " embeddings will be stored. In the latter case, `plm_model_name` must be"
                " provided."
            )
        }
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation data file, with the same format of training data."},
    )
    test_data: Optional[str] = field(
        default=None,
        metadata={"help": "Test data file, with the same format of training data."},
    )
    test_fasta_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Fasta file containing wild-type sequences. Identifiers must match the test data."
            )
        },
    )
    plm_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name of the pre-trained protein language model to use for embedding generation."
            )
        },
    )
    struct_context_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory of per-complex pooled structure-context vectors (keyed s1__s2.pt),"
                " fed to the regression head for B1 (struct_context models)."
            )
        },
    )
    add_zs_scores: bool = field(
        default=False,
        metadata={
            "help": (
                "Interpret the last data column as a zero-shot score (e.g. FoldX binding"
                " ddG) and feed it to the add_scores head channel. The model config must"
                " have add_scores=true."
            )
        },
    )
    mint_pair: bool = field(
        default=False,
        metadata={
            "help": (
                "A2 (MINT): read partner-context embeddings from `embeddings_dir` as"
                " wt/<s1>__<s2>.pt + mut/<s1>__<s2>__<muts>.pt bundles (keyed per complex/"
                " mutation) instead of per-label {id}.pt files. Cache must be complete"
                " (out-of-band; see experiments/mint/gen_mint_emb.py)."
            )
        },
    )
    interface_mask_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "A1: directory of per-complex interface contact masks (bool [L1,L2], keyed"
                " s1__s2.pt) fed to the head's interface cross-attention. The model config"
                " must have interface_xattn=true (and xattn_dim set). Build with"
                " experiments/interface_xattn/build_masks.py."
            )
        },
    )


@dataclass
class ModelArguments:
    model_name_or_config_path: str = field(
        metadata={
            "help": (
                "Name of the pre-trained model to fine-tune, or path to config file in JSON"
                " format."
            )
        }
    )
    save_model: bool = field(
        default=False,
        metadata={"help": "Whether to save the model after training."},
    )


def resolve_checkpointing(save_model: bool, early_stopping_patience, has_eval_set: bool):
    """Decide ``(save_strategy, load_best_model_at_end, warning)``.

    Early stopping and checkpointing are coupled, and the coupling is not obvious from the flags.
    ``load_best_model_at_end`` requires a save strategy — HuggingFace cannot restore a checkpoint
    it never wrote. So ``--early_stopping_patience`` with ``--save_model False`` used to stop
    training at the right moment and then **keep the wrong weights**: the run halted N epochs past
    the optimum and every reported test metric came from that degraded endpoint, silently.

    Setting a patience is an unambiguous statement that the best epoch is wanted, so it now
    implies checkpointing. Disk cost is bounded by ``save_total_limit``.
    """
    warning = None
    if early_stopping_patience and not save_model:
        if has_eval_set:
            save_model = True
            warning = (
                "--early_stopping_patience was set without --save_model, so the best epoch could "
                "not have been restored: training would have stopped at the right moment and then "
                "kept the final (over-trained) weights, and every test metric would have come "
                "from those. Enabling checkpointing so the best epoch is restored."
            )
        else:
            warning = (
                "--early_stopping_patience has no effect without an evaluation set: there is no "
                "metric to monitor, so training will run the full --num_epochs."
            )
    return (
        "epoch" if save_model else "no",
        bool(has_eval_set and save_model),
        warning,
    )


@dataclass
class CustomisableTrainingArguments:
    output_dir: str = field(metadata={"help": "Directory where the trained model will be saved."})
    num_epochs: int = field(default=30, metadata={"help": "Number of training epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size."})
    learning_rate: float = field(default=5e-4, metadata={"help": "Learning rate."})
    disable_tqdm: bool = field(
        default=False, metadata={"help": "Whether to disable tqdm progress bars."}
    )
    report_to: Union[None, str, List[str]] = field(
        default="none",
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of epochs without improvement before early stopping. If not set, early"
                " stopping is disabled."
            )
        },
    )
    tail_loss_mode: str = field(
        default="none",
        metadata={
            "help": (
                "C1 (tail-weighted loss): reweight the per-row MSE to stop the head shrinking the"
                " high-|ΔΔG| tail toward the mean. 'none' (default) = plain MSE, a strict no-op that"
                " leaves every existing arm unchanged. 'linear' = weight 1+alpha*|label| (weight"
                " grows with |ΔΔG|). 'focal' = weight (|err|/rms)^gamma (emphasize the currently"
                " hard/large-error rows). Both variants renormalize weights to mean 1 so the loss"
                " scale — and thus the effective learning rate — stays comparable to plain MSE."
            )
        },
    )
    tail_loss_alpha: float = field(
        default=0.0,
        metadata={"help": "C1 'linear' mode slope: MSE weight = 1 + tail_loss_alpha*|label|."},
    )
    tail_loss_gamma: float = field(
        default=1.0,
        metadata={"help": "C1 'focal' mode exponent: MSE weight = (|err|/rms_err)^tail_loss_gamma."},
    )


class MulanTrainer(Trainer):
    """Custom Trainer class adapted for Mulan model training"""

    def __init__(self, *args, tail_loss_mode="none", tail_loss_alpha=0.0, tail_loss_gamma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # C1 tail-weighted loss (see CustomisableTrainingArguments); 'none' => plain MSE.
        self.tail_loss_mode = tail_loss_mode
        self.tail_loss_alpha = tail_loss_alpha
        self.tail_loss_gamma = tail_loss_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None,):
        """
        Computes the loss for MulanDataset inputs for a model that do not return loss values.
        """
        inputs.pop("data")
        outputs = model(
            inputs["inputs_embeds"], inputs.get("zs_scores"),
            struct_ctx=inputs.get("struct_ctx"), iface_mask=inputs.get("iface_mask"),
        )
        labels = inputs.get("labels")
        pred = outputs.view(-1)
        target = labels.view(-1)
        se = (pred - target) ** 2
        if self.tail_loss_mode == "none" or (
            self.tail_loss_mode == "linear" and self.tail_loss_alpha == 0.0
        ):
            loss = se.mean()  # identical to the original mse_loss
        elif self.tail_loss_mode == "linear":
            # weight grows with |ΔΔG|; renormalize to mean 1 so the loss scale is unchanged.
            w = 1.0 + self.tail_loss_alpha * target.detach().abs()
            loss = (w * se).sum() / w.sum()
        elif self.tail_loss_mode == "focal":
            # emphasize the rows the model is currently getting most wrong (regression focal loss).
            with torch.no_grad():
                err = se.detach().clamp_min(1e-8).sqrt()
                w = (err / err.mean().clamp_min(1e-8)) ** self.tail_loss_gamma
                w = w / w.mean().clamp_min(1e-8)  # mean-1 normalize; keeps loss scale comparable
            loss = (w * se).mean()
        else:
            raise ValueError(f"unknown tail_loss_mode {self.tail_loss_mode!r}")
        return (loss, outputs) if return_outputs else loss

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        Adapted from the parent class to handle the case where the input is a custom type.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (MutatedComplexEmbeds, MutatedComplex)):
            return type(data)(*[self._prepare_input(v) for v in data])
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (
                torch.is_floating_point(data) or torch.is_complex(data)
            ):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(
                    {"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()}
                )
            return data.to(**kwargs)
        return data

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Overridden from the parent class to handle the case where the model does not return loss values.
        Support for sagemaker was removed!

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        # print("return_loss", return_loss, "has_labels", has_labels)
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = inputs.get("labels")
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                logits = outputs
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(
                        inputs["inputs_embeds"], inputs.get("zs_scores"),
                        struct_ctx=inputs.get("struct_ctx"), iface_mask=inputs.get("iface_mask"),
                    )
                logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)
