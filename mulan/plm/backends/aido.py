"""AIDO.Protein-16B.

Promoted out of the local working area. Unlike the other HuggingFace backbones this one needs
``trust_remote_code`` and bfloat16 weights, and it emits a trailing ``[SEP]`` with no leading
``[CLS]`` — which is why it cannot share the generic ``AutoModel`` path.

At 16B it is generated offline on a CPU or GPU box and read from cache at training time; it does
not run on Apple Silicon.
"""

from __future__ import annotations

import os

import torch

from .base import Backend


class AIDOBackend(Backend):
    def __init__(self, spec, device):
        if device.type == "mps":
            device = torch.device("cpu")  # registered mps_ok=False
        super().__init__(spec, device)
        from transformers import AutoModel, AutoTokenizer

        threads = os.environ.get("AIDO_NUM_THREADS")
        if threads:
            torch.set_num_threads(int(threads))

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            spec.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).eval().to(device)

    def _forward(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model(**enc)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out.hidden_states[-1]
        # No special-token mask: the trailing [SEP] is removed by the registry's trimming rule,
        # and there is no leading [CLS] to remove.
        return hidden.float(), None
