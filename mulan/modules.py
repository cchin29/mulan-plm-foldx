"""Implementation of Light attention model"""

import warnings
from typing import Union, Sequence, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn

from mulan.config import MulanConfig
from mulan.interface_xattn import InterfaceCrossAttention
from mulan.utils import TorchDevice, get_device


@dataclass
class OutputWithAttention:
    output: torch.Tensor = None
    attention: Union[torch.Tensor, Sequence[torch.Tensor]] = None


class LayerMix(nn.Module):
    """ELMo/SeqVec-style learned scalar-mix over PLM hidden layers.

    Collapses an all-layers embedding stack into a single per-residue embedding using
    a learned softmax over the layer axis and a learned scalar `gamma`:

        out = gamma * Σ_ℓ softmax(w)_ℓ · h_ℓ

    Input  [..., num_layers, D]  ->  output [..., D]. Weights start uniform (w = 0),
    so at init this is a plain mean over layers. Padded residues (all-zero across the
    layer/feature axes) stay zero after mixing, preserving the encoder's pad mask.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        # x: [..., num_layers, D]; weight the layer axis (dim=-2) and sum it out.
        mixed = (x * w.view(*([1] * (x.dim() - 2)), self.num_layers, 1)).sum(dim=-2)
        return self.gamma * mixed


class StructGate(nn.Module):
    """E1: learned scalar gate on the trailing structure block of an AA⊕3Di embedding.

    Input  [B, L, aa_dim + struct_dim]  ->  [B, L, aa_dim + struct_dim] with the structure
    block scaled by `sigmoid(g)`. The gate is a single learnable scalar, init 0 (=> 0.5 at
    start); after training, `sigmoid(g) -> 0` means the model learned to ignore structure
    (run6c lost to dilution), while a retained gate with PCC > run2 means structure helps
    once it can be down-weighted. The leading AA block (channel 0 used for the pad mask) is
    untouched, and zero-padded residues stay zero (gate is a plain scalar multiply).
    """

    def __init__(self, aa_dim: int):
        super().__init__()
        self.aa_dim = aa_dim
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aa = x[..., : self.aa_dim]
        st = x[..., self.aa_dim :] * torch.sigmoid(self.gate)
        return torch.cat([aa, st], dim=-1)


class AttentionMeanK(nn.Module):
    "Light Attention MuLAN encoder"

    def __init__(self, config: MulanConfig):
        super().__init__()

        self.padding_value = config.padding_value
        kernel_sizes = list(config.kernel_sizes)
        self.n_attn = len(kernel_sizes)
        self.hidden_size = config.hidden_size
        self.last_hidden_size = config.last_hidden_size

        # C3: split the trailing interface channel(s) off the embedding and add them to the
        # attention logits with a learnable strength (init 0 => no bias at start).
        self.interface_bias = getattr(config, "interface_bias", False)
        self.aa_dim = config.aa_dim
        if self.interface_bias:
            self.iface_alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.conv_dropout)

        self.conv_heads = nn.ModuleList(
            [
                nn.LazyConv1d(self.hidden_size, kernel_size, stride=1, padding=kernel_size // 2)
                for kernel_size in kernel_sizes
            ]
        )

        self.attn_heads = nn.ModuleList(
            [
                nn.LazyConv1d(self.hidden_size, kernel_size, stride=1, padding=kernel_size // 2)
                for kernel_size in kernel_sizes
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(
                self.hidden_size * self.n_attn * 2, self.last_hidden_size * self.n_attn
            ),  # n_attn + concatenated maxpool
            nn.Dropout(config.hidden_dropout_prob),
            nn.LeakyReLU(0.5),
            nn.Linear(self.last_hidden_size * self.n_attn, self.last_hidden_size // 2),
            nn.LeakyReLU(0.5),
        )

    def forward(self, x: torch.Tensor):

        # C3: peel off the trailing interface channel, then conv/attend on the AA part only
        iface_bias = 0.0
        if self.interface_bias:
            iface_bias = self.iface_alpha * x[..., self.aa_dim]   # [batch, length]
            iface_bias = iface_bias[:, None, :]                   # broadcast over attn channels
            x = x[..., : self.aa_dim]

        # feature convolution
        batch_size, length, hidden = x.shape
        # Padding is inferred from channel 0 == padding_value (the collator pads with it).
        # Assumes no real residue has an exactly-`padding_value` channel-0 embedding — safe
        # for float PLM embeddings (probability ~0) and inherent to MuLAN's design. Left as-is
        # deliberately: threading an explicit pad mask from the collator would change the mask
        # on existing runs and invalidate trained checkpoints / reported results.
        mask = x[..., 0] != self.padding_value
        x = x.transpose(-1, -2)
        o = torch.stack([conv(x) for conv in self.conv_heads], dim=1)
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]

        # attention weights (interface bias up-weights interface residues before softmax)
        attn_weights = torch.stack(
            [
                self.softmax(
                    (head(x) + iface_bias).masked_fill(mask[:, None, :] == False, -1e9)
                )
                for head in self.attn_heads
            ],
            dim=1,
        )
        o1 = torch.sum(o * attn_weights, dim=-1).view(batch_size, -1)

        # max pooling
        o2, _ = torch.max(o, dim=-1)
        o2 = o2.view(batch_size, -1)

        # mlp
        o = torch.cat([o1, o2], dim=-1)
        o = self.fc(o)
        output = OutputWithAttention(o, attn_weights)

        return output


class LightAttModel(nn.Module):
    "Light Attention MuLAN model"

    def __init__(self, config: MulanConfig):
        super().__init__()

        self.config = config
        if getattr(config, "layer_mix", False):
            self.layer_mix = LayerMix(config.num_plm_layers)
        if getattr(config, "struct_gate", False):
            self.struct_gate = StructGate(config.aa_dim)
        # A1: interface cross-attention runs on the per-residue reps before the encoder.
        # Non-lazy (needs D at construction), so it is built here from config.xattn_dim and
        # its params are captured by the optimizer built over model.parameters().
        if getattr(config, "interface_xattn", False):
            if not getattr(config, "xattn_dim", 0):
                raise ValueError("interface_xattn=True requires xattn_dim (the PLM embed dim) > 0")
            self.iface_xattn = InterfaceCrossAttention(config.xattn_dim, config.xattn_heads)
        self.encoder = AttentionMeanK(config)

        last_hidden_size = config.last_hidden_size
        if config.add_scores:
            last_hidden_size = last_hidden_size + 1
        # Stage 2: a small MLP over a decomposed FoldX term vector (zs_input_dim terms)
        # whose scalar output feeds the same +1 add_scores slot (head width unchanged).
        if getattr(config, "zs_mlp", False):
            self.zs_mlp = nn.Sequential(
                nn.Linear(config.zs_input_dim, config.zs_mlp_hidden),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.zs_mlp_hidden, 1),
            )
        # B1: project the pooled WT structure context and append it to the head input
        if getattr(config, "struct_context", False):
            self.ctx_proj = nn.Sequential(
                nn.Linear(config.struct_context_dim, config.struct_proj_dim),
                nn.LeakyReLU(0.5),
            )
            last_hidden_size = last_hidden_size + config.struct_proj_dim

        self.linear = nn.Linear(last_hidden_size, 1)

    def forward(
        self,
        inputs_embeds: Sequence[torch.FloatTensor],
        zs_scores: Optional[torch.FloatTensor] = None,
        struct_ctx: Optional[torch.FloatTensor] = None,
        iface_mask: Optional[torch.BoolTensor] = None,
        output_attentions=False,
    ):

        batch_size = inputs_embeds[0].shape[0]
        # learned scalar-mix over the cached all-layers stack, if enabled
        if getattr(self.config, "layer_mix", False):
            inputs_embeds = [self.layer_mix(emb) for emb in inputs_embeds]
        # E1: learned gate on the structure block of an AA⊕3Di concat embedding
        if getattr(self.config, "struct_gate", False):
            inputs_embeds = [self.struct_gate(emb) for emb in inputs_embeds]
        # A1: prime chain-1/chain-2 reps with interface cross-attention before pooling.
        # The order is [wt1, wt2, mut1, mut2]; the same interface mask primes both passes.
        # pad masks come from the channel-0 pad convention (== padding_value), matching the
        # encoder. Gated init-0 => this is an exact no-op until trained.
        if getattr(self.config, "interface_xattn", False) and iface_mask is not None:
            pv = self.config.padding_value
            pad1 = inputs_embeds[0][..., 0] != pv
            pad2 = inputs_embeds[1][..., 0] != pv
            wt1, wt2 = self.iface_xattn(inputs_embeds[0], inputs_embeds[1], iface_mask, pad1, pad2)
            mut1, mut2 = self.iface_xattn(inputs_embeds[2], inputs_embeds[3], iface_mask, pad1, pad2)
            inputs_embeds = [wt1, wt2, mut1, mut2]
        # Siamese encoder
        encodings = [
            self.encoder(emb) for emb in inputs_embeds
        ]  # The order is [wt1, wt2, mut1, mut2]

        # features combination
        x_wt = torch.cat(
            [
                encodings[0].output * encodings[1].output,
                torch.abs(encodings[0].output - encodings[1].output),
            ],
            dim=1,
        )
        x_mut = torch.cat(
            [
                encodings[2].output * encodings[3].output,
                torch.abs(encodings[2].output - encodings[3].output),
            ],
            dim=1,
        )
        output = x_mut - x_wt

        # B1: append pooled WT structure context, outside the siamese mut-wt difference
        if getattr(self.config, "struct_context", False) and struct_ctx is not None:
            output = torch.cat((self.ctx_proj(struct_ctx), output), dim=-1)
        if self.config.add_scores and zs_scores is None:
            # The head reserves a +1 (or +zs_input_dim) input slot when add_scores=True;
            # without zs_scores the concat below is skipped and self.linear gets an input
            # that is short by that slot -> a cryptic shape error. Fail with the cause.
            raise ValueError(
                "add_scores=True model requires zs_scores at forward time (the regression "
                "head reserves an input slot for them); got zs_scores=None. Provide the "
                "scores column (--add_zs_scores) or load a non-add_scores checkpoint."
            )
        if zs_scores is not None and self.config.add_scores:
            if getattr(self.config, "zs_mlp", False):
                zs = self.zs_mlp(zs_scores.view(batch_size, self.config.zs_input_dim))
            else:
                zs = zs_scores.view(batch_size, -1)
            output = torch.cat((output, zs), dim=-1)
        output = self.linear(output).squeeze(-1)

        if not output_attentions:
            return output
        else:
            return OutputWithAttention(output, tuple([enc.attention for enc in encodings[:2]]))

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, device: Optional[TorchDevice] = None, **kwargs
    ):
        if device is None:
            device = get_device()
        ckpt = torch.load(pretrained_model_path, map_location=device, weights_only=False)
        config = dict(ckpt["config"])
        config.update(kwargs)
        # from_dict (not MulanConfig(**config)) so a checkpoint whose saved config
        # carries a key that was renamed/removed across versions loads leniently
        # (unknown keys dropped, missing keys defaulted) instead of raising TypeError.
        config = MulanConfig.from_dict(config)
        state_dict = ckpt["state_dict"]
        model = cls(config).to(device)
        # strict=False lets a checkpoint that predates a head (e.g. struct_context /
        # ctx_proj / zs_mlp) still load — but that would leave the new head at random
        # init and return confidently wrong predictions SILENTLY. So we surface any
        # mismatch loudly: missing keys are params the current arch expects but the
        # checkpoint lacks (left at init); unexpected keys are checkpoint params the
        # arch dropped. A caller can then decide whether the drift is intended.
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            warnings.warn(
                f"LightAttModel.from_pretrained: checkpoint/architecture mismatch for "
                f"{pretrained_model_path!r}. Missing (left at RANDOM INIT): "
                f"{list(incompatible.missing_keys)}; unexpected (ignored): "
                f"{list(incompatible.unexpected_keys)}. Predictions may be invalid if "
                f"any missing key is a real weight rather than a buffer.",
                RuntimeWarning, stacklevel=2,
            )
        return model
