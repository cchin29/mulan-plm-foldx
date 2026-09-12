"""Definition of Config class"""


from typing import Any, Dict, Optional, Tuple, Union, get_type_hints
from dataclasses import dataclass, asdict
import json


@dataclass
class MulanConfig:
    """
    Configuration class for Mulan.
    Attributes:
        hidden_size (int): The size of the hidden layer (corrsponding to convolution filters).
        last_hidden_size (int): The size of the last hidden layer.
        hidden_dropout_prob (float): The dropout probability for the hidden layer.
        padding_value (float): The padding value.
        kernel_sizes (Union[Tuple, int]): The sizes of the kernels.
        conv_dropout (float): The dropout probability for the convolutional layer.
        add_scores (bool): Whether to add scores or not.
        layer_mix (bool): Whether the embeddings are an all-layers stack to be combined
            with a learned scalar-mix (ELMo/SeqVec style) before the encoder. When True,
            each cached embedding is expected to have shape [L, num_plm_layers, D].
        num_plm_layers (int): Number of PLM hidden states in the cached stack (e.g. 25 for
            ProstT5: input embeddings + 24 encoder layers). Only used when layer_mix is True.
    Methods:
        from_json(cls, json_path, strict=False): Creates an instance of MulanConfig from a JSON file.
        from_dict(cls, args_dict, strict=False): Creates an instance of MulanConfig from a dictionary.
        save(self, json_path=None) -> str: Saves the configuration to a JSON file.
    """
    
    hidden_size: int = 64
    last_hidden_size: int = 20
    hidden_dropout_prob: float = 0.1
    padding_value: float = 0
    kernel_sizes: Union[ Tuple, int] = (1, 5, 9)
    conv_dropout: float = 0.1
    add_scores: bool = False
    layer_mix: bool = False
    num_plm_layers: int = 0
    # E1: learned scalar gate on the trailing structure (3Di) block of an AA⊕3Di concat
    # embedding, so the model can down-weight structure instead of being diluted by it.
    struct_gate: bool = False
    aa_dim: int = 0  # width of the leading AA block; the gate scales channels [aa_dim:]
    # B1: pooled WT structure context fed to the regression head, outside the siamese
    # mut-wt difference (the four siamese inputs stay AA-only).
    struct_context: bool = False
    struct_context_dim: int = 0  # input dim of the pooled context vector (e.g. 2048)
    struct_proj_dim: int = 0     # projected dim appended to the head (e.g. 64)
    # C3: bias the Light-Attention pooling toward binding-interface residues. The embedding
    # is [AA (aa_dim) | interface indicator (trailing channels)]; the trailing channel is
    # split off and added to the attention logits with a learnable strength (init 0 => no
    # bias, i.e. starts as the AA-only baseline).
    interface_bias: bool = False
    # Stage 2 (FoldX): a small MLP over a decomposed FoldX term vector, replacing the
    # single add_scores scalar. Its scalar output feeds the same +1 head slot, so the
    # regression head width is unchanged. Requires add_scores=true.
    zs_mlp: bool = False
    zs_input_dim: int = 0    # number of decomposed terms fed to the MLP (e.g. 12)
    zs_mlp_hidden: int = 16  # MLP hidden width
    # A1: residue-level interface cross-attention primes each chain's reps with its
    # cross-chain partner (interface-masked) before the pooling encoder, so the pooled
    # mut-wt difference carries the interface signal the pooled product discards. Gated
    # init-0 => starts as the AA baseline. Needs iface_mask at forward (interface_mask_dir).
    interface_xattn: bool = False
    xattn_heads: int = 2     # cross-attention heads (must divide xattn_dim)
    # PLM embedding width D. Unlike the lazy encoder (LazyConv1d infers D), the cross-
    # attention projections need D at construction, so it is carried here explicitly and
    # set per-PLM in the A1 config JSON (e.g. 1536 for Ankh-large). Only used when
    # interface_xattn=True.
    xattn_dim: int = 0

    @classmethod
    def from_json(cls, json_path, strict=False):
        try:
            with open(json_path, "r") as f:
                args_dict = json.load(f)
            return cls.from_dict(args_dict, strict=strict)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {json_path}")
    
    
    @classmethod
    def from_dict(cls, args_dict: Dict[str, Any], strict: bool = False):
        attributes = {**cls.__annotations__, **get_type_hints(cls)}
        keys_to_exclude = list(set(args_dict.keys()) - set(attributes))
        if not strict:
            if keys_to_exclude:
                print(f"Keys {keys_to_exclude} do not match {cls.__name__} attributes and are thus ignored")
            args_dict = {key: value for key, value in args_dict.items() if key in attributes}
        else:
            if keys_to_exclude:
                raise KeyError(f"Unrecognized keys {keys_to_exclude} found in input dictionary")
        keys_defaulted = list(set(attributes) - set(args_dict.keys()))
        if keys_defaulted:
            print(f"Keys {keys_defaulted} were not found in input dictionary and are initialized to default values")
        return cls(**args_dict)
    
    
    def save(self, json_path: Optional[str] = None) -> str:
        if json_path is None:
            json_path = "config.json"
        with open(json_path, "w") as f:
            json.dump(asdict(self), f, indent=4, default=lambda x: x.__dict__)
        return json_path
