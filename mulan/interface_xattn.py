"""A1 — residue-level interface cross-attention (see docs/history/PLAN_A1_INTERFACE_XATTN.md).

`InterfaceCrossAttention` primes each chain's per-residue reps with its cross-chain partner
*before* the LightAtt pooling encoder, so the pooled `mut - wt` difference can carry the
interface electrostatic signal the current pooled product discards (`modules.py` combines the
two chains only after pooling, with no residue pairing).

A1a (implemented here): a bidirectional cross-attention block added *residually* through a
raw init-0 scalar gate — so at init the block is an exact no-op (output == input) and training
can only turn it on if it helps. This follows the codebase's C3 `iface_alpha` precedent
(`modules.py:84`, an init-0 *additive* scalar), NOT a `sigmoid(gate)` (which would start at
0.5 and perturb the baseline — the earlier plan sketch was wrong on that point).

Standalone module: nothing imports it until the §4 wiring lands, so adding this file is inert
for any running/paused `mulan-train` job. Run the smoke test with:
    ./.venv/bin/python mulan/interface_xattn.py
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InterfaceCrossAttention(nn.Module):
    """Bidirectional, interface-masked cross-attention between two chains' residue reps.

    forward(h1, h2, mask, pad1, pad2):
        h1   [B, L1, D]  chain-1 per-residue embeddings (float)
        h2   [B, L2, D]  chain-2 per-residue embeddings
        mask [B, L1, L2] bool — True where residue i of chain1 contacts residue j of chain2
                         (the interface mask; same M for the wt and mut pass of a complex)
        pad1 [B, L1]     bool — True for real residues, False for padding
        pad2 [B, L2]     bool
    Returns (h1', h2') with the same shapes: h1' = h1 + gate * XAttn(h1<-h2), and symmetric.

    At init `gate == 0`, so (h1', h2') == (h1, h2) exactly (the AA baseline).
    """

    def __init__(self, dim: int, n_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"embed dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        # init-0 additive gate => exact no-op at start (cf. C3 iface_alpha). One scalar shared
        # by both directions (the interface interaction is symmetric).
        self.gate = nn.Parameter(torch.zeros(1))

    def _attend(self, q, kv, allow):
        """q attends to kv where `allow` [B, Lq, Lkv] bool is True; masked-out rows contribute 0.

        Rows of q with no allowed key are un-masked (to avoid an all -inf softmax -> NaN) and
        then zeroed by `valid`, so they receive no update.
        """
        B, Lq, _ = q.shape
        valid = allow.any(dim=-1)                       # [B, Lq] — query has >=1 allowed key
        # MultiheadAttention bool attn_mask: True == DISALLOWED. Allow-all on invalid rows.
        disallow = ~allow & valid.unsqueeze(-1)         # invalid rows -> all False (allow all)
        am = disallow.unsqueeze(1).expand(B, self.n_heads, Lq, allow.shape[-1])
        am = am.reshape(B * self.n_heads, Lq, allow.shape[-1])
        out, _ = self.attn(q, kv, kv, attn_mask=am, need_weights=False)
        return out * valid.unsqueeze(-1)                # zero the no-key rows

    def forward(self, h1, h2, mask, pad1, pad2):
        allow12 = mask & pad2.unsqueeze(1)              # [B, L1, L2] chain1 queries -> chain2 keys
        allow21 = mask.transpose(1, 2) & pad1.unsqueeze(1)  # [B, L2, L1]
        a1 = self._attend(h1, h2, allow12)
        a2 = self._attend(h2, h1, allow21)
        # residual through the gate; zero updates on padded query rows so the downstream
        # channel-0 pad mask (x[...,0] != padding_value) is preserved exactly.
        h1 = h1 + self.gate * a1 * pad1.unsqueeze(-1)
        h2 = h2 + self.gate * a2 * pad2.unsqueeze(-1)
        return h1, h2


def _smoke():
    torch.manual_seed(0)
    B, L1, L2, D = 3, 7, 5, 8
    m = InterfaceCrossAttention(D, n_heads=2)
    h1 = torch.randn(B, L1, D)
    h2 = torch.randn(B, L2, D)
    mask = torch.rand(B, L1, L2) > 0.5
    pad1 = torch.ones(B, L1, dtype=torch.bool)
    pad2 = torch.ones(B, L2, dtype=torch.bool)
    pad1[0, 5:] = False           # item 0 has 5 real chain-1 residues
    pad2[1, 3:] = False           # item 1 has 3 real chain-2 residues
    mask[2] = False               # item 2 has an empty interface (all rows no-key)

    # (a) init gate == 0 => exact no-op
    o1, o2 = m(h1, h2, mask, pad1, pad2)
    assert torch.allclose(o1, h1) and torch.allclose(o2, h2), "gate=0 must be identity"

    # (b) with gate != 0: shapes, no NaN, padded query rows unchanged, empty-interface unchanged
    with torch.no_grad():
        m.gate.fill_(0.7)
    o1, o2 = m(h1, h2, mask, pad1, pad2)
    assert o1.shape == h1.shape and o2.shape == h2.shape
    assert not torch.isnan(o1).any() and not torch.isnan(o2).any(), "no NaN from empty rows"
    assert torch.allclose(o1[0, 5:], h1[0, 5:]), "padded chain-1 query rows must be untouched"
    assert torch.allclose(o2[1, 3:], h2[1, 3:]), "padded chain-2 query rows must be untouched"
    assert torch.allclose(o1[2], h1[2]) and torch.allclose(o2[2], h2[2]), "empty interface => no update"
    # (c) some real interface row actually changed
    assert not torch.allclose(o1[0, :5], h1[0, :5]), "interface rows should update when gate!=0"
    print("interface_xattn smoke OK: identity@init, no-NaN, pad/empty-safe, updates when gated")


if __name__ == "__main__":
    _smoke()
