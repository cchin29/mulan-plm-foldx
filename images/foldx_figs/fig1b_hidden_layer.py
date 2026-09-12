"""Figure 1b — reading a different hidden layer of the embedder."""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(_OUT))
from mulanfig import *
from fig1_common import *

H = 1046
o = [header("1b")]
o.append(arch("1b", enc_mark=True))
o.append(text(ENC_X + ENC_W/2, ROW(0) - 30, "expanded below", 9.5, WARN, weight=650))
o.append(leader(ENC_X + ENC_W/2, 468, 300, 500))

DY = 500
o.append(detail(DY, 504, "Inside the frozen encoder — the layer readout",
                "MuLAN takes last_hidden_state. A linear RidgeCV probe on the site delta (mut − wt at the mutated residue) scores every layer instead. N = 1100."))

def stack(x, y, n, w, readout, best, band, label, anchors):
    """A layer stack shaded by probe PCC. `anchors` = {layer: pcc} measured;
    layers inside `band` are drawn at the measured plateau value."""
    b = []
    lo, hi = band
    for i in range(n + 1):
        pcc = anchors.get(i, anchors["_plateau"] if lo <= i <= hi else anchors["_other"])
        b.append(f'<rect x="{x+i*w}" y="{y}" width="{w-1.2}" height="46" fill="{ramp(pcc)}" '
                 f'stroke="#ffffff" stroke-width="0.6"/>')
    # readout marker (what MuLAN uses)
    rx = x + readout*w + w/2
    b.append(arrow(rx, y - 26, rx, y - 4, INK, 1.8))
    b.append(text(rx, y - 50, "default readout", 10, INK, weight=650))
    b.append(text(rx, y - 37, f"layer {readout}", 9.5, MUTED))
    # probe's best
    bx_ = x + best*w + w/2
    b.append(f'<line x1="{bx_}" y1="{y+46}" x2="{bx_}" y2="{y+62}" stroke="{GOOD}" stroke-width="1.8"/>')
    b.append(text(bx_ - (0 if n < 30 else 74), y + 74, f"probe best: layer {best}", 10, GOOD, weight=650))
    b.append(text(x, y + 108, label, 12, INK, anchor="start", weight=650))
    return "".join(b)

# ---- ProstT5 -------------------------------------------------------------
PX_, PY_ = PANEL_X + 60, DY + 132
P_ANCH = {0: 0.377, 1: 0.62, 2: 0.745, 3: 0.765, 22: 0.730, 23: 0.722, 24: 0.702,
          "_plateau": 0.752, "_other": 0.70}
o.append(stack(PX_, PY_, 24, 24, 24, 3, (2, 21), "ProstT5  ·  24 layers, 1024-d", P_ANCH))
o.append(text(PX_ + 268, PY_ + 108, "— probe best 0.765 at layer 3, vs 0.702 at the default layer 24",
              11, INK_SOFT, anchor="start"))
o.append(text(PX_, PY_ + 128,
              "The final layer is ProstT5's weakest contextualised layer for the local signal — its", 11, INK_SOFT, anchor="start"))
o.append(text(PX_, PY_ + 144,
              "pretraining objective translates AA ↔ 3Di, which discards local AA identity.", 11, INK_SOFT, anchor="start"))

# ---- Ankh ----------------------------------------------------------------
AX_, AY_ = PANEL_X + 60, DY + 320
A_ANCH = {0: 0.377, 1: 0.60, 2: 0.70, 3: 0.74, 47: 0.799, 48: 0.796,
          "_plateau": 0.780, "_other": 0.74}
o.append(stack(AX_, AY_, 48, 12, 48, 47, (8, 46), "Ankh-large  ·  48 layers, 1536-d", A_ANCH))
o.append(text(AX_ + 296, AY_ + 108, "— probe best 0.799 at layer 47, vs 0.796 at the default layer 48: statistically tied",
              11, INK_SOFT, anchor="start"))
o.append(text(AX_, AY_ + 128,
              "Ankh's final layer is already near-optimal — span-denoising keeps the top layer general-", 11, INK_SOFT, anchor="start"))
o.append(text(AX_, AY_ + 144,
              "purpose. So no Ankh mid-layer run was queued: the probe says there is nothing to win.", 11, INK_SOFT, anchor="start"))

# legend
LGX = PANEL_X + 60
o.append(text(LGX, DY + 66, "shaded by measured probe PCC (site delta)", 10, MUTED, anchor="start"))
for i, v in enumerate([0.65, 0.68, 0.71, 0.74, 0.77, 0.80]):
    o.append(f'<rect x="{LGX+246+i*26}" y="{DY+58}" width="26" height="12" fill="{ramp(v)}"/>')
o.append(text(LGX + 246, DY + 84, "≤0.65", 9, MUTED))
o.append(text(LGX + 246 + 6*26, DY + 84, "0.80", 9, MUTED))
o.append(text(LGX + 520, DY + 66, "anchors measured at layers 0–3 and the top three; the interior is drawn at the measured plateau",
              10, MUTED, anchor="start", style="font-style:italic"))

# ---- outcome -------------------------------------------------------------
bx = PANEL_X + 1060
o.append(f'<line x1="{bx-40}" y1="{DY+56}" x2="{bx-40}" y2="{DY+472}" stroke="{HAIRLINE}" stroke-width="1"/>')
o.append(text(bx, DY + 70, "Layer swaps in full MuLAN", 12.5, INK, anchor="start", weight=650))
o.append(text(bx, DY + 90, "ProstT5 only — the probe says Ankh has nothing to win", 10.5, MUTED, anchor="start"))
RUNS = [("run2", "ProstT5, final layer 24 (default)", 0.740, True),
        ("run3a", "ProstT5, layer 7 · 1024-d", 0.648, False),
        ("run3b", "ProstT5, concat 7 ⊕ 10 · 2048-d", 0.429, False)]
BARW = 240
for i, (rid, desc, val, base) in enumerate(RUNS):
    ry = DY + 126 + i*52
    o.append(text(bx, ry, rid, 11.5, INK, anchor="start", weight=650))
    o.append(text(bx + 66, ry, desc, 11, INK_SOFT, anchor="start"))
    w = (val - 0.35) / 0.45 * BARW
    o.append(f'<rect x="{bx}" y="{ry+12}" width="{w:.0f}" height="15" rx="3" '
             f'fill="{"#9fb6d4" if base else BAD}" opacity="{0.9 if base else 0.55}"/>')
    o.append(text(bx + w + 10, ry + 20, f"{val:.3f}", 11.5, INK, anchor="start", weight=650 if base else 400))
o.append(text(bx, DY + 300, "A linear probe on a frozen embedding and a", 11, INK_SOFT, anchor="start"))
o.append(text(bx, DY + 316, "trained attention head do not agree about", 11, INK_SOFT, anchor="start"))
o.append(text(bx, DY + 332, "which layer is useful. The probe's ranking is", 11, INK_SOFT, anchor="start"))
o.append(text(bx, DY + 348, "real, and it does not survive contact with", 11, INK_SOFT, anchor="start"))
o.append(text(bx, DY + 364, "the head.", 11, INK_SOFT, anchor="start"))
o.append(text(bx, DY + 400, "Probe validity: its ordering (Ankh 0.80 >", 10.5, MUTED, anchor="start"))
o.append(text(bx, DY + 416, "ProstT5 0.765) matches full MuLAN", 10.5, MUTED, anchor="start"))
o.append(text(bx, DY + 432, "(Ankh run1 0.757 > ProstT5 run2 0.740).", 10.5, MUTED, anchor="start"))

write(str(_OUT / "fig1b_hidden_layer.svg"), W, H, "".join(o))
print("ok")
