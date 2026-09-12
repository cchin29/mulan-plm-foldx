"""Assemble the four Figure-1 panels into one keyboard-steppable page.

Reads the SVGs the panel generators wrote, so it is always in sync with them.
Run it after fig1a…fig1d.
"""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import pathlib, re

D = pathlib.Path(str(_OUT))
PANELS = [("1a", "late concat",   "fig1a_late_concat.svg"),
          ("1b", "hidden layer",  "fig1b_hidden_layer.svg"),
          ("1c", "input fusion",  "fig1c_input_fusion.svg"),
          ("1d", "score channel", "fig1d_score_channel.svg")]

bar, panes = [], []
for i, (tag, nm, fn) in enumerate(PANELS):
    on = " on" if i == 0 else ""
    bar.append(f'<button class="tb{on}" data-i="{i}"><b>{tag}</b><span>{nm}</span></button>')
    svg = (D / fn).read_text()
    svg = svg[svg.index("<svg"):]                      # drop any XML prolog
    panes.append(f'<section class="pane{on}" id="p{i}">{svg}</section>')

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure 1 · Four routes for structure into MuLAN</title><style>
 body{margin:0;background:#f2f2f0;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;color:#231f20}
 .wrap{max-width:1740px;margin:0 auto;padding:18px 16px 40px}
 .bar{display:flex;gap:8px;margin:0 0 14px;position:sticky;top:0;background:#f2f2f0;padding:12px 0;z-index:5}
 .tb{font:inherit;display:flex;gap:9px;align-items:baseline;padding:9px 16px;border-radius:8px;cursor:pointer;
      border:1.6px solid #b8b5b0;background:#fff;color:#8a8782}
 .tb b{font-size:13px;font-weight:700} .tb span{font-size:12px}
 .tb.on{background:#231f20;border-color:#231f20;color:#fff}
 .tb:hover:not(.on){color:#231f20;border-color:#8a8782}
 .pane{display:none;background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.07);overflow:hidden}
 .pane.on{display:block}
 .hint{font-size:12px;color:#8a8782;margin:12px 2px 0}
 kbd{font:inherit;font-size:11px;background:#fff;border:1px solid #b8b5b0;border-bottom-width:2px;
     border-radius:4px;padding:1px 5px}
</style></head><body><div class="wrap">
<div class="bar">__BAR__</div>__PANES__
<p class="hint">Figure 1, panels 1a&ndash;1d. The architecture at the top is identical across the four panels &mdash;
only the highlighted region changes. Step with <kbd>&larr;</kbd> <kbd>&rarr;</kbd>.
Figure 2 and Figure A1 are separate files.</p>
</div><script>
const bs=[...document.querySelectorAll('.tb')],ps=[...document.querySelectorAll('.pane')];
let cur=0;
function go(i){cur=(i+bs.length)%bs.length;
  bs.forEach((b,k)=>b.classList.toggle('on',k===cur));
  ps.forEach((p,k)=>p.classList.toggle('on',k===cur));}
bs.forEach(b=>b.onclick=()=>go(+b.dataset.i));
addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(cur+1);if(e.key==='ArrowLeft')go(cur-1);});
</script></body></html>"""

out = HTML.replace("__BAR__", "".join(bar)).replace("__PANES__", "".join(panes))
(D / "fig1_panels.html").write_text(out)
print("ok", len(out))
