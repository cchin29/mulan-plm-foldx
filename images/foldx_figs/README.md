# FoldX / structure figures

Hand-authored SVG (editable in Illustrator / Inkscape, scales for a paper figure) with a
2× PNG rendered alongside for slides.

## Numbering

**Figure numbers follow deck order, panel letters follow panel order.** There is exactly one
numbering system on a figure: the panel letter carries the sequence, so no circled ①②③④ and
no separate "tab" label. Appendix figures are `A`-prefixed and never renumber the main set.

| Figure | File | Role |
|---|---|---|
| **1a** | `fig1a_late_concat` | the ProstT5 late concat, and why the 3Di block is constant across the pair |
| **1b** | `fig1b_hidden_layer` | the layer readout, probe-shaded encoder stacks |
| **1c** | `fig1c_input_fusion` | SaProt input fusion, and the FoldX mutant-3Di no-op |
| **1d** | `fig1d_score_channel` | the FoldX arms into `add_scores`, the ceiling, the FoldX-alone coda |
| **1** | `fig1_panels.html` | all four panels in one page, ← / → to step |
| **2** | `fig2_predictor_inputs` | the comparator reframe and the FoldX-alone results — the strongest finding |
| **A1** | `figA1_foldx_channel` | appendix: computation, the twelve terms, the two-arm spec table |
| — | `v1/`, `v2/` | superseded: the single-figure C (v1), and the Ci–Civ / A / B lettering (v2) |

**Figure 1 is the spine.** It tells the whole Part I investigation on its own: four views of
the *same* architecture, one per route. Figure 2 and Figure A1 hang off panel 1d.

**Deck order — 1a → 1b → 1c → 1d → 2, with A1 as appendix.** Story, then evaluation; it ends
on the FoldX-alone result rather than on mechanism. A1 answers review question Q1 in full and
stands alone if sent by itself.

**Two planned figures were absorbed into Figure 1 and no longer need building.** The
"mutant-3Di no-op" figure is panel 1c's right column; the "redundancy decomposition" is
panel 1d's *Channel value, out-of-fold* box.

### Not yet built, in priority order

The gap is no longer mechanism figures. Three of the five review questions still have
no figure at all:

| | Answers | Content |
|---|---|---|
| **3** | **Q2** | the split ladder: leaky per-mutation → by-complex → clustered (CD-HIT ≤60 %) → CATH superfamily, what each holds out, residual leakage. Also settles the two conflicting "clustered" definitions in the older artifacts |
| **4** | **Q3 + Q4** | what CATH(all-mut) contains and how it was constructed, plus train/test mutation and complex counts per configuration (table drafted in §10.1 of the plan) |
| **5** | Part II | the augmentation design: compute-once FoldX, the per-fold inclusion gate, Stage A (channel off) → Stage B, Source A vs Source B |
| **6** | Part II | interface prediction (Task 2) |

## Regenerating

```
python3 fig1a_late_concat.py     # 1a — late concat
python3 fig1b_hidden_layer.py    # 1b — hidden layer
python3 fig1c_input_fusion.py    # 1c — input fusion + the mutant-3Di no-op
python3 fig1d_score_channel.py   # 1d — the score channel
python3 fig2_predictor_inputs.py
python3 figA1_foldx_channel.py
python3 build_panels.py          # rebuilds fig1_panels.html from the four SVGs
node render.mjs fig*.svg         # 2× PNGs, needs playwright + the bundled chromium
```

Each script writes its SVG beside itself, resolved from `__file__`; nothing depends on the
directory it is invoked from or on the machine it was authored on.

`render.mjs` hard-codes `/opt/pw-browsers/chromium`, so it only runs where that browser is
installed. `rsvg-convert -z 2 -b white -o <name>.png <name>.svg` produces the same 2× raster from
the same SVG and was used for the 2026-08-04 Figure 2 redraw; the two rasterizers agree on these
figures, glyph fallbacks included.

`fig1_common.py` holds the shared architecture — the four panels import it, so the top of
each figure is byte-identical apart from the one region that panel changes. Its module-level
coordinates (`SLOT_X`, `VEC_Y`, `ENC_X`, …) are what let 1d route its feed line precisely into
the `add_scores` cell.

All of them import `mulanfig.py`, which holds the shared drawing grammar and palette. Change a
colour or a primitive there and every figure follows.

## The visual grammar

Inherited from `images/visual_abstract.png` so the new figures read as the same system:

| Mark | Meaning |
|---|---|
| grey isometric slab | a tensor (length / thickness carries dimension) |
| pink rounded rect | frozen PLM encoder (❄, blue outline groups the frozen block) |
| cyan rounded rect | Light Attention |
| blue parallelogram | an operation (Linear, 1D Conv) |
| circled glyph | elementwise op (⊖ is the `mut − wt` node) |
| dotted rounded rect | grouping (wild-type pair / mutant pair) |
| boxed letters, red glyph | sequence, mutated residue |

### Layout and labelling rules

- **Every heading is a noun phrase — including panel and column headings.** Not just the
  figure title. Interrogative clauses count as statements: "Where each 3Di string comes
  from" → "Provenance of the two 3Di strings"; "What the channel is worth" → "Channel
  value, out-of-fold"; "Every variant lost to sequence-only" → "The four fusion variants".
  The finding goes in the body text underneath.
- **One numbering system per figure.** The panel letter is the order; nothing else numbers
  the same thing. Cross-references name the target ("panel 1c", "Figure A1"), never "the
  next tab" or "the appendix sheet".
- **Name the encoder on every reported number.** Runs in this set are not all on the same
  PLM: 1a and 1b are ProstT5, 1c is SaProt-650M, 1d is Ankh-large, Figure 2 is Ankh-large
  and ESM-C 6B. Each panel states its encoder in the subtitle and again in the run table.
  Results that are encoder-independent (the 0/39 mutant-3Di probe) say so explicitly.
- **Flow is left→right, then top-down.** Never right-to-left, never bottom-up. Annotations never
  hang off the right edge of a mark, because that competes with the direction of flow — put them
  below, or to the left. (This is why the `add_scores` callout moved and the pooled difference
  vector became horizontal.)
- **The `add_scores` slot is always the same glyph** — a horizontal pooled vector with one amber
  cell *appended at the right end*, so concatenation is literal. Draw it with `hslot()`; never
  hand-roll it.
- **Every quantity carries a unit or a type.** Tensors get a dimension in square brackets
  (`[ L ]`, `[ L × H ]`, `[ H ]`, `[ 1 ]`); energies get `kcal/mol`; correlations are named and
  marked unitless.
- **Terminology is consistent across figures.** A FoldX energy is a *ΔΔG* everywhere — the terms
  read "ΔΔG interaction energy", not "Interaction Energy"; `IE` is defined once where the
  subtraction is shown. Term names are otherwise reproduced exactly as `AnalyseComplex` emits them.

**One family is new.** The paper's figure has no mark for *non-neural physics*, so FoldX gets
**amber** (`PHYS_F` / `PHYS_S`) — deliberately not pink, so it never reads as another learned
module. The `add_scores` slot uses the same amber, which is what makes "the physics enters
here, and only here" visible at a glance.

Status colours (`GOOD` / `BAD` / `WARN`) are reserved for outcome badges and always ship with a
glyph plus a word, never colour alone.

## Numbers on the figures

Every value is traceable:

- Figure 1 — S1102 pooled-Pearson values from `FOLDX_SUMMARY.md`; probe PCCs from
  `experiments/LAYER_PROBE.md`; the 0/39 and 0/232 counts from `FINDINGS_FOLDX_MUTANT_3DI.md`.
- Figure 2 — all ρ values from `experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv`;
  the 0/78, 19/232 and 3/76 tallies from the same sweep.
- Figure A1 — term list and MLP shape from `models/config/lightatt_addscores*_config.json` and
  `scratch/foldx_s1102/build_ddg.py`; timings from `FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md` §9.

Figure 2 was redrawn on 2026-08-04 against the 99.2%-coverage sweep, and the earlier caution here
had it backwards on the point that mattered. **FoldX-alone is independent of training but not of
coverage**: raising the single-point channel from 87.8% to 99.2% moved it 0.363 → 0.418, the
largest single correction in `FOLDX_ALONE_BASELINE_RESULT.md`. Both sides of every contrast on
those tiers moved, which is why the figure's Ankh-large panel keeps its conclusion (lane 3 still
lands on lane 1, Δ = −0.007) while all three of its numbers changed. The CATH panel's FoldX and
base values did not move — that tier was already at 100% coverage — but its 12-term arm did,
0.525 → 0.495.

Anything downstream of a coverage change therefore needs re-running, not just re-reading:
`foldx_alone_baseline.py`, then the two `result_block(...)` calls and the `stats` tallies in
`fig2_predictor_inputs.py`.

## How the four panels of Figure 1 are built

Every panel draws the same architecture via `arch()`, with two reserved columns kept
empty in the default drawing so a panel can fill them without moving anything else:

| Reserved | Filled by | What goes there |
|---|---|---|
| `PRE_X` (before the encoders) | **1c** | the fused AA + 3Di input token |
| `GAP_X` (between encoders and Light Attention) | **1a** | the late 3Di concat |
| `enc_mark` | **1b** | highlights the encoder, expanded below |
| `slot_mark` | **1d** | highlights `add_scores`; its labels move above the cell so the feed line can arrive from below |

Panel 1b shades each encoder layer by its measured probe PCC (`ramp()`), which makes
ProstT5's top-layer decay and Ankh's flat plateau visible at a glance. Only layers
0–3 and the top three are measured; the interior is drawn at the measured plateau
value and the figure says so.
