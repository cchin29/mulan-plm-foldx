# DECK_HOWTO — building, checking and sharing the reveal.js decks

Scope: a deck's source tree, built locally under `scripts_plots/artifacts/<date>/deck/` and not
tracked, and the single self-contained file `bundle_deck.py` produces from it. The published copy of
the current deck is `docs/slides/mulan-plm-foldx-slides.html`.

---

## 1. Build

```bash
# from the repo root; the deck source tree is built locally and is not tracked
python3 scripts_plots/bundle_deck.py scripts_plots/artifacts/<date>/deck/index.html \
    --strict -o docs/slides/mulan-plm-foldx-slides.html
```

`index.html` alone is useless outside its folder (relative refs to `vendor/`
and `assets/`). The bundler inlines CSS, JS, images (base64) and the iframe
widgets (`srcdoc`) into one file that works offline, by double-click, from
`file://`, as an email attachment.

Flags:

| flag | effect |
|---|---|
| `-o OUT` | output path |
| `--fix-viewport-units` | mechanically rewrite `vh`/`vw` → `px` **in the output only** (§2) |
| `--strict` | exit 1 if any audit fails — use in CI / pre-send checks |

Known benign warning: `css @imports inlined/dropped: 0/1`. The dropped import is
`white.css`'s `./fonts/source-sans-pro/source-sans-pro.css`; that folder was
never vendored. Harmless — the deck's own CSS sets `.reveal{font-family:system-ui,…}`.

---

## 2. THE TRAP: never size slide content in `vh` / `vw`

reveal.js lays every slide out inside a **fixed canvas** —
`Reveal.initialize({width:1280, height:720})` — and CSS-transform-scales it to
the window. `vh`/`vw` resolve against the **real browser window**, not the
canvas, so an element sized in `vh` grows the bigger the viewer's window and
overflows/clips. It looked fine on the authoring screen and was cut off on a
different one (2026-07-23 deck, 6/13 slides, window-size-dependent).

**Rule: size everything in `px` (or `%`) against the 1280×720 canvas — never a
viewport unit.** `--fix-viewport-units` is a stopgap for re-bundling an old
deck; the real fix is px values in `index.html`. `%` also works (resolves
against the canvas). `vh`/`vw` never do.

---

## 2b. PREFERRED LAYOUT: wide plots that scroll within the slide (current standard, 2026-07-28)

The plots are the point of this deck. **Prefer large, full-width figures and let
the slide scroll vertically, rather than shrinking figures to fit 720 px.** The
viewer is a browser, not a projector — scrolling a slide is fine and expected.
This supersedes the older "shrink every figure to fit 720 px" guidance for
*figure sizing*; the **no-`vh` rule in §2 still stands** (that fixed the actual
clipping bug — window-dependence — and is independent of this).

The recipe that scrolls **without** re-introducing clipping or breaking reveal's
centering — all four parts are required:

1. **Inner scroll body.** Wrap every `<section>`'s content in
   `<div class="slidebody">…</div>` and scroll *that*, not the `<section>`:

   ```css
   .reveal .slides>section{padding:0}
   .slidebody{box-sizing:border-box;height:690px;overflow-y:auto;overflow-x:hidden;padding:4px 14px 20px}
   .slidebody::-webkit-scrollbar{width:9px}
   .slidebody::-webkit-scrollbar-thumb{background:#c7c6c0;border-radius:6px}
   ```

   Scroll the **inner div**, never the `<section>` itself. Putting
   `height`/`overflow` on the `<section>` makes reveal mis-measure it and apply a
   spurious `translateX` to the slide (it shifts right and clips — verified).
   The `<section>` must stay reveal-normal so reveal centers it.

2. **`transition:'none'`** in `Reveal.initialize`. With the default `'slide'`
   transition the horizontal slides animate in from the side; a scrollable deck
   doesn't want that, and — importantly — a slide measured mid-transition reads
   as horizontally offset (false positive when checking centering). `'none'`
   settles instantly and centered.

3. **px sizing, full-width figures.** `.fig img{max-height:none;max-width:100%}`
   renders wide plots at ~1210 px (full canvas width); tall plots just make the
   slidebody scroll. Iframe widgets get an explicit px height (they scroll
   internally too).

4. **Override reveal's iframe cap.** reveal.css sets `max-height:95%` on
   iframes, which silently clamps widget heights to ~635 px. Beat it with a
   more-specific selector:

   ```css
   .reveal .slides section iframe.framey{height:640px;max-height:none}
   ```

   then per-widget inline `style="height:…"` (e.g. 680–760 px) wins for the
   ones that want to be taller. Without the `max-height:none` override the
   inline heights are ignored.

Rule of thumb: `.slidebody` is 690 px tall (720 − a little). A slide fits with no
scroll if kicker + h2 + take + figure ≤ ~690 px; otherwise it scrolls — which is
fine. Stacking two full plots on one slide (with `.cap` captions) is a normal,
intended pattern here.

---

## 3. Verify the bundle

The bundler's audit catches viewport units, missing assets and residual local
refs. It cannot catch layout problems — those need a render. For the
scroll-within-slide layout the two checks that matter (headless, playwright):

- **Horizontal fit / centering** — after `Reveal.slide(i)` **and a ≥600 ms
  settle** (transitions must finish, or use `transition:'none'`), every slide's
  `section.getBoundingClientRect().right` must be `≤ window.innerWidth` and
  `.left ≥ 0`, at 1280, 1440, 1920, 2560 wide. A section extending past the
  right edge = the reveal-mis-positioning bug in §2b.1.
- **Window-independence** — each `.slidebody.scrollHeight` must be **identical**
  across those viewports. Different heights at different windows = a stray `vh`
  leaked back in.

The old "does the slide overflow 720 px" check no longer applies — with §2b the
tall slides are *meant* to overflow and scroll. Don't gate on it.

Cheap manual check: open the bundle, arrow through all slides maximised on an
external monitor, and scroll each tall one to the bottom.

---

## 4. Hosting

- `deck_selfcontained.html` (~3 MB, fully offline) opens from disk with no server.
- **GitHub Pages** — push the deck, Settings → Pages → deploy from `main`. GitHub
  serves Pages for public repositories on the free plan.
- **Netlify / Cloudflare Pages** — drag-and-drop a folder.
- **PDF fallback** — `?print-pdf` then print-to-PDF. Note: print-pdf renders one
   fixed canvas per slide, so a scrolling slide's off-screen content is **lost**
   in the PDF. For the PDF export, temporarily fit tall slides to 690 px (or
   split them) before printing.

---

## 5. Changelog

- **2026-07-28** — Adopted the **wide-plots-that-scroll** layout as the standard
  (§2b): inner `.slidebody` scroll container (690 px, px-sized), `transition:'none'`,
  full-width `.fig img`, and the `.framey` `max-height:none` override for reveal's
  95% iframe cap. Fixes "plots too small" without re-introducing clipping; still
  window-independent. Diagnosed two gotchas: `height`/`overflow` on the
  `<section>` (not the inner div) makes reveal apply a spurious `translateX` that
  clips right; and measuring centering mid-transition gives a false offset (settle
  ≥600 ms or use `transition:'none'`). Slide 2 now shows both the 50 ep quick-screen
  and the 300 ep converged ΔΔG runs (they are not the same plot; slide 3 panel 1 is
  the 300 ep run).
- **2026-07-28 (earlier)** — Diagnosed the `vh` clipping bug in the 2026-07-23
  deck (6/13 slides, window-size-dependent). Added the viewport-unit audit,
  `--fix-viewport-units` and `--strict` to `bundle_deck.py`. The px viewport fix
  is now applied in `2026-07-26/deck/index.html`.
