#!/usr/bin/env python3
"""Bundle a multi-file reveal.js deck into ONE self-contained .html for sharing.

A deck is built locally as a folder tree, under scripts_plots/artifacts/<date>/deck/ (not tracked):
    index.html  +  vendor/{reveal.css,white.css,reveal.js}  +  assets/{*.png,*.html}
`index.html` alone is useless (relative refs), so this inlines everything into a
single file you can email / double-click / drop anywhere, offline:

  * <link rel=stylesheet href="vendor/*.css">  -> inline <style> (with @import
    and url(...) resolved: local files -> data: URI, missing local -> dropped)
  * <script src="vendor/*.js">                 -> inline <script>
  * <img src="assets/*.png">                   -> src="data:image/png;base64,...">
  * <iframe src="assets/*.html">               -> srcdoc="..."  (widget inlined,
    its own images recursively data-URI'd; outbound http(s) links left intact)

It also AUDITS the deck's own CSS for viewport units (vh/vw/...), which silently
break reveal.js layout -- see `audit_viewport_units` and DECK_HOWTO.md.

Pure stdlib, no deps. Re-run it on each new deck version:

    python3 scripts_plots/bundle_deck.py <deck>/index.html -o docs/slides/mulan-plm-foldx-slides.html
    # <deck> is the local source tree, e.g. scripts_plots/artifacts/<date>/deck

Options:
    -o OUT               output path (default: <deckdir>/../<deckdir_name>_selfcontained.html)
    --fix-viewport-units rewrite vh/vw in the deck's own CSS to px against the
                         Reveal canvas, in the OUTPUT only (source left alone)
    --strict             exit 1 if the audit finds anything
"""
import argparse
import base64
import html
import mimetypes
import re
import sys
from pathlib import Path

TEXT_CSS = {".css"}
TEXT_JS = {".js"}

# reveal.js lays every slide out inside a FIXED canvas (Reveal.initialize
# width/height) and then CSS-transform-scales that canvas to fit the window.
# `vh`/`vw` resolve against the real browser window, NOT the canvas -- so an
# element sized in vh is measured in one coordinate system and placed in
# another. Result: the taller the viewer's window, the more the slide overflows
# and gets clipped at the bottom. Looks fine on the author's screen, cut off on
# everyone else's. Always size slide content in px (or %) against the canvas.
VIEWPORT_UNITS = ("vh", "vw", "vmin", "vmax",
                  "svh", "lvh", "dvh", "svw", "lvw", "dvw")
VP_UNIT_RE = re.compile(
    r"(?<![\w.-])(\d+(?:\.\d+)?)(" + "|".join(VIEWPORT_UNITS) + r")\b")
DEFAULT_CANVAS = (960, 700)  # reveal.js defaults if initialize() omits them


def is_remote(url: str) -> bool:
    return url.startswith(("http://", "https://", "data:", "//", "#", "mailto:"))


def data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def inline_css(css_text: str, css_dir: Path, stats: dict) -> str:
    """Resolve @import and url(...) refs inside a CSS blob so it stands alone."""

    def repl_import(m):
        raw = m.group(1).strip().strip("'\"")
        if is_remote(raw):
            return m.group(0)  # leave remote imports as-is
        target = (css_dir / raw).resolve()
        if target.exists():
            stats["css_imports"] += 1
            return f"/* inlined @import {raw} */\n" + inline_css(
                target.read_text(encoding="utf-8"), target.parent, stats
            )
        stats["dropped_imports"] += 1
        return f"/* dropped missing @import: {raw} */"

    def repl_url(m):
        raw = m.group(1).strip().strip("'\"")
        if is_remote(raw):
            return m.group(0)
        target = (css_dir / raw).resolve()
        if target.exists():
            stats["css_assets"] += 1
            return f"url({data_uri(target)})"
        stats["dropped_css_assets"] += 1
        return m.group(0)  # keep (broken) rather than mangle unknown syntax

    css_text = re.sub(r"@import\s+url\(([^)]+)\)\s*;?", repl_import, css_text)
    css_text = re.sub(r"url\(([^)]+)\)", repl_url, css_text)
    return css_text


def inline_html(html_text: str, base_dir: Path, stats: dict, depth: int = 0) -> str:
    """Inline every local resource referenced by an HTML document. Recursive for
    iframe widgets (which may themselves reference images)."""

    def repl_link(m):
        tag = m.group(0)
        if "stylesheet" not in tag:
            return tag
        hm = re.search(r'href="([^"]+)"', tag)
        if not hm or is_remote(hm.group(1)):
            return tag
        target = (base_dir / hm.group(1)).resolve()
        if not target.exists() or target.suffix.lower() not in TEXT_CSS:
            return tag
        stats["css"] += 1
        media = re.search(r'media="([^"]+)"', tag)
        media_attr = f' media="{media.group(1)}"' if media else ""
        css = inline_css(target.read_text(encoding="utf-8"), target.parent, stats)
        return f'<style data-inlined-from="{hm.group(1)}"{media_attr}>\n{css}\n</style>'

    def repl_script(m):
        tag, inner = m.group(0), m.group("src")
        if is_remote(inner):
            return tag
        target = (base_dir / inner).resolve()
        if not target.exists() or target.suffix.lower() not in TEXT_JS:
            return tag
        stats["js"] += 1
        js = target.read_text(encoding="utf-8")
        return f'<script data-inlined-from="{inner}">\n{js}\n</script>'

    def repl_img(m):
        tag = m.group(0)
        sm = re.search(r'src="([^"]+)"', tag)
        if not sm or is_remote(sm.group(1)):
            return tag
        target = (base_dir / sm.group(1)).resolve()
        if not target.exists():
            stats["missing"].append(sm.group(1))
            return tag
        stats["img"] += 1
        return tag.replace(f'src="{sm.group(1)}"', f'src="{data_uri(target)}"', 1)

    def repl_iframe(m):
        tag = m.group(0)
        sm = re.search(r'src="([^"]+)"', tag)
        if not sm or is_remote(sm.group(1)):
            return tag
        target = (base_dir / sm.group(1)).resolve()
        if not target.exists() or target.suffix.lower() not in {".html", ".htm"}:
            return tag
        stats["iframe"] += 1
        widget = inline_html(
            target.read_text(encoding="utf-8"), target.parent, stats, depth + 1
        )
        srcdoc = html.escape(widget, quote=True)
        # splice srcdoc INTO the opening tag (before its first '>'), dropping the
        # now-defunct src=. tag == '<iframe ...></iframe>'; must not append after it.
        gt = tag.index(">")
        opening, rest = tag[:gt], tag[gt:]          # '<iframe ... attrs' , '></iframe>'
        opening = re.sub(r'\ssrc="[^"]+"', "", opening, count=1)
        return f'{opening} srcdoc="{srcdoc}"{rest}'

    html_text = re.sub(r"<link\b[^>]*>", repl_link, html_text, flags=re.I)
    html_text = re.sub(
        r'<script\b[^>]*\bsrc="(?P<src>[^"]+)"[^>]*>\s*</script>',
        repl_script,
        html_text,
        flags=re.I,
    )
    html_text = re.sub(r"<img\b[^>]*?/?>", repl_img, html_text, flags=re.I)
    html_text = re.sub(
        r"<iframe\b[^>]*>\s*</iframe>", repl_iframe, html_text, flags=re.I | re.S
    )
    return html_text


def parse_canvas(html_text: str) -> tuple:
    """Pull width/height out of Reveal.initialize({...}); fall back to defaults."""
    m = re.search(r"Reveal\.initialize\s*\(\s*\{(.*?)\}\s*\)", html_text, re.S)
    if not m:
        return DEFAULT_CANVAS
    body = m.group(1)
    def num(key, default):
        km = re.search(rf"\b{key}\s*:\s*(\d+(?:\.\d+)?)", body)
        return float(km.group(1)) if km else default
    return num("width", DEFAULT_CANVAS[0]), num("height", DEFAULT_CANVAS[1])


def _author_css_spans(html_text: str):
    """Yield (start, end) spans of CSS the DECK owns: <style> bodies and inline
    style="..." attribute values. Script bodies are masked out first so that JS
    template literals ('100vh' inside reveal.js) are never matched."""
    masked = re.sub(r"(<script\b[^>]*>)(.*?)(</script>)",
                    lambda m: m.group(1) + " " * len(m.group(2)) + m.group(3),
                    html_text, flags=re.I | re.S)
    for m in re.finditer(r"<style\b[^>]*>(.*?)</style>", masked, re.I | re.S):
        yield m.start(1), m.end(1)
    for m in re.finditer(r'\bstyle="([^"]*)"', masked, re.I):
        yield m.start(1), m.end(1)


def audit_viewport_units(html_text: str) -> list:
    """Find viewport-unit lengths in the deck's own CSS. Returns a list of
    (line_no, unit_text, context, suggested_px_note)."""
    w, h = parse_canvas(html_text)
    hits = []
    for start, end in _author_css_spans(html_text):
        for m in VP_UNIT_RE.finditer(html_text, start, end):
            val, unit = float(m.group(1)), m.group(2)
            basis = w if unit.endswith("vw") else h
            if unit == "vmin":
                basis = min(w, h)
            elif unit == "vmax":
                basis = max(w, h)
            px = round(val / 100.0 * basis)
            line = html_text.count("\n", 0, m.start()) + 1
            ctx_start = html_text.rfind(";", start, m.start()) + 1 or m.start()
            ctx = html_text[max(ctx_start, m.start() - 60):m.end() + 20]
            ctx = " ".join(ctx.split())[:80]
            hits.append((line, m.group(0), ctx, f"{px}px"))
    return sorted(hits)


def convert_viewport_units(html_text: str, verbose: bool = True) -> str:
    """Rewrite viewport units in the deck's own CSS to px against the Reveal
    canvas. MECHANICAL: it removes the window-dependence (the actual bug) but
    does not guarantee the slide fits -- verify per DECK_HOWTO.md."""
    w, h = parse_canvas(html_text)
    spans = sorted(_author_css_spans(html_text))
    out, cursor, n = [], 0, 0
    for start, end in spans:
        if start < cursor:
            continue
        out.append(html_text[cursor:start])
        chunk = html_text[start:end]

        def repl(m):
            nonlocal n
            val, unit = float(m.group(1)), m.group(2)
            basis = w if unit.endswith("vw") else h
            if unit == "vmin":
                basis = min(w, h)
            elif unit == "vmax":
                basis = max(w, h)
            n += 1
            return f"{round(val / 100.0 * basis)}px"

        out.append(VP_UNIT_RE.sub(repl, chunk))
        cursor = end
    out.append(html_text[cursor:])
    if verbose:
        print(f"  --fix-viewport-units: rewrote {n} viewport-unit length(s) "
              f"against the {w:.0f}x{h:.0f} canvas (output only; source untouched)")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("index", help="path to the deck's index.html")
    ap.add_argument("-o", "--out", help="output .html (default: "
                    "<deckdir>/../<deckdir_name>_selfcontained.html)")
    ap.add_argument("--fix-viewport-units", action="store_true",
                    help="rewrite vh/vw in the deck's own CSS to px against the "
                         "Reveal canvas (output only; index.html is not modified)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if the layout audit or self-containment audit fails")
    args = ap.parse_args()

    index = Path(args.index).resolve()
    if not index.is_file():
        sys.exit(f"!! not a file: {index}")
    deck_dir = index.parent
    out = Path(args.out).resolve() if args.out else \
        deck_dir.parent / f"{deck_dir.name}_selfcontained.html"

    stats = {k: 0 for k in ("css", "js", "img", "iframe", "css_imports",
                            "css_assets", "dropped_imports", "dropped_css_assets")}
    stats["missing"] = []

    source = index.read_text(encoding="utf-8")

    # LAYOUT AUDIT -- run on the SOURCE, before anything is inlined, so vendor
    # CSS and base64 blobs can never produce false positives.
    vp_hits = audit_viewport_units(source)
    if args.fix_viewport_units:
        source = convert_viewport_units(source)

    bundled = inline_html(source, deck_dir, stats)
    out.write_text(bundled, encoding="utf-8")

    # self-containment audit: any residual local (non-remote, non-data) refs in
    # the actual MARKUP? Strip inlined <script>/<style> bodies first (they carry
    # JS template-literals like src="${e}" that are not real references).
    markup = re.sub(r"<script\b[^>]*>.*?</script>", "", bundled, flags=re.I | re.S)
    markup = re.sub(r"<style\b[^>]*>.*?</style>", "", markup, flags=re.I | re.S)
    residual = [
        r for r in re.findall(r'(?:src|href)="([^"]+)"', markup)
        if not is_remote(r) and not r.startswith("data:") and "${" not in r
    ]

    src_kb = sum(f.stat().st_size for f in deck_dir.rglob("*") if f.is_file()) / 1024
    print(f"[bundle_deck] {index}  ->  {out}")
    print(f"  inlined: {stats['css']} css, {stats['js']} js, {stats['img']} img, "
          f"{stats['iframe']} iframe widgets")
    print(f"  css @imports inlined/dropped: {stats['css_imports']}/{stats['dropped_imports']};"
          f" css url() assets inlined: {stats['css_assets']}")
    print(f"  size: {src_kb:.0f} KB (tree)  ->  {out.stat().st_size/1024:.0f} KB (single file)")
    if stats["missing"]:
        print(f"  !! WARNING missing referenced files: {stats['missing']}")
    if residual:
        print(f"  !! WARNING residual local refs (NOT self-contained): {sorted(set(residual))}")
    else:
        print("  OK fully self-contained (no residual local refs, no external fetches)")

    # layout audit report
    cw, ch = parse_canvas(index.read_text(encoding="utf-8"))
    if vp_hits:
        verb = "rewrote" if args.fix_viewport_units else "found"
        print(f"  !! LAYOUT {verb} {len(vp_hits)} viewport-unit length(s) in the "
              f"deck's own CSS (canvas is {cw:.0f}x{ch:.0f}):")
        for line, unit, ctx, px in vp_hits:
            print(f"       index.html:{line}  {unit:>7}  -> {px:>6}   {ctx}")
        print("     vh/vw resolve against the BROWSER WINDOW, but reveal.js lays")
        print("     slides out in the fixed canvas above and then scales it. Slides")
        print("     sized in vh therefore clip differently on every screen -- fine")
        print("     for you, cut off for whoever you send the deck to.")
        if args.fix_viewport_units:
            print("     Rewritten in the OUTPUT only. This kills the window-dependence")
            print("     but does NOT prove the slide fits -- verify per DECK_HOWTO.md,")
            print("     then fold the values back into deck/index.html.")
        else:
            print("     Fix in deck/index.html (see DECK_HOWTO.md), or re-run with")
            print("     --fix-viewport-units for a mechanical px conversion.")
    else:
        print(f"  OK layout: no viewport units in the deck's CSS "
              f"(canvas {cw:.0f}x{ch:.0f})")

    if args.strict and (vp_hits or residual or stats["missing"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
