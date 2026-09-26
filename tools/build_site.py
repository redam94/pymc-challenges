"""Build a static website (GitHub Pages) from the executed notebooks.

    uv run python tools/build_site.py                  # -> _site/
    uv run python tools/build_site.py --out docs --repo-url https://github.com/<you>/<repo>
    python -m http.server -d _site                     # preview at http://localhost:8000

Nothing is executed: the pages are rendered from the outputs already stored in
notebooks/examples (E*, D*), notebooks/challenges and notebooks/solutions. Only
nbformat, nbconvert (for its markdown renderer), pygments and pyyaml are needed, so
the GitHub Actions workflow does not have to install the modelling stack.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
import yaml
from nbconvert.filters.markdown_mistune import markdown2html_mistune
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

sys.path.insert(0, str(Path(__file__).parent))
import site_content as content  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
ASSETS = Path(__file__).parent / "site_assets"
HINTS = ROOT / "src" / "pymc_challenges" / "hints"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LOCAL_PATH = re.compile(re.escape(str(ROOT)) + r"/?|/Users/[^/\s\"']+/")
MATHJAX = (
    '<script>window.MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]],'
    'displayMath:[["$$","$$"],["\\\\[","\\\\]"]],processEscapes:true,tags:"none"},'
    'options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]},'
    'svg:{fontCache:"global"}};</script>\n'
    '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>'
)
PLOTLY = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"></script>'


# --------------------------------------------------------------------------- catalogue


@dataclass
class Entry:
    id: str  # E01, D02, C05
    kind: str  # example | challenge
    nb_path: Path
    title: str = ""
    data: str = ""
    techniques: str = ""
    stars: int = 0
    series: str = ""
    solution_path: Path | None = None
    meta: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.nb_path.stem

    @property
    def url(self) -> str:
        folder = "examples" if self.kind == "example" else "challenges"
        return f"{folder}/{self.slug}.html"

    @property
    def solution_url(self) -> str:
        return f"solutions/{self.slug}_solution.html"


def md_inline(text: str) -> str:
    """Render one line of markdown without the surrounding <p>."""
    out = markdown2html_mistune(text).strip()
    return re.sub(r"^<p>(.*)</p>$", r"\1", out, flags=re.S)


def parse_readme() -> dict[str, dict]:
    """Topic / data / techniques (or stars) for every row of the README curriculum tables."""
    rows = {}
    for line in (ROOT / "README.md").read_text().splitlines():
        m = re.match(r"^\| \*\*([CDE]\d\d)\*\* \|(.*)\|\s*$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split(" | ")]
        if m.group(1).startswith("C"):
            topic, data, stars, deliver = cells
            rows[m.group(1)] = dict(title=topic, data=data, stars=int(stars), techniques=deliver)
        else:
            topic, data, tech = cells
            rows[m.group(1)] = dict(title=topic, data=data, techniques=tech)
    return rows


def catalogue() -> list[Entry]:
    readme = parse_readme()
    entries = []
    for p in sorted((NB / "examples").glob("*.ipynb")):
        e = Entry(id=p.stem[:3], kind="example", nb_path=p, **readme.get(p.stem[:3], {}))
        entries.append(e)
    for p in sorted((NB / "challenges").glob("*.ipynb")):
        sol = NB / "solutions" / f"{p.stem}_solution.ipynb"
        e = Entry(id=p.stem[:3], kind="challenge", nb_path=p,
                  solution_path=sol if sol.exists() else None, **readme.get(p.stem[:3], {}))
        entries.append(e)
    by_id = {e.id: e for e in entries}
    for s in content.SERIES:
        for i in s["ids"]:
            if i in by_id:
                by_id[i].series = s["key"]
    for e in entries:
        if not e.title:  # not in the README yet: fall back to the notebook's own H1
            nb = nbformat.read(e.nb_path, 4)
            e.title = nb.cells[0].source.splitlines()[0].lstrip("# ").split("·", 1)[-1].strip()
    return entries


# --------------------------------------------------------------------------- rendering

FORMATTER = HtmlFormatter(nowrap=True)
LEXER = PythonLexer()


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", html.unescape(text))
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_-]+", "-", text)[:60] or "section"


def scrub(text: str) -> str:
    return LOCAL_PATH.sub("", ANSI.sub("", text))


def render_markdown(src: str, attachments: dict | None = None) -> str:
    if attachments:
        for name, data in attachments.items():
            for mime, b64 in data.items():
                src = src.replace(f"attachment:{name}", f"data:{mime};base64,{b64}")
    return markdown2html_mistune(src)


XREF_TARGETS: dict[str, str] = {}  # id -> url from the site root, filled by build()


def cross_link(body: str, depth: int, current: str) -> str:
    """Link mentions such as "E02" or "C07" in prose to their pages (not inside code, links or math)."""
    up = "../" * depth
    out, skip = [], 0
    for tok in re.split(r"(<[^>]+>)", body):
        if tok.startswith("<"):
            m = re.match(r"</?(code|a|pre|script|style|h[1-4])\b", tok)
            if m:
                skip += -1 if tok.startswith("</") else 1
            out.append(tok)
        elif skip or "$" in tok:
            out.append(tok)
        else:
            out.append(re.sub(r"\b([CDE]\d\d)\b(?![-_])",
                              lambda m: (f'<a class="xref" href="{up}{XREF_TARGETS[m.group(1)]}">{m.group(1)}</a>'
                                         if m.group(1) in XREF_TARGETS and m.group(1) != current else m.group(1)),
                              tok))
    return "".join(out)


def code_block(src: str, extra_class: str = "") -> str:
    body = highlight(src.rstrip(), LEXER, FORMATTER)
    return f'<div class="code {extra_class}"><pre><code>{body}</code></pre></div>'


class Page:
    """Collects the HTML and side files (figures) of one rendered notebook."""

    def __init__(self, out_dir: Path, stem: str):
        self.current = stem[:3]
        self.out_dir = out_dir
        self.fig_dir = out_dir / "figures" / stem
        self.fig_rel = f"figures/{stem}"
        self.n_fig = 0
        self.needs_plotly = False
        self.parts: list[str] = []

    def save(self, data: bytes, ext: str) -> str:
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.n_fig += 1
        name = f"fig{self.n_fig:02d}.{ext}"
        (self.fig_dir / name).write_bytes(data)
        return f"{self.fig_rel}/{name}"

    # -- outputs -------------------------------------------------------------
    def output(self, o) -> str | None:
        t = o.output_type
        if t == "stream":
            if o.name != "stdout":
                return None  # warnings, sampler banners and progress bars
            text = scrub(o.text)
            text = "\n".join(l.rsplit("\r", 1)[-1] for l in text.split("\n")).strip("\n")
            return f'<pre class="out-text">{html.escape(text)}</pre>' if text.strip() else None
        if t == "error":
            tb = scrub("\n".join(o.get("traceback", [])))
            return f'<pre class="out-text out-error">{html.escape(tb)}</pre>'
        d = o.get("data", {})
        if "application/vnd.plotly.v1+json" in d:
            self.needs_plotly = True
            fig = d["application/vnd.plotly.v1+json"]
            uid = "plotly-" + hashlib.md5(json.dumps(fig, sort_keys=True).encode()).hexdigest()[:10]
            payload = json.dumps(fig).replace("</", "<\\/")
            return (f'<div class="out-plotly" id="{uid}"></div>'
                    f'<script type="application/json" data-plotly="{uid}">{payload}</script>')
        if "image/svg+xml" in d:
            svg = d["image/svg+xml"]
            return f'<figure class="out-fig"><img src="{self.save(svg.encode(), "svg")}" alt="" loading="lazy"></figure>'
        if "image/png" in d or "image/jpeg" in d:
            ext = "png" if "image/png" in d else "jpg"
            raw = base64.b64decode(d[f"image/{'png' if ext == 'png' else 'jpeg'}"])
            src = self.save(raw, ext)
            w = o.get("metadata", {}).get(f"image/{ext}", {}).get("width")
            size = f' style="max-width:{w}px"' if w else ""
            return f'<figure class="out-fig"><img src="{src}" alt="Figure" loading="lazy"{size}></figure>'
        if "text/html" in d:
            h = scrub(d["text/html"])
            if "window.PlotlyConfig" in h:  # plotly's notebook init snippet; the page loads plotly itself
                return None
            if "function Animation" in h or "anim-controls" in h:  # matplotlib to_jshtml
                h = re.sub(r'href="https://maxcdn[^"]*font-awesome[^"]*"',  # maxcdn is gone
                           'href="https://cdn.jsdelivr.net/npm/font-awesome@4.7.0/css/font-awesome.min.css"', h)
                doc = ("<!doctype html><meta charset=utf-8><style>body{margin:0;font:14px system-ui;"
                       "display:flex;justify-content:center}img{max-width:100%}</style>" + h)
                src = self.save(doc.encode(), "html")
                return (f'<figure class="out-anim"><iframe src="{src}" loading="lazy" '
                        f'title="Animation" onload="fitFrame(this)"></iframe></figure>')
            text_only = re.sub(r"<[^>]+>|\s", "", re.sub(r"<style.*?</style>", "", h, flags=re.S))
            if not text_only and "<img" not in h and "<svg" not in h:
                return None
            cls = "out-html"
            if 'class="dataframe"' in h:
                cls += " out-df"
            elif "xr-wrap" in h or "xr-header" in h:
                cls += " out-xr"
            return f'<div class="{cls}">{h}</div>'
        if "text/markdown" in d:
            return f'<div class="out-md">{render_markdown(scrub(d["text/markdown"]))}</div>'
        if "text/latex" in d:
            tex = d["text/latex"].replace("\\_", "_")  # \text{} in MathJax prints underscores literally
            return f'<div class="out-latex">{html.escape(tex)}</div>'
        if "text/plain" in d:
            text = scrub(d["text/plain"])
            if re.fullmatch(r"<(Figure|matplotlib|Axes|IPython)[^>]*>|\[<matplotlib.*\]", text.strip()):
                return None
            return f'<pre class="out-text">{html.escape(text)}</pre>'
        return None

    # -- cells ---------------------------------------------------------------
    def code_cell(self, cell, show_code: bool = True):
        outs = [r for r in (self.output(o) for o in cell.get("outputs", [])) if r]
        if not cell.source.strip() and not outs:
            return
        block = ['<div class="cell">']
        if show_code and cell.source.strip():
            n = cell.source.count("\n") + 1
            block.append(code_block(cell.source, "long" if n > 30 else ""))
        if outs:
            block.append('<div class="outputs">' + "".join(outs) + "</div>")
        block.append("</div>")
        self.parts.append("".join(block))

    def markdown_cell(self, cell):
        body = cross_link(render_markdown(cell.source, cell.get("attachments")), 1, self.current)
        self.parts.append(f'<div class="prose">{body}</div>')


def split_header(src: str) -> tuple[str, list[tuple[str, str]], str]:
    """Pull the H1 and the leading `| **Key** | value |` table out of the first cell."""
    lines = src.splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines.pop(0)[2:].strip()
    meta, rest, in_table, done = [], [], False, False
    for line in lines:
        if not done and line.startswith("|"):
            in_table = True
            m = re.match(r"^\|\s*\*\*(.+?)\*\*\s*\|(.*)\|\s*$", line)
            if m:
                meta.append((m.group(1), m.group(2).strip()))
            continue
        if in_table:
            done = True
        rest.append(line)
    return title, meta, "\n".join(rest).strip()


def headings_and_ids(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Give every h2 a readable id; return the TOC entries."""
    toc, seen = [], set()

    def fix(m):
        level, inner = m.group(1), m.group(3)
        inner = re.sub(r'<a class="anchor-link".*?</a>', "", inner)
        sid = slugify(inner)
        base, k = sid, 2
        while sid in seen:
            sid, k = f"{base}-{k}", k + 1
        seen.add(sid)
        if level == "2":
            toc.append((sid, html.unescape(re.sub(r"<[^>]+>", "", inner)).strip()))
        return f'<h{level} id="{sid}"><a class="h-anchor" href="#{sid}" aria-hidden="true">#</a>{inner}</h{level}>'

    body = re.sub(r'<h([2-4])(?: id="[^"]*")?>(\s*)(.*?)</h\1>', fix, body, flags=re.S)
    return body, toc


# --------------------------------------------------------------------------- page shell


def shell(*, title: str, body: str, depth: int, active: str, description: str = "",
          head_extra: str = "", toc: list | None = None, page_class: str = "") -> str:
    up = "../" * depth
    nav = "".join(
        f'<a href="{up}{href}"{" aria-current=page" if key == active else ""}>{label}</a>'
        for key, href, label in [("examples", "examples.html", "Examples"),
                                 ("challenges", "challenges.html", "Challenges"),
                                 ("guide", "guide.html", "Guide"),
                                 ("about", "about.html", "About")]
    )
    toc_html = ""
    if toc:
        items = "".join(f'<li><a href="#{sid}">{html.escape(text)}</a></li>' for sid, text in toc)
        toc_html = (f'<nav class="toc" aria-label="On this page"><details open><summary>On this page</summary>'
                    f"<ol>{items}</ol></details></nav>")
    desc = html.escape(description or content.TAGLINE, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{desc}">
<link rel="icon" href="{up}assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{up}assets/style.css">
<script>try{{var t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t;if(localStorage.getItem("hideCode")==="1")document.documentElement.classList.add("hide-code")}}catch(e){{}}</script>
{head_extra}
</head>
<body class="{page_class}">
<a class="skip" href="#main">Skip to content</a>
<header class="site-header">
  <div class="bar">
    <a class="brand" href="{up}index.html"><span class="brand-mark" aria-hidden="true">p(θ|y)</span> {content.SITE_NAME}</a>
    <nav class="site-nav" aria-label="Main">{nav}</nav>
    <button class="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode">◐</button>
  </div>
</header>
<div class="layout{' with-toc' if toc else ''}">
{toc_html}
<main id="main">
{body}
</main>
</div>
<footer class="site-footer">
  <p>{content.FOOTER}</p>
</footer>
<script src="{up}assets/site.js"></script>
</body>
</html>
"""


def meta_block(meta: list[tuple[str, str]]) -> str:
    if not meta:
        return ""
    rows = "".join(f"<div><dt>{html.escape(k)}</dt><dd>{md_inline(v)}</dd></div>" for k, v in meta)
    return f'<dl class="meta">{rows}</dl>'


def pager(prev: Entry | None, nxt: Entry | None, depth: int) -> str:
    up = "../" * depth
    def link(e, rel, label):
        if not e:
            return "<span></span>"
        return (f'<a class="pager-{rel}" href="{up}{e.url}"><small>{label}</small>'
                f"<span>{e.id} · {html.escape(e.title)}</span></a>")
    return f'<nav class="pager">{link(prev, "prev", "← Previous")}{link(nxt, "next", "Next →")}</nav>'


def repo_links(e: Entry, repo_url: str | None, path: Path) -> str:
    links = []
    if repo_url:
        rel = path.relative_to(ROOT).as_posix()
        links.append(f'<a href="{repo_url}/blob/main/{rel}">View notebook on GitHub</a>')
        links.append(f'<a href="https://nbviewer.org/github/{repo_url.split("github.com/")[-1]}/blob/main/{rel}">nbviewer</a>')
    links.append('<button type="button" class="code-toggle">Hide code</button>')
    return '<div class="page-actions">' + "".join(links) + "</div>"


def render_example(e: Entry, out: Path, prev, nxt, repo_url) -> None:
    nb = nbformat.read(e.nb_path, 4)
    page = Page(out / "examples", e.slug)
    title, meta, rest = split_header(nb.cells[0].source)
    e.meta = dict(meta)
    if rest:
        page.markdown_cell(nbformat.v4.new_markdown_cell(rest))
    for cell in nb.cells[1:]:
        if cell.cell_type == "markdown":
            page.markdown_cell(cell)
        elif cell.cell_type == "code":
            page.code_cell(cell)
    series = content.SERIES_BY_KEY.get(e.series, {})
    body, toc = headings_and_ids("\n".join(page.parts))
    header = (f'<header class="page-head"><p class="eyebrow"><a href="../examples.html#{e.series}">'
              f'{html.escape(series.get("title", "Examples"))}</a> · {e.id}</p>'
              f"<h1>{md_inline(title.split('·', 1)[-1].strip())}</h1>{meta_block(meta)}"
              f"{repo_links(e, repo_url, e.nb_path)}</header>")
    head = MATHJAX + (PLOTLY if page.needs_plotly else "")
    html_doc = shell(title=f"{e.id} · {re.sub('<[^>]+>', '', md_inline(e.title))} — {content.SITE_NAME}",
                     body=header + f'<article class="notebook">{body}</article>' + pager(prev, nxt, 1),
                     depth=1, active="examples", head_extra=head, toc=toc,
                     description=re.sub("<[^>]+>|`", "", e.techniques)[:300])
    (out / "examples" / f"{e.slug}.html").write_text(html_doc)


def hints_block(task: str, spec: dict) -> str:
    t = spec.get("tasks", {}).get(task)
    if not t:
        return ""
    names = ["Nudge", "Approach", "Skeleton"]
    items = "".join(
        f'<details class="hint"><summary><span class="hint-level">{i + 1}</span> {names[i] if i < 3 else "Hint"}</summary>'
        f'<div class="prose">{render_markdown(h)}</div></details>'
        for i, h in enumerate(t.get("hints", []))
    )
    checks = t.get("checks", {})
    check_html = ""
    if checks:
        lis = "".join(f"<li><code>{html.escape(k)}</code> — {md_inline(v.get('what', ''))}</li>" for k, v in checks.items())
        check_html = f'<div class="checks"><p>Self-checks in the notebook (<code>h.check("{task}", …)</code>):</p><ul>{lis}</ul></div>'
    return (f'<aside class="hints"><p class="hints-title">Hints for this task — open one at a time</p>'
            f"{items}{check_html}</aside>")


def render_challenge(e: Entry, out: Path, prev, nxt, repo_url) -> None:
    nb = nbformat.read(e.nb_path, 4)
    spec = yaml.safe_load((HINTS / f"{e.id}.yaml").read_text()) if (HINTS / f"{e.id}.yaml").exists() else {}
    page = Page(out / "challenges", e.slug)
    title, meta, rest = split_header(nb.cells[0].source)
    e.meta = dict(meta)
    # the challenge notebook is unexecuted; show the data previews from the identical solution cells
    solved = {}
    if e.solution_path:
        for c in nbformat.read(e.solution_path, 4).cells:
            if c.cell_type == "code" and c.get("outputs"):
                solved.setdefault(c.source.strip(), c.outputs)
    if rest:
        page.markdown_cell(nbformat.v4.new_markdown_cell(rest))
    for cell in nb.cells[1:]:
        if cell.cell_type == "markdown":
            page.markdown_cell(cell)
            continue
        src = cell.source.strip()
        m = re.search(r'h\.hint\("(task\d+)"\)', src)
        if m and all(l.strip().startswith("#") for l in src.splitlines()):
            page.parts.append(hints_block(m.group(1), spec))
        elif src == "# YOUR CODE HERE" or "task" in cell.metadata.get("tags", []):
            if not page.parts or "your-turn" not in page.parts[-1]:
                page.parts.append('<div class="your-turn"><span>Your turn</span> Work this task in the notebook.</div>')
        else:
            if not cell.get("outputs") and src in solved:
                cell = nbformat.v4.new_code_cell(cell.source, outputs=solved[src])
            page.code_cell(cell)
    body, toc = headings_and_ids("\n".join(page.parts))
    stars = "★" * e.stars + "☆" * (5 - e.stars)
    sol = ""
    if e.solution_path:
        sol = (f'<section class="solution-cta"><h2 id="reference-solution">Reference solution</h2>'
               f"<p>A complete, executed solution with commentary on every task — including the "
               f"ways the obvious model fails. Try the hint ladder first.</p>"
               f'<a class="button" href="../{e.solution_url}">Open the worked solution →</a></section>')
        toc.append(("reference-solution", "Reference solution"))
    header = (f'<header class="page-head"><p class="eyebrow"><a href="../challenges.html">Challenges</a> · {e.id} · '
              f'<span class="stars" title="difficulty {e.stars} of 5">{stars}</span></p>'
              f"<h1>{md_inline(title.split('·', 1)[-1].strip())}</h1>{meta_block(meta)}"
              f"{repo_links(e, repo_url, e.nb_path)}</header>")
    html_doc = shell(title=f"{e.id} · {re.sub('<[^>]+>', '', md_inline(e.title))} — {content.SITE_NAME}",
                     body=header + f'<article class="notebook">{body}{sol}</article>' + pager(prev, nxt, 1),
                     depth=1, active="challenges", head_extra=MATHJAX, toc=toc,
                     description=re.sub("<[^>]+>|`", "", e.techniques)[:300])
    (out / "challenges" / f"{e.slug}.html").write_text(html_doc)


def render_solution(e: Entry, out: Path, repo_url) -> None:
    nb = nbformat.read(e.solution_path, 4)
    page = Page(out / "solutions", e.solution_path.stem)
    title, meta, rest = split_header(nb.cells[0].source)
    if rest:
        page.markdown_cell(nbformat.v4.new_markdown_cell(rest))
    for cell in nb.cells[1:]:
        if cell.cell_type == "markdown":
            page.markdown_cell(cell)
        elif cell.cell_type == "code":
            if re.fullmatch(r"(#.*\n?)*", cell.source.strip()) and not cell.get("outputs"):
                continue
            page.code_cell(cell)
    body, toc = headings_and_ids("\n".join(page.parts))
    header = (f'<header class="page-head"><p class="eyebrow"><a href="../challenges.html">Challenges</a> · '
              f'<a href="../{e.url}">{e.id}</a> · Solution</p>'
              f"<h1>{md_inline(title.split('·', 1)[-1].strip())}</h1>"
              f'<p class="spoiler">Spoilers. This is the reference solution to <a href="../{e.url}">{e.id}</a>; '
              f"the challenge page has the brief and a ladder of hints.</p>"
              f"{repo_links(e, repo_url, e.solution_path)}</header>")
    html_doc = shell(title=f"{e.id} solution · {re.sub('<[^>]+>', '', md_inline(e.title))} — {content.SITE_NAME}",
                     body=header + f'<article class="notebook">{body}</article>',
                     depth=1, active="challenges", head_extra=MATHJAX + (PLOTLY if page.needs_plotly else ""),
                     toc=toc)
    (out / "solutions" / f"{e.solution_path.stem}.html").write_text(html_doc)


# --------------------------------------------------------------------------- index pages


def card(e: Entry, depth: int = 0) -> str:
    up = "../" * depth
    tech = e.techniques
    if len(tech) > 260:
        tech = tech[:260].rsplit(",", 1)[0] + ", …"
    return (f'<a class="card" href="{up}{e.url}"><span class="card-id">{e.id}</span>'
            f"<span class=\"card-title\">{md_inline(e.title)}</span>"
            f'<span class="card-data">{md_inline(e.data)}</span>'
            f'<span class="card-tech">{md_inline(tech)}</span></a>')


def challenge_rows(entries: list[Entry]) -> str:
    rows = "".join(
        f'<tr><td class="cid"><a href="{e.url}">{e.id}</a></td>'
        f'<td><a href="{e.url}">{md_inline(e.title)}</a><div class="muted">{md_inline(e.data)}</div></td>'
        f'<td class="stars" title="difficulty {e.stars} of 5">{"★" * e.stars}<span class="dim">{"★" * (5 - e.stars)}</span></td>'
        f"<td>{md_inline(e.techniques)}</td></tr>"
        for e in entries
    )
    return (f'<div class="table-wrap"><table class="challenge-table"><thead><tr><th></th><th>Challenge</th>'
            f"<th>Difficulty</th><th>You will have to</th></tr></thead><tbody>{rows}</tbody></table></div>")


def series_sections(examples: list[Entry]) -> str:
    by_id = {e.id: e for e in examples}
    out = []
    for s in content.SERIES:
        es = [by_id[i] for i in s["ids"] if i in by_id]
        if not es:
            continue
        out.append(f'<section class="series" id="{s["key"]}"><div class="series-head"><p class="eyebrow">{s["range"]}</p>'
                   f'<h2>{s["title"]}</h2><p>{s["intro"]}</p></div>'
                   f'<div class="cards">{"".join(card(e) for e in es)}</div></section>')
    return "\n".join(out)


def build(out: Path, repo_url: str | None, only: list[str] | None) -> None:
    entries = catalogue()
    examples = [e for e in entries if e.kind == "example"]
    challenges = [e for e in entries if e.kind == "challenge"]
    XREF_TARGETS.update({e.id: e.url for e in entries})
    order = {i: n for n, s in enumerate(content.SERIES) for i in s["ids"]}
    examples.sort(key=lambda e: (order.get(e.id, 99), e.id))

    if out.exists() and not only:
        shutil.rmtree(out)
    for d in ("examples", "challenges", "solutions", "assets"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for f in ASSETS.iterdir():
        shutil.copy(f, out / "assets" / f.name)
    (out / ".nojekyll").write_text("")

    def wanted(e):
        return not only or e.id in only

    for i, e in enumerate(examples):
        if wanted(e):
            print(f"  {e.id}  {e.slug}", flush=True)
            render_example(e, out, examples[i - 1] if i else None,
                           examples[i + 1] if i + 1 < len(examples) else None, repo_url)
    for i, e in enumerate(challenges):
        if wanted(e):
            print(f"  {e.id}  {e.slug}", flush=True)
            render_challenge(e, out, challenges[i - 1] if i else None,
                             challenges[i + 1] if i + 1 < len(challenges) else None, repo_url)
            if e.solution_path:
                render_solution(e, out, repo_url)

    ctx = dict(examples=examples, challenges=challenges, by_id={e.id: e for e in entries},
               series_sections=series_sections(examples), challenge_table=challenge_rows(challenges),
               card=card, repo_url=repo_url)
    for name, (title, active, body, desc) in content.pages(ctx).items():
        (out / name).write_text(shell(title=title, body=body, depth=0, active=active,
                                      description=desc, page_class=f"page-{name[:-5]}",
                                      head_extra=MATHJAX if "$" in body else ""))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}  ({size:.0f} MB)")


def default_repo_url() -> str | None:
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return f"https://github.com/{m.group(1)}" if m else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="only re-render these notebooks (e.g. E01 C03); index pages always rebuild")
    ap.add_argument("--out", default=str(ROOT / "_site"))
    ap.add_argument("--repo-url", default=None, help="GitHub repository URL for 'view notebook' links "
                                                     "(default: the origin remote, if it is on GitHub)")
    args = ap.parse_args()
    build(Path(args.out).resolve(), args.repo_url or default_repo_url(), args.ids or None)
