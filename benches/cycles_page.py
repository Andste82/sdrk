"""Writes the overview page of the kernel cycles: one self-contained HTML file.

Usage: python3 -I benches/cycles_page.py [--repo DIR] OUT.html

Reads the committed results of the repository DIR (default: the checkout of this script):
`benches/results/cycles/<platform>.json`, `benches/results/wasm32.json` and `benches/limits.json`,
and writes one HTML file with inline CSS and JavaScript and no external resource. Nothing is
uploaded and nothing is fetched. The commit is taken from git when DIR is a git checkout
("unknown" otherwise), the CI run ids from the method of the results files when they record any.

One table row per entry and level, the `_fallback` entries as rows of their own:

 1. Status: the entry of `benches/limits.json` (`release`, `dev`, `info`), marked `accuracy trade`
    (D2760) or `contract trade` (D3000); at `simd128` also the verdict `waived` or `fail`.
 2. Kernel: the entry name without `_fallback`; an entry that is the same code of another entry
    measured against a second opponent (`<entry>_<suffix>`, see `second_opponents`) is shown under
    the name of that entry. The full entry name is the tooltip.
 3. Level: `sse2` ... `neon`, `simd128`; a fallback row `fallback (<level>)`.
 4. Cycles ours: cycles per sample on the reference model / the counter-model of the level
    (`simd128`: instructions per sample, marked `instr`).
 5. Cycles opponent: the same for the comparison, on the same two models.
 6. / 7. SNR ours / opponent in dB against the f64 reference (`exact` for an exact result).
 8. Ratio: ours / opponent, per model.
 9. SNR delta: ours minus opponent in dB.
10. Opponent: `compared_with` of the record; our own code is marked `own (information)`.
11. Justification: the `accuracy_trade`, `contract_trade`, `waiver`, `accuracy_waiver` and
    `cycle_waiver` texts of `benches/limits.json` for the entry and level.

Colours (judged rows only: `release` or `dev` against a library; D2767): a ratio <= 1.00 green,
<= 1.10 yellow, above 1.10 red, and light blue for a ratio above 1.10 and at most 1.20 of an entry
and level with an `accuracy_trade` or a `contract_trade`, the maintainer's decisions that keep it
`release` (D2760, D3000, D3050). A `waiver` never makes a ratio light blue: above 1.10 it names an
open item (D2779). Rows that are `info` or compare with our own code get no colour.

`benches/verify_cycles_page.py` checks a written page against the same files.
"""
import html
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle_tables as ct  # noqa: E402

FALLBACK = "_fallback"
SIMD128 = "simd128"
LEVEL_ORDER = [lvl for _, lvl in ct.LEVELS] + [SIMD128]
# The texts of benches/limits.json that justify or explain a row, in the order shown.
REASONS = [("accuracy_trade", "accuracy trade"), ("contract_trade", "contract trade"),
           ("waiver", "waiver"), ("accuracy_waiver", "accuracy waiver"), ("cycle_waiver", "cycle waiver")]
TRADES = ("accuracy_trade", "contract_trade")
RELEASE, TRADE_CAP = 1.10, 1.20
SHORT = 140


def family(kernel):
    """The family chip of a kernel name: `blocks` for blocks, else the longest known prefix."""
    if kernel.endswith("_block"):
        return "blocks"
    m = [p for p in ct.PREFIXES if kernel == p or kernel.startswith(p + "_")]
    return max(m, key=len) if m else kernel.split("_")[0]


def ours_of(level_entries, name, simd128=False):
    """Our figures of an entry on one level: (reference, counter) cycles, (instructions,) at
    simd128, None without numbers."""
    v = level_entries.get(name)
    if not v:
        return None
    if simd128:
        return (v["ours_instr_per_sample"],)
    if "reference_model" not in v:
        return None
    return (v["reference_model"]["ours"]["cycles_per_sample"], v["counter_model"]["ours"]["cycles_per_sample"])


def second_opponents(levels):
    """{entry: base} for the entries `<base>_<suffix>` that are the code of `base` measured against
    a second opponent: the suffix is not `fallback`, and ours is the same as the base's on every
    level where both have numbers (at least one). `levels` is [(entries, simd128?)]."""
    names = set()
    for ents, _ in levels:
        names |= set(ents)
    out = {}
    for name in sorted(names):
        if name.endswith(FALLBACK):
            continue
        parts = name.split("_")
        for i in range(len(parts) - 1, 0, -1):
            base = "_".join(parts[:i])
            if base not in names:
                continue
            common, same = 0, True
            for ents, simd in levels:
                a, b = ours_of(ents, name, simd), ours_of(ents, base, simd)
                if a is None or b is None:
                    continue
                common += 1
                same &= all(abs(x - y) <= 1e-9 * max(1.0, abs(y)) for x, y in zip(a, b))
            if common and same:
                out[name] = out.get(base, base)
                break
    return out


def kernel_of(name, second):
    """The kernel shown for an entry: the base of a second opponent, without `_fallback`."""
    fb = name.endswith(FALLBACK)
    simd = name[: -len(FALLBACK)] if fb else name
    simd = second.get(simd, simd)
    return simd, fb


def level_label(level, fallback):
    return f"fallback ({level})" if fallback else level


def cut(text, n=SHORT):
    """The visible head of a long text: the first clause (up to `; ` or `. `) when it is short
    enough, else the text cut at a word boundary near `n` characters. Returns (head, rest)."""
    if len(text) <= n + 20:
        return text, ""
    for sep in ("; ", ". "):
        i = text.find(sep)
        if 30 <= i <= n + 40:
            return text[: i + 1], text[i + 1:]
    i = text.rfind(" ", 0, n)
    i = n if i < 40 else i
    return text[:i], text[i:]


def ratio_class(x, judged, justified):
    """The colour class of one ratio: g, y, r, lb (justifiably slower) or "" for unjudged rows."""
    if not judged or x is None:
        return ""
    if x <= 1.0:
        return "g"
    if x <= RELEASE:
        return "y"
    return "lb" if justified and x <= TRADE_CAP else "r"


def band_of(classes):
    """The worst class of a row's ratios, for the filter and the counts."""
    rank = ["", "g", "y", "lb", "r"]
    return max(classes, key=rank.index) if classes else ""


def snr_class(ours, opp, judged):
    """SNR colour of judged rows: r more than 1 dB below the opponent, g at least 3 dB above."""
    if not judged or ours is None or opp is None:
        return ""
    if opp - ours > 1:
        return "r"
    if ours - opp >= 3:
        return "g"
    return ""


def db(x):
    return "–" if x is None else ("exact" if x >= ct.EXACT else f"{x:.1f}")


def snr_delta(ours, opp):
    if ours is None or opp is None:
        return "–", None
    if ours >= ct.EXACT and opp >= ct.EXACT:
        return "0.0", 0.0
    if ours >= ct.EXACT:
        return "ours exact", ct.EXACT
    if opp >= ct.EXACT:
        return "opponent exact", -ct.EXACT
    d = round(ours - opp, 1)
    return f"{d:+.1f}".replace("-0.0", "0.0").replace("+0.0", "0.0"), d


def reasons_of(limits, name, level):
    e = limits.get(name, {}).get(level, {})
    return [(label, e[key]) for key, label in REASONS if isinstance(e.get(key), str) and e[key]]


def is_own(kind, compared_with):
    n = compared_with.lower()
    return kind != ct.LIBRARY or n.startswith("sdrk") or n.startswith("ours")


def rows_of(docs, counts, limits):
    """The rows of the page and the notes of the entries without numbers."""
    levels = []
    for d in docs.values():
        for level, lv in d.get("levels", {}).items():
            levels.append((lv.get("entries", {}), False))
    wasm = counts["levels"][SIMD128]["kernels"] if counts else {}
    if wasm:
        levels.append((wasm, True))
    second = second_opponents(levels)
    trades = ct.trades_of(limits)
    page, notes = ct.page_rows(docs, limits)
    rows = []
    for r in page:
        rows.append(dict(level=r["level"], name=r["name"], status=r["status"], kind=r["kind"],
                         judged=r["judged"], unit="cycles", ours=r["ours"], opp=r["comparison"],
                         ratio=r["ratio"], compared_with=r["compared_with"], snr=r["snr"],
                         own_build=r["own_build"], verdict=""))
    for name, k in wasm.items():
        status = ct.status_of(limits, name, SIMD128)
        kind = ct.judged_kind(status, ct.kind_of(k, name, SIMD128), name, SIMD128)
        snr = k.get("snr_db") or {}
        rows.append(dict(level=SIMD128, name=name, status=status, kind=kind, judged=status in ct.JUDGED,
                         unit="instructions", ours=(k["ours_instr_per_sample"],),
                         opp=(k["other_instr_per_sample"],), ratio=(k["ratio"],),
                         compared_with=k["compared_with"],
                         snr=(snr.get("ours"), snr.get("comparison", snr.get("other"))),
                         own_build=None, verdict={"waived": "waived", "fail": "fail"}.get(k["verdict"], "")))
    for r in rows:
        r["kernel"], r["fallback"] = kernel_of(r["name"], second)
        r["second"] = (r["name"][: -len(FALLBACK)] if r["fallback"] else r["name"]) in second
        r["family"] = family(r["kernel"])
        r["trade"] = trades.get((r["name"], r["level"]), "")
        r["own"] = is_own(r["kind"], r["compared_with"])
        if r["judged"] and r["own"]:
            raise ct.TableError(f"`{r['name']}` at `{r['level']}` is `{r['status']}` against our own code")
        judged = r["judged"] and not r["own"]
        r["classes"] = [ratio_class(x, judged, bool(r["trade"])) for x in r["ratio"]]
        r["band"] = band_of(r["classes"])
        r["snr_class"] = snr_class(r["snr"][0], r["snr"][1], judged)
        r["reasons"] = reasons_of(limits, r["name"], r["level"])
    lv_index = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
    rows.sort(key=lambda r: (lv_index.get(r["level"], 99), ct.order(r["kernel"]), r["fallback"],
                             r["second"], r["name"]))
    notes = sorted(notes, key=lambda n: (lv_index.get(n[0], 99), n[1]))
    return rows, notes


def git(repo, *args):
    try:
        out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def run_ids(docs, counts):
    """The CI run ids that the method of the results files records (keys with `run` in the name)."""
    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if "run" in k.lower() and isinstance(v, (str, int)) and str(v) not in found:
                    found.append(str(v))
                elif isinstance(v, dict):
                    walk(v)
    for d in list(docs.values()) + ([counts] if counts else []):
        walk(d.get("method", {}))
        walk(d.get("metadata", {}))
    return found


def meta_of(repo, docs, counts):
    commit = git(repo, "rev-parse", "--short=10", "HEAD") or "unknown"
    date = git(repo, "log", "-1", "--format=%cs") or "unknown"
    dirty = git(repo, "status", "--porcelain", "--", "benches/results", "benches/limits.json")
    if commit != "unknown" and dirty:
        commit += " with local changes to the results"
    tools = sorted({d.get("method", {}).get("tools", {}).get("llvm-mca", "") for d in docs.values()} - {""})
    return {"commit": commit, "date": date, "runs": run_ids(docs, counts), "llvm_mca": tools}


def esc(s):
    return html.escape(str(s), quote=True)


def pair(values, unit):
    s = " / ".join(f"{x:.2f}" for x in values)
    return s + " instr" if unit == "instructions" else s


def ratio_text(x, unit):
    return f"{x:.2f}" if unit == "instructions" else f"{x:.3f}"


def clip(text):
    """A cell text that shows its head and opens to the full text on a click."""
    head, rest = cut(text)
    if not rest:
        return f'<span class="h">{esc(head)}</span>'
    return f'<span class="h">{esc(head)}</span><span class="m">…</span><span class="t">{esc(rest)}</span>'


def row_html(r, models):
    m = models.get(r["level"], ("instructions per sample",))
    sort_ratio = max(r["ratio"])
    marks = []
    if r["trade"]:
        marks.append(f'<span class="st tr">{esc(r["trade"])}</span>')
    if r["verdict"]:
        marks.append(f'<span class="st vd">{esc(r["verdict"])}</span>')
    title = f'entry {r["name"]}'
    if r["second"]:
        title += f' (the code of {r["kernel"]}{" fallback" if r["fallback"] else ""} against a second opponent)'
    ratios = " / ".join(f'<span class="v {c}">{ratio_text(x, r["unit"])}</span>'
                        for x, c in zip(r["ratio"], r["classes"]))
    dtext, dval = snr_delta(*r["snr"])
    opp = ('<span class="st own">own (information)</span> ' if r["own"] else "") + clip(r["compared_with"])
    if r["own_build"]:
        opp += (f'<div class="sub">ours / own avx2 build {r["own_build"][0]:.3f} / {r["own_build"][1]:.3f} '
                f'(information)</div>')
    why = "".join(f'<div class="why"><b>{esc(label)}:</b> {clip(text)}</div>' for label, text in r["reasons"])
    model_title = " / ".join(m)
    attrs = {"data-level": r["level"], "data-variant": "fallback" if r["fallback"] else "simd",
             "data-status": r["status"], "data-trade": r["trade"], "data-band": r["band"],
             "data-family": r["family"], "data-entry": r["name"], "data-kernel": r["kernel"],
             "data-judged": "1" if r["judged"] and not r["own"] else "0"}
    a = " ".join(f'{k}="{esc(v)}"' for k, v in attrs.items())
    cells = [
        f'<td class="l c-status"><span class="st">{esc(r["status"])}</span>{" " + " ".join(marks) if marks else ""}</td>',
        f'<td class="l c-kernel"><code title="{esc(title)}">{esc(r["kernel"])}</code></td>',
        f'<td class="l c-level" title="{esc(model_title)}">{esc(level_label(r["level"], r["fallback"]))}</td>',
        f'<td class="n c-ours" data-v="{r["ours"][0]!r}">{pair(r["ours"], r["unit"])}</td>',
        f'<td class="n c-opp" data-v="{r["opp"][0]!r}">{pair(r["opp"], r["unit"])}</td>',
        f'<td class="n c-snro {r["snr_class"]}" data-v="{"" if r["snr"][0] is None else repr(r["snr"][0])}">{db(r["snr"][0])}</td>',
        f'<td class="n c-snrc {r["snr_class"]}" data-v="{"" if r["snr"][1] is None else repr(r["snr"][1])}">{db(r["snr"][1])}</td>',
        f'<td class="n c-ratio" data-v="{sort_ratio!r}">{ratios}</td>',
        f'<td class="n c-delta {r["snr_class"]}" data-v="{"" if dval is None else repr(dval)}">{dtext}</td>',
        f'<td class="l c-opponent x">{opp}</td>',
        f'<td class="l c-why x">{why}</td>',
    ]
    return f"<tr {a}>" + "".join(cells) + "</tr>"


def models_of(docs):
    out = {}
    for d in docs.values():
        for level, lv in d.get("levels", {}).items():
            out[level] = (lv["reference_model"], lv["counter_model"])
    return out


def page(repo):
    docs = ct.load(os.path.join(repo, "benches", "results", "cycles"))
    wasm_path = os.path.join(repo, "benches", "results", "wasm32.json")
    counts = None
    if os.path.exists(wasm_path):
        with open(wasm_path) as fh:
            counts = json.load(fh)
    with open(os.path.join(repo, "benches", "limits.json")) as fh:
        limits = json.load(fh)
    rows, notes = rows_of(docs, counts, limits)
    models = models_of(docs)
    meta = meta_of(repo, docs, counts)
    levels = [lvl for lvl in LEVEL_ORDER if any(r["level"] == lvl for r in rows)]
    levels += sorted({r["level"] for r in rows} - set(levels))
    fams = sorted({r["family"] for r in rows}, key=lambda f: ct.order(f))
    body = "\n".join(row_html(r, models) for r in rows)
    note_html = "\n".join(f"<li data-level=\"{esc(lvl)}\"><code>{esc(lvl)}</code> <code>{esc(n)}</code>: {esc(t)}</li>"
                          for lvl, n, t in notes) or "<li>none</li>"
    runs = ", ".join(meta["runs"]) if meta["runs"] else "not recorded in the results files"
    js = json.dumps({"levels": levels, "families": fams,
                     "models": {k: list(v) for k, v in models.items()}}, separators=(",", ":")).replace("</", "<\\/")
    out = TEMPLATE
    for key, value in {"@@COMMIT@@": esc(meta["commit"]), "@@DATE@@": esc(meta["date"]),
                       "@@RUNS@@": esc(runs), "@@MCA@@": esc("; ".join(meta["llvm_mca"]) or "unknown"),
                       "@@ROWS@@": body, "@@NOTES@@": note_html, "@@DATA@@": js}.items():
        out = out.replace(key, value)
    return out, rows, notes


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    repo, out = os.path.dirname(HERE), None
    while args:
        a = args.pop(0)
        if a == "--repo" and args:
            repo = args.pop(0)
        elif out is None and not a.startswith("--"):
            out = a
        else:
            sys.exit(f"unknown argument {a!r}; usage: cycles_page.py [--repo DIR] OUT.html")
    if out is None:
        sys.exit("usage: cycles_page.py [--repo DIR] OUT.html")
    try:
        text, rows, notes = page(repo)
    except ct.TableError as e:
        sys.exit(f"cycles page: {e}")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    blue = sum(1 for r in rows if r["band"] == "lb")
    print(f"{out}: {len(rows)} rows, {len(notes)} notes, {blue} light blue", file=sys.stderr)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDRk kernel cycles</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#e3e2dd;--card:#fff;--g:#d8f0dc;--gf:#1f6b2e;--y:#fbefc5;--yf:#7a5a00;--r:#f8d7d4;--rf:#9b1c13;--lb:#d6ecfb;--lbf:#0b4f7c;--acc:#2f5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--card:#1f1f1d;--g:#1d3a24;--gf:#9fe0ad;--y:#3d3315;--yf:#f1d27a;--r:#45201d;--rf:#f4a59d;--lb:#15344a;--lbf:#9fd4f7;--acc:#8fb8e0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header,main,footer{padding:12px 16px}
h1{font-size:22px;margin:4px 0}
h2{font-size:15px;margin:18px 0 6px}
.meta{color:var(--mut);font-size:13px}
.explain{font-size:13px;color:var(--mut);max-width:1400px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;margin:10px 0}
.controls label{display:flex;flex-direction:column;font-size:12px;color:var(--mut);gap:3px}
select,input{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer}
.chip[aria-pressed=true]{background:var(--acc);border-color:var(--acc);color:#fff}
.summary{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;min-width:110px}
.tile b{display:block;font-size:19px;font-variant-numeric:tabular-nums}
.tile span{font-size:12px;color:var(--mut)}
.wrap{overflow:auto;max-height:calc(100vh - 40px);border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;vertical-align:top}
th{position:sticky;top:0;z-index:1;background:var(--card);font-size:12px;color:var(--mut);font-weight:600;cursor:pointer;user-select:none;box-shadow:0 1px 0 var(--line)}
th.l,td.l{text-align:left}
td.n{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
td.x{white-space:normal;font-size:12px;color:var(--mut);cursor:pointer}
td.c-opponent{min-width:220px;max-width:30vw}
td.c-why{min-width:260px;max-width:36vw}
td.x .t{display:none}td.x.open .t{display:inline}td.x.open .m{display:none}
.m{color:var(--acc)}
.why+.why{margin-top:3px}
.why b{color:var(--fg);font-weight:600}
.sub{font-size:11px;color:var(--mut)}
th .sub{font-weight:400}
.g{background:var(--g);color:var(--gf)}.y{background:var(--y);color:var(--yf)}.r{background:var(--r);color:var(--rf)}.lb{background:var(--lb);color:var(--lbf)}
.v{padding:0 3px;border-radius:3px}
.st{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--mut)}
.st.tr{border-color:var(--acc);color:var(--acc)}
.st.vd{border-color:var(--rf);color:var(--rf)}
.legend{display:flex;gap:12px;font-size:12px;color:var(--mut);align-items:center;flex-wrap:wrap}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px;border:1px solid var(--line)}
footer{color:var(--mut);font-size:12px}
</style>
</head>
<body>
<header>
<h1>SDRk kernel cycles</h1>
<div class="meta">Commit <code>@@COMMIT@@</code> of @@DATE@@ · CI runs: @@RUNS@@ · llvm-mca: @@MCA@@</div>
<p class="explain">Simulated cycles per sample (llvm-mca whole-loop on the execution counts of the counted calls, the corrected llvm-mca of D942) of our entry and of its opponent, on the reference model / the counter-model of the level: the two numbers of a cell are the two models, and ours and the opponent run on the same models, so the cells compare model with model. <code>simd128</code> has no core model: its numbers are instructions per sample (marked <code>instr</code>), its ratio the one of <code>benches/results/wasm32.json</code>. A fallback entry is a row of its own, <code>fallback (&lt;level&gt;)</code>. An entry that is the same code measured against a second opponent is listed under the kernel it measures (the full entry name is the tooltip of the kernel). The status is the entry of <code>benches/limits.json</code>: <code>release</code> at most 1.10 times the opponent on both models, <code>dev</code> at most 1.50, <code>info</code> not judged. An <span class="st tr">accuracy trade</span> keeps a <code>release</code> entry up to 1.20 when it is at least 3 dB more accurate than a library below 100 dB (D2760); a <span class="st tr">contract trade</span> keeps it up to 1.20 when the library provably does less work than our contract (D3000). A comparison with our own code is marked <span class="st own">own (information)</span> and never decides a status (D2764, D2767). SNR in dB against the f64 reference; SNR Δ is ours minus the opponent. The justification is the text of <code>benches/limits.json</code> for the entry and level (click a cell for the full text).</p>
<div class="legend"><span>Ratio, judged rows only (<code>release</code>, <code>dev</code> against a library):</span><span><i class="g"></i>&le; 1.00</span><span><i class="y"></i>&le; 1.10</span><span><i class="lb"></i>justifiably slower: above 1.10 and at most 1.20 under an accuracy or contract trade (D3050)</span><span><i class="r"></i>above 1.10 otherwise (a waiver names an open item, D2779)</span><span>SNR: <i class="r"></i>more than 1 dB below the opponent, <i class="g"></i>at least 3 dB above</span></div>
</header>
<main>
<div class="controls">
<label>Level<select id="level"></select></label>
<label>Variant<select id="variant"><option value="">all</option><option value="simd">SIMD entry</option><option value="fallback">fallback</option></select></label>
<label>Status<select id="status"><option value="">all</option><option>release</option><option>dev</option><option>info</option><option value="t:accuracy trade">accuracy trade</option><option value="t:contract trade">contract trade</option></select></label>
<label>Ratio (judged rows)<select id="band"><option value="">all</option><option value="g">&le; 1.00 on every model</option><option value="y">worst in (1.00, 1.10]</option><option value="lb">justifiably slower</option><option value="r">above 1.10, not justified</option></select></label>
<label>Name<input id="q" type="search" placeholder="kernel or entry"></label>
</div>
<div class="chips" id="fams" aria-label="Family"></div>
<div class="summary" id="summary"></div>
<div class="wrap"><table id="t">
<thead><tr>
<th class="l" data-c="0">Status</th>
<th class="l" data-c="1">Kernel</th>
<th class="l" data-c="2">Level</th>
<th data-c="3" data-num="1">Cycles ours (main / counter)<br><span class="sub m-models">all levels</span></th>
<th data-c="4" data-num="1">Cycles opponent, same models<br><span class="sub m-models">all levels</span></th>
<th data-c="5" data-num="1">SNR ours<br><span class="sub">dB</span></th>
<th data-c="6" data-num="1">SNR opponent<br><span class="sub">dB</span></th>
<th data-c="7" data-num="1">Ratio cycles<br><span class="sub">ours / opponent, per model</span></th>
<th data-c="8" data-num="1">SNR Δ (dB)<br><span class="sub">ours − opponent</span></th>
<th class="l" data-c="9">Opponent</th>
<th class="l" data-c="10">Justification</th>
</tr></thead>
<tbody>
@@ROWS@@
</tbody></table></div>
<h2>Entries without numbers</h2>
<ul id="notes" class="explain">
@@NOTES@@
</ul>
</main>
<footer>Generated by <code>python3 -I benches/cycles_page.py</code> from <code>benches/results/cycles</code>, <code>benches/results/wasm32.json</code> and <code>benches/limits.json</code> of the commit above; checked by <code>benches/verify_cycles_page.py</code>. Cycles are simulated, never wall-clock timings.</footer>
<script>
const D=@@DATA@@;
const el=id=>document.getElementById(id);
const body=document.querySelector("#t tbody");
const all=Array.from(body.rows);
const S={level:"",variant:"",status:"",band:"",q:"",fams:new Set(),sort:null,dir:1};
function init(){
  const ls=el("level");
  ls.innerHTML='<option value="">all levels</option>'+D.levels.map(l=>`<option>${l}</option>`).join("");
  let saved=null;try{saved=localStorage.getItem("sdrk-cycles-level")}catch(e){}
  if(saved&&D.levels.includes(saved)){S.level=saved;ls.value=saved}
  ls.onchange=()=>{S.level=ls.value;try{localStorage.setItem("sdrk-cycles-level",S.level)}catch(e){}render()};
  for(const k of ["variant","status","band"])el(k).onchange=e=>{S[k]=e.target.value;render()};
  el("q").oninput=e=>{S.q=e.target.value.trim().toLowerCase();render()};
  const fs=el("fams");
  fs.innerHTML=["all"].concat(D.families).map(f=>`<button class="chip" data-f="${f}" aria-pressed="${f==="all"}">${f}</button>`).join("");
  fs.onclick=e=>{const b=e.target.closest("button");if(!b)return;const f=b.dataset.f;
    if(f==="all")S.fams.clear();else{S.fams.has(f)?S.fams.delete(f):S.fams.add(f)}
    fs.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",x.dataset.f==="all"?S.fams.size===0:S.fams.has(x.dataset.f)));render()};
  document.querySelector("#t thead").onclick=e=>{const th=e.target.closest("th");if(!th)return;const c=+th.dataset.c;
    if(S.sort===c)S.dir=-S.dir;else{S.sort=c;S.dir=1}render()};
  body.onclick=e=>{const td=e.target.closest("td.x");if(td)td.classList.toggle("open")};
  render();
}
function keep(r){const d=r.dataset;
  return (!S.level||d.level===S.level)&&(!S.variant||d.variant===S.variant)&&
    (!S.status||(S.status.startsWith("t:")?d.trade===S.status.slice(2):d.status===S.status))&&
    (!S.band||d.band===S.band)&&(S.fams.size===0||S.fams.has(d.family))&&
    (!S.q||d.kernel.includes(S.q)||d.entry.includes(S.q))}
function render(){
  const m=S.level&&D.models[S.level];
  document.querySelectorAll(".m-models").forEach(x=>x.textContent=m?m.join(" / "):(S.level==="simd128"?"instructions per sample":"all levels"));
  document.querySelectorAll("#t th").forEach(th=>{th.dataset.arrow=S.sort===+th.dataset.c?(S.dir>0?"▲":"▼"):"";
    th.querySelector(".arrow")?.remove();if(th.dataset.arrow)th.insertAdjacentHTML("beforeend",`<span class="arrow"> ${th.dataset.arrow}</span>`)});
  let rows=all.filter(keep);
  if(S.sort!==null){const c=S.sort,num=document.querySelector(`#t th[data-c="${c}"]`).dataset.num;
    rows=rows.slice().sort((a,b)=>{const v=t=>t.dataset.v??t.textContent.trim();let x=v(a.cells[c]),y=v(b.cells[c]);
      if(x==="")return 1;if(y==="")return -1;if(num){x=+x;y=+y}return (x<y?-1:x>y?1:0)*S.dir})}
  const shown=new Set(rows);
  all.forEach(r=>{r.hidden=!shown.has(r)});
  rows.forEach(r=>body.appendChild(r));
  const judged=rows.filter(r=>r.dataset.judged==="1"),n=k=>judged.filter(r=>r.dataset.band===k).length;
  el("summary").innerHTML=[[rows.length,"rows shown"],[judged.length,"judged (release, dev)"],[n("g"),"at ≤ 1.00"],[n("y"),"in (1.00, 1.10]"],[n("lb"),"justifiably slower"],[n("r"),"above 1.10"],
    [rows.filter(r=>r.dataset.trade==="accuracy trade").length,"accuracy trades"],[rows.filter(r=>r.dataset.trade==="contract trade").length,"contract trades"]]
    .map(([v,t])=>`<div class="tile"><b>${v}</b><span>${t}</span></div>`).join("");
  document.querySelectorAll("#notes li[data-level]").forEach(li=>{li.hidden=!!S.level&&li.dataset.level!==S.level});
}
init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
