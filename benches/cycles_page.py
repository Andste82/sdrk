"""Writes the overview page of the kernel cycles: one self-contained HTML file.

Usage: python3 -I benches/cycles_page.py [--repo DIR] OUT.html

Reads the committed results of the repository DIR (default: the checkout of this script):
`benches/results/cycles/<platform>.json`, `benches/results/wasm32.json`, `benches/limits.json`, and
the VOLK dispatch tables of `xtask/src/bench.rs` (which protokernel VOLK runs at a level), and
writes one HTML file with inline CSS and JavaScript and no external resource. Nothing is uploaded
and nothing is fetched. The header names the commit (from git, when DIR is a checkout) and the
tools that the results files record.

One block per kernel, sorted by name. A block has a header row with the kernel name and under it
the levels in a fixed order (`sse2`, `sse4.2`, `avx2`, `avx2-runtime`, `avx512`, `neon`,
`simd128`), each followed by an indented row `↳ fallback` for the `_fallback` entry of that level.
An entry `<entry>_<suffix>` that is the code of `<entry>` measured against a second opponent (see
`second_opponents`) is a further row of its level in the block of `<entry>`.

Columns: Level | Status | Cycles ours | Cycles opponent (same CPUs) | Ratio | SNR delta | Opponent |
Justification. A cell `a / b` holds the reference model and the counter-model of the level (the
legend maps the levels to their CPUs); `simd128` has instructions per sample and one ratio.

- Status: the entry of `benches/limits.json` (`release`, `dev`, `info`), marked `accuracy trade`
  (D2760) or `contract trade` (D3000); at `simd128` also the verdict `waived` or `fail`.
- Ratio: ours / opponent per model; its tooltip gives the SIMD entry against its own fallback of
  the level (information; `avx2-runtime`: against our own `avx2` build).
- SNR delta: ours minus the opponent in dB; the tooltip gives both SNR values.
- Opponent: the library and function of `compared_with`, with the VOLK protokernel that runs at the
  level or the build level of the reference; the full text is the tooltip. Our own code is marked
  `own (information)`.
- Justification: the `accuracy_trade`, `contract_trade`, `waiver`, `accuracy_waiver` and
  `cycle_waiver` texts of `benches/limits.json` for the entry and level, the head shown, the full
  text on a click.

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
import re
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
RELEASE, TRADE_CAP, LITTLE = 1.10, 1.20, 0.80
SHORT = 140
NOT_PLANNED = "not simulated: not measured at"
# The names of the core models of llvm-mca as the legend shows them.
CPU_NAMES = {"skylake": "Skylake", "znver2": "Zen 2", "znver3": "Zen 3", "znver4": "Zen 4",
             "alderlake": "Alder Lake", "sapphirerapids": "Sapphire Rapids",
             "cortex-a72": "Cortex-A72", "cortex-a76": "Cortex-A76"}
# Build levels as the references name them in `compared_with`.
BUILD_LEVELS = {"sse2": {"sse2"}, "sse4.2": {"sse4.2", "sse42", "sse4"}, "avx2": {"avx2"},
                "avx2-runtime": {"avx2"}, "avx512": {"avx512"}, "neon": {"neon", "neon64"},
                "simd128": {"simd128"}}


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
    """The kernel of an entry (the base of a second opponent, without `_fallback`) and whether it
    is a fallback."""
    fb = name.endswith(FALLBACK)
    simd = name[: -len(FALLBACK)] if fb else name
    return second.get(simd, simd), fb


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


BIT_EXACT = "bit-exact: output identical to the f64 reference, no error"


def db(x):
    """An SNR for the tooltip: `∞` for a bit-exact result (999 in the records), else dB."""
    return "–" if x is None else ("∞" if x >= ct.EXACT else f"{x:.1f} dB")


def snr_delta(ours, opp):
    """The SNR difference: (text, sort value, tooltip). Bit-exact is infinite: `+∞` when only ours
    is bit-exact, `−∞` when only the opponent is, `0` when both are."""
    if ours is None or opp is None:
        return "–", None, "no SNR"
    if ours >= ct.EXACT and opp >= ct.EXACT:
        return "0", 0.0, f"both bit-exact ({BIT_EXACT})"
    if ours >= ct.EXACT:
        return "+∞", "Infinity", f"SNR ours ∞ ({BIT_EXACT}) / opponent {db(opp)}"
    if opp >= ct.EXACT:
        return "−∞", "-Infinity", f"SNR ours {db(ours)} / opponent ∞ ({BIT_EXACT})"
    d = round(ours - opp, 1)
    return ("0.0" if d == 0 else f"{d:+.1f}"), d, f"SNR ours {db(ours)} / opponent {db(opp)}"


def reasons_of(limits, name, level):
    e = limits.get(name, {}).get(level, {})
    return [(label, e[key]) for key, label in REASONS if isinstance(e.get(key), str) and e[key]]


def is_own(kind, compared_with):
    n = compared_with.lower()
    return kind != ct.LIBRARY or n.startswith("sdrk") or n.startswith("ours")


# VOLK: which protokernel runs at a level.

def volk_tables(repo):
    """{level: {function: (aligned, unaligned)}} and the dispatch names of VOLK, from the tables of
    `xtask/src/bench.rs` (the pinned `volk_config` of each level, VOLK's default dispatch at
    `avx512`, only `generic` at `simd128`); empty when the file is missing."""
    path = os.path.join(repo, "xtask", "src", "bench.rs")
    if not os.path.exists(path):
        return {}, []
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    consts = {}
    for name, body in re.findall(r'const (VOLK_[A-Z0-9_]+): &str = "\\\n(.*?)";', src, re.S):
        consts[name] = {f: (a, u) for f, a, u in re.findall(r"^(\S+) (\S+) (\S+)$", body, re.M)}
    pins = dict(re.findall(r"Self::(\w+) => (VOLK_[A-Z0-9_]+),", src))
    m = re.search(r"VolkPin::Default => (VOLK_[A-Z0-9_]+),", src)
    default = m.group(1) if m else None
    m = re.search(r"pub const VOLK_FUNCTIONS: \[&str; \d+\] = \[(.*?)\];", src, re.S)
    functions = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    out = {}
    for level, refs, pin in re.findall(r'level: "([^"]+)",.*?refs: "([^"]+)",\s*volk_pin: VolkPin::(\w+),', src, re.S):
        if pin == "Default" and refs == "wasm32":
            out[level] = {f: ("generic", "generic") for f in functions}
        else:
            out[level] = consts.get(pins.get(pin) if pin != "Default" else default, {})
    return out, functions


def volk_calls(compared_with, functions):
    """The VOLK dispatch calls named in `compared_with`: [(function, aligned, as written)]; an
    explicit protokernel (`_u_avx`, `_generic`) is no dispatch."""
    out = []
    for f in functions:
        for m in re.finditer(re.escape(f) + r"(\w*)", compared_with):
            if m.group(1) in ("", "_a", "_u"):
                out.append((f, m.group(1) != "_u", m.group(0)))
    return sorted(set(out))


def strip_parens(s):
    """`s` without its parenthesised remarks (a group after a space); a call such as `real(x)`
    stays."""
    out, depth, keep = "", 0, []
    for ch in s:
        if ch == "(":
            keep.append(bool(keep[-1]) if keep else out[-1:].isalnum())
            depth += 1
            if keep[-1]:
                out += ch
        elif ch == ")" and depth:
            depth -= 1
            if keep.pop():
                out += ch
        elif depth == 0 or keep[-1]:
            out += ch
    return re.sub(r"\s+([,;:])", r"\1", re.sub(r"\s+", " ", out)).strip()


def first_clause(s):
    """`s` up to the first `, `, `; ` or `: ` outside brackets."""
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([<":
            depth += 1
        elif ch in ")]>" and depth:
            depth -= 1
        elif depth == 0 and ch in ",;:" and s[i + 1:i + 2] == " ":
            return s[:i]
    return s


def build_level(compared_with, level):
    """The build level of the reference at `level` as its parentheses name it (`avx2`, `neon64`,
    `sse42`), or "": a token of another level (a note on `simd128` in the text of every level) is
    no build level of this one."""
    names = BUILD_LEVELS.get(level, set())
    for group in re.findall(r"\(([^()]*)\)", compared_with):
        for tok in re.split(r"[,\s]+", group):
            if tok in names:
                return tok
    return ""


def short_opponent(compared_with, level, volk, functions):
    """The opponent as the column shows it: the library and function, cut at the first `, `, `; `
    or `: ` outside parentheses, with the protokernel of every VOLK dispatch call at the level, or
    else the build level of the reference."""
    s = first_clause(strip_parens(compared_with)).strip()
    calls = volk_calls(compared_with, functions)
    table = volk.get(level, {})
    marked = False
    for f, aligned, written in calls:
        pk = table.get(f)
        if pk and written in s:
            s = s.replace(written, f"{written} ({pk[0] if aligned else pk[1]})", 1)
            marked = True
    if s.startswith("volk_"):
        s = "VOLK " + s
    if not marked:
        b = build_level(compared_with, level)
        if b and f"({b})" not in s:
            s += f" ({b})"
    if len(s) > 110:
        i = s.rfind(" ", 0, 100)
        s = s[: i if i > 40 else 100] + " …"
    return s


def rows_of(docs, counts, limits, volk=({}, [])):
    """The rows of the page in the block order and the notes of the real gaps."""
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
    by_key = {(r["level"], r["name"]): r for r in rows}
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
        r["opponent"] = short_opponent(r["compared_with"], r["level"], *volk)
        r["vs_fallback"], r["little"] = None, False
    for r in rows:
        fb = None if r["fallback"] else by_key.get((r["level"], r["name"] + FALLBACK))
        if fb is not None:
            r["vs_fallback"] = tuple(a / b for a, b in zip(r["ours"], fb["ours"]))
            if any(round(x, 2) > LITTLE for x in r["vs_fallback"]):
                r["little"] = fb["little"] = True
        # The SIMD row of the same level that a fallback row hangs under in the block view.
        r["child"] = r["fallback"] and (r["level"], r["name"][: -len(FALLBACK)]) in by_key
    lv_index = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
    rows.sort(key=lambda r: (r["kernel"], lv_index.get(r["level"], 99), r["level"], r["fallback"],
                             r["second"], r["name"]))
    real = [n for n in notes if not n[2].startswith(NOT_PLANNED)]
    planned = len(notes) - len(real)
    real.sort(key=lambda n: (lv_index.get(n[0], 99), n[1]))
    return rows, real, planned


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
                if re.search(r"(^|_)run(_|s?$)", k.lower()) and isinstance(v, (str, int)) and str(v) not in found:
                    found.append(str(v))
                elif isinstance(v, dict):
                    walk(v)
    for d in list(docs.values()) + ([counts] if counts else []):
        walk(d.get("method", {}))
        walk(d.get("metadata", {}))
    return found


def header_of(repo, docs, counts):
    """The parts of the header line: the commit and what the results files record."""
    parts = []
    commit = git(repo, "rev-parse", "--short=10", "HEAD")
    if commit:
        date = git(repo, "log", "-1", "--format=%cs")
        dirty = git(repo, "status", "--porcelain", "--", "benches/results", "benches/limits.json")
        parts.append(f"Commit <code>{esc(commit)}</code>" + (f" of {esc(date)}" if date else "")
                     + (" with local changes to the results" if dirty else ""))
    runs = run_ids(docs, counts)
    if runs:
        parts.append("CI runs " + ", ".join(esc(r) for r in runs))
    mca = sorted({d.get("tools", {}).get("llvm-mca", "") for d in docs.values()} - {""})
    if mca:
        parts.append("llvm-mca: " + esc("; ".join(mca)))
    rustc = sorted({d.get("tools", {}).get("rustc", "") for d in docs.values()} - {""})
    if rustc:
        parts.append(esc("; ".join(rustc)))
    if counts and counts.get("levels", {}).get(SIMD128, {}).get("counter"):
        parts.append(f"simd128: {esc(counts['levels'][SIMD128]['counter'])}")
    return " · ".join(parts)


def esc(s):
    return html.escape(str(s), quote=True)


def pair(values):
    return " / ".join(f"{x:.2f}" for x in values)


def ratio_text(x, unit):
    return f"{x:.2f}" if unit == "instructions" else f"{x:.3f}"


def clip(text):
    """A cell text that shows its head and opens to the full text on a click."""
    head, rest = cut(text)
    if not rest:
        return f'<span class="h">{esc(head)}</span>'
    return f'<span class="h">{esc(head)}</span><span class="m">…</span><span class="t">{esc(rest)}</span>'


def cpus(level, models):
    m = models.get(level)
    return " / ".join(CPU_NAMES.get(x, x) for x in m) if m else "instructions per sample, no CPU model"


def row_html(r, models):
    marks = [f'<span class="st">{esc(r["status"])}</span>']
    if r["trade"]:
        marks.append(f'<span class="st tr">{esc(r["trade"])}</span>')
    if r["verdict"]:
        marks.append(f'<span class="st vd">{esc(r["verdict"])}</span>')
    label = (f'↳ fallback<span class="lv"> ({esc(r["level"])})</span>' if r["fallback"] else esc(r["level"]))
    level_title = f'entry {r["name"]} at {r["level"]}: {cpus(r["level"], models)}'
    ratios = " / ".join(f'<span class="v{" t-" + c if c else ""}">{ratio_text(x, r["unit"])}</span>'
                        for x, c in zip(r["ratio"], r["classes"]))
    info = ""
    if r["vs_fallback"]:
        info = "SIMD / own fallback " + " / ".join(f"{x:.2f}" for x in r["vs_fallback"]) + " (information)"
    elif r["own_build"]:
        info = "ours / own avx2 build " + " / ".join(f"{x:.2f}" for x in r["own_build"]) + " (information)"
    info_attr = f' title="{esc(info)}"' if info else ""
    dtext, dval, snr_title = snr_delta(*r["snr"])
    opp = ('<span class="st own">own (information)</span> ' if r["own"] else "") + esc(r["opponent"])
    why = "".join(f'<div class="why"><b>{esc(label_)}:</b> {clip(text)}</div>' for label_, text in r["reasons"])
    attrs = {"data-level": r["level"], "data-variant": "fallback" if r["fallback"] else "simd",
             "data-status": r["status"], "data-trade": r["trade"], "data-band": r["band"],
             "data-family": r["family"], "data-entry": r["name"], "data-kernel": r["kernel"],
             "data-judged": "1" if r["judged"] and not r["own"] else "0",
             "data-little": "1" if r["little"] else "0"}
    cls = "row" + (" child" if r["child"] else "") + (" fb" if r["fallback"] else "")
    a = " ".join(f'{k}="{esc(v)}"' for k, v in attrs.items())
    cells = [
        f'<td class="l c-level" title="{esc(level_title)}"><span class="kn">{esc(r["kernel"])} </span>{label}</td>',
        f'<td class="l c-status">{" ".join(marks)}</td>',
        f'<td class="n c-ours" data-v="{r["ours"][0]!r}">{pair(r["ours"])}</td>',
        f'<td class="n c-opp" data-v="{r["opp"][0]!r}">{pair(r["opp"])}</td>',
        f'<td class="n c-ratio{" " + r["band"] if r["band"] else ""}" data-v="{max(r["ratio"])!r}"{info_attr}>{ratios}</td>',
        f'<td class="n c-delta{" " + r["snr_class"] if r["snr_class"] else ""}" data-v="{"" if dval is None else dval if isinstance(dval, str) else repr(dval)}" title="{esc(snr_title)}">{dtext}</td>',
        f'<td class="l c-opponent" title="{esc(r["compared_with"])}">{opp}</td>',
        f'<td class="l c-why x">{why}</td>',
    ]
    return f'<tr class="{cls}" {a}>' + "".join(cells) + "</tr>"


def models_of(docs):
    out = {}
    for d in docs.values():
        for level, lv in d.get("levels", {}).items():
            out[level] = (lv["reference_model"], lv["counter_model"])
    return out


def body_html(rows, models):
    out, kernel = [], None
    for r in rows:
        if r["kernel"] != kernel:
            kernel = r["kernel"]
            out.append(f'<tr class="kh" data-kernel="{esc(kernel)}"><th colspan="8" class="l">'
                       f'<code>{esc(kernel)}</code></th></tr>')
        out.append(row_html(r, models))
    return "\n".join(out)


def legend_cpus(models):
    seen, parts = {}, []
    for level in LEVEL_ORDER:
        if level in models:
            seen.setdefault(cpus(level, models), []).append(level)
    for text, lvls in seen.items():
        parts.append(f'<span><code>{" / ".join(esc(x) for x in lvls)}</code>: {esc(text)}</span>')
    parts.append("<span><code>simd128</code>: instructions per sample, no CPU model</span>")
    return "".join(parts)


def page(repo):
    docs = ct.load(os.path.join(repo, "benches", "results", "cycles"))
    wasm_path = os.path.join(repo, "benches", "results", "wasm32.json")
    counts = None
    if os.path.exists(wasm_path):
        with open(wasm_path) as fh:
            counts = json.load(fh)
    with open(os.path.join(repo, "benches", "limits.json")) as fh:
        limits = json.load(fh)
    rows, notes, planned = rows_of(docs, counts, limits, volk_tables(repo))
    models = models_of(docs)
    levels = [lvl for lvl in LEVEL_ORDER if any(r["level"] == lvl for r in rows)]
    levels += sorted({r["level"] for r in rows} - set(levels))
    fams = sorted({r["family"] for r in rows})
    note_html = "\n".join(f'<li data-level="{esc(lvl)}"><code>{esc(lvl)}</code> <code>{esc(n)}</code>: {esc(t)}</li>'
                          for lvl, n, t in notes) or "<li>none</li>"
    js = json.dumps({"levels": levels, "families": fams,
                     "cpus": {k: [CPU_NAMES.get(x, x) for x in v] for k, v in models.items()}},
                    separators=(",", ":")).replace("</", "<\\/")
    out = TEMPLATE
    for key, value in {"@@HEADER@@": header_of(repo, docs, counts), "@@LEGEND@@": legend_cpus(models),
                       "@@ROWS@@": body_html(rows, models), "@@NOTES@@": note_html,
                       "@@PLANNED@@": str(planned), "@@DATA@@": js}.items():
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
    kernels = len({r["kernel"] for r in rows})
    print(f"{out}: {kernels} kernels, {len(rows)} rows, {len(notes)} gaps, {blue} light blue", file=sys.stderr)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDRk kernel cycles</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#e3e2dd;--card:#fff;--head:#f1f0ec;--g:#d8f0dc;--gf:#1f6b2e;--y:#fbefc5;--yf:#7a5a00;--r:#f8d7d4;--rf:#9b1c13;--lb:#d6ecfb;--lbf:#0b4f7c;--acc:#2f5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--card:#1f1f1d;--head:#2a2a27;--g:#1d3a24;--gf:#9fe0ad;--y:#3d3315;--yf:#f1d27a;--r:#45201d;--rf:#f4a59d;--lb:#15344a;--lbf:#9fd4f7;--acc:#8fb8e0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header,main,footer{padding:10px 16px}
h1{font-size:22px;margin:4px 0}
h2{font-size:15px;margin:18px 0 6px}
.meta,.explain{color:var(--mut);font-size:13px}
.explain{max-width:1400px;margin:6px 0}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;margin:10px 0}
.controls label{display:flex;flex-direction:column;font-size:12px;color:var(--mut);gap:3px}
.controls label.cb{flex-direction:row;align-items:center;gap:6px;font-size:13px;color:var(--fg)}
select,input,button{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button{cursor:pointer}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border-radius:999px;padding:3px 10px;font-size:12px}
.chip[aria-pressed=true]{background:var(--acc);border-color:var(--acc);color:#fff}
.summary{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;min-width:110px}
.tile b{display:block;font-size:19px;font-variant-numeric:tabular-nums}
.tile span{font-size:12px;color:var(--mut)}
.wrap{overflow:auto;max-height:calc(100vh - 30px);border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:3px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;vertical-align:top}
thead th{position:sticky;top:0;z-index:2;background:var(--card);font-size:12px;color:var(--mut);font-weight:600;cursor:pointer;user-select:none;box-shadow:0 1px 0 var(--line)}
th.l,td.l{text-align:left}
tr.kh th{background:var(--head);font-size:13.5px;padding-top:6px;color:var(--fg);font-weight:600}
td.n{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
tr.fb td.c-level{padding-left:22px;color:var(--mut)}
tr.child .lv{display:none}
.kn{display:none}
.flat .kn{display:inline;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.flat tr.child .lv{display:inline}
.flat tr.kh{display:none}
td.c-opponent{white-space:normal;min-width:200px;max-width:28vw;font-size:12.5px}
td.x{white-space:normal;font-size:12px;color:var(--mut);cursor:pointer;min-width:240px;max-width:40vw}
td.x .t{display:none}td.x.open .t{display:inline}td.x.open .m{display:none}
.m{color:var(--acc)}
.why+.why{margin-top:3px}
.why b{color:var(--fg);font-weight:600}
th .sub{font-weight:400;font-size:11px}
.g{background:var(--g);color:var(--gf)}.y{background:var(--y);color:var(--yf)}.r{background:var(--r);color:var(--rf)}.lb{background:var(--lb);color:var(--lbf)}
.t-g{color:var(--gf)}.t-y{color:var(--yf)}.t-r{color:var(--rf)}.t-lb{color:var(--lbf)}
td.n .v{font-weight:600}
.st{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--mut)}
.st.tr{border-color:var(--acc);color:var(--acc)}
.st.vd{border-color:var(--rf);color:var(--rf)}
.legend{display:flex;gap:14px;font-size:12px;color:var(--mut);align-items:center;flex-wrap:wrap;margin:4px 0}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px;border:1px solid var(--line)}
footer{color:var(--mut);font-size:12px}
</style>
</head>
<body>
<header>
<h1>SDRk kernel cycles</h1>
<div class="meta">@@HEADER@@</div>
<p class="explain">Simulated cycles per sample of our entry and of its opponent (llvm-mca whole-loop on the execution counts of the counted calls). A cell <code>a / b</code> holds the two CPU models of the level, reference model / counter-model; ours and the opponent run on the same models. The ratio is ours / opponent, the SNR Δ ours minus the opponent in dB against the f64 reference (both values in the tooltip; <code>+∞</code> when only ours is bit-exact, <code>−∞</code> when only the opponent is, <code>0</code> when both are). The status is the entry of <code>benches/limits.json</code>: <code>release</code> at most 1.10 times the opponent on both models, <code>dev</code> at most 1.50, <code>info</code> not judged; an <span class="st tr">accuracy trade</span> (at least 3 dB more accurate than a library below 100 dB, D2760) or a <span class="st tr">contract trade</span> (the library provably does less work, D3000) keeps <code>release</code> up to 1.20. A comparison with our own code is marked <span class="st own">own (information)</span> and decides no status (D2767). The tooltip of the ratio of a SIMD row gives it against our own fallback of the level (information). Click a justification for its full text.</p>
<div class="legend"><span>CPUs:</span>@@LEGEND@@</div>
<div class="legend"><span>Ratio, judged rows only (<code>release</code>, <code>dev</code> against a library):</span><span><i class="g"></i>&le; 1.00</span><span><i class="y"></i>&le; 1.10</span><span><i class="lb"></i>justifiably slower: above 1.10 and at most 1.20 under an accuracy or contract trade (D3050)</span><span><i class="r"></i>above 1.10 otherwise (a waiver names an open item, D2779)</span><span>SNR Δ: <i class="r"></i>more than 1 dB below, <i class="g"></i>at least 3 dB above</span><span>A coloured cell has the colour of its worse number (red &gt; light blue &gt; yellow &gt; green); each number keeps its own colour as its text colour.</span><span>∞ = bit-exact: the output is bit-identical to the f64 reference, so the SNR is infinite.</span></div>
</header>
<main>
<div class="controls">
<label>Level<select id="level"></select></label>
<label>Variant<select id="variant"><option value="">all</option><option value="simd">SIMD entry</option><option value="fallback">fallback</option></select></label>
<label>Status<select id="status"><option value="">all</option><option>release</option><option>dev</option><option>info</option><option value="t:accuracy trade">accuracy trade</option><option value="t:contract trade">contract trade</option></select></label>
<label>Ratio (judged rows)<select id="band"><option value="">all</option><option value="g">&le; 1.00 on every model</option><option value="y">worst in (1.00, 1.10]</option><option value="lb">justifiably slower</option><option value="r">above 1.10, not justified</option></select></label>
<label>Name<input id="q" type="search" placeholder="kernel or entry"></label>
<label class="cb"><input id="little" type="checkbox">SIMD brings little (&gt; 0.80 of its fallback)</label>
<button id="blocks" type="button">Block view</button>
</div>
<div class="chips" id="fams" aria-label="Family"></div>
<div class="summary" id="summary"></div>
<div class="wrap"><table id="t">
<thead><tr>
<th class="l" data-c="0">Level</th>
<th class="l" data-c="1">Status</th>
<th data-c="2" data-num="1">Cycles ours<br><span class="sub cpus">a / b: the CPUs of the level</span></th>
<th data-c="3" data-num="1">Cycles opponent (same CPUs)<br><span class="sub cpus">a / b: the CPUs of the level</span></th>
<th data-c="4" data-num="1">Ratio<br><span class="sub">ours / opponent</span></th>
<th data-c="5" data-num="1">SNR Δ (dB)<br><span class="sub">ours − opponent</span></th>
<th class="l" data-c="6">Opponent</th>
<th class="l" data-c="7">Justification</th>
</tr></thead>
<tbody>
@@ROWS@@
</tbody></table></div>
<h2>Entries without numbers</h2>
<p class="explain">Entries with no result at a level where the measurement plan measures them. @@PLANNED@@ entries that the plan does not measure at a level (such as the fallbacks at <code>avx2-runtime</code>) are not listed.</p>
<ul id="notes" class="explain">
@@NOTES@@
</ul>
</main>
<footer>Generated by <code>python3 -I benches/cycles_page.py</code> from <code>benches/results</code> and <code>benches/limits.json</code> of the commit above; checked by <code>benches/verify_cycles_page.py</code>. Cycles are simulated, never wall-clock timings.</footer>
<script>
const D=@@DATA@@;
const el=id=>document.getElementById(id);
const table=el("t"),body=table.tBodies[0];
const all=Array.from(body.rows),data=all.filter(r=>r.classList.contains("row"));
const S={level:"",variant:"",status:"",band:"",q:"",little:false,fams:new Set(),sort:null,dir:1};
function init(){
  const ls=el("level");
  ls.innerHTML='<option value="">all levels</option>'+D.levels.map(l=>`<option>${l}</option>`).join("");
  let saved=null;try{saved=localStorage.getItem("sdrk-cycles-level")}catch(e){}
  if(saved&&D.levels.includes(saved)){S.level=saved;ls.value=saved}
  ls.onchange=()=>{S.level=ls.value;try{localStorage.setItem("sdrk-cycles-level",S.level)}catch(e){}render()};
  for(const k of ["variant","status","band"])el(k).onchange=e=>{S[k]=e.target.value;render()};
  el("q").oninput=e=>{S.q=e.target.value.trim().toLowerCase();render()};
  el("little").onchange=e=>{S.little=e.target.checked;render()};
  el("blocks").onclick=()=>{S.sort=null;render()};
  const fs=el("fams");
  fs.innerHTML=["all"].concat(D.families).map(f=>`<button class="chip" data-f="${f}" aria-pressed="${f==="all"}">${f}</button>`).join("");
  fs.onclick=e=>{const b=e.target.closest("button");if(!b)return;const f=b.dataset.f;
    if(f==="all")S.fams.clear();else{S.fams.has(f)?S.fams.delete(f):S.fams.add(f)}
    fs.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",x.dataset.f==="all"?S.fams.size===0:S.fams.has(x.dataset.f)));render()};
  table.tHead.onclick=e=>{const th=e.target.closest("th");if(!th)return;const c=+th.dataset.c;
    if(S.sort===c)S.dir=-S.dir;else{S.sort=c;S.dir=1}render()};
  body.onclick=e=>{const td=e.target.closest("td.x");if(td)td.classList.toggle("open")};
  render();
}
function keep(r){const d=r.dataset;
  return (!S.level||d.level===S.level)&&(!S.variant||d.variant===S.variant)&&
    (!S.status||(S.status.startsWith("t:")?d.trade===S.status.slice(2):d.status===S.status))&&
    (!S.band||d.band===S.band)&&(!S.little||d.little==="1")&&(S.fams.size===0||S.fams.has(d.family))&&
    (!S.q||d.kernel.includes(S.q)||d.entry.includes(S.q))}
function render(){
  const c=S.level&&D.cpus[S.level];
  document.querySelectorAll(".cpus").forEach(x=>x.textContent=c?c.join(" / "):(S.level==="simd128"?"instructions per sample":"a / b: the CPUs of the level"));
  table.tHead.querySelectorAll("th").forEach(th=>{th.querySelector(".arrow")?.remove();
    if(S.sort===+th.dataset.c)th.insertAdjacentHTML("beforeend",`<span class="arrow"> ${S.dir>0?"▲":"▼"}</span>`)});
  const shown=data.filter(keep),vis=new Set(shown),kern=new Set(shown.map(r=>r.dataset.kernel));
  table.classList.toggle("flat",S.sort!==null);
  let order=all;
  if(S.sort!==null){const i=S.sort,num=table.tHead.querySelector(`th[data-c="${i}"]`).dataset.num;
    const v=t=>t.dataset.v??t.textContent.trim();
    order=shown.slice().sort((a,b)=>{let x=v(a.cells[i]),y=v(b.cells[i]);
      if(x==="")return 1;if(y==="")return -1;if(num){x=+x;y=+y}return (x<y?-1:x>y?1:0)*S.dir})}
  all.forEach(r=>{r.hidden=r.classList.contains("kh")?(S.sort!==null||!kern.has(r.dataset.kernel)):!vis.has(r)});
  order.forEach(r=>body.appendChild(r));
  const judged=shown.filter(r=>r.dataset.judged==="1"),n=k=>judged.filter(r=>r.dataset.band===k).length;
  el("summary").innerHTML=[[kern.size,"kernels shown"],[shown.length,"rows shown"],[judged.length,"judged (release, dev)"],[n("g"),"at ≤ 1.00"],[n("y"),"in (1.00, 1.10]"],[n("lb"),"justifiably slower"],[n("r"),"above 1.10"],
    [shown.filter(r=>r.dataset.trade==="accuracy trade").length,"accuracy trades"],[shown.filter(r=>r.dataset.trade==="contract trade").length,"contract trades"]]
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
