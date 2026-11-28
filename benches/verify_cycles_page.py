"""Checks the overview page of the kernel cycles against the committed results, independently of
benches/cycles_page.py and benches/cycle_tables.py.

Usage: python3 -I benches/verify_cycles_page.py [--repo DIR] [--doc] PAGE.html

Reads `benches/results/cycles/*.json`, `benches/results/wasm32.json` and `benches/limits.json` of
DIR (default: the checkout of this script) and the table of PAGE.html, and compares every row:

- one row per simulated record of every level and per record of `simd128`, and no other row; the
  entries without numbers are listed in the notes;
- the status (limits.json), the trade mark, the `simd128` verdict, the kernel name (without
  `_fallback`, a second opponent under its base entry) and the full entry name, the level label;
- ours, the opponent and the ratio on both models as shown (at `avx2-runtime` the opponent of the
  `avx2` record and the ratio recomputed, our own `avx2` build as information), the SNR of both
  sides and their difference;
- the colour class of every ratio and of the SNR cells, the band of the row and `data-judged`;
- the opponent text and the `own (information)` mark, the justification texts;
- no judged row (`release`, `dev`) compares with our own code.

`--doc` also checks the cycle tables and the `simd128` table of docs/performance.md the same way.
Prints every difference and the total; exits 1 on any difference.
"""
import html.parser
import json
import os
import re
import sys

EXACT = 999.0
JUDGED = {"release", "dev"}
LEVELS = ["sse2", "sse4.2", "avx2", "avx2-runtime", "avx512", "neon", "simd128"]
REASONS = [("accuracy_trade", "accuracy trade"), ("contract_trade", "contract trade"),
           ("waiver", "waiver"), ("accuracy_waiver", "accuracy waiver"), ("cycle_waiver", "cycle waiver")]


class Checker:
    def __init__(self):
        self.diffs = []
        self.checked = {"rows": 0, "values": 0, "doc_rows": 0, "doc_values": 0}

    def diff(self, where, what, got, want):
        g, w = repr(got), repr(want)
        if len(g) + len(w) > 400:
            i = next((k for k, (a, b) in enumerate(zip(g, w)) if a != b), min(len(g), len(w)))
            g, w = "..." + g[max(0, i - 60):i + 100] + "...", "..." + w[max(0, i - 60):i + 100] + "..."
        self.diffs.append(f"{where}: {what}: shown {g}, expected {w}")

    def same(self, where, what, got, want, key="values"):
        self.checked[key] += 1
        if got != want:
            self.diff(where, what, got, want)


def own_name(name):
    n = name.lower()
    return n.startswith("sdrk") or n.startswith("ours") or "own (information)" in n


def f2(x):
    return f"{x:.2f}"


def db(x):
    return "–" if x is None else ("exact" if x >= EXACT else f"{x:.1f}")


def delta(o, c):
    if o is None or c is None:
        return "–"
    if o >= EXACT and c >= EXACT:
        return "0.0"
    if o >= EXACT:
        return "ours exact"
    if c >= EXACT:
        return "opponent exact"
    d = round(o - c, 1)
    return "0.0" if d == 0 else f"{d:+.1f}"


def colour(x, judged, trade):
    """The colour rule of the page (D3050)."""
    if not judged:
        return ""
    if x <= 1.0:
        return "g"
    if x <= 1.10:
        return "y"
    return "lb" if trade and x <= 1.20 else "r"


def snr_colour(o, c, judged):
    if not judged or o is None or c is None:
        return ""
    return "r" if c - o > 1 else ("g" if o - c >= 3 else "")


def load(repo):
    with open(os.path.join(repo, "benches", "limits.json")) as fh:
        limits = json.load(fh)
    cyc = {}
    d = os.path.join(repo, "benches", "results", "cycles")
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                doc = json.load(fh)
            for level, lv in doc["levels"].items():
                cyc[level] = lv
    wp = os.path.join(repo, "benches", "results", "wasm32.json")
    wasm = {}
    if os.path.exists(wp):
        with open(wp) as fh:
            wasm = json.load(fh)["levels"]["simd128"]["kernels"]
    return limits, cyc, wasm


def ours_figures(cyc, wasm):
    """{level: {entry: our figures}} for the second-opponent rule."""
    out = {}
    for level, lv in cyc.items():
        out[level] = {n: (v["reference_model"]["ours"]["cycles_per_sample"], v["counter_model"]["ours"]["cycles_per_sample"])
                      for n, v in lv["entries"].items() if "reference_model" in v}
    out["simd128"] = {n: (k["ours_instr_per_sample"],) for n, k in wasm.items()}
    return out


def second_map(cyc, wasm):
    """{entry: base}: `<base>_<suffix>` (suffix not `fallback`) whose ours equals the base's on every
    level where both are measured."""
    figs = ours_figures(cyc, wasm)
    names = set(n for lv in cyc.values() for n in lv["entries"]) | set(wasm)
    out = {}
    for name in sorted(names):
        if name.endswith("_fallback"):
            continue
        parts = name.split("_")
        for i in range(len(parts) - 1, 0, -1):
            base = "_".join(parts[:i])
            if base not in names:
                continue
            pairs = [(f[name], f[base]) for f in figs.values() if name in f and base in f]
            if pairs and all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for x, y in pairs for a, b in zip(x, y)):
                out[name] = out.get(base, base)
                break
    return out


def expected(limits, cyc, wasm, ck):
    """{(level, entry): the expected cells}, and the notes {(level, entry)}."""
    second = second_map(cyc, wasm)
    rows, notes = {}, set()

    def entry_of(name, level):
        e = limits.get(name, {}).get(level)
        if not isinstance(e, dict) or "status" not in e:
            ck.diff(f"limits {level} {name}", "status", None, "an entry in limits.json")
            return {}
        return e

    def base(level, name, e, kind, cw, ours, opp, ratio, snr, unit, verdict="", own_build=None):
        status = e.get("status")
        trade = "accuracy trade" if "accuracy_trade" in e else ("contract trade" if "contract_trade" in e else "")
        own = kind != "library" or own_name(cw)
        judged = status in JUDGED
        if judged and own:
            ck.diff(f"results {level} {name}", "judged against our own code", kind, "library")
        jl = judged and not own
        fb = name.endswith("_fallback")
        simd = name[: -len("_fallback")] if fb else name
        kernel = second.get(simd, simd)
        so, sc = (snr or {}).get("ours"), (snr or {}).get("comparison", (snr or {}).get("other"))
        fmt = (lambda x: f"{x:.2f}") if unit == "instructions" else (lambda x: f"{x:.3f}")
        tail = " instr" if unit == "instructions" else ""
        classes = [colour(x, jl, trade) for x in ratio]
        rank = ["", "g", "y", "lb", "r"]
        rows[(level, name)] = {
            "status": status, "trade": trade, "verdict": verdict, "kernel": kernel,
            "title_entry": f"entry {name}", "level": f"fallback ({level})" if fb else level,
            "ours": " / ".join(f2(x) for x in ours) + tail, "opp": " / ".join(f2(x) for x in opp) + tail,
            "snr_o": db(so), "snr_c": db(sc), "delta": delta(so, sc),
            "ratio": [fmt(x) for x in ratio], "ratio_classes": classes,
            "snr_class": snr_colour(so, sc, jl), "band": max(classes, key=rank.index),
            "judged": "1" if jl else "0", "own": own, "opponent": cw,
            "own_build": None if own_build is None else f"ours / own avx2 build {own_build[0]:.3f} / {own_build[1]:.3f} (information)",
            "reasons": [f"{label}: {e[key]}" for key, label in REASONS if isinstance(e.get(key), str) and e[key]],
            "variant": "fallback" if fb else "simd",
        }

    for level, lv in cyc.items():
        lib = cyc.get("avx2", {}).get("entries", {}) if level == "avx2-runtime" else None
        for name, v in lv["entries"].items():
            if "reference_model" not in v:
                notes.add((level, name))
                continue
            r, c = v["reference_model"], v["counter_model"]
            ours = (r["ours"]["cycles_per_sample"], c["ours"]["cycles_per_sample"])
            for i, m in enumerate((r, c)):
                want = m["ours"]["cycles_per_sample"] / m["comparison"]["cycles_per_sample"]
                if abs(m["ratio"] - want) > 1e-9 * max(1, want):
                    ck.diff(f"results {level} {name}", f"ratio of model {i}", m["ratio"], "ours / comparison")
            if lib is not None:
                a = lib.get(name)
                if not a or "reference_model" not in a:
                    notes.add((level, name))
                    continue
                opp = (a["reference_model"]["comparison"]["cycles_per_sample"], a["counter_model"]["comparison"]["cycles_per_sample"])
                s, ls = v.get("snr_db"), a.get("snr_db")
                snr = {"ours": s["ours"], "comparison": ls.get("comparison", ls.get("other"))} if s and ls else None
                base(level, name, entry_of(name, level), a.get("comparison_kind"), a.get("compared_with", ""),
                     ours, opp, (ours[0] / opp[0], ours[1] / opp[1]), snr, "cycles",
                     own_build=(r["ratio"], c["ratio"]))
            else:
                opp = (r["comparison"]["cycles_per_sample"], c["comparison"]["cycles_per_sample"])
                base(level, name, entry_of(name, level), v.get("comparison_kind"), v.get("compared_with", ""),
                     ours, opp, (r["ratio"], c["ratio"]), v.get("snr_db"), "cycles")
    for name, k in wasm.items():
        e = entry_of(name, "simd128")
        if k.get("status") != e.get("status"):
            ck.diff(f"wasm32.json simd128 {name}", "status of the record", k.get("status"), e.get("status"))
        base("simd128", name, e, k.get("comparison_kind"), k["compared_with"], (k["ours_instr_per_sample"],),
             (k["other_instr_per_sample"],), (k["ratio"],), k.get("snr_db"), "instructions",
             verdict={"waived": "waived", "fail": "fail"}.get(k["verdict"], ""))
    return rows, notes


class Page(html.parser.HTMLParser):
    """The rows of the table: attributes, and per cell its classes, text and marked spans."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.notes = [], []
        self.stack = []
        self.in_body = self.in_notes = False
        self.row = self.cell = self.note = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "tbody":
            self.in_body = True
        elif tag == "ul" and a.get("id") == "notes":
            self.in_notes = True
        elif tag == "li" and self.in_notes:
            self.note = ""
        elif tag == "tr" and self.in_body:
            self.row = {"attrs": a, "cells": []}
        elif tag == "td" and self.row is not None:
            self.cell = {"cls": cls, "text": "", "spans": [], "divs": [], "title": None}
            self.row["cells"].append(self.cell)
        elif self.cell is not None:
            if tag == "code" and a.get("title"):
                self.cell["title"] = a["title"]
            if tag == "span" and ("st" in cls or "v" in cls):
                self.cell["spans"].append({"cls": cls, "text": ""})
            if tag == "div":
                self.cell["divs"].append({"cls": cls, "text": ""})
        if tag not in ("br", "meta", "input"):
            self.stack.append((tag, cls))

    def handle_endtag(self, tag):
        while self.stack:
            t, _ = self.stack.pop()
            if t == tag:
                break
        if tag == "tbody":
            self.in_body = False
        elif tag == "ul":
            self.in_notes = False
        elif tag == "li" and self.note is not None:
            self.notes.append(self.note)
            self.note = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "td":
            self.cell = None

    def handle_data(self, data):
        if self.note is not None:
            self.note += data
        if self.cell is None:
            return
        if any(t == "span" and "m" in cls for t, cls in self.stack):
            return  # the ellipsis of a closed cell is no text of the page
        self.cell["text"] += data
        tags = [t for t, _ in self.stack]
        if any(t == "span" and ("st" in cls or "v" in cls) for t, cls in self.stack) and self.cell["spans"]:
            self.cell["spans"][-1]["text"] += data
        if "div" in tags and self.cell["divs"]:
            self.cell["divs"][-1]["text"] += data


def check_page(page_path, limits, cyc, wasm, ck):
    exp, notes = expected(limits, cyc, wasm, ck)
    p = Page()
    with open(page_path, encoding="utf-8") as fh:
        p.feed(fh.read())
    seen = set()
    for row in p.rows:
        a = row["attrs"]
        key = (a.get("data-level"), a.get("data-entry"))
        where = f"page {key[0]} {key[1]}"
        if key in seen:
            ck.diff(where, "duplicate row", 2, 1)
        seen.add(key)
        e = exp.get(key)
        if e is None:
            ck.diff(where, "row", "present", "no record")
            continue
        ck.checked["rows"] += 1
        cells = row["cells"]
        if len(cells) != 11:
            ck.diff(where, "cells", len(cells), 11)
            continue
        st, kern, lvl, ours, opp, so, sc, ratio, dl, oppn, why = cells
        ck.same(where, "data-status", a.get("data-status"), e["status"])
        ck.same(where, "data-trade", a.get("data-trade"), e["trade"])
        ck.same(where, "data-band", a.get("data-band"), e["band"])
        ck.same(where, "data-judged", a.get("data-judged"), e["judged"])
        ck.same(where, "data-kernel", a.get("data-kernel"), e["kernel"])
        ck.same(where, "data-variant", a.get("data-variant"), e["variant"])
        marks = [s["text"] for s in st["spans"]]
        ck.same(where, "status", marks, [e["status"]] + [x for x in (e["trade"], e["verdict"]) if x])
        ck.same(where, "kernel", kern["text"].strip(), e["kernel"])
        ck.same(where, "kernel title", (kern["title"] or "").split(" (")[0], e["title_entry"])
        ck.same(where, "level", lvl["text"].strip(), e["level"])
        ck.same(where, "cycles ours", ours["text"].strip(), e["ours"])
        ck.same(where, "cycles opponent", opp["text"].strip(), e["opp"])
        ck.same(where, "SNR ours", so["text"].strip(), e["snr_o"])
        ck.same(where, "SNR opponent", sc["text"].strip(), e["snr_c"])
        ck.same(where, "SNR delta", dl["text"].strip(), e["delta"])
        for name, c in (("SNR ours", so), ("SNR opponent", sc), ("SNR delta", dl)):
            got = [x for x in c["cls"] if x in ("g", "y", "r", "lb")]
            ck.same(where, f"{name} colour", got, [e["snr_class"]] if e["snr_class"] else [])
        vals = [s for s in ratio["spans"] if "v" in s["cls"]]
        ck.same(where, "ratio", [s["text"] for s in vals], e["ratio"])
        got = [next((x for x in s["cls"] if x in ("g", "y", "r", "lb")), "") for s in vals]
        ck.same(where, "ratio colour", got, e["ratio_classes"])
        own_mark = any(s["text"] == "own (information)" for s in oppn["spans"])
        ck.same(where, "own (information) mark", own_mark, e["own"])
        text = oppn["text"]
        sub = oppn["divs"][0]["text"] if oppn["divs"] else None
        if sub:
            text = text[: len(text) - len(sub)]
        text = text.replace("own (information) ", "", 1) if own_mark else text
        ck.same(where, "opponent", text, e["opponent"])
        ck.same(where, "own avx2 build", sub, e["own_build"])
        ck.same(where, "justification", [d["text"] for d in why["divs"]], e["reasons"])
        if e["judged"] == "1" and (own_mark or own_name(text)):
            ck.diff(where, "judged against our own code", text, "a library")
    for key in sorted(set(exp) - seen):
        ck.diff(f"page {key[0]} {key[1]}", "row", "missing", "present")
    shown = set()
    for n in p.notes:
        m = re.match(r"^(\S+) (\S+): ", n)
        if m:
            shown.add((m.group(1), m.group(2)))
    for key in sorted(notes ^ shown):
        ck.diff(f"notes {key[0]} {key[1]}", "note", key in shown, key in notes)


def snr_cell(snr, trade):
    if not snr:
        return "-"
    cell = f"{db(snr['ours'])} / {db(snr.get('comparison', snr.get('other')))}"
    return cell + f" ({trade})" if trade else cell


def check_doc(repo, limits, cyc, wasm, ck):
    """The cycle tables and the simd128 table of docs/performance.md."""
    exp, _ = expected(limits, cyc, wasm, Checker())
    with open(os.path.join(repo, "docs", "performance.md")) as fh:
        text = fh.read()
    blk = text[text.index("<!-- cycle-tables:begin -->"):text.index("<!-- cycle-tables:end -->")]
    sections = re.split(r"^#### Simulated cycles at `([^`]+)` \(`[^`]+`\)$", blk, flags=re.M)
    tables = {sections[i]: sections[i + 1] for i in range(1, len(sections), 2)}
    sb = text[text.index("<!-- simd128-instructions:begin -->"):text.index("<!-- simd128-instructions:end -->")]
    tables["simd128"] = sb
    for level in sorted(set(cyc) | ({"simd128"} if wasm else set())):
        if level not in tables:
            ck.diff("performance.md", "level table", None, level)
            continue
        shown = {}
        for line in tables[level].split("\n"):
            m = re.match(r"^\| `([^`]+)` \|(.*)\|$", line)
            if m:
                if m.group(1) in shown:
                    ck.diff(f"performance.md {level}", "duplicate row", m.group(1), "one row")
                shown[m.group(1)] = [c.strip() for c in re.split(r"(?<!\\)\|", m.group(2))]
        want_rows = {n for (lvl, n) in exp if lvl == level}
        for name in sorted(want_rows ^ set(shown)):
            ck.diff(f"performance.md {level}", "row present", name in shown, name in want_rows)
        for name in sorted(want_rows & set(shown)):
            e, cells = exp[(level, name)], shown[name]
            ck.checked["doc_rows"] += 1
            cw = e["opponent"].replace("|", "\\|")
            cmp_cell = f"own (information): {cw}" if e["own"] else cw
            if level == "simd128":
                k = wasm[name]
                want = [f2(k["ours_instr_per_sample"]), f2(k["other_instr_per_sample"]), f2(k["ratio"]),
                        snr_cell(k.get("snr_db"), ""), " ".join(x for x in (e["status"], e["verdict"]) if x), cmp_cell]
            else:
                o, c = e["ours"].split(" / "), e["opp"].split(" / ")
                lv = cyc[level]["entries"][name]
                if level == "avx2-runtime":
                    a = cyc["avx2"]["entries"][name]
                    rr = (lv["reference_model"]["ours"]["cycles_per_sample"] / a["reference_model"]["comparison"]["cycles_per_sample"],
                          lv["counter_model"]["ours"]["cycles_per_sample"] / a["counter_model"]["comparison"]["cycles_per_sample"])
                    s, ls = lv.get("snr_db"), a.get("snr_db")
                    snr = {"ours": s["ours"], "comparison": ls.get("comparison", ls.get("other"))} if s and ls else None
                else:
                    rr = (lv["reference_model"]["ratio"], lv["counter_model"]["ratio"])
                    snr = lv.get("snr_db")
                want = [o[0], c[0], f2(rr[0]), o[1], c[1], f2(rr[1]), snr_cell(snr, e["trade"])]
                if level == "avx2-runtime":
                    want.append(f"{f2(lv['reference_model']['ratio'])} / {f2(lv['counter_model']['ratio'])}")
                want += [e["status"], cmp_cell]
            if len(cells) != len(want):
                ck.diff(f"performance.md {level} {name}", "cells", len(cells), len(want))
                continue
            for j, (g, w) in enumerate(zip(cells, want)):
                ck.same(f"performance.md {level} {name}", f"column {j + 2}", g, w, key="doc_values")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page, doc = None, False
    while args:
        a = args.pop(0)
        if a == "--repo" and args:
            repo = args.pop(0)
        elif a == "--doc":
            doc = True
        elif page is None and not a.startswith("--"):
            page = a
        else:
            sys.exit(f"unknown argument {a!r}; usage: verify_cycles_page.py [--repo DIR] [--doc] PAGE.html")
    if page is None:
        sys.exit("usage: verify_cycles_page.py [--repo DIR] [--doc] PAGE.html")
    ck = Checker()
    limits, cyc, wasm = load(repo)
    check_page(page, limits, cyc, wasm, ck)
    if doc:
        check_doc(repo, limits, cyc, wasm, ck)
    for d in ck.diffs:
        print(d)
    print(f"checked: {ck.checked}")
    print(f"differences: {len(ck.diffs)}")
    return 1 if ck.diffs else 0


if __name__ == "__main__":
    sys.exit(main())
