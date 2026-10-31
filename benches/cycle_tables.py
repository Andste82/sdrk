"""Writes the tables of simulated cycles per level into docs/performance.md.

Usage: python3 -I benches/cycle_tables.py [--rebuild [--force]] [--doc FILE] [JSON file or directory]...

Reads the `cycles` sections of the `cargo measure` documents (`target/measure/<kernel>.json`
by default) and writes one table per level between the markers `<!-- cycle-tables:begin -->` and
`<!-- cycle-tables:end -->` of docs/performance.md: per entry the cycles per sample of ours and of
the comparison on the reference model and on the counter-model of the level, the two ratios and
the comparison. The judged mode of the documents is taken (`cycle_mode` of the entry,
`whole-loop` by default). An entry that could not be simulated is listed below its table with the
note of the document. Only the ratios mean something (D33, D68); the absolute cycles are no
hardware figures.

Default (merge): the rows of the entries in the given documents are replaced, or inserted at
their place in the order of the results tables; every other row, note and level of the block
stays as it is, byte for byte. `target/measure` holds only what was measured in this checkout,
so a partial set of documents never removes rows. A level that the documents do not mention is
not touched. Entries that were retired cannot be removed this way.

`--rebuild`: writes the block from the given documents alone, as the first version of the script
did. It refuses to run when that would drop entries of the block that the documents do not
contain, unless `--force` is given (use it when every kernel was measured, or to retire
entries). `--doc FILE` writes another file than docs/performance.md (tests, scratch copies).
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "performance.md")
BEGIN = "<!-- cycle-tables:begin -->"
END = "<!-- cycle-tables:end -->"
# Levels in the order of the tables, with their platform.
LEVELS = [("x86_64", "sse2"), ("x86_64", "sse4.2"), ("x86_64", "avx2"),
          ("x86_64-avx512", "avx512"), ("aarch64", "neon")]
# The kernel prefixes in the order of the results tables; entries of a prefix are sorted.
PREFIXES = ["convert", "rotate", "fir", "resample", "mag", "quad_demod", "fm_mod", "squelch",
            "window", "power_db", "agc", "dc_block", "deemphasis", "biquad", "dot", "scale",
            "fft", "rfft"]


def documents(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".json")]
        else:
            files.append(p)
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        yield from documents_from([d])


def documents_from(docs):
    """The documents that carry cycles, from already loaded JSON values."""
    for d in docs:
        if isinstance(d, dict) and isinstance(d.get("cycles"), dict):
            yield d


def order(entry):
    matches = [q for q in PREFIXES if entry == q or entry.startswith(q + "_")]
    if not matches:
        return (len(PREFIXES), entry)
    return (PREFIXES.index(max(matches, key=len)), entry)


def collect(docs):
    levels = {}
    for d in docs:
        for platform, per_level in d["cycles"].items():
            if not isinstance(per_level, dict):
                continue
            for level, entries in per_level.items():
                if not isinstance(entries, dict):
                    continue
                slot = levels.setdefault((platform, level), {"entries": {}, "counts": None})
                for name, v in entries.items():
                    if isinstance(v, dict) and ("reference_model" in v or "note" in v):
                        slot["entries"][name] = v
                        if isinstance(v.get("block_counts"), str) and "reference_model" in v:
                            slot["counts"] = v["block_counts"]
    return levels


def fmt(x):
    return f"{x:.2f}"


def parts(platform, level, slot):
    """The pieces of one level from the documents: (intro, rows, failed, skipped) where rows maps an
    entry to its table line, failed maps an entry to its note and skipped lists the entries that were
    not measured at the level. intro is None when no entry has cycles."""
    rows = sorted(slot["entries"].items(), key=lambda kv: order(kv[0]))
    done = [(n, v) for n, v in rows if "reference_model" in v]
    notes = [(n, v["note"]) for n, v in rows if "reference_model" not in v]
    if not done and not notes:
        return None
    first = done[0][1] if done else None
    ref_cpu = first["reference_model"]["ours"]["cpu"] if first else "reference model"
    cnt_cpu = first["counter_model"]["ours"]["cpu"] if first else "counter-model"
    mode = first.get("cycle_mode", "whole-loop") if first else "whole-loop"
    counts = slot["counts"] or "the counter of the level"
    intro = (f"Reference model `{ref_cpu}`, counter-model `{cnt_cpu}`, mode `{mode}`; "
             f"execution counts: {counts}.")
    head = [f"| Entry | ours {ref_cpu} (cycles/sample) | comparison {ref_cpu} (cycles/sample) "
            f"| ratio {ref_cpu} (ours/comparison) | ours {cnt_cpu} (cycles/sample) "
            f"| comparison {cnt_cpu} (cycles/sample) | ratio {cnt_cpu} (ours/comparison) "
            f"| comparison |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines = {}
    for name, v in done:
        r, c = v["reference_model"], v["counter_model"]
        comp = v.get("compared_with", "").replace("|", "\\|")
        lines[name] = (f"| `{name}` | {fmt(r['ours']['cycles_per_sample'])} "
                       f"| {fmt(r['other']['cycles_per_sample'])} | {fmt(r['ratio'])} "
                       f"| {fmt(c['ours']['cycles_per_sample'])} "
                       f"| {fmt(c['other']['cycles_per_sample'])} | {fmt(c['ratio'])} | {comp} |")
    failed = {n: note for n, note in notes if not note.startswith("not simulated: not measured at")}
    skipped = [n for n, note in notes if note.startswith("not simulated: not measured at")]
    return {"intro": intro, "head": head, "rows": lines, "failed": failed, "skipped": skipped}


def render(platform, level, p):
    """The text lines of one level from its pieces, rows and notes in the order of the results tables."""
    out = [f"#### Simulated cycles at `{level}` (`{platform}`)", "", p["intro"], ""]
    out += p["head"]
    out += [p["rows"][n] for n in sorted(p["rows"], key=order)]
    out.append("")
    for n in sorted(p["failed"], key=order):
        out.append(f"- `{n}`: {p['failed'][n]}.")
    if p["skipped"]:
        out.append("- Not measured at this level: "
                   + ", ".join(f"`{n}`" for n in sorted(p["skipped"], key=order)) + ".")
    if p["failed"] or p["skipped"]:
        out.append("")
    return out


SECTION = re.compile(r"^#### Simulated cycles at `([^`]+)` \(`([^`]+)`\)$")
ROW = re.compile(r"^\| `([^`]+)` \|")
FAILED = re.compile(r"^- `([^`]+)`: (.*)\.$")
SKIPPED = re.compile(r"^- Not measured at this level: (.*)\.$")


def parse_block(block):
    """The sections of an existing block: {(platform, level): pieces + the original lines}, in order."""
    lines = block.split("\n")
    starts = [i for i, l in enumerate(lines) if SECTION.match(l)]
    sections = {}
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        level, platform = SECTION.match(lines[start]).groups()
        body = lines[start:end]
        rows, failed, skipped = {}, {}, []
        head, intro = [], None
        for l in body[1:]:
            if intro is None and l.startswith("Reference model "):
                intro = l
            elif l.startswith("| Entry |") or l.startswith("| --- |"):
                head.append(l)
            elif ROW.match(l):
                rows[ROW.match(l).group(1)] = l
            elif FAILED.match(l):
                failed[FAILED.match(l).group(1)] = FAILED.match(l).group(2)
            elif SKIPPED.match(l):
                skipped = re.findall(r"`([^`]+)`", SKIPPED.match(l).group(1))
        sections[(platform, level)] = {"intro": intro, "head": head, "rows": rows, "failed": failed,
                                       "skipped": skipped, "lines": body}
    return sections


def merge(block, levels):
    """The block with the entries of the given levels replaced or inserted. Returns (text, replaced,
    inserted, warnings). Levels and entries that the documents do not mention are kept byte for byte."""
    old = parse_block(block)
    replaced = inserted = 0
    warnings = []
    new = {}
    for (platform, level), slot in levels.items():
        got = parts(platform, level, slot)
        if got is None:
            continue
        key = (platform, level)
        names = set(slot["entries"])
        if key not in old or not old[key]["intro"]:
            new[key] = render(platform, level, got)
            inserted += len(got["rows"])
            continue
        cur = dict(old[key])
        if cur["intro"] != got["intro"]:
            warnings.append(f"{level} ({platform}): the models or the counter differ from the block "
                            f"(kept: {cur['intro']!r}); measure everything and use --rebuild --force")
        replaced += len(names & set(cur["rows"]))
        inserted += len(set(got["rows"]) - set(cur["rows"]))
        cur["rows"] = {n: l for n, l in cur["rows"].items() if n not in names}
        cur["rows"].update(got["rows"])
        cur["failed"] = {n: t for n, t in cur["failed"].items() if n not in names}
        cur["failed"].update(got["failed"])
        cur["skipped"] = [n for n in cur["skipped"] if n not in names] + got["skipped"]
        new[key] = render(platform, level, cur)
    out = []
    keys = [k for k in old if k not in new]
    for platform, level in LEVELS:
        key = (platform, level)
        if key in new:
            out += new[key]
        elif key in old:
            out += old[key]["lines"]
    for key in keys:
        if key not in LEVELS:
            out += old[key]["lines"]
    for key in new:
        if key not in LEVELS:
            out += new[key]
    return "\n".join(out).rstrip("\n"), replaced, inserted, warnings


def rebuild(levels):
    text = []
    for platform, level in LEVELS:
        slot = levels.get((platform, level))
        got = parts(platform, level, slot) if slot else None
        if got:
            text += render(platform, level, got)
    return "\n".join(text).rstrip("\n")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    flags = {"--rebuild": False, "--force": False}
    doc_path = DOC
    paths = []
    while args:
        a = args.pop(0)
        if a in flags:
            flags[a] = True
        elif a == "--doc":
            doc_path = args.pop(0)
        else:
            paths.append(a)
    if flags["--force"] and not flags["--rebuild"]:
        sys.exit("--force only belongs to --rebuild")
    paths = paths or [os.path.join(ROOT, "target", "measure")]
    levels = collect(documents(paths))
    with open(doc_path) as fh:
        doc = fh.read()
    a, b = doc.index(BEGIN) + len(BEGIN), doc.index(END)
    block = doc[a:b].strip("\n")
    if flags["--rebuild"]:
        text = rebuild(levels)
        kept = {n for sec in parse_block(text).values() for n in sec["rows"]}
        dropped = sorted({n for sec in parse_block(block).values() for n in sec["rows"]} - kept)
        if dropped and not flags["--force"]:
            sys.exit(f"--rebuild from these documents would drop {len(dropped)} entries of the block "
                     f"({', '.join(dropped[:6])}{', ...' if len(dropped) > 6 else ''}); measure them "
                     f"too, or use --rebuild --force to retire them")
        print(f"rebuilt: {len(kept)} rows, {len(dropped)} dropped", file=sys.stderr)
    else:
        text, replaced, inserted, warnings = merge(block, levels)
        for w in warnings:
            print("warning: " + w, file=sys.stderr)
        print(f"merged: {replaced} rows replaced, {inserted} inserted", file=sys.stderr)
    doc = doc[:a] + "\n\n" + text + "\n\n" + doc[b:]
    with open(doc_path, "w") as fh:
        fh.write(doc)


if __name__ == "__main__":
    main()
