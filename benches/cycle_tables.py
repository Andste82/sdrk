"""Writes the tables of simulated cycles per level into docs/performance.md.

Usage: python3 -I benches/cycle_tables.py [JSON file or directory]...

Reads the `cycles` sections of the `cargo measure` documents (`target/measure/<kernel>.json`
by default) and replaces the text between the markers `<!-- cycle-tables:begin -->` and
`<!-- cycle-tables:end -->` of docs/performance.md with one table per level: per entry the
cycles per sample of ours and of the comparison on the reference model and on the
counter-model of the level, the two ratios and the comparison. The judged mode of the documents
is taken (`cycle_mode` of the entry, `whole-loop` by default). An entry that could not be
simulated is listed below its table with the note of the document. Only the ratios mean
something (D33, D68); the absolute cycles are no hardware figures.
"""
import json
import os
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


def table(platform, level, slot):
    rows = sorted(slot["entries"].items(), key=lambda kv: order(kv[0]))
    done = [(n, v) for n, v in rows if "reference_model" in v]
    notes = [(n, v["note"]) for n, v in rows if "reference_model" not in v]
    if not done and not notes:
        return []
    first = done[0][1] if done else None
    ref_cpu = first["reference_model"]["ours"]["cpu"] if first else "reference model"
    cnt_cpu = first["counter_model"]["ours"]["cpu"] if first else "counter-model"
    out = [f"#### Simulated cycles at `{level}` (`{platform}`)", ""]
    mode = first.get("cycle_mode", "whole-loop") if first else "whole-loop"
    counts = slot["counts"] or "the counter of the level"
    out.append(f"Reference model `{ref_cpu}`, counter-model `{cnt_cpu}`, mode `{mode}`; "
               f"execution counts: {counts}.")
    out.append("")
    out.append(f"| Entry | ours {ref_cpu} (cycles/sample) | comparison {ref_cpu} (cycles/sample) "
               f"| ratio {ref_cpu} (ours/comparison) | ours {cnt_cpu} (cycles/sample) "
               f"| comparison {cnt_cpu} (cycles/sample) | ratio {cnt_cpu} (ours/comparison) "
               f"| comparison |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, v in done:
        r, c = v["reference_model"], v["counter_model"]
        comp = v.get("compared_with", "").replace("|", "\\|")
        out.append(f"| `{name}` | {fmt(r['ours']['cycles_per_sample'])} "
                   f"| {fmt(r['other']['cycles_per_sample'])} | {fmt(r['ratio'])} "
                   f"| {fmt(c['ours']['cycles_per_sample'])} "
                   f"| {fmt(c['other']['cycles_per_sample'])} | {fmt(c['ratio'])} | {comp} |")
    out.append("")
    skipped = [n for n, note in notes if note.startswith("not simulated: not measured at")]
    failed = [(n, note) for n, note in notes if not note.startswith("not simulated: not measured at")]
    for n, note in failed:
        out.append(f"- `{n}`: {note}.")
    if skipped:
        out.append("- Not measured at this level: " + ", ".join(f"`{n}`" for n in skipped) + ".")
    if failed or skipped:
        out.append("")
    return out


def main():
    paths = sys.argv[1:] or [os.path.join(ROOT, "target", "measure")]
    levels = collect(documents(paths))
    text = []
    for platform, level in LEVELS:
        slot = levels.get((platform, level))
        if slot:
            text += table(platform, level, slot)
    with open(DOC) as fh:
        doc = fh.read()
    a, b = doc.index(BEGIN) + len(BEGIN), doc.index(END)
    doc = doc[:a] + "\n\n" + "\n".join(text).rstrip("\n") + "\n\n" + doc[b:]
    with open(DOC, "w") as fh:
        fh.write(doc)


if __name__ == "__main__":
    main()
