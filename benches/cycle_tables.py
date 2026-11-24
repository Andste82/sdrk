"""Writes the tables of simulated cycles per level and the table of `simd128` instructions into
docs/performance.md.

Usage: python3 -I benches/cycle_tables.py [--check] [--doc FILE] [--results DIR] [--counts FILE]
       [--limits FILE]

Reads the committed cycle results `benches/results/cycles/<platform>.json` (written by
`cargo measure <kernel> --merge` and `cargo measure --all`, D2704) and writes one table per level
between the markers `<!-- cycle-tables:begin -->` and `<!-- cycle-tables:end -->` of
docs/performance.md: per entry the cycles per sample of ours and of the comparison on the
reference model and on the counter-model of the level, the two ratios, the accuracy of both sides
(the SNR in dB against the f64 reference, `ours / comparison`, D34; `exact` for an exact result) and
the comparison. An entry whose `benches/limits.json` entry at the level has an `accuracy_trade`
(D2760) is marked `(accuracy trade)` in the accuracy column (`--limits FILE` takes another file).
An entry that could not be simulated is listed below its table with its note. Only the ratios mean
something (D33, D68); the absolute cycles are no hardware figures.

The table of `avx2-runtime` compares, like every level, with the reference library (D2764): our
cycles and SNR of the run-time-detection build against the comparison's of the `avx2` record of the
entry (the same workload on the same core models); the ratio of ours to our own fixed `avx2` build,
which the record holds, is an extra column of information.

The block is generated from the results files alone and is not edited by hand: a change of the
numbers is a change of the results files (`cargo measure ... --merge`), then this script.
`--check` writes nothing and fails when the block differs from what the results files give,
naming the rows that differ. `--doc FILE` and `--results DIR` take other files (tests, scratch
copies).

The second block, between `<!-- simd128-instructions:begin -->` and `<!-- simd128-instructions:end -->`,
is the one level without a core model (M13.1): instructions per sample of ours and of the
comparison at `simd128` with the accuracy of both sides, from `benches/results/wasm32.json` (`--counts FILE`, by default next to the
cycle results directory). It is written only when that file exists.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "performance.md")
RESULTS = os.path.join(ROOT, "benches", "results", "cycles")
LIMITS = os.path.join(ROOT, "benches", "limits.json")
EXACT = 999.0
BEGIN = "<!-- cycle-tables:begin -->"
END = "<!-- cycle-tables:end -->"
SIMD_BEGIN = "<!-- simd128-instructions:begin -->"
SIMD_END = "<!-- simd128-instructions:end -->"
# Levels in the order of the tables, with their platform.
LEVELS = [("x86_64", "sse2"), ("x86_64", "sse4.2"), ("x86_64", "avx2"), ("x86_64", "avx2-runtime"),
          ("x86_64-avx512", "avx512"), ("aarch64", "neon")]
# The kernel prefixes in the order of the results tables; entries of a prefix are sorted.
PREFIXES = ["convert", "rotate", "scramble", "fir", "resample", "mag", "quad_demod", "fm_mod", "squelch",
            "window", "power_db", "agc", "dc_block", "deemphasis", "biquad", "dot", "scale",
            "add", "mul", "interleave", "deinterleave", "oscillate", "fft", "rfft"]
NOT_MEASURED = "not simulated: not measured at"
# The level judged against the comparison of another level's record (D2764), and that level.
RUNTIME, RUNTIME_LIBRARY = "avx2-runtime", "avx2"


def load(results_dir):
    """The results documents of the directory, by platform."""
    docs = {}
    for f in sorted(os.listdir(results_dir)):
        if f.endswith(".json"):
            with open(os.path.join(results_dir, f)) as fh:
                d = json.load(fh)
            docs[d["platform"]] = d
    return docs


def order(entry):
    matches = [q for q in PREFIXES if entry == q or entry.startswith(q + "_")]
    if not matches:
        return (len(PREFIXES), entry)
    return (PREFIXES.index(max(matches, key=len)), entry)


def fmt(x):
    return f"{x:.2f}"


def db(x):
    """An SNR in dB, one decimal; `exact` for the value of an exact result."""
    return "exact" if x >= EXACT else f"{x:.1f}"


def accuracy(snr, traded=False):
    """The accuracy cell: `ours / comparison` in dB, `-` without an SNR."""
    if not snr:
        return "-"
    cell = f"{db(snr['ours'])} / {db(snr.get('comparison', snr.get('other')))}"
    return cell + " (accuracy trade)" if traded else cell


def against_library(v, lib):
    """The record `v` of `avx2-runtime` against the comparison of the `avx2` record `lib` (D2764):
    the models and the SNR with the comparison's figures of `lib`, the comparison's name, and the
    ratios to our own `avx2` build as information; None without `lib`."""
    if not lib or "reference_model" not in lib:
        return None
    def m(key):
        o, c = v[key]["ours"]["cycles_per_sample"], lib[key]["comparison"]["cycles_per_sample"]
        return {"cpu": v[key]["cpu"], "ours": {"cycles_per_sample": o},
                "comparison": {"cycles_per_sample": c}, "ratio": o / c}
    out = {"compared_with": lib.get("compared_with", ""), "reference_model": m("reference_model"),
           "counter_model": m("counter_model"),
           "own": (v["reference_model"]["ratio"], v["counter_model"]["ratio"])}
    snr, lsnr = v.get("snr_db"), lib.get("snr_db")
    if snr and lsnr:
        out["snr_db"] = {"ours": snr["ours"], "comparison": lsnr.get("comparison", lsnr.get("other"))}
    return out


def render(platform, level, lv, trades=frozenset(), library=None):
    """The text lines of one level of a results document; None when it has no entry. `library`
    (the entries of the `avx2` level) makes the table of `avx2-runtime` (D2764)."""
    entries = lv.get("entries", {})
    if not entries:
        return None
    runtime = library is not None
    ref_cpu, cnt_cpu = lv["reference_model"], lv["counter_model"]
    counts = lv.get("block_counts") or "the counter of the level"
    out = [f"#### Simulated cycles at `{level}` (`{platform}`)", "",
           f"Reference model `{ref_cpu}`, counter-model `{cnt_cpu}`, mode `{lv['cycle_mode']}`; "
           f"execution counts: {counts}.", ""]
    if runtime:
        out += [f"Judged like every level against the reference library: the comparison is the one of the "
                f"`{RUNTIME_LIBRARY}` record of the entry, its cycles and SNR on the same workload and models (D2764). "
                f"The column `ours / {RUNTIME_LIBRARY} build` is ours against our own fixed `{RUNTIME_LIBRARY}` build, "
                f"information only.", ""]
    extra_head, extra_rule = (f" ours / {RUNTIME_LIBRARY} build ({ref_cpu} / {cnt_cpu}, information) |", " ---: |") \
        if runtime else ("", "")
    out += [f"| Entry | ours {ref_cpu} (cycles/sample) | comparison {ref_cpu} (cycles/sample) "
            f"| ratio {ref_cpu} (ours/comparison) | ours {cnt_cpu} (cycles/sample) "
            f"| comparison {cnt_cpu} (cycles/sample) | ratio {cnt_cpu} (ours/comparison) "
            f"| SNR (dB, ours / comparison) |{extra_head} comparison |",
            f"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |{extra_rule} --- |"]
    failed, skipped = [], []
    for name in sorted(entries, key=order):
        v = entries[name]
        if "reference_model" not in v:
            note = v.get("note") or "not simulated"
            (skipped if note.startswith(NOT_MEASURED) else failed).append((name, note))
            continue
        extra = ""
        if runtime:
            own = f"{fmt(v['reference_model']['ratio'])} / {fmt(v['counter_model']['ratio'])}"
            judged = against_library(v, library.get(name))
            if judged is None:
                failed.append((name, f"no `{RUNTIME_LIBRARY}` record with the comparison; "
                                     f"ours / {RUNTIME_LIBRARY} build {own}"))
                continue
            v, extra = judged, f" {own} |"
        r, c = v["reference_model"], v["counter_model"]
        comp = v.get("compared_with", "").replace("|", "\\|")
        out.append(f"| `{name}` | {fmt(r['ours']['cycles_per_sample'])} "
                   f"| {fmt(r['comparison']['cycles_per_sample'])} | {fmt(r['ratio'])} "
                   f"| {fmt(c['ours']['cycles_per_sample'])} "
                   f"| {fmt(c['comparison']['cycles_per_sample'])} | {fmt(c['ratio'])} "
                   f"| {accuracy(v.get('snr_db'), (name, level) in trades)} |{extra} {comp} |")
    out.append("")
    for name, note in failed:
        out.append(f"- `{name}`: {note}.")
    if skipped:
        out.append("- Not measured at this level: " + ", ".join(f"`{n}`" for n, _ in skipped) + ".")
    if failed or skipped:
        out.append("")
    return out


def trades_of(limits):
    """The (entry, level) pairs of `benches/limits.json` with an `accuracy_trade`."""
    return frozenset((k, l) for k, levels in limits.items() if not k.startswith("_")
                     for l, e in levels.items() if isinstance(e, dict) and "accuracy_trade" in e)


def block(docs, trades=frozenset()):
    """The whole generated block from the results documents, levels in the order of LEVELS (others
    after them, sorted)."""
    present = [(p, l) for p, d in docs.items() for l in d.get("levels", {})]
    keys = [k for k in LEVELS if k in present] + sorted(k for k in present if k not in LEVELS)
    text = []
    for platform, level in keys:
        levels = docs[platform]["levels"]
        library = levels.get(RUNTIME_LIBRARY, {}).get("entries", {}) if level == RUNTIME else None
        lines = render(platform, level, levels[level], trades, library)
        if lines:
            text += lines
    return "\n".join(text).rstrip("\n")


def simd128_block(counts):
    """The table of `simd128` instructions per sample from the results document of wasm32."""
    kernels = counts["levels"]["simd128"]["kernels"]
    out = ["| Entry | ours (instructions/sample) | comparison (instructions/sample) | ratio (ours/comparison) "
           "| SNR (dB, ours / comparison) | status | comparison |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for name in sorted(kernels, key=order):
        k = kernels[name]
        status = k["status"] if k["status"] != "release" else ""
        verdict = {"waived": "waived", "fail": "fail"}.get(k["verdict"], "")
        mark = " ".join(x for x in (status, verdict) if x)
        out.append(f"| `{name}` | {fmt(k['ours_instr_per_sample'])} | {fmt(k['other_instr_per_sample'])} "
                   f"| {fmt(k['ratio'])} | {accuracy(k.get('snr_db'))} | {mark} | {k['compared_with'].replace('|', chr(92) + '|')} |")
    return "\n".join(out)


SECTION = re.compile(r"^#### Simulated cycles at `([^`]+)` \(`([^`]+)`\)$")
ROW = re.compile(r"^\| `([^`]+)` \|")


def rows_of(text):
    """The lines of a block by (level, entry) for the table rows and by (level, line) for the rest."""
    out, level = {}, None
    for line in text.split("\n"):
        m = SECTION.match(line)
        if m:
            level = m.group(1)
        r = ROW.match(line)
        out[(level, r.group(1)) if r else (level, line)] = line
    return out


def differences(old, new):
    """The rows and lines that differ between two blocks: [(level, entry or line, old, new)]."""
    a, b = rows_of(old), rows_of(new)
    return [(k[0], k[1], a.get(k), b.get(k)) for k in sorted(set(a) | set(b), key=str)
            if a.get(k) != b.get(k)]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    doc_path, results_dir, check, counts_path, limits_path = DOC, RESULTS, False, None, LIMITS
    while args:
        a = args.pop(0)
        if a == "--check":
            check = True
        elif a == "--doc" and args:
            doc_path = args.pop(0)
        elif a == "--results" and args:
            results_dir = args.pop(0)
        elif a == "--counts" and args:
            counts_path = args.pop(0)
        elif a == "--limits" and args:
            limits_path = args.pop(0)
        else:
            sys.exit(f"unknown argument {a!r}; usage: cycle_tables.py [--check] [--doc FILE] "
                     f"[--results DIR] [--counts FILE] [--limits FILE]")
    if counts_path is None:
        counts_path = os.path.join(os.path.dirname(os.path.abspath(results_dir)), "wasm32.json")
    trades = frozenset()
    if os.path.exists(limits_path):
        with open(limits_path) as fh:
            trades = trades_of(json.load(fh))
    blocks = [(BEGIN, END, block(load(results_dir), trades))]
    if os.path.exists(counts_path):
        with open(counts_path) as fh:
            blocks.append((SIMD_BEGIN, SIMD_END, simd128_block(json.load(fh))))
    with open(doc_path) as fh:
        doc = fh.read()
    bad = 0
    for begin, end, text in blocks:
        a, b = doc.index(begin) + len(begin), doc.index(end)
        old = doc[a:b].strip("\n")
        if check:
            if old != text:
                diff = differences(old, text)
                for level, key, x, y in diff[:20]:
                    print(f"{level} {key}:\n  in the document: {x}\n  from the results: {y}", file=sys.stderr)
                bad += len(diff)
            continue
        doc = doc[:a] + "\n\n" + text + "\n\n" + doc[b:]
        rows = sum(1 for t in text.split("\n") if ROW.match(t))
        print(f"tables: {rows} rows between {begin} and {end} of {doc_path}", file=sys.stderr)
    if check:
        if bad:
            sys.exit(f"cycle tables: {bad} lines of {doc_path} differ from {results_dir} and {counts_path}; "
                     f"run python3 -I benches/cycle_tables.py")
        print(f"cycle tables: {doc_path} matches {results_dir} and {counts_path}", file=sys.stderr)
        return
    with open(doc_path, "w") as fh:
        fh.write(doc)


if __name__ == "__main__":
    main()
