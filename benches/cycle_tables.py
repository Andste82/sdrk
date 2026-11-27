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
the status of `benches/limits.json` (`--limits FILE` takes another file) and the comparison. An entry
with an `accuracy_trade` (D2760) is marked `(accuracy trade)` in the accuracy column, one with a
`contract_trade` (D3000) `(contract trade)`. A comparison
with our own code (`comparison_kind` other than `library`: our fallback, our build of another level,
our kernel) is marked `own (information)` in the comparison column and is never `release` or `dev`
(D2764, D2767): such a row, a simulated row without an entry in `benches/limits.json` and a record
without `comparison_kind` fail the generation. An entry that could not be simulated is listed below
its table with its note. Only the ratios mean something (D33, D68); the absolute cycles are no
hardware figures.

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
cycle results directory). It is written only when that file exists. Its status column is the entry
of `benches/limits.json`; `--check` also fails when a record's `status` differs from it (run
`cargo xtask count --platform wasm32 --rejudge`).

`page_rows` gives the rows of the standalone page of the cycles with the same rules (status from
`benches/limits.json`, own comparisons marked and never judged, no row dropped silently).
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
# The comparison kinds of the records (D2767); every kind but `library` is our own code.
LIBRARY = "library"
KINDS = {LIBRARY, "own_fallback", "own_build", "own_kernel"}
OWN = "own (information)"
JUDGED = {"release", "dev"}


class TableError(Exception):
    """A row that the tables refuse: no limit, no comparison kind, a status against our own code."""


def status_of(limits, name, level):
    """The status of `name` at `level` in `benches/limits.json`; a missing entry is an error."""
    entry = limits.get(name, {}).get(level)
    if not isinstance(entry, dict) or "status" not in entry:
        raise TableError(f"`{name}` at `{level}` has no entry in benches/limits.json")
    return entry["status"]


def kind_of(v, name, level):
    """The `comparison_kind` of a record; a missing or unknown kind is an error."""
    kind = v.get("comparison_kind")
    if kind not in KINDS:
        raise TableError(f"`{name}` at `{level}`: the record has no valid `comparison_kind` ({kind!r}, D2767)")
    return kind


def judged_kind(status, kind, name, level):
    """The kind, after the rule that no status is decided by our own code (D2764, D2767)."""
    if status in JUDGED and kind != LIBRARY:
        raise TableError(f"`{name}` at `{level}` is `{status}` against our own code (`{kind}`): "
                         f"a comparison with our own code is information only (D2764, D2767)")
    return kind


def comparison_cell(name, kind):
    """The comparison column: the name, marked `own (information)` for our own code."""
    name = name.replace("|", "\\|")
    return name if kind == LIBRARY else f"{OWN}: {name}"


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


def accuracy(snr, trade=None):
    """The accuracy cell: `ours / comparison` in dB, `-` without an SNR; `trade` is the kind of
    trade of the entry (`accuracy trade`, `contract trade`) or `None`."""
    if not snr:
        return "-"
    cell = f"{db(snr['ours'])} / {db(snr.get('comparison', snr.get('other')))}"
    return f"{cell} ({trade})" if trade else cell


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
    out = {"compared_with": lib.get("compared_with", ""), "comparison_kind": lib.get("comparison_kind"),
           "reference_model": m("reference_model"),
           "counter_model": m("counter_model"),
           "own": (v["reference_model"]["ratio"], v["counter_model"]["ratio"])}
    snr, lsnr = v.get("snr_db"), lib.get("snr_db")
    if snr and lsnr:
        out["snr_db"] = {"ours": snr["ours"], "comparison": lsnr.get("comparison", lsnr.get("other"))}
    return out


def render(platform, level, lv, limits, library=None):
    """The text lines of one level of a results document; None when it has no entry. `library`
    (the entries of the `avx2` level) makes the table of `avx2-runtime` (D2764). The status comes
    from `limits` (benches/limits.json)."""
    trades = trades_of(limits)
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
            f"| SNR (dB, ours / comparison) |{extra_head} status | comparison |",
            f"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |{extra_rule} --- | --- |"]
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
        status = status_of(limits, name, level)
        kind = judged_kind(status, kind_of(v, name, level), name, level)
        r, c = v["reference_model"], v["counter_model"]
        comp = comparison_cell(v.get("compared_with", ""), kind)
        out.append(f"| `{name}` | {fmt(r['ours']['cycles_per_sample'])} "
                   f"| {fmt(r['comparison']['cycles_per_sample'])} | {fmt(r['ratio'])} "
                   f"| {fmt(c['ours']['cycles_per_sample'])} "
                   f"| {fmt(c['comparison']['cycles_per_sample'])} | {fmt(c['ratio'])} "
                   f"| {accuracy(v.get('snr_db'), trades.get((name, level)))} |{extra} {status} | {comp} |")
    out.append("")
    for name, note in failed:
        out.append(f"- `{name}`: {note}.")
    if skipped:
        out.append("- Not measured at this level: " + ", ".join(f"`{n}`" for n, _ in skipped) + ".")
    if failed or skipped:
        out.append("")
    return out


def trades_of(limits):
    """The (entry, level) pairs of `benches/limits.json` with an `accuracy_trade` (D2760) or a
    `contract_trade` (D3000), each with the kind of its trade."""
    out = {}
    for k, levels in limits.items():
        if k.startswith("_"):
            continue
        for l, e in levels.items():
            if not isinstance(e, dict):
                continue
            if "accuracy_trade" in e:
                out[(k, l)] = "accuracy trade"
            elif "contract_trade" in e:
                out[(k, l)] = "contract trade"
    return out


def block(docs, limits):
    """The whole generated block from the results documents and benches/limits.json, levels in the
    order of LEVELS (others after them, sorted)."""
    present = [(p, l) for p, d in docs.items() for l in d.get("levels", {})]
    keys = [k for k in LEVELS if k in present] + sorted(k for k in present if k not in LEVELS)
    text = []
    for platform, level in keys:
        levels = docs[platform]["levels"]
        library = levels.get(RUNTIME_LIBRARY, {}).get("entries", {}) if level == RUNTIME else None
        lines = render(platform, level, levels[level], limits, library)
        if lines:
            text += lines
    return "\n".join(text).rstrip("\n")


def simd128_block(counts, limits):
    """The table of `simd128` instructions per sample from the results document of wasm32, with the
    status of `limits` (benches/limits.json) and the verdict of the record."""
    kernels = counts["levels"]["simd128"]["kernels"]
    out = ["| Entry | ours (instructions/sample) | comparison (instructions/sample) | ratio (ours/comparison) "
           "| SNR (dB, ours / comparison) | status | comparison |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for name in sorted(kernels, key=order):
        k = kernels[name]
        status = status_of(limits, name, "simd128")
        kind = judged_kind(status, kind_of(k, name, "simd128"), name, "simd128")
        verdict = {"waived": "waived", "fail": "fail"}.get(k["verdict"], "")
        mark = " ".join(x for x in (status, verdict) if x)
        out.append(f"| `{name}` | {fmt(k['ours_instr_per_sample'])} | {fmt(k['other_instr_per_sample'])} "
                   f"| {fmt(k['ratio'])} | {accuracy(k.get('snr_db'))} | {mark} | {comparison_cell(k['compared_with'], kind)} |")
    return "\n".join(out)


def stale_statuses(counts, limits):
    """The records of the counts whose `status` differs from benches/limits.json: [(name, record, limits)]."""
    kernels = counts["levels"]["simd128"]["kernels"]
    out = []
    for name in sorted(kernels):
        entry = limits.get(name, {}).get("simd128", {})
        if kernels[name].get("status") != entry.get("status"):
            out.append((name, kernels[name].get("status"), entry.get("status")))
    return out


def page_rows(docs, limits):
    """The rows of the standalone page of the cycles, and the notes of the rows that have no
    numbers (nothing is dropped silently). A row: level, entry, status, comparison kind, `judged`
    (release or dev against a library), ours and the comparison on both models, the ratios, the
    comparison, the SNR, the accuracy trade, the ratios to our own `avx2` build at `avx2-runtime`
    and the cycles of the entry's fallback. A simulated row without a limit or a comparison kind,
    or with a status against our own code, raises TableError."""
    trades = trades_of(limits)
    rows, notes = [], []
    for platform, d in docs.items():
        levels = d.get("levels", {})
        for level, lv in levels.items():
            ents = lv.get("entries", {})
            library = levels.get(RUNTIME_LIBRARY, {}).get("entries", {}) if level == RUNTIME else None
            for name, v in ents.items():
                if "reference_model" not in v:
                    notes.append((level, name, v.get("note") or "not simulated"))
                    continue
                own = None
                if library is not None:
                    judged = against_library(v, library.get(name))
                    if judged is None:
                        notes.append((level, name, f"no `{RUNTIME_LIBRARY}` record with the comparison"))
                        continue
                    own, v = judged["own"], judged
                status = status_of(limits, name, level)
                kind = judged_kind(status, kind_of(v, name, level), name, level)
                fb = None if name.endswith("_fallback") else ents.get(name + "_fallback")
                fb = fb if fb and "reference_model" in fb else None
                r, c = v["reference_model"], v["counter_model"]
                snr = v.get("snr_db") or {}
                rows.append({
                    "level": level, "name": name, "status": status, "kind": kind,
                    "judged": status in JUDGED,
                    "ours": (r["ours"]["cycles_per_sample"], c["ours"]["cycles_per_sample"]),
                    "comparison": (r["comparison"]["cycles_per_sample"], c["comparison"]["cycles_per_sample"]),
                    "ratio": (r["ratio"], c["ratio"]),
                    "compared_with": v.get("compared_with", ""),
                    "snr": (snr.get("ours"), snr.get("comparison", snr.get("other"))),
                    "trade": (name, level) in trades,
                    "own_build": own,
                    "fallback": (fb["reference_model"]["ours"]["cycles_per_sample"],
                                 fb["counter_model"]["ours"]["cycles_per_sample"]) if fb else None,
                })
    return rows, notes


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
    with open(limits_path) as fh:
        limits = json.load(fh)
    stale = []
    try:
        blocks = [(BEGIN, END, block(load(results_dir), limits))]
        if os.path.exists(counts_path):
            with open(counts_path) as fh:
                counts = json.load(fh)
            blocks.append((SIMD_BEGIN, SIMD_END, simd128_block(counts, limits)))
            stale = stale_statuses(counts, limits)
    except TableError as e:
        sys.exit(f"cycle tables: {e}")
    if check and stale:
        for name, rec, lim in stale:
            print(f"simd128 {name}: status {rec!r} in {counts_path}, {lim!r} in {limits_path}", file=sys.stderr)
        sys.exit(f"cycle tables: {len(stale)} records of {counts_path} have another status than {limits_path}; "
                 f"run cargo xtask count --platform wasm32 --rejudge")
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
