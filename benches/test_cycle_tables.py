"""Tests of benches/cycle_tables.py: python3 -I benches/test_cycle_tables.py"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle_tables as ct  # noqa: E402

COUNTS = "Intel SDE (`-hsw`)"


def model(cpu, ours, other):
    return {"ours": {"cpu": cpu, "cycles_per_sample": ours},
            "other": {"cpu": cpu, "cycles_per_sample": other}, "ratio": ours / other}


def entry(ours, other, comparison="a library", counts=COUNTS):
    return {"reference_model": model("znver3", ours, other), "counter_model": model("alderlake", ours, other),
            "compared_with": comparison, "cycle_mode": "whole-loop", "block_counts": counts}


def document(entries, level="avx2", platform="x86_64"):
    return {"cycles": {platform: {level: entries}}}


def write_docs(directory, docs):
    for i, d in enumerate(docs):
        with open(os.path.join(directory, f"k{i}.json"), "w") as fh:
            json.dump(d, fh)


def doc_with_block(block):
    return f"# Title\n\ntext\n\n{ct.BEGIN}\n\n{block}\n\n{ct.END}\n\ntail\n"


def rows(text):
    return [l.split("`")[1] for l in text.split("\n") if ct.ROW.match(l)]


class MergeTest(unittest.TestCase):
    def build(self, entries, level="avx2"):
        levels = ct.collect(ct.documents_from([document(entries, level)]))
        return ct.rebuild(levels)

    def test_merge_replaces_and_inserts_and_keeps_the_rest(self):
        base = self.build({"convert_cs8": entry(1.0, 2.0), "mag_sq": entry(3.0, 4.0), "fft_64": entry(5.0, 6.0)})
        levels = ct.collect(ct.documents_from([document({
            "mag_sq": entry(3.5, 4.0), "zzz_new": entry(7.0, 8.0), "fir": entry(9.0, 10.0)})]))
        text, replaced, inserted, warnings = ct.merge(base, levels)
        self.assertEqual((replaced, inserted, warnings), (1, 2, []))
        # order of the results tables: convert, fir, mag, fft, then the unknown prefix at the end
        self.assertEqual(rows(text), ["convert_cs8", "fir", "mag_sq", "fft_64", "zzz_new"])
        old = {l.split("`")[1]: l for l in base.split("\n") if ct.ROW.match(l)}
        new = {l.split("`")[1]: l for l in text.split("\n") if ct.ROW.match(l)}
        self.assertEqual(new["convert_cs8"], old["convert_cs8"])
        self.assertEqual(new["fft_64"], old["fft_64"])
        self.assertNotEqual(new["mag_sq"], old["mag_sq"])
        self.assertIn("3.50", new["mag_sq"])

    def test_merge_is_idempotent(self):
        base = self.build({"convert_cs8": entry(1.0, 2.0), "mag_sq": entry(3.0, 4.0)})
        levels = ct.collect(ct.documents_from([document({"mag_sq": entry(3.5, 4.0), "zzz_new": entry(7.0, 8.0)})]))
        once = ct.merge(base, levels)[0]
        twice = ct.merge(once, levels)[0]
        self.assertEqual(once, twice)

    def test_levels_that_are_not_given_stay_byte_for_byte(self):
        a = self.build({"convert_cs8": entry(1.0, 2.0)}, "avx2")
        levels = ct.collect(ct.documents_from([document({"mag_sq": entry(3.0, 4.0)}, "sse2")]))
        b = ct.rebuild(levels)
        both = a + "\n\n" + b
        # a block with the avx2 and the sse2 section, merged with a new avx2 entry: sse2 untouched
        text = ct.merge(both, ct.collect(ct.documents_from([document({"fft_64": entry(5.0, 6.0)}, "avx2")])))[0]
        sse2_old = ct.parse_block(both)[("x86_64", "sse2")]["lines"]
        sse2_new = ct.parse_block(text)[("x86_64", "sse2")]["lines"]
        self.assertEqual(sse2_old, sse2_new)

    def test_a_note_replaces_a_row_and_a_new_row_removes_the_note(self):
        base = self.build({"mag_sq": entry(3.0, 4.0)})
        noted = ct.collect(ct.documents_from([document({"mag_sq": {"note": "not simulated: no counter"}})]))
        text = ct.merge(base, noted)[0]
        self.assertEqual(rows(text), [])
        self.assertIn("- `mag_sq`: not simulated: no counter.", text)
        back = ct.collect(ct.documents_from([document({"mag_sq": entry(3.0, 4.0)})]))
        text2 = ct.merge(text, back)[0]
        self.assertEqual(rows(text2), ["mag_sq"])
        self.assertNotIn("not simulated", text2)

    def test_other_models_are_reported_and_kept(self):
        base = self.build({"mag_sq": entry(3.0, 4.0)})
        other = {"reference_model": dict(model("skylake", 1.0, 2.0)), "counter_model": model("znver2", 1.0, 2.0),
                 "compared_with": "x", "block_counts": COUNTS}
        other["reference_model"]["ours"]["cpu"] = "skylake"
        text, _, _, warnings = ct.merge(base, ct.collect(ct.documents_from([document({"fft_64": other})])))
        self.assertEqual(len(warnings), 1)
        self.assertIn("Reference model `znver3`", text)

    def test_rebuild_refuses_to_drop_entries_unless_forced(self):
        base = self.build({"convert_cs8": entry(1.0, 2.0), "mag_sq": entry(3.0, 4.0)})
        with tempfile.TemporaryDirectory() as tmp:
            doc = os.path.join(tmp, "performance.md")
            with open(doc, "w") as fh:
                fh.write(doc_with_block(base))
            write_docs(tmp, [document({"mag_sq": entry(3.0, 4.0)})])
            with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
                ct.main(["--rebuild", "--doc", doc, os.path.join(tmp, "k0.json")])
            self.assertIn("would drop 1 entries", str(refused.exception))
            with open(doc) as fh:
                self.assertEqual(rows(fh.read()), ["convert_cs8", "mag_sq"])
            with contextlib.redirect_stderr(io.StringIO()):
                ct.main(["--rebuild", "--force", "--doc", doc, os.path.join(tmp, "k0.json")])
            with open(doc) as fh:
                self.assertEqual(rows(fh.read()), ["mag_sq"])

    def test_main_merges_into_a_copy_and_leaves_the_rest_of_the_file_alone(self):
        base = self.build({"convert_cs8": entry(1.0, 2.0)})
        with tempfile.TemporaryDirectory() as tmp:
            doc = os.path.join(tmp, "performance.md")
            with open(doc, "w") as fh:
                fh.write(doc_with_block(base))
            write_docs(tmp, [document({"mag_sq": entry(3.0, 4.0)})])
            with contextlib.redirect_stderr(io.StringIO()) as err:
                ct.main(["--doc", doc, os.path.join(tmp, "k0.json")])
            self.assertIn("merged: 0 rows replaced, 1 inserted", err.getvalue())
            with open(doc) as fh:
                text = fh.read()
            self.assertTrue(text.startswith("# Title\n\ntext\n\n" + ct.BEGIN))
            self.assertTrue(text.endswith(ct.END + "\n\ntail\n"))
            self.assertEqual(rows(text), ["convert_cs8", "mag_sq"])

    def test_force_without_rebuild_is_an_error(self):
        with self.assertRaises(SystemExit):
            ct.main(["--force"])


class CommittedBlockTest(unittest.TestCase):
    """The parser and the renderer reproduce the block of docs/performance.md byte for byte."""

    def test_round_trip_of_the_committed_tables(self):
        with open(ct.DOC) as fh:
            doc = fh.read()
        if ct.BEGIN not in doc:
            self.skipTest("no cycle tables in this checkout")
        block = doc[doc.index(ct.BEGIN) + len(ct.BEGIN):doc.index(ct.END)].strip("\n")
        sections = ct.parse_block(block)
        self.assertGreaterEqual(len(sections), 1)
        for (platform, level), sec in sections.items():
            again = ct.render(platform, level, sec)
            self.assertEqual("\n".join(again).rstrip("\n"), "\n".join(sec["lines"]).rstrip("\n"),
                             f"{level} ({platform})")
        self.assertEqual(ct.merge(block, {})[0], block)


if __name__ == "__main__":
    unittest.main()
