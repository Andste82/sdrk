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

COUNTS = "Intel SDE -dcfg, block and branch counts (Intel SDE -spr)"


def model(cpu, ours, other):
    return {"cpu": cpu, "ours": {"cycles_per_sample": ours}, "comparison": {"cycles_per_sample": other},
            "ratio": ours / other}


def record(ours, other, comparison="a library"):
    return {"compared_with": comparison, "samples": 4096, "cycle_mode": "whole-loop",
            "reference_model": model("znver4", ours, other),
            "counter_model": model("sapphirerapids", ours, other)}


def results(entries, level="avx512", platform="x86_64-avx512"):
    return {"platform": platform, "levels": {level: {
        "reference_model": "znver4", "counter_model": "sapphirerapids", "cycle_mode": "whole-loop",
        "block_counts": COUNTS, "entries": entries}}}


def doc_with_block(block):
    return f"# Title\n\ntext\n\n{ct.BEGIN}\n\n{block}\n\n{ct.END}\n\ntail\n"


def rows(text):
    return [l.split("`")[1] for l in text.split("\n") if ct.ROW.match(l)]


class GenerateTest(unittest.TestCase):
    def test_rows_follow_the_order_of_the_results_tables(self):
        text = ct.block({"x86_64-avx512": results({
            "zzz_new": record(7.0, 8.0), "fft_64": record(5.0, 6.0), "mag_sq": record(3.0, 4.0),
            "convert_cs8": record(1.0, 2.0)})})
        self.assertEqual(rows(text), ["convert_cs8", "mag_sq", "fft_64", "zzz_new"])
        self.assertIn("| `mag_sq` | 3.00 | 4.00 | 0.75 | 3.00 | 4.00 | 0.75 | a library |", text)
        self.assertIn(f"Reference model `znver4`, counter-model `sapphirerapids`, mode `whole-loop`; "
                      f"execution counts: {COUNTS}.", text)

    def test_notes_are_listed_below_the_table(self):
        text = ct.block({"x86_64-avx512": results({
            "mag_sq": record(3.0, 4.0, "x | y"),
            "airspy": {"note": "not simulated: not measured at the level avx512"},
            "broken": {"note": "not simulated: the run failed"}})})
        self.assertEqual(rows(text), ["mag_sq"])
        self.assertIn("x \\| y |", text)
        self.assertIn("- `broken`: not simulated: the run failed.", text)
        self.assertIn("- Not measured at this level: `airspy`.", text)

    def test_levels_come_in_the_order_of_the_policy(self):
        docs = {"aarch64": results({"a": record(1.0, 1.0)}, "neon", "aarch64"),
                "x86_64": {"platform": "x86_64", "levels": {
                    "avx2-runtime": results({"a": record(1.0, 1.0)})["levels"]["avx512"],
                    "sse2": results({"a": record(1.0, 1.0)})["levels"]["avx512"]}}}
        heads = [l for l in ct.block(docs).split("\n") if l.startswith("####")]
        self.assertEqual(heads, ["#### Simulated cycles at `sse2` (`x86_64`)",
                                 "#### Simulated cycles at `avx2-runtime` (`x86_64`)",
                                 "#### Simulated cycles at `neon` (`aarch64`)"])

    def test_main_writes_the_block_and_check_finds_a_hand_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = os.path.join(tmp, "cycles")
            os.mkdir(res)
            with open(os.path.join(res, "x86_64-avx512.json"), "w") as fh:
                json.dump(results({"mag_sq": record(3.0, 4.0)}), fh)
            doc = os.path.join(tmp, "performance.md")
            with open(doc, "w") as fh:
                fh.write(doc_with_block("| `old` | 1 |"))
            with contextlib.redirect_stderr(io.StringIO()):
                ct.main(["--doc", doc, "--results", res])
                ct.main(["--check", "--doc", doc, "--results", res])
            with open(doc) as fh:
                text = fh.read()
            self.assertTrue(text.startswith("# Title\n\ntext\n\n" + ct.BEGIN))
            self.assertTrue(text.endswith(ct.END + "\n\ntail\n"))
            self.assertEqual(rows(text), ["mag_sq"])
            with open(doc, "w") as fh:
                fh.write(text.replace("| 3.00 |", "| 2.99 |", 1))
            with self.assertRaises(SystemExit) as failed, contextlib.redirect_stderr(io.StringIO()) as err:
                ct.main(["--check", "--doc", doc, "--results", res])
            self.assertIn("1 lines", str(failed.exception))
            self.assertIn("avx512 mag_sq", err.getvalue())


class CommittedBlockTest(unittest.TestCase):
    """The block of docs/performance.md is what the committed results give, byte for byte."""

    def test_the_committed_tables_are_generated(self):
        with open(ct.DOC) as fh:
            doc = fh.read()
        block = doc[doc.index(ct.BEGIN) + len(ct.BEGIN):doc.index(ct.END)].strip("\n")
        self.assertEqual(ct.differences(block, ct.block(ct.load(ct.RESULTS))), [])


if __name__ == "__main__":
    unittest.main()
