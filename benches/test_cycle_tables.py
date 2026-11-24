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


def record(ours, other, comparison="a library", snr=(120.04, 999.0)):
    return {"compared_with": comparison, "samples": 4096, "cycle_mode": "whole-loop",
            "snr_db": {"ours": snr[0], "comparison": snr[1]},
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
        self.assertIn("| `mag_sq` | 3.00 | 4.00 | 0.75 | 3.00 | 4.00 | 0.75 | 120.0 / exact | a library |", text)
        self.assertIn("| SNR (dB, ours / comparison) | comparison |", text)
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

    def test_an_accuracy_trade_is_marked_and_a_record_without_snr_shows_a_dash(self):
        entries = {"mul_real_q15_fallback": record(1.3, 1.0, snr=(91.62, 85.59)), "mag_sq": record(1.0, 1.0)}
        del entries["mag_sq"]["snr_db"]
        text = ct.block({"x86_64-avx512": results(entries)},
                        ct.trades_of({"_limits": {}, "mul_real_q15_fallback": {
                            "avx512": {"status": "release", "accuracy_trade": "91.6 dB against 85.6 dB"},
                            "avx2": {"status": "release"}}}))
        self.assertIn("| 91.6 / 85.6 (accuracy trade) | a library |", text)
        self.assertIn("| 1.00 | - | a library |", text)
        self.assertNotIn("(accuracy trade)", ct.block({"x86_64-avx512": results(entries)}))

    def test_levels_come_in_the_order_of_the_policy(self):
        docs = {"aarch64": results({"a": record(1.0, 1.0)}, "neon", "aarch64"),
                "x86_64": {"platform": "x86_64", "levels": {
                    "avx2-runtime": results({"a": record(1.0, 1.0)})["levels"]["avx512"],
                    "sse2": results({"a": record(1.0, 1.0)})["levels"]["avx512"]}}}
        heads = [l for l in ct.block(docs).split("\n") if l.startswith("####")]
        self.assertEqual(heads, ["#### Simulated cycles at `sse2` (`x86_64`)",
                                 "#### Simulated cycles at `avx2-runtime` (`x86_64`)",
                                 "#### Simulated cycles at `neon` (`aarch64`)"])

    def test_the_run_time_level_compares_with_the_library_of_the_avx2_record(self):
        avx2 = results({"agc": record(13.0, 36.0, "GNU Radio agc_cc", (119.85, 118.88)),
                        "dot_cq15": record(0.3, 1.0, "sdrk fallback")}, "avx2", "x86_64")
        rt = results({"agc": record(16.0, 13.0, "ours at avx2", (119.85, 119.85)),
                      "dot_cq15": record(0.31, 0.3, "ours at avx2"),
                      "lost": record(1.0, 1.0, "ours at avx2")}, "avx2", "x86_64")
        avx2["levels"]["avx2-runtime"] = rt["levels"]["avx2"]
        text = ct.block({"x86_64": avx2})
        part = text[text.index("`avx2-runtime`"):]
        # Ours of the run-time build against the library's cycles and SNR of the avx2 record; the
        # ratio to our avx2 build (16 / 13) is the information column.
        self.assertIn("| `agc` | 16.00 | 36.00 | 0.44 | 16.00 | 36.00 | 0.44 | 119.8 / 118.9 | 1.23 / 1.23 "
                      "| GNU Radio agc_cc |", part)
        self.assertIn("| `dot_cq15` | 0.31 | 1.00 | 0.31 | 0.31 | 1.00 | 0.31 | 120.0 / exact | 1.03 / 1.03 "
                      "| sdrk fallback |", part)
        self.assertIn("| SNR (dB, ours / comparison) | ours / avx2 build (znver4 / sapphirerapids, information) "
                      "| comparison |", part)
        self.assertIn("(D2764)", part)
        self.assertIn("- `lost`: no `avx2` record with the comparison; ours / avx2 build 1.00 / 1.00.", part)
        # The avx2 table itself is unchanged.
        avx2_part = text[:text.index("`avx2-runtime`")]
        self.assertIn("| `agc` | 13.00 | 36.00 | 0.36 | 13.00 | 36.00 | 0.36 | 119.8 / 118.9 | GNU Radio agc_cc |",
                      avx2_part)

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


class Simd128Test(unittest.TestCase):
    def test_rows_status_and_verdict(self):
        def k(ours, other, status="release", verdict="pass", what="VOLK | generic"):
            return {"ours_instr_per_sample": ours, "other_instr_per_sample": other, "ratio": ours / other,
                    "status": status, "verdict": verdict, "compared_with": what,
                    "snr_db": {"ours": 150.98, "other": 999.0}}
        text = ct.simd128_block({"levels": {"simd128": {"kernels": {
            "mag_sq": k(1.0, 2.0), "convert_cs8": k(3.0, 4.0, "info", "info"),
            "mul": k(5.0, 4.0, "release", "waived")}}}})
        lines = text.split("\n")
        self.assertEqual([l.split("`")[1] for l in lines if ct.ROW.match(l)], ["convert_cs8", "mag_sq", "mul"])
        self.assertIn("| `convert_cs8` | 3.00 | 4.00 | 0.75 | 151.0 / exact | info | VOLK \\| generic |", text)
        self.assertIn("| `mag_sq` | 1.00 | 2.00 | 0.50 | 151.0 / exact |  | VOLK \\| generic |", text)
        self.assertIn("| 1.25 | 151.0 / exact | waived |", text)


class CommittedBlockTest(unittest.TestCase):
    """The block of docs/performance.md is what the committed results give, byte for byte."""

    def test_the_committed_tables_are_generated(self):
        with open(ct.DOC) as fh:
            doc = fh.read()
        block = doc[doc.index(ct.BEGIN) + len(ct.BEGIN):doc.index(ct.END)].strip("\n")
        with open(ct.LIMITS) as fh:
            trades = ct.trades_of(json.load(fh))
        self.assertEqual(ct.differences(block, ct.block(ct.load(ct.RESULTS), trades)), [])

    def test_the_committed_simd128_table_is_generated(self):
        with open(ct.DOC) as fh:
            doc = fh.read()
        block = doc[doc.index(ct.SIMD_BEGIN) + len(ct.SIMD_BEGIN):doc.index(ct.SIMD_END)].strip("\n")
        with open(os.path.join(os.path.dirname(ct.RESULTS), "wasm32.json")) as fh:
            self.assertEqual(block, ct.simd128_block(json.load(fh)))


if __name__ == "__main__":
    unittest.main()
