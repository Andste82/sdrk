"""Tests of benches/cycles_page.py and benches/verify_cycles_page.py: python3 -I benches/test_cycles_page.py"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycles_page as cp  # noqa: E402
import verify_cycles_page as vp  # noqa: E402


def model(cpu, ours, other):
    return {"cpu": cpu, "ours": {"cycles_per_sample": ours}, "comparison": {"cycles_per_sample": other},
            "ratio": ours / other}


def record(ours, other, comparison="lib 1.0 kernel", snr=(120.04, 999.0), kind="library"):
    """A cycle record; `ours` and `other` are (reference, counter) or one number for both."""
    o = ours if isinstance(ours, tuple) else (ours, ours)
    c = other if isinstance(other, tuple) else (other, other)
    return {"compared_with": comparison, "comparison_kind": kind, "samples": 4096,
            "snr_db": {"ours": snr[0], "comparison": snr[1]},
            "reference_model": model("znver3", o[0], c[0]), "counter_model": model("alderlake", o[1], c[1])}


def level(entries):
    return {"reference_model": "znver3", "counter_model": "alderlake", "cycle_mode": "whole-loop",
            "block_counts": "callgrind", "entries": entries}


def fixture():
    """Results, counts and limits of a small repository."""
    avx2 = {
        "mag": record(1.0, 2.0),                                   # green
        "mag_fallback": record(2.1, 2.0),                          # yellow
        "mag_sq": record(1.15, 1.0, snr=(60.0, 50.0)),             # trade: light blue
        "mag_sq_fallback": record(1.15, 1.0),                      # no trade: red
        "scale": record(1.25, 1.0, snr=(60.0, 50.0)),              # trade above 1.20: red
        "scale_fallback": record(1.3, 1.0),                        # info: no colour
        "dot": record(1.0, 1.0, "sdrk fallback", kind="own_fallback"),
        "mod_block": record(4.0, 8.0, "GNU Radio mod"),
        "mod_block_liquid": record(4.0, 5.0, "liquid-dsp 1.8.3 mod"),
        "mul": {"note": "not simulated: the run failed"},
    }
    runtime = {"mag": {**record(1.1, 1.0), "comparison_kind": "own_build", "compared_with": "ours at avx2"},
               "mod_block": {**record(4.0, 4.0), "comparison_kind": "own_build", "compared_with": "ours at avx2"}}
    sse2 = {"mag": record(1.0, 1.0), "mod_block": record(3.0, 1.0, "GNU Radio mod"),
            "mod_block_liquid": record(3.0, 6.0, "liquid-dsp 1.8.3 mod")}
    cycles = {"platform": "x86_64", "method": {"run_id": "123"},
              "levels": {"sse2": level(sse2), "avx2": level(avx2), "avx2-runtime": level(runtime)}}
    wasm = {"platform": "wasm32", "levels": {"simd128": {"kernels": {
        "mag": {"compared_with": "VOLK 3.3.0 generic", "comparison_kind": "library", "ours_instr_per_sample": 3.0,
                "other_instr_per_sample": 2.5, "ratio": 1.2, "snr_db": {"ours": 999.0, "other": 999.0},
                "status": "dev", "verdict": "waived"}}}}}
    limits = {"_limits": {"release": 1.1, "dev": 1.5}}
    for lvl, ents in (("sse2", sse2), ("avx2", avx2), ("avx2-runtime", runtime)):
        for n, v in ents.items():
            own = v.get("comparison_kind", "library") != "library"
            limits.setdefault(n, {})[lvl] = {"status": "info" if own else "release"}
    limits["mag_sq"]["avx2"] = {"status": "release", "accuracy_trade": "SNR 60 dB against 50 dB; at 1.15 of its cycles"}
    limits["mag_sq_fallback"]["avx2"] = {"status": "dev", "waiver": "1.15 of the library; open item I1-1"}
    limits["scale"]["avx2"] = {"status": "dev", "contract_trade": "less work: x; gain: 1; attempts: y"}
    limits["scale_fallback"]["avx2"] = {"status": "info", "waiver": "reference does different work " + "w" * 300}
    limits["mag"]["avx2-runtime"] = {"status": "info"}
    limits["mag"]["simd128"] = {"status": "dev", "waiver": "open item I1-2"}
    return cycles, wasm, limits


def write_repo(root, cycles, wasm, limits):
    os.makedirs(os.path.join(root, "benches", "results", "cycles"), exist_ok=True)
    with open(os.path.join(root, "benches", "results", "cycles", "x86_64.json"), "w") as fh:
        json.dump(cycles, fh)
    with open(os.path.join(root, "benches", "results", "wasm32.json"), "w") as fh:
        json.dump(wasm, fh)
    with open(os.path.join(root, "benches", "limits.json"), "w") as fh:
        json.dump(limits, fh)


def verify(repo, page):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = vp.main(["--repo", repo, page])
    return code, out.getvalue()


class PageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = os.path.join(self.tmp.name, "repo")
        self.page = os.path.join(self.tmp.name, "page.html")
        self.cycles, self.wasm, self.limits = fixture()
        write_repo(self.repo, self.cycles, self.wasm, self.limits)
        with contextlib.redirect_stderr(io.StringIO()):
            cp.main(["--repo", self.repo, self.page])
        with open(self.page, encoding="utf-8") as fh:
            self.html = fh.read()

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        return {(r["level"], r["name"]): r for r in cp.rows_of(
            {"x86_64": self.cycles}, self.wasm, self.limits)[0]}

    def tr(self, level, entry):
        m = re.search(rf'<tr data-level="{re.escape(level)}"[^>]*data-entry="{re.escape(entry)}"[^>]*>.*?</tr>', self.html)
        self.assertIsNotNone(m, (level, entry))
        return m.group(0)

    def test_the_checker_accepts_the_generated_page(self):
        code, out = verify(self.repo, self.page)
        self.assertEqual(code, 0, out)
        self.assertIn("differences: 0", out)

    def test_the_page_is_self_contained(self):
        self.assertNotRegex(self.html, r"<link|<script src|src=|@import|url\(")
        self.assertIn("Commit <code>unknown</code>", self.html)
        self.assertIn("CI runs: 123", self.html)

    def test_columns_of_a_row(self):
        row = self.tr("avx2", "mag_fallback")
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row)
        self.assertEqual(len(cells), 11)
        self.assertIn(">mag</code>", cells[1])
        self.assertIn('title="entry mag_fallback"', cells[1])
        self.assertEqual(cells[2], "fallback (avx2)")
        self.assertEqual(cells[3], "2.10 / 2.10")
        self.assertEqual(cells[4], "2.00 / 2.00")
        self.assertIn('<span class="v y">1.050</span> / <span class="v y">1.050</span>', cells[7])
        self.assertEqual(cells[5], "120.0")
        self.assertEqual(cells[6], "exact")
        self.assertEqual(cells[8], "opponent exact")
        simd = re.findall(r"<td[^>]*>(.*?)</td>", self.tr("simd128", "mag"))
        self.assertEqual(simd[3], "3.00 instr")
        self.assertIn('<span class="st vd">waived</span>', simd[0])
        self.assertIn('<span class="v r">1.20</span>', simd[7])

    def test_light_blue_only_under_a_trade_up_to_1_20(self):
        r = self.rows()
        self.assertEqual(r[("avx2", "mag_sq")]["classes"], ["lb", "lb"])
        self.assertEqual(r[("avx2", "mag_sq_fallback")]["classes"], ["r", "r"])  # a waiver is no reason
        self.assertEqual(r[("avx2", "scale")]["classes"], ["r", "r"])  # a trade above 1.20
        self.assertEqual(r[("avx2", "scale_fallback")]["classes"], ["", ""])  # info
        self.assertEqual(r[("avx2", "dot")]["classes"], ["", ""])  # own code
        self.assertEqual(r[("avx2", "mag")]["classes"], ["g", "g"])
        self.assertEqual(cp.ratio_class(1.10, True, False), "y")
        self.assertEqual(cp.ratio_class(1.1001, True, True), "lb")
        self.assertEqual(cp.ratio_class(1.2001, True, True), "r")
        self.assertEqual(cp.ratio_class(1.15, False, True), "")
        self.assertIn('data-band="lb"', self.tr("avx2", "mag_sq"))

    def test_a_second_opponent_is_listed_under_its_entry(self):
        r = self.rows()
        self.assertEqual(r[("avx2", "mod_block_liquid")]["kernel"], "mod_block")
        self.assertTrue(r[("avx2", "mod_block_liquid")]["second"])
        self.assertEqual(r[("avx2", "mag_sq")]["kernel"], "mag_sq")  # ours differs from `mag`
        row = self.tr("avx2", "mod_block_liquid")
        self.assertIn(">mod_block</code>", row)
        self.assertIn("liquid-dsp 1.8.3 mod", row)
        self.assertNotIn(">mod_block_liquid<", row)

    def test_own_code_and_justification(self):
        row = self.tr("avx2", "dot")
        self.assertIn('<span class="st own">own (information)</span>', row)
        self.assertIn('data-judged="0"', row)
        why = self.tr("avx2", "scale_fallback")
        self.assertIn('<b>waiver:</b>', why)
        self.assertIn('<span class="m">…</span><span class="t">', why)
        runtime = self.tr("avx2-runtime", "mod_block")
        self.assertIn("ours / own avx2 build 1.000 / 1.000 (information)", runtime)
        self.assertIn(">GNU Radio mod<", runtime)

    def test_the_notes_list_the_entries_without_numbers(self):
        self.assertIn("<code>avx2</code> <code>mul</code>: not simulated: the run failed", self.html)

    def broken(self, old, new, count=1):
        self.assertIn(old, self.html)
        with open(self.page, "w", encoding="utf-8") as fh:
            fh.write(self.html.replace(old, new, count))
        return verify(self.repo, self.page)

    def test_the_checker_finds_a_changed_number(self):
        code, out = self.broken(">2.10 / 2.10<", ">2.01 / 2.10<")
        self.assertEqual(code, 1)
        self.assertIn("page avx2 mag_fallback: cycles ours", out)

    def test_the_checker_finds_a_changed_status(self):
        old = self.tr("avx2", "mag")
        code, out = self.broken(old, old.replace('<span class="st">release</span>', '<span class="st">dev</span>'))
        self.assertEqual(code, 1)
        self.assertIn("page avx2 mag: status", out)

    def test_the_checker_finds_a_wrong_colour(self):
        old = self.tr("avx2", "mag_sq")
        code, out = self.broken(old, old.replace('class="v lb"', 'class="v r"'))
        self.assertEqual(code, 1)
        self.assertIn("page avx2 mag_sq: ratio colour", out)

    def test_the_checker_finds_a_changed_justification(self):
        code, out = self.broken("open item I1-1", "accepted")
        self.assertEqual(code, 1)
        self.assertIn("page avx2 mag_sq_fallback: justification", out)

    def test_the_checker_finds_a_judged_row_with_own_code(self):
        limits = json.loads(json.dumps(self.limits))
        limits["dot"]["avx2"] = {"status": "release"}
        write_repo(self.repo, self.cycles, self.wasm, limits)
        code, out = verify(self.repo, self.page)
        self.assertEqual(code, 1)
        self.assertIn("results avx2 dot: judged against our own code", out)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cp.main(["--repo", self.repo, os.path.join(self.tmp.name, "other.html")])

    def test_the_checker_finds_a_missing_row(self):
        old = self.tr("sse2", "mag")
        code, out = self.broken(old, "")
        self.assertEqual(code, 1)
        self.assertIn("page sse2 mag: row: shown 'missing'", out)


class RepositoryTest(unittest.TestCase):
    def test_the_committed_results_give_a_page_without_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = os.path.join(tmp, "page.html")
            with contextlib.redirect_stderr(io.StringIO()):
                cp.main([page])
            code, out = verify(os.path.dirname(HERE), page)
            self.assertEqual(code, 0, out[-3000:])


if __name__ == "__main__":
    unittest.main()
