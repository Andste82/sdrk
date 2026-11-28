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
        "mag": record(1.0, 2.0, "volk_32fc_magnitude_32f"),        # green
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
               "mod_block": {**record(4.0, 4.0), "comparison_kind": "own_build", "compared_with": "ours at avx2"},
               "mag_fallback": {"note": "not simulated: not measured at the level avx2-runtime"}}
    sse2 = {"mag": record(1.0, 1.0), "mod_block": record(3.0, 1.0, "GNU Radio mod"),
            "mod_block_liquid": record(3.0, 6.0, "liquid-dsp 1.8.3 mod")}
    cycles = {"platform": "x86_64", "method": {"run_id": "123"}, "tools": {"llvm-mca": "LLVM 23.1.1 test"},
              "levels": {"sse2": level(sse2), "avx2": level(avx2), "avx2-runtime": level(runtime)}}
    wasm = {"platform": "wasm32", "levels": {"simd128": {"counter": "wasmtime fuel", "kernels": {
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
    limits["scale_fallback"]["avx2"] = {"status": "info", "waiver": "reference does different work " + "w " * 150}
    limits["mag"]["avx2-runtime"] = {"status": "info"}
    limits["mag"]["simd128"] = {"status": "dev", "waiver": "open item I1-2"}
    return cycles, wasm, limits


# The VOLK tables of xtask/src/bench.rs in the form of the real file.
BENCH_RS = r'''
        level: "avx2",
        refs: "x86_64",
        volk_pin: VolkPin::Avx2,
        level: "simd128",
        refs: "wasm32",
        volk_pin: VolkPin::Default,
            Self::Avx2 => VOLK_AVX2_CONFIG,
        VolkPin::Default => VOLK_AVX512_DISPATCH,
const VOLK_AVX2_CONFIG: &str = "\
volk_32fc_magnitude_32f a_avx2 u_avx2
";
pub const VOLK_AVX512_DISPATCH: &str = "\
volk_32fc_magnitude_32f a_avx512f u_avx512f
";
pub const VOLK_FUNCTIONS: [&str; 1] = [
    "volk_32fc_magnitude_32f",
];
'''


def write_repo(root, cycles, wasm, limits):
    os.makedirs(os.path.join(root, "benches", "results", "cycles"), exist_ok=True)
    os.makedirs(os.path.join(root, "xtask", "src"), exist_ok=True)
    with open(os.path.join(root, "xtask", "src", "bench.rs"), "w") as fh:
        fh.write(BENCH_RS)
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
            {"x86_64": self.cycles}, self.wasm, self.limits, cp.volk_tables(self.repo))[0]}

    def tr(self, level, entry):
        m = re.search(rf'<tr class="[^"]*" data-level="{re.escape(level)}"[^>]*data-entry="{re.escape(entry)}"[^>]*>.*?</tr>',
                      self.html)
        self.assertIsNotNone(m, (level, entry))
        return m.group(0)

    def cells(self, level, entry):
        return re.findall(r"<td[^>]*>(.*?)</td>", self.tr(level, entry))

    def test_the_checker_accepts_the_generated_page(self):
        code, out = verify(self.repo, self.page)
        self.assertEqual(code, 0, out)
        self.assertIn("differences: 0", out)

    def test_the_page_is_self_contained(self):
        self.assertNotRegex(self.html, r"<link|<script src|src=|@import|url\(")
        header = self.html.split("<header>")[1].split("</header>")[0]
        self.assertNotIn("unknown", header)
        self.assertIn("CI runs 123 · llvm-mca: LLVM 23.1.1 test · simd128: wasmtime fuel", header)
        self.assertIn("<code>sse2 / avx2 / avx2-runtime</code>: Zen 3 / Alder Lake", header)

    def test_columns_of_a_row(self):
        c = self.cells("avx2", "mag_fallback")
        self.assertEqual(len(c), 8)
        self.assertEqual(c[0], '<span class="kn">mag </span>↳ fallback<span class="lv"> (avx2)</span>')
        self.assertEqual(c[1], '<span class="st">release</span>')
        self.assertEqual(c[2], "2.10 / 2.10")
        self.assertEqual(c[3], "2.00 / 2.00")
        self.assertEqual(c[4], '<span class="v t-y">1.050</span> / <span class="v t-y">1.050</span>')
        self.assertEqual(c[5], "−∞")
        self.assertEqual(c[6], "lib 1.0 kernel")
        row = self.tr("avx2", "mag_fallback")
        self.assertIn('title="SNR ours 120.0 dB / opponent ∞ (bit-exact: output identical to the f64 reference, '
                      'no error)"', row)
        self.assertIn('<td class="n c-ratio y"', row)
        self.assertIn('<td class="n c-delta r" data-v="-Infinity"', row)
        self.assertIn('title="entry mag_fallback at avx2: Zen 3 / Alder Lake"', row)
        simd = self.cells("simd128", "mag")
        self.assertEqual(simd[2], "3.00")
        self.assertIn('<span class="st vd">waived</span>', simd[1])
        self.assertEqual(simd[4], '<span class="v t-r">1.20</span>')
        self.assertIn('<td class="n c-delta" data-v="0.0" title="both bit-exact', self.tr("simd128", "mag"))
        self.assertEqual(simd[5], "0")
        self.assertIn("instructions per sample, no CPU model", self.tr("simd128", "mag"))

    def test_the_opponent_names_the_volk_protokernel_of_the_level(self):
        self.assertIn(">VOLK volk_32fc_magnitude_32f (a_avx2)</td>", self.tr("avx2", "mag"))
        self.assertIn('title="volk_32fc_magnitude_32f"', self.tr("avx2", "mag"))
        self.assertEqual(cp.short_opponent(
            "KFR 7.2.0 kfr::iir, iir_state<float, 2> (avx2, -ffp-contract=off), two calls", "avx2", {}, []),
            "KFR 7.2.0 kfr::iir (avx2)")
        self.assertEqual(cp.short_opponent(
            "liquid-dsp 1.8.3 m (x), dot products (the portable loop at simd128)", "sse2", {}, []),
            "liquid-dsp 1.8.3 m")
        self.assertEqual(cp.short_opponent("KFR 7.2.0 real(c) (benches/refs/kfr-ops.cpp, avx512)", "avx512", {}, []),
                         "KFR 7.2.0 real(c) (avx512)")
        self.assertEqual(cp.short_opponent("CMSIS-DSP 1.18.0 arm_clip_f32 to [-1, 1], generic C (-O3, sse2)",
                                           "sse2", {}, []), "CMSIS-DSP 1.18.0 arm_clip_f32 to [-1, 1] (sse2)")

    def test_blocks_by_kernel_with_levels_in_order(self):
        body = self.html.split("<tbody>")[1].split("</tbody>")[0]
        heads = re.findall(r'<tr class="kh" data-kernel="([^"]+)">', body)
        self.assertEqual(heads, ["dot", "mag", "mag_sq", "mod_block", "scale"])
        mag = body.split('<tr class="kh" data-kernel="mag">')[1].split('<tr class="kh"')[0]
        order = re.findall(r'data-level="([^"]+)" data-variant="(\w+)"', mag)
        self.assertEqual(order, [("sse2", "simd"), ("avx2", "simd"), ("avx2", "fallback"),
                                 ("avx2-runtime", "simd"), ("simd128", "simd")])
        self.assertIn('<tr class="row child fb"', self.tr("avx2", "mag_fallback"))

    def test_simd_against_its_fallback_is_information(self):
        r = self.rows()
        self.assertEqual(r[("avx2", "mag")]["vs_fallback"], (1.0 / 2.1, 1.0 / 2.1))
        self.assertIn('title="SIMD / own fallback 0.48 / 0.48 (information)"', self.tr("avx2", "mag"))
        self.assertIn('data-little="0"', self.tr("avx2", "mag"))
        self.assertIn('data-little="1"', self.tr("avx2", "mag_sq"))  # 1.15 / 1.15 of its fallback
        self.assertIn('data-little="1"', self.tr("avx2", "mag_sq_fallback"))
        self.assertIn('title="ours / own avx2 build 1.10 / 1.10 (information)"', self.tr("avx2-runtime", "mag"))

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

    def test_a_second_opponent_is_a_further_row_of_its_entry(self):
        r = self.rows()
        self.assertEqual(r[("avx2", "mod_block_liquid")]["kernel"], "mod_block")
        self.assertTrue(r[("avx2", "mod_block_liquid")]["second"])
        self.assertEqual(r[("avx2", "mag_sq")]["kernel"], "mag_sq")  # ours differs from `mag`
        row = self.tr("avx2", "mod_block_liquid")
        self.assertIn('<span class="kn">mod_block </span>avx2</td>', row)
        self.assertIn(">liquid-dsp 1.8.3 mod</td>", row)
        self.assertNotIn('data-kernel="mod_block_liquid"', self.html)

    def test_own_code_and_justification(self):
        row = self.tr("avx2", "dot")
        self.assertIn('<span class="st own">own (information)</span> sdrk fallback', row)
        self.assertIn('data-judged="0"', row)
        why = self.tr("avx2", "scale_fallback")
        self.assertIn("<b>waiver:</b>", why)
        self.assertIn('<span class="m">…</span><span class="t">', why)
        runtime = self.tr("avx2-runtime", "mod_block")
        self.assertIn("ours / own avx2 build 1.00 / 1.00 (information)", runtime)
        self.assertIn(">GNU Radio mod</td>", runtime)

    def test_the_notes_list_only_the_real_gaps(self):
        notes = self.html.split('<ul id="notes"')[1].split("</ul>")[0]
        self.assertIn("<code>avx2</code> <code>mul</code>: not simulated: the run failed", notes)
        self.assertNotIn("mag_fallback", notes)
        self.assertIn("1 entries that the plan does not measure", self.html)

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
        code, out = self.broken(old, old.replace('class="v t-lb"', 'class="v t-r"'))
        self.assertEqual(code, 1)
        self.assertIn("page avx2 mag_sq: ratio colour", out)

    def test_a_cell_has_the_colour_of_its_worse_number(self):
        r = cp.rows_of({"x86_64": self.cycles}, self.wasm, self.limits)[0]
        self.assertEqual(cp.band_of(["g", "r"]), "r")
        self.assertEqual(cp.band_of(["lb", "y"]), "lb")
        self.assertEqual(cp.band_of(["g", "y"]), "y")
        self.assertTrue(all("c-ratio" in self.tr(x["level"], x["name"]) for x in r))
        self.assertNotRegex(self.html, r'class="v (g|y|r|lb)"')  # no inner background boxes
        self.assertEqual(cp.snr_delta(999.0, 50.0)[:2], ("+∞", "Infinity"))
        self.assertIn("∞ = bit-exact: the output is bit-identical to the f64 reference, so the SNR is infinite.",
                      self.html)

    def test_the_checker_finds_a_wrong_cell_colour(self):
        old = self.tr("avx2", "mag_sq_fallback")
        code, out = self.broken(old, old.replace('<td class="n c-ratio r"', '<td class="n c-ratio y"'))
        self.assertEqual(code, 1)
        self.assertIn("page avx2 mag_sq_fallback: ratio cell colour", out)

    def test_the_checker_finds_a_changed_justification(self):
        code, out = self.broken("open item I1-1", "accepted")
        self.assertEqual(code, 1)
        self.assertIn("page avx2 mag_sq_fallback: justification", out)

    def test_the_checker_finds_a_changed_snr_tooltip(self):
        code, out = self.broken("SNR ours 60.0 dB / opponent 50.0 dB", "SNR ours 61.0 dB / opponent 50.0 dB")
        self.assertEqual(code, 1)
        self.assertIn("SNR delta tooltip", out)

    def test_the_checker_finds_a_wrong_protokernel(self):
        code, out = self.broken("(a_avx2)</td>", "(u_avx2)</td>")
        self.assertEqual(code, 1)
        self.assertIn("page avx2 mag: protokernel of volk_32fc_magnitude_32f", out)

    def test_the_checker_finds_a_row_in_another_block(self):
        row = self.tr("avx2", "scale")
        html = self.html.replace(row + "\n", "", 1)
        html = html.replace('<tr class="kh" data-kernel="mag">', row + '\n<tr class="kh" data-kernel="mag">', 1)
        with open(self.page, "w", encoding="utf-8") as fh:
            fh.write(html)
        code, out = verify(self.repo, self.page)
        self.assertEqual(code, 1)
        self.assertIn("page avx2 scale: block", out)

    def test_the_checker_finds_a_listed_planned_entry(self):
        code, out = self.broken('<ul id="notes" class="explain">\n',
                                '<ul id="notes" class="explain">\n'
                                '<li><code>avx2-runtime</code> <code>mag_fallback</code>: x</li>\n')
        self.assertEqual(code, 1)
        self.assertIn("notes avx2-runtime mag_fallback: listed", out)

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
