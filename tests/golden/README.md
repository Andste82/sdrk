# Golden vectors

Golden vectors validate the Rust kernels against an independent implementation: NumPy/SciPy for float
kernels and the Python Q15 integer model (`sdrk_golden/q15.py`) for bit-exact Q15 kernels. The files in
`vectors/` are committed to git and read by `sdrk_testkit::golden` (feature `std`). Never edit them by
hand and never generate them with code from this repository.

The golden of a tolerance case is the mathematical optimum: the result computed in `f64`, treated as
exact (except `squelch`, `stereo_pll` and `preemph_limit`, whose cases come from an exact float32 or integer model, see `docs/kernels.md`). It is the
first accuracy target of a kernel, not only a pass/fail check. The `snr_db_min` stored in a case is the contract
floor; the accuracy policy in [`docs/kernels.md`](../../docs/kernels.md#accuracy-policy) adds the
achievable optimum and a regression guard close to the measured SNR. A bit-exact Q15 case proves agreement
with our own integer model, not closeness to the optimum; the policy adds an SNR check against an `f64`
golden for those kernels.

**Report.** When the environment variable `SDRK_GOLDEN_REPORT` names a file, every comparison of
`sdrk_testkit::golden::assert_matches` appends one JSON line to it: the case, the sample type, the class,
the floor, the achieved SNR, the achievable optimum (the expected values rounded once to the sample type) and
whether it passed. `assert_matches_block_float` does the same for a block-floating-point `CQ15` output; its
optimum is the expected values rounded to a Q15 mantissa with the block exponent they need. `cargo measure <kernel>` uses it (`docs/performance.md`).

## Setup

```sh
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r tests/golden/requirements.txt
```

On Windows activate with `.venv\Scripts\activate`. The versions in `requirements.txt` are pinned exactly;
byte-identical output is only guaranteed with those versions and Python 3.12, **on Linux x86-64**.

**Generate the vectors on Linux x86-64 only.** The generator runs on every host (R1.3, checked by CI),
but NumPy and SciPy evaluate `cos`, `exp`, the FFT and the filter design with the math library and the
build of the platform, so on Windows and Linux arm64 about 30 to 40 cases differ in the last bit (an ulp,
or an LSB after quantization). The Rust tests compare with the committed files on every platform and are
not affected; only a regeneration on another host would change unrelated cases. Making those cases
independent of the platform (own scalar math, as `sdrk_golden/scalar_math.py` already does for some cases)
is planned (see `docs/milestones/BACKLOG.md`).

The filter cases of M23 follow this rule (D107): their float expected values are exact sums (`scalar_math.fir_exact`), because the BLAS kernels behind `scipy.signal.lfilter` and `np.convolve` add in a CPU-dependent order. Check a new filter case with `OPENBLAS_CORETYPE=Nehalem python tests/golden/generate.py --kernel <name> --check` as well.

## Commands

```sh
python tests/golden/generate.py                # generate every case into tests/golden/vectors
python tests/golden/generate.py --kernel scale # only the cases of one kernel
python tests/golden/generate.py --list         # kernel/case lines
python tests/golden/generate.py --check        # compare byte for byte with the committed files
python tests/golden/generate.py --out DIR      # use another output directory
```

`--check` prints every missing, differing or stale file (a committed file without a registered case) and
exits with 1 on any difference. The script runs from any working directory. Before generating, the Q15 model
runs a self-test of the R7 spot values, and so does the float32 model (below).

## Format (version 1)

`<kernel>/<case>.json` plus one `<case>.<array>.bin` per array. The JSON is written with sorted keys and
2-space indent. The fields are `format_version`, `kernel`, `case`, `seed`, `params`, `arrays`
(`file`, `dtype`, `len`), `tolerance`, `notes` and `provenance`. Binary arrays are raw little-endian without a
header. Dtypes: `u8`, `i8`, `i16`, `i32`, `f32`, `f64`, `ci16` (interleaved re, im), `cf32`, `cf64`. `len`
counts elements (a complex sample counts once). Tolerance classes:

- `{"class": "bit_exact"}`: raw bits are equal.
- `{"class": "tolerance", "snr_db_min": x, "skip": n}` with optional `"max_abs_err"` and optional `"snr_db_guard"`, the
  regression guard of the accuracy policy (the lowest measured SNR minus 1 dB), checked like `snr_db_min`.
- `{"class": "max_abs_err", "max_abs_err": x, "skip": n}` with optional `"min_ref_db"` and then required
  `"ref_unit": "db" | "linear"`; only samples whose reference is at or above the level are checked.

SNR is `10 log10(sum |ref|^2 / sum |ref - out|^2)` over samples `skip..`. A case stays at or below 512 KiB, all
vectors together at or below 16 MiB; the generator fails otherwise.

## Adding a case

1. Copy the template of `sdrk_golden/cases/scale.py` into a module of the kernel (new modules are imported in
   `sdrk_golden/cases/__init__.py`).
2. Choose a new seed. The `rng` argument is the only source of randomness; the draw order is part of the case.
3. Quantize the inputs yourself (`astype(np.complex64)` or `q15.from_float`) and compute `expected` from the
   quantized input, so that only kernel error is measured. Tolerance cases store `f64`/`cf64` (exception:
   `agc/steps_c32_24000` stores the f64 model rounded once to `cf32`, because a `cf64` file would exceed
   512 KiB); bit-exact cases store the exact output dtype.
4. Pick the tolerance class from the contract table in `docs/kernels.md`.
5. Run `python tests/golden/generate.py --kernel <name>`, then `--check`.
6. Commit the vectors together with the code and the Rust test that loads them.

In Rust: `let case = sdrk_testkit::golden::load("scale", "c32_random_4096")?;`, run the kernel, then
`golden::assert_matches(&case, case.array("expected")?, &actual)?`. The environment variable
`SDRK_GOLDEN_DIR` overrides the vector directory.

## The exact float32 model

Kernels whose output depends on a discrete decision (the gate of `squelch`) cannot be compared with an f64
reference, because the decision may flip one sample earlier or later. Their arithmetic is fixed by the
specification and `sdrk_golden/f32model.py` evaluates it with NumPy `float32` scalars, one operation at a time
(`add`, `sub`, `mul`, `flush_denormal`, `is_finite`); NumPy rounds every scalar operation to float32 and never
fuses a multiply with an add, so the model equals Rust `f32` code compiled without FMA. Such a case is
`bit_exact` and its JSON carries `"model": "float32"` at the top level. `f32model.self_test()` checks the
helpers against hand-computed values (for example that `0.1f + 0.2f == 0.3f` and that a product and a subtraction
are not fused) and runs in front of every generate and check run.

## The stereo cases (seeds 438 to 443 and 454)

`stereo_pll` (seeds 438 to 441 and 454) and `stereo_matrix` (442, 443) follow the model rule above: `sdrk_golden/cases/stereo.py` builds the cosine tables of `rotate` itself (`float32(cos(2 pi k / 4096))` with `scalar_math.cos`, the Q15 rule from the same f64 values) and stores their SHA-256 in every `stereo_pll` case, so a Rust test fails with a clear message when `sdrk-core` has other tables. The coefficients are stored as IEEE bits (`kp_bits`, ...). Besides the bit-exact model, every `lock_*` and `fade_*` case stores `snr_db_guard`, the regression guard of decision D111: the SNR of the difference channel (from sample `guard_skip`) of the model against the same loop in float64 with exact sine and cosine, minus 1 dB, rounded down to 0.1 dB. The float64 output is not committed (it depends on the libm of the host); the Rust test runs its own float64 loop on the stored inputs. Everything that feeds a committed byte uses plain Python arithmetic, `scalar_math` and the float32 helpers (no BLAS, no SIMD transcendental), so the files are identical on every CPU.

## The preemph_limit cases (seeds 444 to 449; `clamp_f32_4096` has no random input, seed 0)

`preemph_limit` follows the model rule above: `sdrk_golden/cases/preemph_limit.py` implements the float32 model of the `f32` variant with the `f32model` helpers and the integer model of the `Q15` variant with Python integers (no NumPy arithmetic, no `sin`/`exp`/`lfilter`; the tone bursts use `math.sin`), so the files are identical on every CPU. The cases are `preemph_f32_8192` (444, limiter idle), `limit_f32_8192` (445, limit 0.5 with two tone bursts above the limit), `limit_q15_8192` (446, the input of 445 quantized), `preemph_q15_8192` (447, tau 530 us), `nonfinite_f32_4096` (448, NaN and infinities at 20 listed positions) `retune_f32_8192` (449, `set_limit(0.5)` before sample 4096) and `clamp_f32_4096` (seed 0, 24 finite values beyond the input limit 1024, which the model clamps). The coefficients are stored as IEEE bits (`b0_bits`, `b1_bits`, `a1_bits`) and the Rust tests build the state from them. The generator asserts that the limiter visits the states the case is meant to cover. The design cases `design/preemphasis_*` (seed 0) store the values of the formulas of `sdrk_golden/design.py:preemphasis_coefs` in plain `math` floats; the generator asserts that `scipy.signal.bilinear` agrees within 1e-12.
