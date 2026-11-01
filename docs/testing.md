# Testing with virtual devices and the loopback channel

The testkit (`sdrk-testkit`, layer `crates/testing`) has virtual devices for the stream traits and a channel simulator. Unit tests of the crates, the driver conformance checks ([device.md](device.md#conformance-checks)) and the system tests of phase 3 use them.

## Virtual devices

- `VirtualRx` (M03): a deterministic signal source delivered in chunks, with gaps and an anchor.
- `VirtualTx` (`no_std`): `TxStream` that stores what is written in caller storage, records every call (start, flags, `tx_time_ns`, sample count, position; at most 64 calls) and injects `UNDERRUN` and `LATE` statuses for `poll_status` at configured sample indices.

## Loopback channel

`channel::Loopback::new(LoopbackParams)` returns the two sides of a channel: `LoopbackTx` (a `TxStream`) and `LoopbackRx` (an `RxStream`), connected through shared state (`split()` gives the two sides, both are `Send`). They share a timeline: receive index 0 is the time `t0_ns`. At the receive index `n` the channel produces

```text
y[n] = g * x(n - d) * exp(j 2 pi f n / fs) + noise[n],     g = 10^(-attenuation_db / 20)
```

- `x` is the transmitted signal in `f64`, placed on the timeline: a write with `tx_time_ns` starts a burst at the index `round((tx_time - t0) * rate / 1e9)` (exact integer arithmetic; a burst before `t0` is dropped and reported as a `LATE` status), a write without a time continues the current burst, and continuous transmission starts at index 0. `BURST_END` closes the current burst. Overlapping bursts add. Silence is zero.
- The delay `d = delay_samples`: the integer part shifts the index, the fractional part is a 63-tap Kaiser(beta 8)-windowed-sinc interpolator computed in `f64` (`g[j] = sinc(j - frac) * kaiser(j - frac)`, `j = -31..31`). An integer delay is an exact shift; silent stretches cost nothing.
- The frequency offset `f` rotates by the phase `2 pi frac(f n / fs)` of the absolute index.
- The noise has independent normal I and Q components of the standard deviation `noise_sigma` (units: 1.0 is full scale). It comes from PCG32 positioned by the absolute index (8 draws per sample, Box-Muller), so any chunking gives the same samples; `seed` selects the sequence.
- The result is encoded into the raw format with saturation. The format must have a whole number of bytes per sample.
- The receiver never ends. It produces what the transmitter has written so far: transmit first, then receive. The first receive block carries the anchor `index 0 = t0_ns` (source `Host`).

## System tests

The workspace member `sdrk-system-tests` (`tests/system`) closes phase 3:

| Test | What it shows |
| --- | --- |
| `fm_over_loopback_f32`, `fm_over_loopback_q15` | `fm_transmitter.json` in pull mode, a channel with 1234.37 samples of delay, 6 dB attenuation, +1.5 kHz offset and noise for 30 dB carrier-to-noise ratio in 200 kHz, `fm_receiver.json` with `cs8` input: SINAD over the second half of the audio at least 30 dB (f32) and 25 dB (Q15). |
| `loop_fm_flat_f32`, `loop_fm_flat_q15` (M25, TX-FLAT) | Tones of 100 Hz, 1, 5, 10 and 12 kHz (0.3 s each, amplitude 0.1) through `fm_transmitter.json` in pull mode, the `cs8` bytes pushed (no channel) into `fm_receiver.json` with `cs8` input: every level within 0.5 dB (`f32`) or 1.0 dB (Q15) of the 1 kHz level (0.04 dB measured); the same test without the `shape` node fails (10 kHz about 10 dB low). |
| `loop_fm_peak_f32`, `loop_fm_peak_q15` (M25, TX-PEAK) | 1 s of band-limited noise (Butterworth 4 at 15 kHz, sigma 3, seed 450) through `fm_transmitter.json`; a `ddc` to 240 kHz and `fm_demod` without de-emphasis measure the instantaneous frequency: at most 1.02 times 75 kHz (0.998 `f32`, 0.983 Q15); without the `shape` node 1.60 in `f32` (above 1.5) and 1.04 in Q15 (the Q15 input saturates at full scale, open item I25-2). |
| `loop_nfm_f32`, `loop_nfm_q15` (M25, NFM) | A 1 kHz tone at amplitude 0.3 through `nfm_transmitter.json` into `nfm_receiver.json` (`cs8`, `tune.frequency_hz` 200000): SINAD at least 30 dB (`f32`) and 25 dB (Q15) (56.5 and 55.4 dB measured); with the TX-PEAK noise the peak deviation stays below 1.0 times 5 kHz (0.80 and 0.79). |
| `burst_timing_loopback` | Two 100 ms bursts of a constant carrier at `t0 + 1 s` and `t0 + 1.5 s`; the first down-converter output sample above half the plateau, converted with the delay-compensated anchor, lies within one output sample period of `tx_time + delay`. |
| `two_stations_no_crosstalk` | Two stations at -400 kHz and +400 kHz, one flow with two down-converter / demodulator branches: SINAD at least 35 dB, the other tone at most -50 dB. |
| `long_run_no_drift` | Start index `2^50`, 20 s at 240 kHz through the tuned resampler (+37.5 ppm) and `fm_demod`: the output counts follow the exact formula and the time of the last anchor equals the analytic value to the nanosecond. The ten-minute variant is `#[ignore]` and runs nightly (`cargo test --release -p sdrk-system-tests -- --ignored`). |
| `event_propagation_exact` | 50 random gap, anchor, retune and overflow events appear on `audio` at exactly the index of the mapping rule of [timing.md](timing.md#index-domains-in-the-pipeline). |
| `sigmf_roundtrip_pipeline` | A recording with a gap and annotations through `SigmfWriter` and `SigmfReader` gives the same audio and events as pushing the blocks directly. |

## The SIMD backend in the suite and in CI

**E2E-STEREO (M24).** `sdrk_testkit::e2e_stereo` receives a stereo broadcast (2.4 MS/s, +600 kHz, deviation 75 kHz, `L` 1 kHz and `R` 3 kHz at amplitude 1, pilot 0.1, scaled to 0.9, complex noise sigma 0.003, `CU8`) with the blocks `convert`, `ddc` (decimation 5, cutoff 130 kHz, transition 30 kHz) and `fm_stereo_demod` and measures from 0.25 s on without storing a sample: `MultitoneAccumulator` (the normal equations of a joint least-squares fit of 1 kHz, 3 kHz and the 19 kHz pilot per channel) gives the separation (`f32` at least 40 dB, Q15 at least 35 dB), the SINAD per channel with both tones removed (55 and 45 dB), the level against `0.9 |H_de|` within 0.5 dB (Q15 scaled by `2 deviation / fs`, the unit of the Q15 discriminator) and the pilot residue (at most -70 and -60 dBFS); `pilot_locked()` is polled after every block of 500 samples and must become true within 0.1 s and stay true. The switching case (pilot from 0.3 s to 0.7 s of 1.0 s) needs stereo within 0.1 s after the pilot appears, mono within 0.2 s after it vanishes and bit-identical channels while mono; the mono cases (a mono transmission, noise without a carrier) never report stereo. The pilot of the switching signal is gated by adding its phase integral over the window (`StereoSignal::switching`; the edges are multiples of 1 ms, i.e. whole pilot periods, so the multiplex signal stays continuous). The cases `block_fm_stereo_demod_separation` and `block_fm_stereo_demod_mono` (the block alone at 480 kHz) and the E2E cases `e2e_fm_stereo_f32` and `e2e_fm_stereo_q15` (0.5 s) are in the quick profile; `e2e_fm_stereo_mono` (0.25 s each of a mono transmission and noise), `e2e_fm_stereo_{f32,q15}_2s` and `e2e_fm_stereo_switch_f32` are in the full profile, because the signal at 2.4 MS/s with a nine-term multiplex generator is expensive on the Cortex-M4 (the quick profile takes 728 of its 900 s with them, decision D128). The host numbers are in `crates/testing/sdrk-testkit/tests/e2e_stereo_m24.rs`.

**Loop tests (M25).** The three groups in `tests/system/tests/loop_fm_m25.rs` (table above) feed the `cs8` bytes of the transmitter flow, pulled in pull mode, into the receiver flow without a channel and measure the audio with `ToneMeter` and `sinad_db`. The peak is measured by a separate small flow (`ddc` from 2.4 MS/s to 240 kS/s and `fm_demod` with `deemphasis_us` 0 and `audio_decimation` 1, i.e. `quad_demod` with the gain `240 kHz / (2 pi deviation_hz)`, always in `f32`, because the Q15 discriminator cannot read above 1.0). The noise is generated in the test: `Pcg32::new(450, 1)`, a fourth-order Butterworth low-pass at 15 kHz (two RBJ sections in `f64`), scaled to sigma 3 after the filter. For Q15 the flow gets `settings.precision` `q15` and `inputs.audio.type` `q15` (the input type must fit the precision) and a pull source that converts with saturation. They need the pipeline and therefore `std`, so they are not suite cases and cost no emulator time; they take about 35 s in a debug build. The suite cases of M25 (kernel cases `preemph_limit_f32` and `preemph_limit_q15` with `Fallback` against `Reference` and 20 chunk plans, `block_preemph_limit_peak` with random input up to 20 times full scale, all quick; `extremes_m25`, full) are listed in `crates/testing/sdrk-testkit/src/cases.rs`; `tests/suite_m25.rs` runs them on the host in both profiles and in the 80 KiB arena of the quick profile.

The suite of `sdrk-testkit` runs through the facade, so every case exercises the SIMD backend where the platform has one. Fifteen cases `backend_vs_fallback_*` (`convert`, `convert_out`, `rotate`, `fir`, `fir_decimate`, `fir_interpolate`, `biquad`, `mag_sq`, `mag`, `quad_demod`, `fm_mod`, `fft`, and the M23 cases `halfband` for `halfband_decimate` and `halfband_decimate_real`, and `real_to_complex`, and the M24 case `stereo_matrix`) cover every accelerated kernel and type (the `C32` `agc` is compared with the fallback bit for bit by its own case `agc_c32`) with the lengths 0, 1, 3, 15, 16, 17, 255 and 4096 (the lengths that fit the arena of the tier: the quick tier takes the short ones), extreme inputs (full scale, -32768, NaN and infinity in `f32` where defined) and, for the stateful kernels, random chunk plans that must give the bits of one call. Bit-exact kernels are compared bit for bit with the fallback; tolerance kernels with the contract of `docs/kernels.md` against the reference (`rotate` 120 dB, `fir` family 110 dB, `quad_demod` 100 dB / 2 LSB, `fft` 110 dB for `C32`; for `CQ15` the exact transform is the measure, fixed scaling within 1 dB of the fallback with its exponent and block floating point checked by the reference cases of M07, `fm_mod` 120 dB; the `C32` and `f32` halfband and `real_to_complex` kernels 110 dB against the reference, the `Q15` and `CQ15` ones bit for bit with the fallback including the full-scale saturation corners, and `real_to_complex` also over random chunk plans that must keep the phase across calls).

The tests of the crate `sdrk-backend-simd` (`crates/backends/sdrk-backend-simd/tests`) compare the kernels with the fallback for all lengths of the vector tails, with `LENGTHS = 0, 1, 2, 3, 7, 15, 16, 17, 31, 255, 257, 4096` (`mag` and `mag_sq` at every length up to 4096 and at four start offsets; the `mag` CQ15 integer square root also around every perfect square and every rounding boundary `r (r + 1)`, and in the ignored test `mag_cq15_exhaustive_every_raw_pair` for all 2^32 raw pairs, run by the `sweep_*` jobs), with random chunk plans, extreme inputs and invalid calls (same error values as the fallback). They run at the level the build selects: the default build at SSE2, `--features std` at the best level of the CPU, `-C target-cpu=...` at the level of the flag. A workspace run enables `std` of the backend (the pipeline, `sdrk-testkit/std` and the system tests ask for it), so it always detects the level at run time; the static levels are tested with the backend and the facade alone (`cargo test -p sdrk-backend-simd -p sdrk`). The backend alone (without `-p sdrk`) also runs its own default build, without the static tables `nco-tables` and `fft-tables`, which the facade switches on. `SDRK_EXPECT_LEVEL=<name>` makes `tests/api.rs` check that `level_name()` is that level; every CI step that tests a level sets it, so a build or an emulator that selects another level fails instead of passing at the wrong one. The exhaustive test of `convert_out` (`convert_out_exhaustive_every_f32`, every `f32` bit pattern against the fallback, the proof of the per-byte CU8 bound) is `#[ignore]` and runs nightly in the jobs `sweep_*`; `SDRK_SWEEP_PART=k/n` runs the `k`-th of `n` interleaved parts, each of which covers every exponent.

`sdrk::backend_for` is tested in `crates/facade/sdrk/tests/facade.rs`: every accelerated kernel reports `simd` when `sdrk::SIMD_BACKEND` is true and `fallback` otherwise (`--cfg sdrk_fallback_only`, no `simd` feature, or a target without SIMD).

### CI

CI is three workflow files. `ci.yml` runs on pull requests and on pushes to `main`, `nightly.yml` every night and on demand, and both call `tree.yml`, the tree of jobs. A job starts when the jobs it needs have passed; when one fails or is skipped, everything below it is skipped too.

| Trigger | File | What runs |
| --- | --- | --- |
| pull request | `ci.yml` | `scope` (what changed), then the tree with scope `pr`: `check`, every build and the tests marked `pr` below. A pull request that only changes Markdown files no test reads runs `check` only |
| `gh workflow run nightly.yml --ref <branch>` | `nightly.yml` | the tree with scope `full` on any branch, for example before merging a change of the CI or of a long-running job |
| push to `main` | `ci.yml` | `smoke` (build, `xtask check`, `bare-metal`, the unit tests). The pull request has tested the same tree, because a branch must be up to date with `main` before it is merged |
| every night | `nightly.yml` | the tree with scope `full`: every `pr` job and the jobs marked `full`. When `main` is unchanged since the last green nightly run, the `pr` tests are skipped and only the `full` jobs run, except on Sunday |
| Sunday night | `nightly.yml` | as every night, with the `pr` tests always |
| manual (`workflow_dispatch`) | `nightly.yml` | as at night; the input chooses whether the `pr` tests are skipped |

The branch protection of `main` requires one check, `result` of `ci.yml`: it is green only when `scope` and every job of the tree that ran passed, and the branch is up to date with `main`. A failed nightly run opens an issue with the label `nightly-failure`, or comments on the open one. Every job has a timeout; the cargo caches are saved by runs on `main` only (pull requests read them), one cache per platform and build.

The tree, with `[pr]` for every run and `[full]` for nightly and manual runs. The SIMD variants that depend only on the CPU architecture (fixed levels, without the static tables, fallback only) run on one host per architecture: x86-64 on Linux, AArch64 on Linux arm64. Build, the whole test suite, `xtask` and the golden generator run on all three hosts (R1.3).

```text
check                              gate: fmt · deny · clippy (x86-64, wasm32-wasip1 with SIMD128, sdrk-wasm on
│                                  wasm32-unknown-unknown, thumbv7em) · xtask check (with cargo doc -D warnings) ·
│                                  schema · golden (pr: when tests/golden/ changed) · the benches workspace
│                                  builds (pr: when kernels, benches, xtask or the toolchain changed)
├── check_msrv                     cargo +<rust-version> check --workspace                                    [pr]
├── golden_x64_win, golden_arm64_linux   the golden generator runs on the other hosts; the byte comparison
│                                  is information, the vectors are generated on Linux x86-64          [full]
├── build_x64_linux                bare-metal · nextest archive · unit tests · doc tests · pipeline without YAML
│   ├── test_x64_linux             the whole suite from the archive                                           [pr]
│   ├── test_x64_linux_fallback    --cfg sdrk_fallback_only                                                   [pr]
│   ├── test_x64_linux_fixed_arch  sse2 | sse4.2 | avx2 fixed at compile time; without the tables at sse2, avx2 [pr]
│   ├── test_x64_linux_sde_runtime sse2 | sse4.2 CPUIDs under SDE: the arms of the run-time dispatch          [pr]
│   ├── count_x64                  instruction counts with the variants and the pipeline (D175)               [full]
│   ├── fuzz                       pipeline_json, five minutes                                                [full]
│   └── soak                       the ten-minute long_run_no_drift                                           [full]
├── build_x64_linux_sde_avx512     target-cpu=sapphirerapids, profile emulated, nextest archive
│   ├── test_x64_linux_sde_avx512  the suite under SDE, two partitions                                        [pr]
│   └── test_x64_linux_sde_avx512_fixed_arch   the backend with AVX-512 fixed at compile time                 [pr]
├── build_x64_win                  clippy · xtask check · bare-metal · nextest archive · unit tests
│   └── test_x64_win               the whole suite from the archive                                           [pr]
├── build_arm64_linux              clippy · bare-metal · nextest archive · unit tests
│   ├── test_arm64_linux           the whole suite from the archive, SDRK_EXPECT_LEVEL=neon                   [pr]
│   ├── test_arm64_linux_fallback  --cfg sdrk_fallback_only                                                   [pr]
│   └── test_arm64_linux_fixed_arch  neon without std, with and without the tables                            [pr]
├── test_wasm32                    WASI under wasmtime: scalar, simd128, relaxed SIMD, the static tables
├── test_wasm_web                  the package in Node, Chrome and Firefox, JavaScript tests on the tarball
├── build_wasm_pkg                 the package on arm64_linux and x64_win
├── build_m4_qemu                  the suite on the Cortex-M4F, quick profile
│   └── test_m4_qemu_selfcheck     a failing case must fail the run (pr: when xtask/ or targets/ changed)    [pr]
├── build_m55_qemu, build_rv32_qemu (emulator image; the ESP32 targets are paused in CI, D201)
├── test_emu                       m4 full; m55, rv32 quick and full                                          [full]
├── count_x64_linux_sde_avx512, count_arm64_linux, count_wasm32, count_m4_qemu                                [full]
└── sweep_x64_linux (sse2 | sse4.2 | avx2), sweep_x64_linux_sde_avx512 (4 parts), sweep_arm64_linux (native),
    sweep_wasm32, sweep_fallback_int: convert_out for every f32, mag CQ15 for every raw pair                  [full]
```

The test programs are built once per platform. The build job packs them with `cargo nextest archive`; the test job of the same platform downloads the archive, extracts it into its checkout (the same path, so the paths the tests compiled in stay valid) and runs it without building. nextest does not run doc tests, so `build_x64_linux` runs them with `cargo test --doc`. The jobs that need other flags (fixed levels, the fallback, the sweeps in release) build themselves; their cargo caches are separate.

Run the same commands locally:

The jobs under SDE use the profile `emulated` (optimised like release but without LTO and with overflow checks; debug builds of the end-to-end tests take hours in an emulator) and leave out the command-line examples, whose smoke tests start the program under test as a child process (natively covered by `test_x64_linux`).

```sh
cargo nextest run --workspace --all-features && cargo test --workspace --all-features --doc
RUSTFLAGS="--cfg sdrk_fallback_only" cargo test --workspace --all-features --target-dir target/fallback-only
SDRK_EXPECT_LEVEL=sse4.2 RUSTFLAGS="-C target-cpu=x86-64-v2" cargo test -p sdrk-backend-simd -p sdrk \
    --features sdrk/fft-tables,sdrk/nco-tables --target-dir target/static-x86-64-v2
SDRK_EXPECT_LEVEL=avx2 RUSTFLAGS="-C target-cpu=x86-64-v3" cargo test -p sdrk-backend-simd \
    --target-dir target/backend-no-tables-x86-64-v3
cargo xtask sde-cpuid sse2 "$PWD/target/cpuid-sse2.def"
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -cpuid_in $PWD/target/cpuid-sse2.def -chip_check_cpuid 1 --" \
    cargo test -p sdrk-backend-simd --features std,fft-tables,nco-tables --profile emulated \
    --target x86_64-unknown-linux-gnu --target-dir target/sde-runtime
cargo test --release -p sdrk-backend-simd --features fft-tables,nco-tables --test convert_out_sweep -- --ignored
cargo test --release -p sdrk-backend-simd --features fft-tables,nco-tables --test spectrum_mag -- --ignored
RUSTFLAGS="-C target-cpu=sapphirerapids" CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -spr --" \
    cargo test --workspace --all-targets --profile emulated --exclude fm-receiver --exclude fm-transmitter \
    --target x86_64-unknown-linux-gnu --target-dir target/avx512
RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=$PWD --" \
    cargo test --target wasm32-wasip1 -p sdrk-core -p sdrk-backend-simd -p sdrk -p sdrk-blocks
RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=$PWD --" \
    cargo test --target wasm32-wasip1 -p sdrk-core -p sdrk-backend-simd -p sdrk -p sdrk-blocks --features sdrk/relaxed-simd
RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=$PWD --" \
    cargo test --target wasm32-wasip1 -p sdrk-core -p sdrk-backend-simd -p sdrk -p sdrk-blocks --features sdrk/nco-tables,sdrk/fft-tables
cargo xtask wasm-pkg --relaxed && cargo xtask wasm-test --env node --variant all
node --test "bindings/wasm/js-tests/*.test.mjs"
```

## The suite on microcontrollers

The `no_std` suite of the testkit (`all_cases()`, `run_suite`, the line protocol) runs bare-metal in five emulators: a Cortex-M4F (every commit), a Cortex-M55, RISC-V `rv32imac` (nightly), the ESP32-P4 and the ESP32-S3 (paused in CI, D201; local runs only), with the quick and the full profile. `cargo xtask emu` builds, runs and judges them; the CI jobs are `build_m4_qemu` (every pull request), `build_*_qemu` (build only, every pull request) and `test_emu` (nightly). See [embedded.md](embedded.md).
