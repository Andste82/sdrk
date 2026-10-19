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
| `burst_timing_loopback` | Two 100 ms bursts of a constant carrier at `t0 + 1 s` and `t0 + 1.5 s`; the first down-converter output sample above half the plateau, converted with the delay-compensated anchor, lies within one output sample period of `tx_time + delay`. |
| `two_stations_no_crosstalk` | Two stations at -400 kHz and +400 kHz, one flow with two down-converter / demodulator branches: SINAD at least 35 dB, the other tone at most -50 dB. |
| `long_run_no_drift` | Start index `2^50`, 20 s at 240 kHz through the tuned resampler (+37.5 ppm) and `fm_demod`: the output counts follow the exact formula and the time of the last anchor equals the analytic value to the nanosecond. The ten-minute variant is `#[ignore]` and runs nightly (`cargo test --release -p sdrk-system-tests -- --ignored`). |
| `event_propagation_exact` | 50 random gap, anchor, retune and overflow events appear on `audio` at exactly the index of the mapping rule of [timing.md](timing.md#index-domains-in-the-pipeline). |
| `sigmf_roundtrip_pipeline` | A recording with a gap and annotations through `SigmfWriter` and `SigmfReader` gives the same audio and events as pushing the blocks directly. |

## The SIMD backend in the suite and in CI

The suite of `sdrk-testkit` runs through the facade, so every case exercises the SIMD backend where the platform has one. Ten cases `backend_vs_fallback_*` (`convert`, `convert_out`, `rotate`, `fir`, `fir_decimate`, `fir_interpolate`, `mag_sq`, `quad_demod`, `fm_mod`, `fft`) cover every accelerated kernel and type with the lengths 0, 1, 3, 15, 16, 17, 255 and 4096 (the lengths that fit the arena of the tier: the quick tier takes the short ones), extreme inputs (full scale, -32768, NaN and infinity in `f32` where defined) and, for the stateful kernels, random chunk plans that must give the bits of one call. Bit-exact kernels are compared bit for bit with the fallback; tolerance kernels with the contract of `docs/kernels.md` against the reference (`rotate` 120 dB, `fir` family 110 dB, `quad_demod` 100 dB / 2 LSB, `fft` 110 dB, `fm_mod` 120 dB).

The tests of the crate `sdrk-backend-simd` (`crates/backends/sdrk-backend-simd/tests`) compare the kernels with the fallback for all lengths of the vector tails, with `LENGTHS = 0, 1, 2, 3, 7, 15, 16, 17, 31, 255, 257, 4096`, with random chunk plans, extreme inputs and invalid calls (same error values as the fallback). They run at the level the build selects: the default build at SSE2, `--features std` at the best level of the CPU, `-C target-cpu=...` at the level of the flag.

`sdrk::backend_for` is tested in `crates/facade/sdrk/tests/facade.rs`: every accelerated kernel reports `simd` when `sdrk::SIMD_BACKEND` is true and `fallback` otherwise (`--cfg sdrk_fallback_only`, no `simd` feature, or a target without SIMD).

| CI job | What |
| --- | --- |
| `tier1` | the whole suite with the default `simd` backend on three hosts, a second run with `-C target-cpu=x86-64-v3` on x86-64 hosts, a third run with `RUSTFLAGS="--cfg sdrk_fallback_only"` so that the fallback stays covered |
| `avx512-sde` | `cargo test --workspace --profile emulated` with `-C target-cpu=sapphirerapids` under Intel SDE (`sde64 -spr --` as the cargo runner) |
| `aarch64-qemu` | `cargo test --workspace --profile emulated` cross-compiled for `aarch64-unknown-linux-gnu` under `qemu-aarch64` |
| `wasm-wasmtime` | the crates that build for WASI (`sdrk-device`, `sdrk-core`, `sdrk-backend-simd`, `sdrk`, `sdrk-blocks`) for `wasm32-wasip1` under wasmtime, with `+simd128`, without it and with `+simd128,+relaxed-simd` (feature `sdrk/relaxed-simd`), with wasmtime's default stack |
| `wasm-bindings` | the WebAssembly package: `cargo xtask wasm-pkg --relaxed`, `wasm-bindgen-test` in Node, headless Chrome and headless Firefox for the variants, `node --test` on the packed tarball; Windows and arm64 build the package only ([wasm.md](wasm.md)) |
| `count` | `cargo xtask count --platform x86_64` against VOLK and rustfft; nightly also AVX-512, AArch64, wasm32 and the pipeline ([performance.md](performance.md)) |

Run the same commands locally:

The two emulator jobs use the profile `emulated` (optimised like release but without LTO and with overflow checks; debug builds of the end-to-end tests take hours in an emulator) and leave out the command-line examples, whose smoke tests start the program under test as a child process (natively covered by `tier1`).

```sh
RUSTFLAGS="--cfg sdrk_fallback_only" cargo test --workspace --all-features --target-dir target/fallback-only
RUSTFLAGS="-C target-cpu=sapphirerapids" CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -spr --" \
    cargo test --workspace --all-targets --profile emulated --exclude fm-receiver --exclude fm-transmitter \
    --target x86_64-unknown-linux-gnu --target-dir target/avx512
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" \
    cargo test --workspace --all-targets --profile emulated --exclude fm-receiver --exclude fm-transmitter \
    --target aarch64-unknown-linux-gnu --target-dir target/aarch64
RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=$PWD --" \
    cargo test --target wasm32-wasip1 -p sdrk-core -p sdrk-backend-simd -p sdrk -p sdrk-blocks
RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=$PWD --" \
    cargo test --target wasm32-wasip1 -p sdrk-core -p sdrk-backend-simd -p sdrk -p sdrk-blocks --features sdrk/relaxed-simd
cargo xtask wasm-pkg --relaxed && cargo xtask wasm-test --env node --variant all
node --test "bindings/wasm/js-tests/*.test.mjs"
```
