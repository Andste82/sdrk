# WebAssembly bindings

The crate `sdrk-wasm` (`bindings/wasm`, `cdylib` and `rlib`) exposes the pipeline API of [pipeline.md](pipeline.md) to JavaScript, in browsers and in Node 20 or later. The bindings only configure: they build a pipeline from JSON (or from the `FlowBuilder`), set parameters (immediately or at a sample index or time), push and pull data and read streams, frames and events. All processing stays in Rust. Data crosses the boundary once per block: `push` copies the typed array into a buffer that was allocated when the pipeline was built, `readStream` copies out of one; nothing allocates per call.

The package `sdrk-wasm` (unscoped; whether the name is free on npm is not checked yet, see the decisions log and `docs/open-items.md`) is assembled by `cargo xtask wasm-pkg`. There is no `wasm-pack` (its upstream organisation was archived): xtask calls `cargo build --target wasm32-unknown-unknown --release` and `wasm-bindgen --target web` directly.

```js
import { init, Pipeline, FlowBuilder } from "sdrk-wasm";

await init();                                   // SIMD128 build where available, else scalar
const pipeline = Pipeline.fromJson(flowJson);   // throws SdrkError with diagnostics
pipeline.push("rf", cu8Bytes);                  // Uint8Array
const audio = new Float32Array(4096);
const n = pipeline.readStream("audio", audio);  // samples read
pipeline.free();
```

## Layers of the crate

- `engine` (plain Rust, no `wasm-bindgen`): `Engine` wraps a `sdrk_pipeline::Pipeline` with the buffers of the data path and a `WasmError` that carries the fields of the JavaScript error. It runs, and is tested, on the host (`tests/engine.rs`, `tests/fm_native.rs`).
- `lib.rs`: the exports of `wasm-bindgen` (camelCase names) and the conversion between JavaScript values and the engine.
- `js/error.js`: the class `SdrkError`, imported by `wasm-bindgen` as a snippet (it ends up in `snippets/` of the package).
- `pkg-template/`: `package.json`, the loader `index.js`, `builder.js`, the hand-written types `index.d.ts` and `builder.d.ts`, `README.md`.

The crate does not add `#![forbid(unsafe_code)]`. It relies on the workspace lint `unsafe_code = "deny"` like every other crate and needed no `allow` for the expansion of the macros of `wasm-bindgen` 0.2.129: it contains no `unsafe` code and no `allow(unsafe_code)`. `console_error_panic_hook` is installed in debug builds only; the library code cannot panic for any input (the lints of R5.4 are on).

## API

```ts
version(): string
schema(): string                       // JSON Schema text
validate(flowJson: string): Report     // never throws on an invalid flow
backends(): Record<string, string>     // kernel -> "simd" | "fallback" | "chained" for this build

class Pipeline {
  static fromJson(flowJson: string): Pipeline;
  push(input: string, data: Samples, meta?: BlockMeta): void;
  pull(output: string, out: Samples, source: PullSource): PullResult;
  readStream(output: string, out: Samples): number;
  readFrame(output: string, frame: Float32Array): FrameMeta | null;
  available(output: string): number;
  chunkSize(output: string): number;
  dropped(output: string): bigint;
  setParam(node: string, name: string, value: number | boolean | string, at?: ParamTime): void;
  pollEvent(): SdrkEvent | null;
  beginBurst(output: string, txTimeNs?: bigint): void;
  endBurst(output: string): void;
  latency(output: string): { groupDelay: number; maxBuffered: number };
  config(): string;                    // lossless JSON
  report(): Report; stats(): NodeStats[]; reset(): void; free(): void;
}
```

`index.d.ts` has the complete types (`Report`, `Diagnostic`, `SdrkEvent`, `BlockMeta`, `FrameMeta`, ...). Beyond the specified surface the package has `chunkSize` and `dropped` (needed to size the buffers and to see what a drop policy discarded), `BlockFlags` (the bit values of `BlockMeta.flags`), `simdSupported()`, `defaultVariant()` and `loadedVariant()`.

### Data layout

The typed array must match the stream; a mismatch throws `SdrkError { kind: "InvalidArgument" }` (a complex stream whose array has an odd number of elements throws `LengthMismatch`).

| Stream | Typed array | Layout of the elements |
| --- | --- | --- |
| raw bytes (an input with a `format`; an output with a `format`) | `Uint8Array` | the bytes of the format, for example CU8 `I Q I Q ...` |
| `c32` (complex `f32`) | `Float32Array` | interleaved: `re, im, re, im, ...` |
| `f32` | `Float32Array` | one value per sample |
| `cq15` | `Int16Array` | interleaved `re, im, ...`; value = raw / 32768 |
| `q15` | `Int16Array` | one value per sample |

All counts are in samples; for a complex stream a typed array of length `2 * n` holds `n` samples, and `readStream` returns `n`. For raw streams the unit is the byte. `push` takes at most `settings.max_chunk` samples (bytes for raw: `max_chunk` times the bytes of one sample, rounded up); more throws `LengthMismatch`. `readStream` reads as much as fits `out` (it copies through the output buffer of `chunkSize(output)` samples, that is `max_chunk` samples, at most 65 536, and four times as many bytes for a raw output, in as many steps as needed). `pull` makes one pull call through that buffer and may return fewer samples than `out` can hold, so a caller that needs more calls it again. A burst boundary ends a pull call (see [pipeline.md](pipeline.md)).

`readFrame` needs a `Float32Array` of exactly the frame length (else `LengthMismatch`) and returns `{ firstSample: bigint, averaged: number, exponent: number }` or `null`.

### Metadata, 64-bit values and time

`meta` of `push` is `{ start?, flags?, anchor?, txTimeNs? }`. Without `start` the block follows the previous one (the next index of the input), so that a block with only an `anchor` or `flags` does not create a gap. `flags` is an OR of `BlockFlags` (`GapBefore = 1`, `Overflow = 2`, `Underrun = 4`, `BurstStart = 8`, `BurstEnd = 16`, `Late = 32`, `Retuned = 64`). An `anchor` is `{ index, unixNs, source? }` with `source` one of `"device"`, `"host"`, `"gps"`, `"unknown"` (default). 64-bit values (`index`, `start`, `txTimeNs`, `unixNs`, `firstSample`, `missing`, `count`, the counters of `stats()`) are `bigint` in everything the library returns; where it takes a value, a `number` that is an integer in the safe range is accepted as well.

`setParam(node, name, value, at)`: `at` is `undefined` (the next sample the node processes), `{ input, index }` (at a sample index of a pipeline input) or `{ unixNs }` (at an absolute time; needs an anchor). The change shows up as a `paramApplied` event. Timed changes and their rules are in [timing.md](timing.md).

### Events

`pollEvent()` returns `null` or an object with `type`, `output` (name) and `index` (`bigint`):

| `type` | More fields |
| --- | --- |
| `gap` | `missing: bigint` |
| `overflow`, `underrun`, `retuned`, `burstEnd`, `late` | |
| `anchor` | `anchor: { index, unixNs, source }` (delay-compensated for the output) |
| `burstStart` | `txTimeNs?: bigint` |
| `paramApplied` | `node`, `param` (name), `value`, `late: boolean` |
| `eventsLost` | `count: bigint` |

### Pull mode

`pull(output, out, source)` produces `out`'s worth of a transmit output. `source(input, max)` is called whenever the path needs input data: it returns a typed array of the input's type with at most `max` samples, or `null` when it is exhausted. A source that returns the wrong type throws `InvalidArgument`, one that returns more than `max` samples `LengthMismatch`; an exception thrown by the source comes out of `pull` unchanged. `pull` returns `{ samples, meta }` where `meta` is `{ start: bigint, flags: number, anchor?, txTimeNs? }` of the first sample. `beginBurst(output, txTimeNs?)` and `endBurst(output)` frame bursts; see the burst rules in [pipeline.md](pipeline.md).

### Errors

Every failing call throws an `SdrkError` (a subclass of `Error`, `name === "SdrkError"`) with:

| Field | Meaning |
| --- | --- |
| `kind` | the name of the Rust `ErrorKind`: `"InvalidArgument"`, `"LengthMismatch"`, `"CapacityExceeded"` (an output with the drop policy `error` is full; read, then push again), `"InvalidState"`, `"Validation"`, ... |
| `code` | for a failed build: the diagnostic code of the first problem (`"out_of_range"`, ...) |
| `path` | for a failed build: its location (`"nodes.tune.params.decimation"`) |
| `diagnostics` | for a failed build: all problems, `{ path, code, message }` |
| `message` | text for humans |

`validate()` returns the same diagnostics in `errors` of the `Report` and never throws, so a UI can show every problem at once, including syntax errors (`code: "syntax"`, with line and column in the message). The field names of the report are those of the Rust `Report` (`memory_bytes`, `cost_per_input_sample`, `nodes`, `outputs`, ...). Calls that throw leave the pipeline usable. The bindings never panic; an exception of a source function is not turned into an `SdrkError`.

## Package

```text
package.json          "sdrk-wasm", type "module", exports "./index.js", types "./index.d.ts"
index.js              loader: init({ variant, wasm }), simdSupported(), BlockFlags, FlowBuilder
builder.js            FlowBuilder
index.d.ts, builder.d.ts
sdrk_simd.js/.wasm    built with +simd128
sdrk_scalar.js/.wasm  built with --cfg sdrk_fallback_only (backends() reports only "fallback"/"chained")
sdrk_relaxed.js/.wasm built with +simd128,+relaxed-simd and the feature relaxed-simd (only with --relaxed)
snippets/             SdrkError
README.md, LICENSE-*
```

`init({ variant: "auto" })` (the default) validates a one-function module with a `v128` instruction (inline bytes, no dependency): `sdrk_simd` where that works, `sdrk_scalar` otherwise. The relaxed build is chosen only by `init({ variant: "relaxed" })`. `init` returns the API of the build it loaded (`{ variant, Pipeline, validate, schema, backends, version, SdrkError }`), and the first successful call also binds the module-level exports of the package (ES live bindings). Two builds can be loaded side by side (each has its own `SdrkError` class). In Node the file is read with `fs` (Node 20 or later); in a browser it is fetched next to `index.js` (`instantiateStreaming`; serve `application/wasm`); `init({ wasm })` takes bytes, a compiled module or a URL.

| Variant | Instructions | Results |
| --- | --- | --- |
| `scalar` | no SIMD; the portable fallback of `sdrk-core` everywhere | identical bits on every platform and engine (tested in Node, Chrome and Firefox) |
| `simd` | WebAssembly SIMD128 | meets the numeric contracts of [kernels.md](kernels.md); bits differ from the scalar build in the tolerance kernels |
| `relaxed` | SIMD128 and relaxed SIMD | as `simd`, but the multiply-adds of the `C32` kernels may be fused or not depending on the host CPU, so the last bits differ between hosts, see [backends.md](backends.md) |

### Build

```sh
cargo install wasm-bindgen-cli --version 0.2.129 --locked   # must equal the pinned wasm-bindgen
cargo xtask wasm-pkg [--relaxed] [--no-opt] [--no-pack]
```

`wasm-pkg` checks that `wasm-bindgen --version` equals the version pinned (`=x.y.z`) in `bindings/wasm/Cargo.toml`, builds the variants with their `RUSTFLAGS` in `target/wasm-pkg-build/<variant>`, runs `wasm-bindgen --target web`, runs `wasm-opt -O3` if it is in `PATH` (skipped otherwise, with a message), assembles `target/wasm-pkg/` and runs `npm pack` (tarball in `target/wasm-pkg-tgz/`) if `npm` exists. It prints the sizes of the modules and warns above the soft limit of 1.5 MiB per variant.

Sizes of the modules with `wasm-opt -O3` (version 0.0.1): `sdrk_simd.wasm` 0.95 MiB, `sdrk_scalar.wasm` 0.93 MiB, `sdrk_relaxed.wasm` 0.95 MiB (999 832, 976 342 and 999 900 bytes). The tarball is 1.1 MB.

### Workers

A `Pipeline` is not shared between threads; it lives in the WebAssembly memory of the instance that created it. Use **one pipeline per Worker**: create the Worker, `init()` and build the pipeline inside it, and send data as transferable `ArrayBuffer`s (`postMessage(buffer, [buffer])`), which moves them without a copy. Results come back the same way. Where the page is cross-origin isolated (response headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, so that `crossOriginIsolated` is true) you can use `SharedArrayBuffer` ring buffers between a producer (for example a WebUSB reader on the main thread) and the Worker that owns the pipeline; the Worker copies from the ring into `push`. The WebAssembly memory of the pipeline itself is never shared, and the package does not use threads.

## Tests

| Test | What | How to run |
| --- | --- | --- |
| `bindings/wasm/tests/engine.rs`, `fm_native.rs` | the engine on the host: layouts, errors, events, timed parameters, pull and bursts, FM receive | part of `cargo test --workspace` |
| `bindings/wasm/tests/wasm.rs` | the JavaScript API from Rust with `wasm-bindgen-test` | `cargo xtask wasm-test [--env node\|chrome\|firefox] [--variant simd\|scalar\|relaxed\|all]` |
| `bindings/wasm/js-tests/*.test.mjs` | the packed tarball installed into a temporary project with `npm install`: FM receive with a JS meter, the builder against `examples/flows/fm_receiver.json`, the loader | `node --test "bindings/wasm/js-tests/*.test.mjs"` after `cargo xtask wasm-pkg --relaxed` |

`wasm-test` runs `cargo test -p sdrk-wasm --release --test wasm --target wasm32-unknown-unknown` with `wasm-bindgen-test-runner` and the `RUSTFLAGS` of the variant (the tests need `--release`: unoptimised the filters are too slow). The browsers need `chromedriver` or `geckodriver` in `PATH` (or `CHROMEDRIVER`/`GECKODRIVER`); `WASM_BINDGEN_TEST_WEBDRIVER_JSON` points to a `webdriver.json` for options (for example `--no-sandbox` for Chrome as root).

**The FM receive test.** The E2E-RX signal (an FM broadcast at +600 kHz with a 1 kHz tone, 0.25 s of CU8 at 2.4 MS/s, generated by the testkit) goes through `fm_receiver.json` in blocks of 4096 samples; the audio must reach SINAD >= 40 dB (the f32 flow reaches 100 dB).

**Reproducibility.** `bindings/wasm/tests/fixtures/fm_receiver.json` holds the SHA-256 of the input, the number of audio samples, the SHA-256 of the audio bytes (`f32`, little endian) and the SINAD of a native run with `RUSTFLAGS="--cfg sdrk_fallback_only"`. The native test `fixture_matches_the_fallback_build` (only built with that flag; `SDRK_UPDATE_FIXTURE=1` rewrites the fixture instead of comparing) pins it, and the **scalar** WASM build must produce the same audio bit for bit, in Node, Chrome and Firefox and through the JS loader (`variant: "scalar"`). Everything runs on the fallback and `libm`, so the bits are the same everywhere. The SIMD128 and the relaxed build are different backends and are only bound to the contracts: their SINAD must be at least the fixture SINAD minus 0.5 dB. The JS tests generate the same input bytes with a port of the generator (the `Math.sin` and `Math.cos` of V8 and the `libm` crate implement the same fdlibm algorithms) and check its SHA-256 first.

CI job `wasm-bindings` (ubuntu-24.04): builds the package with `--relaxed`, runs `wasm-test` in Node, Chrome and Firefox for the variants, and `node --test`. The Windows and Linux arm64 jobs build the package only (no browsers).
