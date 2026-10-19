# sdrk-wasm

SDRk for JavaScript. Describe a software-defined-radio signal chain as JSON (or with the
`FlowBuilder`), build it, push samples in and read samples, frames and events out. All the signal
processing runs in WebAssembly built from Rust; JavaScript only configures and moves blocks of
data (one copy per block).

```js
import { init, Pipeline } from "sdrk-wasm";

await init(); // loads the SIMD128 build where the engine has it, the scalar build otherwise

const pipeline = Pipeline.fromJson(flowJson); // throws SdrkError with diagnostics if invalid
pipeline.push("rf", cu8Bytes); // Uint8Array from the radio
const audio = new Float32Array(4096);
const n = pipeline.readStream("audio", audio); // samples read
pipeline.setParam("tune", "frequency_hz", 610_000); // retune now ...
pipeline.setParam("tune", "frequency_hz", 620_000, { input: "rf", index: 1_000_000n }); // ... or at a sample
for (let e; (e = pipeline.pollEvent()); ) console.log(e.type, e.output, e.index);
```

The full reference is `docs/wasm.md` in the repository. The types are in `index.d.ts`.

## Typed arrays

The typed array must match the stream; a mismatch throws `SdrkError` with `kind` `"InvalidArgument"`.

| Stream | Typed array | Layout |
| --- | --- | --- |
| raw bytes (an input with a `format`) | `Uint8Array` | bytes of the format, for example CU8 `I Q I Q ...` |
| `c32` (complex `f32`) | `Float32Array` | interleaved `re, im, re, im, ...` |
| `f32` (real) | `Float32Array` | one value per sample |
| `cq15` (complex Q15) | `Int16Array` | interleaved `re, im, ...`, value = raw / 32768 |
| `q15` (real Q15) | `Int16Array` | one value per sample |

Counts are in samples (bytes for raw streams): a complex stream of 100 samples is a typed array
of length 200. 64-bit values (`index`, `txTimeNs`, `unixNs`) are `bigint`.

## Variants

| File | Build | Used when |
| --- | --- | --- |
| `sdrk_simd.wasm` | WebAssembly SIMD128 | automatically, if the engine validates a v128 instruction |
| `sdrk_scalar.wasm` | no SIMD, portable fallback only | automatically otherwise |
| `sdrk_relaxed.wasm` | SIMD128 and relaxed SIMD | only with `init({ variant: "relaxed" })` (if the package was built with `--relaxed`) |

The scalar build gives the same bits on every platform. The SIMD builds meet the numeric contracts
of the kernels but differ in the last bits; the relaxed build lets the engine fuse
multiply-adds, so its last bits depend on the host CPU.

## Workers

A `Pipeline` belongs to one thread. Run one pipeline per Worker and move data with transferable
`ArrayBuffer`s (`postMessage(buffer, [buffer])`). When the page is cross-origin isolated (headers
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`) you can use
`SharedArrayBuffer` ring buffers between a producer (for example a WebUSB reader) and the Worker
instead. The WebAssembly memory itself is not shared; the Worker copies from the ring buffer into
the pipeline.

## Serving the files

In a browser `init()` fetches `sdrk_<variant>.wasm` next to `index.js`. Serve it as
`application/wasm`. To load it from elsewhere pass `init({ wasm: url })`. In Node 20 or later
the file is read from disk.
