<p align="center"><img src="assets/sdrk-logo.webp" alt="SDRk Logo"></p>

# SDRk

**Software-defined radio building blocks in Rust: fast, exact, and the same everywhere — from a microcontroller to the browser.**

SDRk (pronounced like the German word *stark* [ʃtaʁk], "strong") is a library of signal-processing kernels and blocks for receiving and transmitting radio signals: mixers, filters, resamplers, synchronizers and complete modulators and demodulators. You wire them together as a JSON flow or directly in Rust, push in samples from a file, an SDR dongle or a WebSocket, and read out audio, spectra, bits or decoded packets.

```text
  IQ samples ──► ddc ──► fm_demod ──► audio
              └► spectrum ──────────► FFT frames
```

## Why another SDR library?

Mature SDR toolkits exist. SDRk is built for the places they do not reach, and it is held to measurable standards in all of them.

- **One code base for every target.** The same kernels and blocks run on a PC, a Raspberry Pi, in the browser as WebAssembly, and heap-free (`no_std`) on microcontrollers. A flow you design on the desktop runs unchanged in a web app.
- **Speed you can verify.** The kernels have hand-written SIMD paths for SSE2, SSE4.2, AVX2, AVX-512, NEON and WebAssembly SIMD where vectors pay off, and a portable fallback that runs everywhere; the facade picks the fastest path for the target. Speed is measured in simulated CPU cycles per sample on Intel, AMD and Arm core models and compared with established libraries such as VOLK, GNU Radio, CMSIS-DSP and FFTW at the same instruction-set level.
- **Exact results.** Every kernel is checked against a 64-bit floating-point golden model (generated with NumPy and SciPy) with a stated accuracy. Results are deterministic: the same input gives the same bits on every platform and for any way the input is cut into pieces.
- **Rust all the way down.** Memory safety without a garbage collector, no hidden allocations in the signal path, and a strict layering: everything below the pipeline is `no_std` and works on memory you provide.
- **Self-describing.** Every block publishes its ports, parameters, units, ranges and timing. Flows are validated against a JSON Schema before they run, with diagnostics that point at the exact problem — useful for editors, UIs and AI assistants that generate flows.

## Getting started

SDRk is a Cargo workspace. You need a stable Rust toolchain.

```sh
git clone https://github.com/Andste82/sdrk.git
cd sdrk
cargo build --release
cargo test --workspace
```

Try a complete FM broadcast receiver on a recording:

```sh
cargo run --release -p fm-receiver -- \
    --input rec.cu8 --format cu8 --rate 2400000 \
    --flow examples/flows/fm_receiver.json --output audio.wav
```

The `examples/flows` folder holds ready-made receivers and transmitters as JSON, each with a commented YAML twin.

### Using SDRk in your project

SDRk is not published on crates.io yet. Use it as a Git dependency:

```toml
[dependencies]
sdrk-pipeline = { git = "https://github.com/Andste82/sdrk" }   # JSON / YAML flows
sdrk-blocks   = { git = "https://github.com/Andste82/sdrk" }   # blocks and chains in Rust
sdrk          = { git = "https://github.com/Andste82/sdrk" }   # the kernels, with SIMD dispatch
```

Pin a `rev` or `tag` for reproducible builds. The crates are dual-licensed under MIT or Apache-2.0.

## Flows: SDR as JSON

A flow describes a signal graph: **inputs** (sample streams or packets), **nodes** (blocks with their parameters), **edges**, and **outputs** (streams, frames or packets). This is the FM receiver from the diagram above:

```json
{
  "sdrk": 1,
  "name": "fm-receiver",
  "settings": { "precision": "f32", "max_chunk": 4096 },
  "inputs":  { "rf": { "format": "cu8", "rate_hz": 2400000 } },
  "nodes": {
    "tune":  { "kind": "ddc",      "params": { "frequency_hz": 600000, "decimation": 10, "cutoff_hz": 100000 } },
    "demod": { "kind": "fm_demod", "params": { "deviation_hz": 75000, "deemphasis_us": 50, "audio_decimation": 5 } },
    "fft":   { "kind": "spectrum", "params": { "fft_size": 1024, "averaging": 4 } }
  },
  "edges": [
    { "from": "rf",    "to": "tune" },
    { "from": "rf",    "to": "fft" },
    { "from": "tune",  "to": "demod" },
    { "from": "demod", "to": "audio" },
    { "from": "fft",   "to": "spectrum" }
  ],
  "outputs": {
    "audio":    { "type": "stream", "capacity": 16384, "drop": "error" },
    "spectrum": { "type": "frames", "queue": 4, "drop": "drop_oldest" }
  }
}
```

When a flow is built, SDRk checks every block, port type, sample rate and parameter. It inserts the format conversion of the input, computes the latency of every output, and refuses what cannot work: a wrong type on an edge, a rate a block cannot handle, a value out of range. The same model can be written in YAML. A flow runs in `f32` or, for small processors, in `q15` fixed point.

### Sample formats

Raw samples go straight into a flow, in the format the hardware delivers; the input names it, for example `"format": "cu8"`, and an exact, SIMD-accelerated conversion follows automatically:

| | Formats |
| --- | --- |
| **In** | `cu8` (RTL-SDR), `cs8` (HackRF), `cs16`, `sc16_q11` (bladeRF), `cs12_lsb16`, `packed12`, `packed10`, `i2s32_msb` (I2S on microcontrollers), `cf32`; real `rs16` and `rf32` (Airspy R2 and Mini, direct sampling, sound cards); and custom formats described by container, bit width, alignment, byte order and scale |
| **Processing** | `f32` / `c32` floating point, or `q15` / `cq15` fixed point for small processors |
| **Out** | back to `cs16`, `cs8`, `cu8` and real `rs16`, `rs8`, `ru8` for transmitters and DACs |

Running it in Rust:

```rust
use sdrk_pipeline::{BuildOptions, InputData, Registry};

let flow = std::fs::read_to_string("examples/flows/fm_receiver.json")?;
let mut pipeline = Registry::standard().build_json(&flow, &BuildOptions::default())?;

let mut audio = vec![0.0f32; 4096];
for chunk in iq_bytes.chunks(2 * 4096) {
    pipeline.push("rf", InputData::Raw(chunk))?;          // feed the input
    let n = pipeline.read_stream("audio", &mut audio)?;  // read 48 kHz audio
    play(&audio[..n]);
}
pipeline.set_param("tune", "frequency_hz", serde_json::json!(700_000))?; // retune while running
```

Beyond this, a running pipeline gives you:
- parameters that change at runtime, including at an exact sample time;
- metadata and events carried with the samples (timestamps, gaps, lock and squelch changes);
- a pull mode for transmitters, which produces samples on demand;
- packet inputs and outputs for framed data.

## Chaining blocks in Rust

When you do not need JSON, connect blocks directly. `chain!` puts blocks in a row and is itself a block, so chains nest. It needs no heap: every buffer comes from an arena you provide, which is how SDRk runs on a microcontroller.

```rust
use sdrk::{Rate, C32};
use sdrk_blocks::{
    chain, AlignedBytes, ArenaStorage, Block, ConvertBlock, ConvertConfig,
    DdcBlock, DdcConfig, FmDemodBlock, FmDemodConfig, ParamValue,
};

let rate = Rate::hz(2_400_000)?;
let mut mem = AlignedBytes([0u8; 65536]);            // all memory of the chain
let mut arena = ArenaStorage::new(&mut mem.0);

let ddc = DdcConfig { frequency_hz: 600e3, decimation: 10, ..DdcConfig::default() };
let mut rx = chain!(&mut arena, 256;
    ConvertBlock::<C32, _>::new(&ConvertConfig::default(), rate, &mut arena)?,
    DdcBlock::<C32, _>::new(&ddc, rate, &mut arena)?,
    FmDemodBlock::<C32, _>::new(&FmDemodConfig::default(), Rate::hz(240_000)?, &mut arena)?
);

rx.set_param("1.0.frequency_hz", ParamValue::F64(610e3))?;  // stage 1.0 is the ddc
```

Every block knows its timing (rate ratio, delay, minimum input) and its storage needs, so a chain can tell you exactly how much memory it takes before you build it.

## In the browser: WebAssembly

The pipeline is available to JavaScript and TypeScript, in browsers and in Node. The package ships a WebAssembly SIMD build and picks it automatically where the browser supports it.

```sh
cargo install wasm-bindgen-cli --version 0.2.129 --locked
cargo xtask wasm-pkg          # builds the npm package sdrk-wasm
```

```js
import { init, Pipeline } from "sdrk-wasm";

await init();                                  // SIMD build where available
const pipeline = Pipeline.fromJson(flowJson);  // the same JSON flow as in Rust
pipeline.push("rf", iqBytes);                  // Uint8Array from a WebSocket or file
const audio = new Float32Array(4096);
const n = pipeline.readStream("audio", audio); // hand it to Web Audio
pipeline.setParam("tune", "frequency_hz", 700000);
```

Validation, the JSON Schema and a `FlowBuilder` are available in JavaScript as well, so a web UI can build and check flows before it runs them. One pipeline per Web Worker keeps the audio thread free.

## On microcontrollers

Everything up to the blocks is `no_std` and allocation-free. It is tested bare-metal in emulators on:
- Arm Cortex-M4 and Cortex-M55;
- RISC-V 32;
- ESP32-P4 and ESP32-S3.

The same test suite runs on every target as on the desktop, against the same golden data. Fixed-point `q15` paths keep receivers practical on cores without a floating-point unit.

## Hardware and files

SDRk includes drivers for RTL-SDR dongles and the HackRF One, behind a common device interface. It reads and writes raw IQ files, SigMF recordings and WAV audio.

## Documentation

- [Architecture](docs/architecture.md): the layers and their rules
- [Pipeline and flows](docs/pipeline.md): the JSON model, validation, runtime, YAML and the JSON Schema
- [Blocks](docs/blocks.md): the block interface, chains, composite blocks and parameters
- [Backends and SIMD](docs/backends.md) and [performance](docs/performance.md): how speed is measured and compared
- [WebAssembly](docs/wasm.md) and [embedded targets](docs/embedded.md)
- [Devices](docs/device.md), [file I/O](docs/io.md) and [testing](docs/testing.md)
- [Examples](examples/README.md) and [contributing](CONTRIBUTING.md)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
