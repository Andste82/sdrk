# M37 — Packet ports and the crate `sdrk-proto`

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M34](M34-multi-input.md) · Next: [M38](M38-symbol-sync.md)

## Goal

Blocks and pipelines can carry **packets** (byte messages with a sample index and flags) next to sample streams. A new crate `sdrk-proto` holds protocol logic without signal processing (packet type, CRCs, a simple test framing). Pipeline inputs and outputs of type `packets`, a port pair `Clocked<S, Packets>` for encoders that are driven by a sample clock, JSON Lines packet files in `sdrk-io`, and the two kinds `frame_encode`/`frame_decode` prove the plumbing end to end.

## Why

Every digital mode of M38–M44 ends in bytes: a demodulator gives soft bits, a decoder turns them into packets, and a transmitter turns packets back into symbols. Without a packet type each mode would invent its own output format, and a flow could not deliver decoded data to the application.

## Order

M37 depends on M34 (multi-input nodes; `Clocked` is a two-input port pair and must follow its rules). It can run in parallel with M35, M36 and M38. **From M37 on, rule R2 includes `crates/proto`** (see the table below).
## Ground rules (identical in every milestone — read them completely before you start)

These rules apply to every line of code in this repository. They are repeated in every milestone file so that each milestone is self-contained. If a rule and a milestone-specific instruction ever conflict, stop and ask; do not guess.

### R1 — Work order and scope

1. A milestone starts when every milestone named in its *Depends on* line is delivered (merged on `main`); up to M25.1 the order is strictly numeric. Milestones that do not depend on each other may be implemented in parallel, each on its own branch; the second one to merge rebases onto the first. A milestone may only use what its dependencies (and their dependencies) delivered. Never implement items of a later milestone "while you are at it".
2. Every milestone must leave the repository in a state where **all** of these commands succeed from a clean checkout:
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - `cargo test --workspace --all-features`
   - `cargo xtask check` (exists from M00 on; runs the architecture checks in R2–R6)
   - `cargo xtask bare-metal` (exists from M00 on; builds every heap-free crate for `thumbv7em-none-eabihf`)
   - plus every additional command listed in the milestone's *Acceptance* section.
3. **Host platforms:** everything (build, tests, `xtask`, golden generator) must work on Linux x86_64, Linux arm64 (Raspberry Pi, 64-bit OS) and Windows x86_64. No shell scripts or Makefiles; tools are Rust (`xtask`) or the Python golden generator; paths are built with `Path::join`; line endings are fixed by `.gitattributes`. CI builds the workspace and runs the whole test suite, `xtask` and the golden generator on all three hosts; the variants that depend only on the CPU architecture (SIMD levels fixed at compile time, builds without the static tables, the fallback-only build) run on one host per architecture (x86-64 on Linux, AArch64 on Linux arm64).
4. When something is ambiguous, choose the simplest solution that satisfies the written tests, and record the decision in `docs/decisions-log.md` (one line: date, milestone, decision, reason). Never silently change a public API defined by an earlier milestone; if a change is unavoidable, update every caller and the affected docs in the same milestone and log it.
5. Naming: write **SDRk** only in prose, code comments (incl. rustdoc) and documentation; everything else in code is lowercase `sdrk` (identifiers, crate/package/module/file names, string literals such as error messages). Exception: fixed upper-case machine tokens — the `SDRK_` environment-variable prefix, the emulator protocol markers `SDRK-SUITE`/`SDRK-CASE`/`SDRK-PANIC` and the wire magic `b"SDRK"`.
6. Plan (living document): https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1 — snapshot in [`docs/PLAN.md`](../PLAN.md). The plan explains *why*; the milestones say *what exactly* to build.

### R2 — Layers and dependencies

The folder of a crate is its layer. A crate may depend only on crates in **lower** layers (or, for the testkit, on any library layer). `cargo xtask check` enforces this by reading `cargo metadata` and the crate paths.

| Layer (folder) | Crates | May depend on (workspace crates) | `no_std`, no heap |
| --- | --- | --- | --- |
| `crates/core` (lowest) | `sdrk-device` | — | yes |
| `crates/core` | `sdrk-core` | `sdrk-device` | yes |
| `crates/backends` | `sdrk-backend-*` | `sdrk-core`, `sdrk-device` | yes |
| `crates/facade` | `sdrk` | core, backends | yes |
| `crates/proto` | `sdrk-proto` | `sdrk-device` | yes |
| `crates/blocks` | `sdrk-blocks` | facade, core, proto | yes |
| `crates/pipeline` | `sdrk-pipeline` | blocks, proto, facade, core | no (std/alloc allowed) |
| `crates/io` | `sdrk-io` | core (types only) | no |
| `crates/testing` | `sdrk-testkit` | any library crate above | `no_std` by default; feature `std` adds file I/O and proptest helpers |
| `bindings/*`, `examples/*`, `targets/*`, `benches/*`, `xtask` | — | anything | no |

Additional rules:
- Within `crates/core` the rank is by crate name: `sdrk-device` (rank 0) < `sdrk-core` (rank 1); `sdrk-core → sdrk-device` is allowed, the reverse is not. All other layers are one rank per folder.
- Backends never depend on each other. Only the facade `sdrk` knows the backends.
- `sdrk-proto` (from M37 on) holds protocol logic without signal processing (bit and byte framing, checksums, packet types); it depends on `sdrk-device` only, and nothing below `crates/blocks` depends on it.
- Only `sdrk-testkit` and test code use the `reference` feature of `sdrk-core`.
- External crates allowed in heap-free crates: `num-complex` (default-features = false, features = ["bytemuck"]), `libm`, `bytemuck`, `heapless`, `fearless_simd` (backends only). Anything else needs a line in `docs/decisions-log.md` and must pass `cargo deny`.

### R3 — `no_std` and no heap (applies to every crate marked "yes" above)

1. The crate root starts with `#![no_std]`. The crate never writes `extern crate alloc;` and never uses `Vec`, `Box`, `String`, `Rc`, `Arc`, `BTreeMap` or anything else that allocates. `cargo xtask bare-metal` builds these crates for `thumbv7em-none-eabihf`, which has no global allocator, so any accidental allocation fails the build.
2. Memory is provided by the caller. State types are generic over their storage: a field that holds samples has a type parameter `S: AsRef<[T]> + AsMut<[T]>` (read-only data: `S: AsRef<[T]>`). On a microcontroller the caller passes `[T; N]` or `&mut [T]`; in upper layers the caller may pass `Vec<T>`. Constructors validate lengths and return `Err`, never panic.
3. Fixed-capacity containers come from `heapless` when needed.
4. No function may keep more than 4 KiB on its stack (WASM and MCU stacks are small). Large buffers belong in caller-provided storage. `clippy.toml` sets `stack-size-threshold = 4096` and the workspace enables `clippy::large_stack_frames` and `clippy::large_stack_arrays` as errors.
5. Math functions come from `libm` (e.g. `libm::sinf`, `libm::atan2f`, `libm::log10f`), never from `f32::sin()` & co. This keeps code `no_std` **and** makes results identical on every platform because `libm` is pure Rust.
6. Upper layers (`pipeline`, `io`, `bindings`, `examples`, `xtask`, testkit feature `std`) may use `std` and allocate. The only `std` use allowed below the heap boundary besides the testkit's `std` feature is runtime CPU-feature detection in backend crates behind their `std` feature (M13); it never allocates. The pipeline allocates only while *building*; after `build()` returns, processing must not allocate (tested with a counting global allocator in M10).

### R4 — Clean interfaces

1. Each crate exposes a small, documented public API through `lib.rs` re-exports; everything else is `pub(crate)`. Module layout inside a crate is an implementation detail.
2. Public enums that may grow are `#[non_exhaustive]`. Public structs with invariants have private fields, a validating constructor (`new(...) -> Result<Self>`) and getters.
3. Kernels follow the **one trait per kernel** pattern defined in M02: the trait is generic over the sample type, `Fallback` (in `sdrk-core`) implements it for every supported sample type, `Reference` (feature `reference`) implements it as the f64/integer specification, backends implement only the variants they accelerate, and the facade type `sdrk::Sdrk` implements all of them by delegating to the chosen implementation.
4. Kernel calls never allocate, never block, never read a clock, never use randomness, and never print.
5. Every public item has a rustdoc comment; every public function with non-trivial behaviour has a doc example that compiles (`#![deny(missing_docs)]` in every library crate; `cargo test --doc` runs the examples).
6. All public types are `Send`; state types and blocks are also `Sync` unless they need interior mutability (they should not). Add compile-time assertions in each crate's tests: `fn _assert_send<T: Send>() {}` called for every public type.

### R5 — One error-handling scheme for the whole workspace

1. The lowest crate `sdrk-device` defines the shared classification:
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
   #[non_exhaustive]
   pub enum ErrorKind {
       InvalidArgument, LengthMismatch, BufferTooSmall, OutOfRange, Unsupported,
       CapacityExceeded, InvalidState, ResourceLimit, Parse, Validation,
       Io, Device, Timeout, Disconnected, Internal,
   }
   pub trait SdrkError: core::error::Error {
       fn kind(&self) -> ErrorKind;
   }
   ```
2. Every library crate defines **its own** `pub enum Error` (`#[non_exhaustive]`, `Debug`, `Clone`, `PartialEq`) and `pub type Result<T, E = Error> = core::result::Result<T, E>;`. The enum implements `core::fmt::Display` (lowercase message, no trailing period), `core::error::Error` and `SdrkError`. Heap-free crates keep their error variants allocation-free (`&'static str` and integers only). Upper crates may carry `String` paths (e.g. `nodes.ddc.params.frequency_hz`).
3. Wrapping: an upper crate's error has a variant like `Core(sdrk_core::Error)` with `From` impls; `kind()` forwards to the inner error. `source()` returns the wrapped error.
4. **No panics in library code for any input that can reach a public API.** Every library crate root contains:
   `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::todo, clippy::unimplemented, clippy::indexing_slicing)]`
   Tests may opt out with `#[cfg_attr(test, allow(...))]` on the test module only. Validate lengths and parameters once at the public entry point; inner loops then iterate with `zip`, `chunks_exact`, `iter_mut` instead of indexing. Where indexing is unavoidable, use `.get()` and return `Error::Internal` if it ever fails.
5. Integer arithmetic that may overflow is written explicitly as `wrapping_*`, `saturating_*` or `checked_*`. The test profile keeps `overflow-checks = true`.

### R6 — Determinism and global state

1. No global mutable state: no `static mut`, no `static` with interior mutability (`Cell`, `RefCell`, `Mutex`, atomics, `OnceCell`, `LazyLock`), no `lazy_static`. Lookup tables are `const`/`static` immutable data, generated by `build.rs` where needed. `cargo xtask check` greps the source of library crates for these patterns and fails.
2. Same backend + same input = bit-identical output, independent of how the input is split into blocks. Every stateful kernel and block gets a block-size-invariance test (testkit helper from M03).
3. No reliance on CPU floating-point modes (flush-to-zero, rounding modes). Feedback states are flushed explicitly: `sdrk_core::num::flush_denormal(x)` returns 0.0 when `|x| < 1e-30`.
4. No hash-map iteration order in any output; use `BTreeMap` or ordered `Vec`s in upper layers.
5. Fused multiply-add is used only where the numeric contract allows it; the reference and the fallback never use FMA.
6. Little-endian only: `sdrk-device` contains `#[cfg(target_endian = "big")] compile_error!("sdrk supports little-endian targets only");`.

### R7 — Numeric conventions (from the plan's Decisions table)

| Topic | Rule |
| --- | --- |
| Q15 format | `Q15(i16)`, value = raw / 32768, range −1 … 1 − 2⁻¹⁵ |
| Rounding | round half up: add ½ LSB, then arithmetic right shift (`(x + (1 << (s - 1))) >> s` on a signed integer) |
| Overflow | always saturate to `[-32768, 32767]`, never wrap |
| Multiply | `sat16((a as i32 * b as i32 + (1 << 14)) >> 15)`; therefore `-1 × -1 = 32767` |
| Accumulation | reference and fallback accumulate Q30 products in `i64`, round and saturate once at the end; accelerated variants may use narrower accumulators if their headroom is documented |
| f32 → Q15 | computed in **f64**: `v = (x as f64) * 32768.0`; NaN → 0; otherwise `floor(v + 0.5)` clamped to `[-32768, 32767]` (so +1.0 → 32767, ±∞ → ±limit). Doing this in f32 would round twice and differ from the Python model |
| Q15 → f32 | `raw as f32 / 32768.0` (exact) |
| Complex Q15 | `CQ15 { re: Q15, im: Q15 }`, `#[repr(C)]`, I before Q — byte-identical to interleaved little-endian `cs16` |
| Denormals | flush feedback state to 0.0 below 1e-30 (R6.3) |
| Kernel classes | *bit-exact* kernels: fallback output equals the reference output bit for bit (Q15 and f32). *tolerance* kernels: output must reach the minimum SNR stated in the kernel's numeric contract (`docs/kernels.md`) against the f64 reference |

### R8 — Test vectors and golden samples (full specification in M04)

1. Golden samples live in `tests/golden/vectors/<kernel>/<case>.json` plus binary arrays `<case>.<array>.bin`. They are produced **only** by `tests/golden/generate.py` (Python 3.12, NumPy and SciPy with versions pinned exactly in `tests/golden/requirements.txt`). Never edit them by hand; never generate them with code from this repository.
2. Every case is deterministic: fixed seed via `numpy.random.default_rng(seed)`, fixed sizes, parameters stored in the JSON. Regenerating must produce byte-identical files (`python tests/golden/generate.py --check`).
3. Expected values are computed in float64 (`cf64`/`f64`). For bit-exact Q15 cases the generator computes the expected result with the integer model in `tests/golden/sdrk_golden/q15.py`, which implements R7 in NumPy `int64`.
4. Rust tests load cases with `sdrk_testkit::golden::load("<kernel>", "<case>")` (feature `std`). Tolerances live in the case JSON, not in Rust code.
5. Signal metric used everywhere: `SNR_dB = 10·log10( Σ|ref|² / Σ|ref − out|² )` over samples `skip..len` (`skip` stored per case; `+∞` when the error is zero). Bit-exact comparisons use `==` on raw bits. Tolerance classes (M04): `bit_exact`, `tolerance` (minimum SNR), `max_abs_err` (largest absolute error, optionally only where the reference is above a level).
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 16 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).
7. Tests that use `sdrk-testkit` live in integration tests (`<crate>/tests/*.rs`) or in `crates/testing/sdrk-testkit/tests/`, never in `#[cfg(test)]` unit modules of a crate the testkit itself depends on (a dev-dependency cycle would compile two copies of that crate and produce confusing trait errors).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** the crate `crates/proto/sdrk-proto` (`no_std`, depends on `sdrk-device` only): `Packet`, flags, `crc` (CRC-16/CCITT-FALSE, CRC-16/X-25, CRC-32/ISO-HDLC, CRC-24/ADS-B as table-driven functions), the test framing `frame` (§3); the port type `Packets` and the port pair `Clocked<S, Packets>` in `sdrk-blocks`; blocks `FrameEncodeBlock` and `FrameDecodeBlock`; the pipeline model for packet inputs, outputs and edges; the runner API `push_packets`/`read_packets`; pull mode for packet inputs; JSON Lines packet files in `sdrk-io`; the WebAssembly binding; schema; `cargo xtask` layer check for `crates/proto`; docs (`docs/architecture.md`, `docs/pipeline.md`, `docs/blocks.md`, `docs/io.md`, `docs/wasm.md`, `README.md` crate list), `CHANGELOG.md`, decisions log.

**Out:** any modulation (M38–M44), protocol decoders (AX.25, RDS: M39, M41), SigMF annotations of packets (backlog), packets larger than 256 bytes (backlog: a fragmenting framing), forward error correction (backlog).

## Specification

### 1. `sdrk-proto`

```rust
pub const MAX_PACKET_BYTES: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Packet {
    /// Sample index at the decoder's input where the packet starts (decoders), or 0 (sources);
    /// a TX encoder ignores it.
    pub index: u64,
    pub flags: PacketFlags,
    len: u16,
    data: [u8; MAX_PACKET_BYTES],
}
impl Packet {
    pub fn new(index: u64, flags: PacketFlags, data: &[u8]) -> Result<Self, ProtoError>; // len > 256 → TooLong
    pub fn data(&self) -> &[u8];
    pub fn len(&self) -> usize;
}
// `PacketFlags(u32)`: CRC_OK = 1, CRC_FAIL = 2, CORRECTED = 4 (bits were corrected), TRUNCATED = 8; bits 16–31 are free for a protocol (documented per decoder).
pub mod crc { pub fn crc16_ccitt_false(d: &[u8]) -> u16; pub fn crc16_x25(d: &[u8]) -> u16; pub fn crc32(d: &[u8]) -> u32; pub fn crc24_adsb(d: &[u8]) -> u32; }
pub mod frame { /* §3 */ }
```

No `bitflags` dependency: a newtype with `const`s and `contains`. The CRCs are tested against the check values of the CRC catalogue (`"123456789"`: 0x29B1, 0x906E, 0xCBF43926, and for CRC-24/ADS-B the residue 0 of a known Mode S frame `8D4840D6202CC371C32CE0576098`).

R2 table from M37 on (also in `docs/architecture.md` and `cargo xtask check layers`):

| Folder | Crate | May depend on | `no_std` |
| --- | --- | --- | --- |
| `crates/proto` | `sdrk-proto` | `sdrk-device` | yes |
| `crates/blocks` | `sdrk-blocks` | facade, core, proto | yes |
| `crates/pipeline` | `sdrk-pipeline` | blocks, proto, facade, core | no |

### 2. Ports

- `PortType::Packets` is new (description, schema `"packets"`).
- `Packets` implements `Ports` (M26 §1) with `Ref<'a> = &'a [Packet]`, `Mut<'a> = &'a mut [Packet]`, `COUNT = 1`. `Packet` is not a `StreamElem`: `Chain` and `Fork` do not accept packet ports (compile error, documented).
- A block with a packet output reports `BlockTiming { ratio: Rate::ZERO, kind: RatioKind::Packets { max_per_input: Rate }, delay }`; `max_per_input` bounds the packets per input sample for buffer sizing (e.g. `1 / min_frame_bits / samples_per_bit` for a decoder). A block with a packet input and a stream output reports the ratio of output samples per **input stream sample** of its clock (`Clocked`) or, for a pure encoder, `RatioKind::Source`: its output rate is the `rate_hz` of the packet input (§4).
- `Clocked<S, Packets>`: port a is a stream `S` (the clock; its values may be ignored), port b packets. Lockstep applies to port a only: `process` gets `n` clock samples and the queued packets, and reports in `Progress` how many clock samples and how many packets it consumed. Packet port b carries no clock domain and no group delay; the pipeline neither checks its rate nor aligns it (M34 §3, §4 apply only to stream ports).

### 3. Test framing (`sdrk_proto::frame`) and the kinds `frame_encode`/`frame_decode`

The framing is for tests and simple custom links:

| Field | Bits | Content |
| --- | --- | --- |
| preamble | `preamble_bytes`·8 | `0x55` repeated (default 4 bytes) |
| sync word | 32 | `0x1ACFFC1D` (CCSDS ASM) |
| length | 8 | payload length `L` (0 … 253; `L` + 3 ≤ 256) |
| payload | `8·L` | the packet bytes |
| CRC | 16 | CRC-16/CCITT-FALSE over length and payload, big-endian |

Bits are sent most significant bit first. `frame::encode(packet, out: &mut [u8]) -> usize` writes the bits as bytes `0`/`1`; `frame::Decoder` is a bit-by-bit state machine (`push_bit(bit, index) -> Option<Packet>`) that searches the sync word with at most `max_sync_errors` bit errors (default 0), then reads length, payload and CRC; a CRC mismatch gives the packet with `CRC_FAIL` only when `emit_bad` is set (default off).

- `frame_encode` (`FrameEncodeBlock`): In `Packets`, Out `f32`, one output sample per bit: `+1.0` for 1, `−1.0` for 0; between packets it emits `idle` samples (`0.0`, Choice `zero`/`preamble`: `preamble` sends `0x55` bits). Parameters `preamble_bytes` (I64, build, 4; 0 … 32), `idle` (Choice, build, `zero`). Timing `RatioKind::Source`.
- `frame_decode` (`FrameDecodeBlock`): In `f32` (soft bits, one per bit; `> 0` is 1), Out `Packets`. Parameters `max_sync_errors` (I64, build, 0; 0 … 4), `emit_bad` (Bool, build, false). `index` is the input index of the first sync word bit. `max_per_input` = 1 / 56.
- Kinds with cost 1 each.

### 4. Pipeline

- **Inputs:** `"type": "packets"` with `rate_hz`: the sample rate of the stream that a pure encoder (`RatioKind::Source`) produces from this input. It is required when a `Source` encoder reads the input (`missing_param` otherwise) and ignored with a warning when only `Clocked` port b reads it (that encoder takes its rate from its clock). A packet input starts its own clock domain (its name), so a stream that an encoder produces from it cannot be mixed with the stream of another input (M34 `clock_domain`).
- **Outputs:** `"type": "packets"` with `queue` (packets, 1 … 65 536, default 64) and the existing `drop` policies, counted in packets.
- **Edges:** packet ports connect only to packet ports (`type_mismatch` otherwise); a packet source may fan out (every reader gets every packet). Port `b` of a `Clocked` node is the packet port.
- **Runner:** `push_packets(input, &[Packet])` (all or nothing, as `push`), `read_packets(output, &mut [Packet]) -> usize`, `available` counts packets. A pure encoder in push mode produces the samples of the packets it has; with no packet queued it produces nothing.
- **Pull mode:** `PullSource::provide` may return `InputData::Packets(&[Packet])` for a packet input. An empty slice from a packet input means "nothing to send now", not "exhausted": the encoder then emits its idle signal for the samples the output still needs (an idle encoder never underruns). A packet input ends only with `end_burst` of the output.
- **Buffers:** a packet buffer holds `ceil(max_chunk · max_per_input) + 4` packets; `Report::memory_bytes` counts `size_of::<Packet>()` per slot.
- **Schema, WebAssembly:** `"packets"` as input and output type; the binding exposes `pushPackets(input, array)` and `readPackets(output)` with packets as `{ index, flags, data }` objects (`data` a hex string), as in the JSON Lines format.

### 5. `sdrk-io`: JSON Lines

`sdrk_io::packets::{write_jsonl, read_jsonl}`: one packet per line, `{"index":123,"flags":1,"data":"1acf"}` (lower-case hex, no spaces). The reader rejects unknown fields and data longer than 256 bytes with the line number.

### 6. Proof flow and loop test

`examples/flows/frame_loop.json` (+ YAML): `msg` packets (`rate_hz` 48000) → `enc` = `frame_encode` → `dec` = `frame_decode` → `out` packets. **Test `frame_loop_exact`:** 1000 random packets (lengths 0 … 253, seed 479) pushed in random groups come out unchanged with `CRC_OK`, in order, with `index` equal to the sample position of their sync word; in pull mode with an idle source between packets the result is the same.

## Tests

1. `sdrk-proto`: CRC check values; `Packet::new` limits; `frame` round trip for every length 0 … 253; the decoder finds a frame after arbitrary leading bits and with `max_sync_errors` errors in the sync word; property test with random bit errors: a corrupted frame never gives `CRC_OK` unless the corruption is in the preamble.
2. Blocks: description rules (packet ports appear as `packets`), exact storage, chunk invariance of `frame_decode` with 20 chunk plans.
3. Pipeline: validation of packet edges (`type_mismatch` corpus file), packet input without `rate_hz` (`missing_param`), clock domains with a packet input, `drop_oldest` on a full packet queue, the loop test of §6 in push and pull mode.
4. `sdrk-io`: JSON Lines round trip and error lines.
5. `cargo xtask check layers` rejects a dependency `sdrk-core → sdrk-proto` (negative test fixture as for the existing layers).

## Acceptance

R1.2 commands; `cargo test -p sdrk-proto`; `cargo build -p sdrk-proto --target thumbv7em-none-eabihf`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo test -p sdrk-io`; `cargo xtask schema --check`; `cargo xtask check layers`; `cargo xtask bare-metal`.

## Deliverables checklist

- [ ] Crate `sdrk-proto` with `Packet`, flags, CRCs, `frame`; layer table and xtask check updated
- [ ] `PortType::Packets`, `Packets` ports, `Clocked<S, Packets>`, timing kinds `Packets` and `Source`
- [ ] Blocks and kinds `frame_encode`, `frame_decode`
- [ ] Pipeline packet inputs, outputs, edges, push and pull mode, schema, WebAssembly
- [ ] JSON Lines in `sdrk-io`
- [ ] Flow `frame_loop` with YAML twin and loop test (seed 479)
- [ ] Docs (`architecture.md`, `pipeline.md`, `blocks.md`, `io.md`, `wasm.md`, README), CHANGELOG, decisions log
