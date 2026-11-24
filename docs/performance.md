# Performance

SDRk's speed is judged in **simulated cycles per sample**, against an established library (VOLK, FFTW, rustfft, KissFFT, KFR, CMSIS-DSP, GNU Radio and liquid-dsp where they have the function; ESP-DSP for the embedded backends to come) at equal accuracy against the f64 reference. The cycles are simulated by `llvm-mca` for every level with a core model (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `avx2-runtime`, and the scalar fallback of each). Instructions per sample are measured only where no core model exists: `simd128` (wasm32) and the in-order embedded cores (Cortex-M4 today). The [measurement policy](#measurement-policy) says what is compared with what; the committed results are in `benches/results/cycles/<platform>.json` and in the generated [cycle tables](#cycle-results) below.

| Cycles per sample vs. reference library, on both core models of the level | Meaning |
| --- | --- |
| ≤ 1.00× | goal: equal or better (recorded as `goal` in the results) |
| ≤ 1.10× | release limit; `cargo xtask check cycles` fails above it for a `release` entry |
| ≤ 1.50× | allowed only while a kernel is under development (`dev`), never in a release |

The history of how these numbers came about (rounds, tunings, decisions) is in [decisions-log.md](decisions-log.md) by decision number, and the pre-cleanup text of this file, with its per-milestone narrative and the instruction tables of the out-of-order levels, is in the git history before M13.1 (commit `d73066a4` and earlier).

## Measurement policy

This is the maintainer's policy (D32 to D34, D2700 to D2711). It applies to every new or changed kernel, entry of `benches/limits.json` and row of the results.

**Gates.** One speed measure on the levels with a core model: the cycles per sample, judged against the comparison of the entry on **both** models of the level (table below).

| Gate | Rule | Where |
| --- | --- | --- |
| Release (D14 in cycles, D2708) | `release` entry: ours ≤ 1.10 × comparison; `dev` entry: ≤ 1.50 ×; `info` is not judged. Both the reference model and the counter-model must meet the cap; `goal` (1.00) and `warn` (1.10) are remarks | `cargo xtask check cycles`, part of `cargo xtask check` and of CI |
| SIMD not slower than its fallback (D321) | `cycles(<k>) ≤ cycles(<k>_fallback)` on both models of every level, in the two decimals of the tables; a violation needs a `cycle_waiver` naming an open item | `cargo xtask check cycles` |
| Run-time detection (`avx2-runtime`, D2702) | ours with run-time detection ≤ 1.02 × our own fixed `avx2` build on the models of `avx2`: the cost of the dispatch | `cargo xtask check cycles` |
| `simd128` and Cortex-M4 | instructions per sample: `simd128` judged against the references with the same caps (1.10, 1.50) by `cargo xtask count --platform wasm32`; Cortex-M4 report only, no limits | CI nightly (`count_wasm32`, `count_m4_qemu`) |
| Accuracy (D34, D2749) | SNR against the f64 golden by the golden tests; per entry and level the SNR of ours and of the comparison on the workload of the cycles (`snr_db` of the cycle records): ours at most 1 dB below the comparison, else an `accuracy_waiver` | `cargo test`, `cargo measure` golden cases, `cargo xtask check cycles` |
| Accuracy trade (D2760; the 3 dB threshold is proposed, the maintainer's to confirm) | a `release` entry over 1.10 × in cycles stays `release` with an `accuracy_trade` while its SNR is at least 3 dB above the comparison's and its ratio within 1.50 × on both models | `cargo xtask check cycles` |

A violation without a waiver fails the gate; an entry over its cap is never loosened: it becomes `dev` or `info` only with an open item that carries its numbers. The cap of 1.10 stays. [Gates in detail](#gates-in-detail) names the statuses and waivers.

Each rule below is marked as **decided** (applies now), **planned** (decided, but the tooling does not exist yet) or **proposed**.

### Three questions, three measures

| Question | Measure | Source | Kind |
| --- | --- | --- | --- |
| How fast is it in the core? | cycles per sample, and the efficiency (below) | `llvm-mca` on the execution counts of callgrind, SDE or QEMU ([level 2](#speed-simulated-cycles-per-sample-level-2)) | simulated, relative only |
| How much work does it do? | instructions per sample, only for `simd128` (wasmtime fuel) and Cortex-M4 (QEMU), where no core model exists or instructions are close to cycles | `cargo xtask count --platform wasm32` / `cortex-m4` | measured, exact |
| How accurate is it? | SNR against the f64 golden | golden vectors ([kernels.md](kernels.md#accuracy-policy)) | measured, exact |

Speed and accuracy are judged independently. A kernel that is faster but less accurate than allowed fails, and so does one that is accurate but too slow.

### What is compared with what (D32)

Decided. The goal is an optimal branch for every kernel at every SIMD level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`), and an efficient portable fallback. Whether a branch is optimal is only known by comparing it with a well-known, well-tested library that does the same thing.

| Our side | Fair comparison | Examples |
| --- | --- | --- |
| SIMD branch of a level | SIMD code of an established library at the **same level**, the **same algorithm class** and the **same work** | VOLK, KFR, FFTW, IPP, rustfft |
| Scalar fallback | established **scalar** code of the same algorithm | CMSIS-DSP generic C, the loops of GNU Radio |
| A branch for which no such library exists | the efficiency of the branch ([below](#efficiency-a-measure-without-a-reference)), recorded per kernel | the `Q15` and `CQ15` biquads, which no library has in SIMD on our levels |
| SIMD branch against our own fallback | never the judged comparison against a library (at most `info`); but in simulated cycles per sample the branch must be at least as fast as the fallback of its level on both models (D321, gated by `cargo xtask check cycles`; a violation needs a `cycle_waiver` naming an open item) | a SIMD branch that is not faster than the scalar loop (possibly auto-vectorised by the compiler) has no reason to exist; beating it says nothing yet about whether the branch is good, which is what the library comparison is for |

"The same work" is meant literally: no apples against oranges. Where the reference does less (for example no NaN rule) or more, the difference is named next to the entry, and the entry is only `release` if the comparison is still fair.

### Speed: simulated cycles per sample (level 2)

Instructions per sample measure work, not speed: `cycles per sample = instructions per sample / IPC`, and the IPC ranges from about 0.3 to 5 depending on the code. The usual optimizations for out-of-order cores *add* instructions and make the code faster: several accumulators to break an add chain (`dot`, `fir`), loop unrolling, the wavefront of D18 in `biquad`. A gate on instructions would penalize them. Example: one `f32` biquad section takes 15 instructions and 10.0 cycles per sample on Zen 4 (an IPC of 1.5); the ports would allow 2.5 cycles, but the chain add → broadcast → mul → sub limits it. No instruction count shows that.

The method is decided (D33) and implemented for every level with a core model (D35 for `avx2`, D68 for `sse2`, `sse4.2`, `avx512` and `neon`):

1. **Optimize per instruction-set level, not per CPU.** The branch of a level must use its instruction set efficiently. Different CPU generations run it at different speeds, and that is accepted. There are no vendor- or model-specific branches.
2. **Metric.** Cycles per sample simulated by `llvm-mca` in a pinned LLVM version, so the numbers are deterministic. The judgement uses no real hardware and no calibration, and absolute cycles or MS/s are not published as hardware figures: the tables below show them only so that the ratios can be checked.
3. **Only ratios.** `llvm-mca` is typically 10 to 30 % off, with outliers above 2×, and the execution counts of an emulator have weaknesses of their own. Only the ratio ours / reference with the same tool, the same core model and the same counter is used: ours and the reference run in the same emulator, on the same emulated CPU, with the same variant and the same optimization (D32), so systematic errors of the model and of the emulator largely cancel in it (D68). Absolute values may deviate; they do not matter.
4. **One reference model per level and one counter-model of another vendor or core generation.** A branch is judged on the reference model. On the counter-model its ratio may be only slightly worse (the proposed bound is below), so that it is not tuned to the quirks of one model.

   | Level | Reference model | Counter-model |
   | --- | --- | --- |
   | `sse2`, `sse4.2` | `skylake` | `znver2` |
   | `avx2` | `znver3` | `alderlake` |
   | `avx512` (built for `x86-64-v4`, vendor-neutral) | `znver4` | `sapphirerapids` |
   | `neon` | `cortex-a72` (Raspberry Pi 4) | `cortex-a76` (Raspberry Pi 5) |

   **Models of the `neon` pair:** with the pinned LLVM 23.1.1, `cortex-a72` is simulated with the scheduling model of `cortex-a57` (its predecessor) and `cortex-a76` with the model of `neoverse-n1` (the same core design), so the two differ. In LLVM 21 (the Ubuntu `llvm` package) `cortex-a76` also used the `cortex-a57` model and the pair checked nothing; this is one reason why the LLVM version is pinned.

5. **The whole kernel, not a hand-picked loop.** The counters of the levels (profilers; they count executions, the cycles are not their output) give the execution count of every basic block (callgrind at `avx2`, Intel SDE's DCFG at `sse2`, `sse4.2` and `avx512`, a QEMU plugin at `neon`). The hot blocks are found from these counts, and their simulated cycles are weighted by them. This covers FFT stages, prologue, tail and dispatch, and it works for reference libraries without markers (VOLK, KFR, FFTW codelets).
6. **Known blind spots**, recorded with every result: cache and memory bandwidth (every count uses 4096 samples, which fit in L1); memory dependencies across iterations (in-place FFT, filter state in memory, store-to-load forwarding); branch mispredictions in prologue and tail; gathers, divisions, square roots and microcode (low confidence: no value is published for them).


**Instructions are the measure only where no core model exists** (decided; D68, D2700). On in-order cores (Cortex-M, ESP32) instructions are close to cycles, and `simd128` has no core model (the engine's JIT generates the machine code; cycles of it would be informational at most and are not simulated). There the instruction ratio is the gate and the target. On every other level the instruction count is not measured, not gated and not reported (D2700): two measures for one decision invite contradictions, and the cycles track speed.

**Optional plausibility check** (proposed): `perf stat` on a Raspberry Pi 4 (Cortex-A72) for the `neon` branch. It checks the simulated ratios for plausibility and does not calibrate them.


**Derived figure, for orientation only.** `MS/s per GHz = 1000 / cycles per sample` may be computed to estimate how much signal one core can process: it is the simulated in-core throughput with the data in L1, an upper bound. For large blocks of streaming kernels the memory caps it: `samples/s = min(f_CPU / cycles per sample, bandwidth / bytes per sample)`. It is neither published as a result nor gated.

### Gates in detail

`benches/limits.json` names, per kernel and level, `{ "status": "release" | "dev" | "info" }`, and the caps in `_limits` (`release` 1.10, `dev` 1.50, `warn` 1.10, `goal` 1.00, `runtime` 1.02; D14, restated in cycles by D2708). `cargo xtask check cycles` reads the committed results `benches/results/cycles/*.json` (`--results DIR` judges other results, `--partial` a selection) and judges them in two parts: D321 (SIMD against its own fallback) and the release gate (every simulated entry and level against its comparison). The judged number is the ratio of the simulated cycles of the record (`ratio`), not the cost with the mispredictions (D1611 decides between forms of a branch; the cost is recorded beside it).

- `release` needs `ours / comparison ≤ _limits.release` on the reference model **and** the counter-model; `dev` the same with `_limits.dev`; `info` is reported, not judged. `avx2-runtime` is judged against our own `avx2` build with `_limits.runtime` and needs no entry of its own.
- Every judged record (`release` and `dev`) needs its accuracy, `snr_db` `{ "ours", "comparison" }` in dB against the f64 reference of the measurement binary (999: exact): ours more than 1 dB below the comparison fails unless the entry carries an `"accuracy_waiver"` (D34, D2749). A record without `snr_db` fails.
- A `release` entry over the release cap may carry an `"accuracy_trade"` text instead of a `waiver` (D2760): it passes, listed as an accuracy trade, while ours is at least 3 dB more accurate than the comparison and within the `dev` cap (1.50) on both models; when either no longer holds it fails, and within the release cap the trade is a remark. The text names both SNR figures and the ratios. An `accuracy_trade` on a `dev` or `info` entry, beside a `waiver`, or at `avx2-runtime` fails. The tables mark it `(accuracy trade)` in the accuracy column. The threshold of 3 dB is the main session's proposal; the maintainer confirms or changes it.
- An entry may carry a `"waiver"` (ratio), a `"cycle_waiver"` (D321) or an `"accuracy_waiver"` text. A waiver that names an open item needs that item in `docs/open-items.md` (`cargo xtask release-check`); one that names none states its reason. A waived exceedance is printed and does not fail the command; the numbers are never hidden. A waiver that covers nothing, a `dev` entry within `warn`, and a ratio above `warn` are remarks.
- A simulated level fails on an entry that is missing from `limits.json`, on a `release` or `dev` entry whose record failed, and on a listed entry without a record.
- An entry may carry `"own_target"` (a number > 0): the efficiency target of its own-kernel comparison (`ours:<kernel>`, `Reference::OwnKernel`, information), printed as met or missed; it never fails and does not replace the limit of the judged comparison (D102). The `halfband_decimate` entries use 0.60, the `real_to_complex` entries 1.10.
- Every entry is re-judged in cycles by the full CI measurement of M13.1 (D2708): an entry that was `release` by instructions and misses the cap in cycles is not loosened.

### Working rule for optimizing (D37, D68)

Decided. `cargo measure` simulates the cycles of every level with a core model, so a kernel is optimized like this:

| Code | Steered and checked by |
| --- | --- |
| Code shared by all levels: the portable fallback, generic `fearless_simd` code | the simulated cycles of every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`; each on its reference model and counter-model) |
| Code of one level: a hand-written SIMD path for `sse2`, `neon`, `avx512`, ... | the simulated cycles of that level; the cycles of another level say nothing about it |
| `simd128` and in-order cores (Cortex-M, ESP32) | the instruction counts (no core model) |

Why shared code is not steered by one level alone: for the `Q15` biquad fallback (D36) a shorter dependency chain at `avx2` was also faster at `sse2`, `sse4.2`, `avx512` and `neon` in 13 of 16 comparisons (level and core model), but not in all of them (`neon` `cortex-a72` unchanged for `CQ15`; `avx512` `znver4` uncertain, `llvm-mca` and OSACA disagree). The gain at one level is a good guide for shared code, not a proof for the other levels.

### How the cycles are simulated

Implemented in `xtask/src/mca.rs` (D35) and `xtask/src/blockcount.rs` (D68) and run by `cargo measure`:

1. The counter of the level records how often every instruction and every taken jump of the measured process ran, in a run with the counted calls and in a run without them; the difference belongs to the counted calls, as in the [differential method](#method) (for an FFT, the restore of the input is subtracted as well; executions that a subtraction leaves negative, the harness code that only the restore run executes, are deducted from the total, so the total is the count). The runs have the environment of the level (`xtask/src/bench.rs`): the same measurement binary, the same `volk_config` pin, the same references and library path, the same emulated CPU. `objdump` (`aarch64-linux-gnu-objdump` for AArch64 objects) disassembles the executed code from the object files of the measured process (our binary, the static references, VOLK's shared library, libc).

   | Level | Counter of the cycles | Execution counts |
   | --- | --- | --- |
   | `avx2` | valgrind callgrind (`--dump-instr=yes --collect-jumps=yes`) | per instruction and per taken jump |
   | `sse2`, `sse4.2` | Intel SDE `-dcfg` with the CPUID definition of the count (`-cpuid_in`, `-chip_check_cpuid 1`) | per basic block and per edge, between the SSC markers of the count (`-start_ssc_mark`, `-stop_ssc_mark`) |
   | `avx512` | Intel SDE `-dcfg -spr` | as `sse2` |
   | `neon` | `qemu-aarch64` (`-R 0x200000000`, the sysroot of the level) with the plugin `xtask/plugins/bbcount.c` | per translation block and per control transfer between blocks |

   callgrind writes the cost of a call as the self cost of the call instruction, the inclusive cost after the `calls=` line (skipped), and, for a call through a PLT stub, a second self-cost line of the call instruction with the jump of the stub (callgrind's default `--skip-plt=yes` gives the stub's cost to the call site); that line is kept in the total but not on the call, whose block would otherwise count twice its executions and make the loop around it run twice as often (D948). SDE's DCFG gives every basic block of every image as an offset in the image and every edge with its type; the taken branches (the edge types `*BRANCH`, not fall-through, call, return or `REP`) are the jumps. The blocks of the two runs are subtracted per block and expanded into instructions with the disassembly; a block whose disassembly does not have the number of instructions SDE counted is an error, never a silent zero. The QEMU plugin writes the execution count of every translation block, every transfer from the last instruction of a block to a block that does not follow it, and the file mappings of the QEMU process: user-mode QEMU maps the guest's files at `guest_base + address`, so the mappings give the object and the file offset of every block, in the binary and in the shared libraries (VOLK). The control-flow graph leaves out a transfer from a call or a return (callgrind records none; the plugin records every transfer). xtask builds the plugin into `target/tools/libbbcount.so` against the `qemu-plugin.h` of the pinned QEMU (see [Running the measurements](#running-the-measurements)) and rebuilds it when its source changes. `cargo measure --level avx2 --cycle-counter sde` takes the counts of `avx2` from SDE on its Haswell CPU (`-hsw`) instead of callgrind, to validate the SDE path (D68).
2. The code is cut into units for `llvm-mca` in one of two modes (`--cycle-mode`, below); the default is `whole-loop`.
3. Starting from the most executed instructions, units are simulated until 99 % of the executed instructions are covered (at most 64 units). `llvm-mca` simulates each unit as a loop of 200 iterations on the model and gives its cycles per iteration and its port bound (`Block RThroughput`).
4. The cycles and port bounds of the units, weighted by how often each runs, give the cycles per sample and the port bound per sample. The instructions outside the simulated units get the cycles per instruction of the simulated ones (the coverage is reported). The efficiency is `port bound / cycles`; the bottleneck class is `throughput` from an efficiency of 0.8 up and `latency` below.

The two modes:

- **`whole-loop` (default).** The control-flow graph of the executed basic blocks is rebuilt from the counts: a block starts at a jump target, after a jump, call or return, after code that never ran and where the count changes; a taken jump is an edge with its count, the rest of a block's executions falls through to the next block (also over a call: a called function is a graph of its own). The loops are the strongly connected components of the graph; the header of a loop is the block entered most often from outside (its executions minus the edges from inside, so a loop at the start of a function is entered by its calls), and the loops inside it are the components left without the header. Each loop body is simulated as one region per frequent path of one iteration: the blocks from the header to the edge back to it (or out of the loop) in execution order, an inner loop passed as one step (it is simulated as a loop of its own). The paths are searched most frequent first, a path's share of the iterations being the product of the frequencies of its branches, until 99 % of the iterations are covered (at most 8 paths; paths below 0.5 % are left out). Weighted by these shares, the paths reproduce the execution count of every block of the body when its branches are independent. A body whose 8 most frequent paths cover less than half of its iterations (many independent branches) and the blocks outside every loop are simulated as in the per-block mode. Register moves on `znver3` and `znver4`: their `llvm-mca` 23.1.1 models lose the dependency of later instructions on an eliminated `mov r32, r32` (the release path of the `agc` `CQ15` fallback: 24.7 instead of 79.0 cycles per iteration on `znver3`, 21.7 instead of 81.0 on `znver4`; `alderlake`, `sapphirerapids`, `skylake` and `znver2` within 2 cycles of the form below), and an eliminated 64-bit move on a loop-carried chain can lengthen the chain instead (the same path with only the 32-bit move rewritten: 89.0 on `znver3`). On these two models a loop path with moves between general registers is therefore simulated twice: as compiled, and with every such move written as `lea` (`mov %eax,%ecx` -> `lea (%rax),%ecx`, `mov %rax,%rcx` -> `lea (%rax),%rcx`, the same result without an eliminated move). The `lea` form keeps every dependency and costs at most one cycle per rewritten move more than real move elimination, so the true cycles lie in `[lea - moves, lea]`; the form as compiled is taken when it lies in that range, the `lea` form otherwise (then up to one cycle per move on the chain high; `moves_as_lea` in the JSON counts these paths). The other models keep the moves as compiled: on `alderlake` and `sapphirerapids` the unpatched models gave `lea` a throughput of one per cycle (four `lea` 4.0 cycles, four moves 0.04), which would have made the `lea` form a throughput artefact there; D942 corrected that throughput (0.20, 0.33 with a scale), and the rule stays as it is.
- **`per-block`** (the method of D35, kept for comparison). The executed instructions are cut into basic blocks: runs of consecutive instructions with the same count, ended by every jump, call and return. Adjacent blocks with the same count, separated only by code that never ran (at most 256 bytes: a saturation or an error path compiled as a jump), are joined into one loop body. Every unit is simulated as a loop of its own, with the moves as compiled. A loop body that spans several blocks with different counts (a branch whose two sides both run) is cut apart, so a register reused across its blocks becomes a dependency chain of its own or a chain through the blocks is lost: the `agc` `CQ15` fallback reads 26.82 / 50.81 cycles per sample here against 78.32 / 86.42 in `whole-loop` mode (`znver3` / `alderlake`).

AArch64: `objdump` writes an immediate as `#n`, so only `//` starts a comment there, and an address operand (a branch target, the page of `adrp`, a literal) is written as the local label `.Lt` that every simulated unit defines; `llvm-mca` ignores the control flow of a unit.

CET markers, in both modes (D64): code built with `-fcf-protection` (the default of GCC on Ubuntu, so most C references) starts every function and every target of an indirect jump with `endbr64`. On hardware it executes as a `nop` (one decode and dispatch slot, no execution port), but `llvm-mca` 23.1.1 has no scheduling information of its own for it and gives it the generic class `WriteSystem`: 100 cycles of latency on every x86 model, and on `znver3` and `znver4` also the microcoded write of 100 micro-operations on the integer ALUs, so a unit with one `endbr64` took about 51 cycles per iteration on `znver3` and 34 on `znver4` (`endbr64` alone: 51.09 / 34.26 / 0.76 cycles on `znver3` / `znver4` / `alderlake`, `nop` 0.26 / 0.26 / 0.02). Every `endbr64` and `endbr32` is therefore simulated as `nop` on every model (the instruction counts keep it); the `notrack` prefix of indirect jumps is modelled as the plain jump and kept. The references that call a function per sample or per output moved most (whole-loop `znver3` cycles per sample, before / after): liquid-dsp's DC blockers 229.23 / 36.03 (`f32`) and 239.99 / 46.65 (`C32`), CMSIS-DSP's `CQ15` `quad_demod` chain 275.53 / 85.94, the CMSIS-DSP compositions of `rotate` and `fm_mod` `CQ15` 121.32 / 29.23 and 112.18 / 21.41, GNU Radio's `pwr_squelch_cc` 118.10 / 20.51, VOLK's `fir`, `fir_interpolate` and `resample` entries by about 48 to 50 cycles each; on `alderlake` the same entries move by at most 2.4 cycles (the 100 cycles of latency). Our own code is built by `rustc` without CET and has no `endbr64`; it moves only where it calls glibc's `memmove` (Ubuntu builds glibc with CET): `fir_decimate` `C32` 2.42 to 2.40 and `fir_decimate` `CQ15` 29.34 to 24.00 on `znver3`. Every other entry of ours is unchanged to the last digit; the full before / after list is in the decision log (D64).

`cargo measure` simulates both modes. The table shows both (column `mode`, the selected mode first); the JSON of an entry (`target/measure/<kernel>.json`, `cycles.<platform>.<level>.<entry>`) carries `cycle_mode` and the fields of the selected mode at the top (`reference_model`, `counter_model`, `counter_vs_reference`) and the other mode under `other_cycle_mode`; every simulation names its `cycle_mode`, the number of `loops` simulated as regions and of loop paths simulated in the `lea` form (`moves_as_lea`). `--cycle-mode per-block` selects the old mode for the table and the top-level fields. A judgement of a scalar loop with branches takes the `whole-loop` figures; say which mode a number comes from when it is quoted. Validation (`avx2`, `znver3` / `alderlake`): the single-block streaming kernels agree in both modes within 2 % (see the decision log of the round); for the `agc` `CQ15` fallback the release path reads 79.0 / 87.1 and the attack path 82.5 / 91.1 cycles (the whole-path figures of D42, all moves as `lea`: 78.1 / 89.1 and 82.6 / 93.1), uiCA 80.3 and 83.8 on Skylake, 83.3 and 86.0 on Tiger Lake.

Limits of `whole-loop`: a path is simulated as if every iteration took it (branch alternation, mispredictions and the switch between paths are not modelled); the shares assume independent branches (correlated branches can move weight between paths with the same blocks); a loop body that calls a function is cut at the call (the callee is simulated apart); edges between objects or between code clusters more than 4 KiB apart are not followed; the code before and after a loop is simulated as in the per-block mode (each run of blocks as a loop of its own), not as one path per call. `SDRK_MCA_DEBUG=1` prints every simulated unit (mode, addresses, the loop header, path number, share and blocks, instructions, executions, cycles and port bound per iteration), `SDRK_MCA_DEBUG=2` also its instructions. Every unit is simulated as if it repeated back to back with itself. That is the steady state of a loop body; for a block that runs once per call it is an approximation, which is small when loop bodies cover most of the instructions.


### Branches (D1611)

`llvm-mca` does not model branch prediction: every branch is taken as predicted. For a data-dependent branch on the hot path the cost of a form is `cycles + mispredictions × penalty` (D1611), with the mispredictions per sample from `valgrind --tool=cachegrind --branch-sim=yes` on the entry's benchmark workload and a fixed penalty per core model (table below, D1612). `neon` takes no penalty when its assembly is branch-free on the data path, otherwise the rate measured at `sse2`; `simd128` prefers the branch-free form within 10 % of instructions. The lower cost decides, a tie (within 2 %) goes to the branch-free form. The results files record the mispredictions per sample and the cost next to the cycles, for ours and for the reference (`cargo measure --branches` prints them).

| Core model | Level (role) | Misprediction penalty (cycles) | Source |
| --- | --- | --- | --- |
| `skylake` | `sse2`, `sse4.2` (reference) | 20 | Intel 64 and IA-32 Architectures Optimization Reference Manual, 248966-050, section 22.5.6.1 "Branch Mispredicts": "about 20 cycles", the factor of its `%BR.MISP.COST`; the manual gives no figure per core |
| `alderlake` | `avx2` (counter) | 20 | as `skylake` |
| `sapphirerapids` | `avx512` (counter) | 20 | as `skylake` |
| `znver2` | `sse2`, `sse4.2` (counter) | 16 | Software Optimization Guide for AMD Family 17h Models 30h and Greater Processors, 56305 Rev. 3.00, section 2.8 "Optimizing Branching": 12 to 18 cycles, "the common case penalty is 16 cycles" |
| `znver3` | `avx2` (reference) | 13 | Software Optimization Guide for AMD Family 19h Processors, 56665 Rev. 3.00, section 2.8 "Optimizing Branching": 11 to 18 cycles, common case 13 |
| `znver4` | `avx512` (reference) | 13 | Software Optimization Guide for the AMD Zen4 Microarchitecture, 57647 Rev. 1.00, section 2.8 "Optimizing Branching": 11 to 18 cycles, common case 13 |
| `cortex-a72` | `neon` (reference) | 20 (measured) | Measured on a Raspberry Pi 4 (Cortex-A72 r0p3, PMU events `cpu_cycles` and `br_mis_pred`, 2026-11-14): a data-dependent `tbz` with paths of equal length, random against constant, 19.56 cycles per misprediction (24 runs, standard deviation 0.01; 0.4997 extra mispredictions per branch, equal instruction counts), and the slope over a random share of 0 to 100 % 19.61 (standard error 0.06, R² 0.99996); D1628. Arm's Cortex-A72 Software Optimization Guide, UAN 0016A, gives no figure |
| `cortex-a76` | `neon` (counter) | 11 (provisional) | the Arm Cortex-A76 Software Optimization Guide was not available (its download is withdrawn); 11 is Arm's disclosure of the core (2018: 13 stages with the latency of 11), to be confirmed (open item I44-15) |

Where a guide gives a range, the table takes the value the guide calls typical (AMD's "common case"); the ranges are in the source column. The values are in `xtask/src/branches.rs` (`penalty`).

**Counting the mispredictions.** `cargo measure <kernel> --branches` runs, after the cycles, the benchmark workload of every selected entry under `valgrind --tool=cachegrind --cache-sim=no --branch-sim=yes` on the x86-64 build of `sse2`, `sse4.2` and `avx2`, with the differential method ([Method](#method): a run with the counted calls minus a run without them, the same calls and buffers as the cycles, the pinned `volk_config` of the level), for ours and for the judged comparison. Per entry it prints the branches and the mispredictions (`Bcm` + `Bim`) per sample of both sides and, per core model of the level, the penalty, the simulated cycles (in the judged cycle mode, when `--no-cycles` is not given) and the cost `cycles + mispredictions × penalty`; the JSON has it under `branches` (the raw counts `Bc`, `Bcm`, `Bi`, `Bim` included). The build fixes the level of our code; the process runs on the host CPU, so the dispatch of a reference by its own CPU test (rustfft, FFTW) can choose another path than at its profile. Cachegrind's predictor is a simple model (2-bit counters indexed by the address and the recent branch history, the last target for indirect branches), not the one of any core: a branch that follows random data is mispredicted by every predictor at about the same rate, which is what the rule needs, while a branch with a learnable pattern can be predicted better by a core than by cachegrind. `neon` is not branch-simulated (D1611); `avx512` cannot be, because Valgrind runs no AVX-512 (open item I44-16).


### Efficiency: a measure without a reference

`llvm-mca` also gives the port bound of a block (its `Block RThroughput`): the cycles per iteration that the pressure on the execution ports alone would allow, without dependencies. The efficiency of a branch is

```text
efficiency = port bound / simulated cycles
```

For a whole kernel both are sums over its hot blocks, weighted by their execution counts.

- **Near 1:** the execution units are busy; the instruction set is used well, and only less work can make the branch faster.
- **Low:** a dependency chain limits the code (one biquad section: 2.5 / 10 = 0.25); only another algorithm can help, such as the wavefront of D18.

Each result carries its bottleneck class: latency, ports, frontend, or not modelled. A branch without a fair reference gets an efficiency floor of its own: it is set per kernel and level, with its bottleneck class, when level 2 is introduced, and is approved by the maintainer. This replaces the earlier wording "about as efficient as our other SIMD branches".

### Accuracy (D34)

The f64 golden is the mathematical optimum and the first target; an established library is only a yardstick where that optimum cannot be reached. The rules are in [kernels.md](kernels.md#accuracy-policy). For the speed measurement this means: `cargo measure` runs `accuracy <kernel> <impl>` of the measurement binary for both sides of every entry and level, in the environment of the level (SDE, QEMU, the host) and on the workload of the cycles, and writes the SNR against the same f64 model into the cycle record (`snr_db`, D2749); "at most 1 dB below the comparison" is a floor, never the target. A comparison that is less accurate than we are does not lower our goal. `simd128` keeps the same judgement in its count.

**Accuracy trade (D2760).** Accuracy is bought with work: an exact 64-bit sum, a rounding step or a wider intermediate costs cycles a less accurate comparison does not spend. A `release` entry that misses the release cap in cycles therefore stays `release` when it is at least 3 dB more accurate than its comparison and within the `dev` cap, with an `accuracy_trade` text in `benches/limits.json` that names both SNR figures; `cargo xtask check cycles` re-judges it on every measurement and fails it as soon as the advantage drops below 3 dB or the ratio exceeds 1.50. It never covers an accuracy below the comparison, and it is no waiver of the cycles of an entry without that advantage. The threshold is proposed and the maintainer's to confirm.


### Determinism and the reference environment

- **No address-space randomization (D2710, I13.1-1).** Every profiled run (host binary, SDE, QEMU, Valgrind, wasmtime) is started by `setarch -R` where the system permits it, so that stacks, heaps and libraries do not move between runs; the results files record it under `method.address_space`. The development container refuses `personality` (its seccomp profile) and measures without it; the CI container runs with `--security-opt seccomp=unconfined`. The `repeat` input of the CI job measures every partition twice and requires identical files: in run 45359765815 18 of 24 partitions were identical and the rest differed by at most 1.7e-05 relative (D2747), so open item I13.1-1 stays open for the maintainer.
- **The CI job is the reference environment (maintainer, 2026-11-24, D2707).** The committed results are the artifact `cycle-results` of the full CI measurement (`measure.yml`, pinned container image and apt snapshot, the corrected `llvm-mca` of D942 built and cached, `costas` at `neon` 86.13 / 54.10 checked first, D2761). A run on the development machine is for a handful of entries when a step cannot be decided otherwise; it is not committed as a result of record, and a record that drifts from the CI artifact by more than 1 % on either model fails the nightly job, as does a committed record that a complete run lacks (a lost partition); a failed partition fails the run first and leaves no `cycle-results` artifact (D2743).
- **Never wall-clock.** No timing in seconds is a result of this file; the cycles are simulated and the only figures that are published.

### Status of the policy

| Part | State |
| --- | --- |
| Simulated cycles and efficiency | reported by `cargo measure` at `sse2`, `sse4.2` and `avx512` (SDE), `avx2` and `avx2-runtime` (callgrind) and `neon` (QEMU), D68; committed in `benches/results/cycles` (D2704) and gated by `cargo xtask check cycles` (D321, D2708) |
| Release gate in cycles, every entry re-judged | done (D2708, D2746): every entry re-judged on the full CI measurement of D2745; the entries over their cap are `dev` with open items I13.1-4 to I13.1-8 and I24-3 |
| Instructions per sample | only `simd128` and Cortex-M4 (D2700); `count accuracy` stays the accuracy check of the `simd128` counts |
| Fair references for every entry | partly: the entries gated against our own fallback and the kernels without a SIMD branch are listed in issue #20 (groups A to D) |
| Mispredictions per sample of data-dependent branches (D1611) | `cargo measure --branches` at `sse2`, `sse4.2` and `avx2` with the penalties of D1612 (D1613); `avx512` not branch-simulated, treated as `neon` (D1616), the `cortex-a72` penalty measured (D1628), the `cortex-a76` penalty provisional (open item I44-15) |
| `avx512` measurement build and KFR reference at `x86-64-v4` instead of `sapphirerapids` | planned |
| Accuracy against the comparison (D34) | measured per entry and level by `cargo measure` (`snr_db` of the cycle records) and judged by `cargo xtask check cycles` (D2749); the accuracy trade of D2760 with a proposed threshold of 3 dB |
| Accuracy against the achievable optimum | measured and shown per golden case by `cargo measure` (not judged) |
| SNR regression guards, Q15 SNR against an f64 golden | planned ([kernels.md](kernels.md#accuracy-policy)) |
| One command per kernel (`cargo measure`) | golden cases and simulated cycles |

## Method

`cargo measure <kernel>` (alias of `cargo xtask measure`) builds the measurement binary of `benches/kernels` for every level of the platform and profiles it under the counter of the level; `cargo xtask count --platform wasm32|cortex-m4` runs the same binary under an instruction counter.

- **Differential measurement.** `run <kernel> <impl> <k>` does one warm-up call and then `k` calls on the same buffers. It is started twice, with `k = 0` and with `k = K`, and the execution counts (or instructions) per sample are `(count(K) - count(0)) / (K * samples)`. Process start, allocation, set-up (the plans of rustfft, FFTW and KissFFT, the VOLK dispatcher) and the warm-up call cancel. `K = 16` calls of 4096 samples; a kernel with fewer samples per call (the FFT sizes 64 and 1024) gets proportionally more calls, so every measurement covers 65 536 samples.
- **Inputs.** PCG32 noise with the seed `1000 + index of the kernel`, 4096 samples, buffers aligned to 64 bytes for both sides. The FFTs are in place: every call first restores the input, and the restore alone is subtracted from both sides, so the transforms always see the same finite data.
- **Builds and counters per level.**

| Level | Build | Counter |
| --- | --- | --- |
| `sse2` | default x86-64 build | Intel SDE with the CPUID of a strict SSE2 x86-64 CPU (`-cpuid_in`, `-chip_check_cpuid 1`): VOLK, rustfft and FFTW choose the code of that level themselves, and any instruction above it stops the run |
| `sse4.2` | `-C target-cpu=x86-64-v2` | Intel SDE with the CPUID of an x86-64-v2 CPU (up to SSE4.2 and POPCNT), as for `sse2` |
| `avx2` | `-C target-cpu=x86-64-v3` | Valgrind `callgrind` (Linux only) |
| `avx2-runtime` | default build with the feature `std` (run-time detection) | Valgrind `callgrind`; judged against our own numbers at `avx2` (at most 1.02×, the cost of the dispatch) |
| `avx512` | `-C target-cpu=sapphirerapids` | Intel SDE `-spr` |
| `neon` | `aarch64-unknown-linux-gnu` | `qemu-aarch64` in user mode with the plugin `xtask/plugins/bbcount.c` and a reserved guest address space (`-R 0x200000000`), which fixes the memory layout |
| `simd128` | `wasm32-wasip1`, `+simd128` | wasmtime with fuel: the smallest fuel with which the run completes, found by bisection (instructions, no cycles) |

  Under SDE only the region between two SSC markers of the binary (`benches/kernels`, `region_start`/`region_stop`) is profiled. The CPUID definitions keep the XSAVE state to x87 and SSE: on a CPU without OSXSAVE the `cpu_features` library inside VOLK would read the SSE flags of the host from `/proc/cpuinfo` and select a machine that the emulated CPU does not have. SDE's own `-p4` does not run 64-bit code, so it cannot be used for `sse2`.
- **Reproducibility.** The measurement binary is built with `--remap-path-prefix` for the checkout (`/sdrk`) and the Cargo home (`/cargo`), passed through `CARGO_ENCODED_RUSTFLAGS`. Otherwise these paths end up in the panic locations of the binary, and their length moves its data. The two runs of a measurement get arguments of the same length: `k` is written with as many digits as the number of calls (`0000` and `1024`). The start-up code of a process depends on the length of its arguments and environment; with the same length, the start-ups are identical and cancel exactly. Together with the fixed address space (above) the results do not depend on the path of the checkout or on the environment of the caller.
- **Accuracy.** The measurement binary's `accuracy <kernel> <impl>` prints the SNR of one run against the f64 reference (999 for an exact result). `cargo measure` runs it for ours and the comparison of every entry and level and writes `snr_db` into the cycle records; `cargo xtask check cycles` fails an entry whose SNR is more than 1 dB below its comparison's unless it carries an `accuracy_waiver`, and `cargo xtask count --platform wasm32` does the same at `simd128`; speed and accuracy are judged independently, except for the accuracy trade of D2760.

## References (D32)

The C references are built by `benches/refs/build.sh <x86_64|aarch64|wasm32> <prefix>` from pinned release tarballs (each checked against its SHA-256) into `benches/target/refs/<platform>`; xtask runs the script before it builds a platform and copies its manifest (versions, compilers, flags) into the results. Only these builds are used, never a VOLK of the system: the distribution's libvolk is built differently (`-O2`, LTO, ORC) and gives other counts (with the earlier callgrind method, which ran VOLK's `generic` code compiled for AVX2 at `sse2`, `convert_cs8` needed 14.02 instructions per sample with the distribution's library against 1.65 with this build).

| Library | Version | License | Platforms | Use |
| --- | --- | --- | --- | --- |
| VOLK | 3.3.0 | LGPL-3.0-or-later | x86_64, aarch64 (shared, CMake Release, no ORC); wasm32 (the `generic` protokernels, static, `benches/refs/volk-wasm.c`; since the fallback round also `volk_32fc_32f_multiply_32fc`, `volk_32f_log2_32f`, `volk_32f_s32f_multiply_32f`, `volk_32f_s32f_add_32f`, `volk_32f_x2_dot_prod_32f`, `volk_32fc_x2_dot_prod_32fc` and `volk_8i_convert_16i`, so that `volk_sys::FULL` comparisons also run at `simd128`; since M24 also `volk_32fc_deinterleave_32f_x2`, `volk_32f_x2_add_32f`, `volk_32f_x2_subtract_32f` and `volk_32f_x2_interleave_32fc`) | the reference of the streaming kernels; its `generic` `volk_32fc_x2_multiply_conjugate_32fc` is compiled into the `quadrature_demod_cf` reference |
| VOLK without FMA | 3.3.0 | LGPL-3.0-or-later | x86_64 (`avx2`, `avx512`), aarch64 (`neon`); static, one library per level (`benches/refs/volk-nofma.c`: the protokernels `volk_32fc_x2_multiply_32fc_u_avx` and `_neon` and `volk_32fc_x2_dot_prod_32fc_u_avx` and `_neon` of the release, `-O3 -ffp-contract=off` and the `-march` of the level) | information for `mul` `C32` at the levels where VOLK's dispatcher fuses (the gate there from D287 until D326); the gate of `dot_complex` there (D821) |
| FFTW | 3.3.10 | GPL-2.0-or-later | x86_64 (`--enable-sse2`), aarch64 (`--enable-neon`); static, single precision, `FFTW_ESTIMATE` | the `C32` FFT reference at `sse2` (out of place, its usual use; in place as information), information on `neon` |
| KissFFT | 131.1.0 | BSD-3-Clause | all (static, `FIXED_POINT=16`) | the `CQ15` FFT reference |
| Ne10 | 1f059a7 (last commit of the archived repository) | BSD-3-Clause | aarch64 (static, DSP module) | information next to KissFFT on `neon` |
| GNU Radio `agc_cc` | the loop of `agc.h` of GNU Radio main at `ec1bf15`, translated to C in `benches/refs/gr-agc.c` (about 20 lines, in the repository) | GPL-3.0-or-later | all (static, one object file, the generic flags of GNU Radio's build) | the `C32` `agc` reference; the only reference source that is part of the repository (it is bench-only, never linked into a library crate, and carries its license in its header) |
| GNU Radio `pwr_squelch_cc` | the loop of `squelch_base_cc_impl::general_work` with `update_state` and `mute` of `pwr_squelch_cc_impl` and `single_pole_iir<double, double, double>::filter`, GNU Radio main at `ec1bf15`, translated to C in `benches/refs/gr-squelch.c` (the virtual calls through a table of function pointers, the stream tags left out) | GPL-3.0-or-later | all (static, one object file, the flags of `agc_cc`) | the `C32` `squelch` reference; bench-only like `gr-agc.c` |
| GNU Radio `rotator` | the scalar `gr::blocks::rotator::rotate` of `rotator.h` (gr-blocks, main at `ec1bf15`): two complex products per sample with the C99 rules (`__mulsc3` on NaN) and `phase /= cabsf(phase)` every 512 samples, in `benches/refs/gr-rotator.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrrotator.a`, the flags of `agc_cc`) | the scalar reference of the `C32` `rotate` fallback (`cref_sys::GrRotator`); `rotateN` is VOLK's `rotator2`, already declared in `volk-sys` |
| GNU Radio `frequency_modulator_fc` | `work` of `frequency_modulator_fc_impl.cc` (gr-analog, main at `ec1bf15`) with `gr::fxpt::float_to_fixed` and `sincos` and the 1024-entry table that `gen_sine_table.py` generates, in `benches/refs/gr-fm-mod.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrfmmod.a`, the flags of `agc_cc`) | the scalar reference of `fm_mod` (`cref_sys::GrFrequencyModulatorFc`; float phase with `fmodf`, table sine of about 1e-6 accuracy) |
| GNU Radio `quadrature_demod_cf` | `work` of `quadrature_demod_cf_impl.cc` (gr-analog, main at `ec1bf15`): a zero-filled temporary per call, VOLK's `generic` `multiply_conjugate` and `gr::fast_atan2f` (gnuradio-runtime, an object file of its own, called out of line), in `benches/refs/gr-quad-demod.c` and `gr-fast-atan2f.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrquaddemod.a`, the flags of `agc_cc`) | the scalar reference of `quad_demod` (`cref_sys::GrQuadratureDemodCf`; `fast_atan2f` is a 256-entry table with an error of about 6e-7 rad) |
| GNU Radio `fm_deemph` | the taps of `fm_emph.py` (bilinear transform with prewarping, gr-analog, main at `ec1bf15`) and the loop of `iir_filter_ffd` (`iir_filter<float, float, double, double>::filter`, gr-filter): double taps and accumulator, doubled history buffers, in `benches/refs/gr-deemph.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrdeemph.a`, the flags of `agc_cc`) | the scalar reference of `deemphasis` (`cref_sys::GrFmDeemph`); a first-order filter with a zero at `z = -1` where SDRk's has none, the same order of work |
| GNU Radio `crc` | the generic table-driven CRC `gr::digital::crc` of gr-digital (`crc.cc`, `crc.h`, main at `ec1bf15`): the constructor's 256-entry `uint64_t` table of the polynomial (reflected or MSB first), `compute` with a 64-bit remainder and one lookup per byte, the result reflection and the final xor, in `benches/refs/gr-crc.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrcrc.a`, the flags of `agc_cc`) | the reference of the CRCs of `sdrk-proto` (`cref_sys::GrCrc`, D527): the same catalogue parameters and messages |
| GNU Radio `scrambler_bb` and `descrambler_bb` | the multiplicative scrambler and descrambler of gr-digital (`scrambler_bb_impl.cc`, `descrambler_bb_impl.cc` and `gr::digital::lfsr` of `lfsr.h` with `__builtin_parityl`, main at `ec1bf15`), member for member in `benches/refs/gr-scrambler.c`, one unpacked bit per byte | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrscrambler.a`, the flags of `agc_cc`) | the scalar reference of `bit_scramble_block` and `bit_descramble_block` (`scrambler_bb(0x21, 0, 22)`, the polynomial `1 + x^-18 + x^-23` of `x23_x18`, D1533) |
| GNU Radio `costas_loop_cc` | `work` of `costas_loop_cc_impl.cc` (gr-digital, main at `ec1bf15`) with `phase_detector_2` and `phase_detector_4` and the members of `gr::blocks::control_loop` (`advance_loop`, `phase_wrap`, `frequency_limit`, `update_gains`): per sample `sincosf` of the phase, `fast_cc_multiply`, the detector, `branchless_clip` to `+-1` and the second-order loop; one output, no tags, `use_snr` false, in `benches/refs/gr-costas.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrcostas.a`, the flags of `agc_cc`) | the reference of `costas_fallback` (`cref_sys::GrCostasLoopCc`, D607) |
| GNU Radio `linear_equalizer` | `linear_equalizer_impl::equalize` of gr-digital (main at `ec1bf15`) with `sps` 1 and no training sequence (decision-directed from the start), `adaptive_algorithm::error_dd` with the decision of `constellation_qpsk` (points `+-sqrt 2 +-j sqrt 2`), `adaptive_algorithm_nlms::update_taps` (`magsq` from the conjugate dot product of the window, then `taps[i] = conj(conj(taps[i]) + (mu / magsq) in[i] conj(error))`) and the members of `fir_filter_ccc` (gr-filter: reversed taps, `filter` as the dot product of the window, `update_tap` into one aligned copy), the VOLK calls as VOLK's `generic` protokernels, in `benches/refs/gr-equalizer.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrequalizer.a`, the flags of `agc_cc`) | the reference of `equalize_fallback` (`cref_sys::GrLinearEqualizer`, D838) |
| GNU Radio `constellation_receiver_cb` and `linear_equalizer` on a `constellation_rect` | `constellation_receiver_cb_impl::general_work` and `phase_error_tracking` of gr-digital (main at `ec1bf15`) without tags, the derotated sample as the output (the fifth output of the block), `constellation::decision_maker_pe` (`-arg(sample conj(point))`, `cargf`) with the sector decision of `constellation_rect` (`get_sector`, the sector values of `find_sector_values`), the members of `control_loop`, `sincosf` for `gr_expj`; and the loop of gr-equalizer.c with `error_dd` deciding on the same `constellation_rect`; the constellation is SDRk's unit-power 16-QAM grid (4 by 4 sectors of the width `2 / sqrt 10`, no normalization), in `benches/refs/gr-constellation.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrconstellation.a`, the flags of `agc_cc`) | the references of `costas_qam16_fallback` and `equalize_qam16_fallback` (`cref_sys::GrConstellationReceiverCb`, `GrLinearEqualizerRect`, D1210) |
| GNU Radio `fastnoise_source_f` and `fastnoise_source_c` | `work` with `sample()` of `fastnoise_source_impl.cc` (gr-analog, main at `ec1bf15`) and `xoroshiro128p_next`, `_jump`, `_seed` and `splitmix64_next` of `xoroshiro128p.h` (gnuradio-runtime), translated to C in `benches/refs/gr-fastnoise.c`: per output sample one xoroshiro128+ draw and a copy of the pool entry `d_samples[next & d_bitmask]` (the test of `d_bitmask` in the loop, the pool a `std::vector` of 8192 values, the default of GNU Radio Companion); `fastnoise_source_c` divides its amplitude by `sqrtf(2.0f)`. The pool is filled once at construction, outside the counted calls, by a stand-in for `gr::random::gasdev` (the polar Box-Muller method over a second xoroshiro128+ state): the values of the pool are not GNU Radio's, the measured loop is | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrfastnoise.a`, one object file, the flags of `agc_cc`) | information next to `gauss_noise` `f32` and `C32` (M48 §5, reference does less work; `cref_sys::GrFastnoiseSourceF`, `GrFastnoiseSourceC`) |
| GNU Radio `dvbs2_physical_cc` PL scrambling | the per-symbol `switch` on `m_cscram[n]` (`R_n` in 0..3) of `general_work` in `dvbs2_physical_cc_impl.cc` (gr-dtv, main at `ec1bf15`; the payload loop of the `VLSNR_OFF` branch), translated to C in `benches/refs/gr-dvbs2-plscramble.c`: `out = in`, `(-in.im, in.re)`, `-in` or `(in.im, -in.re)` per symbol, `gr_complex` as a struct of two floats. Left out: the PL header and the pilots, the zero sample after every symbol and the generation of `R_n` (`build_symbols`, once per object; the caller passes `R_n = p & 3` of the phase bytes, made at preparation) | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrdvbs2.a`, one object file, the flags of `agc_cc`) | information next to `scramble` `C32` at `bits = 2` (M49 §5, reference does different work; `cref_sys::gr_dvbs2`) |
| liquid-dsp `iirfilt` DC blocker | 1.8.3 (`https://github.com/jgaeddert/liquid-dsp/archive/refs/tags/v1.8.3.tar.gz`, sha256 `18fa83b73db8bb6fe6ea0376e4b5aecf8645970f4604d10d9dadbf609f3f95e2`): `iirfilt_{rrrf,crcf}_create_dc_blocker` with `_execute_block`, `_execute_norm` and liquid's portable dot products (the run-time choice fixed to `LIQUID_RUNTIME_PORT`), in `benches/refs/liquid-dcblock.c` and `liquid-dotprod.c` | MIT | x86_64, aarch64, wasm32 (static, `libliquiddc.a`, liquid's `-O3`) | the scalar reference of `dc_block` and `dc_block_real` (`cref_sys::LiquidDcBlocker`, `LiquidDcBlockerCrcf`): the same pole and zero (`alpha = 1 - r`) in direct form II with the gain `sqrt(1 - alpha)` (`set_scale(1.0)` gives SDRk's gain), not the direct form I of SDRk; no established scalar code computes `y = x - x1 + r y1` literally in single precision (STK's `PoleZero::setBlockZero` does, in double) |
| liquid-dsp halfband decimators | 1.8.3 (the same tarball and sha256 as the DC blocker above), unpacked and compiled from the release: `resamp2.proto.c` and `firhilb.proto.c` through `benches/refs/liquid-halfband.c` (the type macros of `filter_crcf.c` and `filter_rrrf.c`), `dotprod_crcf.c` and `dotprod_rrrf.c` with their SSE, AVX, AVX-512F and NEON loops, the window buffers and the support files (`msb_index.c`, `memory.c`, `error.c`, `logging.c`, `runtime.c`, `libliquid.c`) | MIT | x86_64, aarch64, wasm32 (static, `libliquidhb-<level>.a`, one library per level, `-O3 -ffp-contract=off` and the instruction set of the level) | the reference of `halfband_decimate` (`resamp2_crcf_decim_execute`), `halfband_decimate_real` (`resamp2_rrrf_decim_execute`) and `real_to_complex` (`firhilbf_decim_execute_block`) with liquid's SIMD dot product of the level (SSE at `sse2` and `sse4.2` for `dotprod_crcf`, at `sse4.2` for `dotprod_rrrf` and the portable loop at `sse2` for it, AVX at `avx2`, AVX-512F at `avx512`, NEON at `neon`; liquid has no WebAssembly loop, so the portable loop compiled with `-msimd128` is the reference at `simd128`), and, with the portable dot product (`LIQUID_RUNTIME_PORT`, compiled for the same level), the scalar reference of the three fallback entries (D69) |
| liquid-dsp `firpfbch2_crcf` | 1.8.3 (the same tarball and sha256 as the DC blocker above), unpacked and compiled from the release: `firpfbch2.proto.c` through `benches/refs/liquid-channelizer.c` (the type macros of `firpfbch_crcf.c`, liquid's names renamed per build so that both builds link), with the window buffers and dot products of the halfband library and liquid's FFT (`fftf.c`, `fft_utilities.c`, `dotprod_cccf.c`, `modular_arithmetic.c`); built twice into the halfband library of each level: with FFTW 3.3.10 (`fftw3f_FOUND`, not on wasm32) and with liquid's own FFT | MIT | x86_64, aarch64, wasm32 (static, `libliquidhb-<level>.a`) | the reference of `channelize` (the SIMD dot product of the level and FFTW; liquid's own FFT on wasm32) and of `channelize_fallback` (the portable dot product and liquid's own FFT), D919 |
| liquid-dsp `fftfilt_cccf` | 1.8.3 (the same tarball and sha256 as the DC blocker above), unpacked and compiled from the release: `fftfilt.proto.c` through `benches/refs/liquid-fftfilt.c` (the type macros of `filter_cccf.c`, liquid's names renamed per build so that both builds link; a wrapper that takes any number of samples per call and carries a partial block of `n` samples to the next call, copying element by element), with liquid's FFT of the channelizer row; built twice into the halfband library of each level: with FFTW 3.3.10 (`fftw3f_FOUND`, not on wasm32) and with liquid's own FFT | MIT | x86_64, aarch64, wasm32 (static, `libliquidhb-<level>.a`) | the reference of `xcorr_fft_n512` and `xcorr_fft_n4096` (FFTW; liquid's own FFT on wasm32, information) and of their `_fallback` entries (liquid's own FFT), M50 §5, D2619 |
| liquid-dsp `nco_crcf` | 1.8.3 (the same tarball and sha256 as the DC blocker above): `nco_crcf` of the type `LIQUID_VCO_INTERP` from `src/nco/src/nco.proto.c` with `T = float`, translated to C in `benches/refs/liquid-nco.c` (`nco_crcf_create` with its table of 1024 value and slope pairs, `_fp_sin`, `_sincos`, `_static_index`, `_adjust_frequency`, `_adjust_phase`, `_constrain` with its `while` loops, `_step`; the loop gains of SDRk are applied through `_adjust_frequency` and `_adjust_phase`, the two calls of liquid's `_pll_step`, and `sin(2 theta)` is read from the same table with the phase word shifted left by one bit) | MIT | x86_64, aarch64, wasm32 (static, `libliquidnco.a`, one object file, liquid's `-O3` and the generic flags of `agc_cc`) | the scalar reference of the `f32` `stereo_pll` fallback (`cref_sys::LiquidNcoPll`; a lower accuracy class: the table interpolates with an error of 4.7e-6 against 2.9e-7 of the 4096-entry table of `rotate`, `_constrain` reduces a correction modulo two pi in `float` so that a small negative correction becomes `2 pi - c` rounded to a multiple of 4.8e-7 rad, and the loop has no lock detector and no integrator limit) |
| liquid-dsp `randnf` and `crandnf` | 1.8.3 (the same tarball and sha256 as the DC blocker above): `randnf` and `crandnf` of `src/random/src/randn.c` with `randf` of `src/random/src/rand.c` (`rand() / RAND_MAX` of the libc of the platform: glibc's `random_r` under a lock on x86_64 and aarch64, wasi-libc's generator on wasm32), Box-Muller in single precision (`sqrtf`, `logf`, `sinf` for `randnf`, `cexpf` for `crandnf`) with the rejection loop for `u1 == 0`, in `benches/refs/liquid-randn.c` with two plain loops that scale each value by `sigma` | MIT | x86_64, aarch64, wasm32 (static, `libliquidrandn.a`, one object file, the flags of `agc_cc`) | the comparison of `gauss_noise` `f32` and `C32` and of their fallbacks, information only (M48 §5, reference does different work; `cref_sys::LiquidRandn`) |
| libairspy | 1.0.10 (`https://github.com/airspy/airspyone_host/archive/refs/tags/v1.0.10.tar.gz`, sha256 `fcca23911c9a9da71cebeffeba708c59d1d6401eec6eb2dd73cae35b8ea3c613`): `iqconverter_float.c` and `iqconverter_int16.c` through `benches/refs/airspy-iqconv.c`. The last release under the MIT licence; later releases are under a restrictive licence and are not used | MIT | x86_64, aarch64, wasm32 (static, `libairspy-<level>.a`, `-O3 -ffp-contract=off` and the instruction set of the level; the SSE2 build of the float converter on x86_64) | `real_to_complex`: the SSE2 reference of the `C32` kernel at `sse2` (information; libairspy enables its SSE2 code for FreeBSD only, the file is compiled with that branch), the generic code as information next to the fallback, and `iqconverter_int16` for `CQ15` (a lower accuracy class: truncating `>> 15`, a 32-bit accumulator, the odd samples halved; its status is the maintainer's open decision I23-1). Conversion only, without `remove_dc`; the wrapper clears the delay line and the FIR queue of the int16 converter completely, because its `reset` clears half of each (v1.0.10 uses `sizeof(int16_t)` for 32-bit queue entries), which leaves the first outputs to the allocator |
| rustfft | 6.4 (Cargo dependency of `benches/kernels`) | MIT OR Apache-2.0 | all; `wasm_simd` on wasm32 (target-specific) | the `C32` FFT reference except at `sse2` |
| rand_distr | 0.5.1 with rand 0.9.5 (Cargo dependencies of `benches/kernels` only, checked with `cargo deny --manifest-path benches/Cargo.toml check`): `StandardNormal` (the Ziggurat method in `f64`, sampled as `f32`) over `rand::rngs::SmallRng` (xoshiro256++ on the 64-bit targets, xoshiro128++ on wasm32), times `sigma` | MIT OR Apache-2.0 | all | the informational Ziggurat row of `gauss_noise` `f32` (`gauss_noise_f32_ziggurat`, M48 §5, reference does different work; maintainer decision 2026-11-17) |
| KFR | 7.2.0 | GPL-2.0-or-later | x86_64 (GCC), aarch64 (`clang++ --target=aarch64-linux-gnu`: GCC rejects KFR's NEON code), wasm32 (`clang++ --target=wasm32-wasip1`, KFR's own Emscripten configuration `KFR_ARCH=sse41` with `-msse4.1 -msimd128`, D29); header-only `kfr::iir` behind `benches/refs/kfr-biquad.cpp`, static, one library per level | the SIMD reference of `biquad` C32 and `biquad_real` f32 at every level (D15, D29) |
| CMSIS-DSP | 1.18.0 | Apache-2.0 | all; generic C (`__GNUC_PYTHON__`, `ARM_MATH_LOOPUNROLL`, the configuration of its own host build) behind `benches/refs/cmsis-biquad.c`, static, one library per level; on the Cortex-M4 of the report with CMSIS-Core, the DSP extension and the FPU (`arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -O3 -ffp-contract=off`, `ARM_MATH_LOOPUNROLL`; `benches/refs/cmsis-m4-heap.c` supplies `malloc`) | the scalar reference of the four biquad fallbacks (D16); information next to KFR and next to the Q15 entries (D15; until D29 the gate of `biquad` C32 and `biquad_real` f32 at `simd128`); the comparison of the biquads in the Cortex-M4 report; since M24 also `arm_add_f32`, `arm_sub_f32`, `arm_add_q15`, `arm_sub_q15` (the `stereo_matrix` fallbacks) (D27); since the fallback round the scalar references in `libcmsisdsp-<level>.a` behind `benches/refs/cmsis-dsp.c` (x86_64, aarch64, wasm32; `cref_sys::cmsis_dsp`): the conversions `arm_q7_to_float`, `arm_q15_to_float`, `arm_q7_to_q15`, `arm_float_to_q15`, `arm_float_to_q7`, `arm_q15_to_q7`; `arm_dot_prod_{f32,q15}`, `arm_cmplx_dot_prod_{f32,q15}`; `arm_scale_{f32,q15}`, `arm_mult_{f32,q15}`, `arm_cmplx_mult_real_{f32,q15}`, `arm_cmplx_mult_cmplx_{f32,q15}`, `arm_cmplx_conj_f32`; `arm_cmplx_mag_squared_{f32,q15}`, `arm_cmplx_mag_{f32,q15}`, `arm_vlog_f32`; `arm_sin_cos_{f32,q31}`, `arm_sin_f32`, `arm_cos_f32`, `arm_sin_q15`, `arm_cos_q15`, `arm_atan2_{f32,q15}`; `arm_fir_{f32,q15}`, `arm_fir_decimate_{f32,q15}`, `arm_fir_interpolate_{f32,q15}`; `arm_cfft_{f32,q15}` (64, 1024, 4096 points, the constant instances of `arm_const_structs.c`) and `arm_rfft_fast_f32` |
| wasi-sdk sysroot | 34.0 (`https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sysroot-34.0.tar.gz`, sha256 `9d813544eeebe38b7b8f2244ed591de46b6db812c6dd1a257ff9f0d2a905a2be`) | Apache-2.0 WITH LLVM-exception (libc++), the licenses of wasi-libc | wasm32 only; the headers of libc++ (`include/wasm32-wasip1/noeh/c++/v1`, `include/c++/v1`) and wasi-libc | the C++ standard library headers KFR needs on WASI; the shim links no library of it (D29) |
| Emscripten | 6.0.12 (`https://github.com/emscripten-core/emscripten/archive/refs/tags/6.0.12.tar.gz`, sha256 `795ef929e4dee52bf03de3742c4252bb121e742afac896378d969b5da2db86eb`) | MIT or NCSA | wasm32 only; the `*intrin.h` headers of `system/include/compat` (SSE to SSE4.1 written with SIMD128 intrinsics) | KFR's x86 code on wasm32, as emcc compiles it with `-msse4.1 -msimd128` (D29) |
| CMSIS 6 (CMSIS-Core) | 6.3.0 (`https://github.com/ARM-software/CMSIS_6/archive/refs/tags/v6.3.0.tar.gz`, sha256 `331df74000876b8fb07933658cbf402c4d9f831b429258792d2c34ab5a6c5bf7`) | Apache-2.0 | Cortex-M4 only; the headers of `CMSIS/Core/Include` (`cmsis_compiler.h` and the intrinsics such as `__SMLALD`) | needed by CMSIS-DSP outside its host configuration (D27) |

The libraries are linked only into the measurement binary of `benches/kernels`, which is built on the measuring machine and never distributed; none of them is a dependency of a library crate of SDRk. `benches/` is a Cargo workspace of its own that is not a member of the repository workspace (`exclude = ["benches"]`), so `cargo deny` does not see it. VOLK is linked dynamically (`benches/volk-sys`, hand-written `extern "C"` declarations; VOLK declares its kernels as function-pointer variables that its dispatcher fills in), FFTW, KissFFT, Ne10, the GNU Radio loops, liquid-dsp's DC blocker and halfband decimators, libairspy's converters, KFR and CMSIS-DSP statically through `benches/cref-sys`.

The biquad references are compiled once per level with the instruction set of the level (`-march=x86-64`, `x86-64-v2`, `x86-64-v3`, `sapphirerapids` for `sse2`, `sse4.2`, `avx2`, `avx512`; `armv8-a` for `neon`; `-msimd128` for `simd128`), all with `-O3 -ffp-contract=off` (no FMA on either side, D5), into `libkfrbq-<level>.a` and `libcmsisbq-<level>.a` (and the other CMSIS-DSP references into `libcmsisdsp-<level>.a`); `benches/cref-sys/build.rs` links the libraries of the level that the target features of the measurement build select, the rule of `sdrk-backend-simd`. The level `avx2-runtime` is judged against our `avx2` numbers and runs no reference. KFR's shim is compiled without exceptions and RTTI and allocates with `aligned_alloc`, so it needs no C++ run-time library; its filter loop throws nothing, so this does not change the measured code. On wasm32 KFR has no code of its own: its CMake build for Emscripten selects `KFR_ARCH=sse41` and compiles with `-msse4.1 -msimd128`. `build.sh` repeats that with the system clang: the macros emcc defines for `-msse4.1` (`__SSE__` to `__SSE4_1__`), the SSE headers of Emscripten 6.0.12 and the libc++ and wasi-libc headers of the wasi-sdk 34 sysroot, both pinned by SHA-256 (D29). Most of KFR's filter is written with clang's vector extensions, which compile to SIMD128 directly; the emulated SSE intrinsics are a few shuffles. KFR's generic architecture does not compile in 7.2.0.

What each kernel is compared with:

| Kernel | Comparison |
| --- | --- |
| `convert` CS8 / CS16 to C32 | `volk_8i_s32f_convert_32f` / `volk_16i_s32f_convert_32f` on 2N values |
| `convert` CU8 to C32 | `volk_8i_s32f_convert_32f` on the same 2N bytes with the sign bit flipped (the same work apart from the offset of 0.5 / 128, which VOLK, having no unsigned 8-bit input, does not add); our fallback as information |
| `convert_real` RS16 to f32 | `volk_16i_s32f_convert_32f` on N values |
| `convert` CS8 to CQ15 | `volk_8i_convert_16i` (`x * 256`) on 2N values |
| `convert` CU8 to CQ15 | `volk_8i_convert_16i` on the same 2N bytes with the sign bit flipped (`(u - 128) << 8`, the same work apart from the offset of 128 of `(2u - 255) << 7`, which the accuracy check adds outside the counted region); our fallback as information. `convert` CS16 to CQ15 and `convert_real` RS16 to Q15 are copies and have no entry |
| `convert_out` C32 to CS16 / CS8 / CU8 | `volk_32f_s32f_convert_16i` / `volk_32f_s32f_convert_8i` / `volk_32f_s32f_x2_convert_8u` on 2N values |
| `convert_out_real` f32 to RS16 / RS8 / RU8 | the same VOLK kernels on N values (the same work per value). `Q15` to the real 8-bit formats has no entry: no established library converts Q15 to 8 bits with ties to even (CMSIS-DSP `arm_q15_to_q7` truncates, VOLK's `16i_convert_8i` takes the high byte); `Q15` to `RS16` is a copy |
| `rotate` C32 | `volk_32fc_s32fc_x2_rotator2_32fc` |
| `fir` C32, 64 taps; `fir_decimate` C32, 127 taps, M = 10 | `volk_32fc_32f_dot_prod_32fc` per output, called as GNU Radio's FIR filters call it (aligned entry point, taps pre-shifted, D6) |
| `fir_interpolate` C32, L = 2, 128 taps; `resample` C32, 3 / 2, 192 taps (64 per branch; per output sample) | the same dot product per output with the taps of the output's branch, as GNU Radio's `interp_fir_filter_ccf` and `rational_resampler_ccf` call it |
| `fir_interpolate_short` C32, L = 8, 80 taps (10 per branch, the short-branch form of D1420; per output sample) | the same dot product per output with the taps of the output's branch, as GNU Radio's `interp_fir_filter_ccf` calls it (`volk_32fc_32f_dot_prod_32fc`); `fir_interpolate_short_fallback` against CMSIS-DSP's `arm_fir_interpolate_f32` by 8 on the planar `re` and `im` (CMSIS-DSP needs a multiple of the factor, so 80 taps instead of the 81 of `qam_mod`) |
| `fir_interpolate_real` f32, L = 2, 128 taps; `resample_real` f32, 3 / 2, 192 taps | `volk_32f_x2_dot_prod_32f_a` per output with the taps of the output's branch, as GNU Radio's `interp_fir_filter_fff` and `rational_resampler_fff` call it; VOLK's wasm32 build lacks it, so at `simd128` our fallback, as information |
| `mag_sq` C32 | `volk_32fc_magnitude_squared_32f` |
| `mag` C32 | `volk_32fc_magnitude_32f` |
| `mag` CQ15 | `volk_16ic_magnitude_16i` (VOLK scales to float and back and rounds differently from the integer rule; the entries are `release` (they started as `dev`; every ratio is below 1.00, open item M20-3), and the accuracy rule compares the SNR of both against the unrounded magnitude) |
| `quad_demod` C32 | `volk_32fc_x2_multiply_conjugate_32fc` + `volk_32fc_s32f_atan2_32f` |
| `quad_demod_fallback` C32 (the measured side is our fallback, compiled for the level) | the scalar `quadrature_demod_cf` of GNU Radio (`benches/refs/gr-quad-demod.c`: a zero-filled temporary per call, VOLK's `generic` `multiply_conjugate` and `gr::fast_atan2f` out of line), compiled with the generic flags of GNU Radio's own build; CMSIS-DSP's generic C (`arm_cmplx_conj_f32`, `arm_cmplx_mult_cmplx_f32`, `arm_atan2_f32` per sample, `arm_scale_f32` with the gain) as information. Not the same accuracy class: `fast_atan2f` interpolates a table (114.0 dB on the measured input), ours is `libm::atan2f` bit for bit (143.6 dB, the contract), CMSIS-DSP 136.6 dB; where GNU Radio is cheaper beyond the limit the entry is `info`. Not measured at `avx2-runtime` |
| `quad_demod_cq15_fallback` CQ15 | CMSIS-DSP's generic C: the conjugates of the previous samples (`arm_cmplx_conj_q15` is not in the build, so a plain loop with its saturating negation), `arm_cmplx_mult_cmplx_q15` (Q3.13), `arm_atan2_q15` (Q2.13 radians) and `arm_scale_q15` by `4 / pi` into phase / pi in Q15. Less accurate than ours (62.3 against 81.2 dB). Not measured at `avx2-runtime` |
| `window` C32 (a periodic Hann window) | `volk_32fc_32f_multiply_32fc` (a complex vector times a real one, the same work). At `simd128` its `generic` protokernel from `volk-wasm.c` (since the references of the fallback round), scalar code, so information (D12) |
| `power_db` f32 (powers over 70 dB, offset -3 dB) | VOLK's base-2 logarithm scaled and offset by VOLK: `volk_32f_log2_32f`, `volk_32f_s32f_multiply_32f` by `10 log10(2)` in place, `volk_32f_s32f_add_32f` in place (the same work without the floor). Not the same accuracy class: VOLK's SIMD `log2` is a polynomial with an SNR of 141.6 dB on the measured input against 152.0 dB of ours, which is bit-exact with `libm::log10f` like the fallback; matching its cost would lose accuracy (D34), so the entries where it is cheaper are `info`. At `simd128` the `generic` protokernels from `volk-wasm.c`, scalar code, so information (D12) |
| `window` CQ15, `power_db` Q15 | our fallback (no established library has a complex-by-real Q15 multiply with the rounding of `q15_mul`, or a decibel conversion of Q15 powers), as information with the efficiency verdict |
| `mag_sq_fallback`, `mag_sq_q15_fallback`, `mag_fallback`, `mag_q15_fallback`, `window_fallback`, `window_cq15_fallback`, `power_db_fallback`, `power_db_q15_fallback` (the measured side is our fallback, compiled for the level) | CMSIS-DSP 1.18.0 generic C, compiled for the same level: `arm_cmplx_mag_squared_f32` / `_q15`, `arm_cmplx_mag_f32` / `_q15`, `arm_cmplx_mult_real_f32` / `_q15`, and for `power_db` `arm_vlog_f32` in place and `arm_scale_f32` by `10 / ln 10` with the offset added by a plain loop (`arm_offset_f32` is not in the shim), after `arm_q15_to_float` for Q15. Accepted as the same work: the Q15 squared magnitude in Q3.13 truncating (scaled by 4 for the accuracy column), the Q15 magnitude in Q2.14 from CMSIS-DSP's own `arm_sqrt_q31` (scaled by 2), the Q15 window product truncating. `arm_vlog_f32` is a polynomial of another accuracy class (143 dB against our 152 dB, bit-exact with `libm::log10f`); our fallback is faster at every level, so `power_db_fallback` is `release`, while `power_db_q15_fallback` is information (no library converts Q15 powers to decibels; judged by its efficiency). Not measured at `avx2-runtime` |
| `rotate` CQ15, `mag_sq` CQ15, `quad_demod` CQ15, `fm_mod` CQ15, `fir_interpolate_real` and `resample_real` Q15, `squelch` CQ15, `agc` CQ15, `dc_block_real` f32 and Q15, `deemphasis` f32 and Q15, `dc_block` C32 and CQ15 | our fallback (no established library; `fm_mod` C32 is `info` against GNU Radio's scalar `frequency_modulator_fc` since the fallback round, D58). `mag_sq` CQ15 is `info` with the efficiency verdict since the fallback round; `quad_demod` CQ15, `rotate` CQ15, `fm_mod` CQ15 and the Q15 polyphase entries are `info` with the efficiency verdict (D32; the `quad_demod` fallbacks themselves are gated by `quad_demod_fallback` and `quad_demod_cq15_fallback`); `squelch` CQ15, `agc` CQ15, `dc_block_real`, `deemphasis` and `dc_block` are not accelerated, so their entries are `info` (and their `avx2-runtime` entries compare the baseline build of the fallback with its AVX2 build) |
| `biquad_real` f32, 2 sections | KFR 7.2.0 `kfr::iir` (`iir_state<float, 2>`: the sections in the lanes of one vector, transposed direct form II, the same algorithm class as ours) at every level, `simd128` included (D29); CMSIS-DSP `arm_biquad_cascade_df2T_f32` (generic C) as information (D15) |
| `biquad` C32, 2 sections | KFR as above on the re and im parts as two real channels: two calls of N real samples (the 2N-values rule of `convert`), the deinterleaving done once outside the counted region (the best case for KFR, which has no complex type); CMSIS-DSP `arm_biquad_cascade_stereo_df2T_f32` on the interleaved buffer (its two channels are our re/im layout) as information (D15) |
| `biquad` CQ15, `biquad_real` Q15, 2 sections | our fallback, as information only (D21): no library has a SIMD Q15 biquad on our levels (CMSIS-DSP has one for Helium/MVE only; Intel IPP's 16-bit IIR computes in float, another contract); CMSIS-DSP `arm_biquad_cascade_df1_q15` (generic C, `postShift = 1`) as information, for CQ15 two calls on planar re and im (D15) |
| `biquad_fallback`, `biquad_cq15_fallback`, `biquad_real_fallback`, `biquad_real_q15_fallback` (the measured side is our fallback, compiled for the level) | CMSIS-DSP 1.18.0 generic C, compiled for the same level (D16): `arm_biquad_cascade_stereo_df2T_f32` for C32, `arm_biquad_cascade_df2T_f32` for f32, `arm_biquad_cascade_df1_q15` with `postShift = 1` for Q15 and, on planar re and im, for CQ15. The coefficients are ours with `a1`, `a2` negated (CMSIS-DSP adds the feedback terms); with `postShift = 1` the raw Q15 coefficients are exactly our Q2.14 values. CMSIS-DSP's Q15 cascade truncates where ours rounds half up, and its f32 cascades do not flush the state; both are accepted as the same work. A feedback coefficient of -2.0 (raw -32768) cannot be negated in Q15; the measured Butterworth does not use it. Not measured at `avx2-runtime` |
| `agc` C32 | the loop of `agc_cc` of GNU Radio (`gr-analog/include/gnuradio/analog/agc.h`), a C copy in `benches/refs/gr-agc.c` compiled with the generic flags of GNU Radio's own build (`-O3`, no `-march`) for every level of the platform; our fallback is information. The `agc` of GNU Radio is a different algorithm (additive, no lower limit, no silence rule), so the accuracy column compares each with the f64 model of its own loop |
| `agc_fallback` C32 (the measured side is our fallback, compiled for the level) | the same `agc_cc` loop of GNU Radio, the scalar reference of the fallback; not measured at `avx2-runtime` |
| `costas` C32 | our fallback as information (`info`; not accelerated, D608) |
| `costas_fallback` C32 (the measured side is our fallback, compiled for the level) | the loop of GNU Radio's `costas_loop_cc` (`benches/refs/gr-costas.c`, compiled like `gr-agc.c`) at order 4 with the loop bandwidth `loop_bw / (damping + 1 / (4 damping))` of our constants, so both loops have the same gains; GNU Radio computes the sine and cosine with `sincosf` and has no lock detector, ours reads the `rotate` tables and runs the lock metric, so the accuracy column compares each with the f64 model of its own loop; not measured at `avx2-runtime` |
| `costas_qam16` C32 (M44, the constellation `qam16`) | our fallback as information (`info`; not accelerated, the recursion of D608, D1210) |
| `costas_qam16_fallback` C32 (the measured side is our fallback) | the loop of GNU Radio's `constellation_receiver_cb` with the 16-QAM `constellation_rect` (`benches/refs/gr-constellation.c`, D1210) with the loop bandwidth of our constants and the frequency limits `+-2 pi max_offset`: the same decision-directed loop with the same detector `arg(y conj d)`; GNU Radio computes the sine and cosine with `sincosf`, decides with two divisions per symbol and has no lock detector, ours reads the `rotate` tables, decides with a product and runs the lock metric; 16-QAM at 0.3 rad and 0.0005 cycles per symbol with uniform noise of ±0.05; not measured at `avx2-runtime` |
| `costas_qam64` C32 (I44-11, the constellation `qam64`) | our fallback as information (`info`, D1601) |
| `costas_qam64_fallback` C32 (the measured side is our fallback) | the same loop of GNU Radio's `constellation_receiver_cb` with the 64-QAM `constellation_rect` (8 by 8 sectors of the width `2 / sqrt(42)`, D1601); 64-QAM at 0.05 rad (at 0.3 rad the outer points are decided wrongly) and 0.0005 cycles per symbol with uniform noise of ±0.05; not measured at `avx2-runtime` |
| `costas_qam256`, `costas_qam1024` C32 (M44.2, the constellations `qam256` and `qam1024`; the start phase 0.025 and 0.0125 rad and the noise ±0.012 and ±0.006, the 64-QAM phase and the 16-QAM noise scaled by the level spacing) | our fallback as information (`info`, D1507, D1524) |
| `costas_qam256_fallback` C32 (the measured side is our fallback) | the same loop of GNU Radio's `constellation_receiver_cb` with the 256-QAM `constellation_rect` (16 by 16 sectors of the width `2 / sqrt(170)`, D1524): the workload of `costas_qam256` |
| `costas_qam1024_fallback` C32 (the measured side is our fallback) | the same loop of GNU Radio's `constellation_receiver_cb` with the 1024-QAM `constellation_rect` (32 by 32 sectors of the width `2 / sqrt(682)`; the C translation with `SDRK_GR_MAX_POINTS` 1024, D1545): the workload of `costas_qam1024` |
| `rds_demod_block`, `rds_decode_block` (the blocks `rds_demod` and `rds_decode`, M41) | our own code as information (`info`, `Family::Own`; no library has these blocks, D678) |
| `rds_shape`, `stereo_encode_rds` (M42) | our own fallback as information (`info`, `Family::Own`; no library has an RDS modulator or a stereo encoder with RDS, D757) |
| `squelch` C32 | the loop of `pwr_squelch_cc` of GNU Radio (`squelch_base_cc_impl::general_work` of gr-analog with the overrides of `pwr_squelch_cc_impl`), a C copy in `benches/refs/gr-squelch.c` compiled like `gr-agc.c` for every level of the platform, with our open level as its threshold, the same smoothing coefficient, no ramp and no gating; our fallback is information. GNU Radio averages in double precision without hysteresis and flush, so the accuracy column compares it with the f64 model of its own loop |
| `dot` C32, N = 4096 taps, one call | `volk_32fc_32f_dot_prod_32fc` through its dispatcher (alignment check per call); the `generic` protokernel at `simd128` is information (D12) |
| `dot_real` f32 / `scale` C32 | `volk_32f_x2_dot_prod_32f` / `volk_32f_s32f_multiply_32f` on the 2N floats (the same work); VOLK's wasm32 build has neither, so at `simd128` our fallback is information |
| `dot` CQ15, `dot_real` Q15, `scale` CQ15 | our fallback, as information only: no established SIMD library has the exact 64-bit sums of the dot products (`volk_16ic_x2_dot_prod_16ic` takes complex taps and sums in saturating 16-bit lanes) or the rounding of `q15_mul`; the branches are judged by their efficiency |
| `dot_fallback`, `dot_cq15_fallback`, `dot_real_fallback`, `dot_real_q15_fallback`, `scale_fallback`, `scale_cq15_fallback` (the measured side is our fallback, compiled for the level) | CMSIS-DSP 1.18.0 generic C (`libcmsisdsp-<level>.a`), compiled for the same level, called through its raw declarations: `arm_dot_prod_f32` (f32) and `arm_dot_prod_q15` (Q15); for C32 and CQ15 with real taps the same function twice on the planar re and im parts, deinterleaved once outside the counted region (CMSIS-DSP has no complex dot product with real taps; `arm_cmplx_dot_prod_*` with zero imaginary taps would do twice the work); `arm_scale_f32` and `arm_scale_q15` (shift 0) on the 2N values. `arm_dot_prod_q15` returns the exact Q34.30 sum without our final rounding and saturation (the harness rounds it outside the count for the accuracy); `arm_scale_q15` truncates where `q15_mul` rounds (87.6 against 93.7 dB), accepted as the same work. Not measured at `avx2-runtime` |
| `dot_complex` C32 (M43), N = 4096, one call | `volk_32fc_x2_dot_prod_32fc` through its dispatcher (the same function: `sum a[i] b[i]`, not conjugated) at `sse4.2` (`a_sse3`, `u_sse3`); at `avx2`, `avx512` and `neon`, where the dispatcher fuses (`a_avx_fma`, `neonv8`) and the bit-exact contract of ours does not (D801), VOLK's protokernel without FMA (`u_avx`, `neon` from `benches/refs/volk-nofma.c`) is the gate and the fused dispatch is information (D821, as for `mul` under D287); `info` at `sse2` (VOLK's first implementation is SSE3, so its `sse2_64_mmx` machine runs the scalar `generic` code) and at `simd128` (D12); `dev` until the first count |
| `dot_complex_fallback` C32 (the measured side is our fallback, compiled for the level) | CMSIS-DSP 1.18.0 `arm_cmplx_dot_prod_f32`, generic C compiled for the same level, on the same interleaved buffers (D822); `dev` until the first count; not measured at `avx2-runtime` |
| `equalize` C32 (M43), 11 taps, QPSK, one symbol per sample | our fallback as information (`info`, D823, D838): no SIMD library runs the loop (liquid-dsp's `eqlms_cccf` has no decision feedback and no phase tracker, and its blind mode projects onto the unit circle instead of the CMA error; the translation of GNU Radio's `linear_equalizer` runs VOLK's generic dot products and is the scalar reference of the fallback); every backend runs the fallback (D841) |
| `equalize_fallback` C32 (the measured side is our fallback) | GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` (`benches/refs/gr-equalizer.c`, D838): the same decision-directed NLMS work per symbol (an FIR of 11 complex taps, the QPSK decision, the window power and the normalized update of every tap); ours adds the phase tracker and the smoothed `mse`; `dev` until the first count; not measured at `avx2-runtime` |
| `equalize_qam16` C32 (M44), 11 taps, the constellation `qam16` | our fallback as information (`info`, D823, D1210): every backend runs the fallback's loop (D841) |
| `equalize_qam16_fallback` C32 (the measured side is our fallback) | GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` deciding on the 16-QAM `constellation_rect` (`benches/refs/gr-constellation.c`, the loop of gr-equalizer.c, D1210): the work of `equalize_fallback` with the grid decision; 16-QAM through `1 + 0.2 e^(j0.6) z^-1 + 0.05 e^(-j2) z^-2` at 0.2 rad with uniform noise of ±0.02, the counted calls in decision mode |
| `equalize_qam64` C32 (I44-11), 11 taps, the constellation `qam64` | our fallback as information (`info`, D823, D1601): every backend runs the fallback's loop (D841) |
| `equalize_qam64_fallback` C32 (the measured side is our fallback) | GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` deciding on the 64-QAM `constellation_rect` (D1601): the workload of `equalize_qam16_fallback` with 64-QAM points |
| `equalize_qam256`, `equalize_qam1024` C32 (M44.2), 11 taps, the constellations `qam256` and `qam1024` (the noise ±0.005 and ±0.0025, the 16-QAM noise in units of the level spacing) | our fallback as information (`info`, D823, D1507, D1524): every backend runs the fallback's loop (D841) |
| `equalize_qam256_fallback` C32 (the measured side is our fallback) | GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` deciding on the 256-QAM `constellation_rect` (D1524): the workload of `equalize_qam256` |
| `equalize_qam1024_fallback` C32 (the measured side is our fallback) | GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` deciding on the 1024-QAM `constellation_rect` (D1545): the workload of `equalize_qam1024` |
| `channelize` C32 (M45), 16 channels, 8 taps per branch (the block default), 4096 input samples of noise per call | liquid-dsp 1.8.3 `firpfbch2_crcf` analyzer (`benches/refs/liquid-channelizer.c`, D919): the 2x oversampled M/2-hop bank of ours, once per hop with SDRk's prototype, the SIMD dot product of the level and FFTW 3.3.10 for the transform (liquid's default build where FFTW is found; on wasm32 liquid's own FFT and the portable dot product); liquid also scales the transform by `1 / M`; our fallback as information |
| `channelize_fallback` C32 (the measured side is our fallback) | the same `firpfbch2_crcf` with liquid's portable dot product and its own FFT (portable C) |
| `channelize_block`, `channel_select_block` (the blocks `channelize` and `channel_select`, M45); `channel_bank_channelize`, `channel_bank_ddc` (M45 test 9: one channelizer with eight readers against eight `ddc` blocks) | our own code as information (`info`, `Family::Own`; the library comparison of the channelizer is the kernel entry `channelize`, D928, D930; Channelizer blocks (decisions-log.md)) |
| `band_pulse_detect_block` (the block `band_pulse_detect`, M45.1: 16 lanes at 250 kHz, per lane value) | our own code as information (`info`, `Family::Own`; the natural like-for-like reference, `rtl_433`'s `pulse_detect_package`, is GPL and is not used or linked, and no other library has a pulse detector on the lanes of a channelizer, D962; Band pulse detector (decisions-log.md)) |
| `qam_soft_bits_qam16`, `qam_soft_bits_qam64` (the kernel `qam_soft_bits`, M44 §2, I44-1), 4096 symbols per call, scale 1, per input symbol | our fallback as information (`info`, `Family::Own`, D1201, D1412): no library computes the piecewise-linear soft bits; the SIMD paths are judged by their efficiency and must not be slower than the fallback; QAM slicer (decisions-log.md) |
| `symbol_demap_qam16_block`, `symbol_demap_qam64_block` (the block `symbol_demap` with the constellations `qam16` and `qam64`, M44 §2, around the kernel `qam_soft_bits`), 4096 symbols per call, per input symbol | our own code as information (`info`, `Family::Own`, D1201): no library computes the piecewise-linear soft bits (liquid-dsp's `modemcf_demodulate_soft` and GNU Radio's `calc_soft_dec` search the points for the exact max-log or log-sum value); QAM slicer (decisions-log.md) |
| `qam_mod_block`, `qam_demod_block`, `qam_demod_search_block` (the blocks `qam_mod` and `qam_demod`, M44 §4, 16-QAM at 8 samples per symbol; the search entry on noise without a signal, D1422), 4096 output samples (`qam_mod`) or 4096 input samples (`qam_demod`) per call, per sample; `frame_decode_four_block` (`frame_decode` with `rotations` `four`, 4 bits per symbol), per soft bit | our own code as information (`info`, `Family::Own`, D1216): the blocks are chains of kernels whose library comparisons are their own entries (`fir_interpolate`, `fir`, `symbol_sync`, `costas_qam16_fallback`, `equalize_qam16_fallback`); no library has the same blocks or the test framing; QAM blocks (decisions-log.md) |
| `psk_mod_block` (the block `psk_mod`, coherent QPSK at 8 samples per symbol), `symbol_map_block` (`symbol_map` with `qam16`), `iq_offset_block` (`iq_offset`, `imag`, a delay of 4 samples, in place), `qam_acquire_block` (`qam_acquire`, `correct` false, searching on noise, in place), `qam_frame_encode_block` and `qam_frame_decode_block` (the composites `qam_frame_encode` and `qam_frame_decode`, 16-QAM at 8 samples per symbol, 32-byte frames) (M44.1), 4096 output samples, symbols or input samples per call | our own code as information (`info`, `Family::Own`, D1216): the building blocks and composites of M44.1 have no counterpart in a library (GNU Radio builds the same chains from separate blocks), their kernels have entries of their own; QAM building blocks and composites (decisions-log.md) |
| `pulse_shape_block` (the block `pulse_shape` with the pulse of `qam_mod_block`, 8 samples per symbol, roll-off 0.35, span 10, per output sample), `pulse_match_block` (the block `pulse_match`, the matched filter of `qam_demod_block`, 25 kBd at 200 kHz, per input sample) (M44.1.1), 4096 output or input samples per call | our own code as information (`info`, `Family::Own`, D1216): the building blocks of M44.1.1 with their design; their kernels have entries of their own against library references (`fir_interpolate_short`, `fir`); Pulse filters as building blocks (decisions-log.md) |
| `gauss_noise_f32`, `gauss_noise_c32` (the kernel `gauss_noise`, M48 §5), `sigma` 0.1, stream 1, 4096 output samples per call, per output sample | liquid-dsp 1.8.3 `randnf` / `crandnf` times `sigma` (`benches/refs/liquid-randn.c`): information only, reference does different work (libc `rand()` and single-precision Box-Muller, no random access, no bit-exact sequence); our fallback beside it |
| `gauss_noise_f32_fallback`, `gauss_noise_c32_fallback` (the measured side is our fallback) | the same liquid-dsp generators, information only; the pairs of the SIMD entries for the D321 cycle gate |
| `gauss_noise_q15` | our fallback as information (`info`, `Family::Own`): no library makes `Q15` Gaussian noise |
| `gauss_noise_q15_fallback` (the measured side is our fallback) | the same fallback as information (`info`, `Family::FallbackOwn`, ratio 1): no library makes `Q15` Gaussian noise; the pair of `gauss_noise_q15` for the D321 cycle gate |
| `gauss_noise_f32_fastnoise`, `gauss_noise_c32_fastnoise` (our `gauss_noise` through the facade) | the work loop of GNU Radio's `fastnoise_source_f` / `_c` (`benches/refs/gr-fastnoise.c`, amplitude `sigma` and `sigma * sqrt(2)`, a pool of 8192 values): information only, reference does less work (one xoroshiro128+ draw and a copy from a precomputed pool per sample, no normal mapping, repeating values, the tail ends at the largest pool value, no random access); the floor of what a noise sample can cost |
| `gauss_noise_f32_ziggurat` (our `gauss_noise` through the facade) | rand_distr 0.5.1 `StandardNormal` as `f32` over rand 0.9.5 `SmallRng`, times `sigma`: information only, reference does different work (Ziggurat rejection sampling with a variable number of draws per value, no random access, no bit-exact sequence across platforms; maintainer decision 2026-11-17) |
| `noise_block` (the block `noise`, M48 §6, default configuration on a `C32` clock, per output sample) | our own code as information (`info`, `Family::Own`; one call of `gauss_noise` per block call, whose entries carry the library rows) |
| `scramble_c32_b1` (the kernel `scramble`, M49.1 §5, `bits = 1`, random phase bytes, 4096 samples per call) | the form of srsRAN's `srsran_scrambling_c` (D32, fair: the same task, sign scrambling of complex samples): the `±1.0` `f32` sequence `1 − 2 (p & 1)` of the same index bits formed into a buffer of 256 values (a lookup in a table of two), then VOLK's `volk_32fc_32f_multiply_32fc` (complex times real, two products per sample, no FMA in any of its protokernels) through VOLK's dispatcher on every level (`a_sse` at `sse2` and `sse4.2`, `a_avx` at `avx2` and `avx512`, `neonv8` at `neon`; the aligned protokernels, the buffers are 64-byte aligned); `simd128` is `info` (VOLK's `generic` code, D12) |
| `scramble_c32_b1_fallback` (the measured side is our fallback) | the same sequence plus VOLK's portable `volk_32fc_32f_multiply_32fc_generic` (`benches/refs/volk-generic.c`, `-O3 -ffp-contract=off` and the instruction set of the level) |
| `scramble_c32_b2`, `scramble_c32_b8` (the kernel `scramble`, M49 §5, `bits` 2 and 8, random phase bytes, 4096 samples per call) | the library composition (D32, fair): the phasor `exp(j 2π p / 2^bits)` of every sample looked up from a table of `2^bits` entries (built once from `COS_F32`, the table of the kernel) into a buffer of 256 samples, then VOLK's `volk_32fc_x2_multiply_32fc` on it: through VOLK's dispatcher at `sse2`, `sse4.2` (no FMA there: `generic` at `sse2`, `a_sse3` at `sse4.2`) and `simd128` (the `generic` protokernel, D12), and VOLK's protokernel without FMA (`benches/refs/volk-nofma.c`, `u_avx`, `neon`) at `avx2`, `avx512` and `neon`, where the dispatcher fuses; `sse2` and `simd128` are `info` (VOLK has no SIMD product there). At `bits = 2` ours does less work by design (the quadrant map has no product). At `bits = 8` the composition and the kernel do the same work since M49.1 (one table phasor and one product per sample, the kernel in one pass): `scramble_c32_b8` is fair and gated again (`release` at `sse4.2`, `avx2`, `avx512`, `neon`; D2092 revised for the `C32` rows by the maintainer decision of 2026-11-18) |
| `scramble_c32_b2_fallback`, `scramble_c32_b8_fallback` (the measured side is our fallback) | the same lookup plus VOLK's portable `volk_32fc_x2_multiply_32fc_generic` (`benches/refs/volk-generic.c`, `-O3 -ffp-contract=off` and the instruction set of the level); both are fair and `release` (`scramble_c32_b8_fallback` again since M49.1) |
| `scramble_cq15_b2`, `scramble_cq15_b8` | the same lookup from `COS_Q15` plus CMSIS-DSP's `arm_cmplx_mult_cmplx_q15` (generic C compiled for the level), marked (3.13 output): CMSIS writes Q3.13 (each partial product shifted by 17, truncating, no saturation), its accuracy is taken on the output times 4; our fallback as information. `scramble_cq15_b8` stays information only, reference does less work (D2092, kept by M49.1 §5): the CMSIS product truncates to Q3.13 without rounding or saturation, ours rounds and saturates the Q15 product |
| `scramble_cq15_b2_fallback`, `scramble_cq15_b8_fallback` (the measured side is our fallback) | the same CMSIS-DSP composition (3.13 output); `scramble_cq15_b8_fallback` is information only, reference does less work (D2092) |
| `scramble_c32_b2_dvbs2` (our `scramble` at `bits = 2` through the facade) | the per-symbol `switch` of GNU Radio's `dvbs2_physical_cc` (`benches/refs/gr-dvbs2-plscramble.c`, `R_n = p & 3` as an `int` sequence): information only, reference does different work (a switch per symbol on a precomputed sequence of a fixed frame layout) |
| `scramble_block` (the block `scramble`, M49 §6 with the index production of M49.1 §4, `bits = 8`, key `prbs15` from register 1, `C32`, 4096 samples per pass in calls of at most 256, per sample) | our own code as information (`info`, `Family::Own`; the PRBS indices and one call of `scramble` per `process`, whose entries carry the library rows) |
| `qam_soft_bits_qpsk`, `qam_soft_bits_qam256`, `qam_soft_bits_qam1024` (the kernel `qam_soft_bits` with 4-QAM, the constellation `qpsk`, 256-QAM and 1024-QAM, M44.2 §2), 4096 symbols per call, scale 1, per input symbol; `symbol_demap_qam256_block` (the block `symbol_demap` with `qam256`) | our fallback or our own code as information (`info`, `Family::Own`, D1201, D1504, D1524), as the 16-QAM and 64-QAM entries; QAM of any square order (decisions-log.md) |
| `qam_mod_qam256_block`, `qam_demod_qam256_block` (the blocks `qam_mod` and `qam_demod` with `qam256`, M44.2 §4, 8 samples per symbol, the block defaults; `qam_demod` in the steady state after a warm-up of 15360 symbols, reading the next of 8 windows of 4096 input samples per call), per output or input sample | our own code as information (`info`, `Family::Own`, D1216, D1524): chains of kernels whose library comparisons are their own entries (`costas_qam256_fallback`, `equalize_qam256_fallback`); QAM of any square order (decisions-log.md) |
| `bit_scramble_block`, `bit_descramble_block` (the blocks `bit_scramble` and `bit_descramble` of M44.2, the default `x23_x18` over plain bits, 4096 bits per call, per soft bit), `bit_scramble_in_place_block`, `bit_descramble_in_place_block` (the same with `process_in_place`, the call of the QAM frame composites, D1538) | GNU Radio's `scrambler_bb(0x21, 0, 22)` and `descrambler_bb(0x21, 0, 22)` of gr-digital (`benches/refs/gr-scrambler.c`) on the hard bits of the same input (`Family::GnuRadio`, status `release` at every level as the CRCs of D527, D1533); no SIMD library has a self-synchronizing scrambler, so at the SIMD levels the verdict is the efficiency; QAM of any square order (decisions-log.md) |
| `snr_estimate_block`, `snr_estimate_block_svr` (the block `snr_estimate` of M44.4 §1 through `observe`, `qam16` at Es/N0 20 dB, a window and a call of 4096 symbols, per input symbol) | GNU Radio `mpsk_snr_est_cc` with `SNR_EST_M2M4` and with `SNR_EST_SVR` (`tag_nsamples` 4096, `alpha` 0.001), both without the copy of the pass-through; information, reference does different work (D2504) |
| `fft` C32, N = 64, 1024, 4096 | FFTW at `sse2`, rustfft (`FftPlanner`, forward, in place) elsewhere |
| `fft` CQ15, N = 1024, block float and fixed scaling (noise of 8000 LSB, and of full scale for fixed scaling) | KissFFT `FIXED_POINT=16`; Ne10 int16 as information on `neon` |
| `rfft` f32, N = 1024 | FFTW's real-input transform (`fftwf_plan_dft_r2c_1d`, out of place) at `sse2` and `sse4.2` (FFTW has no codelets above SSE2, so it is the stronger reference there too), rustfft's real-input planner (`realfft` 3.5, forward, out of place) elsewhere; the other of the two and our fallback as information (FFTW also on `neon`) |
| `fft_64_fallback`, `fft_1024_fallback`, `fft_4096_fallback`, `rfft_1024_fallback`, `fft_cq15_1024_fallback` (block float), `fft_cq15_1024_default_fallback` (fixed scaling) (the measured side is our fallback, compiled for the level; the inputs of `fft_*`, `rfft_1024`, `fft_cq15_1024` and `fft_cq15_1024_default`) | CMSIS-DSP 1.18.0 generic C, compiled for the same level: `arm_cfft_f32` (the constant instances of `arm_const_structs.c`, bit reversal on) for C32, `arm_rfft_fast_f32` (packed output, unpacked outside the counted region) for `rfft`, `arm_cfft_q15` for CQ15 with KissFFT `FIXED_POINT=16` as information. The f32 transforms are the same work and accuracy class (137 to 141 dB on both sides), but CMSIS-DSP's radix-8/4 passes need fewer operations than the radix-2 butterflies of our contract; since D170 (three stages per pass off x86, the bit reversal in `N/4` steps) the entries are `release` except `rfft` at `sse4.2` (1.11 instructions) and N = 4096 at `simd128` (1.11), which are `dev`. `arm_cfft_q15` is a lower accuracy class (truncating shifts, 41.6 dB against 46.1 dB fixed scaling and 68.0 dB block float of ours, which rounds every butterfly) and has no block floating point: the block-float entry is `info` everywhere; the fixed-scaling entry is `release` on every level since D171 (no clamps where none can act; it was `info` at `sse2`, `neon` and `simd128`, where CMSIS-DSP was cheaper by more than the release limit). Not measured at `avx2-runtime` |
| `xcorr_fft_n512` C32 (M50 §5: `L` 127 random complex taps, `N` 512, `B` 386, 10 blocks = 3860 samples of noise per call), `xcorr_fft_n4096` C32 (`L` 1023, `N` 4096, `B` 3074, one block per call), per output sample | liquid-dsp 1.8.3 `fftfilt_cccf` (`benches/refs/liquid-fftfilt.c`, D2619): the same work (a streaming complex FFT fast convolution with complex taps in `f32`; liquid by overlap-add with a transform of `2 n` points per block of `n` outputs, ours by overlap-save), with the same taps, the block length `n` the power of two nearest to our `B` (512 and 4096, so liquid transforms 1024 and 8192 points), the same samples per call (the wrapper carries a partial block to the next call), over FFTW 3.3.10 (liquid's default build where FFTW is found; FFTW has no codelets above SSE2 in this build, as for `rfft`); at `simd128` liquid's own FFT, information (FFTW has no wasm32 build); our fallback as information. GNU Radio `fft_filter_ccc` (FFTW plus VOLK `volk_32fc_x2_multiply_32fc`) is the same method and the same work as the FFTW build of liquid and is not added as a separate row |
| `xcorr_fft_n512_fallback`, `xcorr_fft_n4096_fallback` C32 (the measured side is our fallback) | the same `fftfilt_cccf` over liquid's own FFT (portable C, compiled for the level) |
| `correlate_block` (the block `correlate`, M50 §6, the default `prbs7` code: `L` 127, `N` 512, 3860 samples per call, per output sample) | our own code as information (`info`, `Family::Own`; the library comparison of the kernel is the entry `xcorr_fft_n512`) |
| `halfband_decimate` C32, `halfband_decimate_real` f32 (47 taps, 24 non-zero odd taps and the centre tap, decimation by 2) | liquid-dsp `resamp2_crcf_decim_execute` / `resamp2_rrrf_decim_execute` with SDRk's taps in its prototype (liquid's `m = 12`), its SIMD dot product of the level; `halfband_decimate` also reports our `fir_decimate` by 2 with the expanded taps (`halfband_ref_fir_decimate`, `ours:` in the results) as information, the work the halfband structure saves (D69) |
| `halfband_decimate` CQ15, `halfband_decimate_real` Q15 | no SIMD library: our fallback (`info`, efficiency verdict as the Q15 biquads, D21), the CMSIS-DSP composition as information |
| `real_to_complex` C32 | liquid-dsp `firhilbf_decim_execute_block` (a delay branch and one real dot product of 24 taps per output, a sign toggle) with its SIMD dot product; at `sse2` also libairspy's `iqconverter_float` (SSE2) as the second gate, the entry `real_to_complex_airspy` (measured at `sse2` only, `release`: both must pass, D101); the separate shift followed by `halfband_decimate` (`halfband_ref_shifted`) as information |
| `real_to_complex` CQ15 | libairspy's `iqconverter_int16`, provisionally `info` (open item I23-1); our fallback and the separate shift followed by `halfband_decimate` as information |
| `halfband_decimate_fallback` C32, `halfband_decimate_real_fallback` f32, `real_to_complex_fallback` C32 | liquid-dsp with the portable dot product; for `real_to_complex_fallback` libairspy's generic code as information |
| `halfband_decimate_cq15_fallback`, `halfband_decimate_real_q15_fallback` | the CMSIS-DSP composition (CMSIS-DSP has no halfband decimator): the polyphase split of the input, `arm_fir_q15` with the 24 compacted taps on the even samples, `arm_scale_q15` with the centre tap on the odd samples (delayed by 12) and `arm_add_q15`; the complex type on the planar `re` and `im` as for the FIR family (D69) |
| `real_to_complex_cq15_fallback` | libairspy's `iqconverter_int16` (generic C), provisionally `info` (open item I23-1) |
| `stereo_matrix` C32 (`s + 2d`, `s - 2d` on interleaved pairs) | the VOLK 3.3.0 composition at the same level, as a VOLK user writes it (VOLK has no kernel for it): `volk_32fc_deinterleave_32f_x2` into `s` and `d`, `volk_32f_x2_add_32f` for `d2 = d + d`, `volk_32f_x2_add_32f` for `l = s + d2`, `volk_32f_x2_subtract_32f` for `r = s - d2` and `volk_32f_x2_interleave_32fc`, through the dispatchers on 64-byte aligned temporaries; at `simd128` the `generic` protokernels of `volk-wasm.c`. **The composition makes five passes over memory where our kernel makes one** (D108, D109); this is a property of the reference and is not corrected in the count. At `avx512` the dispatcher takes AVX-512F for the first four and AVX for `interleave` (VOLK has no AVX-512 implementation of it). At `neon` the cheapest VOLK combination is information |
| `stereo_matrix` CQ15 (`sat16(s +- 2d)`, `d2` exact) | no fair library for the SIMD branch (VOLK has `volk_16ic_deinterleave_16i_x2` and `volk_16i_x2_add_saturated_16i`, but no saturating subtraction and no 16-bit interleave): our fallback as information with the efficiency verdict (D33, as the `Q15` biquads, D21); the CMSIS-DSP composition on planar data as information |
| `stereo_matrix_fallback` C32, `stereo_matrix_cq15_fallback` (the measured side is our fallback, compiled for the level) | CMSIS-DSP 1.18.0 generic C on planar `s` and `d` (deinterleaved once outside the counted region, the best case for CMSIS-DSP, as for `dot`, D57): `arm_add_f32` for `d + d` and `s + d2`, `arm_sub_f32` for `s - d2`; for `CQ15` `arm_add_q15` and `arm_sub_q15`. The `CQ15` composition saturates `d + d` before the final sum, where our kernel computes `d2` exactly in `i64` and saturates once, so the results differ in the saturation corners (the workload is uniform over the whole `i16` range, which saturates often; the accuracy column shows it: 12.2 dB against exact for CMSIS-DSP, 999 dB for ours). Not measured at `avx2-runtime` |
| `stereo_pll_fallback` f32 (4096 samples at 480 kHz, a locked pilot) | liquid-dsp 1.8.3 `nco_crcf` of the type `LIQUID_VCO_INTERP` with the multiplying phase detector of `nco_pll_real_example.c` (`benches/refs/liquid-nco.c`): per sample `nco_crcf_sincos` at `theta` and the same table lookup at `2 theta`, `e = pilot cos(theta)`, `nco_crcf_adjust_frequency(ki e)`, `nco_crcf_adjust_phase(kp e)` (both through `_constrain`, a `double` modulo per call) and `nco_crcf_step`; the gains are ours (`kp`, `ki` times two pi). liquid does less in the lock detector (it has none, nor the integrator limit) and more in the modulo; both are named here. A lower accuracy class: the 1024-entry table has an error of 4.7e-6 (106.5 dB below full scale) against 2.9e-7 (130.6 dB) of ours, and at the narrow gains of this loop the `float` modulo of `_constrain` rounds the corrections so coarsely that the loop does not follow the pilot as ours does (the accuracy column of the entry, 9.5 dB against 57.0 dB for ours, is the SNR of the difference channel against the ideal oscillator on the same input); the reference is a speed reference (precedent D58, `fm_mod` against GNU Radio) and the limit 1.10 is not relaxed (D110). The workload is the golden case `lock_f32_16384` at 480 kHz (L 1 kHz, R 3 kHz, the pilot 0.1 with real noise sigma 0.01 on the multiplex signal and 0.005 on the pilot), with exactly 162 pilot cycles per call so that the repeated buffer is continuous; every implementation runs 24 calls before the measurement, so the loop is locked and the lock decision is on. `stereo_pll` has no SIMD branch and therefore no per-level entry. Not measured at `avx2-runtime` |
| `stereo_pll_q15_fallback` | no fixed-point PLL in any library (liquid-dsp's `nco_crcf` is float only, CMSIS-DSP has no PLL): the loop of the kernel with the oscillator of CMSIS-DSP (`arm_cos_q15` once and `arm_sin_q15` twice per sample, a 512-entry table with linear interpolation; the arithmetic around it is the kernel's) as information, with the efficiency verdict (D111, as `rotate` and `fm_mod` `CQ15`). Workload: `lock_q15_16384` at 480 kHz (the signals above scaled by 0.625). Not measured at `avx2-runtime` |
| `add` C32, `add_real` f32, `mul` C32, `mul_real` f32 (M30) | VOLK `volk_32fc_x2_add_32fc`, `volk_32f_x2_add_32f`, `volk_32fc_x2_multiply_32fc`, `volk_32f_x2_multiply_32f` on the same buffers (the same function and accuracy class). For `mul` C32 at `avx2`, `avx512` and `neon` VOLK's dispatcher takes a fused implementation (`a_avx2_fma`/`u_avx2_fma`, `neonv8`; GCC also contracts the products of VOLK's `neon` protokernel into `fmls`/`fmla` in VOLK's build), and since D326 ours fuses there as well, so VOLK's dispatch is the gate on every level (150.1 dB at those levels, the SNR of ours there on the count input). From D287 until D326 the gate there was VOLK's protokernel without fused operations, `volk_32fc_x2_multiply_32fc_u_avx` (`avx2`, `avx512`; VOLK has no AVX-512 implementation of the function) and `volk_32fc_x2_multiply_32fc_neon`, from the pinned release unchanged, compiled with `-O3 -ffp-contract=off` and the instruction set of the level (`benches/refs/volk-nofma.c`, `libvolknofma-<level>.a`, 148.5 dB); it stays as information (`information: volk-nofma`) |
| `add` CQ15, `add_real` Q15 | VOLK `volk_16i_x2_add_saturated_16i` (on the `2 n` values for `CQ15`, the same saturating sum) |
| `mul` CQ15, `mul_real` Q15 | no fair SIMD library: VOLK's `volk_16ic_x2_multiply_16ic` multiplies the 16-bit integers without the shift by 15 (the low halves wrap), a different function, and VOLK has no 16-bit real product. Our fallback as information with the efficiency verdict (D33) |
| `interleave` f32 to C32, `deinterleave` C32 and `CQ15` | VOLK `volk_32f_x2_interleave_32fc`, `volk_32fc_deinterleave_32f_x2` (both outputs written) and `volk_16ic_deinterleave_16i_x2`; planar buffers are prepared once outside the counted region. VOLK has no 16-bit interleave, so `interleave` `CQ15` is our fallback as information with the efficiency verdict |
| `oscillate` C32, `CQ15` | no library has an oscillator with this phase model (CMSIS-DSP has no NCO): our fallback as information with the efficiency verdict. The `C32` SIMD path is used only where it beats the fallback (M30 section 6); it does on every level (D289) |
| `add_fallback`, `add_cq15_fallback`, `add_real_fallback`, `add_real_q15_fallback`, `mul_fallback`, `mul_cq15_fallback`, `mul_real_fallback`, `mul_real_q15_fallback` (the measured side is our fallback) | CMSIS-DSP 1.18.0 generic C: `arm_add_f32`, `arm_add_q15` (on the `2 n` values for `CQ15`), `arm_cmplx_mult_cmplx_f32`, `arm_cmplx_mult_cmplx_q15`, `arm_mult_f32`, `arm_mult_q15`. The CMSIS-DSP `Q15` products truncate (Q3.13 for the complex one), where ours round once, so their accuracy column is lower |
| `interleave_fallback`, `interleave_cq15_fallback`, `deinterleave_fallback`, `deinterleave_cq15_fallback` (the measured side is our fallback) | VOLK 3.3.0's portable `generic` protokernels of `volk_32f_x2_interleave_32fc`, `volk_32fc_deinterleave_32f_x2` (both outputs written) and `volk_16ic_deinterleave_16i_x2`, from the pinned release unchanged, compiled with `-O3 -ffp-contract=off` and the `-march` of the level (`benches/refs/volk-generic.c`, `libvolkgeneric-<level>.a`, family `FallbackVsVolkGeneric`, implementation `volk-generic`, D297); CMSIS-DSP has no interleave. VOLK has no 16-bit interleave (`volk_32f_x2_s32f_interleave_16ic` converts scaled floats, a multiply and a rounding per value), so `interleave_cq15_fallback` is gated against the loop of `volk_32f_x2_interleave_32fc_generic` with `int16_t` elements, the mirror of `volk_16ic_deinterleave_16i_x2_generic`. Not measured at `avx2-runtime` |
| `oscillate_fallback` C32, `oscillate_cq15_fallback` | CMSIS-DSP `arm_sin_cos_f32` of the angle in degrees (a 512-entry sine table with cubic Hermite interpolation, the derivative read from the table shifted by a quarter turn) and `arm_cos_q15` plus `arm_sin_q15` (512 entries, linear interpolation) per sample, with the phase accumulator of our loop; a lower accuracy class for `Q15`; the f32 function is more accurate than our table at harmonic 1 (140.8 against 133.2 dB), so `oscillate_fallback` C32 carries an accuracy waiver (open item I30-1) |
| `am_mod` C32, `am_mod_cq15` (M35; the block defaults carrier 0.5 and depth 0.8 on a 1 kHz tone of 0.6 plus uniform noise of 0.3 at 48 kHz) | no SIMD library has an AM modulator (VOLK, KFR and CMSIS-DSP have none; liquid-dsp's `ampmodem` is scalar C): our fallback as information with the efficiency verdict; for `CQ15` the CMSIS-DSP composition of `am_mod_cq15_fallback` as information (D440) |
| `am_mod_fallback` C32 (the measured side is our fallback) | liquid-dsp 1.8.3 `ampmodem_modulate_block` of type `LIQUID_AMPMODEM_DSB` with the carrier (`benches/refs/liquid-ampmodem.c`, `-O3`, the generic flags of the other liquid references), `mod_index` = depth: `x * mod_index + 1` per sample as a complex value, without the clamp, the NaN rule and the carrier product of ours, so it does less work; its accuracy is that of its output times the carrier. `dev` with a waiver at `sse2`, `sse4.2` and `neon` (accepted by the maintainer, D586; at `neon` also an accuracy waiver: GCC contracts liquid's loop into `fmla`, 152.1 dB against our 150.8 dB), `release` at `avx2`, `avx512` and `simd128` (D441 to D443; counted 4.52 / 2.78 instructions per sample at `sse2` and `sse4.2`, 2.02 / 2.78 at `avx2`, 1.24 / 2.78 at `avx512`, 3.52 / 1.77 at `neon`, 10.54 / 46.02 at `simd128`, where clang leaves liquid's loop scalar). Not measured at `avx2-runtime` |
| `am_mod_cq15_fallback` (the measured side is our fallback) | the CMSIS-DSP 1.18.0 composition (CMSIS-DSP has no AM modulator): `arm_scale_q15` by `round(c m 32768)` (truncating), `arm_offset_q15` by `round(c 32768)` in place (saturating) and a plain loop that writes `(re, 0)`; one truncated product instead of our two rounded ones. Not measured at `avx2-runtime` |

VOLK's dispatcher is pinned per level by a `volk_config` that xtask writes (`benches/target/<level>/volk-pinned/volk/volk_config`), and the measured processes run without `VOLK_GENERIC` and without the variables that override single kernels:

- `sse2`, `sse4.2`: the defaults of VOLK's `sse2_64_mmx` and `sse4_2_64_mmx` machines, which the emulated CPUs select.
- `avx2`, `avx2-runtime`: the implementations that need no more than AVX2 and FMA, so that a runner with AVX-512 does not dispatch `rotator2`, `atan2` and `dot_prod` to AVX-512 code.
- `avx512`: VOLK's own choice, as our kernels use AVX-512 there too (the `stereo_matrix` composition then runs `deinterleave`, `add` and `subtract` in AVX-512F and `interleave`, which has no AVX-512 implementation, in AVX).
- `neon`: the default dispatch of VOLK's `neonv8` machine (D11), pinned so that a change of VOLK's ranking shows up as a missing implementation instead of a silently different reference. Every VOLK kernel also gets the cheapest combination of VOLK implementations of the machine as information (`goal: cheapest VOLK implementation`).
- `simd128`: VOLK has no wasm32 machine; its `generic` protokernels compiled for wasm32 with `simd128` are what its dispatcher would take. They gate `mag_sq`, `mag` (D12 extended in M20: the `generic` protokernel of `volk_32fc_magnitude_32f` is the code its dispatcher would take, as for `mag_sq`) and the conversions only (D12); for `rotate`, the FIR family and `quad_demod` they are scalar code and the entries are `info`. Since D29 the biquads no longer follow this rule (D15 applied it with CMSIS-DSP's generic C as the gate): KFR, built in its own Emscripten configuration, gates `biquad` C32 and `biquad_real` f32 as on the other levels.

The halfband references (M23, maintainer decisions of 2026-10-31) get SDRk's taps, so that their accuracy against the same `f64` model is comparable: liquid's `resamp2` takes a prototype of `4 m + 1` taps (the 47 taps between two zero taps, divided by the centre tap because its delay branch has the gain 1, with `set_scale(centre)` restoring it), `firhilbf` the same prototype (its output is `j / centre` times SDRk's `real_to_complex`; the mapping is applied outside the counted region), and libairspy's converters the 47 taps (their output is `-conj` of SDRk's; the Q15 converter hard-wires the centre tap 0.5). liquid's `_create` functions design a Kaiser filter; the wrapper builds the same objects from the given prototype with the code of `_create` that follows the design, so that liquid's design code is not linked. liquid selects the dot product at run time from the CPU; the wrapper fixes it per level (above), so the reference does not depend on the CPU that counts. The SSE loop of `dotprod_rrrf` (`resamp2_rrrf`, `firhilbf`) needs the SSE3 horizontal add (`HAVE_SSE3`, defined by liquid's autotools build): it is the reference from `sse4.2` (compiled with `-DHAVE_SSE3=1`) and at `sse2` the portable loop is (maintainer decision of 2026-10-31); the SSE loop of `dotprod_crcf` needs SSE only. The portable and SIMD methods are compiled without contraction into FMA (D5). Our own kernels on other workloads are reported as `ours:<kernel>` (`Reference::OwnKernel`): information that shows the work the specialised kernel saves, never the judged comparison. The counts are per input sample of the call (4096 inputs, 2048 outputs); libairspy converts in place, so its wrapper copies the input first (a `memcpy` of 4096 values, counted).

### Where no fair reference exists

- VOLK has no SSE implementation of `rotator2` and `atan2`; at `sse2` and `sse4.2` the dispatcher takes their scalar `generic` code, so `rotate` and `quad_demod` are `info` at both levels. VOLK has no implementation of `multiply_conjugate`, `8i_s32f_convert_32f` and `8i_convert_16i` below SSE3 and SSE4.1 either, but the `generic` code of the conversions is vectorized with SSE2 by the compiler, so `convert_cs8`, `convert_cu8`, `convert_cq15_cs8` and `convert_cq15_cu8` are judged against it at `sse2`.
- At `simd128`, see above.
- `stereo_matrix` `CQ15` has no fair SIMD reference (no saturating 16-bit subtraction and no 16-bit interleave in VOLK, CMSIS-DSP is generic C): `info` with the efficiency verdict; its fallback is gated against the CMSIS-DSP composition (`stereo_matrix_cq15_fallback`). `stereo_pll_q15_fallback` has no reference at all (no fixed-point PLL in any library): `info` with the CMSIS-DSP oscillator loop beside it. `stereo_pll` has no SIMD branch (the phase of every sample depends on the detector output of the previous one and the loop is bit-exact), so it has no entry other than the two fallback entries.
- `mul` `CQ15`, `mul_real` `Q15`, `interleave` `CQ15` and `oscillate` (M30) have no fair SIMD reference, see the table above: `info` with the efficiency verdict, with the CMSIS-DSP entries beside them for the fallbacks. All M30 entries in `benches/limits.json` carry the waiver `provisional` where the status is not `release`.
- `info` entries are reported with their numbers and never judged; the reason is the `waiver` text of the entry.
- `preemph_limit` (f32) and `preemph_limit_q15` (M25) have no reference: GNU Radio's `fm_preemph` and liquid-dsp's `iirfilt` implement the one-pole pre-emphasis only, no library has a peak limiter of this kind, so a comparison would compare a partial workload (D195). Both are `info` against our own fallback at every level, `avx2-runtime` included, and carry no efficiency gate.
- `am_mod` (`C32`) and `am_mod_cq15` (M35) have no SIMD reference: no library has a vectorized AM modulator, so both are `info` against our own fallback with the efficiency verdict (D440); the fallbacks are gated against liquid-dsp's `ampmodem` and the CMSIS-DSP composition, see the table above. The block `ssb_mod` is a chain of `interleave`, `rotate` and `fir` and has no entry of its own.
- `noise_blank` (`C32`), `noise_blank_cq15` and `auto_notch` (`f32`) (M33) are per-sample recursions that stay with the fallback on every backend: the facade entry is `info` against our own fallback at every level, `avx2-runtime` included, without an efficiency gate (D420). The workloads: `noise_blank` at 192 kHz with a hold of 0.2 ms and a lookahead of 50 us (the spec's defaults before D425) on noise with an impulse of 3 samples in every 512, `auto_notch` with 32 taps on 12 kHz audio with a 1.5 kHz whistle. The `noise_blank` fallback runs a call in sections of 128 samples (D424): one `memcpy` of the whole call is `rep movsb` in glibc, which the simulated cycles cost per byte (811.18 cycles per sample on `znver3` at `avx2` before, 11.58 after). `auto_notch` is bound by the latency of its sums, which run in a fixed ascending order shared bit for bit with the reference.
- `preemph_limit_stereo` (`C32`) and `stereo_encode` (`C32` to `f32`) (M36) stay with the fallback on every backend (the limiter is a per-sample recursion with one envelope for both channels, the encoder reads the pilot and the subcarrier from the interpolated tables of `rotate` at the running phase): the facade entry is `info` against our own fallback at every level, `avx2-runtime` included, without an efficiency gate and without a `_fallback` entry (D484). No library has a stereo limiter with a linked gain or a stereo multiplex encoder. The workloads (D485): `preemph_limit_stereo` as `preemph_limit` (48 kHz, 50 us, limit 0.5) with a burst on the left channel and a later one on the right, `stereo_encode` with the block defaults at 192 kHz on left 1 kHz 0.6 and right 3 kHz 0.4 plus noise.
- The protocol entries of `sdrk-proto` (M37) have no SIMD branch (D496). The CRCs (`crc16_ccitt_false`, `crc16_x25`, `crc32`, `crc24_adsb`) are gated against GNU Radio's generic CRC `gr::digital::crc` of gr-digital with the same parameters and messages (`Family::GnuRadio`, D527), our code beside it as information; `frame_decode` and the blocks `frame_encode_block` and `frame_decode_block` have no reference (no C library has this framing): `info` against our own code at every level, `avx2-runtime` included; figures in Protocol code (decisions-log.md).
- `symbol_sync` (`C32`) and `symbol_sync_real` (`f32`) (M38) stay with the fallback on every backend (every strobe position depends on the timing error of the strobe before it): the facade entry is `info` against our own fallback at every level, `avx2-runtime` included, without an efficiency gate and without a `_fallback` entry (D562). No library runs the same loop: GNU Radio's `symbol_sync_cc`/`_ff` interpolate with an 8-tap MMSE filter bank or a polyphase filter bank (and keep the detector history and the stream tags in C++ containers), liquid-dsp's `symsync` filters with a polyphase matched filter bank and its derivative; `symbol_sync` uses a 4-point cubic Lagrange interpolator, so either library would count a different amount of work. The workloads (D563) are the golden-case shapes with the block defaults (Gardner): BPSK with a raised cosine 0.35 at 4.1 samples per symbol, and boxcar-filtered real levels at 8 samples per symbol, both with noise; a sample is an input sample (cycles per symbol = cycles per sample times the samples per symbol). The block `symbol_demap` has entries only for its QAM constellations (`symbol_demap_qam16_block`, `symbol_demap_qam64_block`, M44, D1201), which run the kernel `qam_soft_bits` (entries `qam_soft_bits_qam16`, `qam_soft_bits_qam64`, D1412).
- `costas` (`C32`) (M40) stays with the fallback on every backend (the phase of every symbol depends on the detector output of the symbol before it): the facade entry is `info` against our own fallback at every level, `avx2-runtime` included, without an efficiency gate (D608); the fallback is gated against GNU Radio's `costas_loop_cc` in `costas_fallback` (D607). The workload: QPSK with one sample per symbol, a start phase of 0.7 rad, 0.002 cycles per symbol and uniform noise of 0.15 per component, with the constellation `qpsk` and the defaults `loop_bw` 0.02, `damping` 0.7071, `max_offset` 0.05; a sample is a symbol. `design::rrc` has no entry (a filter design, D606).
- `costas_qam16`, `costas_qam64`, `equalize_qam16` and `equalize_qam64` (`C32`) (M44, D1210, D1601): the QAM paths of the carrier loop and of the equalizer. `costas` stays with the fallback (the recursion of D608), `equalize` runs the fallback's loop on every backend (D841); the facade entries are `info` against our own fallback at every level, `avx2-runtime` included. The fallbacks are gated against the C translations of GNU Radio's `constellation_receiver_cb` and of `linear_equalizer` deciding on a `constellation_rect` (`benches/refs/gr-constellation.c`); see QAM carrier loop and equalizer (decisions-log.md).
- `dot_complex` (`C32`) and `equalize` (`C32`) (M43): `dot_complex` is gated against VOLK's `volk_32fc_x2_dot_prod_32fc`, at the levels where VOLK's dispatch fuses against its protokernel without FMA (D821), and its fallback against CMSIS-DSP's `arm_cmplx_dot_prod_f32` (D822); the judged levels are `release` since the count of 2026-11-07. `equalize` has no SIMD library reference: the facade entry is `info` against our own fallback at every level, `avx2-runtime` included (D823), and since D841 every backend runs the fallback, and `equalize_fallback` is gated against the C translation of GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` (D838, `release` since the count of 2026-11-07). The workload is the block default (11 taps, no feedback, QPSK, `mu_cma` 0.05, `mu_dd` 0.02, `cma_symbols` 1000, `phase_bw` 0.02; GNU Radio with the same normalized step 0.02) on QPSK through `1 + 0.3 e^(j0.6) z^-1 + 0.1 e^(-j2) z^-2` with a carrier phase of 0.7 rad and uniform noise of 0.05; the state carries over between the calls, so the counted calls run in decision mode.
- `rds_shape` (`f32`) and `stereo_encode_rds` (`C32` and `f32` to `f32`) (M42) stay with the fallback on every backend (the shaper reads four interpolated values per sample from the pulse table at an f64 position and moves its bit window through a callback, the encoder is `stereo_encode` plus a third lookup in the tables of `rotate` at the tripled phase): the facade entry is `info` against our own fallback at every level, `avx2-runtime` included, without an efficiency gate and without a `_fallback` entry (D757). Neither GNU Radio nor liquid-dsp has an RDS modulator in-tree (GNU Radio's RDS support is the out-of-tree receiver gr-rds; liquid-dsp has no RDS), and no library has a stereo multiplex encoder with RDS. The workloads (D757): `rds_shape` at 192 kHz on random amplitudes +-1 (a stored cycle of 4096 bits), `stereo_encode_rds` the defaults of `fm_stereo_rds_mod` (pilot 19 kHz, `main` 0.87, `pilot` 0.09, `rds` 0.04) at 192 kHz on the audio of the `stereo_encode` workload and the `rds_shape` output of random bits.
- The RDS blocks of M41 (`rds_demod_block`, `rds_decode_block`) are composite blocks without a kernel of their own and have no reference (no C library has them): `info` against our own code at every level, `avx2-runtime` included (D678); `rds_demod` follows the SIMD paths of `interleave`, `rotate` and `fir_decimate` at each level, `rds_decode` is scalar. Figures in RDS blocks (decisions-log.md).
- Under the [measurement policy](#what-is-compared-with-what-d32) our own fallback is no fair gate for a SIMD branch, and since the fallback round no SIMD entry is `release` against it (issue #20, group A). `fm_mod` `C32` is `info` (D58), with GNU Radio's scalar `frequency_modulator_fc` and the fallback as information. `mag_sq` `CQ15` is `info` with the efficiency verdict (no SIMD library has the rounding and saturation of its rule; CMSIS-DSP's scalar `arm_cmplx_mag_squared_q15` gates the fallback entry). `quad_demod` `CQ15` (round 2), `rotate` `CQ15` (D45), `fm_mod` `CQ15` (D46), the biquad Q15 entries (D21, D39), `window` `CQ15` and `power_db` `Q15` (D43), the Q15 polyphase entries (D44) and `dot` `CQ15`, `dot_real` `Q15` and `scale` `CQ15` (D48) are `info` and judged by their simulated cycles and efficiency (see the notes below). `squelch` `C32` is gated against GNU Radio's `pwr_squelch_cc` (D41). The `C32` fallbacks of `agc` and `quad_demod` are gated against GNU Radio (`agc_fallback`, `quad_demod_fallback`) and the `CQ15` fallback of `quad_demod` against CMSIS-DSP (`quad_demod_cq15_fallback`). The fallbacks of `dc_block_real` and `dc_block` (`f32`/`C32`) are gated against liquid-dsp's DC blocker and that of `deemphasis` `f32` against GNU Radio's `fm_deemph` (D53, entries `<kernel>_fallback`); their SIMD entries (the `f64` block scan) stay `info` against the fallback, since no SIMD library has these recursions. The `info` entries of `squelch` `CQ15`, `agc` `CQ15` and the `Q15`/`CQ15` forms of `dc_block_real`, `deemphasis` and `dc_block` compare the fallback with itself and still need a scalar reference (group B; no established fixed-point implementation was found).


## Cycle results

Simulated cycles per sample of our kernel and of its comparison on the reference model and the counter-model of each level, from the committed `benches/results/cycles/<platform>.json` (`x86_64`, `x86_64-avx512`, `aarch64`). The tables are **generated** by `python3 -I benches/cycle_tables.py` from these files and are never edited by hand (`--check` fails when they differ; CI runs it). The files are written by `cargo measure ... --merge` / `--all`; the record of reference is the artifact `cycle-results` of the full CI measurement ([Determinism and the reference environment](#determinism-and-the-reference-environment)). The tables below are the full CI measurement of 2026-11-24 on commit 484abc1b (D2745: runs 45359765815 for `sse2`, `sse4.2` and `avx512`, 45364395319 for `avx2`, `avx2-runtime` and `neon`).

Per row: ours and the comparison in cycles per sample on both models, the two ratios (ours / comparison, lower is better), the accuracy of both sides (SNR in dB against the f64 reference, ours / comparison, `exact` for an exact result, `(accuracy trade)` for an entry kept `release` by D2760) and the comparison. The status of the entry (`release`, `dev`, `info`), its waivers and the goal are in `benches/limits.json` and judged by `cargo xtask check cycles`; a row without a library comparison (`info`) shows our own fallback or an own kernel as the comparison. An entry that could not be simulated is listed below its table with its note. Only the ratios mean something (D33, D68): the absolute cycles are no hardware figures. Models: `sse2` and `sse4.2` `skylake` / `znver2`, `avx2` and `avx2-runtime` `znver3` / `alderlake`, `avx512` `znver4` / `sapphirerapids`, `neon` `cortex-a72` / `cortex-a76`. The level `avx2-runtime` shows our run-time-detection build against our own `avx2` build (limit 1.02). Cycle mode is `whole-loop`.


<!-- cycle-tables:begin -->

#### Simulated cycles at `sse2` (`x86_64`)

Reference model `skylake`, counter-model `znver2`, mode `whole-loop`; execution counts: Intel SDE -dcfg, block and branch counts (Intel SDE, CPUID of a strict SSE2 x86-64 CPU).

| Entry | ours skylake (cycles/sample) | comparison skylake (cycles/sample) | ratio skylake (ours/comparison) | ours znver2 (cycles/sample) | comparison znver2 (cycles/sample) | ratio znver2 (ours/comparison) | comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `convert_cq15_cs8` | 0.26 | 0.33 | 0.80 | 0.26 | 0.42 | 0.63 | volk_8i_convert_16i |
| `convert_cq15_cs8_fallback` | 0.30 | 0.32 | 0.93 | 0.36 | 0.42 | 0.86 | CMSIS-DSP 1.18.0 arm_q7_to_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_cq15_cu8` | 0.26 | 0.33 | 0.80 | 0.32 | 0.42 | 0.75 | volk_8i_convert_16i on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cq15_cu8_fallback` | 0.33 | 0.32 | 1.03 | 0.42 | 0.42 | 1.01 | CMSIS-DSP 1.18.0 arm_q7_to_q15 on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_cs16` | 0.53 | 1.17 | 0.45 | 0.60 | 1.43 | 0.42 | volk_16i_s32f_convert_32f |
| `convert_cs16_fallback` | 0.78 | 0.78 | 1.00 | 0.97 | 0.96 | 1.00 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_cs8` | 0.77 | 1.02 | 0.76 | 0.63 | 1.02 | 0.62 | volk_8i_s32f_convert_32f |
| `convert_cs8_fallback` | 1.03 | 1.02 | 1.02 | 1.09 | 1.02 | 1.08 | CMSIS-DSP 1.18.0 arm_q7_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_cu8` | 0.77 | 1.02 | 0.76 | 0.63 | 1.02 | 0.62 | volk_8i_s32f_convert_32f on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cu8_fallback` | 1.03 | 1.02 | 1.02 | 1.10 | 1.02 | 1.08 | CMSIS-DSP 1.18.0 arm_q7_to_float on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_out_cs16` | 1.10 | 1.03 | 1.06 | 1.03 | 1.03 | 1.00 | volk_32f_s32f_convert_16i |
| `convert_out_cs16_fallback` | 2.05 | 2.97 | 0.69 | 2.04 | 3.47 | 0.59 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_out_cs8` | 1.02 | 1.02 | 1.00 | 1.03 | 0.96 | 1.08 | volk_32f_s32f_convert_8i |
| `convert_out_cs8_fallback` | 2.32 | 3.08 | 0.75 | 2.48 | 3.67 | 0.68 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_out_cu8` | 1.59 | 1.27 | 1.25 | 1.55 | 1.27 | 1.23 | volk_32f_s32f_x2_convert_8u |
| `convert_out_cu8_fallback` | 5.43 | 3.08 | 1.76 | 5.74 | 3.67 | 1.56 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_out_real_rs16` | 0.56 | 0.52 | 1.07 | 0.53 | 0.52 | 1.02 | volk_32f_s32f_convert_16i |
| `convert_out_real_rs16_fallback` | 1.03 | 1.49 | 0.69 | 1.02 | 1.74 | 0.59 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_out_real_rs8` | 0.53 | 0.52 | 1.02 | 0.53 | 0.48 | 1.09 | volk_32f_s32f_convert_8i |
| `convert_out_real_rs8_fallback` | 1.17 | 1.54 | 0.75 | 1.25 | 1.84 | 0.68 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_out_real_ru8` | 0.81 | 0.64 | 1.26 | 0.79 | 0.64 | 1.24 | volk_32f_s32f_x2_convert_8u |
| `convert_out_real_ru8_fallback` | 2.72 | 1.54 | 1.76 | 2.88 | 1.84 | 1.57 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `convert_real_rs16` | 0.28 | 0.59 | 0.47 | 0.31 | 0.72 | 0.44 | volk_16i_s32f_convert_32f |
| `convert_real_rs16_fallback` | 0.40 | 0.39 | 1.03 | 0.49 | 0.48 | 1.01 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `rotate` | 2.08 | 11.05 | 0.19 | 1.99 | 12.13 | 0.16 | volk_32fc_s32fc_x2_rotator2_32fc |
| `rotate_cq15` | 3.79 | 12.10 | 0.31 | 3.79 | 12.37 | 0.31 | sdrk fallback |
| `rotate_cq15_fallback` | 12.10 | 36.49 | 0.33 | 12.37 | 34.93 | 0.35 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `rotate_fallback` | 8.64 | 15.45 | 0.56 | 8.88 | 16.79 | 0.53 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `scramble_block` | 6.21 | 6.21 | 1.00 | 5.99 | 5.99 | 1.00 | sdrk fallback |
| `scramble_c32_b1` | 0.89 | 1.99 | 0.45 | 0.83 | 2.40 | 0.35 | +-1.0 sequence from the index bits into a buffer of 256 + volk_32fc_32f_multiply_32fc (VOLK 3.3.0 dispatcher; the form of srsRAN srsran_scrambling_c) |
| `scramble_c32_b1_fallback` | 2.04 | 2.07 | 0.98 | 1.54 | 2.60 | 0.59 | +-1.0 sequence from the index bits into a buffer of 256 + VOLK 3.3.0 volk_32fc_32f_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2` | 2.85 | 6.39 | 0.45 | 2.77 | 8.80 | 0.31 | phasor lookup into a buffer of 256 + volk_32fc_x2_multiply_32fc (VOLK 3.3.0 dispatcher) |
| `scramble_c32_b2_dvbs2` | 2.85 | 3.55 | 0.80 | 2.77 | 4.44 | 0.62 | GNU Radio dvbs2_physical_cc PL scrambling, the per-symbol switch on R_n (benches/refs/gr-dvbs2-plscramble.c, main at ec1bf15; reference does different work; -O3) |
| `scramble_c32_b2_fallback` | 4.54 | 6.30 | 0.72 | 4.04 | 8.71 | 0.46 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8` | 3.40 | 6.39 | 0.53 | 3.74 | 8.80 | 0.42 | phasor lookup into a buffer of 256 + volk_32fc_x2_multiply_32fc (VOLK 3.3.0 dispatcher) |
| `scramble_c32_b8_fallback` | 6.18 | 6.30 | 0.98 | 6.88 | 8.71 | 0.79 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_cq15_b2` | 2.28 | 7.36 | 0.31 | 2.41 | 7.52 | 0.32 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `scramble_cq15_b2_fallback` | 4.03 | 7.36 | 0.55 | 3.53 | 7.52 | 0.47 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `scramble_cq15_b8` | 3.78 | 7.36 | 0.51 | 3.66 | 7.52 | 0.49 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `scramble_cq15_b8_fallback` | 9.44 | 7.36 | 1.28 | 8.86 | 7.52 | 1.18 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir` | 33.63 | 77.65 | 0.43 | 29.99 | 78.96 | 0.38 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 142.31 | 142.31 | 1.00 | 188.30 | 188.30 | 1.00 | sdrk fallback |
| `fir_cq15_fallback` | 142.31 | 177.35 | 0.80 | 188.30 | 220.90 | 0.85 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_decimate` | 6.87 | 12.46 | 0.55 | 6.08 | 13.37 | 0.45 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 30.40 | 30.40 | 1.00 | 45.20 | 45.20 | 1.00 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 30.40 | 24.93 | 1.22 | 45.20 | 25.79 | 1.75 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_decimate_fallback` | 15.09 | 38.80 | 0.39 | 15.82 | 23.74 | 0.67 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_fallback` | 42.70 | 132.97 | 0.32 | 51.73 | 111.96 | 0.46 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate` | 36.22 | 86.10 | 0.42 | 34.15 | 88.20 | 0.39 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 137.49 | 137.49 | 1.00 | 207.95 | 207.95 | 1.00 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 137.49 | 137.61 | 1.00 | 207.95 | 164.43 | 1.26 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate_fallback` | 69.95 | 141.75 | 0.49 | 71.95 | 114.81 | 0.63 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate_real` | 19.44 | 79.39 | 0.24 | 18.20 | 77.13 | 0.24 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 42.41 | 70.88 | 0.60 | 44.65 | 57.41 | 0.78 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate_real_q15` | 74.97 | 74.97 | 1.00 | 112.89 | 112.89 | 1.00 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 74.97 | 68.81 | 1.09 | 112.89 | 82.21 | 1.37 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate_short` | 7.16 | 49.49 | 0.14 | 8.99 | 58.82 | 0.15 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_short_fallback` | 14.74 | 24.83 | 0.59 | 19.29 | 22.72 | 0.85 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 8 (80 taps), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `resample` | 38.72 | 87.37 | 0.44 | 35.30 | 89.05 | 0.40 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 168.72 | 168.72 | 1.00 | 240.51 | 240.51 | 1.00 | sdrk fallback |
| `resample_cq15_fallback` | 168.72 | 136.68 | 1.23 | 240.51 | 163.55 | 1.47 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `resample_fallback` | 84.56 | 141.22 | 0.60 | 86.87 | 113.81 | 0.76 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `resample_real` | 22.41 | 80.59 | 0.28 | 22.04 | 77.71 | 0.28 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 84.46 | 70.61 | 1.20 | 86.75 | 56.90 | 1.52 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `resample_real_q15` | 103.67 | 103.67 | 1.00 | 144.16 | 144.16 | 1.00 | sdrk fallback |
| `resample_real_q15_fallback` | 103.67 | 68.34 | 1.52 | 144.16 | 81.78 | 1.76 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `mag` | 0.76 | 0.79 | 0.96 | 1.77 | 1.79 | 0.99 | volk_32fc_magnitude_32f |
| `mag_fallback` | 0.79 | 0.85 | 0.93 | 1.79 | 1.86 | 0.96 | CMSIS-DSP 1.18.0 arm_cmplx_mag_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `mag_q15` | 2.27 | 11.17 | 0.20 | 3.44 | 12.29 | 0.28 | volk_16ic_magnitude_16i |
| `mag_q15_fallback` | 4.88 | 39.58 | 0.12 | 6.41 | 45.02 | 0.14 | CMSIS-DSP 1.18.0 arm_cmplx_mag_q15 (Q2.14, arm_sqrt_q31), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `mag_sq` | 0.51 | 0.65 | 0.78 | 0.58 | 0.84 | 0.69 | volk_32fc_magnitude_squared_32f |
| `mag_sq_fallback` | 0.52 | 0.65 | 0.80 | 0.67 | 0.84 | 0.80 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `mag_sq_q15` | 0.33 | 0.82 | 0.40 | 0.33 | 0.90 | 0.37 | sdrk fallback |
| `mag_sq_q15_fallback` | 0.82 | 5.39 | 0.15 | 0.90 | 4.02 | 0.22 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_q15 (Q3.13, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `quad_demod` | 9.38 | 110.71 | 0.08 | 11.36 | 98.69 | 0.12 | volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f |
| `quad_demod_cq15` | 20.33 | 94.87 | 0.21 | 21.68 | 97.68 | 0.22 | sdrk fallback |
| `quad_demod_cq15_fallback` | 94.87 | 133.11 | 0.71 | 97.68 | 119.98 | 0.81 | CMSIS-DSP arm_cmplx_mult_cmplx_q15 + arm_atan2_q15 + arm_scale_q15 (generic C), conjugate in a plain loop |
| `quad_demod_fallback` | 53.33 | 68.55 | 0.78 | 51.61 | 63.79 | 0.81 | GNU Radio quadrature_demod_cf (gr-analog, fast_atan2f, generic compiler flags) |
| `fm_mod` | 7.97 | 47.68 | 0.17 | 8.37 | 60.01 | 0.14 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `fm_mod_cq15` | 8.56 | 10.74 | 0.80 | 8.56 | 11.73 | 0.73 | sdrk fallback |
| `fm_mod_cq15_fallback` | 10.74 | 25.62 | 0.42 | 11.73 | 23.96 | 0.49 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample of an integer phase accumulator, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fm_mod_fallback` | 8.94 | 47.68 | 0.19 | 10.89 | 60.01 | 0.18 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `squelch` | 12.14 | 27.74 | 0.44 | 9.14 | 26.37 | 0.35 | GNU Radio pwr_squelch_cc (squelch_base_cc_impl.cc of gr-analog, generic compiler flags) |
| `squelch_cq15` | 7.60 | 7.60 | 1.00 | 8.11 | 8.11 | 1.00 | sdrk fallback |
| `window` | 0.54 | 0.65 | 0.83 | 0.68 | 0.78 | 0.87 | volk_32fc_32f_multiply_32fc |
| `window_cq15` | 0.88 | 2.11 | 0.42 | 0.95 | 2.65 | 0.36 | sdrk fallback |
| `window_cq15_fallback` | 2.11 | 2.62 | 0.81 | 2.65 | 2.78 | 0.95 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `window_fallback` | 0.64 | 0.65 | 0.98 | 0.79 | 0.84 | 0.95 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `power_db` | 5.43 | 59.69 | 0.09 | 5.30 | 54.12 | 0.10 | volk_32f_log2_32f + volk_32f_s32f_multiply_32f + volk_32f_s32f_add_32f |
| `power_db_fallback` | 8.86 | 55.94 | 0.16 | 11.07 | 49.41 | 0.22 | CMSIS-DSP 1.18.0 arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `power_db_q15` | 5.43 | 10.53 | 0.52 | 5.26 | 13.43 | 0.39 | sdrk fallback |
| `power_db_q15_fallback` | 10.53 | 56.34 | 0.19 | 13.43 | 49.90 | 0.27 | CMSIS-DSP 1.18.0 arm_q15_to_float + arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `agc` | 13.76 | 42.10 | 0.33 | 11.54 | 38.11 | 0.30 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `agc_cq15` | 85.25 | 85.25 | 1.00 | 82.80 | 82.80 | 1.00 | sdrk fallback |
| `agc_fallback` | 18.78 | 42.10 | 0.45 | 17.66 | 38.11 | 0.46 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `dc_block` | 5.71 | 10.05 | 0.57 | 4.79 | 7.07 | 0.68 | sdrk fallback |
| `dc_block_cq15` | 8.06 | 8.06 | 1.00 | 10.05 | 10.05 | 1.00 | sdrk fallback |
| `dc_block_fallback` | 10.05 | 56.29 | 0.18 | 7.07 | 51.26 | 0.14 | liquid-dsp 1.8.3 iirfilt_crcf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real` | 4.61 | 10.03 | 0.46 | 3.69 | 7.04 | 0.52 | sdrk fallback |
| `dc_block_real_fallback` | 10.03 | 46.19 | 0.22 | 7.04 | 43.86 | 0.16 | liquid-dsp 1.8.3 iirfilt_rrrf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real_q15` | 6.06 | 6.06 | 1.00 | 7.55 | 7.55 | 1.00 | sdrk fallback |
| `deemphasis` | 4.15 | 15.09 | 0.28 | 3.59 | 12.09 | 0.30 | sdrk fallback |
| `deemphasis_fallback` | 15.09 | 13.15 | 1.15 | 12.09 | 17.11 | 0.71 | GNU Radio fm_deemph: iir_filter_ffd with the bilinear taps of fm_emph.py (double taps and accumulator, generic compiler flags) |
| `deemphasis_q15` | 7.03 | 7.03 | 1.00 | 8.53 | 8.53 | 1.00 | sdrk fallback |
| `biquad` | 12.69 | 28.36 | 0.45 | 9.42 | 27.58 | 0.34 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (sse2, -ffp-contract=off), two calls on planar re and im (deinterleaving not counted) |
| `biquad_cq15` | 24.94 | 24.94 | 1.00 | 31.17 | 31.17 | 1.00 | sdrk fallback |
| `biquad_cq15_fallback` | 24.94 | 60.21 | 0.41 | 31.17 | 52.19 | 0.60 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `biquad_fallback` | 25.00 | 32.54 | 0.77 | 18.60 | 24.54 | 0.76 | CMSIS-DSP 1.18.0 arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `biquad_real` | 12.31 | 14.18 | 0.87 | 9.05 | 13.79 | 0.66 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (sse2, -ffp-contract=off) |
| `biquad_real_fallback` | 14.27 | 24.04 | 0.59 | 11.38 | 18.03 | 0.63 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df2T_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `biquad_real_q15` | 14.12 | 14.12 | 1.00 | 15.13 | 15.13 | 1.00 | sdrk fallback |
| `biquad_real_q15_fallback` | 14.12 | 30.11 | 0.47 | 15.13 | 26.10 | 0.58 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `dot` | 0.60 | 0.61 | 0.98 | 0.69 | 0.65 | 1.05 | volk_32fc_32f_dot_prod_32fc |
| `dot_complex` | 1.32 | 2.67 | 0.49 | 1.33 | 2.19 | 0.61 | volk_32fc_x2_dot_prod_32fc |
| `dot_complex_fallback` | 5.02 | 8.28 | 0.61 | 3.77 | 6.28 | 0.60 | CMSIS-DSP 1.18.0 arm_cmplx_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `dot_cq15` | 1.01 | 2.20 | 0.46 | 0.82 | 2.53 | 0.33 | sdrk fallback |
| `dot_cq15_fallback` | 2.20 | 1.78 | 1.23 | 2.53 | 2.03 | 1.24 | CMSIS-DSP 1.18.0 arm_dot_prod_q15 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `dot_fallback` | 4.03 | 8.07 | 0.50 | 3.03 | 6.06 | 0.50 | CMSIS-DSP 1.18.0 arm_dot_prod_f32 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `dot_real` | 0.27 | 0.28 | 0.97 | 0.29 | 0.27 | 1.06 | volk_32f_x2_dot_prod_32f |
| `dot_real_fallback` | 4.04 | 4.04 | 1.00 | 3.03 | 3.03 | 1.00 | CMSIS-DSP 1.18.0 arm_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `dot_real_q15` | 0.34 | 0.98 | 0.34 | 0.35 | 0.98 | 0.36 | sdrk fallback |
| `dot_real_q15_fallback` | 0.98 | 0.89 | 1.10 | 0.98 | 1.02 | 0.96 | CMSIS-DSP 1.18.0 arm_dot_prod_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `scale` | 0.51 | 0.64 | 0.80 | 0.51 | 0.79 | 0.64 | volk_32f_s32f_multiply_32f |
| `scale_cq15` | 2.03 | 2.03 | 1.00 | 2.11 | 2.11 | 1.00 | sdrk fallback |
| `scale_cq15_fallback` | 2.03 | 3.78 | 0.54 | 2.11 | 3.65 | 0.58 | CMSIS-DSP 1.18.0 arm_scale_q15 (shift 0, truncating) on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `scale_fallback` | 0.63 | 0.63 | 1.00 | 0.79 | 0.79 | 1.00 | CMSIS-DSP 1.18.0 arm_scale_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `add` | 0.51 | 0.70 | 0.72 | 0.56 | 0.79 | 0.71 | volk_32fc_x2_add_32fc |
| `add_cq15` | 0.25 | 0.35 | 0.72 | 0.28 | 0.40 | 0.71 | volk_16i_x2_add_saturated_16i on the 2 n values |
| `add_cq15_fallback` | 0.35 | 3.66 | 0.10 | 0.46 | 3.78 | 0.12 | CMSIS-DSP 1.18.0 arm_add_q15 on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `add_fallback` | 0.70 | 0.70 | 1.00 | 0.91 | 0.91 | 1.00 | CMSIS-DSP 1.18.0 arm_add_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `add_real` | 0.26 | 0.36 | 0.71 | 0.28 | 0.40 | 0.70 | volk_32f_x2_add_32f |
| `add_real_fallback` | 0.29 | 0.36 | 0.82 | 0.36 | 0.46 | 0.77 | CMSIS-DSP 1.18.0 arm_add_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `add_real_q15` | 0.13 | 0.18 | 0.71 | 0.14 | 0.20 | 0.70 | volk_16i_x2_add_saturated_16i |
| `add_real_q15_fallback` | 0.15 | 1.83 | 0.08 | 0.18 | 1.89 | 0.10 | CMSIS-DSP 1.18.0 arm_add_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `mul` | 1.53 | 4.35 | 0.35 | 1.59 | 6.09 | 0.26 | volk_32fc_x2_multiply_32fc |
| `mul_cq15` | 5.03 | 5.03 | 1.00 | 3.90 | 3.90 | 1.00 | sdrk fallback |
| `mul_cq15_fallback` | 5.03 | 5.29 | 0.95 | 3.90 | 4.78 | 0.82 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_q15 (Q3.13, truncating, scaled by 4), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `mul_fallback` | 1.53 | 2.53 | 0.60 | 1.59 | 2.09 | 0.76 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `mul_real` | 0.26 | 0.36 | 0.71 | 0.28 | 0.40 | 0.70 | volk_32f_x2_multiply_32f |
| `mul_real_fallback` | 0.29 | 0.36 | 0.82 | 0.36 | 0.46 | 0.77 | CMSIS-DSP 1.18.0 arm_mult_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `mul_real_q15` | 0.97 | 0.97 | 1.00 | 1.02 | 1.02 | 1.00 | sdrk fallback |
| `mul_real_q15_fallback` | 0.97 | 1.21 | 0.81 | 1.02 | 1.39 | 0.73 | CMSIS-DSP 1.18.0 arm_mult_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `interleave` | 0.51 | 0.52 | 0.98 | 0.51 | 0.64 | 0.79 | volk_32f_x2_interleave_32fc |
| `interleave_cq15` | 0.26 | 0.26 | 0.99 | 0.26 | 0.31 | 0.84 | sdrk fallback |
| `interleave_cq15_fallback` | 0.26 | 0.26 | 1.00 | 0.31 | 0.33 | 0.95 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic on int16 (VOLK has no 16-bit interleave), benches/refs/volk-generic.c (-O3 -ffp-contract=off, sse2) |
| `interleave_fallback` | 0.51 | 0.52 | 0.98 | 0.61 | 0.64 | 0.94 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, sse2) |
| `deinterleave` | 0.51 | 0.52 | 0.98 | 0.51 | 0.65 | 0.79 | volk_32fc_deinterleave_32f_x2 |
| `deinterleave_cq15` | 0.34 | 1.52 | 0.23 | 0.40 | 0.89 | 0.45 | volk_16ic_deinterleave_16i_x2 |
| `deinterleave_cq15_fallback` | 1.02 | 1.27 | 0.80 | 0.64 | 0.73 | 0.88 | VOLK 3.3.0 volk_16ic_deinterleave_16i_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, sse2) |
| `deinterleave_fallback` | 0.51 | 0.58 | 0.88 | 0.54 | 0.71 | 0.77 | VOLK 3.3.0 volk_32fc_deinterleave_32f_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, sse2) |
| `oscillate` | 2.30 | 4.90 | 0.47 | 2.28 | 6.35 | 0.36 | sdrk fallback |
| `oscillate_cq15` | 3.78 | 5.34 | 0.71 | 3.79 | 6.82 | 0.55 | sdrk fallback |
| `oscillate_cq15_fallback` | 5.34 | 21.81 | 0.24 | 6.82 | 19.33 | 0.35 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample (512-entry table, linear interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `oscillate_fallback` | 4.90 | 56.29 | 0.09 | 6.35 | 51.64 | 0.12 | CMSIS-DSP 1.18.0 arm_sin_cos_f32 of the angle in degrees per sample (512-entry sine table, cubic Hermite interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_1024` | 7.19 | 6.91 | 1.04 | 6.33 | 6.78 | 0.93 | FFTW (fftwf_plan_dft_1d, FFTW_ESTIMATE, forward, out of place) |
| `fft_1024_fallback` | 31.44 | 26.16 | 1.20 | 29.67 | 28.91 | 1.03 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_4096` | 8.14 | 10.41 | 0.78 | 7.28 | 8.77 | 0.83 | FFTW (fftwf_plan_dft_1d, FFTW_ESTIMATE, forward, out of place) |
| `fft_4096_fallback` | 37.59 | 30.53 | 1.23 | 35.67 | 32.90 | 1.08 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_64` | 4.58 | 5.86 | 0.78 | 4.03 | 5.51 | 0.73 | FFTW (fftwf_plan_dft_1d, FFTW_ESTIMATE, forward, out of place) |
| `fft_64_fallback` | 20.78 | 15.15 | 1.37 | 19.54 | 17.76 | 1.10 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_cq15_1024` | 8.77 | 52.83 | 0.17 | 9.70 | 63.00 | 0.15 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 7.36 | 52.83 | 0.14 | 7.78 | 63.00 | 0.12 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 50.20 | 44.75 | 1.12 | 54.56 | 56.34 | 0.97 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_cq15_1024_fallback` | 83.13 | 44.75 | 1.86 | 91.71 | 56.34 | 1.63 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_cq15_1024_full_scale` | 7.68 | 52.83 | 0.15 | 8.03 | 63.00 | 0.13 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 4.71 | 5.63 | 0.84 | 4.40 | 4.58 | 0.96 | FFTW (fftwf_plan_dft_r2c_1d, FFTW_ESTIMATE, forward, out of place) |
| `rfft_1024_fallback` | 16.75 | 14.25 | 1.18 | 16.47 | 15.04 | 1.09 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `am_mod` | 1.09 | 1.09 | 1.00 | 1.16 | 1.16 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 2.62 | 2.62 | 1.00 | 3.12 | 3.12 | 1.00 | sdrk fallback |
| `am_mod_cq15_fallback` | 2.62 | 4.17 | 0.63 | 3.12 | 4.64 | 0.67 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `am_mod_fallback` | 1.09 | 0.59 | 1.84 | 1.16 | 0.72 | 1.61 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 233.12 | 233.12 | 1.00 | 207.68 | 207.68 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 53.73 | 53.73 | 1.00 | 56.40 | 56.40 | 1.00 | sdrk fallback |
| `bit_descramble_block` | 0.64 | 6.36 | 0.10 | 0.87 | 7.10 | 0.12 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_descramble_in_place_block` | 2.02 | 6.36 | 0.32 | 2.50 | 7.10 | 0.35 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_block` | 1.27 | 5.34 | 0.24 | 1.60 | 6.57 | 0.24 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_in_place_block` | 1.51 | 5.34 | 0.28 | 1.90 | 6.57 | 0.29 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `channel_bank_channelize` | 46.76 | 46.76 | 1.00 | 53.22 | 53.22 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 228.06 | 228.06 | 1.00 | 214.45 | 214.45 | 1.00 | sdrk fallback |
| `channel_select_block` | 0.60 | 0.60 | 1.00 | 0.77 | 0.77 | 1.00 | sdrk fallback |
| `channelize` | 25.93 | 171.64 | 0.15 | 27.90 | 176.13 | 0.16 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 27.50 | 27.50 | 1.00 | 28.52 | 28.52 | 1.00 | sdrk fallback |
| `channelize_fallback` | 55.45 | 166.47 | 0.33 | 59.82 | 182.49 | 0.33 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `correlate_block` | 21.75 | 21.75 | 1.00 | 20.30 | 20.30 | 1.00 | sdrk fallback |
| `costas` | 75.14 | 75.14 | 1.00 | 63.16 | 63.16 | 1.00 | sdrk fallback |
| `costas_fallback` | 75.14 | 86.26 | 0.87 | 63.16 | 80.13 | 0.79 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `costas_qam1024` | 155.27 | 155.27 | 1.00 | 139.40 | 139.40 | 1.00 | sdrk fallback |
| `costas_qam1024_fallback` | 155.27 | 287.31 | 0.54 | 139.40 | 247.17 | 0.56 | GNU Radio constellation_receiver_cb with constellation_rect 1024-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam16` | 156.35 | 156.35 | 1.00 | 140.33 | 140.33 | 1.00 | sdrk fallback |
| `costas_qam16_fallback` | 156.35 | 205.09 | 0.76 | 140.33 | 181.00 | 0.78 | GNU Radio constellation_receiver_cb with constellation_rect 16-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam256` | 155.62 | 155.62 | 1.00 | 139.70 | 139.70 | 1.00 | sdrk fallback |
| `costas_qam256_fallback` | 155.62 | 210.25 | 0.74 | 139.70 | 185.16 | 0.75 | GNU Radio constellation_receiver_cb with constellation_rect 256-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam64` | 156.24 | 156.24 | 1.00 | 140.23 | 140.23 | 1.00 | sdrk fallback |
| `costas_qam64_fallback` | 156.24 | 206.06 | 0.76 | 140.23 | 181.48 | 0.77 | GNU Radio constellation_receiver_cb with constellation_rect 64-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 13.55 | 15.15 | 0.89 | 11.55 | 13.14 | 0.88 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 8.06 | 8.15 | 0.99 | 7.06 | 7.14 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 13.05 | 15.15 | 0.86 | 11.06 | 13.14 | 0.84 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 8.06 | 8.15 | 0.99 | 7.06 | 7.15 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 138.46 | 355.18 | 0.39 | 145.11 | 390.47 | 0.37 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_fallback` | 138.46 | 168.48 | 0.82 | 145.11 | 201.30 | 0.72 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam1024` | 160.64 | 160.64 | 1.00 | 164.28 | 164.28 | 1.00 | sdrk fallback |
| `equalize_qam1024_fallback` | 160.64 | 191.53 | 0.84 | 164.28 | 213.78 | 0.77 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 1024-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam16` | 159.59 | 357.76 | 0.45 | 158.25 | 396.73 | 0.40 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam16_fallback` | 159.59 | 191.53 | 0.83 | 158.25 | 213.78 | 0.74 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam256` | 160.64 | 160.64 | 1.00 | 164.28 | 164.28 | 1.00 | sdrk fallback |
| `equalize_qam256_fallback` | 160.64 | 191.53 | 0.84 | 164.28 | 213.78 | 0.77 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 256-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam64` | 159.59 | 357.76 | 0.45 | 158.25 | 396.73 | 0.40 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam64_fallback` | 159.59 | 191.53 | 0.83 | 158.25 | 213.78 | 0.74 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 6.20 | 6.20 | 1.00 | 8.43 | 8.43 | 1.00 | sdrk fallback |
| `frame_decode_block` | 6.03 | 6.03 | 1.00 | 7.98 | 7.98 | 1.00 | sdrk fallback |
| `frame_decode_four_block` | 34.77 | 34.77 | 1.00 | 33.53 | 33.53 | 1.00 | sdrk fallback |
| `frame_encode_block` | 1.87 | 1.87 | 1.00 | 1.82 | 1.82 | 1.00 | sdrk fallback |
| `gauss_noise_c32` | 15.60 | 281.32 | 0.06 | 13.81 | 246.97 | 0.06 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fallback` | 24.74 | 281.32 | 0.09 | 28.14 | 246.97 | 0.11 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fastnoise` | 15.60 | 4.04 | 3.86 | 13.81 | 4.03 | 3.42 | GNU Radio fastnoise_source_c, amplitude sigma * sqrt(2) (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32` | 7.81 | 168.33 | 0.05 | 6.91 | 146.51 | 0.05 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fallback` | 12.37 | 168.33 | 0.07 | 14.07 | 146.51 | 0.10 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fastnoise` | 7.81 | 4.04 | 1.93 | 6.91 | 4.05 | 1.71 | GNU Radio fastnoise_source_f, amplitude sigma (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32_ziggurat` | 7.81 | 14.35 | 0.54 | 6.91 | 15.45 | 0.45 | rand_distr 0.5.1 StandardNormal as f32 times sigma over rand 0.9.5 SmallRng (Ziggurat in f64 with rejection: a variable number of draws per value, no random access, no bit-exact sequence across platforms; reference does different work) |
| `gauss_noise_q15` | 9.58 | 14.71 | 0.65 | 10.30 | 17.35 | 0.59 | sdrk fallback |
| `gauss_noise_q15_fallback` | 14.71 | 14.71 | 1.00 | 17.35 | 17.35 | 1.00 | sdrk fallback |
| `halfband_decimate` | 6.83 | 59.03 | 0.12 | 7.85 | 50.49 | 0.16 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its SSE dot product (-O3 -ffp-contract=off, sse2) |
| `halfband_decimate_cq15` | 16.67 | 39.56 | 0.42 | 16.20 | 46.15 | 0.35 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 39.56 | 41.57 | 0.95 | 46.15 | 50.10 | 0.92 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `halfband_decimate_fallback` | 15.16 | 82.14 | 0.18 | 17.48 | 69.57 | 0.25 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `halfband_decimate_real` | 3.66 | 68.61 | 0.05 | 3.99 | 55.10 | 0.07 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `halfband_decimate_real_fallback` | 12.87 | 68.61 | 0.19 | 13.53 | 55.10 | 0.25 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `halfband_decimate_real_q15` | 6.99 | 22.40 | 0.31 | 6.51 | 27.26 | 0.24 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 22.40 | 20.77 | 1.08 | 27.26 | 25.03 | 1.09 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `halfband_ref_fir_decimate` | 13.12 | 29.62 | 0.44 | 12.55 | 29.80 | 0.42 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 60.56 | 60.56 | 1.00 | 86.06 | 86.06 | 1.00 | sdrk fallback |
| `halfband_ref_shifted` | 9.54 | 17.92 | 0.53 | 11.35 | 21.03 | 0.54 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 19.59 | 42.48 | 0.46 | 20.37 | 50.32 | 0.40 | sdrk fallback |
| `iq_offset_block` | 2.27 | 2.27 | 1.00 | 2.89 | 2.89 | 1.00 | sdrk fallback |
| `morse_decode_block` | 16.81 | 16.81 | 1.00 | 20.51 | 20.51 | 1.00 | sdrk fallback |
| `morse_encode_block` | 6.60 | 6.60 | 1.00 | 7.58 | 7.58 | 1.00 | sdrk fallback |
| `noise_blank` | 15.92 | 15.92 | 1.00 | 11.94 | 11.94 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 9.59 | 9.59 | 1.00 | 10.59 | 10.59 | 1.00 | sdrk fallback |
| `noise_block` | 15.60 | 15.60 | 1.00 | 13.81 | 13.81 | 1.00 | sdrk fallback |
| `preemph_limit` | 16.10 | 16.10 | 1.00 | 11.93 | 11.93 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 51.64 | 51.64 | 1.00 | 43.84 | 43.84 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 21.13 | 21.13 | 1.00 | 22.70 | 22.70 | 1.00 | sdrk fallback |
| `psk_demod_block` | 75.73 | 447.26 | 0.17 | 66.83 | 367.86 | 0.18 | GNU Radio fir_filter_ccf + symbol_sync_cc + costas_loop_cc + constellation_decoder_cb (the sum of the blocks of the receiver chain, generic compiler flags; no acquisition and no lock watches) |
| `psk_mod_block` | 8.98 | 198.05 | 0.05 | 11.34 | 195.97 | 0.06 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `psk_mod_block_liquid` | 8.98 | 77.95 | 0.12 | 11.34 | 66.01 | 0.17 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `pulse_match_block` | 43.14 | 94.30 | 0.46 | 38.95 | 96.26 | 0.40 | GNU Radio fir_filter_ccf (fir_filter_ccf_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_match_block_liquid` | 43.14 | 227.97 | 0.19 | 38.95 | 186.24 | 0.21 | liquid-dsp 1.8.3 firfilt_crcf (firfilt.proto.c), the SIMD dot product of the level (the portable loop at simd128) |
| `pulse_shape_block` | 8.18 | 48.24 | 0.17 | 10.30 | 53.98 | 0.19 | GNU Radio interp_fir_filter_ccf (interp_fir_filter_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_shape_block_liquid` | 8.18 | 75.35 | 0.11 | 10.30 | 63.54 | 0.16 | liquid-dsp 1.8.3 firinterp_crcf (firinterp.proto.c, firpfb.proto.c), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_acquire_block` | 52.13 | 52.13 | 1.00 | 44.08 | 44.08 | 1.00 | sdrk fallback |
| `qam_demod_block` | 108.18 | 485.33 | 0.22 | 98.64 | 406.59 | 0.24 | GNU Radio fir_filter_ccf + symbol_sync_cc + linear_equalizer + constellation_receiver_cb (the sum of the blocks of the receiver chain, generic compiler flags and VOLK generic dot products; no acquisition, no blind equalizer stage and no lock watches) |
| `qam_demod_qam256_block` | 108.63 | 108.63 | 1.00 | 99.73 | 99.73 | 1.00 | sdrk fallback |
| `qam_demod_search_block` | 115.48 | 115.48 | 1.00 | 104.89 | 104.89 | 1.00 | sdrk fallback |
| `qam_frame_decode_block` | 126.16 | 126.16 | 1.00 | 115.95 | 115.95 | 1.00 | sdrk fallback |
| `qam_frame_encode_block` | 10.14 | 10.14 | 1.00 | 12.17 | 12.17 | 1.00 | sdrk fallback |
| `qam_mod_block` | 9.11 | 198.63 | 0.05 | 11.19 | 196.54 | 0.06 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_block_liquid` | 9.11 | 78.46 | 0.12 | 11.19 | 66.52 | 0.17 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_mod_qam256_block` | 10.06 | 199.78 | 0.05 | 12.33 | 197.69 | 0.06 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_qam256_block_liquid` | 10.06 | 79.47 | 0.13 | 12.33 | 67.53 | 0.18 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_soft_bits_qam1024` | 7.89 | 7.88 | 1.00 | 9.05 | 9.05 | 1.00 | sdrk fallback |
| `qam_soft_bits_qam16` | 1.35 | 1864.75 | 0.00 | 1.49 | 1588.53 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam16_liquid` | 1.35 | 224.89 | 0.01 | 1.49 | 227.63 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam16_lut` | 1.35 | 78.15 | 0.02 | 1.49 | 66.61 | 0.02 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qam256` | 6.02 | 6.56 | 0.92 | 6.02 | 7.06 | 0.85 | sdrk fallback |
| `qam_soft_bits_qam64` | 3.03 | 6763.49 | 0.00 | 3.34 | 5733.38 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam64_liquid` | 3.03 | 274.50 | 0.01 | 3.34 | 273.82 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam64_lut` | 3.03 | 78.15 | 0.04 | 3.34 | 66.61 | 0.05 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qpsk` | 0.52 | 0.52 | 0.99 | 0.52 | 0.59 | 0.88 | sdrk fallback |
| `rds_decode_block` | 9.18 | 9.18 | 1.00 | 11.91 | 11.91 | 1.00 | sdrk fallback |
| `rds_demod_block` | 41.90 | 41.90 | 1.00 | 39.11 | 39.11 | 1.00 | sdrk fallback |
| `rds_shape` | 83.94 | 83.94 | 1.00 | 75.83 | 75.83 | 1.00 | sdrk fallback |
| `real_to_complex` | 5.32 | 70.40 | 0.08 | 5.52 | 57.25 | 0.10 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `real_to_complex_airspy` | 5.32 | 11.28 | 0.47 | 5.52 | 9.37 | 0.59 | libairspy 1.0.10 iqconverter_float, conversion without remove_dc (USE_SSE2) (-O3 -ffp-contract=off, sse2) |
| `real_to_complex_cq15` | 8.98 | 15.69 | 0.57 | 8.97 | 17.83 | 0.50 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, sse2) |
| `real_to_complex_cq15_fallback` | 29.10 | 15.69 | 1.85 | 34.91 | 17.83 | 1.96 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, sse2) |
| `real_to_complex_fallback` | 17.76 | 70.40 | 0.25 | 20.65 | 57.25 | 0.36 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `snr_estimate_block` | 7.43 | 60.76 | 0.12 | 6.66 | 56.65 | 0.12 | GNU Radio mpsk_snr_est_cc(SNR_EST_M2M4, 4096, 0.001): the moments \|x\|^2 and \|x\|^4 averaged in double, six cabsf per symbol (reference does different work: no decisions, a PSK kurtosis; generic compiler flags) |
| `snr_estimate_block_svr` | 7.43 | 88.58 | 0.08 | 6.66 | 80.92 | 0.08 | GNU Radio mpsk_snr_est_cc(SNR_EST_SVR, 4096, 0.001): \|x\|^4 and \|x_n\|^2 \|x_(n-1)\|^2 averaged in double, two cabsf per symbol (reference does different work: no decisions, a PSK constellation; generic compiler flags) |
| `stereo_encode` | 10.15 | 10.15 | 1.00 | 10.88 | 10.88 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 18.68 | 18.68 | 1.00 | 17.96 | 17.96 | 1.00 | sdrk fallback |
| `stereo_matrix` | 1.02 | 2.10 | 0.49 | 0.99 | 2.48 | 0.40 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.51 | 1.02 | 0.50 | 0.53 | 1.14 | 0.47 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 1.02 | 5.50 | 0.19 | 1.14 | 5.68 | 0.20 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `stereo_matrix_fallback` | 1.55 | 1.07 | 1.45 | 1.67 | 1.38 | 1.21 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `stereo_pll_fallback` | 88.81 | 71.63 | 1.24 | 77.78 | 63.55 | 1.22 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, sse2) |
| `stereo_pll_q15_fallback` | 38.04 | 55.54 | 0.68 | 38.06 | 58.48 | 0.65 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `symbol_demap_qam16_block` | 1.35 | 1864.75 | 0.00 | 1.49 | 1588.53 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam16_block_liquid` | 1.35 | 224.88 | 0.01 | 1.49 | 227.62 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam16_block_lut` | 1.35 | 66.25 | 0.02 | 1.49 | 52.13 | 0.03 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_demap_qam256_block` | 6.02 | 6.02 | 1.00 | 6.03 | 6.03 | 1.00 | sdrk fallback |
| `symbol_demap_qam64_block` | 3.03 | 6763.49 | 0.00 | 3.34 | 5733.38 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam64_block_liquid` | 3.03 | 274.49 | 0.01 | 3.34 | 273.82 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam64_block_lut` | 3.03 | 78.15 | 0.04 | 3.34 | 66.60 | 0.05 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_map_block` | 7.10 | 13.20 | 0.54 | 6.61 | 15.02 | 0.44 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_liquid` | 7.10 | 22.45 | 0.32 | 6.61 | 23.09 | 0.29 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qam64` | 10.11 | 17.84 | 0.57 | 10.12 | 19.62 | 0.52 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qam64_liquid` | 10.11 | 26.53 | 0.38 | 10.12 | 27.16 | 0.37 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qpsk` | 6.01 | 8.56 | 0.70 | 7.74 | 10.43 | 0.74 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qpsk_liquid` | 6.01 | 18.37 | 0.33 | 7.74 | 19.02 | 0.41 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_sync` | 38.58 | 38.58 | 1.00 | 30.05 | 30.05 | 1.00 | sdrk fallback |
| `symbol_sync_fallback` | 38.58 | 57.52 | 0.67 | 30.05 | 57.78 | 0.52 | GNU Radio symbol_sync_cc (symbol_sync_cc_impl.cc of gr-digital, TED_GARDNER, IR_MMSE_8TAP, generic compiler flags; reference does different work: an 8-tap MMSE interpolator against our 4-point Lagrange polynomial) |
| `symbol_sync_real` | 19.03 | 19.03 | 1.00 | 14.53 | 14.53 | 1.00 | sdrk fallback |
| `xcorr_fft_n4096` | 25.47 | 68.23 | 0.37 | 23.94 | 66.22 | 0.36 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n4096_fallback` | 104.39 | 363.78 | 0.29 | 101.22 | 397.16 | 0.25 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |
| `xcorr_fft_n512` | 21.75 | 45.86 | 0.47 | 20.30 | 50.34 | 0.40 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n512_fallback` | 84.54 | 252.42 | 0.33 | 83.81 | 278.96 | 0.30 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |

#### Simulated cycles at `sse4.2` (`x86_64`)

Reference model `skylake`, counter-model `znver2`, mode `whole-loop`; execution counts: Intel SDE -dcfg, block and branch counts (Intel SDE, CPUID of an x86-64-v2 (SSE4.2) CPU).

| Entry | ours skylake (cycles/sample) | comparison skylake (cycles/sample) | ratio skylake (ours/comparison) | ours znver2 (cycles/sample) | comparison znver2 (cycles/sample) | ratio znver2 (ours/comparison) | comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `convert_cq15_cs8` | 0.26 | 0.39 | 0.68 | 0.26 | 0.39 | 0.68 | volk_8i_convert_16i |
| `convert_cq15_cs8_fallback` | 0.30 | 0.39 | 0.79 | 0.30 | 0.36 | 0.84 | CMSIS-DSP 1.18.0 arm_q7_to_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_cq15_cu8` | 0.26 | 0.39 | 0.67 | 0.32 | 0.39 | 0.81 | volk_8i_convert_16i on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cq15_cu8_fallback` | 0.36 | 0.39 | 0.92 | 0.36 | 0.36 | 1.01 | CMSIS-DSP 1.18.0 arm_q7_to_q15 on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_cs16` | 0.53 | 0.78 | 0.68 | 0.54 | 0.96 | 0.56 | volk_16i_s32f_convert_32f |
| `convert_cs16_fallback` | 0.70 | 0.78 | 0.91 | 0.72 | 0.84 | 0.86 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_cs8` | 0.53 | 0.89 | 0.59 | 0.54 | 0.77 | 0.70 | volk_8i_s32f_convert_32f |
| `convert_cs8_fallback` | 0.70 | 1.14 | 0.62 | 0.72 | 0.83 | 0.87 | CMSIS-DSP 1.18.0 arm_q7_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_cu8` | 0.76 | 0.89 | 0.85 | 0.63 | 0.77 | 0.82 | volk_8i_s32f_convert_32f on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cu8_fallback` | 0.79 | 1.14 | 0.69 | 0.85 | 0.83 | 1.02 | CMSIS-DSP 1.18.0 arm_q7_to_float on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_out_cs16` | 1.10 | 1.03 | 1.06 | 1.03 | 1.03 | 1.00 | volk_32f_s32f_convert_16i |
| `convert_out_cs16_fallback` | 1.95 | 1.03 | 1.90 | 2.03 | 1.25 | 1.63 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_out_cs8` | 1.02 | 1.02 | 1.00 | 1.03 | 0.96 | 1.08 | volk_32f_s32f_convert_8i |
| `convert_out_cs8_fallback` | 1.91 | 1.18 | 1.61 | 1.88 | 1.39 | 1.35 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_out_cu8` | 1.57 | 1.33 | 1.18 | 1.52 | 1.39 | 1.09 | volk_32f_s32f_x2_convert_8u |
| `convert_out_cu8_fallback` | 4.70 | 1.18 | 3.98 | 4.41 | 1.39 | 3.17 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_out_real_rs16` | 0.56 | 0.52 | 1.07 | 0.53 | 0.52 | 1.02 | volk_32f_s32f_convert_16i |
| `convert_out_real_rs16_fallback` | 0.98 | 0.52 | 1.91 | 1.02 | 0.63 | 1.64 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_out_real_rs8` | 0.53 | 0.51 | 1.02 | 0.53 | 0.48 | 1.09 | volk_32f_s32f_convert_8i |
| `convert_out_real_rs8_fallback` | 0.96 | 0.59 | 1.62 | 0.94 | 0.70 | 1.35 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_out_real_ru8` | 0.80 | 0.67 | 1.19 | 0.78 | 0.70 | 1.11 | volk_32f_s32f_x2_convert_8u |
| `convert_out_real_ru8_fallback` | 2.36 | 0.59 | 3.98 | 2.21 | 0.70 | 3.17 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `convert_real_rs16` | 0.27 | 0.39 | 0.69 | 0.28 | 0.49 | 0.57 | volk_16i_s32f_convert_32f |
| `convert_real_rs16_fallback` | 0.36 | 0.39 | 0.93 | 0.37 | 0.42 | 0.87 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `rotate` | 2.08 | 9.16 | 0.23 | 1.98 | 11.13 | 0.18 | volk_32fc_s32fc_x2_rotator2_32fc |
| `rotate_cq15` | 2.45 | 9.55 | 0.26 | 2.66 | 10.37 | 0.26 | sdrk fallback |
| `rotate_cq15_fallback` | 9.55 | 34.89 | 0.27 | 10.37 | 33.66 | 0.31 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `rotate_fallback` | 7.15 | 15.45 | 0.46 | 8.62 | 16.79 | 0.51 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `scramble_block` | 6.21 | 6.21 | 1.00 | 5.99 | 5.99 | 1.00 | sdrk fallback |
| `scramble_c32_b1` | 0.89 | 1.99 | 0.45 | 0.83 | 2.39 | 0.35 | +-1.0 sequence from the index bits into a buffer of 256 + volk_32fc_32f_multiply_32fc (VOLK 3.3.0 dispatcher; the form of srsRAN srsran_scrambling_c) |
| `scramble_c32_b1_fallback` | 1.53 | 2.07 | 0.74 | 1.42 | 2.59 | 0.55 | +-1.0 sequence from the index bits into a buffer of 256 + VOLK 3.3.0 volk_32fc_32f_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2` | 2.35 | 3.63 | 0.65 | 1.96 | 4.29 | 0.46 | phasor lookup into a buffer of 256 + volk_32fc_x2_multiply_32fc (VOLK 3.3.0 dispatcher) |
| `scramble_c32_b2_dvbs2` | 2.35 | 3.55 | 0.66 | 1.96 | 4.44 | 0.44 | GNU Radio dvbs2_physical_cc PL scrambling, the per-symbol switch on R_n (benches/refs/gr-dvbs2-plscramble.c, main at ec1bf15; reference does different work; -O3) |
| `scramble_c32_b2_fallback` | 4.04 | 8.05 | 0.50 | 3.55 | 9.73 | 0.36 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8` | 3.40 | 3.63 | 0.94 | 3.74 | 4.29 | 0.87 | phasor lookup into a buffer of 256 + volk_32fc_x2_multiply_32fc (VOLK 3.3.0 dispatcher) |
| `scramble_c32_b8_fallback` | 5.88 | 8.05 | 0.73 | 6.87 | 9.73 | 0.71 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_cq15_b2` | 1.78 | 5.77 | 0.31 | 1.34 | 6.26 | 0.21 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `scramble_cq15_b2_fallback` | 2.79 | 5.77 | 0.48 | 2.28 | 6.26 | 0.36 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `scramble_cq15_b8` | 3.61 | 5.77 | 0.63 | 3.72 | 6.26 | 0.59 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `scramble_cq15_b8_fallback` | 6.18 | 5.77 | 1.07 | 6.87 | 6.26 | 1.10 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir` | 33.63 | 86.68 | 0.39 | 29.99 | 88.48 | 0.34 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 47.76 | 142.31 | 0.34 | 49.10 | 188.30 | 0.26 | sdrk fallback |
| `fir_cq15_fallback` | 142.31 | 177.24 | 0.80 | 188.30 | 220.40 | 0.85 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_decimate` | 6.86 | 14.45 | 0.47 | 6.06 | 14.17 | 0.43 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 8.87 | 30.40 | 0.29 | 9.34 | 45.20 | 0.21 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 30.40 | 27.95 | 1.09 | 45.20 | 22.79 | 1.98 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_decimate_fallback` | 15.09 | 38.81 | 0.39 | 15.82 | 23.69 | 0.67 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_fallback` | 42.70 | 132.97 | 0.32 | 51.73 | 111.96 | 0.46 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate` | 36.22 | 95.13 | 0.38 | 34.15 | 97.71 | 0.35 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 51.44 | 137.49 | 0.37 | 53.59 | 207.95 | 0.26 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 137.49 | 137.61 | 1.00 | 207.95 | 164.43 | 1.26 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate_fallback` | 69.95 | 141.75 | 0.49 | 71.95 | 114.81 | 0.63 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate_real` | 19.44 | 72.80 | 0.27 | 18.20 | 62.85 | 0.29 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 42.42 | 70.88 | 0.60 | 44.66 | 57.41 | 0.78 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate_real_q15` | 26.58 | 74.97 | 0.35 | 32.31 | 112.89 | 0.29 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 74.97 | 68.81 | 1.09 | 112.89 | 82.21 | 1.37 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate_short` | 7.20 | 49.96 | 0.14 | 8.86 | 61.82 | 0.14 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_short_fallback` | 12.79 | 24.83 | 0.52 | 17.63 | 22.72 | 0.78 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 8 (80 taps), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `resample` | 38.72 | 96.79 | 0.40 | 35.30 | 98.80 | 0.36 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 49.95 | 168.72 | 0.30 | 51.77 | 240.51 | 0.22 | sdrk fallback |
| `resample_cq15_fallback` | 168.72 | 136.68 | 1.23 | 240.51 | 163.55 | 1.47 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `resample_fallback` | 84.56 | 141.22 | 0.60 | 86.87 | 113.81 | 0.76 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `resample_real` | 22.29 | 73.99 | 0.30 | 21.79 | 63.40 | 0.34 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 84.49 | 70.61 | 1.20 | 75.06 | 56.90 | 1.32 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `resample_real_q15` | 25.84 | 103.67 | 0.25 | 31.40 | 144.16 | 0.22 | sdrk fallback |
| `resample_real_q15_fallback` | 103.67 | 68.34 | 1.52 | 144.16 | 81.78 | 1.76 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mag` | 0.76 | 0.79 | 0.96 | 1.77 | 1.80 | 0.98 | volk_32fc_magnitude_32f |
| `mag_fallback` | 0.79 | 0.85 | 0.93 | 1.79 | 1.86 | 0.96 | CMSIS-DSP 1.18.0 arm_cmplx_mag_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mag_q15` | 2.27 | 6.47 | 0.35 | 3.44 | 6.78 | 0.51 | volk_16ic_magnitude_16i |
| `mag_q15_fallback` | 3.34 | 39.58 | 0.08 | 4.30 | 45.02 | 0.10 | CMSIS-DSP 1.18.0 arm_cmplx_mag_q15 (Q2.14, arm_sqrt_q31), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mag_sq` | 0.51 | 0.61 | 0.83 | 0.39 | 0.65 | 0.60 | volk_32fc_magnitude_squared_32f |
| `mag_sq_fallback` | 0.52 | 0.65 | 0.80 | 0.67 | 0.84 | 0.80 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mag_sq_q15` | 0.33 | 0.41 | 0.80 | 0.33 | 0.48 | 0.69 | sdrk fallback |
| `mag_sq_q15_fallback` | 0.41 | 3.71 | 0.11 | 0.48 | 2.77 | 0.17 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_q15 (Q3.13, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `quad_demod` | 8.50 | 107.41 | 0.08 | 10.10 | 93.93 | 0.11 | volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f |
| `quad_demod_cq15` | 12.77 | 94.87 | 0.13 | 12.48 | 97.68 | 0.13 | sdrk fallback |
| `quad_demod_cq15_fallback` | 94.87 | 130.34 | 0.73 | 97.68 | 119.37 | 0.82 | CMSIS-DSP arm_cmplx_mult_cmplx_q15 + arm_atan2_q15 + arm_scale_q15 (generic C), conjugate in a plain loop |
| `quad_demod_fallback` | 51.20 | 68.55 | 0.75 | 49.82 | 63.79 | 0.78 | GNU Radio quadrature_demod_cf (gr-analog, fast_atan2f, generic compiler flags) |
| `fm_mod` | 6.98 | 47.68 | 0.15 | 6.61 | 60.01 | 0.11 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `fm_mod_cq15` | 6.57 | 12.31 | 0.53 | 4.95 | 11.93 | 0.41 | sdrk fallback |
| `fm_mod_cq15_fallback` | 12.31 | 25.62 | 0.48 | 11.93 | 23.96 | 0.50 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample of an integer phase accumulator, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fm_mod_fallback` | 8.94 | 47.68 | 0.19 | 10.89 | 60.01 | 0.18 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `squelch` | 12.13 | 27.74 | 0.44 | 9.13 | 26.37 | 0.35 | GNU Radio pwr_squelch_cc (squelch_base_cc_impl.cc of gr-analog, generic compiler flags) |
| `squelch_cq15` | 7.60 | 7.60 | 1.00 | 8.11 | 8.11 | 1.00 | sdrk fallback |
| `window` | 0.54 | 0.65 | 0.83 | 0.68 | 0.78 | 0.87 | volk_32fc_32f_multiply_32fc |
| `window_cq15` | 0.48 | 1.46 | 0.33 | 0.53 | 1.49 | 0.36 | sdrk fallback |
| `window_cq15_fallback` | 1.46 | 1.53 | 0.95 | 1.49 | 1.53 | 0.97 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `window_fallback` | 0.64 | 0.65 | 0.98 | 0.79 | 0.84 | 0.95 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `power_db` | 6.83 | 4.62 | 1.48 | 7.79 | 4.98 | 1.56 | volk_32f_log2_32f + volk_32f_s32f_multiply_32f + volk_32f_s32f_add_32f |
| `power_db_fallback` | 8.96 | 55.94 | 0.16 | 10.71 | 49.41 | 0.22 | CMSIS-DSP 1.18.0 arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `power_db_q15` | 5.30 | 9.62 | 0.55 | 5.08 | 11.84 | 0.43 | sdrk fallback |
| `power_db_q15_fallback` | 9.62 | 56.34 | 0.17 | 11.84 | 49.84 | 0.24 | CMSIS-DSP 1.18.0 arm_q15_to_float + arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `agc` | 13.76 | 42.10 | 0.33 | 11.54 | 38.11 | 0.30 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `agc_cq15` | 85.25 | 85.25 | 1.00 | 82.80 | 82.80 | 1.00 | sdrk fallback |
| `agc_fallback` | 18.78 | 42.10 | 0.45 | 17.66 | 38.11 | 0.46 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `dc_block` | 5.04 | 10.03 | 0.50 | 4.28 | 7.04 | 0.61 | sdrk fallback |
| `dc_block_cq15` | 8.06 | 8.06 | 1.00 | 10.05 | 10.05 | 1.00 | sdrk fallback |
| `dc_block_fallback` | 10.03 | 56.29 | 0.18 | 7.04 | 51.26 | 0.14 | liquid-dsp 1.8.3 iirfilt_crcf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real` | 4.09 | 10.03 | 0.41 | 3.65 | 7.04 | 0.52 | sdrk fallback |
| `dc_block_real_fallback` | 10.03 | 46.19 | 0.22 | 7.04 | 43.86 | 0.16 | liquid-dsp 1.8.3 iirfilt_rrrf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real_q15` | 6.06 | 6.06 | 1.00 | 7.55 | 7.55 | 1.00 | sdrk fallback |
| `deemphasis` | 3.77 | 15.09 | 0.25 | 3.64 | 11.09 | 0.33 | sdrk fallback |
| `deemphasis_fallback` | 15.09 | 13.15 | 1.15 | 11.09 | 17.11 | 0.65 | GNU Radio fm_deemph: iir_filter_ffd with the bilinear taps of fm_emph.py (double taps and accumulator, generic compiler flags) |
| `deemphasis_q15` | 7.03 | 7.03 | 1.00 | 8.53 | 8.53 | 1.00 | sdrk fallback |
| `biquad` | 12.68 | 27.87 | 0.46 | 9.42 | 28.09 | 0.34 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (sse42, -ffp-contract=off), two calls on planar re and im (deinterleaving not counted) |
| `biquad_cq15` | 16.13 | 24.43 | 0.66 | 15.63 | 31.16 | 0.50 | sdrk fallback |
| `biquad_cq15_fallback` | 24.43 | 60.21 | 0.41 | 31.16 | 52.19 | 0.60 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `biquad_fallback` | 25.00 | 32.54 | 0.77 | 18.60 | 24.54 | 0.76 | CMSIS-DSP 1.18.0 arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `biquad_real` | 12.12 | 13.94 | 0.87 | 9.05 | 14.05 | 0.64 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (sse42, -ffp-contract=off) |
| `biquad_real_fallback` | 14.52 | 24.04 | 0.60 | 11.31 | 18.03 | 0.63 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df2T_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `biquad_real_q15` | 12.08 | 14.13 | 0.85 | 11.32 | 15.13 | 0.75 | sdrk fallback |
| `biquad_real_q15_fallback` | 14.13 | 30.11 | 0.47 | 15.13 | 26.10 | 0.58 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `dot` | 0.60 | 0.61 | 0.98 | 0.69 | 0.65 | 1.05 | volk_32fc_32f_dot_prod_32fc |
| `dot_complex` | 1.31 | 2.06 | 0.64 | 1.33 | 1.56 | 0.86 | volk_32fc_x2_dot_prod_32fc |
| `dot_complex_fallback` | 2.65 | 8.28 | 0.32 | 2.40 | 6.28 | 0.38 | CMSIS-DSP 1.18.0 arm_cmplx_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `dot_cq15` | 0.82 | 2.03 | 0.40 | 0.75 | 1.72 | 0.44 | sdrk fallback |
| `dot_cq15_fallback` | 2.03 | 2.04 | 1.00 | 1.72 | 1.49 | 1.15 | CMSIS-DSP 1.18.0 arm_dot_prod_q15 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `dot_fallback` | 4.03 | 8.07 | 0.50 | 3.03 | 6.06 | 0.50 | CMSIS-DSP 1.18.0 arm_dot_prod_f32 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `dot_real` | 0.27 | 0.52 | 0.51 | 0.29 | 0.34 | 0.86 | volk_32f_x2_dot_prod_32f |
| `dot_real_fallback` | 4.04 | 4.04 | 1.00 | 3.03 | 3.03 | 1.00 | CMSIS-DSP 1.18.0 arm_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `dot_real_q15` | 0.35 | 1.03 | 0.34 | 0.35 | 0.72 | 0.49 | sdrk fallback |
| `dot_real_q15_fallback` | 1.03 | 1.02 | 1.01 | 0.72 | 0.75 | 0.95 | CMSIS-DSP 1.18.0 arm_dot_prod_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `scale` | 0.51 | 0.64 | 0.80 | 0.51 | 0.79 | 0.64 | volk_32f_s32f_multiply_32f |
| `scale_cq15` | 0.26 | 1.10 | 0.24 | 0.26 | 1.02 | 0.26 | sdrk fallback |
| `scale_cq15_fallback` | 1.10 | 1.78 | 0.62 | 1.02 | 1.65 | 0.62 | CMSIS-DSP 1.18.0 arm_scale_q15 (shift 0, truncating) on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `scale_fallback` | 0.52 | 0.63 | 0.82 | 0.58 | 0.79 | 0.74 | CMSIS-DSP 1.18.0 arm_scale_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `add` | 0.51 | 0.70 | 0.72 | 0.56 | 0.79 | 0.71 | volk_32fc_x2_add_32fc |
| `add_cq15` | 0.26 | 0.35 | 0.72 | 0.28 | 0.40 | 0.71 | volk_16i_x2_add_saturated_16i on the 2 n values |
| `add_cq15_fallback` | 0.35 | 2.28 | 0.15 | 0.46 | 1.46 | 0.31 | CMSIS-DSP 1.18.0 arm_add_q15 on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `add_fallback` | 0.70 | 0.70 | 1.00 | 0.91 | 0.91 | 1.00 | CMSIS-DSP 1.18.0 arm_add_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `add_real` | 0.26 | 0.36 | 0.71 | 0.28 | 0.40 | 0.70 | volk_32f_x2_add_32f |
| `add_real_fallback` | 0.29 | 0.36 | 0.82 | 0.36 | 0.46 | 0.77 | CMSIS-DSP 1.18.0 arm_add_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `add_real_q15` | 0.13 | 0.18 | 0.71 | 0.14 | 0.20 | 0.70 | volk_16i_x2_add_saturated_16i |
| `add_real_q15_fallback` | 0.15 | 1.14 | 0.13 | 0.18 | 0.73 | 0.25 | CMSIS-DSP 1.18.0 arm_add_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mul` | 1.52 | 1.55 | 0.98 | 1.19 | 1.55 | 0.77 | volk_32fc_x2_multiply_32fc |
| `mul_cq15` | 1.43 | 3.04 | 0.47 | 1.48 | 2.89 | 0.51 | sdrk fallback |
| `mul_cq15_fallback` | 3.04 | 3.71 | 0.82 | 2.89 | 3.53 | 0.82 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_q15 (Q3.13, truncating, scaled by 4), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mul_fallback` | 1.52 | 1.53 | 0.99 | 1.23 | 1.34 | 0.92 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mul_real` | 0.26 | 0.36 | 0.71 | 0.28 | 0.40 | 0.70 | volk_32f_x2_multiply_32f |
| `mul_real_fallback` | 0.29 | 0.36 | 0.82 | 0.36 | 0.46 | 0.77 | CMSIS-DSP 1.18.0 arm_mult_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mul_real_q15` | 0.20 | 0.67 | 0.30 | 0.24 | 0.55 | 0.43 | sdrk fallback |
| `mul_real_q15_fallback` | 0.67 | 0.64 | 1.04 | 0.55 | 0.71 | 0.78 | CMSIS-DSP 1.18.0 arm_mult_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `interleave` | 0.51 | 0.52 | 0.98 | 0.51 | 0.64 | 0.79 | volk_32f_x2_interleave_32fc |
| `interleave_cq15` | 0.26 | 0.26 | 0.99 | 0.26 | 0.31 | 0.84 | sdrk fallback |
| `interleave_cq15_fallback` | 0.26 | 0.26 | 1.00 | 0.31 | 0.33 | 0.95 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic on int16 (VOLK has no 16-bit interleave), benches/refs/volk-generic.c (-O3 -ffp-contract=off, sse4.2) |
| `interleave_fallback` | 0.51 | 0.52 | 0.98 | 0.61 | 0.64 | 0.94 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, sse4.2) |
| `deinterleave` | 0.51 | 0.52 | 0.98 | 0.51 | 0.65 | 0.79 | volk_32fc_deinterleave_32f_x2 |
| `deinterleave_cq15` | 0.51 | 0.52 | 0.99 | 0.30 | 0.49 | 0.63 | volk_16ic_deinterleave_16i_x2 |
| `deinterleave_cq15_fallback` | 0.52 | 0.51 | 1.00 | 0.48 | 0.55 | 0.89 | VOLK 3.3.0 volk_16ic_deinterleave_16i_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, sse4.2) |
| `deinterleave_fallback` | 0.51 | 0.58 | 0.88 | 0.54 | 0.71 | 0.77 | VOLK 3.3.0 volk_32fc_deinterleave_32f_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, sse4.2) |
| `oscillate` | 2.29 | 4.90 | 0.47 | 2.57 | 6.35 | 0.40 | sdrk fallback |
| `oscillate_cq15` | 2.61 | 8.08 | 0.32 | 3.04 | 7.08 | 0.43 | sdrk fallback |
| `oscillate_cq15_fallback` | 8.08 | 21.81 | 0.37 | 7.08 | 19.33 | 0.37 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample (512-entry table, linear interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `oscillate_fallback` | 4.90 | 51.98 | 0.09 | 6.35 | 50.82 | 0.12 | CMSIS-DSP 1.18.0 arm_sin_cos_f32 of the angle in degrees per sample (512-entry sine table, cubic Hermite interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_1024` | 7.19 | 9.32 | 0.77 | 6.33 | 9.55 | 0.66 | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 31.38 | 25.79 | 1.22 | 29.98 | 28.60 | 1.05 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_4096` | 8.14 | 10.68 | 0.76 | 7.28 | 10.77 | 0.68 | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 37.53 | 30.53 | 1.23 | 35.98 | 32.90 | 1.09 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_64` | 4.58 | 8.57 | 0.53 | 4.03 | 8.08 | 0.50 | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 20.71 | 15.17 | 1.37 | 19.86 | 17.78 | 1.12 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_cq15_1024` | 6.27 | 52.83 | 0.12 | 7.28 | 63.00 | 0.12 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 5.40 | 52.83 | 0.10 | 6.16 | 63.00 | 0.10 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 45.87 | 44.80 | 1.02 | 46.95 | 56.34 | 0.83 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_cq15_1024_fallback` | 76.35 | 44.80 | 1.70 | 77.17 | 56.34 | 1.37 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_cq15_1024_full_scale` | 5.83 | 52.83 | 0.11 | 6.27 | 63.00 | 0.10 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 4.71 | 5.63 | 0.84 | 4.40 | 4.58 | 0.96 | FFTW (fftwf_plan_dft_r2c_1d, FFTW_ESTIMATE, forward, out of place) |
| `rfft_1024_fallback` | 16.63 | 13.39 | 1.24 | 16.16 | 14.58 | 1.11 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `am_mod` | 1.09 | 1.09 | 1.00 | 1.16 | 1.16 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 0.31 | 1.23 | 0.25 | 0.37 | 1.15 | 0.32 | sdrk fallback |
| `am_mod_cq15_fallback` | 1.23 | 3.17 | 0.39 | 1.15 | 3.60 | 0.32 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `am_mod_fallback` | 1.09 | 0.59 | 1.84 | 1.16 | 0.72 | 1.61 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 233.12 | 233.12 | 1.00 | 207.68 | 207.68 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 53.06 | 53.06 | 1.00 | 55.95 | 55.95 | 1.00 | sdrk fallback |
| `bit_descramble_block` | 0.64 | 6.36 | 0.10 | 0.87 | 7.10 | 0.12 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_descramble_in_place_block` | 2.03 | 6.36 | 0.32 | 2.50 | 7.10 | 0.35 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_block` | 1.27 | 5.34 | 0.24 | 1.60 | 6.57 | 0.24 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_in_place_block` | 1.51 | 5.34 | 0.28 | 1.90 | 6.57 | 0.29 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `channel_bank_channelize` | 45.78 | 45.78 | 1.00 | 53.10 | 53.10 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 227.75 | 227.75 | 1.00 | 214.03 | 214.03 | 1.00 | sdrk fallback |
| `channel_select_block` | 0.60 | 0.60 | 1.00 | 0.77 | 0.77 | 1.00 | sdrk fallback |
| `channelize` | 21.71 | 171.63 | 0.13 | 28.26 | 176.12 | 0.16 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 26.54 | 26.54 | 1.00 | 28.38 | 28.38 | 1.00 | sdrk fallback |
| `channelize_fallback` | 55.89 | 166.73 | 0.34 | 59.69 | 182.23 | 0.33 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `correlate_block` | 21.66 | 21.66 | 1.00 | 20.37 | 20.37 | 1.00 | sdrk fallback |
| `costas` | 73.14 | 73.14 | 1.00 | 62.15 | 62.15 | 1.00 | sdrk fallback |
| `costas_fallback` | 73.14 | 86.26 | 0.85 | 62.15 | 80.13 | 0.78 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `costas_qam1024` | 155.27 | 155.27 | 1.00 | 139.40 | 139.40 | 1.00 | sdrk fallback |
| `costas_qam1024_fallback` | 155.27 | 287.31 | 0.54 | 139.40 | 247.17 | 0.56 | GNU Radio constellation_receiver_cb with constellation_rect 1024-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam16` | 156.35 | 156.35 | 1.00 | 140.33 | 140.33 | 1.00 | sdrk fallback |
| `costas_qam16_fallback` | 156.35 | 205.09 | 0.76 | 140.33 | 181.00 | 0.78 | GNU Radio constellation_receiver_cb with constellation_rect 16-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam256` | 155.62 | 155.62 | 1.00 | 139.70 | 139.70 | 1.00 | sdrk fallback |
| `costas_qam256_fallback` | 155.62 | 210.25 | 0.74 | 139.70 | 185.16 | 0.75 | GNU Radio constellation_receiver_cb with constellation_rect 256-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam64` | 156.24 | 156.24 | 1.00 | 140.23 | 140.23 | 1.00 | sdrk fallback |
| `costas_qam64_fallback` | 156.24 | 206.06 | 0.76 | 140.23 | 181.48 | 0.77 | GNU Radio constellation_receiver_cb with constellation_rect 64-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 13.55 | 15.15 | 0.89 | 11.55 | 13.14 | 0.88 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 8.06 | 8.15 | 0.99 | 7.06 | 7.14 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 13.05 | 15.15 | 0.86 | 11.06 | 13.14 | 0.84 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 8.06 | 8.15 | 0.99 | 7.06 | 7.15 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 135.40 | 271.02 | 0.50 | 133.75 | 292.38 | 0.46 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_fallback` | 135.40 | 168.48 | 0.80 | 133.75 | 201.30 | 0.66 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam1024` | 159.18 | 159.18 | 1.00 | 158.86 | 158.86 | 1.00 | sdrk fallback |
| `equalize_qam1024_fallback` | 159.18 | 191.53 | 0.83 | 158.86 | 213.78 | 0.74 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 1024-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam16` | 160.09 | 273.34 | 0.59 | 152.75 | 297.82 | 0.51 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam16_fallback` | 160.09 | 191.53 | 0.84 | 152.75 | 213.78 | 0.71 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam256` | 159.18 | 159.18 | 1.00 | 158.86 | 158.86 | 1.00 | sdrk fallback |
| `equalize_qam256_fallback` | 159.18 | 191.53 | 0.83 | 158.86 | 213.78 | 0.74 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 256-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam64` | 160.09 | 273.34 | 0.59 | 152.75 | 297.82 | 0.51 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam64_fallback` | 160.09 | 191.53 | 0.84 | 152.75 | 213.78 | 0.71 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 6.27 | 6.27 | 1.00 | 8.52 | 8.52 | 1.00 | sdrk fallback |
| `frame_decode_block` | 6.06 | 6.06 | 1.00 | 7.98 | 7.98 | 1.00 | sdrk fallback |
| `frame_decode_four_block` | 34.82 | 34.82 | 1.00 | 33.27 | 33.27 | 1.00 | sdrk fallback |
| `frame_encode_block` | 1.87 | 1.87 | 1.00 | 1.82 | 1.82 | 1.00 | sdrk fallback |
| `gauss_noise_c32` | 14.94 | 281.32 | 0.05 | 14.45 | 246.97 | 0.06 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fallback` | 24.14 | 281.32 | 0.09 | 27.45 | 246.97 | 0.11 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fastnoise` | 14.94 | 4.04 | 3.70 | 14.45 | 4.03 | 3.58 | GNU Radio fastnoise_source_c, amplitude sigma * sqrt(2) (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32` | 7.48 | 168.33 | 0.04 | 7.24 | 146.51 | 0.05 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fallback` | 12.07 | 168.33 | 0.07 | 13.73 | 146.51 | 0.09 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fastnoise` | 7.48 | 4.04 | 1.85 | 7.24 | 4.05 | 1.79 | GNU Radio fastnoise_source_f, amplitude sigma (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32_ziggurat` | 7.48 | 15.50 | 0.48 | 7.24 | 19.58 | 0.37 | rand_distr 0.5.1 StandardNormal as f32 times sigma over rand 0.9.5 SmallRng (Ziggurat in f64 with rejection: a variable number of draws per value, no random access, no bit-exact sequence across platforms; reference does different work) |
| `gauss_noise_q15` | 9.85 | 14.71 | 0.67 | 10.12 | 17.35 | 0.58 | sdrk fallback |
| `gauss_noise_q15_fallback` | 14.71 | 14.71 | 1.00 | 17.35 | 17.35 | 1.00 | sdrk fallback |
| `halfband_decimate` | 6.78 | 59.03 | 0.11 | 7.78 | 50.49 | 0.15 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its SSE dot product (-O3 -ffp-contract=off, sse4.2) |
| `halfband_decimate_cq15` | 13.13 | 23.94 | 0.55 | 12.99 | 21.82 | 0.60 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 23.94 | 38.80 | 0.62 | 21.82 | 47.25 | 0.46 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `halfband_decimate_fallback` | 15.73 | 82.14 | 0.19 | 17.79 | 69.57 | 0.26 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, sse4.2) |
| `halfband_decimate_real` | 3.66 | 53.52 | 0.07 | 3.99 | 48.28 | 0.08 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its SSE (SSE3 horizontal add) dot product (-O3 -ffp-contract=off, sse4.2) |
| `halfband_decimate_real_fallback` | 12.87 | 68.61 | 0.19 | 13.53 | 55.10 | 0.25 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse4.2) |
| `halfband_decimate_real_q15` | 6.21 | 16.50 | 0.38 | 5.38 | 16.82 | 0.32 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 16.50 | 19.39 | 0.85 | 16.82 | 23.61 | 0.71 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `halfband_ref_fir_decimate` | 13.07 | 29.62 | 0.44 | 12.48 | 29.80 | 0.42 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 19.29 | 60.56 | 0.32 | 19.65 | 86.06 | 0.23 | sdrk fallback |
| `halfband_ref_shifted` | 9.52 | 18.49 | 0.51 | 11.31 | 21.34 | 0.53 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 16.01 | 26.85 | 0.60 | 17.14 | 25.98 | 0.66 | sdrk fallback |
| `iq_offset_block` | 3.55 | 3.55 | 1.00 | 3.56 | 3.56 | 1.00 | sdrk fallback |
| `morse_decode_block` | 16.62 | 16.62 | 1.00 | 20.29 | 20.29 | 1.00 | sdrk fallback |
| `morse_encode_block` | 6.60 | 6.60 | 1.00 | 7.58 | 7.58 | 1.00 | sdrk fallback |
| `noise_blank` | 15.92 | 15.92 | 1.00 | 11.93 | 11.93 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 9.59 | 9.59 | 1.00 | 10.59 | 10.59 | 1.00 | sdrk fallback |
| `noise_block` | 14.94 | 14.94 | 1.00 | 14.45 | 14.45 | 1.00 | sdrk fallback |
| `preemph_limit` | 16.10 | 16.10 | 1.00 | 11.44 | 11.44 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 51.64 | 51.64 | 1.00 | 43.84 | 43.84 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 20.85 | 20.85 | 1.00 | 21.65 | 21.65 | 1.00 | sdrk fallback |
| `psk_demod_block` | 75.60 | 447.26 | 0.17 | 66.68 | 367.86 | 0.18 | GNU Radio fir_filter_ccf + symbol_sync_cc + costas_loop_cc + constellation_decoder_cb (the sum of the blocks of the receiver chain, generic compiler flags; no acquisition and no lock watches) |
| `psk_mod_block` | 8.57 | 216.10 | 0.04 | 10.62 | 214.99 | 0.05 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `psk_mod_block_liquid` | 8.57 | 77.95 | 0.11 | 10.62 | 66.00 | 0.16 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `pulse_match_block` | 43.14 | 103.32 | 0.42 | 38.95 | 105.55 | 0.37 | GNU Radio fir_filter_ccf (fir_filter_ccf_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_match_block_liquid` | 43.14 | 228.29 | 0.19 | 38.95 | 186.56 | 0.21 | liquid-dsp 1.8.3 firfilt_crcf (firfilt.proto.c), the SIMD dot product of the level (the portable loop at simd128) |
| `pulse_shape_block` | 7.77 | 50.59 | 0.15 | 9.58 | 57.06 | 0.17 | GNU Radio interp_fir_filter_ccf (interp_fir_filter_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_shape_block_liquid` | 7.77 | 75.35 | 0.10 | 9.58 | 63.53 | 0.15 | liquid-dsp 1.8.3 firinterp_crcf (firinterp.proto.c, firpfb.proto.c), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_acquire_block` | 51.31 | 51.31 | 1.00 | 43.41 | 43.41 | 1.00 | sdrk fallback |
| `qam_demod_block` | 108.35 | 485.33 | 0.22 | 97.91 | 406.59 | 0.24 | GNU Radio fir_filter_ccf + symbol_sync_cc + linear_equalizer + constellation_receiver_cb (the sum of the blocks of the receiver chain, generic compiler flags and VOLK generic dot products; no acquisition, no blind equalizer stage and no lock watches) |
| `qam_demod_qam256_block` | 108.52 | 108.52 | 1.00 | 98.98 | 98.98 | 1.00 | sdrk fallback |
| `qam_demod_search_block` | 114.49 | 114.49 | 1.00 | 103.14 | 103.14 | 1.00 | sdrk fallback |
| `qam_frame_decode_block` | 126.38 | 126.38 | 1.00 | 115.09 | 115.09 | 1.00 | sdrk fallback |
| `qam_frame_encode_block` | 9.74 | 9.74 | 1.00 | 11.45 | 11.45 | 1.00 | sdrk fallback |
| `qam_mod_block` | 8.71 | 216.68 | 0.04 | 10.47 | 215.56 | 0.05 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_block_liquid` | 8.71 | 78.45 | 0.11 | 10.47 | 66.51 | 0.16 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_mod_qam256_block` | 9.97 | 217.84 | 0.05 | 11.45 | 216.71 | 0.05 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_qam256_block_liquid` | 9.97 | 79.47 | 0.13 | 11.45 | 67.52 | 0.17 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_soft_bits_qam1024` | 7.20 | 7.19 | 1.00 | 8.07 | 8.06 | 1.00 | sdrk fallback |
| `qam_soft_bits_qam16` | 1.35 | 1864.75 | 0.00 | 1.49 | 1588.53 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam16_liquid` | 1.35 | 220.40 | 0.01 | 1.49 | 226.71 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam16_lut` | 1.35 | 78.15 | 0.02 | 1.49 | 66.61 | 0.02 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qam256` | 5.65 | 5.82 | 0.97 | 5.72 | 6.56 | 0.87 | sdrk fallback |
| `qam_soft_bits_qam64` | 3.03 | 6763.49 | 0.00 | 3.34 | 5733.38 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam64_liquid` | 3.03 | 269.15 | 0.01 | 3.34 | 274.90 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam64_lut` | 3.03 | 78.15 | 0.04 | 3.34 | 66.61 | 0.05 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qpsk` | 0.52 | 0.52 | 0.99 | 0.52 | 0.59 | 0.89 | sdrk fallback |
| `rds_decode_block` | 8.74 | 8.74 | 1.00 | 11.50 | 11.50 | 1.00 | sdrk fallback |
| `rds_demod_block` | 41.84 | 41.84 | 1.00 | 38.66 | 38.66 | 1.00 | sdrk fallback |
| `rds_shape` | 78.95 | 78.95 | 1.00 | 71.83 | 71.83 | 1.00 | sdrk fallback |
| `real_to_complex` | 5.33 | 55.36 | 0.10 | 5.44 | 50.25 | 0.11 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its SSE (SSE3 horizontal add) dot product (-O3 -ffp-contract=off, sse4.2) |
| `real_to_complex_cq15` | 7.55 | 10.87 | 0.69 | 6.96 | 13.90 | 0.50 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, sse4.2) |
| `real_to_complex_cq15_fallback` | 22.70 | 10.87 | 2.09 | 23.69 | 13.90 | 1.70 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, sse4.2) |
| `real_to_complex_fallback` | 17.36 | 70.45 | 0.25 | 19.91 | 57.08 | 0.35 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse4.2) |
| `snr_estimate_block` | 7.30 | 60.76 | 0.12 | 7.28 | 56.65 | 0.13 | GNU Radio mpsk_snr_est_cc(SNR_EST_M2M4, 4096, 0.001): the moments \|x\|^2 and \|x\|^4 averaged in double, six cabsf per symbol (reference does different work: no decisions, a PSK kurtosis; generic compiler flags) |
| `snr_estimate_block_svr` | 7.30 | 88.58 | 0.08 | 7.28 | 80.92 | 0.09 | GNU Radio mpsk_snr_est_cc(SNR_EST_SVR, 4096, 0.001): \|x\|^4 and \|x_n\|^2 \|x_(n-1)\|^2 averaged in double, two cabsf per symbol (reference does different work: no decisions, a PSK constellation; generic compiler flags) |
| `stereo_encode` | 10.15 | 10.15 | 1.00 | 10.88 | 10.88 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 14.16 | 14.16 | 1.00 | 15.14 | 15.14 | 1.00 | sdrk fallback |
| `stereo_matrix` | 1.02 | 2.10 | 0.48 | 0.99 | 2.48 | 0.40 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.51 | 1.52 | 0.34 | 0.53 | 0.83 | 0.64 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 1.52 | 3.43 | 0.44 | 0.83 | 2.20 | 0.38 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `stereo_matrix_fallback` | 1.52 | 1.07 | 1.42 | 1.46 | 1.38 | 1.06 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `stereo_pll_fallback` | 88.29 | 71.63 | 1.23 | 77.34 | 63.55 | 1.22 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, sse4.2) |
| `stereo_pll_q15_fallback` | 38.04 | 55.54 | 0.68 | 38.06 | 58.48 | 0.65 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `symbol_demap_qam16_block` | 1.35 | 1864.75 | 0.00 | 1.49 | 1588.53 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam16_block_liquid` | 1.35 | 220.40 | 0.01 | 1.49 | 226.70 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam16_block_lut` | 1.35 | 66.25 | 0.02 | 1.49 | 52.13 | 0.03 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_demap_qam256_block` | 5.66 | 5.66 | 1.00 | 5.72 | 5.72 | 1.00 | sdrk fallback |
| `symbol_demap_qam64_block` | 3.03 | 6763.49 | 0.00 | 3.34 | 5733.38 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam64_block_liquid` | 3.03 | 269.14 | 0.01 | 3.34 | 274.89 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam64_block_lut` | 3.03 | 78.15 | 0.04 | 3.34 | 66.60 | 0.05 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_map_block` | 7.10 | 13.20 | 0.54 | 6.61 | 15.02 | 0.44 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_liquid` | 7.10 | 22.45 | 0.32 | 6.61 | 23.09 | 0.29 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qam64` | 10.11 | 17.84 | 0.57 | 10.12 | 19.62 | 0.52 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qam64_liquid` | 10.11 | 26.53 | 0.38 | 10.12 | 27.16 | 0.37 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qpsk` | 6.01 | 8.56 | 0.70 | 7.74 | 10.43 | 0.74 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qpsk_liquid` | 6.01 | 18.37 | 0.33 | 7.74 | 19.02 | 0.41 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_sync` | 38.82 | 38.82 | 1.00 | 30.04 | 30.04 | 1.00 | sdrk fallback |
| `symbol_sync_fallback` | 38.82 | 57.52 | 0.67 | 30.04 | 57.78 | 0.52 | GNU Radio symbol_sync_cc (symbol_sync_cc_impl.cc of gr-digital, TED_GARDNER, IR_MMSE_8TAP, generic compiler flags; reference does different work: an 8-tap MMSE interpolator against our 4-point Lagrange polynomial) |
| `symbol_sync_real` | 19.03 | 19.03 | 1.00 | 14.53 | 14.53 | 1.00 | sdrk fallback |
| `xcorr_fft_n4096` | 25.43 | 69.62 | 0.37 | 24.00 | 65.57 | 0.37 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n4096_fallback` | 104.08 | 373.34 | 0.28 | 101.88 | 393.11 | 0.26 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |
| `xcorr_fft_n512` | 21.66 | 47.25 | 0.46 | 20.38 | 49.69 | 0.41 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n512_fallback` | 83.74 | 259.57 | 0.32 | 81.89 | 276.67 | 0.30 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |

- Not measured at this level: `real_to_complex_airspy`.

#### Simulated cycles at `avx2` (`x86_64`)

Reference model `znver3`, counter-model `alderlake`, mode `whole-loop`; execution counts: valgrind callgrind, execution counts per instruction and taken jumps.

| Entry | ours znver3 (cycles/sample) | comparison znver3 (cycles/sample) | ratio znver3 (ours/comparison) | ours alderlake (cycles/sample) | comparison alderlake (cycles/sample) | ratio alderlake (ours/comparison) | comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `convert_cq15_cs8` | 0.14 | 0.14 | 0.97 | 0.14 | 0.18 | 0.76 | volk_8i_convert_16i |
| `convert_cq15_cs8_fallback` | 0.14 | 0.26 | 0.53 | 0.14 | 0.20 | 0.69 | CMSIS-DSP 1.18.0 arm_q7_to_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_cq15_cu8` | 0.14 | 0.14 | 0.97 | 0.14 | 0.18 | 0.80 | volk_8i_convert_16i on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cq15_cu8_fallback` | 0.14 | 0.26 | 0.53 | 0.15 | 0.20 | 0.74 | CMSIS-DSP 1.18.0 arm_q7_to_q15 on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_cs16` | 0.26 | 0.32 | 0.81 | 0.27 | 0.38 | 0.71 | volk_16i_s32f_convert_32f |
| `convert_cs16_fallback` | 0.27 | 0.39 | 0.68 | 0.29 | 0.40 | 0.72 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_cs8` | 0.26 | 0.48 | 0.55 | 0.27 | 0.32 | 0.84 | volk_8i_s32f_convert_32f |
| `convert_cs8_fallback` | 0.27 | 0.58 | 0.46 | 0.29 | 0.58 | 0.50 | CMSIS-DSP 1.18.0 arm_q7_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_cu8` | 0.27 | 0.48 | 0.55 | 0.27 | 0.32 | 0.84 | volk_8i_s32f_convert_32f on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cu8_fallback` | 0.28 | 0.58 | 0.49 | 0.37 | 0.58 | 0.64 | CMSIS-DSP 1.18.0 arm_q7_to_float on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_out_cs16` | 0.45 | 0.46 | 0.99 | 0.55 | 0.53 | 1.04 | volk_32f_s32f_convert_16i |
| `convert_out_cs16_fallback` | 0.77 | 0.46 | 1.65 | 0.73 | 0.58 | 1.25 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_out_cs8` | 0.43 | 0.50 | 0.87 | 0.56 | 0.52 | 1.07 | volk_32f_s32f_convert_8i |
| `convert_out_cs8_fallback` | 1.06 | 0.51 | 2.06 | 1.02 | 0.63 | 1.63 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_out_cu8` | 0.53 | 0.54 | 0.99 | 0.67 | 0.60 | 1.11 | volk_32f_s32f_x2_convert_8u |
| `convert_out_cu8_fallback` | 1.67 | 0.51 | 3.24 | 1.82 | 0.63 | 2.90 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_out_real_rs16` | 0.24 | 0.24 | 1.00 | 0.28 | 0.27 | 1.06 | volk_32f_s32f_convert_16i |
| `convert_out_real_rs16_fallback` | 0.39 | 0.24 | 1.66 | 0.37 | 0.29 | 1.27 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_out_real_rs8` | 0.23 | 0.25 | 0.89 | 0.29 | 0.26 | 1.09 | volk_32f_s32f_convert_8i |
| `convert_out_real_rs8_fallback` | 0.54 | 0.26 | 2.06 | 0.52 | 0.32 | 1.64 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_out_real_ru8` | 0.28 | 0.28 | 1.00 | 0.35 | 0.30 | 1.14 | volk_32f_s32f_x2_convert_8u |
| `convert_out_real_ru8_fallback` | 0.84 | 0.26 | 3.22 | 0.92 | 0.32 | 2.91 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_real_rs16` | 0.14 | 0.17 | 0.84 | 0.14 | 0.19 | 0.73 | volk_16i_s32f_convert_32f |
| `convert_real_rs16_fallback` | 0.14 | 0.20 | 0.69 | 0.15 | 0.20 | 0.74 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `rotate` | 0.66 | 3.19 | 0.21 | 0.57 | 3.05 | 0.19 | volk_32fc_s32fc_x2_rotator2_32fc |
| `rotate_cq15` | 1.18 | 4.49 | 0.26 | 1.25 | 5.49 | 0.23 | sdrk fallback |
| `rotate_cq15_fallback` | 4.49 | 29.18 | 0.15 | 5.49 | 25.57 | 0.21 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `rotate_fallback` | 2.65 | 10.63 | 0.25 | 4.41 | 11.17 | 0.39 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `scramble_block` | 2.85 | 2.85 | 1.00 | 3.88 | 3.88 | 1.00 | sdrk fallback |
| `scramble_c32_b1` | 0.41 | 1.36 | 0.30 | 0.30 | 1.48 | 0.20 | +-1.0 sequence from the index bits into a buffer of 256 + volk_32fc_32f_multiply_32fc (VOLK 3.3.0 dispatcher; the form of srsRAN srsran_scrambling_c) |
| `scramble_c32_b1_fallback` | 0.64 | 1.36 | 0.47 | 0.78 | 1.31 | 0.59 | +-1.0 sequence from the index bits into a buffer of 256 + VOLK 3.3.0 volk_32fc_32f_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2` | 0.95 | 2.37 | 0.40 | 1.04 | 2.41 | 0.43 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2_dvbs2` | 0.95 | 2.98 | 0.32 | 1.04 | 2.42 | 0.43 | GNU Radio dvbs2_physical_cc PL scrambling, the per-symbol switch on R_n (benches/refs/gr-dvbs2-plscramble.c, main at ec1bf15; reference does different work; -O3) |
| `scramble_c32_b2_fallback` | 1.98 | 6.24 | 0.32 | 1.70 | 7.00 | 0.24 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8` | 1.31 | 2.37 | 0.55 | 1.17 | 2.41 | 0.49 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8_fallback` | 2.57 | 6.24 | 0.41 | 3.04 | 7.00 | 0.43 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_cq15_b2` | 0.42 | 3.15 | 0.13 | 0.66 | 3.31 | 0.20 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `scramble_cq15_b2_fallback` | 1.05 | 3.15 | 0.33 | 1.10 | 3.31 | 0.33 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `scramble_cq15_b8` | 1.21 | 3.15 | 0.38 | 1.13 | 3.31 | 0.34 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `scramble_cq15_b8_fallback` | 3.69 | 3.15 | 1.17 | 3.25 | 3.31 | 0.98 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir` | 9.63 | 58.05 | 0.17 | 8.85 | 72.63 | 0.12 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 40.62 | 40.62 | 1.00 | 61.15 | 61.15 | 1.00 | sdrk fallback |
| `fir_cq15_fallback` | 40.62 | 144.83 | 0.28 | 61.15 | 154.65 | 0.40 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_decimate` | 2.42 | 9.29 | 0.26 | 2.08 | 12.60 | 0.17 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 21.61 | 21.61 | 1.00 | 22.05 | 22.05 | 1.00 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 21.61 | 42.02 | 0.51 | 22.05 | 17.52 | 1.26 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_decimate_fallback` | 11.63 | 71.28 | 0.16 | 11.54 | 32.88 | 0.35 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_fallback` | 31.56 | 51.44 | 0.61 | 30.50 | 80.42 | 0.38 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate` | 12.44 | 66.84 | 0.19 | 11.40 | 76.70 | 0.15 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 52.45 | 52.45 | 1.00 | 71.49 | 71.49 | 1.00 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 52.45 | 136.94 | 0.38 | 71.49 | 136.97 | 0.52 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate_fallback` | 41.03 | 101.68 | 0.40 | 38.06 | 101.03 | 0.38 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate_real` | 7.27 | 83.60 | 0.09 | 6.78 | 93.79 | 0.07 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 25.89 | 50.84 | 0.51 | 24.02 | 50.51 | 0.48 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate_real_q15` | 42.13 | 42.13 | 1.00 | 38.06 | 38.06 | 1.00 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 42.13 | 68.47 | 0.62 | 38.06 | 68.49 | 0.56 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate_short` | 3.31 | 64.34 | 0.05 | 3.12 | 67.82 | 0.05 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_short_fallback` | 10.02 | 18.52 | 0.54 | 9.80 | 17.80 | 0.55 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 8 (80 taps), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `resample` | 14.92 | 68.15 | 0.22 | 14.17 | 77.87 | 0.18 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 117.53 | 117.53 | 1.00 | 113.20 | 113.20 | 1.00 | sdrk fallback |
| `resample_cq15_fallback` | 117.53 | 136.18 | 0.86 | 113.20 | 136.22 | 0.83 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `resample_fallback` | 64.91 | 100.83 | 0.64 | 61.64 | 100.27 | 0.61 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `resample_real` | 9.82 | 84.38 | 0.12 | 9.13 | 94.98 | 0.10 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 60.00 | 50.42 | 1.19 | 69.79 | 50.13 | 1.39 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `resample_real_q15` | 60.80 | 112.73 | 0.54 | 66.72 | 99.45 | 0.67 | sdrk fallback |
| `resample_real_q15_fallback` | 112.73 | 68.09 | 1.66 | 99.45 | 68.11 | 1.46 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mag` | 0.76 | 0.91 | 0.84 | 0.76 | 0.78 | 0.97 | volk_32fc_magnitude_32f |
| `mag_fallback` | 0.90 | 1.15 | 0.78 | 0.78 | 0.83 | 0.94 | CMSIS-DSP 1.18.0 arm_cmplx_mag_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mag_q15` | 0.97 | 1.06 | 0.92 | 0.98 | 0.92 | 1.07 | volk_16ic_magnitude_16i |
| `mag_q15_fallback` | 1.93 | 79.45 | 0.02 | 1.30 | 83.02 | 0.02 | CMSIS-DSP 1.18.0 arm_cmplx_mag_q15 (Q2.14, arm_sqrt_q31), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mag_sq` | 0.22 | 0.33 | 0.68 | 0.26 | 0.52 | 0.50 | volk_32fc_magnitude_squared_32f |
| `mag_sq_fallback` | 0.29 | 0.52 | 0.55 | 0.32 | 0.53 | 0.61 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mag_sq_q15` | 0.14 | 0.20 | 0.69 | 0.18 | 0.23 | 0.80 | sdrk fallback |
| `mag_sq_q15_fallback` | 0.20 | 2.17 | 0.09 | 0.23 | 1.84 | 0.12 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_q15 (Q3.13, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `quad_demod` | 2.60 | 3.61 | 0.72 | 2.12 | 3.70 | 0.57 | volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f |
| `quad_demod_cq15` | 4.40 | 71.20 | 0.06 | 4.01 | 66.54 | 0.06 | sdrk fallback |
| `quad_demod_cq15_fallback` | 71.20 | 84.76 | 0.84 | 66.54 | 64.37 | 1.03 | CMSIS-DSP arm_cmplx_mult_cmplx_q15 + arm_atan2_q15 + arm_scale_q15 (generic C), conjugate in a plain loop |
| `quad_demod_fallback` | 39.05 | 865.81 | 0.05 | 32.80 | 130.25 | 0.25 | GNU Radio quadrature_demod_cf (gr-analog, fast_atan2f, generic compiler flags) |
| `fm_mod` | 2.06 | 105.76 | 0.02 | 2.13 | 25.57 | 0.08 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `fm_mod_cq15` | 1.93 | 7.53 | 0.26 | 2.16 | 7.91 | 0.27 | sdrk fallback |
| `fm_mod_cq15_fallback` | 7.53 | 21.41 | 0.35 | 7.91 | 19.58 | 0.40 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample of an integer phase accumulator, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fm_mod_fallback` | 5.34 | 105.76 | 0.05 | 6.42 | 25.57 | 0.25 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `squelch` | 9.13 | 20.51 | 0.45 | 10.13 | 15.26 | 0.66 | GNU Radio pwr_squelch_cc (squelch_base_cc_impl.cc of gr-analog, generic compiler flags) |
| `squelch_cq15` | 5.56 | 5.56 | 1.00 | 6.95 | 6.95 | 1.00 | sdrk fallback |
| `window` | 0.33 | 0.38 | 0.87 | 0.26 | 0.53 | 0.49 | volk_32fc_32f_multiply_32fc |
| `window_cq15` | 0.22 | 0.56 | 0.39 | 0.21 | 0.56 | 0.37 | sdrk fallback |
| `window_cq15_fallback` | 0.56 | 0.75 | 0.75 | 0.56 | 0.82 | 0.68 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `window_fallback` | 0.39 | 0.41 | 0.97 | 0.36 | 0.38 | 0.94 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `power_db` | 1.85 | 1.18 | 1.57 | 1.88 | 1.52 | 1.24 | volk_32f_log2_32f + volk_32f_s32f_multiply_32f + volk_32f_s32f_add_32f |
| `power_db_fallback` | 2.76 | 41.06 | 0.07 | 2.08 | 46.55 | 0.04 | CMSIS-DSP 1.18.0 arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `power_db_q15` | 1.89 | 3.07 | 0.62 | 1.90 | 2.46 | 0.77 | sdrk fallback |
| `power_db_q15_fallback` | 3.07 | 41.25 | 0.07 | 2.46 | 46.74 | 0.05 | CMSIS-DSP 1.18.0 arm_q15_to_float + arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `agc` | 13.18 | 36.12 | 0.36 | 14.78 | 39.15 | 0.38 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `agc_cq15` | 78.32 | 78.32 | 1.00 | 85.53 | 85.53 | 1.00 | sdrk fallback |
| `agc_fallback` | 17.93 | 36.12 | 0.50 | 18.42 | 39.15 | 0.47 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `dc_block` | 2.95 | 7.03 | 0.42 | 2.76 | 9.04 | 0.31 | sdrk fallback |
| `dc_block_cq15` | 6.30 | 6.30 | 1.00 | 6.13 | 6.13 | 1.00 | sdrk fallback |
| `dc_block_fallback` | 7.03 | 46.65 | 0.15 | 9.04 | 39.57 | 0.23 | liquid-dsp 1.8.3 iirfilt_crcf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real` | 2.12 | 7.03 | 0.30 | 1.88 | 9.04 | 0.21 | sdrk fallback |
| `dc_block_real_fallback` | 7.03 | 36.03 | 0.20 | 9.04 | 34.31 | 0.26 | liquid-dsp 1.8.3 iirfilt_rrrf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real_q15` | 5.62 | 5.62 | 1.00 | 5.56 | 5.56 | 1.00 | sdrk fallback |
| `deemphasis` | 2.09 | 9.55 | 0.22 | 1.84 | 11.58 | 0.16 | sdrk fallback |
| `deemphasis_fallback` | 9.55 | 10.80 | 0.88 | 11.58 | 23.14 | 0.50 | GNU Radio fm_deemph: iir_filter_ffd with the bilinear taps of fm_emph.py (double taps and accumulator, generic compiler flags) |
| `deemphasis_q15` | 5.53 | 5.53 | 1.00 | 5.54 | 5.54 | 1.00 | sdrk fallback |
| `biquad` | 9.37 | 20.59 | 0.46 | 10.57 | 19.11 | 0.55 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (avx2, -ffp-contract=off), two calls on planar re and im (deinterleaving not counted) |
| `biquad_cq15` | 9.16 | 21.19 | 0.43 | 10.56 | 21.26 | 0.50 | sdrk fallback |
| `biquad_cq15_fallback` | 21.19 | 54.13 | 0.39 | 21.26 | 56.47 | 0.38 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `biquad_fallback` | 18.46 | 24.04 | 0.77 | 20.84 | 26.04 | 0.80 | CMSIS-DSP 1.18.0 arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `biquad_real` | 9.12 | 10.30 | 0.89 | 10.06 | 9.56 | 1.05 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (avx2, -ffp-contract=off) |
| `biquad_real_fallback` | 10.13 | 18.04 | 0.56 | 11.77 | 20.05 | 0.59 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df2T_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `biquad_real_q15` | 8.18 | 11.11 | 0.74 | 10.21 | 12.14 | 0.84 | sdrk fallback |
| `biquad_real_q15_fallback` | 11.11 | 27.07 | 0.41 | 12.14 | 28.24 | 0.43 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot` | 0.26 | 0.27 | 0.97 | 0.26 | 0.52 | 0.50 | volk_32fc_32f_dot_prod_32fc |
| `dot_complex` | 0.46 | 0.78 | 0.59 | 0.58 | 0.78 | 0.74 | VOLK 3.3.0 volk_32fc_x2_dot_prod_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `dot_complex_fallback` | 1.94 | 6.03 | 0.32 | 1.68 | 6.03 | 0.28 | CMSIS-DSP 1.18.0 arm_cmplx_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot_cq15` | 0.30 | 1.04 | 0.29 | 0.40 | 0.77 | 0.52 | sdrk fallback |
| `dot_cq15_fallback` | 1.04 | 1.07 | 0.97 | 0.77 | 1.03 | 0.75 | CMSIS-DSP 1.18.0 arm_dot_prod_q15 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot_fallback` | 3.03 | 6.07 | 0.50 | 3.03 | 6.07 | 0.50 | CMSIS-DSP 1.18.0 arm_dot_prod_f32 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot_real` | 0.14 | 0.52 | 0.26 | 0.10 | 0.77 | 0.13 | volk_32f_x2_dot_prod_32f |
| `dot_real_fallback` | 3.03 | 3.04 | 1.00 | 3.03 | 3.04 | 1.00 | CMSIS-DSP 1.18.0 arm_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot_real_q15` | 0.13 | 0.32 | 0.40 | 0.16 | 0.38 | 0.42 | sdrk fallback |
| `dot_real_q15_fallback` | 0.32 | 0.54 | 0.59 | 0.38 | 0.52 | 0.74 | CMSIS-DSP 1.18.0 arm_dot_prod_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `scale` | 0.26 | 0.27 | 0.94 | 0.18 | 0.29 | 0.64 | volk_32f_s32f_multiply_32f |
| `scale_cq15` | 0.13 | 0.28 | 0.47 | 0.10 | 0.42 | 0.24 | sdrk fallback |
| `scale_cq15_fallback` | 0.28 | 0.65 | 0.44 | 0.42 | 0.78 | 0.54 | CMSIS-DSP 1.18.0 arm_scale_q15 (shift 0, truncating) on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `scale_fallback` | 0.26 | 0.27 | 0.95 | 0.20 | 0.29 | 0.69 | CMSIS-DSP 1.18.0 arm_scale_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `add` | 0.38 | 0.40 | 0.96 | 0.26 | 0.35 | 0.73 | volk_32fc_x2_add_32fc |
| `add_cq15` | 0.19 | 0.20 | 0.95 | 0.13 | 0.18 | 0.74 | volk_16i_x2_add_saturated_16i on the 2 n values |
| `add_cq15_fallback` | 0.19 | 1.03 | 0.19 | 0.14 | 1.03 | 0.13 | CMSIS-DSP 1.18.0 arm_add_q15 on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `add_fallback` | 0.38 | 0.40 | 0.97 | 0.27 | 0.35 | 0.77 | CMSIS-DSP 1.18.0 arm_add_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `add_real` | 0.19 | 0.20 | 0.95 | 0.13 | 0.18 | 0.73 | volk_32f_x2_add_32f |
| `add_real_fallback` | 0.19 | 0.20 | 0.96 | 0.14 | 0.18 | 0.77 | CMSIS-DSP 1.18.0 arm_add_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `add_real_q15` | 0.10 | 0.10 | 0.93 | 0.07 | 0.09 | 0.73 | volk_16i_x2_add_saturated_16i |
| `add_real_q15_fallback` | 0.10 | 0.52 | 0.19 | 0.07 | 0.52 | 0.13 | CMSIS-DSP 1.18.0 arm_add_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mul` | 0.45 | 0.49 | 0.93 | 0.51 | 0.54 | 0.94 | volk_32fc_x2_multiply_32fc |
| `mul_cq15` | 0.51 | 0.83 | 0.61 | 0.63 | 1.37 | 0.46 | sdrk fallback |
| `mul_cq15_fallback` | 0.83 | 1.27 | 0.65 | 1.37 | 1.41 | 0.98 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_q15 (Q3.13, truncating, scaled by 4), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mul_fallback` | 0.51 | 0.53 | 0.97 | 0.51 | 0.79 | 0.65 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mul_real` | 0.19 | 0.20 | 0.95 | 0.13 | 0.18 | 0.73 | volk_32f_x2_multiply_32f |
| `mul_real_fallback` | 0.19 | 0.20 | 0.96 | 0.14 | 0.18 | 0.77 | CMSIS-DSP 1.18.0 arm_mult_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mul_real_q15` | 0.10 | 0.28 | 0.35 | 0.09 | 0.39 | 0.22 | sdrk fallback |
| `mul_real_q15_fallback` | 0.28 | 0.30 | 0.92 | 0.39 | 0.31 | 1.25 | CMSIS-DSP 1.18.0 arm_mult_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `interleave` | 0.26 | 0.27 | 0.97 | 0.51 | 0.52 | 0.97 | volk_32f_x2_interleave_32fc |
| `interleave_cq15` | 0.13 | 0.13 | 1.01 | 0.13 | 0.13 | 1.00 | sdrk fallback |
| `interleave_cq15_fallback` | 0.13 | 0.14 | 0.98 | 0.13 | 0.14 | 0.96 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic on int16 (VOLK has no 16-bit interleave), benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx2) |
| `interleave_fallback` | 0.26 | 0.27 | 0.98 | 0.51 | 0.52 | 0.98 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx2) |
| `deinterleave` | 0.26 | 0.39 | 0.66 | 0.26 | 0.40 | 0.65 | volk_32fc_deinterleave_32f_x2 |
| `deinterleave_cq15` | 0.17 | 0.26 | 0.63 | 0.20 | 0.26 | 0.74 | volk_16ic_deinterleave_16i_x2 |
| `deinterleave_cq15_fallback` | 0.23 | 0.20 | 1.14 | 0.23 | 0.26 | 0.86 | VOLK 3.3.0 volk_16ic_deinterleave_16i_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx2) |
| `deinterleave_fallback` | 0.26 | 1.02 | 0.26 | 0.26 | 0.52 | 0.50 | VOLK 3.3.0 volk_32fc_deinterleave_32f_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx2) |
| `oscillate` | 1.40 | 2.27 | 0.62 | 1.45 | 4.03 | 0.36 | sdrk fallback |
| `oscillate_cq15` | 1.90 | 4.51 | 0.42 | 1.49 | 4.80 | 0.31 | sdrk fallback |
| `oscillate_cq15_fallback` | 4.51 | 16.77 | 0.27 | 4.80 | 16.78 | 0.29 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample (512-entry table, linear interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `oscillate_fallback` | 2.27 | 31.51 | 0.07 | 4.03 | 80.72 | 0.05 | CMSIS-DSP 1.18.0 arm_sin_cos_f32 of the angle in degrees per sample (512-entry sine table, cubic Hermite interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_1024` | 2.35 | 2.70 | 0.87 | 2.26 | 3.38 | 0.67 | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 18.39 | 20.03 | 0.92 | 18.85 | 19.56 | 0.96 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_4096` | 2.62 | 3.37 | 0.78 | 2.51 | 3.83 | 0.66 | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 21.85 | 23.30 | 0.94 | 22.32 | 22.33 | 1.00 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_64` | 1.72 | 2.04 | 0.84 | 1.85 | 1.89 | 0.98 | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 12.55 | 12.51 | 1.00 | 13.01 | 12.47 | 1.04 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_cq15_1024` | 2.47 | 46.26 | 0.05 | 2.79 | 48.58 | 0.06 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 2.31 | 46.26 | 0.05 | 2.34 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 28.16 | 37.69 | 0.75 | 30.91 | 38.63 | 0.80 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_cq15_1024_fallback` | 46.25 | 37.69 | 1.23 | 51.39 | 38.63 | 1.33 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_cq15_1024_full_scale` | 2.41 | 46.26 | 0.05 | 2.45 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 2.11 | 2.71 | 0.78 | 2.08 | 2.82 | 0.74 | realfft on rustfft (RealFftPlanner, forward, out of place) |
| `rfft_1024_fallback` | 9.66 | 9.77 | 0.99 | 9.94 | 9.84 | 1.01 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `am_mod` | 0.65 | 0.65 | 1.00 | 0.47 | 0.47 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 0.20 | 0.43 | 0.47 | 0.21 | 0.56 | 0.37 | sdrk fallback |
| `am_mod_cq15_fallback` | 0.43 | 2.91 | 0.15 | 0.56 | 2.30 | 0.24 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `am_mod_fallback` | 0.65 | 0.53 | 1.22 | 0.47 | 0.54 | 0.87 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 164.98 | 164.98 | 1.00 | 164.85 | 164.85 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 47.93 | 47.93 | 1.00 | 46.70 | 46.70 | 1.00 | sdrk fallback |
| `bit_descramble_block` | 0.33 | 4.75 | 0.07 | 0.32 | 5.55 | 0.06 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_descramble_in_place_block` | 1.48 | 4.75 | 0.31 | 1.23 | 5.55 | 0.22 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_block` | 0.77 | 4.41 | 0.17 | 0.81 | 5.18 | 0.16 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_in_place_block` | 1.02 | 4.41 | 0.23 | 0.99 | 5.18 | 0.19 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `channel_bank_channelize` | 20.88 | 20.88 | 1.00 | 25.55 | 25.55 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 73.01 | 73.01 | 1.00 | 65.32 | 65.32 | 1.00 | sdrk fallback |
| `channel_select_block` | 0.76 | 0.76 | 1.00 | 0.52 | 0.52 | 1.00 | sdrk fallback |
| `channelize` | 11.83 | 119.62 | 0.10 | 12.74 | 101.71 | 0.13 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 12.08 | 12.08 | 1.00 | 12.81 | 12.81 | 1.00 | sdrk fallback |
| `channelize_fallback` | 34.73 | 147.96 | 0.23 | 32.88 | 133.26 | 0.25 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `correlate_block` | 7.60 | 7.60 | 1.00 | 7.15 | 7.15 | 1.00 | sdrk fallback |
| `costas` | 56.17 | 56.17 | 1.00 | 68.16 | 68.16 | 1.00 | sdrk fallback |
| `costas_fallback` | 56.17 | 76.46 | 0.73 | 68.16 | 66.01 | 1.03 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `costas_qam1024` | 131.34 | 131.34 | 1.00 | 152.10 | 152.10 | 1.00 | sdrk fallback |
| `costas_qam1024_fallback` | 131.34 | 211.17 | 0.62 | 152.10 | 254.58 | 0.60 | GNU Radio constellation_receiver_cb with constellation_rect 1024-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam16` | 124.35 | 124.35 | 1.00 | 148.39 | 148.39 | 1.00 | sdrk fallback |
| `costas_qam16_fallback` | 124.35 | 166.34 | 0.75 | 148.39 | 172.78 | 0.86 | GNU Radio constellation_receiver_cb with constellation_rect 16-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam256` | 131.66 | 131.66 | 1.00 | 152.53 | 152.53 | 1.00 | sdrk fallback |
| `costas_qam256_fallback` | 131.66 | 169.39 | 0.78 | 152.53 | 178.14 | 0.86 | GNU Radio constellation_receiver_cb with constellation_rect 256-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam64` | 128.23 | 128.23 | 1.00 | 154.25 | 154.25 | 1.00 | sdrk fallback |
| `costas_qam64_fallback` | 128.23 | 166.51 | 0.77 | 154.25 | 173.10 | 0.89 | GNU Radio constellation_receiver_cb with constellation_rect 64-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 11.81 | 13.17 | 0.90 | 12.80 | 16.19 | 0.79 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 7.05 | 10.39 | 0.68 | 8.05 | 9.40 | 0.86 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 11.04 | 13.17 | 0.84 | 12.04 | 16.19 | 0.74 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 7.05 | 10.39 | 0.68 | 8.05 | 9.40 | 0.86 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 111.94 | 1076.92 | 0.10 | 152.23 | 287.12 | 0.53 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_fallback` | 111.94 | 931.44 | 0.12 | 152.23 | 220.19 | 0.69 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam1024` | 142.06 | 142.06 | 1.00 | 174.29 | 174.29 | 1.00 | sdrk fallback |
| `equalize_qam1024_fallback` | 142.06 | 943.08 | 0.15 | 174.29 | 277.17 | 0.63 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 1024-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam16` | 119.94 | 1088.28 | 0.11 | 164.27 | 294.40 | 0.56 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam16_fallback` | 119.94 | 943.86 | 0.13 | 164.27 | 277.23 | 0.59 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam256` | 142.06 | 142.06 | 1.00 | 174.29 | 174.29 | 1.00 | sdrk fallback |
| `equalize_qam256_fallback` | 142.06 | 943.86 | 0.15 | 174.29 | 277.23 | 0.63 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 256-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam64` | 123.94 | 1088.28 | 0.11 | 170.30 | 294.40 | 0.58 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam64_fallback` | 123.94 | 943.86 | 0.13 | 170.30 | 277.23 | 0.61 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 5.57 | 5.57 | 1.00 | 6.18 | 6.18 | 1.00 | sdrk fallback |
| `frame_decode_block` | 6.91 | 6.91 | 1.00 | 5.80 | 5.80 | 1.00 | sdrk fallback |
| `frame_decode_four_block` | 23.43 | 23.43 | 1.00 | 20.78 | 20.78 | 1.00 | sdrk fallback |
| `frame_encode_block` | 1.64 | 1.64 | 1.00 | 1.73 | 1.73 | 1.00 | sdrk fallback |
| `gauss_noise_c32` | 3.42 | 194.93 | 0.02 | 4.97 | 191.42 | 0.03 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fallback` | 18.12 | 194.93 | 0.09 | 16.34 | 191.42 | 0.09 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fastnoise` | 3.42 | 4.04 | 0.85 | 4.97 | 6.09 | 0.82 | GNU Radio fastnoise_source_c, amplitude sigma * sqrt(2) (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32` | 1.72 | 117.09 | 0.01 | 2.49 | 125.14 | 0.02 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fallback` | 9.06 | 117.09 | 0.08 | 8.17 | 125.14 | 0.07 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fastnoise` | 1.72 | 4.05 | 0.42 | 2.49 | 6.10 | 0.41 | GNU Radio fastnoise_source_f, amplitude sigma (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32_ziggurat` | 1.72 | 9.92 | 0.17 | 2.49 | 10.75 | 0.23 | rand_distr 0.5.1 StandardNormal as f32 times sigma over rand 0.9.5 SmallRng (Ziggurat in f64 with rejection: a variable number of draws per value, no random access, no bit-exact sequence across platforms; reference does different work) |
| `gauss_noise_q15` | 2.33 | 10.63 | 0.22 | 3.02 | 9.46 | 0.32 | sdrk fallback |
| `gauss_noise_q15_fallback` | 10.63 | 10.63 | 1.00 | 9.46 | 9.46 | 1.00 | sdrk fallback |
| `halfband_decimate` | 3.43 | 87.24 | 0.04 | 2.97 | 34.43 | 0.09 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its AVX dot product (-O3 -ffp-contract=off, avx2) |
| `halfband_decimate_cq15` | 6.56 | 12.86 | 0.51 | 7.11 | 13.27 | 0.54 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 12.86 | 30.91 | 0.42 | 13.27 | 33.32 | 0.40 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `halfband_decimate_fallback` | 61.26 | 68.43 | 0.90 | 13.84 | 62.30 | 0.22 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, avx2) |
| `halfband_decimate_real` | 1.81 | 65.32 | 0.03 | 1.36 | 32.31 | 0.04 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its AVX dot product (-O3 -ffp-contract=off, avx2) |
| `halfband_decimate_real_fallback` | 9.11 | 80.18 | 0.11 | 8.76 | 50.72 | 0.17 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, avx2) |
| `halfband_decimate_real_q15` | 2.95 | 11.33 | 0.26 | 3.83 | 10.11 | 0.38 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 11.33 | 15.45 | 0.73 | 10.11 | 16.66 | 0.61 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `halfband_ref_fir_decimate` | 5.53 | 21.84 | 0.25 | 5.48 | 21.35 | 0.26 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 40.97 | 40.97 | 1.00 | 39.60 | 39.60 | 1.00 | sdrk fallback |
| `halfband_ref_shifted` | 5.80 | 63.70 | 0.09 | 5.42 | 16.31 | 0.33 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 9.34 | 15.65 | 0.60 | 10.19 | 16.35 | 0.62 | sdrk fallback |
| `iq_offset_block` | 2.89 | 2.89 | 1.00 | 2.95 | 2.95 | 1.00 | sdrk fallback |
| `morse_decode_block` | 12.77 | 12.77 | 1.00 | 16.08 | 16.08 | 1.00 | sdrk fallback |
| `morse_encode_block` | 4.44 | 4.44 | 1.00 | 4.49 | 4.49 | 1.00 | sdrk fallback |
| `noise_blank` | 11.58 | 11.58 | 1.00 | 11.37 | 11.37 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 8.48 | 8.48 | 1.00 | 9.30 | 9.30 | 1.00 | sdrk fallback |
| `noise_block` | 3.42 | 3.42 | 1.00 | 4.97 | 4.97 | 1.00 | sdrk fallback |
| `preemph_limit` | 15.37 | 15.37 | 1.00 | 13.18 | 13.18 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 27.62 | 27.62 | 1.00 | 18.69 | 18.69 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 16.71 | 16.71 | 1.00 | 22.32 | 22.32 | 1.00 | sdrk fallback |
| `psk_demod_block` | 37.72 | 1161.39 | 0.03 | 38.80 | 415.11 | 0.09 | GNU Radio fir_filter_ccf + symbol_sync_cc + costas_loop_cc + constellation_decoder_cb (the sum of the blocks of the receiver chain, generic compiler flags; no acquisition and no lock watches) |
| `psk_mod_block` | 4.37 | 180.44 | 0.02 | 4.27 | 249.90 | 0.02 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `psk_mod_block_liquid` | 4.37 | 57.11 | 0.08 | 4.27 | 57.04 | 0.07 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `pulse_match_block` | 12.32 | 878.31 | 0.01 | 11.29 | 149.29 | 0.08 | GNU Radio fir_filter_ccf (fir_filter_ccf_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_match_block_liquid` | 12.32 | 109.99 | 0.11 | 11.29 | 94.89 | 0.12 | liquid-dsp 1.8.3 firfilt_crcf (firfilt.proto.c), the SIMD dot product of the level (the portable loop at simd128) |
| `pulse_shape_block` | 3.57 | 64.78 | 0.06 | 3.54 | 69.12 | 0.05 | GNU Radio interp_fir_filter_ccf (interp_fir_filter_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_shape_block_liquid` | 3.57 | 55.18 | 0.06 | 3.54 | 55.16 | 0.06 | liquid-dsp 1.8.3 firinterp_crcf (firinterp.proto.c, firpfb.proto.c), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_acquire_block` | 26.92 | 26.92 | 1.00 | 32.63 | 32.63 | 1.00 | sdrk fallback |
| `qam_demod_block` | 62.75 | 1192.78 | 0.05 | 70.69 | 455.70 | 0.16 | GNU Radio fir_filter_ccf + symbol_sync_cc + linear_equalizer + constellation_receiver_cb (the sum of the blocks of the receiver chain, generic compiler flags and VOLK generic dot products; no acquisition, no blind equalizer stage and no lock watches) |
| `qam_demod_qam256_block` | 66.52 | 66.52 | 1.00 | 72.62 | 72.62 | 1.00 | sdrk fallback |
| `qam_demod_search_block` | 65.65 | 65.65 | 1.00 | 74.37 | 74.37 | 1.00 | sdrk fallback |
| `qam_frame_decode_block` | 74.73 | 74.73 | 1.00 | 81.50 | 81.50 | 1.00 | sdrk fallback |
| `qam_frame_encode_block` | 5.26 | 5.26 | 1.00 | 5.48 | 5.48 | 1.00 | sdrk fallback |
| `qam_mod_block` | 4.39 | 180.83 | 0.02 | 4.57 | 250.43 | 0.02 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_block_liquid` | 4.39 | 57.62 | 0.08 | 4.57 | 57.80 | 0.08 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_mod_qam256_block` | 4.96 | 181.61 | 0.03 | 4.74 | 251.49 | 0.02 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_qam256_block_liquid` | 4.96 | 58.64 | 0.08 | 4.74 | 59.33 | 0.08 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_soft_bits_qam1024` | 5.53 | 5.53 | 1.00 | 4.31 | 4.31 | 1.00 | sdrk fallback |
| `qam_soft_bits_qam16` | 0.52 | 1246.72 | 0.00 | 1.02 | 1327.29 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam16_liquid` | 0.52 | 164.02 | 0.00 | 1.02 | 163.46 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam16_lut` | 0.52 | 56.48 | 0.01 | 1.02 | 55.13 | 0.02 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qam256` | 2.51 | 2.51 | 1.00 | 4.06 | 4.05 | 1.00 | sdrk fallback |
| `qam_soft_bits_qam64` | 1.55 | 4422.74 | 0.00 | 1.42 | 4747.33 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam64_liquid` | 1.55 | 234.13 | 0.01 | 1.42 | 212.09 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam64_lut` | 1.55 | 56.48 | 0.03 | 1.42 | 55.13 | 0.03 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qpsk` | 0.26 | 0.26 | 1.00 | 0.20 | 0.20 | 0.99 | sdrk fallback |
| `rds_decode_block` | 7.28 | 7.28 | 1.00 | 7.98 | 7.98 | 1.00 | sdrk fallback |
| `rds_demod_block` | 16.60 | 16.60 | 1.00 | 14.43 | 14.43 | 1.00 | sdrk fallback |
| `rds_shape` | 50.41 | 50.41 | 1.00 | 78.60 | 78.60 | 1.00 | sdrk fallback |
| `real_to_complex` | 2.49 | 91.87 | 0.03 | 2.27 | 35.24 | 0.06 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its AVX dot product (-O3 -ffp-contract=off, avx2) |
| `real_to_complex_cq15` | 3.60 | 7.27 | 0.50 | 4.87 | 8.83 | 0.55 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, avx2) |
| `real_to_complex_cq15_fallback` | 14.90 | 7.27 | 2.05 | 14.49 | 8.83 | 1.64 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, avx2) |
| `real_to_complex_fallback` | 12.67 | 106.72 | 0.12 | 12.75 | 53.64 | 0.24 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, avx2) |
| `snr_estimate_block` | 2.70 | 50.03 | 0.05 | 2.78 | 57.26 | 0.05 | GNU Radio mpsk_snr_est_cc(SNR_EST_M2M4, 4096, 0.001): the moments \|x\|^2 and \|x\|^4 averaged in double, six cabsf per symbol (reference does different work: no decisions, a PSK kurtosis; generic compiler flags) |
| `snr_estimate_block_svr` | 2.70 | 71.84 | 0.04 | 2.78 | 78.62 | 0.04 | GNU Radio mpsk_snr_est_cc(SNR_EST_SVR, 4096, 0.001): \|x\|^4 and \|x_n\|^2 \|x_(n-1)\|^2 averaged in double, two cabsf per symbol (reference does different work: no decisions, a PSK constellation; generic compiler flags) |
| `stereo_encode` | 4.47 | 4.47 | 1.00 | 5.92 | 5.92 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 7.17 | 7.17 | 1.00 | 8.91 | 8.91 | 1.00 | sdrk fallback |
| `stereo_matrix` | 0.32 | 1.26 | 0.26 | 0.45 | 1.45 | 0.31 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.15 | 0.39 | 0.39 | 0.20 | 0.54 | 0.37 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 0.39 | 1.55 | 0.25 | 0.54 | 1.55 | 0.35 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `stereo_matrix_fallback` | 0.63 | 0.60 | 1.06 | 0.86 | 0.53 | 1.61 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `stereo_pll_fallback` | 63.32 | 57.06 | 1.11 | 79.41 | 78.06 | 1.02 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, avx2) |
| `stereo_pll_q15_fallback` | 37.04 | 51.46 | 0.72 | 38.06 | 34.89 | 1.09 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `symbol_demap_qam16_block` | 0.52 | 1246.72 | 0.00 | 1.02 | 1327.29 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam16_block_liquid` | 0.52 | 164.01 | 0.00 | 1.02 | 163.45 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam16_block_lut` | 0.52 | 47.02 | 0.01 | 1.02 | 35.13 | 0.03 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_demap_qam256_block` | 2.51 | 2.51 | 1.00 | 4.06 | 4.06 | 1.00 | sdrk fallback |
| `symbol_demap_qam64_block` | 1.55 | 4422.74 | 0.00 | 1.42 | 4747.33 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam64_block_liquid` | 1.55 | 234.67 | 0.01 | 1.42 | 212.20 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam64_block_lut` | 1.55 | 56.48 | 0.03 | 1.42 | 55.13 | 0.03 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_map_block` | 5.78 | 10.17 | 0.57 | 7.55 | 11.66 | 0.65 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_liquid` | 5.78 | 18.15 | 0.32 | 7.55 | 20.14 | 0.38 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qam64` | 7.65 | 13.29 | 0.58 | 7.14 | 15.92 | 0.45 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qam64_liquid` | 7.65 | 22.23 | 0.34 | 7.14 | 26.23 | 0.27 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qpsk` | 5.56 | 7.05 | 0.79 | 5.15 | 7.40 | 0.70 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qpsk_liquid` | 5.56 | 14.07 | 0.39 | 5.15 | 14.05 | 0.37 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_sync` | 29.31 | 29.31 | 1.00 | 32.97 | 32.97 | 1.00 | sdrk fallback |
| `symbol_sync_fallback` | 29.31 | 43.55 | 0.67 | 32.97 | 36.52 | 0.90 | GNU Radio symbol_sync_cc (symbol_sync_cc_impl.cc of gr-digital, TED_GARDNER, IR_MMSE_8TAP, generic compiler flags; reference does different work: an 8-tap MMSE interpolator against our 4-point Lagrange polynomial) |
| `symbol_sync_real` | 14.41 | 14.41 | 1.00 | 16.16 | 16.16 | 1.00 | sdrk fallback |
| `xcorr_fft_n4096` | 8.60 | 854.13 | 0.01 | 8.00 | 115.28 | 0.07 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n4096_fallback` | 69.36 | 4330.92 | 0.02 | 64.06 | 605.13 | 0.11 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |
| `xcorr_fft_n512` | 7.60 | 35.33 | 0.22 | 7.15 | 35.79 | 0.20 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n512_fallback` | 55.59 | 213.22 | 0.26 | 51.79 | 200.93 | 0.26 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |

- Not measured at this level: `real_to_complex_airspy`.

#### Simulated cycles at `avx2-runtime` (`x86_64`)

Reference model `znver3`, counter-model `alderlake`, mode `whole-loop`; execution counts: valgrind callgrind, execution counts per instruction and taken jumps.

| Entry | ours znver3 (cycles/sample) | comparison znver3 (cycles/sample) | ratio znver3 (ours/comparison) | ours alderlake (cycles/sample) | comparison alderlake (cycles/sample) | ratio alderlake (ours/comparison) | comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `convert_cq15_cs8` | 0.14 | 0.14 | 1.03 | 0.14 | 0.14 | 1.01 | ours at avx2 |
| `convert_cq15_cu8` | 0.14 | 0.14 | 1.04 | 0.14 | 0.14 | 1.01 | ours at avx2 |
| `convert_cs16` | 0.26 | 0.26 | 1.01 | 0.27 | 0.27 | 1.01 | ours at avx2 |
| `convert_cs8` | 0.27 | 0.26 | 1.01 | 0.27 | 0.27 | 1.00 | ours at avx2 |
| `convert_cu8` | 0.27 | 0.27 | 1.01 | 0.27 | 0.27 | 1.00 | ours at avx2 |
| `convert_out_cs16` | 0.45 | 0.45 | 1.01 | 0.56 | 0.55 | 1.01 | ours at avx2 |
| `convert_out_cs8` | 0.43 | 0.43 | 1.00 | 0.56 | 0.56 | 1.00 | ours at avx2 |
| `convert_out_cu8` | 0.53 | 0.53 | 0.99 | 0.66 | 0.67 | 0.99 | ours at avx2 |
| `convert_out_real_rs16` | 0.24 | 0.24 | 1.01 | 0.29 | 0.28 | 1.01 | ours at avx2 |
| `convert_out_real_rs8` | 0.23 | 0.23 | 1.01 | 0.29 | 0.29 | 1.01 | ours at avx2 |
| `convert_out_real_ru8` | 0.27 | 0.28 | 0.99 | 0.34 | 0.35 | 0.99 | ours at avx2 |
| `convert_real_rs16` | 0.14 | 0.14 | 1.02 | 0.14 | 0.14 | 1.01 | ours at avx2 |
| `rotate` | 0.67 | 0.66 | 1.01 | 0.57 | 0.57 | 1.01 | ours at avx2 |
| `rotate_cq15` | 1.18 | 1.18 | 1.00 | 1.24 | 1.25 | 0.99 | ours at avx2 |
| `scramble_block` | 3.06 | 2.85 | 1.07 | 4.56 | 3.88 | 1.18 | ours at avx2 |
| `scramble_c32_b1` | 0.41 | 0.41 | 1.02 | 0.31 | 0.30 | 1.05 | ours at avx2 |
| `scramble_c32_b2` | 0.96 | 0.95 | 1.01 | 1.05 | 1.04 | 1.01 | ours at avx2 |
| `scramble_c32_b8` | 1.32 | 1.31 | 1.01 | 1.18 | 1.17 | 1.01 | ours at avx2 |
| `scramble_cq15_b2` | 0.42 | 0.42 | 1.00 | 0.66 | 0.66 | 1.00 | ours at avx2 |
| `scramble_cq15_b8` | 1.21 | 1.21 | 1.01 | 1.13 | 1.13 | 1.01 | ours at avx2 |
| `fir` | 9.64 | 9.63 | 1.00 | 8.85 | 8.85 | 1.00 | ours at avx2 |
| `fir_cq15` | 84.10 | 40.62 | 2.07 | 98.96 | 61.15 | 1.62 | ours at avx2 |
| `fir_decimate` | 2.43 | 2.42 | 1.00 | 2.09 | 2.08 | 1.00 | ours at avx2 |
| `fir_decimate_cq15` | 18.97 | 21.61 | 0.88 | 20.01 | 22.05 | 0.91 | ours at avx2 |
| `fir_interpolate` | 12.39 | 12.44 | 1.00 | 11.40 | 11.40 | 1.00 | ours at avx2 |
| `fir_interpolate_cq15` | 97.70 | 52.45 | 1.86 | 110.63 | 71.49 | 1.55 | ours at avx2 |
| `fir_interpolate_real` | 7.15 | 7.27 | 0.98 | 6.80 | 6.78 | 1.00 | ours at avx2 |
| `fir_interpolate_real_q15` | 60.92 | 42.13 | 1.45 | 67.03 | 38.06 | 1.76 | ours at avx2 |
| `fir_interpolate_short` | 3.34 | 3.31 | 1.01 | 3.12 | 3.12 | 1.00 | ours at avx2 |
| `resample` | 15.10 | 14.92 | 1.01 | 14.19 | 14.17 | 1.00 | ours at avx2 |
| `resample_cq15` | 97.21 | 117.53 | 0.83 | 110.41 | 113.20 | 0.98 | ours at avx2 |
| `resample_real` | 9.74 | 9.82 | 0.99 | 9.18 | 9.13 | 1.01 | ours at avx2 |
| `resample_real_q15` | 60.54 | 60.80 | 1.00 | 66.87 | 66.72 | 1.00 | ours at avx2 |
| `mag` | 0.77 | 0.76 | 1.01 | 0.77 | 0.76 | 1.01 | ours at avx2 |
| `mag_q15` | 0.98 | 0.97 | 1.01 | 0.98 | 0.98 | 1.00 | ours at avx2 |
| `mag_sq` | 0.23 | 0.22 | 1.01 | 0.26 | 0.26 | 1.01 | ours at avx2 |
| `mag_sq_q15` | 0.14 | 0.14 | 1.01 | 0.18 | 0.18 | 1.01 | ours at avx2 |
| `quad_demod` | 2.61 | 2.60 | 1.00 | 2.13 | 2.12 | 1.00 | ours at avx2 |
| `quad_demod_cq15` | 4.47 | 4.40 | 1.01 | 4.11 | 4.01 | 1.02 | ours at avx2 |
| `fm_mod` | 2.17 | 2.06 | 1.05 | 2.17 | 2.13 | 1.02 | ours at avx2 |
| `fm_mod_cq15` | 2.05 | 1.93 | 1.06 | 2.21 | 2.16 | 1.02 | ours at avx2 |
| `squelch` | 9.13 | 9.13 | 1.00 | 10.14 | 10.13 | 1.00 | ours at avx2 |
| `squelch_cq15` | 6.11 | 5.56 | 1.10 | 9.09 | 6.95 | 1.31 | ours at avx2 |
| `window` | 0.34 | 0.33 | 1.01 | 0.26 | 0.26 | 1.01 | ours at avx2 |
| `window_cq15` | 0.23 | 0.22 | 1.04 | 0.22 | 0.21 | 1.04 | ours at avx2 |
| `power_db` | 1.86 | 1.85 | 1.01 | 1.88 | 1.88 | 1.00 | ours at avx2 |
| `power_db_q15` | 2.03 | 1.89 | 1.08 | 1.90 | 1.90 | 1.00 | ours at avx2 |
| `agc` | 16.08 | 13.18 | 1.22 | 21.04 | 14.78 | 1.42 | ours at avx2 |
| `agc_cq15` | 80.43 | 78.32 | 1.03 | 90.62 | 85.53 | 1.06 | ours at avx2 |
| `dc_block` | 2.80 | 2.95 | 0.95 | 2.77 | 2.76 | 1.01 | ours at avx2 |
| `dc_block_cq15` | 8.03 | 6.30 | 1.28 | 8.06 | 6.13 | 1.31 | ours at avx2 |
| `dc_block_real` | 2.09 | 2.12 | 0.99 | 1.91 | 1.88 | 1.01 | ours at avx2 |
| `dc_block_real_q15` | 5.10 | 5.62 | 0.91 | 6.11 | 5.56 | 1.10 | ours at avx2 |
| `deemphasis` | 2.05 | 2.09 | 0.98 | 1.83 | 1.84 | 1.00 | ours at avx2 |
| `deemphasis_q15` | 7.03 | 5.53 | 1.27 | 7.06 | 5.54 | 1.27 | ours at avx2 |
| `biquad` | 9.37 | 9.37 | 1.00 | 10.57 | 10.57 | 1.00 | ours at avx2 |
| `biquad_cq15` | 8.06 | 9.16 | 0.88 | 10.32 | 10.56 | 0.98 | ours at avx2 |
| `biquad_real` | 9.12 | 9.12 | 1.00 | 10.06 | 10.06 | 1.00 | ours at avx2 |
| `biquad_real_q15` | 14.39 | 8.18 | 1.76 | 15.84 | 10.21 | 1.55 | ours at avx2 |
| `dot` | 0.27 | 0.26 | 1.02 | 0.27 | 0.26 | 1.01 | ours at avx2 |
| `dot_complex` | 0.46 | 0.46 | 0.99 | 0.58 | 0.58 | 1.00 | ours at avx2 |
| `dot_cq15` | 0.31 | 0.30 | 1.01 | 0.40 | 0.40 | 1.00 | ours at avx2 |
| `dot_real` | 0.14 | 0.14 | 1.02 | 0.11 | 0.10 | 1.02 | ours at avx2 |
| `dot_real_q15` | 0.13 | 0.13 | 1.04 | 0.16 | 0.16 | 1.03 | ours at avx2 |
| `scale` | 0.26 | 0.26 | 1.02 | 0.20 | 0.18 | 1.07 | ours at avx2 |
| `scale_cq15` | 0.14 | 0.13 | 1.03 | 0.10 | 0.10 | 1.03 | ours at avx2 |
| `add` | 0.39 | 0.38 | 1.02 | 0.26 | 0.26 | 1.01 | ours at avx2 |
| `add_cq15` | 0.20 | 0.19 | 1.03 | 0.13 | 0.13 | 1.02 | ours at avx2 |
| `add_real` | 0.20 | 0.19 | 1.04 | 0.13 | 0.13 | 1.03 | ours at avx2 |
| `add_real_q15` | 0.10 | 0.10 | 1.05 | 0.07 | 0.07 | 1.04 | ours at avx2 |
| `mul` | 0.45 | 0.45 | 1.00 | 0.51 | 0.51 | 1.00 | ours at avx2 |
| `mul_cq15` | 0.51 | 0.51 | 1.00 | 0.63 | 0.63 | 1.00 | ours at avx2 |
| `mul_real` | 0.20 | 0.19 | 1.03 | 0.13 | 0.13 | 1.02 | ours at avx2 |
| `mul_real_q15` | 0.10 | 0.10 | 1.05 | 0.09 | 0.09 | 1.05 | ours at avx2 |
| `interleave` | 0.26 | 0.26 | 1.01 | 0.51 | 0.51 | 1.00 | ours at avx2 |
| `interleave_cq15` | 0.14 | 0.13 | 1.01 | 0.13 | 0.13 | 1.01 | ours at avx2 |
| `deinterleave` | 0.26 | 0.26 | 1.01 | 0.26 | 0.26 | 1.00 | ours at avx2 |
| `deinterleave_cq15` | 0.17 | 0.17 | 1.01 | 0.20 | 0.20 | 1.01 | ours at avx2 |
| `oscillate` | 1.40 | 1.40 | 1.00 | 1.45 | 1.45 | 1.00 | ours at avx2 |
| `oscillate_cq15` | 1.90 | 1.90 | 1.00 | 1.50 | 1.49 | 1.00 | ours at avx2 |
| `fft_1024` | 2.36 | 2.35 | 1.00 | 2.25 | 2.26 | 1.00 | ours at avx2 |
| `fft_4096` | 2.63 | 2.62 | 1.00 | 2.51 | 2.51 | 1.00 | ours at avx2 |
| `fft_64` | 2.00 | 1.72 | 1.16 | 2.00 | 1.85 | 1.08 | ours at avx2 |
| `fft_cq15_1024` | 2.54 | 2.47 | 1.03 | 2.80 | 2.79 | 1.00 | ours at avx2 |
| `fft_cq15_1024_default` | 2.44 | 2.31 | 1.05 | 2.38 | 2.34 | 1.01 | ours at avx2 |
| `fft_cq15_1024_full_scale` | 2.44 | 2.41 | 1.01 | 2.46 | 2.45 | 1.00 | ours at avx2 |
| `rfft_1024` | 2.17 | 2.11 | 1.03 | 2.11 | 2.08 | 1.01 | ours at avx2 |
| `am_mod` | 0.65 | 0.65 | 1.01 | 0.47 | 0.47 | 1.01 | ours at avx2 |
| `am_mod_cq15` | 0.20 | 0.20 | 1.02 | 0.21 | 0.21 | 1.02 | ours at avx2 |
| `auto_notch` | 156.14 | 164.98 | 0.95 | 189.93 | 164.85 | 1.15 | ours at avx2 |
| `band_pulse_detect_block` | 46.41 | 47.93 | 0.97 | 56.47 | 46.70 | 1.21 | ours at avx2 |
| `bit_descramble_block` | 0.59 | 0.33 | 1.79 | 0.70 | 0.32 | 2.19 | ours at avx2 |
| `bit_descramble_in_place_block` | 1.67 | 1.48 | 1.13 | 1.59 | 1.23 | 1.29 | ours at avx2 |
| `bit_scramble_block` | 1.12 | 0.77 | 1.45 | 1.18 | 0.81 | 1.46 | ours at avx2 |
| `bit_scramble_in_place_block` | 1.35 | 1.02 | 1.33 | 1.41 | 0.99 | 1.42 | ours at avx2 |
| `channel_bank_channelize` | 20.71 | 20.88 | 0.99 | 25.62 | 25.55 | 1.00 | ours at avx2 |
| `channel_bank_ddc` | 72.79 | 73.01 | 1.00 | 66.53 | 65.32 | 1.02 | ours at avx2 |
| `channel_select_block` | 0.77 | 0.76 | 1.00 | 0.59 | 0.52 | 1.15 | ours at avx2 |
| `channelize` | 11.96 | 11.83 | 1.01 | 12.63 | 12.74 | 0.99 | ours at avx2 |
| `channelize_block` | 11.96 | 12.08 | 0.99 | 12.87 | 12.81 | 1.00 | ours at avx2 |
| `correlate_block` | 7.72 | 7.60 | 1.02 | 7.14 | 7.15 | 1.00 | ours at avx2 |
| `costas` | 59.17 | 56.17 | 1.05 | 71.13 | 68.16 | 1.04 | ours at avx2 |
| `costas_qam1024` | 131.27 | 131.34 | 1.00 | 157.92 | 152.10 | 1.04 | ours at avx2 |
| `costas_qam16` | 132.30 | 124.35 | 1.06 | 159.34 | 148.39 | 1.07 | ours at avx2 |
| `costas_qam256` | 131.63 | 131.66 | 1.00 | 158.43 | 152.53 | 1.04 | ours at avx2 |
| `costas_qam64` | 132.21 | 128.23 | 1.03 | 159.23 | 154.25 | 1.03 | ours at avx2 |
| `crc16_ccitt_false` | 11.55 | 11.81 | 0.98 | 12.54 | 12.80 | 0.98 | ours at avx2 |
| `crc16_x25` | 7.06 | 7.05 | 1.00 | 8.05 | 8.05 | 1.00 | ours at avx2 |
| `crc24_adsb` | 11.05 | 11.04 | 1.00 | 12.04 | 12.04 | 1.00 | ours at avx2 |
| `crc32` | 7.06 | 7.05 | 1.00 | 8.05 | 8.05 | 1.00 | ours at avx2 |
| `equalize` | 105.10 | 111.94 | 0.94 | 100.09 | 152.23 | 0.66 | ours at avx2 |
| `equalize_qam1024` | 134.21 | 142.06 | 0.94 | 116.33 | 174.29 | 0.67 | ours at avx2 |
| `equalize_qam16` | 129.16 | 119.94 | 1.08 | 111.96 | 164.27 | 0.68 | ours at avx2 |
| `equalize_qam256` | 134.21 | 142.06 | 0.94 | 116.33 | 174.29 | 0.67 | ours at avx2 |
| `equalize_qam64` | 129.16 | 123.94 | 1.04 | 112.47 | 170.30 | 0.66 | ours at avx2 |
| `frame_decode` | 6.92 | 5.57 | 1.24 | 6.22 | 6.18 | 1.01 | ours at avx2 |
| `frame_decode_block` | 6.90 | 6.91 | 1.00 | 5.79 | 5.80 | 1.00 | ours at avx2 |
| `frame_decode_four_block` | 23.39 | 23.43 | 1.00 | 21.19 | 20.77 | 1.02 | ours at avx2 |
| `frame_encode_block` | 1.61 | 1.64 | 0.98 | 1.69 | 1.73 | 0.98 | ours at avx2 |
| `gauss_noise_c32` | 3.42 | 3.42 | 1.00 | 4.97 | 4.97 | 1.00 | ours at avx2 |
| `gauss_noise_f32` | 1.72 | 1.72 | 1.00 | 2.50 | 2.49 | 1.00 | ours at avx2 |
| `gauss_noise_q15` | 2.18 | 2.33 | 0.93 | 2.97 | 3.02 | 0.98 | ours at avx2 |
| `halfband_decimate` | 3.43 | 3.43 | 1.00 | 2.98 | 2.97 | 1.00 | ours at avx2 |
| `halfband_decimate_cq15` | 6.55 | 6.56 | 1.00 | 7.11 | 7.11 | 1.00 | ours at avx2 |
| `halfband_decimate_real` | 2.00 | 1.81 | 1.10 | 1.34 | 1.36 | 0.99 | ours at avx2 |
| `halfband_decimate_real_q15` | 2.98 | 2.94 | 1.01 | 3.84 | 3.82 | 1.01 | ours at avx2 |
| `iq_offset_block` | 2.02 | 2.89 | 0.70 | 2.28 | 2.95 | 0.77 | ours at avx2 |
| `morse_decode_block` | 13.35 | 12.77 | 1.05 | 16.29 | 16.08 | 1.01 | ours at avx2 |
| `morse_encode_block` | 5.27 | 4.44 | 1.19 | 5.43 | 4.49 | 1.21 | ours at avx2 |
| `noise_blank` | 8.09 | 11.58 | 0.70 | 12.46 | 11.37 | 1.10 | ours at avx2 |
| `noise_blank_cq15` | 8.51 | 8.48 | 1.00 | 9.31 | 9.30 | 1.00 | ours at avx2 |
| `noise_block` | 3.42 | 3.42 | 1.00 | 4.97 | 4.97 | 1.00 | ours at avx2 |
| `preemph_limit` | 16.03 | 15.37 | 1.04 | 15.16 | 13.18 | 1.15 | ours at avx2 |
| `preemph_limit_q15` | 28.42 | 27.62 | 1.03 | 18.69 | 18.69 | 1.00 | ours at avx2 |
| `preemph_limit_stereo` | 19.59 | 16.71 | 1.17 | 22.55 | 22.32 | 1.01 | ours at avx2 |
| `psk_demod_block` | 37.55 | 37.72 | 1.00 | 38.85 | 38.80 | 1.00 | ours at avx2 |
| `psk_mod_block` | 4.40 | 4.37 | 1.01 | 4.23 | 4.27 | 0.99 | ours at avx2 |
| `psk_mod_block_liquid` | 4.40 | 4.37 | 1.01 | 4.23 | 4.27 | 0.99 | ours at avx2 |
| `pulse_match_block` | 12.40 | 12.32 | 1.01 | 11.32 | 11.29 | 1.00 | ours at avx2 |
| `pulse_match_block_liquid` | 12.40 | 12.32 | 1.01 | 11.32 | 11.29 | 1.00 | ours at avx2 |
| `pulse_shape_block` | 3.59 | 3.57 | 1.01 | 3.51 | 3.54 | 0.99 | ours at avx2 |
| `pulse_shape_block_liquid` | 3.59 | 3.57 | 1.01 | 3.51 | 3.54 | 0.99 | ours at avx2 |
| `qam_acquire_block` | 26.55 | 26.92 | 0.99 | 33.04 | 32.63 | 1.01 | ours at avx2 |
| `qam_demod_block` | 64.42 | 62.75 | 1.03 | 65.10 | 70.69 | 0.92 | ours at avx2 |
| `qam_demod_qam256_block` | 65.05 | 66.52 | 0.98 | 65.71 | 72.62 | 0.90 | ours at avx2 |
| `qam_demod_search_block` | 68.30 | 65.65 | 1.04 | 69.82 | 74.37 | 0.94 | ours at avx2 |
| `qam_frame_decode_block` | 76.50 | 74.73 | 1.02 | 76.11 | 81.50 | 0.93 | ours at avx2 |
| `qam_frame_encode_block` | 5.17 | 5.26 | 0.98 | 5.28 | 5.48 | 0.96 | ours at avx2 |
| `qam_mod_block` | 4.31 | 4.39 | 0.98 | 4.39 | 4.57 | 0.96 | ours at avx2 |
| `qam_mod_block_liquid` | 4.31 | 4.39 | 0.98 | 4.39 | 4.57 | 0.96 | ours at avx2 |
| `qam_mod_qam256_block` | 4.89 | 4.96 | 0.99 | 5.12 | 4.74 | 1.08 | ours at avx2 |
| `qam_mod_qam256_block_liquid` | 4.89 | 4.96 | 0.99 | 5.12 | 4.74 | 1.08 | ours at avx2 |
| `qam_soft_bits_qam1024` | 4.67 | 5.53 | 0.84 | 4.89 | 4.31 | 1.13 | ours at avx2 |
| `qam_soft_bits_qam16` | 0.52 | 0.52 | 1.00 | 1.02 | 1.02 | 1.00 | ours at avx2 |
| `qam_soft_bits_qam16_liquid` | 0.52 | 0.52 | 1.00 | 1.02 | 1.02 | 1.00 | ours at avx2 |
| `qam_soft_bits_qam16_lut` | 0.52 | 0.52 | 1.00 | 1.02 | 1.02 | 1.00 | ours at avx2 |
| `qam_soft_bits_qam256` | 2.51 | 2.51 | 1.00 | 4.06 | 4.06 | 1.00 | ours at avx2 |
| `qam_soft_bits_qam64` | 1.55 | 1.55 | 1.00 | 1.42 | 1.42 | 1.00 | ours at avx2 |
| `qam_soft_bits_qam64_liquid` | 1.55 | 1.55 | 1.00 | 1.42 | 1.42 | 1.00 | ours at avx2 |
| `qam_soft_bits_qam64_lut` | 1.55 | 1.55 | 1.00 | 1.42 | 1.42 | 1.00 | ours at avx2 |
| `qam_soft_bits_qpsk` | 0.27 | 0.26 | 1.02 | 0.20 | 0.20 | 1.01 | ours at avx2 |
| `rds_decode_block` | 7.77 | 7.28 | 1.07 | 8.56 | 7.98 | 1.07 | ours at avx2 |
| `rds_demod_block` | 16.79 | 16.60 | 1.01 | 14.71 | 14.43 | 1.02 | ours at avx2 |
| `rds_shape` | 47.87 | 50.41 | 0.95 | 69.18 | 78.60 | 0.88 | ours at avx2 |
| `real_to_complex` | 2.55 | 2.49 | 1.02 | 2.30 | 2.27 | 1.01 | ours at avx2 |
| `real_to_complex_cq15` | 3.63 | 3.60 | 1.01 | 4.93 | 4.87 | 1.01 | ours at avx2 |
| `snr_estimate_block` | 4.64 | 2.70 | 1.72 | 5.03 | 2.78 | 1.81 | ours at avx2 |
| `snr_estimate_block_svr` | 4.64 | 2.70 | 1.72 | 5.03 | 2.78 | 1.81 | ours at avx2 |
| `stereo_encode` | 7.65 | 4.47 | 1.71 | 23.10 | 5.92 | 3.90 | ours at avx2 |
| `stereo_encode_rds` | 12.50 | 7.17 | 1.74 | 30.19 | 8.91 | 3.39 | ours at avx2 |
| `stereo_matrix` | 0.33 | 0.32 | 1.01 | 0.45 | 0.45 | 1.01 | ours at avx2 |
| `stereo_matrix_cq15` | 0.15 | 0.15 | 1.03 | 0.20 | 0.20 | 1.02 | ours at avx2 |
| `symbol_demap_qam16_block` | 0.52 | 0.52 | 1.00 | 1.02 | 1.02 | 1.00 | ours at avx2 |
| `symbol_demap_qam16_block_liquid` | 0.52 | 0.52 | 1.00 | 1.02 | 1.02 | 1.00 | ours at avx2 |
| `symbol_demap_qam16_block_lut` | 0.52 | 0.52 | 1.00 | 1.02 | 1.02 | 1.00 | ours at avx2 |
| `symbol_demap_qam256_block` | 2.51 | 2.51 | 1.00 | 4.06 | 4.06 | 1.00 | ours at avx2 |
| `symbol_demap_qam64_block` | 1.55 | 1.55 | 1.00 | 1.43 | 1.42 | 1.00 | ours at avx2 |
| `symbol_demap_qam64_block_liquid` | 1.55 | 1.55 | 1.00 | 1.43 | 1.42 | 1.00 | ours at avx2 |
| `symbol_demap_qam64_block_lut` | 1.55 | 1.55 | 1.00 | 1.43 | 1.42 | 1.00 | ours at avx2 |
| `symbol_map_block` | 4.93 | 5.78 | 0.85 | 6.38 | 7.55 | 0.84 | ours at avx2 |
| `symbol_map_block_liquid` | 4.93 | 5.78 | 0.85 | 6.38 | 7.55 | 0.84 | ours at avx2 |
| `symbol_map_block_qam64` | 7.15 | 7.65 | 0.94 | 9.20 | 7.14 | 1.29 | ours at avx2 |
| `symbol_map_block_qam64_liquid` | 7.15 | 7.65 | 0.94 | 9.20 | 7.14 | 1.29 | ours at avx2 |
| `symbol_map_block_qpsk` | 5.56 | 5.56 | 1.00 | 5.05 | 5.15 | 0.98 | ours at avx2 |
| `symbol_map_block_qpsk_liquid` | 5.56 | 5.56 | 1.00 | 5.05 | 5.15 | 0.98 | ours at avx2 |
| `symbol_sync` | 27.35 | 29.31 | 0.93 | 31.75 | 32.97 | 0.96 | ours at avx2 |
| `symbol_sync_real` | 13.53 | 14.41 | 0.94 | 17.66 | 16.16 | 1.09 | ours at avx2 |
| `xcorr_fft_n4096` | 8.66 | 8.60 | 1.01 | 8.03 | 8.00 | 1.00 | ours at avx2 |
| `xcorr_fft_n512` | 7.72 | 7.60 | 1.02 | 7.14 | 7.15 | 1.00 | ours at avx2 |

- Not measured at this level: `convert_cq15_cs8_fallback`, `convert_cq15_cu8_fallback`, `convert_cs16_fallback`, `convert_cs8_fallback`, `convert_cu8_fallback`, `convert_out_cs16_fallback`, `convert_out_cs8_fallback`, `convert_out_cu8_fallback`, `convert_out_real_rs16_fallback`, `convert_out_real_rs8_fallback`, `convert_out_real_ru8_fallback`, `convert_real_rs16_fallback`, `rotate_cq15_fallback`, `rotate_fallback`, `scramble_c32_b1_fallback`, `scramble_c32_b2_dvbs2`, `scramble_c32_b2_fallback`, `scramble_c32_b8_fallback`, `scramble_cq15_b2_fallback`, `scramble_cq15_b8_fallback`, `fir_cq15_fallback`, `fir_decimate_cq15_fallback`, `fir_decimate_fallback`, `fir_fallback`, `fir_interpolate_cq15_fallback`, `fir_interpolate_fallback`, `fir_interpolate_real_fallback`, `fir_interpolate_real_q15_fallback`, `fir_interpolate_short_fallback`, `resample_cq15_fallback`, `resample_fallback`, `resample_real_fallback`, `resample_real_q15_fallback`, `mag_fallback`, `mag_q15_fallback`, `mag_sq_fallback`, `mag_sq_q15_fallback`, `quad_demod_cq15_fallback`, `quad_demod_fallback`, `fm_mod_cq15_fallback`, `fm_mod_fallback`, `window_cq15_fallback`, `window_fallback`, `power_db_fallback`, `power_db_q15_fallback`, `agc_fallback`, `dc_block_fallback`, `dc_block_real_fallback`, `deemphasis_fallback`, `biquad_cq15_fallback`, `biquad_fallback`, `biquad_real_fallback`, `biquad_real_q15_fallback`, `dot_complex_fallback`, `dot_cq15_fallback`, `dot_fallback`, `dot_real_fallback`, `dot_real_q15_fallback`, `scale_cq15_fallback`, `scale_fallback`, `add_cq15_fallback`, `add_fallback`, `add_real_fallback`, `add_real_q15_fallback`, `mul_cq15_fallback`, `mul_fallback`, `mul_real_fallback`, `mul_real_q15_fallback`, `interleave_cq15_fallback`, `interleave_fallback`, `deinterleave_cq15_fallback`, `deinterleave_fallback`, `oscillate_cq15_fallback`, `oscillate_fallback`, `fft_1024_fallback`, `fft_4096_fallback`, `fft_64_fallback`, `fft_cq15_1024_default_fallback`, `fft_cq15_1024_fallback`, `rfft_1024_fallback`, `am_mod_cq15_fallback`, `am_mod_fallback`, `channelize_fallback`, `costas_fallback`, `costas_qam1024_fallback`, `costas_qam16_fallback`, `costas_qam256_fallback`, `costas_qam64_fallback`, `equalize_fallback`, `equalize_qam1024_fallback`, `equalize_qam16_fallback`, `equalize_qam256_fallback`, `equalize_qam64_fallback`, `gauss_noise_c32_fallback`, `gauss_noise_c32_fastnoise`, `gauss_noise_f32_fallback`, `gauss_noise_f32_fastnoise`, `gauss_noise_f32_ziggurat`, `gauss_noise_q15_fallback`, `halfband_decimate_cq15_fallback`, `halfband_decimate_fallback`, `halfband_decimate_real_fallback`, `halfband_decimate_real_q15_fallback`, `halfband_ref_fir_decimate`, `halfband_ref_fir_decimate_cq15`, `halfband_ref_shifted`, `halfband_ref_shifted_cq15`, `real_to_complex_airspy`, `real_to_complex_cq15_fallback`, `real_to_complex_fallback`, `stereo_matrix_cq15_fallback`, `stereo_matrix_fallback`, `stereo_pll_fallback`, `stereo_pll_q15_fallback`, `symbol_sync_fallback`, `xcorr_fft_n4096_fallback`, `xcorr_fft_n512_fallback`.

#### Simulated cycles at `avx512` (`x86_64-avx512`)

Reference model `znver4`, counter-model `sapphirerapids`, mode `whole-loop`; execution counts: Intel SDE -dcfg, block and branch counts (Intel SDE -spr).

| Entry | ours znver4 (cycles/sample) | comparison znver4 (cycles/sample) | ratio znver4 (ours/comparison) | ours sapphirerapids (cycles/sample) | comparison sapphirerapids (cycles/sample) | ratio sapphirerapids (ours/comparison) | comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `convert_cq15_cs8` | 0.09 | 0.14 | 0.66 | 0.07 | 0.18 | 0.41 | volk_8i_convert_16i |
| `convert_cq15_cs8_fallback` | 0.14 | 0.13 | 1.02 | 0.14 | 0.20 | 0.69 | CMSIS-DSP 1.18.0 arm_q7_to_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_cq15_cu8` | 0.10 | 0.14 | 0.71 | 0.11 | 0.18 | 0.61 | volk_8i_convert_16i on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cq15_cu8_fallback` | 0.14 | 0.13 | 1.02 | 0.15 | 0.20 | 0.74 | CMSIS-DSP 1.18.0 arm_q7_to_q15 on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_cs16` | 0.24 | 0.25 | 1.00 | 0.21 | 0.22 | 0.94 | volk_16i_s32f_convert_32f |
| `convert_cs16_fallback` | 0.27 | 0.28 | 0.93 | 0.29 | 0.40 | 0.72 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_cs8` | 0.25 | 0.25 | 1.00 | 0.21 | 0.22 | 0.94 | volk_8i_s32f_convert_32f |
| `convert_cs8_fallback` | 0.27 | 0.34 | 0.79 | 0.29 | 0.58 | 0.50 | CMSIS-DSP 1.18.0 arm_q7_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_cu8` | 0.25 | 0.25 | 1.00 | 0.21 | 0.22 | 0.94 | volk_8i_s32f_convert_32f on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cu8_fallback` | 0.28 | 0.34 | 0.83 | 0.37 | 0.58 | 0.64 | CMSIS-DSP 1.18.0 arm_q7_to_float on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_out_cs16` | 0.39 | 0.42 | 0.94 | 0.39 | 0.40 | 0.97 | volk_32f_s32f_convert_16i |
| `convert_out_cs16_fallback` | 0.64 | 0.41 | 1.58 | 0.61 | 0.59 | 1.03 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_out_cs8` | 0.39 | 0.43 | 0.91 | 0.39 | 0.39 | 1.00 | volk_32f_s32f_convert_8i |
| `convert_out_cs8_fallback` | 0.73 | 0.38 | 1.92 | 0.84 | 0.61 | 1.38 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_out_cu8` | 0.62 | 0.51 | 1.20 | 0.52 | 0.59 | 0.88 | volk_32f_s32f_x2_convert_8u |
| `convert_out_cu8_fallback` | 1.22 | 0.38 | 3.20 | 1.33 | 0.61 | 2.16 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_out_real_rs16` | 0.20 | 0.21 | 0.95 | 0.20 | 0.20 | 1.00 | volk_32f_s32f_convert_16i |
| `convert_out_real_rs16_fallback` | 0.33 | 0.21 | 1.58 | 0.31 | 0.30 | 1.05 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_out_real_rs8` | 0.21 | 0.22 | 0.93 | 0.21 | 0.20 | 1.04 | volk_32f_s32f_convert_8i |
| `convert_out_real_rs8_fallback` | 0.37 | 0.19 | 1.92 | 0.43 | 0.31 | 1.39 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_out_real_ru8` | 0.32 | 0.26 | 1.21 | 0.27 | 0.30 | 0.90 | volk_32f_s32f_x2_convert_8u |
| `convert_out_real_ru8_fallback` | 0.61 | 0.19 | 3.17 | 0.67 | 0.31 | 2.17 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_real_rs16` | 0.13 | 0.13 | 1.01 | 0.11 | 0.11 | 0.96 | volk_16i_s32f_convert_32f |
| `convert_real_rs16_fallback` | 0.14 | 0.15 | 0.94 | 0.15 | 0.20 | 0.74 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `rotate` | 0.71 | 3.89 | 0.18 | 0.44 | 3.36 | 0.13 | volk_32fc_s32fc_x2_rotator2_32fc |
| `rotate_cq15` | 0.79 | 4.59 | 0.17 | 3.22 | 5.64 | 0.57 | sdrk fallback |
| `rotate_cq15_fallback` | 4.59 | 27.49 | 0.17 | 5.64 | 25.81 | 0.22 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `rotate_fallback` | 1.05 | 10.63 | 0.10 | 9.30 | 11.17 | 0.83 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `scramble_block` | 2.10 | 2.10 | 1.00 | 5.36 | 5.36 | 1.00 | sdrk fallback |
| `scramble_c32_b1` | 0.23 | 0.66 | 0.35 | 0.15 | 1.59 | 0.09 | +-1.0 sequence from the index bits into a buffer of 256 + volk_32fc_32f_multiply_32fc (VOLK 3.3.0 dispatcher; the form of srsRAN srsran_scrambling_c) |
| `scramble_c32_b1_fallback` | 0.51 | 0.70 | 0.73 | 0.52 | 1.42 | 0.36 | +-1.0 sequence from the index bits into a buffer of 256 + VOLK 3.3.0 volk_32fc_32f_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2` | 0.64 | 2.36 | 0.27 | 0.78 | 2.41 | 0.32 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2_dvbs2` | 0.64 | 2.98 | 0.21 | 0.78 | 2.42 | 0.32 | GNU Radio dvbs2_physical_cc PL scrambling, the per-symbol switch on R_n (benches/refs/gr-dvbs2-plscramble.c, main at ec1bf15; reference does different work; -O3) |
| `scramble_c32_b2_fallback` | 1.49 | 6.24 | 0.24 | 1.30 | 7.00 | 0.19 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8` | 0.58 | 2.36 | 0.25 | 2.66 | 2.41 | 1.11 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8_fallback` | 0.95 | 6.24 | 0.15 | 3.92 | 7.00 | 0.56 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_cq15_b2` | 0.40 | 2.92 | 0.14 | 0.41 | 3.56 | 0.12 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `scramble_cq15_b2_fallback` | 0.67 | 2.92 | 0.23 | 0.85 | 3.56 | 0.24 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `scramble_cq15_b8` | 0.49 | 2.92 | 0.17 | 2.43 | 3.56 | 0.68 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `scramble_cq15_b8_fallback` | 4.04 | 2.92 | 1.38 | 3.95 | 3.56 | 1.11 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir` | 8.65 | 63.03 | 0.14 | 4.82 | 97.97 | 0.05 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 40.60 | 40.60 | 1.00 | 45.49 | 45.49 | 1.00 | sdrk fallback |
| `fir_cq15_fallback` | 40.60 | 144.59 | 0.28 | 45.49 | 154.66 | 0.29 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_decimate` | 2.05 | 8.83 | 0.23 | 1.61 | 15.25 | 0.11 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 11.57 | 16.80 | 0.69 | 11.17 | 18.22 | 0.61 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 16.80 | 37.21 | 0.45 | 18.22 | 20.44 | 0.89 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_decimate_fallback` | 11.12 | 71.32 | 0.16 | 10.28 | 38.89 | 0.26 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_fallback` | 31.12 | 51.41 | 0.61 | 30.50 | 80.41 | 0.38 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate` | 11.16 | 71.37 | 0.16 | 7.51 | 102.01 | 0.07 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 54.34 | 54.34 | 1.00 | 63.17 | 63.17 | 1.00 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 54.34 | 136.84 | 0.40 | 63.17 | 136.97 | 0.46 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate_fallback` | 39.43 | 101.17 | 0.39 | 38.06 | 100.78 | 0.38 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate_real` | 6.69 | 81.93 | 0.08 | 4.79 | 118.66 | 0.04 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 25.47 | 50.59 | 0.50 | 24.02 | 50.39 | 0.48 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate_real_q15` | 22.72 | 22.72 | 1.00 | 29.27 | 29.27 | 1.00 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 22.72 | 68.42 | 0.33 | 29.27 | 68.49 | 0.43 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate_short` | 3.63 | 79.65 | 0.05 | 2.68 | 85.17 | 0.03 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_short_fallback` | 9.75 | 18.21 | 0.54 | 9.87 | 17.74 | 0.56 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 8 (80 taps), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `resample` | 13.83 | 71.87 | 0.19 | 12.45 | 102.76 | 0.12 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 66.76 | 84.05 | 0.79 | 69.75 | 89.10 | 0.78 | sdrk fallback |
| `resample_cq15_fallback` | 84.05 | 136.08 | 0.62 | 89.10 | 136.22 | 0.65 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `resample_fallback` | 60.87 | 100.42 | 0.61 | 55.19 | 100.10 | 0.55 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `resample_real` | 9.25 | 82.64 | 0.11 | 8.96 | 119.58 | 0.07 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 40.67 | 50.21 | 0.81 | 243.27 | 50.05 | 4.86 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `resample_real_q15` | 33.26 | 100.33 | 0.33 | 36.27 | 85.43 | 0.42 | sdrk fallback |
| `resample_real_q15_fallback` | 100.33 | 68.04 | 1.47 | 85.43 | 68.11 | 1.25 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `mag` | 0.64 | 1.28 | 0.50 | 0.76 | 1.54 | 0.49 | volk_32fc_magnitude_32f |
| `mag_fallback` | 0.64 | 0.66 | 0.97 | 0.76 | 0.78 | 0.97 | CMSIS-DSP 1.18.0 arm_cmplx_mag_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `mag_q15` | 0.72 | 0.84 | 0.85 | 0.93 | 0.92 | 1.01 | volk_16ic_magnitude_16i |
| `mag_q15_fallback` | 0.96 | 78.55 | 0.01 | 1.11 | 83.02 | 0.01 | CMSIS-DSP 1.18.0 arm_cmplx_mag_q15 (Q2.14, arm_sqrt_q31), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `mag_sq` | 0.29 | 0.33 | 0.88 | 0.16 | 0.52 | 0.31 | volk_32fc_magnitude_squared_32f |
| `mag_sq_fallback` | 0.37 | 0.31 | 1.21 | 0.26 | 0.27 | 0.94 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `mag_sq_q15` | 0.18 | 0.19 | 0.92 | 0.14 | 0.19 | 0.71 | sdrk fallback |
| `mag_sq_q15_fallback` | 0.19 | 0.90 | 0.22 | 0.19 | 1.87 | 0.10 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_q15 (Q3.13, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `quad_demod` | 1.64 | 2.11 | 0.77 | 1.28 | 2.11 | 0.61 | volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f |
| `quad_demod_cq15` | 2.29 | 77.19 | 0.03 | 2.68 | 76.98 | 0.03 | sdrk fallback |
| `quad_demod_cq15_fallback` | 77.19 | 81.69 | 0.94 | 76.98 | 65.14 | 1.18 | CMSIS-DSP arm_cmplx_mult_cmplx_q15 + arm_atan2_q15 + arm_scale_q15 (generic C), conjugate in a plain loop |
| `quad_demod_fallback` | 41.29 | 868.77 | 0.05 | 43.08 | 130.25 | 0.33 | GNU Radio quadrature_demod_cf (gr-analog, fast_atan2f, generic compiler flags) |
| `fm_mod` | 1.37 | 56.04 | 0.02 | 2.11 | 25.57 | 0.08 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `fm_mod_cq15` | 1.31 | 7.60 | 0.17 | 1.48 | 7.84 | 0.19 | sdrk fallback |
| `fm_mod_cq15_fallback` | 7.60 | 21.56 | 0.35 | 7.84 | 19.58 | 0.40 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample of an integer phase accumulator, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fm_mod_fallback` | 2.56 | 56.04 | 0.05 | 5.74 | 25.57 | 0.22 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `squelch` | 9.13 | 18.42 | 0.50 | 10.14 | 15.26 | 0.66 | GNU Radio pwr_squelch_cc (squelch_base_cc_impl.cc of gr-analog, generic compiler flags) |
| `squelch_cq15` | 5.56 | 5.56 | 1.00 | 6.95 | 6.95 | 1.00 | sdrk fallback |
| `window` | 0.20 | 0.34 | 0.60 | 0.15 | 0.53 | 0.28 | volk_32fc_32f_multiply_32fc |
| `window_cq15` | 0.13 | 0.48 | 0.28 | 0.22 | 0.55 | 0.39 | sdrk fallback |
| `window_cq15_fallback` | 0.48 | 0.65 | 0.74 | 0.55 | 0.98 | 0.56 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `window_fallback` | 0.28 | 0.41 | 0.69 | 0.16 | 0.38 | 0.42 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `power_db` | 1.56 | 1.01 | 1.54 | 1.41 | 0.97 | 1.46 | volk_32f_log2_32f + volk_32f_s32f_multiply_32f + volk_32f_s32f_add_32f |
| `power_db_fallback` | 2.42 | 40.37 | 0.06 | 2.05 | 46.31 | 0.04 | CMSIS-DSP 1.18.0 arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `power_db_q15` | 1.48 | 2.98 | 0.50 | 1.46 | 2.32 | 0.63 | sdrk fallback |
| `power_db_q15_fallback` | 2.98 | 40.52 | 0.07 | 2.32 | 46.52 | 0.05 | CMSIS-DSP 1.18.0 arm_q15_to_float + arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `agc` | 13.08 | 36.12 | 0.36 | 15.28 | 39.15 | 0.39 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `agc_cq15` | 82.44 | 82.44 | 1.00 | 87.71 | 87.71 | 1.00 | sdrk fallback |
| `agc_fallback` | 18.34 | 36.12 | 0.51 | 18.42 | 39.15 | 0.47 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `dc_block` | 2.65 | 9.03 | 0.29 | 3.19 | 11.04 | 0.29 | sdrk fallback |
| `dc_block_cq15` | 8.27 | 8.27 | 1.00 | 8.03 | 8.03 | 1.00 | sdrk fallback |
| `dc_block_fallback` | 9.03 | 43.11 | 0.21 | 11.04 | 39.57 | 0.28 | liquid-dsp 1.8.3 iirfilt_crcf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real` | 2.26 | 7.03 | 0.32 | 2.41 | 8.04 | 0.30 | sdrk fallback |
| `dc_block_real_fallback` | 7.03 | 33.06 | 0.21 | 8.04 | 34.31 | 0.23 | liquid-dsp 1.8.3 iirfilt_rrrf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real_q15` | 5.36 | 5.36 | 1.00 | 5.46 | 5.46 | 1.00 | sdrk fallback |
| `deemphasis` | 2.24 | 10.21 | 0.22 | 2.15 | 11.11 | 0.19 | sdrk fallback |
| `deemphasis_fallback` | 10.21 | 10.80 | 0.95 | 11.11 | 23.14 | 0.48 | GNU Radio fm_deemph: iir_filter_ffd with the bilinear taps of fm_emph.py (double taps and accumulator, generic compiler flags) |
| `deemphasis_q15` | 5.76 | 5.76 | 1.00 | 5.53 | 5.53 | 1.00 | sdrk fallback |
| `biquad` | 9.44 | 24.92 | 0.38 | 10.57 | 28.90 | 0.37 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (avx512, -ffp-contract=off), two calls on planar re and im (deinterleaving not counted) |
| `biquad_cq15` | 9.12 | 21.05 | 0.43 | 10.85 | 20.25 | 0.54 | sdrk fallback |
| `biquad_cq15_fallback` | 21.05 | 66.14 | 0.32 | 20.25 | 76.17 | 0.27 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `biquad_fallback` | 18.46 | 24.04 | 0.77 | 20.71 | 26.04 | 0.80 | CMSIS-DSP 1.18.0 arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `biquad_real` | 9.31 | 12.46 | 0.75 | 10.32 | 14.45 | 0.71 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (avx512, -ffp-contract=off) |
| `biquad_real_fallback` | 9.69 | 18.04 | 0.54 | 11.46 | 20.05 | 0.57 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df2T_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `biquad_real_q15` | 9.04 | 12.08 | 0.75 | 10.62 | 13.08 | 0.81 | sdrk fallback |
| `biquad_real_q15_fallback` | 12.08 | 33.07 | 0.37 | 13.08 | 38.08 | 0.34 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `dot` | 0.20 | 0.27 | 0.75 | 0.21 | 0.77 | 0.27 | volk_32fc_32f_dot_prod_32fc |
| `dot_complex` | 0.45 | 0.78 | 0.58 | 0.58 | 0.78 | 0.74 | VOLK 3.3.0 volk_32fc_x2_dot_prod_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `dot_complex_fallback` | 1.52 | 6.03 | 0.25 | 1.69 | 6.03 | 0.28 | CMSIS-DSP 1.18.0 arm_cmplx_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `dot_cq15` | 0.38 | 0.58 | 0.66 | 0.35 | 0.79 | 0.45 | sdrk fallback |
| `dot_cq15_fallback` | 0.58 | 0.65 | 0.89 | 0.79 | 1.28 | 0.61 | CMSIS-DSP 1.18.0 arm_dot_prod_q15 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `dot_fallback` | 3.03 | 6.07 | 0.50 | 3.03 | 6.07 | 0.50 | CMSIS-DSP 1.18.0 arm_dot_prod_f32 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `dot_real` | 0.11 | 0.27 | 0.40 | 0.11 | 0.77 | 0.15 | volk_32f_x2_dot_prod_32f |
| `dot_real_fallback` | 3.03 | 3.04 | 1.00 | 3.03 | 3.04 | 1.00 | CMSIS-DSP 1.18.0 arm_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `dot_real_q15` | 0.13 | 0.32 | 0.40 | 0.15 | 0.38 | 0.39 | sdrk fallback |
| `dot_real_q15_fallback` | 0.32 | 0.32 | 0.99 | 0.38 | 0.64 | 0.59 | CMSIS-DSP 1.18.0 arm_dot_prod_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `scale` | 0.13 | 0.27 | 0.48 | 0.13 | 0.29 | 0.45 | volk_32f_s32f_multiply_32f |
| `scale_cq15` | 0.07 | 0.36 | 0.19 | 0.07 | 0.39 | 0.17 | sdrk fallback |
| `scale_cq15_fallback` | 0.36 | 0.59 | 0.61 | 0.39 | 1.04 | 0.38 | CMSIS-DSP 1.18.0 arm_scale_q15 (shift 0, truncating) on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `scale_fallback` | 0.13 | 0.27 | 0.49 | 0.14 | 0.29 | 0.47 | CMSIS-DSP 1.18.0 arm_scale_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `add` | 0.19 | 0.40 | 0.48 | 0.13 | 0.35 | 0.37 | volk_32fc_x2_add_32fc |
| `add_cq15` | 0.11 | 0.20 | 0.54 | 0.07 | 0.18 | 0.38 | volk_16i_x2_add_saturated_16i on the 2 n values |
| `add_cq15_fallback` | 0.11 | 0.52 | 0.21 | 0.09 | 1.03 | 0.09 | CMSIS-DSP 1.18.0 arm_add_q15 on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `add_fallback` | 0.20 | 0.40 | 0.51 | 0.18 | 0.35 | 0.50 | CMSIS-DSP 1.18.0 arm_add_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `add_real` | 0.10 | 0.11 | 0.91 | 0.07 | 0.09 | 0.73 | volk_32f_x2_add_32f |
| `add_real_fallback` | 0.19 | 0.20 | 0.96 | 0.14 | 0.18 | 0.77 | CMSIS-DSP 1.18.0 arm_add_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `add_real_q15` | 0.06 | 0.10 | 0.53 | 0.04 | 0.09 | 0.39 | volk_16i_x2_add_saturated_16i |
| `add_real_q15_fallback` | 0.10 | 0.26 | 0.37 | 0.07 | 0.52 | 0.13 | CMSIS-DSP 1.18.0 arm_add_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `mul` | 0.58 | 0.49 | 1.18 | 0.39 | 0.54 | 0.71 | volk_32fc_x2_multiply_32fc |
| `mul_cq15` | 0.70 | 0.81 | 0.86 | 0.59 | 1.08 | 0.55 | sdrk fallback |
| `mul_cq15_fallback` | 0.81 | 1.06 | 0.77 | 1.08 | 1.66 | 0.65 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_q15 (Q3.13, truncating, scaled by 4), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `mul_fallback` | 0.77 | 0.53 | 1.47 | 0.78 | 0.79 | 0.98 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `mul_real` | 0.10 | 0.12 | 0.84 | 0.07 | 0.09 | 0.71 | volk_32f_x2_multiply_32f |
| `mul_real_fallback` | 0.19 | 0.20 | 0.96 | 0.14 | 0.18 | 0.76 | CMSIS-DSP 1.18.0 arm_mult_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `mul_real_q15` | 0.07 | 0.24 | 0.29 | 0.05 | 0.26 | 0.21 | sdrk fallback |
| `mul_real_q15_fallback` | 0.24 | 0.27 | 0.89 | 0.26 | 0.39 | 0.65 | CMSIS-DSP 1.18.0 arm_mult_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `interleave` | 0.20 | 0.27 | 0.74 | 0.13 | 0.52 | 0.25 | volk_32f_x2_interleave_32fc |
| `interleave_cq15` | 0.10 | 0.13 | 0.78 | 0.13 | 0.13 | 1.00 | sdrk fallback |
| `interleave_cq15_fallback` | 0.13 | 0.14 | 0.97 | 0.13 | 0.26 | 0.51 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic on int16 (VOLK has no 16-bit interleave), benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx512) |
| `interleave_fallback` | 0.26 | 0.27 | 0.97 | 0.14 | 0.27 | 0.50 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx512) |
| `deinterleave` | 0.20 | 0.27 | 0.73 | 0.13 | 0.27 | 0.49 | volk_32fc_deinterleave_32f_x2 |
| `deinterleave_cq15` | 0.10 | 0.33 | 0.32 | 0.13 | 0.26 | 0.50 | volk_16ic_deinterleave_16i_x2 |
| `deinterleave_cq15_fallback` | 0.13 | 0.20 | 0.68 | 0.26 | 0.27 | 0.97 | VOLK 3.3.0 volk_16ic_deinterleave_16i_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx512) |
| `deinterleave_fallback` | 0.26 | 0.39 | 0.67 | 0.26 | 0.27 | 0.96 | VOLK 3.3.0 volk_32fc_deinterleave_32f_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx512) |
| `oscillate` | 0.43 | 0.76 | 0.57 | 2.08 | 3.79 | 0.55 | sdrk fallback |
| `oscillate_cq15` | 0.38 | 4.22 | 0.09 | 2.03 | 5.40 | 0.38 | sdrk fallback |
| `oscillate_cq15_fallback` | 4.22 | 15.95 | 0.26 | 5.40 | 16.78 | 0.32 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample (512-entry table, linear interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `oscillate_fallback` | 0.76 | 78.52 | 0.01 | 3.79 | 90.69 | 0.04 | CMSIS-DSP 1.18.0 arm_sin_cos_f32 of the angle in degrees per sample (512-entry sine table, cubic Hermite interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_1024` | 2.44 | 2.62 | 0.93 | 1.79 | 3.27 | 0.55 | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 17.02 | 17.82 | 0.96 | 16.76 | 18.53 | 0.90 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_4096` | 2.70 | 3.03 | 0.89 | 1.92 | 3.71 | 0.52 | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 20.55 | 20.57 | 1.00 | 20.23 | 21.21 | 0.95 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_64` | 1.64 | 1.82 | 0.90 | 1.66 | 1.82 | 0.91 | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 11.13 | 11.76 | 0.95 | 10.93 | 11.39 | 0.96 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_cq15_1024` | 2.32 | 45.37 | 0.05 | 2.65 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 2.08 | 45.37 | 0.05 | 2.35 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 28.66 | 37.40 | 0.77 | 31.56 | 39.49 | 0.80 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_cq15_1024_fallback` | 42.83 | 37.40 | 1.15 | 50.08 | 39.49 | 1.27 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_cq15_1024_full_scale` | 2.14 | 45.37 | 0.05 | 2.40 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 1.93 | 2.57 | 0.75 | 1.61 | 2.83 | 0.57 | realfft on rustfft (RealFftPlanner, forward, out of place) |
| `rfft_1024_fallback` | 9.01 | 8.63 | 1.04 | 8.87 | 9.37 | 0.95 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `am_mod` | 0.38 | 0.38 | 1.00 | 0.30 | 0.30 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 0.14 | 0.32 | 0.43 | 0.20 | 0.48 | 0.43 | sdrk fallback |
| `am_mod_cq15_fallback` | 0.32 | 2.82 | 0.11 | 0.48 | 2.37 | 0.20 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `am_mod_fallback` | 0.38 | 0.53 | 0.73 | 0.30 | 0.54 | 0.56 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 161.09 | 161.09 | 1.00 | 163.36 | 163.36 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 53.15 | 53.15 | 1.00 | 51.10 | 51.10 | 1.00 | sdrk fallback |
| `bit_descramble_block` | 0.32 | 4.76 | 0.07 | 0.31 | 5.55 | 0.06 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_descramble_in_place_block` | 1.36 | 4.76 | 0.28 | 1.19 | 5.55 | 0.21 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_block` | 0.77 | 4.41 | 0.17 | 0.74 | 5.18 | 0.14 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_in_place_block` | 1.01 | 4.41 | 0.23 | 0.96 | 5.18 | 0.18 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `channel_bank_channelize` | 783.82 | 783.82 | 1.00 | 97.13 | 97.13 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 513.98 | 513.98 | 1.00 | 85.10 | 85.10 | 1.00 | sdrk fallback |
| `channel_select_block` | 0.28 | 0.28 | 1.00 | 1.83 | 1.83 | 1.00 | sdrk fallback |
| `channelize` | 12.20 | 103.02 | 0.12 | 16.04 | 85.19 | 0.19 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 12.22 | 12.22 | 1.00 | 16.16 | 16.16 | 1.00 | sdrk fallback |
| `channelize_fallback` | 37.56 | 105.93 | 0.35 | 32.73 | 118.74 | 0.28 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `correlate_block` | 7.62 | 7.62 | 1.00 | 5.50 | 5.50 | 1.00 | sdrk fallback |
| `costas` | 55.09 | 55.09 | 1.00 | 68.07 | 68.07 | 1.00 | sdrk fallback |
| `costas_fallback` | 55.09 | 68.66 | 0.80 | 68.07 | 66.16 | 1.03 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `costas_qam1024` | 127.33 | 127.33 | 1.00 | 148.31 | 148.31 | 1.00 | sdrk fallback |
| `costas_qam1024_fallback` | 127.33 | 208.62 | 0.61 | 148.31 | 254.58 | 0.58 | GNU Radio constellation_receiver_cb with constellation_rect 1024-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam16` | 119.28 | 119.28 | 1.00 | 137.31 | 137.31 | 1.00 | sdrk fallback |
| `costas_qam16_fallback` | 119.28 | 158.59 | 0.75 | 137.31 | 172.86 | 0.79 | GNU Radio constellation_receiver_cb with constellation_rect 16-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam256` | 127.65 | 127.65 | 1.00 | 148.63 | 148.63 | 1.00 | sdrk fallback |
| `costas_qam256_fallback` | 127.65 | 161.83 | 0.79 | 148.63 | 178.14 | 0.83 | GNU Radio constellation_receiver_cb with constellation_rect 256-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam64` | 121.28 | 121.28 | 1.00 | 140.32 | 140.32 | 1.00 | sdrk fallback |
| `costas_qam64_fallback` | 121.28 | 159.13 | 0.76 | 140.32 | 173.45 | 0.81 | GNU Radio constellation_receiver_cb with constellation_rect 64-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 11.81 | 13.17 | 0.90 | 12.80 | 16.19 | 0.79 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 7.05 | 10.38 | 0.68 | 8.04 | 9.40 | 0.86 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 11.04 | 13.17 | 0.84 | 12.04 | 16.19 | 0.74 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 7.05 | 10.38 | 0.68 | 8.04 | 9.40 | 0.86 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 104.01 | 1094.85 | 0.09 | 146.19 | 323.95 | 0.45 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_fallback` | 104.01 | 929.61 | 0.11 | 146.19 | 220.19 | 0.66 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam1024` | 125.31 | 125.31 | 1.00 | 173.38 | 173.38 | 1.00 | sdrk fallback |
| `equalize_qam1024_fallback` | 125.31 | 937.23 | 0.13 | 173.38 | 277.16 | 0.63 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 1024-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam16` | 121.27 | 1100.18 | 0.11 | 169.39 | 328.53 | 0.52 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam16_fallback` | 121.27 | 938.01 | 0.13 | 169.39 | 277.23 | 0.61 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam256` | 125.31 | 125.31 | 1.00 | 173.38 | 173.38 | 1.00 | sdrk fallback |
| `equalize_qam256_fallback` | 125.31 | 938.01 | 0.13 | 173.38 | 277.23 | 0.63 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 256-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam64` | 121.24 | 1100.18 | 0.11 | 168.37 | 328.53 | 0.51 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam64_fallback` | 121.24 | 938.01 | 0.13 | 168.37 | 277.23 | 0.61 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 5.69 | 5.69 | 1.00 | 6.18 | 6.18 | 1.00 | sdrk fallback |
| `frame_decode_block` | 6.90 | 6.90 | 1.00 | 5.79 | 5.79 | 1.00 | sdrk fallback |
| `frame_decode_four_block` | 22.08 | 22.08 | 1.00 | 21.17 | 21.17 | 1.00 | sdrk fallback |
| `frame_encode_block` | 1.62 | 1.62 | 1.00 | 1.72 | 1.72 | 1.00 | sdrk fallback |
| `gauss_noise_c32` | 1.64 | 187.46 | 0.01 | 3.09 | 192.60 | 0.02 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fallback` | 10.68 | 187.46 | 0.06 | 18.59 | 192.60 | 0.10 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fastnoise` | 1.64 | 4.04 | 0.41 | 3.09 | 6.09 | 0.51 | GNU Radio fastnoise_source_c, amplitude sigma * sqrt(2) (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32` | 0.84 | 116.24 | 0.01 | 1.56 | 124.74 | 0.01 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fallback` | 5.34 | 116.24 | 0.05 | 9.30 | 124.74 | 0.07 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fastnoise` | 0.84 | 4.05 | 0.21 | 1.56 | 6.10 | 0.26 | GNU Radio fastnoise_source_f, amplitude sigma (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32_ziggurat` | 0.84 | 9.82 | 0.09 | 1.56 | 10.43 | 0.15 | rand_distr 0.5.1 StandardNormal as f32 times sigma over rand 0.9.5 SmallRng (Ziggurat in f64 with rejection: a variable number of draws per value, no random access, no bit-exact sequence across platforms; reference does different work) |
| `gauss_noise_q15` | 1.04 | 9.98 | 0.10 | 1.83 | 9.20 | 0.20 | sdrk fallback |
| `gauss_noise_q15_fallback` | 9.98 | 9.98 | 1.00 | 9.20 | 9.20 | 1.00 | sdrk fallback |
| `halfband_decimate` | 2.58 | 107.89 | 0.02 | 3.44 | 34.07 | 0.10 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its AVX-512F dot product (-O3 -ffp-contract=off, avx512) |
| `halfband_decimate_cq15` | 7.15 | 11.94 | 0.60 | 8.52 | 13.27 | 0.64 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 11.94 | 30.34 | 0.39 | 13.27 | 33.28 | 0.40 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `halfband_decimate_fallback` | 60.73 | 118.25 | 0.51 | 12.98 | 65.16 | 0.20 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, avx512) |
| `halfband_decimate_real` | 1.63 | 27.50 | 0.06 | 1.06 | 20.10 | 0.05 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its AVX-512F dot product (-O3 -ffp-contract=off, avx512) |
| `halfband_decimate_real_fallback` | 8.91 | 78.71 | 0.11 | 8.78 | 52.09 | 0.17 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, avx512) |
| `halfband_decimate_real_q15` | 3.82 | 11.68 | 0.33 | 4.51 | 12.40 | 0.36 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 11.68 | 15.19 | 0.77 | 12.40 | 16.66 | 0.74 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `halfband_ref_fir_decimate` | 7.56 | 20.86 | 0.36 | 7.77 | 18.94 | 0.41 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 30.90 | 27.19 | 1.14 | 28.35 | 28.95 | 0.98 | sdrk fallback |
| `halfband_ref_shifted` | 4.95 | 63.16 | 0.08 | 5.89 | 15.45 | 0.38 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 9.93 | 14.74 | 0.67 | 11.61 | 16.34 | 0.71 | sdrk fallback |
| `iq_offset_block` | 2.89 | 2.89 | 1.00 | 2.95 | 2.95 | 1.00 | sdrk fallback |
| `morse_decode_block` | 12.75 | 12.75 | 1.00 | 17.03 | 17.03 | 1.00 | sdrk fallback |
| `morse_encode_block` | 4.20 | 4.20 | 1.00 | 4.64 | 4.64 | 1.00 | sdrk fallback |
| `noise_blank` | 11.45 | 11.45 | 1.00 | 11.33 | 11.33 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 8.41 | 8.41 | 1.00 | 9.25 | 9.25 | 1.00 | sdrk fallback |
| `noise_block` | 1.64 | 1.64 | 1.00 | 3.09 | 3.09 | 1.00 | sdrk fallback |
| `preemph_limit` | 19.13 | 19.13 | 1.00 | 12.17 | 12.17 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 26.71 | 26.71 | 1.00 | 18.69 | 18.69 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 15.64 | 15.64 | 1.00 | 13.07 | 13.07 | 1.00 | sdrk fallback |
| `psk_demod_block` | 35.14 | 1156.12 | 0.03 | 33.62 | 415.04 | 0.08 | GNU Radio fir_filter_ccf + symbol_sync_cc + costas_loop_cc + constellation_decoder_cb (the sum of the blocks of the receiver chain, generic compiler flags; no acquisition and no lock watches) |
| `psk_mod_block` | 4.66 | 216.63 | 0.02 | 3.57 | 302.04 | 0.01 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `psk_mod_block_liquid` | 4.66 | 49.54 | 0.09 | 3.57 | 57.22 | 0.06 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `pulse_match_block` | 10.99 | 878.24 | 0.01 | 6.23 | 178.86 | 0.03 | GNU Radio fir_filter_ccf (fir_filter_ccf_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_match_block_liquid` | 10.99 | 73.62 | 0.15 | 6.23 | 69.98 | 0.09 | liquid-dsp 1.8.3 firfilt_crcf (firfilt.proto.c), the SIMD dot product of the level (the portable loop at simd128) |
| `pulse_shape_block` | 3.91 | 168.93 | 0.02 | 2.86 | 86.35 | 0.03 | GNU Radio interp_fir_filter_ccf (interp_fir_filter_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_shape_block_liquid` | 3.91 | 47.79 | 0.08 | 2.86 | 55.35 | 0.05 | liquid-dsp 1.8.3 firinterp_crcf (firinterp.proto.c, firpfb.proto.c), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_acquire_block` | 26.62 | 26.62 | 1.00 | 30.30 | 30.30 | 1.00 | sdrk fallback |
| `qam_demod_block` | 59.62 | 1287.05 | 0.05 | 65.02 | 463.62 | 0.14 | GNU Radio fir_filter_ccf + symbol_sync_cc + linear_equalizer + constellation_receiver_cb (the sum of the blocks of the receiver chain, generic compiler flags and VOLK generic dot products; no acquisition, no blind equalizer stage and no lock watches) |
| `qam_demod_qam256_block` | 61.08 | 61.08 | 1.00 | 66.84 | 66.84 | 1.00 | sdrk fallback |
| `qam_demod_search_block` | 62.95 | 62.99 | 1.00 | 68.84 | 68.90 | 1.00 | sdrk fallback |
| `qam_frame_decode_block` | 71.06 | 71.06 | 1.00 | 76.04 | 76.04 | 1.00 | sdrk fallback |
| `qam_frame_encode_block` | 5.02 | 5.02 | 1.00 | 4.12 | 4.12 | 1.00 | sdrk fallback |
| `qam_mod_block` | 4.17 | 217.02 | 0.02 | 3.21 | 302.57 | 0.01 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_block_liquid` | 4.17 | 50.05 | 0.08 | 3.21 | 57.98 | 0.06 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_mod_qam256_block` | 4.41 | 217.79 | 0.02 | 3.54 | 303.62 | 0.01 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_qam256_block_liquid` | 4.41 | 51.08 | 0.09 | 3.54 | 59.51 | 0.06 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_soft_bits_qam1024` | 2.05 | 2.05 | 1.00 | 2.06 | 2.06 | 1.00 | sdrk fallback |
| `qam_soft_bits_qam16` | 0.36 | 1224.82 | 0.00 | 0.33 | 1330.90 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam16_liquid` | 0.36 | 157.17 | 0.00 | 0.33 | 172.10 | 0.00 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam16_lut` | 0.36 | 56.64 | 0.01 | 0.33 | 55.46 | 0.01 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qam256` | 1.21 | 1.21 | 1.00 | 1.01 | 1.01 | 1.00 | sdrk fallback |
| `qam_soft_bits_qam64` | 0.77 | 4356.96 | 0.00 | 0.77 | 4764.87 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam64_liquid` | 0.77 | 209.23 | 0.00 | 0.77 | 220.38 | 0.00 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam64_lut` | 0.77 | 56.64 | 0.01 | 0.77 | 55.46 | 0.01 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qpsk` | 0.14 | 0.26 | 0.52 | 0.13 | 0.20 | 0.67 | sdrk fallback |
| `rds_decode_block` | 7.20 | 7.20 | 1.00 | 7.96 | 7.96 | 1.00 | sdrk fallback |
| `rds_demod_block` | 476.92 | 476.92 | 1.00 | 50.29 | 50.29 | 1.00 | sdrk fallback |
| `rds_shape` | 42.06 | 42.06 | 1.00 | 39.42 | 39.42 | 1.00 | sdrk fallback |
| `real_to_complex` | 2.00 | 53.98 | 0.04 | 1.57 | 23.02 | 0.07 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its AVX-512F dot product (-O3 -ffp-contract=off, avx512) |
| `real_to_complex_cq15` | 4.30 | 220.92 | 0.02 | 5.04 | 25.98 | 0.19 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, avx512) |
| `real_to_complex_cq15_fallback` | 14.45 | 220.09 | 0.07 | 15.62 | 25.92 | 0.60 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, avx512) |
| `real_to_complex_fallback` | 12.02 | 105.23 | 0.11 | 12.14 | 55.01 | 0.22 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, avx512) |
| `snr_estimate_block` | 2.64 | 48.13 | 0.05 | 2.35 | 56.75 | 0.04 | GNU Radio mpsk_snr_est_cc(SNR_EST_M2M4, 4096, 0.001): the moments \|x\|^2 and \|x\|^4 averaged in double, six cabsf per symbol (reference does different work: no decisions, a PSK kurtosis; generic compiler flags) |
| `snr_estimate_block_svr` | 2.64 | 69.08 | 0.04 | 2.35 | 77.87 | 0.03 | GNU Radio mpsk_snr_est_cc(SNR_EST_SVR, 4096, 0.001): \|x\|^4 and \|x_n\|^2 \|x_(n-1)\|^2 averaged in double, two cabsf per symbol (reference does different work: no decisions, a PSK constellation; generic compiler flags) |
| `stereo_encode` | 1.15 | 1.15 | 1.00 | 5.53 | 5.53 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 1.59 | 1.59 | 1.00 | 5.54 | 5.54 | 1.00 | sdrk fallback |
| `stereo_matrix` | 0.35 | 0.85 | 0.42 | 0.32 | 1.07 | 0.30 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.20 | 0.39 | 0.51 | 0.13 | 0.58 | 0.23 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 0.39 | 0.79 | 0.49 | 0.58 | 1.55 | 0.37 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `stereo_matrix_fallback` | 0.48 | 0.60 | 0.80 | 0.54 | 0.53 | 1.02 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `stereo_pll_fallback` | 61.34 | 56.30 | 1.09 | 74.40 | 78.06 | 0.95 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, avx512) |
| `stereo_pll_q15_fallback` | 37.04 | 45.61 | 0.81 | 38.05 | 34.89 | 1.09 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `symbol_demap_qam16_block` | 0.36 | 1224.82 | 0.00 | 0.33 | 1330.90 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam16_block_liquid` | 0.36 | 157.17 | 0.00 | 0.33 | 172.10 | 0.00 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam16_block_lut` | 0.36 | 47.18 | 0.01 | 0.33 | 35.46 | 0.01 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_demap_qam256_block` | 1.21 | 1.21 | 1.00 | 1.01 | 1.01 | 1.00 | sdrk fallback |
| `symbol_demap_qam64_block` | 0.77 | 4356.96 | 0.00 | 0.77 | 4764.87 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam64_block_liquid` | 0.77 | 209.33 | 0.00 | 0.77 | 220.52 | 0.00 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam64_block_lut` | 0.77 | 56.64 | 0.01 | 0.77 | 55.46 | 0.01 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_map_block` | 1.28 | 10.17 | 0.13 | 2.17 | 11.66 | 0.19 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_liquid` | 1.28 | 17.26 | 0.07 | 2.17 | 20.14 | 0.11 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qam64` | 2.40 | 13.29 | 0.18 | 3.55 | 15.92 | 0.22 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qam64_liquid` | 2.40 | 21.34 | 0.11 | 3.55 | 26.23 | 0.14 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qpsk` | 5.19 | 7.05 | 0.74 | 5.03 | 7.40 | 0.68 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qpsk_liquid` | 5.19 | 13.18 | 0.39 | 5.03 | 14.05 | 0.36 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_sync` | 23.21 | 23.21 | 1.00 | 27.84 | 27.84 | 1.00 | sdrk fallback |
| `symbol_sync_fallback` | 23.21 | 36.87 | 0.63 | 27.84 | 36.52 | 0.76 | GNU Radio symbol_sync_cc (symbol_sync_cc_impl.cc of gr-digital, TED_GARDNER, IR_MMSE_8TAP, generic compiler flags; reference does different work: an 8-tap MMSE interpolator against our 4-point Lagrange polynomial) |
| `symbol_sync_real` | 11.28 | 11.28 | 1.00 | 13.78 | 13.78 | 1.00 | sdrk fallback |
| `xcorr_fft_n4096` | 9.11 | 854.02 | 0.01 | 6.21 | 115.30 | 0.05 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n4096_fallback` | 64.05 | 7488.43 | 0.01 | 57.48 | 872.20 | 0.07 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |
| `xcorr_fft_n512` | 7.61 | 839.33 | 0.01 | 5.49 | 100.11 | 0.05 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n512_fallback` | 51.59 | 4207.81 | 0.01 | 46.21 | 529.97 | 0.09 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |

- Not measured at this level: `real_to_complex_airspy`.

#### Simulated cycles at `neon` (`aarch64`)

Reference model `cortex-a72`, counter-model `cortex-a76`, mode `whole-loop`; execution counts: qemu-aarch64 with the bbcount plugin, translation-block and transfer counts.

| Entry | ours cortex-a72 (cycles/sample) | comparison cortex-a72 (cycles/sample) | ratio cortex-a72 (ours/comparison) | ours cortex-a76 (cycles/sample) | comparison cortex-a76 (cycles/sample) | ratio cortex-a76 (ours/comparison) | comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `convert_cq15_cs8` | 0.52 | 1.52 | 0.34 | 0.49 | 1.02 | 0.48 | volk_8i_convert_16i |
| `convert_cq15_cs8_fallback` | 0.54 | 0.64 | 0.84 | 0.53 | 0.64 | 0.83 | CMSIS-DSP 1.18.0 arm_q7_to_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_cq15_cu8` | 0.61 | 1.52 | 0.40 | 0.61 | 1.02 | 0.60 | volk_8i_convert_16i on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cq15_cu8_fallback` | 0.66 | 0.64 | 1.03 | 0.66 | 0.64 | 1.02 | CMSIS-DSP 1.18.0 arm_q7_to_q15 on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_cs16` | 1.35 | 2.29 | 0.59 | 1.08 | 2.03 | 0.53 | volk_16i_s32f_convert_32f |
| `convert_cs16_fallback` | 1.55 | 2.56 | 0.61 | 1.29 | 2.54 | 0.51 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_cs8` | 1.63 | 1.96 | 0.83 | 1.06 | 1.83 | 0.58 | volk_8i_s32f_convert_32f |
| `convert_cs8_fallback` | 1.92 | 1.90 | 1.01 | 1.29 | 1.65 | 0.78 | CMSIS-DSP 1.18.0 arm_q7_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_cu8` | 1.19 | 1.96 | 0.61 | 0.95 | 1.83 | 0.52 | volk_8i_s32f_convert_32f on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cu8_fallback` | 1.55 | 1.90 | 0.81 | 1.29 | 1.65 | 0.78 | CMSIS-DSP 1.18.0 arm_q7_to_float on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_out_cs16` | 1.69 | 2.78 | 0.61 | 1.44 | 1.90 | 0.76 | volk_32f_s32f_convert_16i |
| `convert_out_cs16_fallback` | 4.08 | 1.79 | 2.28 | 2.56 | 1.78 | 1.44 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_out_cs8` | 1.63 | 2.65 | 0.62 | 1.35 | 1.71 | 0.79 | volk_32f_s32f_convert_8i |
| `convert_out_cs8_fallback` | 5.80 | 1.78 | 3.27 | 3.09 | 1.39 | 2.22 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_out_cu8` | 2.91 | 2.66 | 1.09 | 2.09 | 2.15 | 0.97 | volk_32f_s32f_x2_convert_8u |
| `convert_out_cu8_fallback` | 7.14 | 1.78 | 4.02 | 4.35 | 1.39 | 3.13 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_out_real_rs16` | 0.86 | 1.40 | 0.62 | 0.74 | 0.96 | 0.77 | volk_32f_s32f_convert_16i |
| `convert_out_real_rs16_fallback` | 2.05 | 0.90 | 2.29 | 1.30 | 0.89 | 1.45 | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_out_real_rs8` | 0.83 | 1.33 | 0.62 | 0.69 | 0.86 | 0.80 | volk_32f_s32f_convert_8i |
| `convert_out_real_rs8_fallback` | 2.91 | 0.89 | 3.26 | 1.55 | 0.70 | 2.21 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_out_real_ru8` | 1.47 | 1.34 | 1.10 | 1.06 | 1.08 | 0.98 | volk_32f_s32f_x2_convert_8u |
| `convert_out_real_ru8_fallback` | 3.58 | 0.89 | 4.01 | 2.18 | 0.70 | 3.11 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `convert_real_rs16` | 0.69 | 1.15 | 0.60 | 0.56 | 1.02 | 0.55 | volk_16i_s32f_convert_32f |
| `convert_real_rs16_fallback` | 0.79 | 1.28 | 0.62 | 0.66 | 1.27 | 0.52 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `rotate` | 2.74 | 4.63 | 0.59 | 1.96 | 3.71 | 0.53 | volk_32fc_s32fc_x2_rotator2_32fc |
| `rotate_cq15` | 4.54 | 14.12 | 0.32 | 3.66 | 15.09 | 0.24 | sdrk fallback |
| `rotate_cq15_fallback` | 14.12 | 79.06 | 0.18 | 15.09 | 72.65 | 0.21 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `rotate_fallback` | 11.17 | 11.10 | 1.01 | 13.08 | 14.09 | 0.93 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `scramble_block` | 8.02 | 8.02 | 1.00 | 5.95 | 5.95 | 1.00 | sdrk fallback |
| `scramble_c32_b1` | 1.71 | 4.43 | 0.39 | 1.33 | 4.18 | 0.32 | +-1.0 sequence from the index bits into a buffer of 256 + volk_32fc_32f_multiply_32fc (VOLK 3.3.0 dispatcher; the form of srsRAN srsran_scrambling_c) |
| `scramble_c32_b1_fallback` | 1.83 | 4.22 | 0.43 | 1.45 | 3.96 | 0.37 | +-1.0 sequence from the index bits into a buffer of 256 + VOLK 3.3.0 volk_32fc_32f_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2` | 3.04 | 8.52 | 0.36 | 2.46 | 6.76 | 0.36 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_neon without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2_dvbs2` | 3.04 | 6.18 | 0.49 | 2.46 | 6.80 | 0.36 | GNU Radio dvbs2_physical_cc PL scrambling, the per-symbol switch on R_n (benches/refs/gr-dvbs2-plscramble.c, main at ec1bf15; reference does different work; -O3) |
| `scramble_c32_b2_fallback` | 3.93 | 11.23 | 0.35 | 2.92 | 11.21 | 0.26 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8` | 5.51 | 8.52 | 0.65 | 3.54 | 6.76 | 0.52 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_neon without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8_fallback` | 8.14 | 11.23 | 0.73 | 9.08 | 11.21 | 0.81 | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_cq15_b2` | 1.63 | 8.57 | 0.19 | 1.47 | 6.55 | 0.22 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `scramble_cq15_b2_fallback` | 3.02 | 8.57 | 0.35 | 2.14 | 6.55 | 0.33 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `scramble_cq15_b8` | 4.27 | 8.57 | 0.50 | 4.09 | 6.55 | 0.62 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `scramble_cq15_b8_fallback` | 5.20 | 8.57 | 0.61 | 5.08 | 6.55 | 0.78 | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir` | 37.96 | 126.85 | 0.30 | 24.33 | 99.19 | 0.25 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 106.10 | 107.72 | 0.98 | 92.51 | 98.30 | 0.94 | sdrk fallback |
| `fir_cq15_fallback` | 107.72 | 180.05 | 0.60 | 98.30 | 181.88 | 0.54 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_decimate` | 9.89 | 20.50 | 0.48 | 9.08 | 16.88 | 0.54 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 26.40 | 29.57 | 0.89 | 23.88 | 27.63 | 0.86 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 29.57 | 16.40 | 1.80 | 27.63 | 16.90 | 1.63 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_decimate_fallback` | 32.76 | 34.10 | 0.96 | 25.71 | 41.23 | 0.62 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_fallback` | 124.44 | 132.43 | 0.94 | 84.01 | 133.90 | 0.63 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate` | 42.73 | 138.57 | 0.31 | 28.61 | 113.64 | 0.25 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 75.80 | 127.87 | 0.59 | 76.00 | 145.98 | 0.52 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 127.87 | 141.59 | 0.90 | 145.98 | 141.30 | 1.03 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate_fallback` | 143.96 | 135.30 | 1.06 | 102.02 | 152.44 | 0.67 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate_real` | 23.66 | 107.89 | 0.22 | 23.25 | 85.10 | 0.27 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 82.83 | 67.65 | 1.22 | 62.42 | 76.22 | 0.82 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate_real_q15` | 73.09 | 73.09 | 1.00 | 73.97 | 73.97 | 1.00 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 73.09 | 70.80 | 1.03 | 73.97 | 70.65 | 1.05 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate_short` | 11.77 | 66.29 | 0.18 | 10.87 | 63.78 | 0.17 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_short_fallback` | 30.89 | 27.28 | 1.13 | 23.59 | 29.90 | 0.79 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 8 (80 taps), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `resample` | 55.27 | 140.12 | 0.39 | 50.97 | 114.43 | 0.45 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 145.75 | 193.78 | 0.75 | 118.45 | 184.91 | 0.64 | sdrk fallback |
| `resample_cq15_fallback` | 193.78 | 140.28 | 1.38 | 184.91 | 139.74 | 1.32 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `resample_fallback` | 175.82 | 134.66 | 1.31 | 142.09 | 151.50 | 0.94 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `resample_real` | 32.36 | 109.41 | 0.30 | 29.37 | 85.87 | 0.34 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 139.68 | 67.33 | 2.07 | 121.24 | 75.75 | 1.60 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `resample_real_q15` | 97.68 | 113.65 | 0.86 | 78.50 | 107.66 | 0.73 | sdrk fallback |
| `resample_real_q15_fallback` | 113.65 | 70.14 | 1.62 | 107.66 | 69.87 | 1.54 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `mag` | 8.58 | 8.59 | 1.00 | 1.77 | 1.79 | 0.99 | volk_32fc_magnitude_32f |
| `mag_fallback` | 8.56 | 8.82 | 0.97 | 1.77 | 2.04 | 0.87 | CMSIS-DSP 1.18.0 arm_cmplx_mag_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `mag_q15` | 9.91 | 9.10 | 1.09 | 3.79 | 3.68 | 1.03 | volk_16ic_magnitude_16i |
| `mag_q15_fallback` | 9.88 | 35.41 | 0.28 | 4.07 | 32.99 | 0.12 | CMSIS-DSP 1.18.0 arm_cmplx_mag_q15 (Q2.14, arm_sqrt_q31), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `mag_sq` | 1.28 | 1.78 | 0.72 | 0.91 | 1.53 | 0.60 | volk_32fc_magnitude_squared_32f |
| `mag_sq_fallback` | 1.52 | 2.28 | 0.67 | 1.14 | 1.53 | 0.75 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `mag_sq_q15` | 0.71 | 1.01 | 0.70 | 0.64 | 0.91 | 0.71 | sdrk fallback |
| `mag_sq_q15_fallback` | 1.01 | 1.40 | 0.72 | 0.91 | 1.15 | 0.79 | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_q15 (Q3.13, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `quad_demod` | 14.55 | 15.15 | 0.96 | 6.91 | 10.40 | 0.66 | volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f |
| `quad_demod_cq15` | 19.37 | 120.12 | 0.16 | 10.61 | 115.54 | 0.09 | sdrk fallback |
| `quad_demod_cq15_fallback` | 120.12 | 138.91 | 0.86 | 115.54 | 136.18 | 0.85 | CMSIS-DSP arm_cmplx_mult_cmplx_q15 + arm_atan2_q15 + arm_scale_q15 (generic C), conjugate in a plain loop |
| `quad_demod_fallback` | 71.50 | 72.39 | 0.99 | 74.45 | 55.28 | 1.35 | GNU Radio quadrature_demod_cf (gr-analog, fast_atan2f, generic compiler flags) |
| `fm_mod` | 10.17 | 54.90 | 0.19 | 5.26 | 57.43 | 0.09 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `fm_mod_cq15` | 8.34 | 18.86 | 0.44 | 4.59 | 15.77 | 0.29 | sdrk fallback |
| `fm_mod_cq15_fallback` | 18.86 | 54.29 | 0.35 | 15.77 | 47.65 | 0.33 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample of an integer phase accumulator, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fm_mod_fallback` | 18.66 | 54.90 | 0.34 | 16.45 | 57.43 | 0.29 | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `squelch` | 12.15 | 37.71 | 0.32 | 8.21 | 37.23 | 0.22 | GNU Radio pwr_squelch_cc (squelch_base_cc_impl.cc of gr-analog, generic compiler flags) |
| `squelch_cq15` | 11.08 | 11.08 | 1.00 | 9.12 | 9.12 | 1.00 | sdrk fallback |
| `window` | 1.96 | 2.15 | 0.91 | 1.45 | 1.65 | 0.88 | volk_32fc_32f_multiply_32fc |
| `window_cq15` | 1.05 | 1.65 | 0.64 | 0.83 | 1.40 | 0.59 | sdrk fallback |
| `window_cq15_fallback` | 1.65 | 2.04 | 0.81 | 1.40 | 1.78 | 0.78 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `window_fallback` | 2.03 | 2.53 | 0.80 | 1.53 | 2.02 | 0.75 | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `power_db` | 14.40 | 7.23 | 1.99 | 6.61 | 5.45 | 1.21 | volk_32f_log2_32f + volk_32f_s32f_multiply_32f + volk_32f_s32f_add_32f |
| `power_db_fallback` | 18.37 | 58.97 | 0.31 | 8.81 | 39.50 | 0.22 | CMSIS-DSP 1.18.0 arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `power_db_q15` | 13.74 | 14.39 | 0.96 | 6.46 | 6.66 | 0.97 | sdrk fallback |
| `power_db_q15_fallback` | 14.39 | 60.25 | 0.24 | 6.66 | 40.77 | 0.16 | CMSIS-DSP 1.18.0 arm_q15_to_float + arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `agc` | 25.67 | 40.13 | 0.64 | 18.01 | 28.09 | 0.64 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `agc_cq15` | 130.48 | 130.48 | 1.00 | 88.18 | 88.18 | 1.00 | sdrk fallback |
| `agc_fallback` | 25.67 | 40.13 | 0.64 | 18.01 | 28.09 | 0.64 | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `dc_block` | 10.07 | 15.04 | 0.67 | 4.79 | 8.04 | 0.60 | sdrk fallback |
| `dc_block_cq15` | 11.04 | 11.04 | 1.00 | 12.54 | 12.54 | 1.00 | sdrk fallback |
| `dc_block_fallback` | 15.04 | 54.62 | 0.28 | 8.04 | 55.67 | 0.14 | liquid-dsp 1.8.3 iirfilt_crcf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real` | 6.74 | 11.07 | 0.61 | 3.34 | 8.05 | 0.41 | sdrk fallback |
| `dc_block_real_fallback` | 11.07 | 52.71 | 0.21 | 8.05 | 58.60 | 0.14 | liquid-dsp 1.8.3 iirfilt_rrrf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real_q15` | 11.04 | 11.04 | 1.00 | 10.06 | 10.06 | 1.00 | sdrk fallback |
| `deemphasis` | 5.99 | 15.07 | 0.40 | 3.09 | 10.06 | 0.31 | sdrk fallback |
| `deemphasis_fallback` | 15.07 | 18.10 | 0.83 | 10.06 | 21.07 | 0.48 | GNU Radio fm_deemph: iir_filter_ffd with the bilinear taps of fm_emph.py (double taps and accumulator, generic compiler flags) |
| `deemphasis_q15` | 12.05 | 12.05 | 1.00 | 11.05 | 11.05 | 1.00 | sdrk fallback |
| `biquad` | 17.66 | 26.24 | 0.67 | 7.68 | 24.13 | 0.32 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (neon64, -ffp-contract=off), two calls on planar re and im (deinterleaving not counted) |
| `biquad_cq15` | 16.38 | 24.13 | 0.68 | 15.44 | 38.12 | 0.40 | sdrk fallback |
| `biquad_cq15_fallback` | 24.13 | 200.18 | 0.12 | 38.12 | 124.16 | 0.31 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `biquad_fallback` | 25.50 | 32.04 | 0.80 | 14.83 | 18.91 | 0.78 | CMSIS-DSP 1.18.0 arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `biquad_real` | 12.19 | 13.12 | 0.93 | 7.42 | 12.07 | 0.61 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (neon64, -ffp-contract=off) |
| `biquad_real_fallback` | 13.16 | 24.04 | 0.55 | 10.11 | 14.03 | 0.72 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df2T_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `biquad_real_q15` | 14.22 | 15.11 | 0.94 | 12.90 | 28.09 | 0.46 | sdrk fallback |
| `biquad_real_q15_fallback` | 15.11 | 100.10 | 0.15 | 28.09 | 62.08 | 0.45 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `dot` | 1.34 | 1.29 | 1.04 | 0.96 | 1.04 | 0.92 | volk_32fc_32f_dot_prod_32fc |
| `dot_complex` | 3.66 | 3.54 | 1.03 | 1.77 | 2.03 | 0.87 | VOLK 3.3.0 volk_32fc_x2_dot_prod_32fc_neon without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `dot_complex_fallback` | 4.27 | 8.03 | 0.53 | 4.52 | 5.27 | 0.86 | CMSIS-DSP 1.18.0 arm_cmplx_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `dot_cq15` | 0.90 | 0.90 | 1.00 | 0.90 | 0.90 | 1.00 | sdrk fallback |
| `dot_cq15_fallback` | 0.90 | 1.54 | 0.58 | 0.90 | 1.29 | 0.70 | CMSIS-DSP 1.18.0 arm_dot_prod_q15 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `dot_fallback` | 4.15 | 8.08 | 0.51 | 2.27 | 4.05 | 0.56 | CMSIS-DSP 1.18.0 arm_dot_prod_f32 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `dot_real` | 0.52 | 0.66 | 0.79 | 0.42 | 0.53 | 0.80 | volk_32f_x2_dot_prod_32f |
| `dot_real_fallback` | 4.04 | 4.05 | 1.00 | 2.02 | 2.03 | 1.00 | CMSIS-DSP 1.18.0 arm_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `dot_real_q15` | 0.39 | 0.52 | 0.76 | 0.40 | 0.52 | 0.77 | sdrk fallback |
| `dot_real_q15_fallback` | 0.52 | 0.77 | 0.67 | 0.52 | 0.65 | 0.80 | CMSIS-DSP 1.18.0 arm_dot_prod_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `scale` | 1.39 | 1.78 | 0.78 | 1.14 | 1.53 | 0.74 | volk_32f_s32f_multiply_32f |
| `scale_cq15` | 0.70 | 1.52 | 0.46 | 0.58 | 1.39 | 0.41 | sdrk fallback |
| `scale_cq15_fallback` | 1.52 | 1.80 | 0.84 | 1.39 | 1.55 | 0.90 | CMSIS-DSP 1.18.0 arm_scale_q15 (shift 0, truncating) on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `scale_fallback` | 1.52 | 2.03 | 0.75 | 1.27 | 2.03 | 0.62 | CMSIS-DSP 1.18.0 arm_scale_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `add` | 1.64 | 2.03 | 0.81 | 1.39 | 1.77 | 0.78 | volk_32fc_x2_add_32fc |
| `add_cq15` | 0.70 | 0.90 | 0.78 | 0.70 | 0.90 | 0.78 | volk_16i_x2_add_saturated_16i on the 2 n values |
| `add_cq15_fallback` | 0.76 | 1.79 | 0.42 | 0.76 | 1.53 | 0.50 | CMSIS-DSP 1.18.0 arm_add_q15 on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `add_fallback` | 1.77 | 2.54 | 0.70 | 1.52 | 2.04 | 0.75 | CMSIS-DSP 1.18.0 arm_add_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `add_real` | 0.82 | 1.02 | 0.80 | 0.70 | 0.90 | 0.78 | volk_32f_x2_add_32f |
| `add_real_fallback` | 1.14 | 1.28 | 0.89 | 1.01 | 1.03 | 0.99 | CMSIS-DSP 1.18.0 arm_add_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `add_real_q15` | 0.35 | 0.46 | 0.77 | 0.35 | 0.46 | 0.77 | volk_16i_x2_add_saturated_16i |
| `add_real_q15_fallback` | 0.51 | 0.90 | 0.56 | 0.51 | 0.77 | 0.66 | CMSIS-DSP 1.18.0 arm_add_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `mul` | 2.27 | 2.78 | 0.82 | 1.83 | 2.27 | 0.81 | volk_32fc_x2_multiply_32fc |
| `mul_cq15` | 1.46 | 1.78 | 0.82 | 1.37 | 1.65 | 0.83 | sdrk fallback |
| `mul_cq15_fallback` | 1.78 | 3.31 | 0.54 | 1.65 | 2.29 | 0.72 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_q15 (Q3.13, truncating, scaled by 4), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `mul_fallback` | 3.29 | 3.53 | 0.93 | 2.03 | 3.03 | 0.67 | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `mul_real` | 0.82 | 1.02 | 0.80 | 0.70 | 0.90 | 0.78 | volk_32f_x2_multiply_32f |
| `mul_real_fallback` | 1.14 | 1.28 | 0.89 | 1.01 | 1.03 | 0.99 | CMSIS-DSP 1.18.0 arm_mult_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `mul_real_q15` | 0.41 | 0.89 | 0.46 | 0.41 | 0.82 | 0.50 | sdrk fallback |
| `mul_real_q15_fallback` | 0.89 | 0.90 | 0.98 | 0.82 | 0.78 | 1.06 | CMSIS-DSP 1.18.0 arm_mult_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `interleave` | 1.02 | 1.40 | 0.73 | 0.83 | 1.15 | 0.72 | volk_32f_x2_interleave_32fc |
| `interleave_cq15` | 0.51 | 0.51 | 1.00 | 0.42 | 0.45 | 0.93 | sdrk fallback |
| `interleave_cq15_fallback` | 0.51 | 0.64 | 0.80 | 0.45 | 0.52 | 0.87 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic on int16 (VOLK has no 16-bit interleave), benches/refs/volk-generic.c (-O3 -ffp-contract=off, neon) |
| `interleave_fallback` | 1.02 | 1.27 | 0.80 | 0.89 | 1.03 | 0.87 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, neon) |
| `deinterleave` | 1.05 | 1.28 | 0.82 | 1.12 | 1.40 | 0.80 | volk_32fc_deinterleave_32f_x2 |
| `deinterleave_cq15` | 0.53 | 0.65 | 0.82 | 0.57 | 0.71 | 0.79 | volk_16ic_deinterleave_16i_x2 |
| `deinterleave_cq15_fallback` | 0.58 | 0.65 | 0.89 | 0.64 | 0.90 | 0.71 | VOLK 3.3.0 volk_16ic_deinterleave_16i_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, neon) |
| `deinterleave_fallback` | 1.15 | 1.28 | 0.90 | 1.27 | 1.78 | 0.71 | VOLK 3.3.0 volk_32fc_deinterleave_32f_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, neon) |
| `oscillate` | 5.55 | 8.10 | 0.68 | 3.28 | 10.06 | 0.33 | sdrk fallback |
| `oscillate_cq15` | 3.82 | 7.74 | 0.49 | 3.29 | 9.05 | 0.36 | sdrk fallback |
| `oscillate_cq15_fallback` | 7.74 | 51.51 | 0.15 | 9.05 | 45.00 | 0.20 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample (512-entry table, linear interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `oscillate_fallback` | 8.10 | 52.62 | 0.15 | 10.06 | 46.15 | 0.22 | CMSIS-DSP 1.18.0 arm_sin_cos_f32 of the angle in degrees per sample (512-entry sine table, cubic Hermite interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_1024` | 12.82 | 17.22 | 0.74 | 8.54 | 13.55 | 0.63 | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 33.58 | 38.54 | 0.87 | 34.04 | 39.16 | 0.87 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_4096` | 15.05 | 20.37 | 0.74 | 10.02 | 16.16 | 0.62 | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 37.84 | 39.40 | 0.96 | 38.05 | 43.25 | 0.88 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_64` | 10.62 | 13.02 | 0.82 | 8.89 | 10.68 | 0.83 | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 22.43 | 22.44 | 1.00 | 20.36 | 24.31 | 0.84 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_cq15_1024` | 11.77 | 67.26 | 0.17 | 10.10 | 77.63 | 0.13 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 10.18 | 67.26 | 0.15 | 8.48 | 77.63 | 0.11 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 50.87 | 64.42 | 0.79 | 51.01 | 52.33 | 0.97 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_cq15_1024_fallback` | 120.91 | 64.42 | 1.88 | 100.37 | 52.33 | 1.92 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_cq15_1024_full_scale` | 10.93 | 67.26 | 0.16 | 9.01 | 77.63 | 0.12 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 8.81 | 10.88 | 0.81 | 5.73 | 10.04 | 0.57 | realfft on rustfft (RealFftPlanner, forward, out of place) |
| `rfft_1024_fallback` | 18.49 | 18.65 | 0.99 | 16.67 | 18.61 | 0.90 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `am_mod` | 2.91 | 2.91 | 1.00 | 1.52 | 1.52 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 0.90 | 1.72 | 0.52 | 0.77 | 1.33 | 0.58 | sdrk fallback |
| `am_mod_cq15_fallback` | 1.72 | 4.30 | 0.40 | 1.33 | 4.49 | 0.30 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `am_mod_fallback` | 2.91 | 1.28 | 2.27 | 1.52 | 1.03 | 1.48 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 263.70 | 263.70 | 1.00 | 208.69 | 208.69 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 89.93 | 89.93 | 1.00 | 77.43 | 77.43 | 1.00 | sdrk fallback |
| `bit_descramble_block` | 1.97 | 8.10 | 0.24 | 1.73 | 8.08 | 0.21 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_descramble_in_place_block` | 3.34 | 8.10 | 0.41 | 3.24 | 8.08 | 0.40 | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_block` | 2.70 | 8.12 | 0.33 | 2.36 | 8.09 | 0.29 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_in_place_block` | 2.94 | 8.12 | 0.36 | 2.63 | 8.09 | 0.32 | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `channel_bank_channelize` | 60.05 | 60.05 | 1.00 | 42.71 | 42.71 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 299.57 | 299.57 | 1.00 | 263.36 | 263.36 | 1.00 | sdrk fallback |
| `channel_select_block` | 1.27 | 1.27 | 1.00 | 1.27 | 1.27 | 1.00 | sdrk fallback |
| `channelize` | 30.92 | 294.02 | 0.11 | 22.58 | 240.53 | 0.09 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 30.98 | 30.98 | 1.00 | 22.71 | 22.71 | 1.00 | sdrk fallback |
| `channelize_fallback` | 97.98 | 221.93 | 0.44 | 80.80 | 165.45 | 0.49 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `correlate_block` | 44.05 | 44.05 | 1.00 | 30.56 | 30.56 | 1.00 | sdrk fallback |
| `costas` | 86.13 | 86.13 | 1.00 | 54.09 | 54.09 | 1.00 | sdrk fallback |
| `costas_fallback` | 86.13 | 82.06 | 1.05 | 54.09 | 74.73 | 0.72 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `costas_qam1024` | 163.08 | 163.08 | 1.00 | 109.88 | 109.88 | 1.00 | sdrk fallback |
| `costas_qam1024_fallback` | 163.08 | 268.24 | 0.61 | 109.88 | 256.78 | 0.43 | GNU Radio constellation_receiver_cb with constellation_rect 1024-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam16` | 153.38 | 153.38 | 1.00 | 102.26 | 102.26 | 1.00 | sdrk fallback |
| `costas_qam16_fallback` | 153.38 | 194.14 | 0.79 | 102.26 | 175.11 | 0.58 | GNU Radio constellation_receiver_cb with constellation_rect 16-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam256` | 163.08 | 163.08 | 1.00 | 109.03 | 109.03 | 1.00 | sdrk fallback |
| `costas_qam256_fallback` | 163.08 | 199.25 | 0.82 | 109.03 | 180.58 | 0.60 | GNU Radio constellation_receiver_cb with constellation_rect 256-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam64` | 146.21 | 146.21 | 1.00 | 110.16 | 110.16 | 1.00 | sdrk fallback |
| `costas_qam64_fallback` | 146.21 | 194.79 | 0.75 | 110.16 | 175.70 | 0.63 | GNU Radio constellation_receiver_cb with constellation_rect 64-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 10.13 | 10.16 | 1.00 | 7.10 | 9.15 | 0.78 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 8.11 | 8.17 | 0.99 | 7.10 | 7.15 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 8.10 | 10.16 | 0.80 | 7.09 | 9.15 | 0.78 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 8.11 | 8.17 | 0.99 | 7.10 | 7.16 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 216.21 | 329.80 | 0.66 | 183.98 | 359.68 | 0.51 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_fallback` | 216.21 | 228.84 | 0.94 | 183.98 | 236.11 | 0.78 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam1024` | 252.46 | 252.46 | 1.00 | 186.09 | 186.09 | 1.00 | sdrk fallback |
| `equalize_qam1024_fallback` | 252.46 | 244.41 | 1.03 | 186.09 | 243.21 | 0.77 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 1024-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam16` | 229.29 | 357.37 | 0.64 | 190.02 | 388.90 | 0.49 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam16_fallback` | 229.29 | 244.41 | 0.94 | 190.02 | 243.21 | 0.78 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam256` | 252.46 | 252.46 | 1.00 | 186.09 | 186.09 | 1.00 | sdrk fallback |
| `equalize_qam256_fallback` | 252.46 | 244.41 | 1.03 | 186.09 | 243.21 | 0.77 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 256-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam64` | 244.37 | 357.37 | 0.68 | 189.01 | 388.90 | 0.49 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter) |
| `equalize_qam64_fallback` | 244.37 | 244.41 | 1.00 | 189.01 | 243.21 | 0.78 | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 8.84 | 8.84 | 1.00 | 10.38 | 10.38 | 1.00 | sdrk fallback |
| `frame_decode_block` | 8.81 | 8.81 | 1.00 | 9.48 | 9.48 | 1.00 | sdrk fallback |
| `frame_decode_four_block` | 47.75 | 47.75 | 1.00 | 45.47 | 45.47 | 1.00 | sdrk fallback |
| `frame_encode_block` | 2.11 | 2.11 | 1.00 | 1.97 | 1.97 | 1.00 | sdrk fallback |
| `gauss_noise_c32` | 15.80 | 326.12 | 0.05 | 14.11 | 289.13 | 0.05 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fallback` | 27.54 | 326.12 | 0.08 | 20.98 | 289.13 | 0.07 | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fastnoise` | 15.80 | 5.04 | 3.13 | 14.11 | 5.04 | 2.80 | GNU Radio fastnoise_source_c, amplitude sigma * sqrt(2) (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32` | 7.92 | 211.38 | 0.04 | 7.07 | 188.65 | 0.04 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fallback` | 13.77 | 211.38 | 0.07 | 10.50 | 188.65 | 0.06 | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fastnoise` | 7.92 | 5.04 | 1.57 | 7.07 | 5.04 | 1.40 | GNU Radio fastnoise_source_f, amplitude sigma (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32_ziggurat` | 7.92 | 13.30 | 0.60 | 7.07 | 15.03 | 0.47 | rand_distr 0.5.1 StandardNormal as f32 times sigma over rand 0.9.5 SmallRng (Ziggurat in f64 with rejection: a variable number of draws per value, no random access, no bit-exact sequence across platforms; reference does different work) |
| `gauss_noise_q15` | 8.15 | 10.65 | 0.77 | 6.80 | 10.51 | 0.65 | sdrk fallback |
| `gauss_noise_q15_fallback` | 10.65 | 10.65 | 1.00 | 10.51 | 10.51 | 1.00 | sdrk fallback |
| `halfband_decimate` | 13.17 | 84.75 | 0.16 | 12.58 | 83.63 | 0.15 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its NEON dot product (-O3 -ffp-contract=off, neon) |
| `halfband_decimate_cq15` | 16.38 | 29.04 | 0.56 | 15.23 | 25.78 | 0.59 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 29.04 | 39.92 | 0.73 | 25.78 | 40.69 | 0.63 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `halfband_decimate_fallback` | 35.63 | 93.29 | 0.38 | 26.81 | 74.24 | 0.36 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, neon) |
| `halfband_decimate_real` | 7.75 | 84.61 | 0.09 | 6.79 | 75.02 | 0.09 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its NEON dot product (-O3 -ffp-contract=off, neon) |
| `halfband_decimate_real_fallback` | 25.66 | 78.52 | 0.33 | 21.30 | 58.40 | 0.36 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, neon) |
| `halfband_decimate_real_q15` | 7.60 | 22.82 | 0.33 | 6.44 | 21.15 | 0.30 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 22.82 | 19.75 | 1.16 | 21.15 | 20.11 | 1.05 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `halfband_ref_fir_decimate` | 19.74 | 64.01 | 0.31 | 17.93 | 51.01 | 0.35 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 78.40 | 60.69 | 1.29 | 75.20 | 56.59 | 1.33 | sdrk fallback |
| `halfband_ref_shifted` | 18.85 | 41.31 | 0.46 | 19.09 | 33.38 | 0.57 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 23.42 | 36.10 | 0.65 | 21.76 | 32.32 | 0.67 | sdrk fallback |
| `iq_offset_block` | 7.95 | 7.95 | 1.00 | 7.44 | 7.44 | 1.00 | sdrk fallback |
| `morse_decode_block` | 22.03 | 22.03 | 1.00 | 28.42 | 28.42 | 1.00 | sdrk fallback |
| `morse_encode_block` | 8.35 | 8.35 | 1.00 | 9.45 | 9.45 | 1.00 | sdrk fallback |
| `noise_blank` | 14.61 | 14.61 | 1.00 | 8.80 | 8.80 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 9.51 | 9.51 | 1.00 | 10.53 | 10.53 | 1.00 | sdrk fallback |
| `noise_block` | 15.80 | 15.80 | 1.00 | 14.11 | 14.11 | 1.00 | sdrk fallback |
| `preemph_limit` | 15.43 | 15.43 | 1.00 | 15.41 | 15.41 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 55.11 | 55.11 | 1.00 | 40.70 | 40.70 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 24.12 | 24.12 | 1.00 | 22.90 | 22.90 | 1.00 | sdrk fallback |
| `psk_demod_block` | 81.89 | 277.55 | 0.30 | 55.84 | 272.91 | 0.20 | GNU Radio fir_filter_ccf + symbol_sync_cc + costas_loop_cc + constellation_decoder_cb (the sum of the blocks of the receiver chain, generic compiler flags; no acquisition and no lock watches) |
| `psk_mod_block` | 14.05 | 332.40 | 0.04 | 13.06 | 264.83 | 0.05 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `psk_mod_block_liquid` | 14.05 | 110.13 | 0.13 | 13.06 | 109.07 | 0.12 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `pulse_match_block` | 48.91 | 173.40 | 0.28 | 31.64 | 129.18 | 0.24 | GNU Radio fir_filter_ccf (fir_filter_ccf_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_match_block_liquid` | 48.91 | 296.38 | 0.17 | 31.64 | 246.95 | 0.13 | liquid-dsp 1.8.3 firfilt_crcf (firfilt.proto.c), the SIMD dot product of the level (the portable loop at simd128) |
| `pulse_shape_block` | 13.05 | 79.64 | 0.16 | 11.83 | 52.71 | 0.22 | GNU Radio interp_fir_filter_ccf (interp_fir_filter_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter) |
| `pulse_shape_block_liquid` | 13.05 | 107.19 | 0.12 | 11.83 | 106.13 | 0.11 | liquid-dsp 1.8.3 firinterp_crcf (firinterp.proto.c, firpfb.proto.c), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_acquire_block` | 61.73 | 61.73 | 1.00 | 48.39 | 48.39 | 1.00 | sdrk fallback |
| `qam_demod_block` | 121.49 | 322.25 | 0.38 | 88.29 | 315.54 | 0.28 | GNU Radio fir_filter_ccf + symbol_sync_cc + linear_equalizer + constellation_receiver_cb (the sum of the blocks of the receiver chain, generic compiler flags and VOLK generic dot products; no acquisition, no blind equalizer stage and no lock watches) |
| `qam_demod_qam256_block` | 125.74 | 125.74 | 1.00 | 88.91 | 88.91 | 1.00 | sdrk fallback |
| `qam_demod_search_block` | 130.89 | 130.89 | 1.00 | 96.89 | 96.89 | 1.00 | sdrk fallback |
| `qam_frame_decode_block` | 146.14 | 146.14 | 1.00 | 111.79 | 111.79 | 1.00 | sdrk fallback |
| `qam_frame_encode_block` | 15.16 | 15.16 | 1.00 | 14.13 | 14.13 | 1.00 | sdrk fallback |
| `qam_mod_block` | 14.04 | 333.28 | 0.04 | 13.08 | 265.71 | 0.05 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_block_liquid` | 14.04 | 110.64 | 0.13 | 13.08 | 109.57 | 0.12 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_mod_qam256_block` | 14.65 | 335.05 | 0.04 | 13.93 | 267.47 | 0.05 | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level |
| `qam_mod_qam256_block_liquid` | 14.65 | 111.66 | 0.13 | 13.93 | 110.59 | 0.13 | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_soft_bits_qam1024` | 15.10 | 15.10 | 1.00 | 9.08 | 9.08 | 1.00 | sdrk fallback |
| `qam_soft_bits_qam16` | 3.03 | 1757.11 | 0.00 | 2.03 | 1412.59 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam16_liquid` | 3.03 | 280.89 | 0.01 | 2.03 | 301.70 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam16_lut` | 3.03 | 44.47 | 0.07 | 2.03 | 45.96 | 0.04 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qam256` | 6.05 | 6.04 | 1.00 | 4.29 | 4.28 | 1.00 | sdrk fallback |
| `qam_soft_bits_qam64` | 4.54 | 6285.06 | 0.00 | 3.03 | 5351.51 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam64_liquid` | 4.54 | 334.63 | 0.01 | 3.03 | 361.03 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam64_lut` | 4.54 | 44.47 | 0.10 | 3.03 | 45.96 | 0.07 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qpsk` | 1.78 | 1.78 | 1.00 | 1.53 | 1.53 | 1.00 | sdrk fallback |
| `rds_decode_block` | 12.14 | 12.14 | 1.00 | 13.67 | 13.67 | 1.00 | sdrk fallback |
| `rds_demod_block` | 54.29 | 54.29 | 1.00 | 49.29 | 49.29 | 1.00 | sdrk fallback |
| `rds_shape` | 53.93 | 53.93 | 1.00 | 47.22 | 47.22 | 1.00 | sdrk fallback |
| `real_to_complex` | 10.14 | 86.75 | 0.12 | 8.59 | 76.87 | 0.11 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its NEON dot product (-O3 -ffp-contract=off, neon) |
| `real_to_complex_cq15` | 10.68 | 17.90 | 0.60 | 8.74 | 17.14 | 0.51 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, neon) |
| `real_to_complex_cq15_fallback` | 33.36 | 17.90 | 1.86 | 31.07 | 17.14 | 1.81 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, neon) |
| `real_to_complex_fallback` | 31.17 | 80.68 | 0.39 | 30.71 | 60.25 | 0.51 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, neon) |
| `snr_estimate_block` | 12.81 | 71.51 | 0.18 | 6.58 | 49.49 | 0.13 | GNU Radio mpsk_snr_est_cc(SNR_EST_M2M4, 4096, 0.001): the moments \|x\|^2 and \|x\|^4 averaged in double, six cabsf per symbol (reference does different work: no decisions, a PSK kurtosis; generic compiler flags) |
| `snr_estimate_block_svr` | 12.81 | 103.78 | 0.12 | 6.58 | 72.57 | 0.09 | GNU Radio mpsk_snr_est_cc(SNR_EST_SVR, 4096, 0.001): \|x\|^4 and \|x_n\|^2 \|x_(n-1)\|^2 averaged in double, two cabsf per symbol (reference does different work: no decisions, a PSK constellation; generic compiler flags) |
| `stereo_encode` | 15.13 | 15.13 | 1.00 | 16.08 | 16.08 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 21.13 | 21.13 | 1.00 | 22.07 | 22.07 | 1.00 | sdrk fallback |
| `stereo_matrix` | 1.90 | 5.74 | 0.33 | 1.39 | 5.23 | 0.27 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.77 | 0.77 | 1.00 | 0.76 | 0.76 | 1.00 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 0.77 | 2.70 | 0.28 | 0.76 | 2.31 | 0.33 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `stereo_matrix_fallback` | 1.90 | 3.83 | 0.50 | 1.39 | 3.07 | 0.45 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `stereo_pll_fallback` | 84.51 | 86.93 | 0.97 | 56.44 | 57.23 | 0.99 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, neon) |
| `stereo_pll_q15_fallback` | 41.07 | 100.31 | 0.41 | 39.08 | 88.00 | 0.44 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `symbol_demap_qam16_block` | 3.04 | 1757.10 | 0.00 | 2.03 | 1412.58 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam16_block_liquid` | 3.04 | 280.88 | 0.01 | 2.03 | 301.69 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam16_block_lut` | 3.04 | 44.47 | 0.07 | 2.03 | 45.96 | 0.04 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_demap_qam256_block` | 6.06 | 6.06 | 1.00 | 4.29 | 4.29 | 1.00 | sdrk fallback |
| `symbol_demap_qam64_block` | 4.55 | 6285.06 | 0.00 | 3.04 | 5351.51 | 0.00 | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam64_block_liquid` | 4.55 | 337.22 | 0.01 | 3.04 | 361.39 | 0.01 | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam64_block_lut` | 4.55 | 44.47 | 0.10 | 3.04 | 45.96 | 0.07 | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_map_block` | 7.08 | 22.59 | 0.31 | 9.07 | 21.73 | 0.42 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_liquid` | 7.08 | 25.31 | 0.28 | 9.07 | 26.29 | 0.35 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qam64` | 10.09 | 29.68 | 0.34 | 12.07 | 28.81 | 0.42 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qam64_liquid` | 10.09 | 29.39 | 0.34 | 12.07 | 30.36 | 0.40 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qpsk` | 7.08 | 15.50 | 0.46 | 8.81 | 14.65 | 0.60 | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qpsk_liquid` | 7.08 | 21.23 | 0.33 | 8.81 | 22.22 | 0.40 | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_sync` | 35.91 | 35.91 | 1.00 | 22.00 | 22.00 | 1.00 | sdrk fallback |
| `symbol_sync_fallback` | 35.91 | 37.11 | 0.97 | 22.00 | 29.01 | 0.76 | GNU Radio symbol_sync_cc (symbol_sync_cc_impl.cc of gr-digital, TED_GARDNER, IR_MMSE_8TAP, generic compiler flags; reference does different work: an 8-tap MMSE interpolator against our 4-point Lagrange polynomial) |
| `symbol_sync_real` | 17.29 | 17.29 | 1.00 | 10.16 | 10.16 | 1.00 | sdrk fallback |
| `xcorr_fft_n4096` | 51.90 | 144.73 | 0.36 | 35.61 | 112.25 | 0.32 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n4096_fallback` | 113.30 | 451.32 | 0.25 | 112.39 | 482.83 | 0.23 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |
| `xcorr_fft_n512` | 44.05 | 113.11 | 0.39 | 30.56 | 96.13 | 0.32 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10 |
| `xcorr_fft_n512_fallback` | 89.58 | 319.42 | 0.28 | 85.31 | 339.49 | 0.25 | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |

- Not measured at this level: `real_to_complex_airspy`.

<!-- cycle-tables:end -->

The details of what each kernel and block does, the workloads of the entries that have none in the table of [what each kernel is compared with](#references-d32), the tunings and the rounds that led to these numbers are in [decisions-log.md](decisions-log.md) (search the entry name or its milestone) and in the milestone specs. Entries of one milestone are listed in the milestone file.

## `simd128` instructions (wasm32, no core model)

Instructions per sample (wasmtime fuel, differential) of our kernel divided by those of the comparison, from the committed `benches/results/wasm32.json`; this is the one level whose gate stays in instructions (`cargo xtask count --platform wasm32`, nightly job `count_wasm32`; limits 1.10 `release`, 1.50 `dev`, `info` not judged). The table is **generated** by `python3 -I benches/cycle_tables.py` from that file. The comparison at `simd128` is VOLK's `generic` protokernels compiled for wasm32 with `simd128` (it has no wasm32 machine), CMSIS-DSP generic C, KFR and the other references of [References](#references-d32) built with `-msimd128`; a status other than `release` and its waiver are in `benches/limits.json`. An entry whose SIMD path equals the fallback shows the ratio 1.00 `info`. `waived` is a ratio above the limit under a waiver, `fail` a `release` entry that does not pass: its ratio exceeds the limit, or the comparison is more accurate than ours (`oscillate_fallback` against CMSIS-DSP `arm_sin_cos_f32`).

<!-- simd128-instructions:begin -->

| Entry | ours (instructions/sample) | comparison (instructions/sample) | ratio (ours/comparison) | status | comparison |
| --- | ---: | ---: | ---: | --- | --- |
| `convert_cq15_cs8` | 2.84 | 4.78 | 0.59 |  | volk_8i_convert_16i |
| `convert_cq15_cs8_fallback` | 4.81 | 22.27 | 0.22 |  | CMSIS-DSP 1.18.0 arm_q7_to_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_cq15_cu8` | 3.37 | 4.78 | 0.71 |  | volk_8i_convert_16i on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cq15_cu8_fallback` | 5.31 | 22.27 | 0.24 |  | CMSIS-DSP 1.18.0 arm_q7_to_q15 on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_cs16` | 4.34 | 10.03 | 0.43 |  | volk_16i_s32f_convert_32f |
| `convert_cs16_fallback` | 10.07 | 12.28 | 0.82 |  | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_cs8` | 4.22 | 11.03 | 0.38 |  | volk_8i_s32f_convert_32f |
| `convert_cs8_fallback` | 11.06 | 12.03 | 0.92 |  | CMSIS-DSP 1.18.0 arm_q7_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_cu8` | 3.96 | 11.03 | 0.36 |  | volk_8i_s32f_convert_32f on the bytes with the sign bit flipped (the CU8 conversion without its offset) |
| `convert_cu8_fallback` | 12.06 | 12.03 | 1.00 |  | CMSIS-DSP 1.18.0 arm_q7_to_float on the bytes with the sign bit flipped (the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_out_cs16` | 5.67 | 29.03 | 0.20 |  | volk_32f_s32f_convert_16i |
| `convert_out_cs16_fallback` | 17.57 | 58.52 | 0.30 | info | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_out_cs8` | 9.17 | 29.03 | 0.32 |  | volk_32f_s32f_convert_8i |
| `convert_out_cs8_fallback` | 17.57 | 58.52 | 0.30 | info | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_out_cu8` | 15.17 | 29.03 | 0.52 |  | volk_32f_s32f_x2_convert_8u |
| `convert_out_cu8_fallback` | 26.57 | 58.52 | 0.45 | info | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_out_real_rs16` | 2.88 | 14.53 | 0.20 |  | volk_32f_s32f_convert_16i |
| `convert_out_real_rs16_fallback` | 8.82 | 29.27 | 0.30 | info | CMSIS-DSP 1.18.0 arm_float_to_q15 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_out_real_rs8` | 4.62 | 14.53 | 0.32 |  | volk_32f_s32f_convert_8i |
| `convert_out_real_rs8_fallback` | 8.82 | 29.27 | 0.30 | info | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_out_real_ru8` | 7.62 | 14.53 | 0.52 |  | volk_32f_s32f_x2_convert_8u |
| `convert_out_real_ru8_fallback` | 13.31 | 29.27 | 0.45 | info | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `convert_real_rs16` | 2.20 | 5.03 | 0.44 |  | volk_16i_s32f_convert_32f |
| `convert_real_rs16_fallback` | 5.06 | 6.15 | 0.82 |  | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `rotate` | 15.97 | 79.23 | 0.20 | info | volk_32fc_s32fc_x2_rotator2_32fc |
| `rotate_cq15` | 28.29 | 39.16 | 0.72 | info | sdrk fallback |
| `rotate_cq15_fallback` | 39.16 | 229.23 | 0.17 |  | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `rotate_fallback` | 35.29 | 92.21 | 0.38 |  | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `scramble_block` | 44.45 | 44.45 | 1.00 | info | sdrk fallback |
| `scramble_c32_b1` | 8.07 | 21.68 | 0.37 | info | +-1.0 sequence from the index bits into a buffer of 256 + volk_32fc_32f_multiply_32fc (VOLK 3.3.0 dispatcher; the form of srsRAN srsran_scrambling_c) |
| `scramble_c32_b1_fallback` | 11.79 | 21.70 | 0.54 |  | +-1.0 sequence from the index bits into a buffer of 256 + VOLK 3.3.0 volk_32fc_32f_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b2` | 19.57 | 85.43 | 0.23 | info | phasor lookup into a buffer of 256 + volk_32fc_x2_multiply_32fc (VOLK 3.3.0 dispatcher) |
| `scramble_c32_b2_dvbs2` | 19.57 | 41.78 | 0.47 | info | GNU Radio dvbs2_physical_cc PL scrambling, the per-symbol switch on R_n (benches/refs/gr-dvbs2-plscramble.c, main at ec1bf15; reference does different work; -O3) |
| `scramble_c32_b2_fallback` | 31.04 | 85.44 | 0.36 |  | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_c32_b8` | 27.07 | 85.43 | 0.32 | info | phasor lookup into a buffer of 256 + volk_32fc_x2_multiply_32fc (VOLK 3.3.0 dispatcher) |
| `scramble_c32_b8_fallback` | 27.79 | 85.44 | 0.33 |  | phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off) |
| `scramble_cq15_b2` | 13.57 | 50.15 | 0.27 |  | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `scramble_cq15_b2_fallback` | 14.79 | 50.15 | 0.29 |  | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `scramble_cq15_b8` | 29.82 | 50.15 | 0.59 | info | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `scramble_cq15_b8_fallback` | 26.79 | 50.15 | 0.53 | info | CMSIS-DSP 1.18.0 phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir` | 209.41 | 1980.03 | 0.11 | info | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 841.39 | 934.94 | 0.90 | info | sdrk fallback |
| `fir_cq15_fallback` | 934.94 | 785.69 | 1.19 | dev | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir_decimate` | 55.44 | 383.41 | 0.14 | info | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 185.86 | 204.82 | 0.91 | info | sdrk fallback |
| `fir_decimate_cq15_fallback` | 204.82 | 321.21 | 0.64 |  | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir_decimate_fallback` | 147.83 | 138.12 | 1.07 | waived | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir_fallback` | 509.86 | 676.70 | 0.75 |  | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir_interpolate` | 236.37 | 2024.03 | 0.12 | info | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 638.42 | 638.42 | 1.00 | info | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 638.42 | 1183.72 | 0.54 |  | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir_interpolate_fallback` | 603.69 | 592.97 | 1.02 |  | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir_interpolate_real` | 147.91 | 947.03 | 0.16 | info | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 351.14 | 296.50 | 1.18 | dev | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir_interpolate_real_q15` | 451.84 | 451.84 | 1.00 | info | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 451.84 | 591.87 | 0.76 |  | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fir_interpolate_short` | 81.43 | 426.53 | 0.19 | info | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_short_fallback` | 176.00 | 127.10 | 1.38 | dev | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 8 (80 taps), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `resample` | 309.31 | 2026.03 | 0.15 | info | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 977.01 | 1075.61 | 0.91 | info | sdrk fallback |
| `resample_cq15_fallback` | 1075.61 | 1180.20 | 0.91 | info | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `resample_fallback` | 781.66 | 589.97 | 1.32 | info | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `resample_real` | 179.99 | 949.03 | 0.19 | info | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 687.91 | 295.00 | 2.33 | info | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `resample_real_q15` | 598.41 | 875.74 | 0.68 | info | sdrk fallback |
| `resample_real_q15_fallback` | 875.74 | 590.11 | 1.48 | info | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `mag` | 6.86 | 8.53 | 0.80 |  | volk_32fc_magnitude_32f |
| `mag_fallback` | 8.28 | 10.91 | 0.76 |  | CMSIS-DSP 1.18.0 arm_cmplx_mag_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `mag_q15` | 9.03 | 18.28 | 0.49 |  | volk_16ic_magnitude_16i |
| `mag_q15_fallback` | 12.90 | 208.59 | 0.06 |  | CMSIS-DSP 1.18.0 arm_cmplx_mag_q15 (Q2.14, arm_sqrt_q31), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `mag_sq` | 4.96 | 8.28 | 0.60 |  | volk_32fc_magnitude_squared_32f |
| `mag_sq_fallback` | 8.03 | 9.35 | 0.86 |  | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `mag_sq_q15` | 3.02 | 7.53 | 0.40 | info | sdrk fallback |
| `mag_sq_q15_fallback` | 7.53 | 7.85 | 0.96 |  | CMSIS-DSP 1.18.0 arm_cmplx_mag_squared_q15 (Q3.13, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `quad_demod` | 36.15 | 201.21 | 0.18 | info | volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f |
| `quad_demod_cq15` | 40.79 | 332.25 | 0.12 | info | sdrk fallback |
| `quad_demod_cq15_fallback` | 332.25 | 603.82 | 0.55 |  | CMSIS-DSP arm_cmplx_mult_cmplx_q15 + arm_atan2_q15 + arm_scale_q15 (generic C), conjugate in a plain loop |
| `quad_demod_fallback` | 188.90 | 155.40 | 1.22 | info | GNU Radio quadrature_demod_cf (gr-analog, fast_atan2f, generic compiler flags) |
| `fm_mod` | 34.85 | 125.97 | 0.28 | info | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `fm_mod_cq15` | 48.23 | 67.27 | 0.72 | info | sdrk fallback |
| `fm_mod_cq15_fallback` | 67.27 | 116.02 | 0.58 |  | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample of an integer phase accumulator, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fm_mod_fallback` | 61.27 | 125.97 | 0.49 |  | GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags) |
| `squelch` | 63.60 | 98.71 | 0.64 |  | GNU Radio pwr_squelch_cc (squelch_base_cc_impl.cc of gr-analog, generic compiler flags) |
| `squelch_cq15` | 67.05 | 67.05 | 1.00 | info | sdrk fallback |
| `window` | 7.73 | 10.79 | 0.72 | info | volk_32fc_32f_multiply_32fc |
| `window_cq15` | 3.95 | 9.91 | 0.40 | info | sdrk fallback |
| `window_cq15_fallback` | 9.91 | 14.93 | 0.66 |  | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `window_fallback` | 10.79 | 26.79 | 0.40 |  | CMSIS-DSP 1.18.0 arm_cmplx_mult_real_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `power_db` | 28.78 | 98.80 | 0.29 | info | volk_32f_log2_32f + volk_32f_s32f_multiply_32f + volk_32f_s32f_add_32f |
| `power_db_fallback` | 28.28 | 77.68 | 0.36 |  | CMSIS-DSP 1.18.0 arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `power_db_q15` | 29.03 | 28.28 | 1.03 | info | sdrk fallback |
| `power_db_q15_fallback` | 28.28 | 86.83 | 0.33 | info | CMSIS-DSP 1.18.0 arm_q15_to_float + arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `agc` | 64.61 | 64.05 | 1.01 |  | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `agc_cq15` | 137.07 | 137.07 | 1.00 | info | sdrk fallback |
| `agc_fallback` | 64.11 | 64.05 | 1.00 |  | GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags) |
| `dc_block` | 26.53 | 26.53 | 1.00 | info | sdrk fallback |
| `dc_block_cq15` | 53.03 | 53.03 | 1.00 | info | sdrk fallback |
| `dc_block_fallback` | 26.53 | 277.07 | 0.10 |  | liquid-dsp 1.8.3 iirfilt_crcf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real` | 26.81 | 27.54 | 0.97 | info | sdrk fallback |
| `dc_block_real_fallback` | 27.54 | 240.06 | 0.11 |  | liquid-dsp 1.8.3 iirfilt_rrrf_create_dc_blocker (direct form II, portable dot product, -O3) |
| `dc_block_real_q15` | 49.03 | 49.03 | 1.00 | info | sdrk fallback |
| `deemphasis` | 24.81 | 27.53 | 0.90 | info | sdrk fallback |
| `deemphasis_fallback` | 27.53 | 150.07 | 0.18 | waived | GNU Radio fm_deemph: iir_filter_ffd with the bilinear taps of fm_emph.py (double taps and accumulator, generic compiler flags) |
| `deemphasis_q15` | 42.03 | 42.03 | 1.00 | info | sdrk fallback |
| `biquad` | 43.04 | 96.78 | 0.44 |  | KFR 7.2.0 kfr::iir, iir_state<float, 2> (sse41, -ffp-contract=off), two calls on planar re and im (deinterleaving not counted) |
| `biquad_cq15` | 70.53 | 203.11 | 0.35 | info | sdrk fallback |
| `biquad_cq15_fallback` | 203.11 | 272.13 | 0.75 |  | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `biquad_fallback` | 68.47 | 118.83 | 0.58 |  | CMSIS-DSP 1.18.0 arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `biquad_real` | 38.93 | 48.39 | 0.80 |  | KFR 7.2.0 kfr::iir, iir_state<float, 2> (sse41, -ffp-contract=off) |
| `biquad_real_fallback` | 50.79 | 57.94 | 0.88 |  | CMSIS-DSP 1.18.0 arm_biquad_cascade_df2T_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `biquad_real_q15` | 41.21 | 96.08 | 0.43 | info | sdrk fallback |
| `biquad_real_q15_fallback` | 96.08 | 136.07 | 0.71 |  | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `dot` | 7.20 | 29.55 | 0.24 | info | volk_32fc_32f_dot_prod_32fc |
| `dot_complex` | 13.20 | 40.55 | 0.33 | info | volk_32fc_x2_dot_prod_32fc |
| `dot_complex_fallback` | 13.20 | 35.30 | 0.37 |  | CMSIS-DSP 1.18.0 arm_cmplx_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `dot_cq15` | 6.74 | 17.56 | 0.38 | info | sdrk fallback |
| `dot_cq15_fallback` | 17.56 | 25.58 | 0.69 |  | CMSIS-DSP 1.18.0 arm_dot_prod_q15 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `dot_fallback` | 16.55 | 25.57 | 0.65 |  | CMSIS-DSP 1.18.0 arm_dot_prod_f32 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `dot_real` | 3.35 | 6.17 | 0.54 | info | sdrk fallback |
| `dot_real_fallback` | 6.17 | 12.79 | 0.48 |  | CMSIS-DSP 1.18.0 arm_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `dot_real_q15` | 3.23 | 11.55 | 0.28 | info | sdrk fallback |
| `dot_real_q15_fallback` | 11.55 | 12.79 | 0.90 |  | CMSIS-DSP 1.18.0 arm_dot_prod_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `scale` | 5.65 | 6.16 | 0.92 | info | sdrk fallback |
| `scale_cq15` | 2.81 | 8.15 | 0.34 | info | sdrk fallback |
| `scale_cq15_fallback` | 8.15 | 8.28 | 0.98 |  | CMSIS-DSP 1.18.0 arm_scale_q15 (shift 0, truncating) on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `scale_fallback` | 6.16 | 5.41 | 1.14 | dev | CMSIS-DSP 1.18.0 arm_scale_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `add` | 7.17 | 8.79 | 0.82 |  | volk_32fc_x2_add_32fc |
| `add_cq15` | 2.33 | 10.78 | 0.22 |  | volk_16i_x2_add_saturated_16i on the 2 n values |
| `add_cq15_fallback` | 2.33 | 3.62 | 0.64 |  | CMSIS-DSP 1.18.0 arm_add_q15 on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `add_fallback` | 7.79 | 31.54 | 0.25 |  | CMSIS-DSP 1.18.0 arm_add_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `add_real` | 3.61 | 6.04 | 0.60 |  | volk_32f_x2_add_32f |
| `add_real_fallback` | 6.04 | 15.79 | 0.38 |  | CMSIS-DSP 1.18.0 arm_add_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `add_real_q15` | 1.19 | 5.41 | 0.22 |  | volk_16i_x2_add_saturated_16i |
| `add_real_q15_fallback` | 1.19 | 1.83 | 0.65 |  | CMSIS-DSP 1.18.0 arm_add_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `mul` | 16.79 | 56.03 | 0.30 |  | volk_32fc_x2_multiply_32fc |
| `mul_cq15` | 15.17 | 15.17 | 1.00 | info | sdrk fallback |
| `mul_cq15_fallback` | 15.17 | 20.37 | 0.74 |  | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_q15 (Q3.13, truncating, scaled by 4), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `mul_fallback` | 15.29 | 18.62 | 0.82 |  | CMSIS-DSP 1.18.0 arm_cmplx_mult_cmplx_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `mul_real` | 3.61 | 6.04 | 0.60 |  | volk_32f_x2_multiply_32f |
| `mul_real_fallback` | 6.04 | 15.79 | 0.38 |  | CMSIS-DSP 1.18.0 arm_mult_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `mul_real_q15` | 1.83 | 3.77 | 0.49 | info | sdrk fallback |
| `mul_real_q15_fallback` | 3.77 | 3.93 | 0.96 |  | CMSIS-DSP 1.18.0 arm_mult_q15 (truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `interleave` | 5.89 | 7.79 | 0.76 |  | volk_32f_x2_interleave_32fc |
| `interleave_cq15` | 2.97 | 3.92 | 0.76 | info | sdrk fallback |
| `interleave_cq15_fallback` | 3.92 | 3.92 | 1.00 |  | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic on int16 (VOLK has no 16-bit interleave), benches/refs/volk-generic.c (-O3 -ffp-contract=off, simd128) |
| `interleave_fallback` | 7.79 | 7.79 | 1.00 |  | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, simd128) |
| `deinterleave` | 5.89 | 7.79 | 0.76 |  | volk_32fc_deinterleave_32f_x2 |
| `deinterleave_cq15` | 2.97 | 3.92 | 0.76 |  | volk_16ic_deinterleave_16i_x2 |
| `deinterleave_cq15_fallback` | 3.91 | 3.92 | 1.00 |  | VOLK 3.3.0 volk_16ic_deinterleave_16i_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, simd128) |
| `deinterleave_fallback` | 7.79 | 7.80 | 1.00 |  | VOLK 3.3.0 volk_32fc_deinterleave_32f_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, simd128) |
| `oscillate` | 22.04 | 28.28 | 0.78 | info | sdrk fallback |
| `oscillate_cq15` | 17.54 | 33.66 | 0.52 | info | sdrk fallback |
| `oscillate_cq15_fallback` | 33.66 | 105.01 | 0.32 |  | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample (512-entry table, linear interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `oscillate_fallback` | 28.28 | 175.02 | 0.16 | fail | CMSIS-DSP 1.18.0 arm_sin_cos_f32 of the angle in degrees per sample (512-entry sine table, cubic Hermite interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fft_1024` | 58.57 | 91.17 | 0.64 |  | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 220.35 | 209.86 | 1.05 |  | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fft_4096` | 67.64 | 105.58 | 0.64 |  | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 244.92 | 221.28 | 1.11 | dev | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fft_64` | 35.50 | 63.23 | 0.56 |  | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 121.02 | 117.59 | 1.03 |  | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fft_cq15_1024` | 48.92 | 460.87 | 0.11 |  | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 43.57 | 460.87 | 0.09 |  | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 405.94 | 378.19 | 1.07 |  | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fft_cq15_1024_fallback` | 537.22 | 378.19 | 1.42 | info | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `fft_cq15_1024_full_scale` | 46.17 | 460.87 | 0.10 |  | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 41.91 | 59.40 | 0.71 |  | realfft on rustfft (RealFftPlanner, forward, out of place) |
| `rfft_1024_fallback` | 105.27 | 98.59 | 1.07 |  | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `am_mod` | 10.54 | 10.54 | 1.00 | info | sdrk fallback |
| `am_mod_cq15` | 3.92 | 8.80 | 0.45 | info | sdrk fallback |
| `am_mod_cq15_fallback` | 8.80 | 31.69 | 0.28 |  | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `am_mod_fallback` | 10.54 | 46.02 | 0.23 |  | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 1700.29 | 1700.29 | 1.00 | info | sdrk fallback |
| `band_pulse_detect_block` | 324.05 | 324.05 | 1.00 | info | sdrk fallback |
| `bit_descramble_block` | 10.27 | 42.02 | 0.24 |  | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_descramble_in_place_block` | 22.16 | 42.02 | 0.53 |  | GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_block` | 14.63 | 42.02 | 0.35 |  | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `bit_scramble_in_place_block` | 15.63 | 42.02 | 0.37 |  | GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags) |
| `channel_bank_channelize` | 534.31 | 534.31 | 1.00 | info | sdrk fallback |
| `channel_bank_ddc` | 1657.26 | 1657.26 | 1.00 | info | sdrk fallback |
| `channel_select_block` | 10.54 | 10.54 | 1.00 | info | sdrk fallback |
| `channelize` | 270.93 | 949.71 | 0.29 |  | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `channelize_block` | 270.93 | 270.93 | 1.00 | info | sdrk fallback |
| `channelize_fallback` | 436.07 | 949.71 | 0.46 |  | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `correlate_block` | 186.81 | 186.81 | 1.00 | info | sdrk fallback |
| `costas` | 222.04 | 222.04 | 1.00 | info | sdrk fallback |
| `costas_fallback` | 222.04 | 255.33 | 0.87 |  | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `costas_qam1024` | 289.73 | 289.73 | 1.00 | info | sdrk fallback |
| `costas_qam1024_fallback` | 289.54 | 1062.64 | 0.27 |  | GNU Radio constellation_receiver_cb with constellation_rect 1024-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam16` | 346.36 | 346.36 | 1.00 | info | sdrk fallback |
| `costas_qam16_fallback` | 346.36 | 471.32 | 0.73 |  | GNU Radio constellation_receiver_cb with constellation_rect 16-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam256` | 288.45 | 288.45 | 1.00 | info | sdrk fallback |
| `costas_qam256_fallback` | 288.45 | 507.71 | 0.57 |  | GNU Radio constellation_receiver_cb with constellation_rect 256-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `costas_qam64` | 346.31 | 346.31 | 1.00 | info | sdrk fallback |
| `costas_qam64_fallback` | 346.31 | 473.53 | 0.73 |  | GNU Radio constellation_receiver_cb with constellation_rect 64-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 21.79 | 26.91 | 0.81 |  | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 19.79 | 22.88 | 0.86 |  | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 23.79 | 26.91 | 0.88 |  | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 19.79 | 22.89 | 0.86 |  | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 1014.47 | 1014.47 | 1.00 | info | sdrk fallback |
| `equalize_fallback` | 1014.47 | 1900.06 | 0.53 |  | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam1024` | 1017.48 | 1017.48 | 1.00 | info | sdrk fallback |
| `equalize_qam1024_fallback` | 1017.48 | 1952.06 | 0.52 |  | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 1024-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam16` | 1062.48 | 1062.48 | 1.00 | info | sdrk fallback |
| `equalize_qam16_fallback` | 1062.48 | 1952.06 | 0.54 |  | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam256` | 1017.48 | 1017.48 | 1.00 | info | sdrk fallback |
| `equalize_qam256_fallback` | 1017.48 | 1952.06 | 0.52 |  | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 256-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `equalize_qam64` | 1061.48 | 1061.48 | 1.00 | info | sdrk fallback |
| `equalize_qam64_fallback` | 1061.48 | 1952.06 | 0.54 |  | GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 61.50 | 61.50 | 1.00 | info | sdrk fallback |
| `frame_decode_block` | 60.51 | 60.51 | 1.00 | info | sdrk fallback |
| `frame_decode_four_block` | 166.41 | 166.41 | 1.00 | info | sdrk fallback |
| `frame_encode_block` | 8.97 | 8.97 | 1.00 | info | sdrk fallback |
| `gauss_noise_c32` | 66.68 | 343.24 | 0.19 | info | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fallback` | 138.04 | 343.24 | 0.40 | info | liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_c32_fastnoise` | 66.68 | 58.01 | 1.15 | info | GNU Radio fastnoise_source_c, amplitude sigma * sqrt(2) (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32` | 33.42 | 197.64 | 0.17 | info | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fallback` | 69.04 | 197.64 | 0.35 | info | liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3) |
| `gauss_noise_f32_fastnoise` | 33.42 | 39.02 | 0.86 | info | GNU Radio fastnoise_source_f, amplitude sigma (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3) |
| `gauss_noise_f32_ziggurat` | 33.42 | 127.08 | 0.26 | info | rand_distr 0.5.1 StandardNormal as f32 times sigma over rand 0.9.5 SmallRng (Ziggurat in f64 with rejection: a variable number of draws per value, no random access, no bit-exact sequence across platforms; reference does different work) |
| `gauss_noise_q15` | 36.30 | 99.05 | 0.37 | info | sdrk fallback |
| `gauss_noise_q15_fallback` | 99.05 | 99.05 | 1.00 | info | sdrk fallback |
| `halfband_decimate` | 92.52 | 440.36 | 0.21 |  | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, simd128) |
| `halfband_decimate_cq15` | 120.65 | 134.25 | 0.90 | info | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 134.25 | 188.86 | 0.71 |  | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `halfband_decimate_fallback` | 122.90 | 440.36 | 0.28 |  | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, simd128) |
| `halfband_decimate_real` | 42.94 | 263.60 | 0.16 |  | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, simd128) |
| `halfband_decimate_real_fallback` | 86.29 | 263.60 | 0.33 |  | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, simd128) |
| `halfband_decimate_real_q15` | 54.98 | 100.19 | 0.55 | info | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 100.19 | 94.37 | 1.06 |  | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `halfband_ref_fir_decimate` | 102.07 | 269.35 | 0.38 | info | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 476.03 | 377.01 | 1.26 | info | sdrk fallback |
| `halfband_ref_shifted` | 128.04 | 158.42 | 0.81 | info | sdrk fallback |
| `halfband_ref_shifted_cq15` | 164.17 | 177.77 | 0.92 | info | sdrk fallback |
| `iq_offset_block` | 40.77 | 40.77 | 1.00 | info | sdrk fallback |
| `morse_decode_block` | 149.15 | 149.15 | 1.00 | info | sdrk fallback |
| `morse_encode_block` | 49.57 | 49.57 | 1.00 | info | sdrk fallback |
| `noise_blank` | 77.93 | 77.93 | 1.00 | info | sdrk fallback |
| `noise_blank_cq15` | 74.15 | 74.15 | 1.00 | info | sdrk fallback |
| `noise_block` | 66.69 | 66.69 | 1.00 | info | sdrk fallback |
| `preemph_limit` | 98.56 | 98.56 | 1.00 | info | sdrk fallback |
| `preemph_limit_q15` | 448.58 | 448.58 | 1.00 | info | sdrk fallback |
| `preemph_limit_stereo` | 126.84 | 126.84 | 1.00 | info | sdrk fallback |
| `psk_demod_block` | 367.78 | 2233.43 | 0.16 | info | GNU Radio fir_filter_ccf + symbol_sync_cc + costas_loop_cc + constellation_decoder_cb (the sum of the blocks of the receiver chain, generic compiler flags; no acquisition and no lock watches) |
| `psk_mod_block` | 96.37 | 5353.10 | 0.02 | info | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's generic dot product (the wasm32 build of VOLK has no SIMD) |
| `psk_mod_block_liquid` | 96.37 | 426.59 | 0.23 | info | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `pulse_match_block` | 266.12 | 2475.18 | 0.11 | info | GNU Radio fir_filter_ccf (fir_filter_ccf_impl.cc of gr-filter) with VOLK's generic dot product (the wasm32 build of VOLK has no SIMD), generic compiler flags |
| `pulse_match_block_liquid` | 266.12 | 2291.19 | 0.12 | info | liquid-dsp 1.8.3 firfilt_crcf (firfilt.proto.c), the SIMD dot product of the level (the portable loop at simd128) |
| `pulse_shape_block` | 88.47 | 387.30 | 0.23 | info | GNU Radio interp_fir_filter_ccf (interp_fir_filter_impl.cc of gr-filter) with VOLK's generic dot product (the wasm32 build of VOLK has no SIMD), generic compiler flags |
| `pulse_shape_block_liquid` | 88.47 | 414.07 | 0.21 | info | liquid-dsp 1.8.3 firinterp_crcf (firinterp.proto.c, firpfb.proto.c), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_acquire_block` | 348.39 | 348.39 | 1.00 | info | sdrk fallback |
| `qam_demod_block` | 540.18 | 2501.07 | 0.22 | info | GNU Radio fir_filter_ccf + symbol_sync_cc + linear_equalizer + constellation_receiver_cb (the sum of the blocks of the receiver chain, generic compiler flags and VOLK generic dot products; no acquisition, no blind equalizer stage and no lock watches) |
| `qam_demod_qam256_block` | 529.05 | 529.05 | 1.00 | info | sdrk fallback |
| `qam_demod_search_block` | 587.01 | 587.01 | 1.00 | info | sdrk fallback |
| `qam_frame_decode_block` | 624.74 | 624.74 | 1.00 | info | sdrk fallback |
| `qam_frame_encode_block` | 97.81 | 97.81 | 1.00 | info | sdrk fallback |
| `qam_mod_block` | 93.11 | 5357.73 | 0.02 | info | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's generic dot product (the wasm32 build of VOLK has no SIMD) |
| `qam_mod_block_liquid` | 93.11 | 427.59 | 0.22 | info | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_mod_qam256_block` | 95.80 | 5366.98 | 0.02 | info | GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's generic dot product (the wasm32 build of VOLK has no SIMD) |
| `qam_mod_qam256_block_liquid` | 95.80 | 433.34 | 0.22 | info | liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128) |
| `qam_soft_bits_qam1024` | 39.06 | 39.04 | 1.00 | info | sdrk fallback |
| `qam_soft_bits_qam16` | 14.80 | 4080.04 | 0.00 | info | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam16_liquid` | 14.80 | 1310.32 | 0.01 | info | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam16_lut` | 14.80 | 139.04 | 0.11 | info | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qam256` | 32.56 | 32.54 | 1.00 | info | sdrk fallback |
| `qam_soft_bits_qam64` | 23.80 | 17702.04 | 0.00 | info | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `qam_soft_bits_qam64_liquid` | 23.80 | 1800.52 | 0.01 | info | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `qam_soft_bits_qam64_lut` | 23.80 | 147.04 | 0.16 | info | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `qam_soft_bits_qpsk` | 6.30 | 9.54 | 0.66 | info | sdrk fallback |
| `rds_decode_block` | 84.97 | 84.97 | 1.00 | info | sdrk fallback |
| `rds_demod_block` | 338.75 | 338.75 | 1.00 | info | sdrk fallback |
| `rds_shape` | 168.41 | 169.41 | 0.99 | info | sdrk fallback |
| `real_to_complex` | 52.87 | 275.00 | 0.19 |  | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, simd128) |
| `real_to_complex_cq15` | 67.97 | 126.54 | 0.54 | info | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, simd128) |
| `real_to_complex_cq15_fallback` | 120.47 | 126.54 | 0.95 | info | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, simd128) |
| `real_to_complex_fallback` | 103.61 | 274.99 | 0.38 |  | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, simd128) |
| `snr_estimate_block` | 46.15 | 570.08 | 0.08 | info | GNU Radio mpsk_snr_est_cc(SNR_EST_M2M4, 4096, 0.001): the moments \|x\|^2 and \|x\|^4 averaged in double, six cabsf per symbol (reference does different work: no decisions, a PSK kurtosis; generic compiler flags) |
| `snr_estimate_block_svr` | 46.15 | 215.08 | 0.21 | info | GNU Radio mpsk_snr_est_cc(SNR_EST_SVR, 4096, 0.001): \|x\|^4 and \|x_n\|^2 \|x_(n-1)\|^2 averaged in double, two cabsf per symbol (reference does different work: no decisions, a PSK constellation; generic compiler flags) |
| `stereo_encode` | 37.54 | 37.54 | 1.00 | info | sdrk fallback |
| `stereo_encode_rds` | 53.05 | 53.05 | 1.00 | info | sdrk fallback |
| `stereo_matrix` | 11.53 | 33.63 | 0.34 |  | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 5.40 | 5.40 | 1.00 | info | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 5.40 | 5.47 | 0.99 |  | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `stereo_matrix_fallback` | 10.78 | 47.34 | 0.23 |  | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `stereo_pll_fallback` | 190.06 | 143.86 | 1.32 | waived | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, simd128) |
| `stereo_pll_q15_fallback` | 183.05 | 228.04 | 0.80 | info | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, simd128) |
| `symbol_demap_qam16_block` | 14.80 | 4080.03 | 0.00 | info | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam16_block_liquid` | 14.80 | 1310.23 | 0.01 | info | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam16_block_lut` | 14.80 | 139.03 | 0.11 | info | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_demap_qam256_block` | 32.56 | 32.56 | 1.00 | info | sdrk fallback |
| `symbol_demap_qam64_block` | 23.80 | 17702.03 | 0.00 | info | GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code) |
| `symbol_demap_qam64_block_liquid` | 23.80 | 1800.37 | 0.01 | info | liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol) |
| `symbol_demap_qam64_block_lut` | 23.80 | 147.03 | 0.16 | info | GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation) |
| `symbol_map_block` | 33.53 | 140.03 | 0.24 | info | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_liquid` | 33.53 | 108.02 | 0.31 | info | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qam64` | 40.53 | 177.03 | 0.23 | info | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qam64_liquid` | 40.53 | 150.02 | 0.27 | info | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_map_block_qpsk` | 59.56 | 103.03 | 0.58 | info | GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags |
| `symbol_map_block_qpsk_liquid` | 59.56 | 100.02 | 0.60 | info | liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper |
| `symbol_sync` | 101.80 | 101.80 | 1.00 | info | sdrk fallback |
| `symbol_sync_fallback` | 101.80 | 156.98 | 0.65 | info | GNU Radio symbol_sync_cc (symbol_sync_cc_impl.cc of gr-digital, TED_GARDNER, IR_MMSE_8TAP, generic compiler flags; reference does different work: an 8-tap MMSE interpolator against our 4-point Lagrange polynomial) |
| `symbol_sync_real` | 46.96 | 46.96 | 1.00 | info | sdrk fallback |
| `xcorr_fft_n4096` | 217.07 | 2752.17 | 0.08 | info | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |
| `xcorr_fft_n4096_fallback` | 693.03 | 2752.17 | 0.25 |  | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |
| `xcorr_fft_n512` | 186.81 | 1977.98 | 0.09 | info | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |
| `xcorr_fft_n512_fallback` | 517.29 | 1977.98 | 0.26 |  | liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT |

<!-- simd128-instructions:end -->

## Efficiency verdicts without a SIMD library (issue #169)

The ratios below were taken from the whole-loop simulations of the local runs of D2544 and D2547 (cycles per sample, reference model / counter model); the cycle tables above are the current numbers of the CI measurement.

**Efficiency verdict of `costas`, `costas_qam16` and `costas_qam64` at the SIMD levels (issue #169, D2544).** GNU Radio has no SIMD path for these loops (`costas_loop_cc` and `constellation_receiver_cb` are scalar), so there is no SIMD library to compare with; the facade entries stay with our fallback (the recursion of D608) and are `info`. The verdict is the efficiency of the whole-loop simulation (the port bound of the executed instructions per sample divided by the simulated cycles), reference / counter model:

| Entry | `sse2` | `sse4.2` | `avx2` | `avx512` | `neon` |
| --- | --- | --- | --- | --- | --- |
| `costas` (QPSK) | 0.27 / 0.46 | 0.27 / 0.46 | 0.31 / 0.25 | 0.31 / 0.24 | 0.34 / 0.59 |
| `costas_qam16` | 0.21 / 0.32 | 0.21 / 0.32 | 0.20 / 0.19 | 0.19 / 0.17 | 0.27 / 0.49 |
| `costas_qam64` | 0.21 / 0.32 | 0.21 / 0.32 | 0.21 / 0.21 | 0.20 / 0.18 | 0.39 / 0.64 |

Every row is bound by latency: the phase of the next symbol is the phase of this one plus the frequency integrator and the proportional correction of this symbol's detector output, so the NCO step, the derotation, the detector and the loop filter form one chain per symbol that no SIMD width shortens (the lanes would need the next symbol's phase before it exists). The distance to the port bound (16 to 70 cycles per symbol against 55 to 180 simulated) is that chain, not unused throughput of the level; the verdict is that the loops are at the length of their recursion (the chain was shortened with the same bits in D1100 and in the QAM speed round, D1416 to D1418) and no SIMD path is missing. The rows are measured with `rustc 1.99.1`; the `x86_64` rows reproduce the committed ones, the `neon` row of `costas` does not (94.19 / 55.08 against 86.13 / 54.09, D2540, D2544).

**Efficiency verdicts of the entries without a library at the SIMD levels (issue #169, D2547).** No library has `qam_frame_encode_block`, `qam_frame_decode_block`, `qam_acquire_block`, `iq_offset_block`, the framing of `sdrk-proto` (`frame_decode`, `frame_decode_block`, `frame_decode_four_block`, `frame_encode_block`), and none has a SIMD form of `symbol_sync` and `symbol_sync_real` (D562, the loop of the strobe positions; the fallback entry `symbol_sync_fallback` has GNU Radio's `symbol_sync_cc`). The verdict is the efficiency of the whole-loop simulation (port bound of the executed instructions divided by the simulated cycles) and its bottleneck, `T` throughput and `L` latency, reference / counter model:

| Entry | `sse2` | `sse4.2` | `avx2` | `avx512` | `neon` |
| --- | --- | --- | --- | --- | --- |
| `qam_frame_encode_block` | 0.85T / 0.95T | 0.85T / 0.95T | 0.84T / 0.79L | 0.84T / 0.82T | 0.89T / 0.91T |
| `qam_frame_decode_block` | 0.63L / 0.74L | 0.63L / 0.74L | 0.61L / 0.52L | 0.58L / 0.50L | 0.69L / 0.80L |
| `qam_acquire_block` | 0.59L / 0.90T | 0.59L / 0.90T | 0.74L / 0.61L | 0.78L / 0.61L | 0.82T / 0.90T |
| `iq_offset_block` | 0.94T / 1.00T | 0.85T / 0.92T | 0.82T / 0.70L | 0.82T / 0.70L | 0.66L / 0.76L |
| `symbol_sync` | 0.26L / 0.45L | 0.25L / 0.44L | 0.34L / 0.24L | 0.34L / 0.24L | 0.32L / 0.62L |
| `symbol_sync_real` | 0.28L / 0.47L | 0.27L / 0.46L | 0.32L / 0.23L | 0.33L / 0.23L | 0.36L / 0.73L |
| `frame_decode` | 0.80L / 0.84T | 0.80L / 0.84T | 0.82T / 0.79L | 0.82T / 0.79L | 0.90T / 0.88T |
| `frame_decode_block` | 0.79L / 0.81T | 0.79L / 0.81T | 0.62L / 0.75L | 0.62L / 0.75L | 0.90T / 0.93T |
| `frame_decode_four_block` | 0.74L / 0.78L | 0.74L / 0.78L | 0.78L / 0.81T | 0.81T / 0.78L | 0.83T / 0.80L |
| `frame_encode_block` | 0.33L / 0.47L | 0.33L / 0.47L | 0.33L / 0.33L | 0.33L / 0.33L | 0.60L / 0.67L |

Seven of the ten entries are at 0.5 or above on every level and model, bound by the throughput of their work or by a short dependency chain; nothing is left that a SIMD form would take: the framing is byte- and bit-serial by its definition (the sync-word registers, the whitening and the CRC of `sdrk-proto` have no SIMD form, and the code is the same at every level), and `iq_offset_block` is a delay of one component. The three entries below 0.5 are bound by latency: `symbol_sync` and `symbol_sync_real` (0.23 to 0.73) by the recursion of the strobe positions (every position is the previous one plus the period and the correction of the previous error, M38), and `frame_encode_block` (0.33 to 0.67). Two checks of the issue:

- **The QAM part of the frame entries matches `qam_mod_block` and `qam_demod_block`.** `qam_frame_encode_block` against `qam_mod_block` per output sample, reference / counter model: `sse2` 1.11 / 1.09; `sse4.2` 1.12 / 1.09; `avx2` 1.20 / 1.20; `avx512` 1.21 / 1.28; `neon` 1.08 / 1.08 (the ratio; the rest is `frame_encode` of the framing). `qam_frame_decode_block` against `qam_demod_block` per input sample, the difference in cycles (and the ratio): `sse2` +17.9 / +15.1 (1.16 / 1.15); `sse4.2` +18.0 / +14.9 (1.16 / 1.15); `avx2` +14.3 / +10.9 (1.22 / 1.15); `avx512` +13.7 / +11.2 (1.23 / 1.16); `neon` +24.7 / +23.5 (1.18 / 1.26). The QAM part is the same code with the same cost; the frame functions add 8 to 28 % when encoding and 15 to 26 % when decoding.
- **The share of the 1024-point FFT in `qam_acquire_block`** (one `fft_1024` per 512 symbols, a sample is a symbol; the `fft_1024` row of the same run, cycles per transform against the cycles of the block per 512 symbols), reference / counter model: `sse2` 7366 of 26689 = 28 %, 6541 of 22628 = 29 %; `sse4.2` 7366 of 26272 = 28 %, 6541 of 22287 = 29 %; `avx2` 2402 of 13769 = 17 %, 2318 of 16682 = 14 %; `avx512` 2079 of 13190 = 16 %, 2000 of 15547 = 13 %; `neon` 13138 of 39560 = 33 %, 8749 of 24798 = 35 %. The rest is the normalized fourth power of every symbol and the peak search. The `fft_1024` row itself is compared with FFTW and KFR in its own entry.

## PSK receiver against GNU Radio (BER and alignment changes, issue #169 §D)

`cargo run --release --bin psk_ber [symbols [first seed [seeds]]]` in `benches/kernels` (with `SDRK_REFS_DIR` as for `cargo xtask count`; the code is `benches/kernels/src/psk_ber.rs`) runs `psk_demod` and GNU Radio's receiver chain (`fir_filter_ccf` with the matched filter, `symbol_sync_cc`, `costas_loop_cc` of the order 2, 4 or 8, the decision of `constellation_decoder_cb` as the nearest point, `diff_decoder_bb` for the differential case; the C translations of `benches/refs`) on the same signals. The signal is the output of `psk_mod` for random bits (8 samples per symbol, roll-off 0.35, span 10) turned by 0.3 rad with a carrier offset of 0.003 cycles per symbol, delayed by 3 samples, with white Gaussian noise for the Eb/N0 of the row (the power of the samples measured); both receivers run `carrier_bw` 0.02 and damping 0.707, `psk_demod` its default `sync_bw` 0.005 (D2553; 0.01 before) and GNU Radio's chain `sync_bw` 0.01 (`PSK_BER_SYNC_BW` sets that of `psk_demod`), in calls of 4096 samples. The bits are compared with the sent ones per window of 2000 symbols at the best alignment (the phase ambiguity of the carrier loop and the lag in symbols, searched around the alignment of the window before); the first 6000 symbols and the last 128 (still in the filters when the input ends) are not counted; an alignment change is a window whose best alignment differs from the one before. Eight seeds (2601 to 2608) of 200000 symbols are summed per row. The theory is `Q(sqrt(2 Eb/N0))` for BPSK and QPSK, `2/3 Q(sqrt(6 Eb/N0) sin(pi/8))` for 8PSK and `exp(-Eb/N0)/2` for non-coherent differential BPSK. Both receivers lose 0.2 to 0.4 dB against the theory (the filters, the timing and the carrier loops).

Since D2553 the last 128 symbols are left out (before, the symbols still in the filters counted as errors of both receivers, about 10 symbols per seed) and the history of GNU Radio's matched filter is right in the last, shorter call (`GrFront` took it from the length of that call, which turned about 5 symbols of the chain wrong per seed); GNU Radio's BER at 8 dB is 1.96e-4 instead of 2.66e-4 (BPSK), and the tables of D2551 are replaced by these.

Block default (`sync_bw` 0.005, D2553; the PSK detector of `costas` limited to [-1, 1], D2554):

| constellation | mode | Eb/N0 dB | theory | sdrk BER | GNU Radio BER | sdrk / GNU Radio | sdrk changes per 1e5 | GNU Radio changes per 1e5 | bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BPSK | coherent | 2 | 3.751e-2 | 4.134e-2 | 3.934e-2 | 1.05 | 0.00 | 0.00 | 1536000 |
| BPSK | coherent | 4 | 1.250e-2 | 1.378e-2 | 1.306e-2 | 1.06 | 0.00 | 0.00 | 1536000 |
| BPSK | coherent | 6 | 2.388e-3 | 2.660e-3 | 2.503e-3 | 1.06 | 0.00 | 0.00 | 1536000 |
| BPSK | coherent | 8 | 1.909e-4 | 2.142e-4 | 1.960e-4 | 1.09 | 0.00 | 0.00 | 1536000 |
| QPSK | coherent | 2 | 3.751e-2 | 5.189e-2 | 5.050e-2 | 1.03 | 3.26 | 2.73 | 3072000 |
| QPSK | coherent | 4 | 1.250e-2 | 1.442e-2 | 1.416e-2 | 1.02 | 0.07 | 0.07 | 3072000 |
| QPSK | coherent | 6 | 2.388e-3 | 2.829e-3 | 2.761e-3 | 1.02 | 0.00 | 0.00 | 3072000 |
| QPSK | coherent | 8 | 1.909e-4 | 2.542e-4 | 2.497e-4 | 1.02 | 0.00 | 0.00 | 3072000 |
| 8PSK | coherent | 6 | 2.048e-2 | 3.946e-2 | 3.848e-2 | 1.03 | 5.14 | 5.92 | 4608000 |
| 8PSK | coherent | 8 | 6.181e-3 | 7.441e-3 | 7.387e-3 | 1.01 | 0.00 | 0.00 | 4608000 |
| 8PSK | coherent | 10 | 1.011e-3 | 1.288e-3 | 1.255e-3 | 1.03 | 0.00 | 0.00 | 4608000 |
| 8PSK | coherent | 12 | 6.340e-5 | 9.266e-5 | 8.984e-5 | 1.03 | 0.00 | 0.00 | 4608000 |
| BPSK | differential | 2 | 1.025e-1 | 1.078e-1 | 7.603e-2 | 1.42 | 0.00 | 0.00 | 1536000 |
| BPSK | differential | 4 | 4.056e-2 | 4.290e-2 | 2.604e-2 | 1.65 | 0.00 | 0.00 | 1536000 |
| BPSK | differential | 6 | 9.333e-3 | 9.995e-3 | 5.052e-3 | 1.98 | 0.00 | 0.00 | 1536000 |
| BPSK | differential | 8 | 9.094e-4 | 1.014e-3 | 3.997e-4 | 2.54 | 0.00 | 0.00 | 1536000 |

`sync_bw` 0.01 (the default before D2553), with the limit of D2554:

| constellation | mode | Eb/N0 dB | theory | sdrk BER | GNU Radio BER | sdrk / GNU Radio | sdrk changes per 1e5 | GNU Radio changes per 1e5 | bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BPSK | coherent | 2 | 3.751e-2 | 7.307e-2 | 3.934e-2 | 1.86 | 9.05 | 0.00 | 1536000 |
| BPSK | coherent | 4 | 1.250e-2 | 1.614e-2 | 1.306e-2 | 1.24 | 0.20 | 0.00 | 1536000 |
| BPSK | coherent | 6 | 2.388e-3 | 3.072e-3 | 2.503e-3 | 1.23 | 0.00 | 0.00 | 1536000 |
| BPSK | coherent | 8 | 1.909e-4 | 2.611e-4 | 1.960e-4 | 1.33 | 0.00 | 0.00 | 1536000 |
| QPSK | coherent | 2 | 3.751e-2 | 5.554e-2 | 5.050e-2 | 1.10 | 3.84 | 2.73 | 3072000 |
| QPSK | coherent | 4 | 1.250e-2 | 1.503e-2 | 1.416e-2 | 1.06 | 0.07 | 0.07 | 3072000 |
| QPSK | coherent | 6 | 2.388e-3 | 2.981e-3 | 2.761e-3 | 1.08 | 0.00 | 0.00 | 3072000 |
| QPSK | coherent | 8 | 1.909e-4 | 2.686e-4 | 2.497e-4 | 1.08 | 0.00 | 0.00 | 3072000 |
| 8PSK | coherent | 6 | 2.048e-2 | 3.863e-2 | 3.848e-2 | 1.00 | 5.34 | 5.92 | 4608000 |
| 8PSK | coherent | 8 | 6.181e-3 | 7.807e-3 | 7.387e-3 | 1.06 | 0.07 | 0.00 | 4608000 |
| 8PSK | coherent | 10 | 1.011e-3 | 1.351e-3 | 1.255e-3 | 1.08 | 0.00 | 0.00 | 4608000 |
| 8PSK | coherent | 12 | 6.340e-5 | 1.020e-4 | 8.984e-5 | 1.14 | 0.00 | 0.00 | 4608000 |
| BPSK | differential | 2 | 1.025e-1 | 1.373e-1 | 7.603e-2 | 1.81 | 8.46 | 0.00 | 1536000 |
| BPSK | differential | 4 | 4.056e-2 | 4.738e-2 | 2.604e-2 | 1.82 | 0.20 | 0.00 | 1536000 |
| BPSK | differential | 6 | 9.333e-3 | 1.104e-2 | 5.052e-3 | 2.18 | 0.00 | 0.00 | 1536000 |
| BPSK | differential | 8 | 9.094e-4 | 1.148e-3 | 3.997e-4 | 2.87 | 0.00 | 0.00 | 1536000 |

Timing settling: `cargo run --release --bin psk_ber -- acquisition [symbols [first seed [seeds [sync_bw...]]]]` sends 8000 symbols at a high Eb/N0 (12 dB, 8PSK 18 dB) for each delay of 0 to 7 samples (every symbol phase) and seeds 2701 to 2704 (32 runs per row; carrier offset and phase as above) and counts the symbols up to the last decision error at the alignment of symbols 6872 to 7871 (a run whose final alignment still has errors is "not settled"):

| constellation | Eb/N0 dB | receiver | median | 90 % | max | not settled |
| --- | --- | --- | --- | --- | --- | --- |
| BPSK | 12 | sdrk, sync_bw 0.005 | 0 | 147 | 258 | 0 |
| BPSK | 12 | sdrk, sync_bw 0.01 | 0 | 47 | 258 | 0 |
| BPSK | 12 | GNU Radio, sync_bw 0.01 | 0 | 862 | 2078 | 0 |
| QPSK | 12 | sdrk, sync_bw 0.005 | 17 | 190 | 353 | 0 |
| QPSK | 12 | sdrk, sync_bw 0.01 | 12 | 78 | 142 | 0 |
| QPSK | 12 | GNU Radio, sync_bw 0.01 | 27 | 1185 | 2400 | 0 |
| 8PSK | 18 | sdrk, sync_bw 0.005 | 60 | 213 | 306 | 0 |
| 8PSK | 18 | sdrk, sync_bw 0.01 | 19 | 115 | 237 | 0 |
| 8PSK | 18 | GNU Radio, sync_bw 0.01 | 281 | 2018 | 2032 | 0 |

- **Coherent BPSK, QPSK and 8PSK.** With the block defaults `psk_demod` is within 1 to 9 % of GNU Radio's chain on every row: BPSK 1.05 to 1.09 with 0 alignment changes, QPSK 1.02 to 1.03 (2 dB: 3.26 alignment changes per 1e5 symbols against 2.73), 8PSK 1.01 to 1.03. Two changes brought it there. The timing loop (D2553): the same nominal `sync_bw` is a wider loop in `psk_demod` than in `symbol_sync_cc`, so at 0.01 BPSK is 1.23 to 1.86 times GNU Radio's BER with 9.05 alignment changes at 2 dB; the default is 0.005. The carrier loop (D2554): GNU Radio's `costas_loop_cc` limits the phase error to +-1 before the loop filter, ours did not, so at 2 dB QPSK slipped more (4.56 changes, 1.11 of GNU Radio's BER, before the limit). The narrower timing loop keeps the settling well below that of the chain: at most 258 symbols for BPSK, 353 for QPSK and 306 for 8PSK (0.01: 258, 142 and 237), GNU Radio 2078, 2400 and 2032. With the loop before D2554, 0.0075 and 0.004 made 8PSK at 6 dB slip (2.49 and 1.78 times, D2553).

- **Differential BPSK.** `psk_demod` with `differential` is a non-coherent detector (no carrier loop, the product of two symbols) and its BER is the theory of the non-coherent detector plus 5 to 12 % (1.0e-3 at 8 dB against `exp(-Eb/N0)/2` = 9.1e-4); GNU Radio's chain is coherent detection (`costas_loop_cc`) followed by the differential decoder, which has about twice the BER of coherent BPSK (4.0e-4 at 8 dB), so it is 1.4 to 2.5 times better at 2 to 8 dB (1.8 to 2.9 with `sync_bw` 0.01). These are two receivers with different work (the differential mode of `psk_demod` needs no carrier lock and follows frequency offsets that the loop of the chain would have to pull in), reported as information.
- **Not compared** (the maintainer's decision D2555: the reason stands, no further comparison). Differential QPSK and differential 8PSK: the differential `psk_mod` steps the phase by the mapped phase (odd multiples of 45 degrees for QPSK and 8PSK, M40 and M40.1), so no point set of GNU Radio's `costas_loop_cc` (orders 4 and 8 lock to one point set) follows the signal, and GNU Radio's `diff_encoder_bb` steps by 0, 90, 180 and 270 degrees, another signal; `pi4` and `offset` likewise (no GNU Radio block). The receiver with the equalizer (`equalizer_taps`): the translations of GNU Radio's `linear_equalizer` in `benches/refs` run the decision-directed NLMS (QPSK points and the rectangular constellation of square QAM) without the CMA start of the QAM comparison, and a constellation of BPSK or 8PSK points is not among them.

## Running the measurements

```sh
cargo measure biquad_real                          # the platform of the host, every level: golden cases and cycles
cargo measure biquad_real --level avx2             # one level (repeatable)
cargo measure fir --platform x86_64-avx512         # another platform (repeatable)
cargo measure fft --no-cycles                      # only the golden cases (--no-golden: only the cycles)
cargo measure fft --no-golden --merge --platform x86_64 --platform x86_64-avx512 --platform aarch64
                                                   # the cycles and branches of the entry into benches/results/cycles
cargo measure --all --no-golden --platform x86_64  # every entry: benches/results/cycles/x86_64.json anew
cargo measure fir --branches                       # also the mispredictions and the cost of the data-dependent branches
python3 -I benches/cycle_tables.py                 # the cycle and simd128 tables above from the committed results
cargo xtask check cycles                           # the gates: D321 and the release gate in cycles
cargo xtask count --platform wasm32                # simd128 instructions; needs wasmtime and clang with the wasi-libc headers
cargo xtask count --platform cortex-m4             # Cortex-M4 report; needs qemu-system-arm with plugins, arm-none-eabi-gcc and the newlib headers
```

`cargo measure <kernel>` (alias of `cargo xtask measure`) runs every measurement of the [measurement policy](#measurement-policy) that exists for one kernel and prints it as tables; `target/measure/<kernel>.json` gets the same as JSON (`--json` prints it instead of the tables). The name selects every entry and golden directory that is the name or starts with it and `_` (`biquad_real` selects `biquad_real`, `biquad_real_q15`, `biquad_real_fallback`, `biquad_real_q15_fallback` and the golden cases of `biquad_real`). `--platform` selects `x86_64`, `x86_64-avx512` or `aarch64` (`cargo xtask count` takes `wasm32` and `cortex-m4` only, and rejects the others with a message that names `cargo measure`).

- **Golden cases:** the golden tests of `sdrk-testkit` (the host build, which detects the level at run time) with the report of `sdrk_testkit::golden` (`SDRK_GOLDEN_REPORT`, [tests/golden/README.md](../tests/golden/README.md)). Per case: the contract floor, the measured SNR (the lowest of all comparisons of the case), the achievable optimum (the expected values rounded once to the output format) and the distance to it ([kernels.md](kernels.md#accuracy-policy)).
- **Cycles** (`--no-cycles` skips them): for every level with a core model, the execution counts of ours and of the judged comparison from the counter of the level in its environment, simulated by `llvm-mca` on the reference model and the counter-model of the level ([how](#how-the-cycles-are-simulated)). Per entry, cycle mode (`whole-loop` and `per-block`, the one of `--cycle-mode` first; default `whole-loop`) and model: the cycles per sample of both sides, their ratio, our efficiency, the bottleneck class and the coverage. An entry that cannot be simulated gets a `note`; `simd128` is listed as not simulated. `--cycle-counter sde` takes the counts of `avx2` from SDE (`-hsw`) instead of callgrind (validation, D68). Only the ratios mean something; the cycles are no hardware figures.
- **Branches** (`--branches`, off by default): the mispredictions per sample of ours and of the comparison from cachegrind's branch simulation and the cost `cycles + mispredictions × penalty` on both models of `sse2`, `sse4.2` and `avx2` ([Branches](#branches-d1611)).
- **Results:** `--merge` updates the given entries of `benches/results/cycles/<platform>.json` only; `--all` measures every entry (`--results DIR` writes elsewhere, `--shard I/N` takes one of N partitions). `cargo xtask results merge|compare|repro` merge partitions, compare two result sets (1 % on either model) and check the reproduction row.

The command fails (exit code 1) when a golden case fails or a platform cannot be profiled.

Tools.


- **The C references** need `curl`, `sha256sum`, `tar`, `make`, `cmake`, `ninja` and `python3` with `mako` (VOLK), and for `aarch64`, `wasm32` and `cortex-m4` the cross compilers (`gcc-aarch64-linux-gnu`, `g++-aarch64-linux-gnu`; `clang` and `llvm-ar` with the wasi-libc headers; `gcc-arm-none-eabi` with `libnewlib-arm-none-eabi`, whose headers CMSIS-DSP includes). `benches/refs/build.sh` returns at once when the prefix already holds a build of the same script with the same compilers. Environment: `SDRK_REFS_DOWNLOADS` (download cache), `SDRK_REFS_JOBS` (default 2).
- **Intel SDE** is pinned in CI to version 9.58.0 (`https://downloadmirror.intel.com/859732/sde-external-9.58.0-2025-06-16-lin.tar.xz`, sha256 `f849acecad4c9b108259c643b2688fd65c35723cd23368abe5dd64b917cc18c0`) and cached; it is proprietary freeware of Intel and is not part of the repository.
- **QEMU** must be configured with `--enable-plugins`; the packaged `qemu-aarch64` of the distributions is not. CI builds QEMU 10.2.1 (`configure --target-list=aarch64-linux-user --enable-plugins`, then `ninja qemu-aarch64`) in the nightly job. Point `SDRK_QEMU_AARCH64` at the binary. xtask builds the `insn` plugin into `target/tools/libinsn.so` if it is missing, from the QEMU tree named by `SDRK_QEMU_SRC` or from the two files `tests/tcg/plugins/insn.c` and `include/qemu/qemu-plugin.h` that it downloads for the tag `SDRK_QEMU_TAG` (default `v10.2.1`; the location of the plugin source has moved between QEMU releases). Needs `gcc`, `pkg-config` and `libglib2.0-dev`. For the simulated cycles of `neon` it builds the block-count plugin of the repository (`xtask/plugins/bbcount.c`) into `target/tools/libbbcount.so` the same way, against `include/qemu/qemu-plugin.h` of that tree (downloaded once if missing), and rebuilds it when the source changes (its SHA-256 is kept in `target/tools/libbbcount.so.sha256`). `SDRK_AARCH64_SYSROOT` (default `/usr/aarch64-linux-gnu`, from `libc6-dev-arm64-cross`) is the sysroot of the target libc.
- **wasmtime** with fuel (`-W fuel=N`).
- **Cycle simulation** (level 2 of the [measurement policy](#measurement-policy); installed in the development container, `.devcontainer/Dockerfile`; the counters are the ones above, and `aarch64-linux-gnu-objdump` from `binutils-aarch64-linux-gnu` disassembles the AArch64 objects, `SDRK_AARCH64_OBJDUMP`):
  - `llvm-mca` of LLVM 23.1.1, built from the release source (`llvm-project-23.1.1.src.tar.xz`, sha256 `ebe9be46fe8756d58c5b198ffad0fa2a766257add81a4dc52179bfacc7888ee6`, the digest of the release attestation) with the targets X86 and AArch64 and SDRk's corrections of the scheduling models (`assets/llvm-mca-23.1.1-sdrk-sched.patch`, D942), in `/opt/llvm-mca-23.1.1/bin` (`SDRK_LLVM_MCA`); the same build without the patch is kept in `/opt/llvm-mca-23.1.1-upstream/bin` (`SDRK_LLVM_MCA_UPSTREAM`) for comparisons. The patch replaces numbers that a one-time comparison of every instruction form in the simulated loops found wrong: on x86 against the measurements of uops.info (stores on `znver2`, `znver3` and `znver4`, `lea` on `znver2`, `alderlake` and `sapphirerapids`, sign and zero extension and element insertion from memory on `znver3` and `znver4`, the second 512-bit unit of `sapphirerapids`, lane moves and 512-bit loads on `znver4`, divide and square root on `alderlake` and `sapphirerapids`), on AArch64 against Arm's software optimization guides (the FP latencies of `cortex-a72`, which LLVM simulates with the `cortex-a57` model, and divide and square root on both cores). Every changed definition names its source; no model was added. The `llvm-mca` of the distribution's `llvm` package is another LLVM version and must not be mixed into the ratios. In LLVM 23.1.1 `cortex-a72` uses the `cortex-a57` model and `cortex-a76` the `neoverse-n1` model; in LLVM 21 both used the `cortex-a57` model (see the core-model table).
  - Second opinions, used only where a decision depends on a single result: OSACA 0.7.1 (PyPI; models for Zen 1 to 4, Intel from Sandy Bridge to Sapphire Rapids, Neoverse N1 and V2, Cortex-A72; `SDRK_OSACA`) and uiCA at commit `6ebeeab` (Intel cores; `SDRK_UICA`), both AGPL-3.0 development tools that only read assembly and are never linked into SDRk. uiCA's setup takes the current `instructions.xml` of uops.info, which has no version; its SHA-256 is kept in `/opt/uiCA/instructions.xml.sha256`.
- For `aarch64-unknown-linux-gnu` cross builds xtask sets `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc`.

## CI

The measurement jobs run in a pinned container image (`ubuntu:26.04` by digest) with the packages of a pinned apt snapshot, so that the compilers of the C references do not change under the measurements; the references are cached by the build script, the image and the snapshot. The Cortex-M4 job `count_m4_qemu` uses the same image (the snapshot has `gcc-arm-none-eabi` 15:14.2.rel1-1, the compiler of the committed CMSIS-DSP numbers).

- **Cycles** (`.github/workflows/measure.yml`, D2709): `nightly.yml` calls it as `measure_nightly` for every entry and level with a core model; a branch measures a selection alone with `gh workflow run nightly.yml --ref <branch> -f measure_only=true -f kernels="<names>" -f levels="<levels>"` (`-f repeat=true` measures every partition twice and fails unless the two results files are identical; `-f shards=N` splits a full run). The job `llvm_mca` builds `llvm-mca` 23.1.1 with `assets/llvm-mca-23.1.1-sdrk-sched.patch` (D942) and caches it by the patch, the LLVM version, the image and the snapshot; `reproduction` checks `costas` at `neon` (86.13 / 54.10, D2761) first; `measure` runs `cargo measure --merge --results` per level, the whole set split into partitions, with SDE, Valgrind and QEMU with plugin support, and with `setarch -R` (the container runs with `seccomp=unconfined`, I13.1-1); `results` merges the partitions (`cargo xtask results merge`), fails on a record that differs from the committed `benches/results/cycles` by more than 1 % in ours or the comparison on either model, or that is new or lost (`cargo xtask results compare`), runs `cargo xtask check cycles --results` on the fresh results, and keeps them as the artifact `cycle-results`. The CI container is the reference environment of the committed results: they are this artifact. Run measurements in CI, not on the development machine.
- **Instruction counts** (`count_wasm32`, `count_m4_qemu`): the nightly and manual runs only (D175 as restated by M13.1: the nightly speed net for the out-of-order levels is the cycle job above). They fail only when `cargo xtask count` fails (a limit of `simd128` or a measurement error); the difference of the fresh results against the committed files is shown for information (`git diff`) and the fresh files are kept as an artifact. The counts drift with the `stable` toolchain of the day (for example 0.01 instructions per sample in wasm32).
- A pull request that touches kernels, backends, `benches`, `xtask` or the toolchain checks that the measurement harness builds (`cargo check --manifest-path benches/Cargo.toml --workspace --all-targets` in `check`, without the C references; input `run_bench` of `tree.yml`); `cargo xtask check` judges the committed cycle results (`check cycles`) and `benches/test_cycle_tables.py` checks the generated tables.
- `xtask` removes every `QEMU_*` variable from the environment of `qemu-aarch64`, because qemu-user reads them as options (`QEMU_VERSION` is `-version`). wasmtime is installed from its GitHub release tarball, pinned by version and SHA-256 (`WASMTIME_SHA256`).


## Cortex-M4 (fallback, report only)

`cargo xtask count --platform cortex-m4` counts twenty-two kernels of the portable fallback on a Cortex-M4F in `qemu-system-arm` (the `insn` plugin) and writes `benches/results/cortex-m4.json`; the nightly job `count_m4_qemu` runs it. The method is the differential one above, with the workload binary `count` of `targets/qemu-cortex-m4`. The report has no limits and no verdicts. Details in [embedded.md](embedded.md#instruction-counts-cortex-m4).

Since M22 (open item M22-10, D27) the four biquad kernels are in the report next to CMSIS-DSP 1.18.0 built for the M4F with the DSP extension and the FPU (the tarball and SHA-256 of the host build, plus the CMSIS-Core headers of CMSIS 6.3.0; `benches/refs/build.sh cortex-m4`). Both sides filter the input of the host counts (the generator, seeds, filter and 4096 samples per call of `benches/kernels`) and are compared on the same input with the f64 model of `benches/kernels`; our SNRs are those of the host counts to the last digit, which confirms the input. Our side is called through a function that is never inlined, with the coefficients behind `black_box`, as on the host (CMSIS-DSP sees them at run time too). Both sides are compiled for the same core (D30): the workload with `-C target-cpu=cortex-m4` (`targets/qemu-cortex-m4/.cargo/config.toml`, recorded as `build.target_cpu` in the report), CMSIS-DSP with `-mcpu=cortex-m4`. Without the flag LLVM tunes for a generic Armv7E-M and turns every non-fused multiply-add of the `f32` kernels into one `vmla.f32`/`vmls.f32` (3 cycles on the M4 against 1 + 1 for `vmul.f32` and `vadd.f32`, which GCC's Cortex-M4 tuning selects), so the instruction counts compared instruction selections, not work: until 2026-10-28 the report had 36.77 (0.82), 73.29 (1.11), 13.88 (0.62) and 33.53 (1.02) in the order of the table.

| Kernel (2 sections) | Ours | CMSIS-DSP | Ratio | SNR ours / CMSIS-DSP (dB) |
| --- | --- | --- | --- | --- |
| `biquad` C32 (CMSIS-DSP `arm_biquad_cascade_stereo_df2T_f32`, re and im as the two channels) | 50.27 | 45.00 | 1.12 | 136.06 / 134.90 |
| `biquad` CQ15 (`arm_biquad_cascade_df1_q15`, postShift 1, two calls on planar re and im, deinterleaving not counted) | 82.05 | 66.05 | 1.24 | 71.59 / 56.73 |
| `biquad_real` f32 (`arm_biquad_cascade_df2T_f32`) | 21.88 | 22.50 | 0.97 | 136.72 / 136.72 |
| `biquad_real` Q15 (`arm_biquad_cascade_df1_q15`, postShift 1) | 37.78 | 33.03 | 1.14 | 71.45 / 56.63 |

The report has no limits, so these ratios get no verdict; measured against the goal of 1.00 that the host entries have, `biquad` C32 (1.12), `biquad` CQ15 (1.24) and `biquad_real` Q15 (1.14) miss it and `biquad_real` f32 (0.97) meets it. The gap is expected and reported as information (D31): the Cortex-M4 has no SIMD or DSP-extension backend yet (M18), so the portable scalar fallback is compared with CMSIS-DSP code that uses the DSP extension. Whether the comparison is like with like: it is the same algorithm (transposed direct form II in `f32`, direct form I in Q15 with a 64-bit accumulator), the same core and the same instruction set (`thumbv7em-none-eabihf` has the DSP extension and the FPU, so both compilers may use `smlal`, `ssat` and the FPU), and CMSIS-DSP is the established library for it; but CMSIS-DSP's Q15 cascade is hand-written for the DSP extension (`__SMLALD`, two 16 x 16 bit products per instruction on packed sample pairs), an optimization that our portable fallback by design does not contain (no target-specific code, one `smlal` per product). The like-with-like counterpart of that path is a biquad of the `arm-dsp` backend (milestone [M18](milestones/M18-arm-dsp-backend.md), not implemented yet; its kernel list has no biquad), not the fallback; the Q15 rows therefore compare the portable fallback with the best established implementation for the core and are information about how far the fallback is from it. CMSIS-DSP truncates where ours rounds (14.8 to 14.9 dB more SNR for ours). On 32-bit targets other than wasm32 our Q15 cascade keeps the state in `i32` and runs the two components of `CQ15` in two passes (D28); with the `i64` state of the 64-bit targets it counts 322.06 (`CQ15`) and 103.28 (`Q15`) instructions per sample here (326.06 and 103.78 with the generic tuning; the 64-bit products of `i64` operands are full 64 x 64 bit multiplications on the M4).

Since M48 the report has `gauss_noise` (`sigma` 0.1, stream 1, 1024 samples per call, the table `GAUSS_ICDF` where the linker puts it): 40.34 instructions per sample for `f32`, 72.08 for `Q15` and 80.59 for `C32` (two values per sample), counted on 2026-11-18 (only these three rows were merged into `benches/results/cortex-m4.json`). `Q15` costs 31.74 instructions more per value than `f32`: the exact R7 rounding of `zq · ms` (a 32 x 32 to 64-bit product, the rounding constant and a 64-bit shift by the variable `sh`, then the saturation), where `f32` takes one `VCVT` and one `VMUL`. This is the number for the re-evaluation of the cheaper `SMMULR` + `SSAT` form (M48, maintainer decision 3). Instruction counts, not cycles: QEMU reads the flash without wait states.

Note that the M4F has a single-precision FPU only: `convert_out_cs16` rounds in `f64` (R7) and so counts about a thousand instructions per sample in software floating point.

## Open gaps

The open items of the measurement are in [open-items.md](open-items.md); the ones that decide how far the numbers above can be trusted or are still above a gate:

- **I13.1-1:** the profiles of `cargo measure` are not bit-reproducible from run to run; with the address space fixed in CI (D2710) the spread is below 1.7e-05 relative, not zero (D2747).
- **I13.1-4, I13.1-5:** `agc` and `biquad_real_q15` at `avx2-runtime` are above 1.02 of the `avx2` build from the simulation of the same loops (decision of the maintainer: info status, or a change of the measuring tool).
- **I13.1-6 to I13.1-8, I24-3:** the entries over their cap in cycles after the re-judgement of D2746, `dev` with their numbers: 28 at `avx2-runtime` (I13.1-6), 33 fallback entries and levels against their library reference (I13.1-7), 11 SIMD entries and levels (I13.1-8) and `stereo_matrix_fallback` (I24-3).
- **I44-15, I44-16:** the `cortex-a76` misprediction penalty is provisional; `avx512` cannot be branch-simulated.
- **Issue #20** (fair references, groups A to D), **#134** (`Q15` fallbacks above CMSIS-DSP's cycles), **#169** (new references).
- **Planned:** the `avx512` build and KFR at `x86-64-v4` instead of `sapphirerapids`; SNR regression guards ([kernels.md](kernels.md#accuracy-policy)); the instruction counts of ESP32-P4 and RISC-V when M16 and M17 come.
