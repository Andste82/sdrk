# Performance

Performance is compared with an established library (VOLK, FFTW, rustfft, KissFFT, KFR, CMSIS-DSP; ESP-DSP for the embedded backends to come) at equal accuracy against the f64 reference. The speed target is the simulated cycles per sample on every level with a core model (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, and the scalar fallbacks of each, D68); executed instructions per sample, counted in an emulator, are the gate that catches gross outliers, and the only measure where no core model exists (`simd128`, the in-order cores). The [measurement policy](#measurement-policy) below says what is compared with what. Until the maintainer decides the cycle gates, the gate of `benches/limits.json` and CI is the instruction ratio with the limits below.

| Instructions per sample vs. reference library | Meaning |
| --- | --- |
| ≤ 1.00× | goal: equal or better (recorded as `goal` in the results) |
| ≤ 1.10× | release limit; CI fails above it for a `release` entry |
| ≤ 1.50× | allowed only while a kernel is under development (`dev`), never in a release |

Instruction counts are a strong proxy on in-order cores (Cortex-M, ESP32) and a weaker one on out-of-order CPUs and for memory-bound kernels, which is why the simulated cycles are the speed target there (D68). They say nothing about cache behaviour, the cost of one instruction (a gather is one instruction and many cycles) or the clock; they are the portable, reproducible measure of how much work a kernel does. Kernels that use gathers (`fm_mod` at `avx512`, and the NCO lookups of `rotate` `CQ15` there) are therefore not shown to be faster by their counts. There is no real hardware for a wall-clock check (D33), and the simulation of the next section has low confidence for gathers. Issue #20 proposes to try a gather-free form first: in-register tables with permutes (`vpermps`, `vpermi2ps`).

## Measurement policy

This section is the maintainer's policy (D32 to D34; worked out in issue #20). It applies to every new or changed kernel, entry of `benches/limits.json` and row of the results. Where the current entries do not follow it yet, issue #20 lists them; the [status](#status-of-the-policy) at the end of this section says what exists today.

Each rule below is marked as one of three kinds: **decided** (applies now), **planned** (decided, but the tooling does not exist yet) or **proposed** (a threshold that still needs the maintainer's confirmation).

**Gates today:** instructions per sample stay the gate on every platform, with the limits of the table above; on the levels with a core model their role is to catch gross outliers, and the speed target is the simulated cycles (D68). The cycles are reported by `cargo measure` on every such level; one rule on them is gated (D321): no SIMD entry may be slower than its own fallback in simulated cycles per sample on the reference model or on the counter-model of its level (`cargo xtask check cycles`, part of `cargo xtask check` in CI, on the committed [cycle tables](#results)). Gates on the cycle ratios against the libraries are a separate decision of the maintainer. A SIMD branch without a fair reference is `info` and its `waiver` text names the reason. `count accuracy` stays the accuracy check of the counts.

### Three questions, three measures

| Question | Measure | Source | Kind |
| --- | --- | --- | --- |
| How much work does the kernel do? | instructions per sample | SDE, callgrind, QEMU plug-in, wasmtime fuel (this file, [Method](#method)) | measured, exact |
| How fast is it in the core? | cycles per sample, and the efficiency (below) | `llvm-mca` on the execution counts of callgrind, SDE or QEMU ([level 2](#speed-simulated-cycles-per-sample-level-2)) | simulated, relative only |
| How accurate is it? | SNR against the f64 golden | golden vectors and `count accuracy` ([kernels.md](kernels.md#accuracy-policy)) | measured, exact |

Speed and accuracy are judged independently. A kernel that is faster but less accurate than allowed fails, and so does one that is accurate but too slow.

### What is compared with what (D32)

Decided. The goal is an optimal branch for every kernel at every SIMD level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`), and an efficient portable fallback. Whether a branch is optimal is only known by comparing it with a well-known, well-tested library that does the same thing.

| Our side | Fair comparison | Examples |
| --- | --- | --- |
| SIMD branch of a level | SIMD code of an established library at the **same level**, the **same algorithm class** and the **same work** | VOLK, KFR, FFTW, IPP, rustfft |
| Scalar fallback | established **scalar** code of the same algorithm | CMSIS-DSP generic C, the loops of GNU Radio |
| A branch for which no such library exists | the efficiency of the branch ([below](#efficiency-a-measure-without-a-reference)), recorded per kernel | the `Q15` and `CQ15` biquads, which no library has in SIMD on our levels |
| SIMD branch against our own fallback | never the judged comparison in instructions (at most `info`); but in simulated cycles per sample the branch must be at least as fast as the fallback of its level on both models (D321, gated by `cargo xtask check cycles`; a violation needs a `cycle_waiver` naming an open item) | a SIMD branch that is not faster than the scalar loop (possibly auto-vectorised by the compiler) has no reason to exist; beating it says nothing yet about whether the branch is good, which is what the library comparison is for |

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

5. **The whole kernel, not a hand-picked loop.** The counters of the levels give the execution count of every basic block (callgrind at `avx2`, Intel SDE's DCFG at `sse2`, `sse4.2` and `avx512`, a QEMU plugin at `neon`). The hot blocks are found from these counts, and their simulated cycles are weighted by them. This covers FFT stages, prologue, tail and dispatch, and it works for reference libraries without markers (VOLK, KFR, FFTW codelets).
6. **Known blind spots**, recorded with every result: cache and memory bandwidth (every count uses 4096 samples, which fit in L1); memory dependencies across iterations (in-place FFT, filter state in memory, store-to-load forwarding); branch mispredictions in prologue and tail; gathers, divisions, square roots and microcode (low confidence: no value is published for them).

**Where instructions stay the gate** (decided; D68). On in-order cores (Cortex-M, ESP32) instructions are close to cycles, and `simd128` has no core model (the engine's JIT generates the machine code; cycles of it would be informational at most and are not simulated). There the instruction ratio stays the gate and the target. On x86-64 and AArch64 it is the gate for gross outliers next to the simulated cycles, which are the speed target.

**Proposed gates** (issue #20, not yet in `benches/limits.json`): on the reference model ours / reference ≤ 1.00 is the goal and the release limit of the table above applies; on the counter-model the ratio may be at most about 5 % worse than on the reference model (2 % would be inside the modelling error and cause churn).

**Optional plausibility check** (proposed): `perf stat` on a Raspberry Pi 4 (Cortex-A72) for the `neon` branch. It checks the simulated ratios for plausibility and does not calibrate them.

**Derived figure, for orientation only.** `MS/s per GHz = 1000 / cycles per sample` may be computed to estimate how much signal one core can process: it is the simulated in-core throughput with the data in L1, an upper bound. For large blocks of streaming kernels the memory caps it: `samples/s = min(f_CPU / cycles per sample, bandwidth / bytes per sample)`. It is neither published as a result nor gated.

### Working rule for optimizing (D37, D68)

Decided. Since D68 `cargo measure` simulates the cycles of every level with a core model, so a kernel is optimized like this:

| Code | Steered by | Checked by |
| --- | --- | --- |
| Code shared by all levels: the portable fallback, generic `fearless_simd` code | the simulated cycles of every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`; each on its reference model and counter-model) | the instruction counts of every level, as the gate for gross outliers |
| Code of one level: a hand-written SIMD path for `sse2`, `neon`, `avx512`, ... | the simulated cycles of that level | the instruction counts of that level; the cycles of another level say nothing about it |
| `simd128` and in-order cores (Cortex-M, ESP32) | the instruction counts | nothing else: `simd128` has no core model (its cycles would be informational at most), and on in-order cores instructions are close to cycles |

Before D68 (D37) the shared code was steered by the cycles at `avx2` alone, because only callgrind gave execution counts per instruction. Why that was not enough: for the `Q15` biquad fallback (D36) a shorter dependency chain at `avx2` was also faster at `sse2`, `sse4.2`, `avx512` and `neon` in 13 of 16 comparisons (level and core model), but not in all of them (`neon` `cortex-a72` unchanged for `CQ15`; `avx512` `znver4` uncertain, `llvm-mca` and OSACA disagree). The gain at `avx2` is a good guide for shared code, not a proof for the other levels.

### How the cycles are simulated

Implemented in `xtask/src/mca.rs` (D35) and `xtask/src/blockcount.rs` (D68) and run by `cargo measure`:

1. The counter of the level records how often every instruction and every taken jump of the measured process ran, in a run with the counted calls and in a run without them; the difference belongs to the counted calls, as in the differential method above (for an FFT, the restore of the input is subtracted as well; executions that a subtraction leaves negative, the harness code that only the restore run executes, are deducted from the total, so the total is the count). The runs have the environment of `cargo xtask count` for the level: the same measurement binary, the same `volk_config` pin, the same references and library path, the same emulated CPU. `objdump` (`aarch64-linux-gnu-objdump` for AArch64 objects) disassembles the executed code from the object files of the measured process (our binary, the static references, VOLK's shared library, libc).

   | Level | Counter of the cycles | Execution counts |
   | --- | --- | --- |
   | `avx2` | valgrind callgrind (`--dump-instr=yes --collect-jumps=yes`) | per instruction and per taken jump |
   | `sse2`, `sse4.2` | Intel SDE `-dcfg` with the CPUID definition of the count (`-cpuid_in`, `-chip_check_cpuid 1`) | per basic block and per edge, between the SSC markers of the count (`-start_ssc_mark`, `-stop_ssc_mark`) |
   | `avx512` | Intel SDE `-dcfg -spr` | as `sse2` |
   | `neon` | `qemu-aarch64` (`-R 0x200000000`, the sysroot of the count) with the plugin `xtask/plugins/bbcount.c` | per translation block and per control transfer between blocks |

   SDE's DCFG gives every basic block of every image as an offset in the image and every edge with its type; the taken branches (the edge types `*BRANCH`, not fall-through, call, return or `REP`) are the jumps. The blocks of the two runs are subtracted per block and expanded into instructions with the disassembly; a block whose disassembly does not have the number of instructions SDE counted is an error, never a silent zero. The QEMU plugin writes the execution count of every translation block, every transfer from the last instruction of a block to a block that does not follow it, and the file mappings of the QEMU process: user-mode QEMU maps the guest's files at `guest_base + address`, so the mappings give the object and the file offset of every block, in the binary and in the shared libraries (VOLK). The control-flow graph leaves out a transfer from a call or a return (callgrind records none; the plugin records every transfer). xtask builds the plugin into `target/tools/libbbcount.so` against the `qemu-plugin.h` of the pinned QEMU (see [Running the measurements](#running-the-measurements)) and rebuilds it when its source changes. `cargo measure --level avx2 --cycle-counter sde` takes the counts of `avx2` from SDE on its Haswell CPU (`-hsw`) instead of callgrind, to validate the SDE path (D68).
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

The f64 golden is the mathematical optimum and the first target; an established library is only a yardstick where that optimum cannot be reached. The rules are in [kernels.md](kernels.md#accuracy-policy). For the counts this means: `count accuracy` measures both sides against the same f64 model, and "at most 1 dB below the comparison" is a floor, never the target. A comparison that is less accurate than we are does not lower our goal.

### Status of the policy

| Part | State |
| --- | --- |
| Instructions per sample, `count accuracy` | in place (this file) |
| Fair references for every entry | partly: the entries gated against our own fallback, the kernels without a SIMD branch or without a count are listed in issue #20 (groups A to D) |
| Simulated cycles and efficiency | reported by `cargo measure` at `sse2`, `sse4.2` and `avx512` (SDE), `avx2` (callgrind) and `neon` (QEMU), D68; the speed target and not written to `benches/results`. Gated since D321: SIMD entry ≤ its `_fallback` entry on both models of every level, read from the committed cycle tables of this file by `cargo xtask check cycles` (CI runs no `llvm-mca`); `cargo measure` prints the same verdict for every pair it simulates. Gates on the ratios against the libraries are a separate decision |
| `avx512` measurement build and KFR reference at `x86-64-v4` instead of `sapphirerapids` | planned |
| Accuracy against the achievable optimum | measured and shown per golden case by `cargo measure` (not judged) |
| SNR regression guards, Q15 SNR against an f64 golden | planned ([kernels.md](kernels.md#accuracy-policy)) |
| One command per kernel (`cargo measure`) | instructions, count accuracy, golden cases and simulated cycles |

## Method

`cargo xtask count --platform <platform>` builds the measurement binary of `benches/kernels` for every level of the platform and runs it under an instruction counter.

- **Differential count.** `count run <kernel> <impl> <k>` does one warm-up call and then `k` calls on the same buffers. It is started twice, with `k = 0` and with `k = K`, and the instructions per sample are `(count(K) - count(0)) / (K * samples)`. Process start, allocation, set-up (the plans of rustfft, FFTW and KissFFT, the VOLK dispatcher) and the warm-up call cancel. `K = 16` calls of 4096 samples; a kernel with fewer samples per call (the FFT sizes 64 and 1024) gets proportionally more calls, so every measurement covers 65 536 samples.
- **Inputs.** PCG32 noise with the seed `1000 + index of the kernel`, 4096 samples, buffers aligned to 64 bytes for both sides. The FFTs are in place: every call first restores the input, and the count of the restore alone is subtracted from both sides, so the transforms always see the same finite data.
- **Variants (information).** The streaming kernels are also measured with buffers misaligned by one sample, and the conversions with blocks of 64 and 100 samples. These lines are reported in the results and never judged.
- **Counters.**

| Level | Build | Counter |
| --- | --- | --- |
| `sse2` | default x86-64 build | Intel SDE with the CPUID of a strict SSE2 x86-64 CPU (`-cpuid_in`, `-chip_check_cpuid 1`): VOLK, rustfft and FFTW choose the code of that level themselves, and any instruction above it stops the run |
| `sse4.2` | `-C target-cpu=x86-64-v2` | Intel SDE with the CPUID of an x86-64-v2 CPU (up to SSE4.2 and POPCNT), as for `sse2` |
| `avx2` | `-C target-cpu=x86-64-v3` | Valgrind `callgrind` (Linux; not available on Windows, so the count job runs on Linux only) |
| `avx2-runtime` | default build with the feature `std` (run-time detection) | Valgrind `callgrind`; judged against our own numbers at `avx2` (at most 1.02×, the cost of the dispatch) |
| `avx512` | `-C target-cpu=sapphirerapids` | Intel SDE `-spr` |
| `neon` | `aarch64-unknown-linux-gnu` | `qemu-aarch64` in user mode with the `insn` plugin and a reserved guest address space (`-R 0x200000000`), which fixes the memory layout: identical runs give identical counts |
| `simd128` | `wasm32-wasip1`, `+simd128` | wasmtime with fuel: the smallest fuel with which the run completes, found by bisection |

  Under SDE only the region between two SSC markers of the binary (`benches/kernels`, `region_start`/`region_stop`) is counted, which makes the counts exact; each count is still the median of up to three runs (the third run is made only when the first two differ). The CPUID definitions keep the XSAVE state to x87 and SSE: on a CPU without OSXSAVE the `cpu_features` library inside VOLK would read the SSE flags of the host from `/proc/cpuinfo` and select a machine that the emulated CPU does not have. SDE's own `-p4` does not run 64-bit code, so it cannot be used for `sse2`.
- **Reproducibility.** The measurement binary is built with `--remap-path-prefix` for the checkout (`/sdrk`) and the Cargo home (`/cargo`), passed through `CARGO_ENCODED_RUSTFLAGS`. Otherwise these paths end up in the panic locations of the binary, and their length moves its data and with it some counts. The two runs of a measurement get arguments of the same length: `k` is written with as many digits as the number of calls (`0000` and `1024`). The start-up code of a process depends on the length of its arguments and environment; with the same length, the start-ups are identical and cancel exactly. With both measures the counts no longer depend on the path of the checkout or on the environment of the caller: a checkout at another path gives identical lines on every platform.
- **Run time.** The levels of a platform are counted in parallel: first the levels judged against the references, then the run-time detection level, which is judged against our numbers at `avx2`. The C references are built once before. `x86_64` with the variants takes about 14 minutes instead of 20 on four cores.
- **Accuracy.** `count accuracy <kernel> <impl>` prints the SNR of one run against the f64 reference. An entry fails if its SNR is more than 1 dB below the SNR of its comparison, unless it carries an `accuracy_waiver`. The instruction ratio and the accuracy are judged independently.

### References

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
| GNU Radio `costas_loop_cc` | `work` of `costas_loop_cc_impl.cc` (gr-digital, main at `ec1bf15`) with `phase_detector_2` and `phase_detector_4` and the members of `gr::blocks::control_loop` (`advance_loop`, `phase_wrap`, `frequency_limit`, `update_gains`): per sample `sincosf` of the phase, `fast_cc_multiply`, the detector, `branchless_clip` to `+-1` and the second-order loop; one output, no tags, `use_snr` false, in `benches/refs/gr-costas.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrcostas.a`, the flags of `agc_cc`) | the reference of `costas_fallback` (`cref_sys::GrCostasLoopCc`, D607) |
| GNU Radio `linear_equalizer` | `linear_equalizer_impl::equalize` of gr-digital (main at `ec1bf15`) with `sps` 1 and no training sequence (decision-directed from the start), `adaptive_algorithm::error_dd` with the decision of `constellation_qpsk` (points `+-sqrt 2 +-j sqrt 2`), `adaptive_algorithm_nlms::update_taps` (`magsq` from the conjugate dot product of the window, then `taps[i] = conj(conj(taps[i]) + (mu / magsq) in[i] conj(error))`) and the members of `fir_filter_ccc` (gr-filter: reversed taps, `filter` as the dot product of the window, `update_tap` into one aligned copy), the VOLK calls as VOLK's `generic` protokernels, in `benches/refs/gr-equalizer.c` | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, `libgrequalizer.a`, the flags of `agc_cc`) | the reference of `equalize_fallback` (`cref_sys::GrLinearEqualizer`, D838) |
| liquid-dsp `iirfilt` DC blocker | 1.8.3 (`https://github.com/jgaeddert/liquid-dsp/archive/refs/tags/v1.8.3.tar.gz`, sha256 `18fa83b73db8bb6fe6ea0376e4b5aecf8645970f4604d10d9dadbf609f3f95e2`): `iirfilt_{rrrf,crcf}_create_dc_blocker` with `_execute_block`, `_execute_norm` and liquid's portable dot products (the run-time choice fixed to `LIQUID_RUNTIME_PORT`), in `benches/refs/liquid-dcblock.c` and `liquid-dotprod.c` | MIT | x86_64, aarch64, wasm32 (static, `libliquiddc.a`, liquid's `-O3`) | the scalar reference of `dc_block` and `dc_block_real` (`cref_sys::LiquidDcBlocker`, `LiquidDcBlockerCrcf`): the same pole and zero (`alpha = 1 - r`) in direct form II with the gain `sqrt(1 - alpha)` (`set_scale(1.0)` gives SDRk's gain), not the direct form I of SDRk; no established scalar code computes `y = x - x1 + r y1` literally in single precision (STK's `PoleZero::setBlockZero` does, in double) |
| liquid-dsp halfband decimators | 1.8.3 (the same tarball and sha256 as the DC blocker above), unpacked and compiled from the release: `resamp2.proto.c` and `firhilb.proto.c` through `benches/refs/liquid-halfband.c` (the type macros of `filter_crcf.c` and `filter_rrrf.c`), `dotprod_crcf.c` and `dotprod_rrrf.c` with their SSE, AVX, AVX-512F and NEON loops, the window buffers and the support files (`msb_index.c`, `memory.c`, `error.c`, `logging.c`, `runtime.c`, `libliquid.c`) | MIT | x86_64, aarch64, wasm32 (static, `libliquidhb-<level>.a`, one library per level, `-O3 -ffp-contract=off` and the instruction set of the level) | the reference of `halfband_decimate` (`resamp2_crcf_decim_execute`), `halfband_decimate_real` (`resamp2_rrrf_decim_execute`) and `real_to_complex` (`firhilbf_decim_execute_block`) with liquid's SIMD dot product of the level (SSE at `sse2` and `sse4.2` for `dotprod_crcf`, at `sse4.2` for `dotprod_rrrf` and the portable loop at `sse2` for it, AVX at `avx2`, AVX-512F at `avx512`, NEON at `neon`; liquid has no WebAssembly loop, so the portable loop compiled with `-msimd128` is the reference at `simd128`), and, with the portable dot product (`LIQUID_RUNTIME_PORT`, compiled for the same level), the scalar reference of the three fallback entries (D69) |
| liquid-dsp `firpfbch2_crcf` | 1.8.3 (the same tarball and sha256 as the DC blocker above), unpacked and compiled from the release: `firpfbch2.proto.c` through `benches/refs/liquid-channelizer.c` (the type macros of `firpfbch_crcf.c`, liquid's names renamed per build so that both builds link), with the window buffers and dot products of the halfband library and liquid's FFT (`fftf.c`, `fft_utilities.c`, `dotprod_cccf.c`, `modular_arithmetic.c`); built twice into the halfband library of each level: with FFTW 3.3.10 (`fftw3f_FOUND`, not on wasm32) and with liquid's own FFT | MIT | x86_64, aarch64, wasm32 (static, `libliquidhb-<level>.a`) | the reference of `channelize` (the SIMD dot product of the level and FFTW; liquid's own FFT on wasm32) and of `channelize_fallback` (the portable dot product and liquid's own FFT), D919 |
| liquid-dsp `nco_crcf` | 1.8.3 (the same tarball and sha256 as the DC blocker above): `nco_crcf` of the type `LIQUID_VCO_INTERP` from `src/nco/src/nco.proto.c` with `T = float`, translated to C in `benches/refs/liquid-nco.c` (`nco_crcf_create` with its table of 1024 value and slope pairs, `_fp_sin`, `_sincos`, `_static_index`, `_adjust_frequency`, `_adjust_phase`, `_constrain` with its `while` loops, `_step`; the loop gains of SDRk are applied through `_adjust_frequency` and `_adjust_phase`, the two calls of liquid's `_pll_step`, and `sin(2 theta)` is read from the same table with the phase word shifted left by one bit) | MIT | x86_64, aarch64, wasm32 (static, `libliquidnco.a`, one object file, liquid's `-O3` and the generic flags of `agc_cc`) | the scalar reference of the `f32` `stereo_pll` fallback (`cref_sys::LiquidNcoPll`; a lower accuracy class: the table interpolates with an error of 4.7e-6 against 2.9e-7 of the 4096-entry table of `rotate`, `_constrain` reduces a correction modulo two pi in `float` so that a small negative correction becomes `2 pi - c` rounded to a multiple of 4.8e-7 rad, and the loop has no lock detector and no integrator limit) |
| libairspy | 1.0.10 (`https://github.com/airspy/airspyone_host/archive/refs/tags/v1.0.10.tar.gz`, sha256 `fcca23911c9a9da71cebeffeba708c59d1d6401eec6eb2dd73cae35b8ea3c613`): `iqconverter_float.c` and `iqconverter_int16.c` through `benches/refs/airspy-iqconv.c`. The last release under the MIT licence; later releases are under a restrictive licence and are not used | MIT | x86_64, aarch64, wasm32 (static, `libairspy-<level>.a`, `-O3 -ffp-contract=off` and the instruction set of the level; the SSE2 build of the float converter on x86_64) | `real_to_complex`: the SSE2 reference of the `C32` kernel at `sse2` (information; libairspy enables its SSE2 code for FreeBSD only, the file is compiled with that branch), the generic code as information next to the fallback, and `iqconverter_int16` for `CQ15` (a lower accuracy class: truncating `>> 15`, a 32-bit accumulator, the odd samples halved; its status is the maintainer's open decision I23-1). Conversion only, without `remove_dc`; the wrapper clears the delay line and the FIR queue of the int16 converter completely, because its `reset` clears half of each (v1.0.10 uses `sizeof(int16_t)` for 32-bit queue entries), which leaves the first outputs to the allocator |
| rustfft | 6.4 (Cargo dependency of `benches/kernels`) | MIT OR Apache-2.0 | all; `wasm_simd` on wasm32 (target-specific) | the `C32` FFT reference except at `sse2` |
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
| `channelize` C32 (M45), 16 channels, 8 taps per branch (the block default), 4096 input samples of noise per call | liquid-dsp 1.8.3 `firpfbch2_crcf` analyzer (`benches/refs/liquid-channelizer.c`, D919): the 2x oversampled M/2-hop bank of ours, once per hop with SDRk's prototype, the SIMD dot product of the level and FFTW 3.3.10 for the transform (liquid's default build where FFTW is found; on wasm32 liquid's own FFT and the portable dot product); liquid also scales the transform by `1 / M`; our fallback as information |
| `channelize_fallback` C32 (the measured side is our fallback) | the same `firpfbch2_crcf` with liquid's portable dot product and its own FFT (portable C) |
| `channelize_block`, `channel_select_block` (the blocks `channelize` and `channel_select`, M45); `channel_bank_channelize`, `channel_bank_ddc` (M45 test 9: one channelizer with eight readers against eight `ddc` blocks) | our own code as information (`info`, `Family::Own`; the library comparison of the channelizer is the kernel entry `channelize`, D928, D930; [Channelizer blocks](#channelizer-blocks-and-channel-banks-m45)) |
| `band_pulse_detect_block` (the block `band_pulse_detect`, M45.1: 16 lanes at 250 kHz, per lane value) | our own code as information (`info`, `Family::Own`; the natural like-for-like reference, `rtl_433`'s `pulse_detect_package`, is GPL and is not used or linked, and no other library has a pulse detector on the lanes of a channelizer, D962; [Band pulse detector](#band-pulse-detector-m451)) |
| `fft` C32, N = 64, 1024, 4096 | FFTW at `sse2`, rustfft (`FftPlanner`, forward, in place) elsewhere |
| `fft` CQ15, N = 1024, block float and fixed scaling (noise of 8000 LSB, and of full scale for fixed scaling) | KissFFT `FIXED_POINT=16`; Ne10 int16 as information on `neon` |
| `rfft` f32, N = 1024 | FFTW's real-input transform (`fftwf_plan_dft_r2c_1d`, out of place) at `sse2` and `sse4.2` (FFTW has no codelets above SSE2, so it is the stronger reference there too), rustfft's real-input planner (`realfft` 3.5, forward, out of place) elsewhere; the other of the two and our fallback as information (FFTW also on `neon`) |
| `fft_64_fallback`, `fft_1024_fallback`, `fft_4096_fallback`, `rfft_1024_fallback`, `fft_cq15_1024_fallback` (block float), `fft_cq15_1024_default_fallback` (fixed scaling) (the measured side is our fallback, compiled for the level; the inputs of `fft_*`, `rfft_1024`, `fft_cq15_1024` and `fft_cq15_1024_default`) | CMSIS-DSP 1.18.0 generic C, compiled for the same level: `arm_cfft_f32` (the constant instances of `arm_const_structs.c`, bit reversal on) for C32, `arm_rfft_fast_f32` (packed output, unpacked outside the counted region) for `rfft`, `arm_cfft_q15` for CQ15 with KissFFT `FIXED_POINT=16` as information. The f32 transforms are the same work and accuracy class (137 to 141 dB on both sides), but CMSIS-DSP's radix-8/4 passes need fewer operations than the radix-2 butterflies of our contract; since D170 (three stages per pass off x86, the bit reversal in `N/4` steps) the entries are `release` except `rfft` at `sse4.2` (1.11 instructions) and N = 4096 at `simd128` (1.11), which are `dev`. `arm_cfft_q15` is a lower accuracy class (truncating shifts, 41.6 dB against 46.1 dB fixed scaling and 68.0 dB block float of ours, which rounds every butterfly) and has no block floating point: the block-float entry is `info` everywhere; the fixed-scaling entry is `release` on every level since D171 (no clamps where none can act; it was `info` at `sse2`, `neon` and `simd128`, where CMSIS-DSP was cheaper by more than the release limit). Not measured at `avx2-runtime` |
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
- The protocol entries of `sdrk-proto` (M37) have no SIMD branch (D496). The CRCs (`crc16_ccitt_false`, `crc16_x25`, `crc32`, `crc24_adsb`) are gated against GNU Radio's generic CRC `gr::digital::crc` of gr-digital with the same parameters and messages (`Family::GnuRadio`, D527), our code beside it as information; `frame_decode` and the blocks `frame_encode_block` and `frame_decode_block` have no reference (no C library has this framing): `info` against our own code at every level, `avx2-runtime` included; figures in [Protocol code](#protocol-code-sdrk-proto-m37).
- `symbol_sync` (`C32`) and `symbol_sync_real` (`f32`) (M38) stay with the fallback on every backend (every strobe position depends on the timing error of the strobe before it): the facade entry is `info` against our own fallback at every level, `avx2-runtime` included, without an efficiency gate and without a `_fallback` entry (D562). No library runs the same loop: GNU Radio's `symbol_sync_cc`/`_ff` interpolate with an 8-tap MMSE filter bank or a polyphase filter bank (and keep the detector history and the stream tags in C++ containers), liquid-dsp's `symsync` filters with a polyphase matched filter bank and its derivative; `symbol_sync` uses a 4-point cubic Lagrange interpolator, so either library would count a different amount of work. The workloads (D563) are the golden-case shapes with the block defaults (Gardner): BPSK with a raised cosine 0.35 at 4.1 samples per symbol, and boxcar-filtered real levels at 8 samples per symbol, both with noise; a sample is an input sample (cycles per symbol = cycles per sample times the samples per symbol). The block `slice` has no kernel and no entry.
- `costas` (`C32`) (M40) stays with the fallback on every backend (the phase of every symbol depends on the detector output of the symbol before it): the facade entry is `info` against our own fallback at every level, `avx2-runtime` included, without an efficiency gate (D608); the fallback is gated against GNU Radio's `costas_loop_cc` in `costas_fallback` (D607). The workload: QPSK with one sample per symbol, a start phase of 0.7 rad, 0.002 cycles per symbol and uniform noise of 0.15 per component, with the defaults order 4, `loop_bw` 0.02, `damping` 0.7071, `max_offset` 0.05; a sample is a symbol. `design::rrc` has no entry (a filter design, D606).
- `dot_complex` (`C32`) and `equalize` (`C32`) (M43): `dot_complex` is gated against VOLK's `volk_32fc_x2_dot_prod_32fc`, at the levels where VOLK's dispatch fuses against its protokernel without FMA (D821), and its fallback against CMSIS-DSP's `arm_cmplx_dot_prod_f32` (D822); the judged levels are `release` since the count of 2026-11-07. `equalize` has no SIMD library reference: the facade entry is `info` against our own fallback at every level, `avx2-runtime` included (D823), and since D841 every backend runs the fallback, and `equalize_fallback` is gated against the C translation of GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` (D838, `release` since the count of 2026-11-07). The workload is the block default (11 taps, no feedback, QPSK, `mu_cma` 0.05, `mu_dd` 0.02, `cma_symbols` 1000, `phase_bw` 0.02; GNU Radio with the same normalized step 0.02) on QPSK through `1 + 0.3 e^(j0.6) z^-1 + 0.1 e^(-j2) z^-2` with a carrier phase of 0.7 rad and uniform noise of 0.05; the state carries over between the calls, so the counted calls run in decision mode.
- `rds_shape` (`f32`) and `stereo_encode_rds` (`C32` and `f32` to `f32`) (M42) stay with the fallback on every backend (the shaper reads four interpolated values per sample from the pulse table at an f64 position and moves its bit window through a callback, the encoder is `stereo_encode` plus a third lookup in the tables of `rotate` at the tripled phase): the facade entry is `info` against our own fallback at every level, `avx2-runtime` included, without an efficiency gate and without a `_fallback` entry (D757). Neither GNU Radio nor liquid-dsp has an RDS modulator in-tree (GNU Radio's RDS support is the out-of-tree receiver gr-rds; liquid-dsp has no RDS), and no library has a stereo multiplex encoder with RDS. The workloads (D757): `rds_shape` at 192 kHz on random amplitudes +-1 (a stored cycle of 4096 bits), `stereo_encode_rds` the defaults of `fm_stereo_rds_mod` (pilot 19 kHz, `main` 0.87, `pilot` 0.09, `rds` 0.04) at 192 kHz on the audio of the `stereo_encode` workload and the `rds_shape` output of random bits.
- The RDS blocks of M41 (`rds_demod_block`, `rds_decode_block`) are composite blocks without a kernel of their own and have no reference (no C library has them): `info` against our own code at every level, `avx2-runtime` included (D678); `rds_demod` follows the SIMD paths of `interleave`, `rotate` and `fir_decimate` at each level, `rds_decode` is scalar. Figures in [RDS blocks](#rds-blocks-m41).
- Under the [measurement policy](#what-is-compared-with-what-d32) our own fallback is no fair gate for a SIMD branch, and since the fallback round no SIMD entry is `release` against it (issue #20, group A). `fm_mod` `C32` is `info` (D58), with GNU Radio's scalar `frequency_modulator_fc` and the fallback as information. `mag_sq` `CQ15` is `info` with the efficiency verdict (no SIMD library has the rounding and saturation of its rule; CMSIS-DSP's scalar `arm_cmplx_mag_squared_q15` gates the fallback entry). `quad_demod` `CQ15` (round 2), `rotate` `CQ15` (D45), `fm_mod` `CQ15` (D46), the biquad Q15 entries (D21, D39), `window` `CQ15` and `power_db` `Q15` (D43), the Q15 polyphase entries (D44) and `dot` `CQ15`, `dot_real` `Q15` and `scale` `CQ15` (D48) are `info` and judged by their simulated cycles and efficiency (see the notes below). `squelch` `C32` is gated against GNU Radio's `pwr_squelch_cc` (D41). The `C32` fallbacks of `agc` and `quad_demod` are gated against GNU Radio (`agc_fallback`, `quad_demod_fallback`) and the `CQ15` fallback of `quad_demod` against CMSIS-DSP (`quad_demod_cq15_fallback`). The fallbacks of `dc_block_real` and `dc_block` (`f32`/`C32`) are gated against liquid-dsp's DC blocker and that of `deemphasis` `f32` against GNU Radio's `fm_deemph` (D53, entries `<kernel>_fallback`); their SIMD entries (the `f64` block scan) stay `info` against the fallback, since no SIMD library has these recursions. The `info` entries of `squelch` `CQ15`, `agc` `CQ15` and the `Q15`/`CQ15` forms of `dc_block_real`, `deemphasis` and `dc_block` compare the fallback with itself and still need a scalar reference (group B; no established fixed-point implementation was found).

## Limits

`benches/limits.json` names, per kernel and level, `{ "status": "release" | "dev" | "info" }`, and the caps in `_limits` (`release` 1.10, `dev` 1.50, `warn` 1.10, `goal` 1.00, `runtime` 1.02; D14). `cargo xtask count` writes `benches/results/<platform>.json` (counts, ratio, SNR of ours and of the comparison, the verdict and whether the entry reaches the goal) and fails if a `release` entry exceeds 1.10× or a `dev` entry 1.50×, or if an entry is less accurate than its comparison by more than 1 dB. A kernel and level that is not listed fails. An entry may carry `"own_target"` (a number > 0): the efficiency target of its own-kernel comparison (`ours:<kernel>`, [`Reference::OwnKernel`], information), recorded in the results as `target` and `meets_target` next to the ratio and printed as `target 0.60: met` or `missed`; it never fails the command and does not replace the limit of the judged comparison (D102). The `halfband_decimate` entries use 0.60 (the work saved against `fir_decimate` by 2 with the expanded taps), the `real_to_complex` entries 1.10. The level `avx2-runtime` needs no entries: it is judged against our own `avx2` numbers with the cap `runtime`.

An entry may carry a `"waiver"` (ratio) or an `"accuracy_waiver"` text that names the open item in `docs/open-items.md` which tracks a limit that is not met yet. A waived exceedance is printed (`waived`) and recorded in the results file, but does not fail the command; the numbers are never hidden. A waiver that no longer covers anything, and a `dev` entry at or below 1.10×, are reported as a warning; a passing entry above `warn` (a `dev` entry between 1.10× and 1.50×) gets the verdict `warn`. `--strict` (the release gate of open item M13-4) additionally fails for every entry that is not a clean `release` entry: `dev`, `info`, a waiver, or a failure. The unit tests of the limit logic (`xtask/src/count.rs`) include a deliberately slow dummy kernel with status `release`, which makes the command fail.

## Results

Instructions per sample of our kernel divided by those of the comparison (lower is better), from the committed `benches/results/*.json`; `dev` marks an entry whose status is `dev`, `info` one that is reported but not judged, `waived` one whose ratio exceeds its limit under a waiver, `(accuracy waiver)` a `release` entry within its limit whose comparison is more accurate under an accuracy waiver, and `-` a level without the entry (the `<kernel>_fallback` entries are not measured at `avx2-runtime`, whose fallback is the baseline build). All other entries are `release`, and every entry at or below 1.00 reaches the goal. Measured with VOLK 3.3.0, FFTW 3.3.10, KissFFT 131.1.0 and rustfft 6.4 as built by `benches/refs/build.sh` on the development machine (the counts are deterministic: two complete runs gave identical lines on every platform). The fallback round (D51 to D62) added the `<kernel>_fallback` entries against CMSIS-DSP 1.18.0 generic C, GNU Radio `ec1bf15` and liquid-dsp 1.8.3 compiled for the same level, and the `CQ15` FIR entries; the tables were counted on the integrated tree on 2026-10-31 (123 entries, all four platforms, every entry within its limit). The round-2 follow-ups (D63 to D67: the block-float `CQ15` optimum, the CET markers simulated as `nop`, the grouped fallback where the `Q15`/`CQ15` FIR SIMD branch was slower, and the vectorizable square root of `mag_fallback`) were counted again on all four platforms and measured with `cargo measure --level avx2` on 2026-10-31 (123 entries, every entry within its limit). Round 3 (D166 to D174: `power_db`, `convert_out`, `biquad_real` and `fft` at 64 points on AVX-512 in the SIMD backend; the `C32` and `CQ15` FFT, `fir_decimate` `CQ15`, `fir_interpolate` `Q15`/`CQ15` and, on AArch64, `convert_cu8` fallbacks) was counted on all four platforms and measured with `cargo measure` on all five levels with a core model on 2026-11-01 (123 entries, every entry within its limit). M30 (D250 to D275: `add`, `add_real`, `mul`, `mul_real`, `interleave`, `deinterleave` and `oscillate`, 24 entries with their fallbacks) was counted at `sse2`, `sse4.2`, `avx2`, `avx2-runtime`, `avx512` and `neon`; its `simd128` cells were filled from `benches/results/wasm32.json` after the rounds below (2026-11-03), so a `-` in an M30 row means a level without the entry. The rows of `interleave` and `deinterleave` (both sample types) were counted again on all four platforms after their tuning (D281 to D286, 2026-11-03, `cargo xtask count --kernel ... --merge`), with their `simd128` cells; every one of them is within its limit, and the waivers of open item I30-1 for them are gone. The rows of `mul` C32 and `oscillate` C32 were counted again on all four platforms after their tuning (D287 to D290, 2026-11-03, `cargo xtask count --kernel mul --kernel oscillate --merge`), with their `simd128` cells: `mul` is gated against VOLK's protokernel without FMA at `avx2`, `avx512` and `neon` (D287), is within its limit at every level and has no waiver left; the `oscillate` SIMD path beats the fallback at every level (D289). The rows of `mul_fallback`, `mul_cq15_fallback` and `mul` (both sample types) were counted again on all four platforms after the tuning of the two fallbacks (D291, D292, 2026-11-03, `cargo xtask count --kernel mul_fallback --kernel mul_cq15_fallback --kernel mul --kernel mul_cq15 --merge`), with their `simd128` cells: both fallbacks are within the release limit at every level and their waivers are gone (0.48 to 0.93 of CMSIS-DSP's instructions). The `mul` `CQ15` SIMD entries at `sse2` and `simd128` run the fallback (backends.md) and follow it; at `avx512` the SIMD path (the AVX2 path) now counts 2.20 against 1.89 of the fallback (1.16, `info`). The `avx2-runtime` row of `mul` `CQ15` (1.06 against the fixed `avx2` build, before 1.11) is unchanged by this work: the dispatch overhead of the SIMD path, not the fallback; it fails the runtime limit of 1.02 like `add_real` `Q15` (1.13) and `mul_real` `Q15` (1.20), which open item I30-2 tracks. After D291 the fallback at `sse4.2` (4.89) counted fewer instructions than the `mul` `C32` SIMD path (6.27, the split form), so the SIMD path at `sse4.2` got the 128-bit form of the `avx2` product (D295, the same bits): 4.71 instructions per sample (0.78 of VOLK, before 1.04; `cargo xtask count --platform x86_64 --kernel mul --merge`) and 1.51 / 1.18 simulated cycles on `skylake` / `znver2` (before 1.53 / 1.59; the fallback 1.52 / 1.23). The rows of `add_real_q15_fallback`, `mul_real_q15_fallback` and of `add_real` and `mul_real` `Q15` were counted again at `simd128` after the tuning of the two fallbacks there (D294, 2026-11-03, `cargo xtask count --platform wasm32 --kernel add_real_q15_fallback --kernel mul_real_q15_fallback --kernel add_real_q15 --kernel mul_real_q15 --merge`), with their `simd128` cells, and the two fallbacks on the other three platforms (the same instructions per sample as committed, so not written): the fallbacks count 1.19 and 3.77 instructions per sample there (before 3.04 and 6.17), 0.65 and 0.96 of CMSIS-DSP, and their waivers are gone; the other levels are unchanged (the grouped loop is compiled for wasm32 only). At `simd128` the `add_real` `Q15` SIMD path (the token's `i16x8.add_sat_s`, 1.84) now counts more than the fallback (1.19); the entry stays `release` and passes its gate against VOLK (0.34). The `mul_real` `Q15` SIMD path (`i16x8.q15mulr_sat`, 1.83) stays `info` against the fallback (0.49), and `mul` `CQ15` at `simd128` runs the fallback (1.00 `info`). The entries `interleave_fallback`, `interleave_cq15_fallback`, `deinterleave_fallback` and `deinterleave_cq15_fallback` were added against VOLK's `generic` protokernels (D297) and counted on all four platforms after the tuning of the two fallbacks (D298, 2026-11-03, `cargo xtask count --kernel interleave_fallback --kernel interleave_cq15_fallback --kernel deinterleave_fallback --kernel deinterleave_cq15_fallback --kernel interleave_cq15 --merge`); before the tuning `deinterleave_fallback` and `deinterleave_cq15_fallback` failed their limit at `neon` (2.82, 5.10: the two passes over the input stayed scalar on AArch64) and `simd128` (1.42, 1.62), `deinterleave_cq15_fallback` at `sse2` (1.13), and `interleave` of both types was at 1.10 at `sse2` and `sse4.2`; now every one is within its goal (0.55 to 1.00). The `interleave` `CQ15` SIMD row follows the faster fallback at `sse2` and `sse4.2` (1.28 against 1.21, 1.05 `info`, before 0.92). At `simd128` the `oscillate` `CQ15` row is that of its new SIMD128 path (D299, 17.54 against 33.66 of the fallback, before 34.54, 1.03) and the `add_cq15_fallback` row that of the `Q15` stream through `zip_q15` (D318, 2.33 against 3.62 of CMSIS-DSP, before 3.92, 1.08; `cargo xtask count --platform wasm32 --kernel add_cq15_fallback --kernel oscillate_cq15 --merge`). The seven entries of M37 (`crc16_ccitt_false`, `crc16_x25`, `crc32`, `crc24_adsb`, `frame_decode`, `frame_encode_block`, `frame_decode_block`; per byte or per bit, see [Protocol code](#protocol-code-sdrk-proto-m37)) were counted on all four platforms on 2026-11-05 (`cargo xtask count --platform <p> --kernel ... --merge`) and their cycles measured with `cargo measure` on the five levels with a core model; every one of them is within its limit. `fail` marks a `release` entry that does not pass: its ratio exceeds the limit, or the comparison is more accurate than ours (`oscillate_fallback` C32 against CMSIS-DSP `arm_sin_cos_f32`).

| Kernel | sse2 | sse4.2 | avx2 | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | 0.62 | 0.70 | 0.61 | 0.62 | 0.57 | 0.38 |
| `convert` CS16 to C32 | 0.43 | 0.56 | 0.61 | 0.62 | 0.48 | 0.43 |
| `convert` CU8 to C32 | 0.62 | 0.83 | 0.61 | 0.62 | 0.67 | 0.36 |
| `convert_real` RS16 to f32 | 0.44 | 0.57 | 0.63 | 0.66 | 0.49 | 0.44 |
| `convert` CS8 to CQ15 | 0.61 | 0.66 | 0.59 | 0.31 | 0.46 | 0.59 |
| `convert` CU8 to CQ15 | 0.76 | 0.83 | 0.76 | 0.40 | 0.66 | 0.71 |
| `convert_out` C32 to CS16 | 0.92 | 0.92 | 0.92 | 0.71 | 0.60 | 0.20 |
| `convert_out` C32 to CS8 | 0.99 | 1.03 | 0.97 | 0.72 | 0.65 | 0.32 |
| `convert_out` C32 to CU8 | 1.38 dev | 1.08 | 1.06 | 0.63 | 0.95 | 0.52 |
| `convert_out_real` f32 to RS16 | 0.92 | 0.92 | 0.93 | 0.74 | 0.61 | 0.20 |
| `convert_out_real` f32 to RS8 | 1.00 | 1.04 | 0.99 | 0.77 | 0.66 | 0.32 |
| `convert_out_real` f32 to RU8 | 1.39 dev | 1.09 | 1.08 | 0.65 | 0.96 | 0.52 |
| `convert_cs8_fallback` (CMSIS-DSP) | 1.07 | 0.85 | 0.74 | 0.74 | 0.86 | 0.92 |
| `convert_cs16_fallback` (CMSIS-DSP) | 1.01 | 0.85 | 0.75 | 0.75 | 0.40 | 0.82 |
| `convert_cu8_fallback` (CMSIS-DSP) | 1.07 | 1.01 | 0.90 | 0.90 | 0.86 | 1.00 |
| `convert_real_rs16_fallback` (CMSIS-DSP) | 1.01 | 0.86 | 0.76 | 0.76 | 0.41 | 0.82 |
| `convert_cq15_cs8_fallback` (CMSIS-DSP) | 0.86 | 0.84 | 0.72 | 0.72 | 0.87 | 0.22 |
| `convert_cq15_cu8_fallback` (CMSIS-DSP) | 1.01 | 1.01 | 0.89 | 0.89 | 1.20 dev | 0.24 |
| `convert_out_cs16_fallback` (CMSIS-DSP) | 0.70 info | 1.48 info | 1.23 info | 1.21 info | 2.01 info | 0.30 info |
| `convert_out_cs8_fallback` (CMSIS-DSP) | 0.82 info | 1.40 info | 1.82 info | 1.47 info | 2.38 info | 0.30 info |
| `convert_out_cu8_fallback` (CMSIS-DSP) | 1.44 info | 2.50 info | 2.78 info | 3.15 info | 3.80 info | 0.45 info |
| `convert_out_real_rs16_fallback` (CMSIS-DSP) | 0.70 info | 1.48 info | 1.24 info | 1.22 info | 2.01 info | 0.30 info |
| `convert_out_real_rs8_fallback` (CMSIS-DSP) | 0.83 info | 1.41 info | 1.82 info | 1.48 info | 2.38 info | 0.30 info |
| `convert_out_real_ru8_fallback` (CMSIS-DSP) | 1.44 info | 2.50 info | 2.77 info | 3.14 info | 3.79 info | 0.45 info |
| `rotate` C32 | 0.16 info | 0.18 info | 0.45 | 0.25 | 0.60 | 0.20 info |
| `rotate` CQ15 | 0.26 info | 0.24 info | 0.36 info | 0.20 info | 0.18 info | 0.72 info |
| `rotate_fallback` C32 (GNU Radio) | 0.57 | 0.56 | 0.22 | 0.08 | 0.93 | 0.38 |
| `rotate_cq15_fallback` (CMSIS-DSP composition) | 0.45 | 0.38 | 0.16 | 0.10 | 0.41 | 0.17 |
| `fir` C32, 64 taps | 0.49 | 0.48 | 0.13 | 0.06 | 0.36 | 0.11 info |
| `fir_decimate` C32, 127 taps, M = 10 | 0.51 | 0.50 | 0.21 | 0.18 | 0.52 | 0.14 info |
| `fir_interpolate` C32, L = 2, 128 taps | 0.50 | 0.50 | 0.19 | 0.12 | 0.38 | 0.12 info |
| `resample` C32, 3 / 2, 192 taps | 0.51 | 0.50 | 0.24 | 0.19 | 0.49 | 0.15 info |
| `fir_interpolate_real` f32, L = 2, 128 taps | 0.43 | 0.41 | 0.17 | 0.12 | 0.34 | 0.16 info |
| `resample_real` f32, 3 / 2, 192 taps | 0.50 | 0.47 | 0.26 | 0.24 | 0.39 | 0.19 info |
| `fir_interpolate_real` Q15, L = 2, 128 taps | 1.00 info | 0.36 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `resample_real` Q15, 3 / 2, 192 taps | 1.00 info | 0.26 info | 0.61 info | 0.57 info | 0.65 info | 0.68 info |
| `fir` CQ15, 64 taps (fallback) | 1.00 info | 0.35 info | 1.00 info | 1.00 info | 1.08 info | 0.90 info |
| `fir_decimate` CQ15, 127 taps, M = 10 (fallback) | 1.00 info | 0.30 info | 1.00 info | 0.63 info | 0.62 info | 0.91 info |
| `fir_interpolate` CQ15, L = 2, 128 taps (fallback) | 1.00 info | 0.34 info | 1.00 info | 1.00 info | 0.69 info | 1.00 info |
| `resample` CQ15, 3 / 2, 192 taps (fallback) | 1.00 info | 0.31 info | 1.00 info | 0.80 info | 0.61 info | 0.91 info |
| `mag_sq` C32 | 0.71 | 0.62 | 0.78 | 0.39 | 0.52 | 0.60 |
| `mag_sq` CQ15 | 0.37 info | 0.69 info | 0.72 info | 0.82 info | 0.54 info | 0.40 info |
| `quad_demod` C32 | 0.13 info | 0.14 info | 0.65 | 0.45 | 0.58 | 0.18 info |
| `quad_demod` CQ15 | 0.19 info | 0.14 info | 0.07 info | 0.02 info | 0.10 info | 0.12 info |
| `quad_demod_fallback` C32 (GNU Radio) | 1.32 info | 1.31 info | 1.14 info | 1.12 info | 2.14 info | 1.22 info |
| `quad_demod_cq15_fallback` (CMSIS-DSP) | 0.64 | 0.65 | 0.64 | 0.65 | 0.70 | 0.55 |
| `fm_mod` C32 | 0.22 info | 0.18 info | 0.11 info | 0.04 info | 0.19 info | 0.28 info |
| `fm_mod` CQ15 | 0.46 info | 0.28 info | 0.41 info | 0.18 info | 0.25 info | 0.72 info |
| `fm_mod_fallback` C32 (GNU Radio) | 0.49 | 0.49 | 0.26 | 0.12 | 0.60 | 0.49 |
| `fm_mod_cq15_fallback` (CMSIS-DSP composition) | 0.60 | 0.61 | 0.32 | 0.23 | 0.70 | 0.58 |
| `squelch` C32 | 0.41 | 0.42 | 0.38 | 0.37 | 0.41 | 0.64 |
| `squelch` CQ15 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `mag` C32 | 0.73 | 0.66 | 0.64 | 0.52 | 0.72 | 0.80 |
| `mag` CQ15 | 0.24 | 0.46 | 0.85 | 0.46 | 0.72 | 0.49 |
| `window` C32 | 0.90 | 0.90 | 0.65 | 0.37 | 0.82 | 0.72 info |
| `window` CQ15 | 0.43 info | 0.40 info | 0.36 info | 0.31 info | 0.33 info | 0.40 info |
| `power_db` f32 | 0.28 | 1.07 | 1.36 info | 0.95 | 1.14 info | 0.29 info |
| `power_db` Q15 | 0.98 info | 1.01 info | 0.86 info | 0.47 info | 0.98 info | 1.03 info |
| `mag_sq_fallback` C32 (CMSIS-DSP) | 0.81 | 0.81 | 0.67 | 0.67 | 0.78 | 0.86 |
| `mag_sq_q15_fallback` (CMSIS-DSP) | 0.33 | 0.22 | 0.23 | 0.16 | 0.79 | 0.96 |
| `mag_fallback` C32 (CMSIS-DSP) | 0.76 | 0.76 | 0.74 | 0.65 | 0.67 | 0.76 |
| `mag_q15_fallback` (CMSIS-DSP) | 0.09 | 0.06 | 0.04 | 0.02 | 0.06 | 0.06 |
| `window_fallback` C32 (CMSIS-DSP) | 0.96 | 0.96 | 0.93 | 0.40 | 0.58 | 0.40 |
| `window_cq15_fallback` (CMSIS-DSP) | 0.91 | 0.95 | 0.75 | 0.59 | 0.82 | 0.66 |
| `power_db_fallback` f32 (CMSIS-DSP) | 0.30 | 0.29 | 0.17 | 0.14 | 0.28 | 0.36 |
| `power_db_q15_fallback` (CMSIS-DSP) | 0.29 info | 0.28 info | 0.17 info | 0.14 info | 0.21 info | 0.33 info |
| `agc` C32 | 0.99 | 0.96 | 0.72 | 0.67 | 1.32 dev | 1.01 |
| `agc` CQ15 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `agc_fallback` C32 (GNU Radio) | 1.18 dev | 1.18 dev | 0.97 | 0.97 | 1.32 dev | 1.00 |
| `dc_block_real` f32 | 1.19 info | 1.13 info | 0.75 info | 0.63 info | 0.46 info | 0.97 info |
| `dc_block_real` Q15 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `deemphasis` f32 | 0.55 info | 0.62 info | 0.49 info | 0.35 info | 0.47 info | 0.90 info |
| `deemphasis` Q15 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `dc_block` C32 | 1.16 info | 1.28 info | 1.05 info | 0.69 info | 1.00 info | 1.00 info |
| `dc_block` CQ15 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `dc_block_real_fallback` f32 (liquid-dsp) | 0.11 | 0.11 | 0.09 | 0.09 | 0.18 | 0.11 |
| `deemphasis_fallback` f32 (GNU Radio) | 0.37 (accuracy waiver) | 0.33 (accuracy waiver) | 0.22 (accuracy waiver) | 0.25 (accuracy waiver) | 0.31 (accuracy waiver) | 0.18 (accuracy waiver) |
| `dc_block_fallback` C32 (liquid-dsp) | 0.12 | 0.10 | 0.07 | 0.06 | 0.14 | 0.10 |
| `biquad` C32, 2 sections (KFR) | 0.21 | 0.23 | 0.25 | 0.18 | 0.27 | 0.44 |
| `biquad` CQ15, 2 sections (fallback) | 1.00 info | 0.51 info | 0.18 info | 0.18 info | 0.43 info | 0.35 info |
| `biquad_real` f32, 2 sections (KFR) | 0.45 | 0.47 | 0.52 | 0.36 | 0.59 | 0.80 |
| `biquad_real` Q15, 2 sections (fallback) | 1.00 info | 0.46 info | 0.40 info | 0.40 info | 0.46 info | 0.43 info |
| `biquad_fallback` C32 (CMSIS-DSP) | 0.56 | 0.56 | 0.53 | 0.53 | 0.52 | 0.58 |
| `biquad_cq15_fallback` (CMSIS-DSP) | 0.66 | 0.66 | 0.72 | 0.71 | 0.58 | 0.75 |
| `biquad_real_fallback` f32 (CMSIS-DSP) | 0.99 | 0.98 | 0.95 | 0.89 | 0.97 | 0.88 |
| `biquad_real_q15_fallback` (CMSIS-DSP) | 0.61 | 0.61 | 0.64 | 0.64 | 0.52 | 0.71 |
| `fir_fallback` C32 (CMSIS-DSP) | 0.43 | 0.43 | 0.78 | 0.78 | 0.51 | 0.75 |
| `fir_cq15_fallback` (CMSIS-DSP) | 0.64 | 0.64 | 0.26 | 0.18 | 0.44 | 1.19 dev |
| `fir_decimate_fallback` C32 (CMSIS-DSP) | 0.68 (accuracy waiver) | 0.68 (accuracy waiver) | 0.99 (accuracy waiver) | 0.95 (accuracy waiver) | 0.63 (accuracy waiver) | 1.07 (accuracy waiver) |
| `fir_decimate_cq15_fallback` (CMSIS-DSP) | 1.36 dev | 1.58 waived | 1.34 dev | 1.07 | 2.09 waived | 0.64 |
| `fir_interpolate_fallback` C32 (CMSIS-DSP) | 0.57 | 0.57 | 0.41 | 0.40 | 0.52 | 1.02 |
| `fir_interpolate_cq15_fallback` (CMSIS-DSP) | 1.19 dev | 1.19 dev | 0.53 | 0.40 | 1.01 | 0.54 |
| `resample_fallback` C32 (CMSIS-DSP interpolator) | 0.73 info | 0.73 info | 0.87 info | 0.85 info | 0.88 info | 1.32 info |
| `resample_cq15_fallback` (CMSIS-DSP interpolator) | 1.28 info | 1.28 info | 0.74 info | 0.59 info | 1.94 info | 0.91 info |
| `fir_interpolate_real_fallback` f32 (CMSIS-DSP) | 0.66 | 0.66 | 0.52 | 0.52 | 0.63 | 1.18 dev |
| `fir_interpolate_real_q15_fallback` (CMSIS-DSP) | 1.28 dev | 1.28 dev | 0.49 | 0.38 | 0.86 | 0.76 |
| `resample_real_fallback` f32 (CMSIS-DSP interpolator) | 1.44 info | 1.23 info | 1.64 info | 0.95 info | 1.28 info | 2.33 info |
| `resample_real_q15_fallback` (CMSIS-DSP interpolator) | 1.70 info | 1.70 info | 1.47 info | 1.30 info | 2.35 info | 1.48 info |
| `dot` C32, 4096 samples | 1.07 | 1.07 | 0.65 | 0.92 | 1.04 | 0.24 info |
| `dot` CQ15, 4096 samples | 0.35 info | 0.44 info | 0.55 info | 0.28 info | 1.00 info | 0.38 info |
| `dot_real` f32, 4096 samples | 1.09 | 0.88 | 0.50 | 0.58 | 0.69 | 0.54 info |
| `dot_real` Q15, 4096 samples | 0.36 info | 0.50 info | 0.49 info | 0.49 info | 0.63 info | 0.28 info |
| `scale` C32 | 0.56 | 0.56 | 0.48 | 0.23 | 0.62 | 0.92 info |
| `scale` CQ15 | 1.00 info | 0.25 info | 0.19 info | 0.17 info | 0.23 info | 0.34 info |
| `dot_fallback` C32 (CMSIS-DSP) | 0.80 | 0.68 | 0.87 | 0.67 | 0.77 | 0.65 |
| `dot_cq15_fallback` (CMSIS-DSP) | 1.48 dev | 1.28 dev | 1.09 | 1.10 | 0.73 | 0.69 |
| `dot_real_fallback` f32 (CMSIS-DSP) | 1.00 | 0.88 | 0.95 | 0.95 | 0.77 | 0.48 |
| `dot_real_q15_fallback` (CMSIS-DSP) | 1.23 dev | 1.05 | 1.06 | 1.02 | 0.83 | 0.90 |
| `scale_fallback` C32 (CMSIS-DSP) | 1.00 | 0.75 | 0.55 | 0.36 | 0.42 | 1.14 dev |
| `scale_cq15_fallback` (CMSIS-DSP) | 0.69 | 0.61 | 0.62 | 0.44 | 0.79 | 0.98 |
| `add` C32 (VOLK) | 0.79 | 0.79 | 0.59 | 0.29 | 0.57 | 0.82 |
| `add` CQ15 (VOLK `volk_16i_x2_add_saturated_16i`) | 0.73 | 0.73 | 0.57 | 0.29 | 0.64 | 0.22 |
| `add_real` f32 (VOLK) | 0.79 | 0.79 | 0.59 | 0.59 | 0.58 | 0.60 |
| `add_real` Q15 (VOLK) | 0.73 | 0.73 | 0.58 | 0.30 | 0.64 | 0.22 |
| `mul` C32 (VOLK; fused at `avx2`, `avx512`, `neon` as ours) | 0.27 | 0.78 | 0.77 | 0.38 | 0.74 | 0.30 |
| `mul` CQ15 (our fallback) | 1.00 info | 0.61 info | 0.64 info | 1.16 info | 0.81 info | 1.00 info |
| `mul_real` f32 (VOLK) | 0.79 | 0.79 | 0.59 | 0.57 | 0.57 | 0.60 |
| `mul_real` Q15 (our fallback) | 1.00 info | 0.44 info | 0.31 info | 0.34 info | 0.28 info | 0.49 info |
| `interleave` f32 (VOLK) | 0.90 | 0.90 | 0.76 | 0.35 | 0.50 | 0.76 |
| `interleave` CQ15 (our fallback) | 1.05 info | 1.05 info | 0.89 info | 0.70 info | 0.59 info | 0.76 info |
| `deinterleave` C32 (VOLK) | 0.90 | 0.90 | 0.77 | 0.47 | 0.78 | 0.76 |
| `deinterleave` CQ15 (VOLK) | 0.58 | 0.63 | 0.70 | 0.27 | 0.78 | 0.76 |
| `oscillate` C32 (our fallback) | 0.36 info | 0.33 info | 0.42 info | 0.47 info | 0.30 info | 0.78 info |
| `oscillate` CQ15 (our fallback) | 0.38 info | 0.26 info | 0.36 info | 0.16 info | 0.26 info | 0.52 info |
| `add_fallback` C32 (CMSIS-DSP) | 1.00 | 1.00 | 0.75 | 0.50 | 0.43 | 0.25 |
| `add_cq15_fallback` (CMSIS-DSP) | 0.15 | 0.31 | 0.21 | 0.16 | 0.23 | 0.64 |
| `add_real_fallback` f32 (CMSIS-DSP) | 0.79 | 0.79 | 0.75 | 0.75 | 0.71 | 0.38 |
| `add_real_q15_fallback` (CMSIS-DSP) | 0.12 | 0.24 | 0.21 | 0.25 | 0.39 | 0.65 |
| `mul_fallback` C32 (CMSIS-DSP) | 0.76 | 0.93 | 0.81 | 0.79 | 0.48 | 0.82 |
| `mul_cq15_fallback` (CMSIS-DSP, truncating Q3.13 product) | 0.90 | 0.78 | 0.80 | 0.52 | 0.57 | 0.74 |
| `mul_real_fallback` f32 (CMSIS-DSP) | 0.79 | 0.79 | 0.75 | 0.75 | 0.71 | 0.38 |
| `mul_real_q15_fallback` (CMSIS-DSP, truncating) | 0.88 | 0.81 | 0.89 | 0.54 | 1.00 | 0.96 |
| `oscillate_fallback` C32 (CMSIS-DSP `arm_sin_cos_f32`) | 0.25 fail | 0.26 fail | 0.13 fail | 0.04 fail | 0.25 fail | 0.16 fail |
| `oscillate_cq15_fallback` (CMSIS-DSP `arm_cos_q15`, `arm_sin_q15`) | 0.46 | 0.47 | 0.24 | 0.15 | 0.43 | 0.32 |
| `interleave_fallback` f32 (VOLK `generic`) | 0.95 | 0.95 | 0.86 | 0.55 | 1.00 | 1.00 |
| `interleave_cq15_fallback` (VOLK's `generic` interleave loop on `int16`) | 0.95 | 0.95 | 0.86 | 0.56 | 1.00 | 1.00 |
| `deinterleave_fallback` C32 (VOLK `generic`) | 0.77 | 0.77 | 0.86 | 0.83 | 0.83 | 1.00 |
| `deinterleave_cq15_fallback` (VOLK `generic`) | 0.74 | 0.88 | 0.90 | 0.61 | 0.84 | 1.00 |
| `fft` C32, N = 64 | 0.79 | 0.59 | 0.99 | 0.87 | 0.57 | 0.56 |
| `fft` C32, N = 1024 | 0.98 | 0.74 | 0.85 | 0.50 | 0.59 | 0.64 |
| `fft` C32, N = 4096 | 0.93 | 0.73 | 0.82 | 0.50 | 0.56 | 0.64 |
| `fft` CQ15, N = 1024, block float | 0.12 | 0.10 | 0.04 | 0.02 | 0.09 | 0.11 |
| `fft` CQ15, N = 1024, fixed scaling | 0.11 | 0.09 | 0.04 | 0.02 | 0.07 | 0.09 |
| `fft` CQ15, N = 1024, fixed scaling, full scale | 0.12 | 0.09 | 0.04 | 0.02 | 0.08 | 0.10 |
| `rfft` f32, N = 1024 | 1.07 | 1.07 | 0.60 | 0.40 | 0.56 | 0.71 |
| `fft_64_fallback` C32 (CMSIS-DSP) | 1.06 | 1.04 | 0.91 | 0.84 | 0.81 | 1.03 |
| `fft_1024_fallback` C32 (CMSIS-DSP) | 1.03 | 1.02 | 0.88 | 0.82 | 0.86 | 1.05 |
| `fft_4096_fallback` C32 (CMSIS-DSP) | 1.09 | 1.07 | 0.92 | 0.87 | 0.88 | 1.11 dev |
| `fft_cq15_1024_fallback` block float (CMSIS-DSP) | 1.55 info | 1.28 info | 1.19 info | 1.12 info | 1.84 info | 1.42 info |
| `fft_cq15_1024_default_fallback` fixed scaling (CMSIS-DSP) | 0.97 | 0.72 | 0.71 | 0.78 | 0.96 | 1.07 |
| `rfft_1024_fallback` f32 (CMSIS-DSP) | 1.10 | 1.11 dev | 0.97 | 0.90 | 0.87 | 1.07 |
| `halfband_decimate` C32 (liquid-dsp `resamp2_crcf`) | 0.23 | 0.23 | 0.12 | 0.08 | 0.15 | 0.21 |
| `halfband_decimate_real` f32 (liquid-dsp `resamp2_rrrf`) | 0.14 | 0.12 | 0.08 | 0.05 | 0.09 | 0.16 |
| `real_to_complex` C32 (liquid-dsp `firhilbf`) | 0.22 | 0.18 | 0.13 | 0.08 | 0.14 | 0.19 |
| `real_to_complex_airspy` C32 (libairspy `iqconverter_float`, SSE2, second gate) | 0.60 | - | - | - | - | - |
| `halfband_decimate` CQ15 (our fallback) | 0.34 info | 0.61 info | 0.61 info | 0.57 info | 0.69 info | 0.90 info |
| `halfband_decimate_real` Q15 (our fallback) | 0.25 info | 0.31 info | 0.27 info | 0.21 info | 0.34 info | 0.55 info |
| `real_to_complex` CQ15 (libairspy `iqconverter_int16`, provisional) | 0.51 info | 0.60 info | 0.54 info | 0.36 info | 0.61 info | 0.54 info |
| `halfband_decimate_fallback` C32 (liquid-dsp portable) | 0.31 | 0.32 | 0.22 | 0.25 | 0.35 | 0.28 |
| `halfband_decimate_real_fallback` f32 (liquid-dsp portable) | 0.53 | 0.53 | 0.52 | 0.52 | 0.44 | 0.33 |
| `real_to_complex_fallback` C32 (liquid-dsp portable) | 0.78 | 0.74 | 0.76 | 0.75 | 0.69 | 0.38 |
| `halfband_decimate_cq15_fallback` (CMSIS-DSP composition) | 0.94 | 0.44 | 0.33 | 0.27 | 0.50 | 0.71 |
| `halfband_decimate_real_q15_fallback` (CMSIS-DSP composition) | 1.02 | 0.69 | 0.59 | 0.53 | 0.81 | 1.06 |
| `real_to_complex_cq15_fallback` (libairspy `iqconverter_int16`, provisional) | 1.74 info | 1.77 info | 1.99 info | 1.70 info | 1.79 info | 0.95 info |
| `stereo_matrix` C32 (VOLK composition) | 0.41 | 0.41 | 0.31 | 0.21 | 0.28 | 0.34 |
| `stereo_matrix` CQ15 (our fallback) | 0.49 info | 0.64 info | 0.65 info | 0.50 info | 1.00 info | 1.00 info |
| `stereo_matrix_fallback` C32 (CMSIS-DSP planar) | 1.23 waived | 1.09 | 0.90 | 0.57 | 0.38 | 0.23 |
| `stereo_matrix_cq15_fallback` (CMSIS-DSP planar) | 0.24 | 0.38 | 0.32 | 0.24 | 0.23 | 0.99 |
| `stereo_pll_fallback` f32 (liquid-dsp `nco_crcf`) | 1.33 waived | 1.30 waived | 1.15 waived | 1.08 | 1.15 waived | 1.32 waived |
| `stereo_pll_q15_fallback` (CMSIS-DSP oscillator) | 0.63 info | 0.63 info | 0.63 info | 0.63 info | 0.51 info | 0.80 info |
| `preemph_limit` f32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `preemph_limit_q15` (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `noise_blank` C32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `noise_blank_cq15` (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `auto_notch` f32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `am_mod` C32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `am_mod_cq15` (our fallback) | 1.00 info | 0.45 info | 0.41 info | 0.78 info | 0.44 info | 0.45 info |
| `am_mod_fallback` C32 (liquid-dsp `ampmodem`) | 1.63 waived | 1.63 waived | 0.73 | 0.45 | 1.99 waived | 0.23 |
| `am_mod_cq15_fallback` (CMSIS-DSP composition) | 0.47 | 0.23 | 0.14 | 0.07 | 0.28 | 0.28 |
| `preemph_limit_stereo` C32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `stereo_encode` C32 to f32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `crc16_ccitt_false` (GNU Radio `gr::digital::crc`, per byte) | 0.80 | 0.80 | 0.67 | 0.67 | 0.77 | 0.81 |
| `crc16_x25` (GNU Radio `gr::digital::crc`, per byte) | 0.93 | 0.93 | 0.84 | 0.79 | 0.99 | 0.86 |
| `crc32` (GNU Radio `gr::digital::crc`, per byte) | 0.81 | 0.81 | 0.72 | 0.67 | 0.99 | 0.86 |
| `crc24_adsb` (GNU Radio `gr::digital::crc`, per byte) | 0.75 | 0.75 | 0.67 | 0.67 | 0.88 | 0.88 |
| `frame_decode` (our code, per bit) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `frame_encode_block` (our code, per output sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `frame_decode_block` (our code, per input sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `symbol_sync` C32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `symbol_sync_real` f32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `costas` C32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `costas_fallback` C32 (GNU Radio `costas_loop_cc`) | 0.76 | 0.74 | 0.70 | 0.69 | 0.85 | 0.89 |
| `rds_demod_block` (our code, per input sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `rds_decode_block` (our code, per input sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `rds_shape` f32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 0.99 info |
| `stereo_encode_rds` C32 and f32 to f32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `morse_encode_block` (our code, per output sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `morse_decode_block` (our code, per input sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `dot_complex` C32 (VOLK `volk_32fc_x2_dot_prod_32fc`) | 0.62 info | 0.88 | 0.78 | 0.78 | 0.88 | 0.33 info |
| `dot_complex_fallback` C32 (CMSIS-DSP `arm_cmplx_dot_prod_f32`) | 0.96 | 0.58 | 0.65 | 0.49 | 0.78 | 0.37 |
| `equalize` C32 (our fallback) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `equalize_fallback` C32 (GNU Radio `linear_equalizer`) | 0.70 | 0.69 | 0.73 | 0.67 | 0.89 | 0.54 |
| `channelize` C32 (liquid-dsp `firpfbch2_crcf`, SIMD dot products and FFTW) | 0.28 | 0.28 | 0.15 | 0.17 | 0.12 | 0.29 |
| `channelize_fallback` C32 (liquid-dsp `firpfbch2_crcf`, portable) | 0.45 | 0.44 | 0.35 | 0.36 | 0.51 | 0.46 |
| `channelize_block` (our code, per input sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `channel_select_block` (our code, per output sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `channel_bank_channelize` (our code, per input sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `channel_bank_ddc` (our code, per input sample) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `band_pulse_detect_block` (our code, per lane value) | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |

Instructions per sample, ours / comparison (`avx2-runtime`: ours only, judged against ours at `avx2`):

| Kernel | sse2 | sse4.2 | avx2 | avx2-runtime | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | 2.50 / 4.02 | 2.12 / 3.02 | 1.08 / 1.77 | 1.08 | 0.56 / 0.90 | 1.40 / 2.46 | 4.22 / 11.03 |
| `convert` CS16 to C32 | 2.38 / 5.52 | 2.12 / 3.77 | 1.08 / 1.77 | 1.08 | 0.55 / 0.89 | 1.46 / 3.02 | 4.34 / 10.03 |
| `convert` CU8 to C32 | 2.50 / 4.02 | 2.50 / 3.02 | 1.08 / 1.77 | 1.09 | 0.56 / 0.89 | 1.65 / 2.46 | 3.96 / 11.03 |
| `convert_real` RS16 to f32 | 1.21 / 2.77 | 1.08 / 1.89 | 0.56 / 0.89 | 0.57 | 0.30 / 0.46 | 0.75 / 1.52 | 2.20 / 5.03 |
| `convert` CS8 to CQ15 | 1.00 / 1.65 | 1.00 / 1.52 | 0.46 / 0.77 | 0.46 | 0.23 / 0.77 | 0.59 / 1.27 | 2.84 / 4.78 |
| `convert` CU8 to CQ15 | 1.26 / 1.65 | 1.26 / 1.52 | 0.58 / 0.77 | 0.59 | 0.31 / 0.77 | 0.84 / 1.27 | 3.37 / 4.78 |
| `convert_out` C32 to CS16 | 3.68 / 4.02 | 3.68 / 4.02 | 1.97 / 2.15 | 1.98 | 0.72 / 1.02 | 2.04 / 3.39 | 5.67 / 29.03 |
| `convert_out` C32 to CS8 | 3.62 / 3.65 | 3.62 / 3.52 | 1.90 / 1.96 | 1.91 | 0.74 / 1.02 | 2.17 / 3.33 | 9.17 / 29.03 |
| `convert_out` C32 to CU8 | 5.56 / 4.02 | 5.43 / 5.02 | 2.34 / 2.21 | 2.35 | 1.40 / 2.21 | 3.95 / 4.15 | 15.17 / 29.03 |
| `convert_out_real` f32 to RS16 | 1.87 / 2.02 | 1.87 / 2.02 | 1.01 / 1.08 | 1.02 | 0.39 / 0.52 | 1.04 / 1.71 | 2.88 / 14.53 |
| `convert_out_real` f32 to RS8 | 1.84 / 1.83 | 1.84 / 1.77 | 0.98 / 0.99 | 0.99 | 0.40 / 0.52 | 1.10 / 1.68 | 4.62 / 14.53 |
| `convert_out_real` f32 to RU8 | 2.81 / 2.02 | 2.75 / 2.52 | 1.20 / 1.12 | 1.21 | 0.73 / 1.12 | 2.00 / 2.08 | 7.62 / 14.53 |
| `convert_cs8_fallback` (CMSIS-DSP) | 4.29 / 4.02 | 2.79 / 3.27 | 1.22 / 1.64 | - | 1.22 / 1.64 | 2.17 / 2.51 | 11.06 / 12.03 |
| `convert_cs16_fallback` (CMSIS-DSP) | 3.79 / 3.76 | 2.79 / 3.26 | 1.22 / 1.64 | - | 1.22 / 1.64 | 1.79 / 4.51 | 10.07 / 12.28 |
| `convert_cu8_fallback` (CMSIS-DSP) | 4.29 / 4.02 | 3.29 / 3.27 | 1.47 / 1.64 | - | 1.47 / 1.64 | 2.17 / 2.51 | 12.06 / 12.03 |
| `convert_real_rs16_fallback` (CMSIS-DSP) | 1.91 / 1.89 | 1.41 / 1.64 | 0.63 / 0.83 | - | 0.63 / 0.83 | 0.92 / 2.26 | 5.06 / 6.15 |
| `convert_cq15_cs8_fallback` (CMSIS-DSP) | 1.41 / 1.64 | 1.16 / 1.39 | 0.51 / 0.70 | - | 0.51 / 0.70 | 0.67 / 0.76 | 4.81 / 22.27 |
| `convert_cq15_cu8_fallback` (CMSIS-DSP) | 1.66 / 1.64 | 1.41 / 1.39 | 0.63 / 0.70 | - | 0.63 / 0.70 | 0.92 / 0.76 | 5.31 / 22.27 |
| `convert_out_cs16_fallback` (CMSIS-DSP) | 6.79 / 9.77 | 7.04 / 4.76 | 2.48 / 2.02 | - | 1.98 / 1.64 | 5.54 / 2.76 | 17.57 / 58.52 |
| `convert_out_cs8_fallback` (CMSIS-DSP) | 8.04 / 9.77 | 7.04 / 5.02 | 3.79 / 2.08 | - | 2.23 / 1.52 | 6.29 / 2.64 | 17.57 / 58.52 |
| `convert_out_cu8_fallback` (CMSIS-DSP) | 14.04 / 9.77 | 12.54 / 5.02 | 5.79 / 2.08 | - | 4.79 / 1.52 | 10.04 / 2.64 | 26.57 / 58.52 |
| `convert_out_real_rs16_fallback` (CMSIS-DSP) | 3.42 / 4.89 | 3.54 / 2.39 | 1.26 / 1.02 | - | 1.01 / 0.83 | 2.79 / 1.39 | 8.82 / 29.27 |
| `convert_out_real_rs8_fallback` (CMSIS-DSP) | 4.04 / 4.89 | 3.54 / 2.52 | 1.92 / 1.05 | - | 1.14 / 0.77 | 3.17 / 1.33 | 8.82 / 29.27 |
| `convert_out_real_ru8_fallback` (CMSIS-DSP) | 7.04 / 4.89 | 6.29 / 2.52 | 2.92 / 1.05 | - | 2.42 / 0.77 | 5.04 / 1.33 | 13.31 / 29.27 |
| `rotate` C32 | 7.42 / 46.11 | 7.41 / 41.11 | 2.39 / 5.28 | 2.41 | 1.24 / 4.99 | 3.58 / 5.99 | 15.97 / 79.23 |
| `rotate` CQ15 | 12.78 / 49.01 | 10.02 / 41.01 | 5.78 / 15.85 | 5.79 | 2.06 / 10.27 | 7.02 / 40.01 | 28.29 / 39.16 |
| `rotate_fallback` C32 (GNU Radio) | 35.01 / 61.10 | 34.01 / 61.10 | 13.40 / 61.10 | - | 5.02 / 61.10 | 26.01 / 28.09 | 35.29 / 92.21 |
| `rotate_cq15_fallback` (CMSIS-DSP composition) | 49.01 / 108.61 | 41.01 / 107.14 | 15.85 / 101.60 | - | 10.27 / 100.72 | 40.01 / 97.02 | 39.16 / 229.23 |
| `fir` C32, 64 taps | 118.50 / 244.02 | 117.97 / 247.02 | 24.92 / 193.77 | 24.94 | 11.16 / 197.52 | 55.59 / 153.02 | 209.41 / 1980.03 |
| `fir_decimate` C32, 127 taps, M = 10 | 23.53 / 46.47 | 23.55 / 47.27 | 6.84 / 32.55 | 6.88 | 4.45 / 25.03 | 14.06 / 26.84 | 55.44 / 383.41 |
| `fir_interpolate` C32, L = 2, 128 taps | 131.97 / 262.02 | 131.67 / 265.02 | 39.59 / 211.77 | 39.67 | 25.24 / 215.52 | 65.59 / 172.51 | 236.37 / 2024.03 |
| `resample` C32, 3 / 2, 192 taps | 135.44 / 266.35 | 134.82 / 269.35 | 52.57 / 215.52 | 53.09 | 42.35 / 218.42 | 84.58 / 174.35 | 309.31 / 2026.03 |
| `fir_interpolate_real` f32, L = 2, 128 taps | 70.60 / 164.02 | 70.26 / 171.02 | 26.25 / 153.27 | 24.99 | 19.00 / 165.15 | 45.67 / 134.77 | 147.91 / 947.03 |
| `resample_real` f32, 3 / 2, 192 taps | 82.87 / 166.19 | 82.23 / 173.19 | 40.99 / 155.85 | 39.79 | 40.14 / 167.33 | 53.45 / 136.68 | 179.99 / 949.03 |
| `fir_interpolate_real` Q15, L = 2, 128 taps | 321.33 / 321.33 | 114.51 / 321.33 | 125.89 / 125.89 | 225.72 | 96.97 / 96.97 | 110.37 / 110.37 | 451.84 / 451.84 |
| `resample_real` Q15, 3 / 2, 192 taps | 423.58 / 423.58 | 111.86 / 423.58 | 225.31 / 371.06 | 224.46 | 186.78 / 329.03 | 192.59 / 297.56 | 598.41 / 875.74 |
| `fir` CQ15, 64 taps (fallback) | 530.46 / 530.46 | 187.33 / 530.46 | 214.08 / 214.08 | 237.75 | 153.06 / 153.06 | 227.65 / 211.54 | 841.39 / 934.94 |
| `fir_decimate` CQ15, 127 taps, M = 10 (fallback) | 118.11 / 118.11 | 35.72 / 118.11 | 68.42 / 68.42 | 50.15 | 33.82 / 53.59 | 50.04 / 80.46 | 185.86 / 204.82 |
| `fir_interpolate` CQ15, L = 2, 128 taps (fallback) | 593.52 / 593.52 | 201.70 / 593.50 | 268.38 / 268.38 | 318.91 | 205.58 / 205.58 | 178.71 / 259.40 | 638.42 / 638.42 |
| `resample` CQ15, 3 / 2, 192 taps (fallback) | 635.23 / 635.23 | 196.06 / 635.23 | 373.40 / 373.40 | 316.59 | 237.44 / 295.89 | 299.95 / 490.10 | 977.01 / 1075.61 |
| `mag_sq` C32 | 2.32 / 3.27 | 1.57 / 2.52 | 1.18 / 1.52 | 1.20 | 0.60 / 1.52 | 1.18 / 2.27 | 4.96 / 8.28 |
| `mag_sq` CQ15 | 1.31 / 3.51 | 1.31 / 1.89 | 0.73 / 1.01 | 0.75 | 0.36 / 0.44 | 1.02 / 1.89 | 3.02 / 7.53 |
| `quad_demod` C32 | 18.80 / 149.03 | 18.30 / 131.54 | 7.05 / 10.92 | 7.07 | 3.05 / 6.73 | 11.55 / 19.79 | 36.15 / 201.21 |
| `quad_demod` CQ15 | 33.15 / 170.85 | 24.65 / 171.16 | 10.65 / 162.72 | 10.60 | 4.03 / 163.66 | 16.65 / 173.61 | 40.79 / 332.25 |
| `quad_demod_fallback` C32 (GNU Radio) | 105.17 / 79.88 | 104.48 / 79.88 | 98.86 / 87.01 | - | 97.28 / 87.01 | 110.52 / 51.64 | 188.90 / 155.40 |
| `quad_demod_cq15_fallback` (CMSIS-DSP) | 170.85 / 266.90 | 171.16 / 263.04 | 162.72 / 253.85 | - | 163.66 / 252.52 | 173.61 / 246.64 | 332.25 / 603.82 |
| `fm_mod` C32 | 16.52 / 76.53 | 14.00 / 76.53 | 8.77 / 76.53 | 8.91 | 3.41 / 76.53 | 11.23 / 60.00 | 34.85 / 125.97 |
| `fm_mod` CQ15 | 18.72 / 40.82 | 11.47 / 41.34 | 8.98 / 21.78 | 9.12 | 2.94 / 15.90 | 9.58 / 37.73 | 48.23 / 67.27 |
| `fm_mod_fallback` C32 (GNU Radio) | 37.20 / 76.53 | 37.20 / 76.53 | 20.15 / 76.53 | - | 8.93 / 76.53 | 36.10 / 60.00 | 61.27 / 125.97 |
| `fm_mod_cq15_fallback` (CMSIS-DSP composition) | 40.82 / 68.01 | 41.34 / 68.01 | 21.78 / 68.01 | - | 15.90 / 68.01 | 37.73 / 54.01 | 67.27 / 116.02 |
| `squelch` C32 | 22.81 / 55.39 | 23.25 / 55.39 | 21.25 / 55.39 | 22.80 | 20.25 / 55.39 | 18.25 / 44.21 | 63.60 / 98.71 |
| `squelch` CQ15 | 27.03 / 27.03 | 27.03 / 27.03 | 25.53 / 25.53 | 27.03 | 25.53 / 25.53 | 21.02 / 21.02 | 67.05 / 67.05 |
| `mag` C32 | 2.57 / 3.52 | 1.82 / 2.77 | 1.04 / 1.64 | 1.06 | 0.66 / 1.27 | 1.46 / 2.02 | 6.86 / 8.53 |
| `mag` CQ15 | 5.09 / 21.27 | 5.09 / 11.02 | 2.25 / 2.64 | 2.26 | 1.22 / 2.64 | 3.26 / 4.52 | 9.03 / 18.28 |
| `window` C32 | 2.71 / 3.02 | 2.71 / 3.02 | 0.99 / 1.52 | 1.01 | 0.57 / 1.52 | 1.77 / 2.15 | 7.73 / 10.79 |
| `window` CQ15 | 3.36 / 7.77 | 2.11 / 5.27 | 0.76 / 2.08 | 0.78 | 0.39 / 1.27 | 0.95 / 2.89 | 3.95 / 9.91 |
| `power_db` f32 | 16.33 / 59.05 | 14.52 / 13.56 | 6.55 / 4.81 | 6.60 | 2.90 / 3.05 | 11.78 / 10.30 | 28.78 / 98.80 |
| `power_db` Q15 | 15.45 / 15.77 | 15.20 / 15.02 | 6.49 / 7.52 | 6.69 | 2.80 / 6.02 | 11.53 / 11.77 | 29.03 / 28.28 |
| `mag_sq_fallback` C32 (CMSIS-DSP) | 2.64 / 3.27 | 2.64 / 3.27 | 1.36 / 2.02 | - | 1.11 / 1.64 | 1.76 / 2.26 | 8.03 / 9.35 |
| `mag_sq_q15_fallback` (CMSIS-DSP) | 3.51 / 10.64 | 1.89 / 8.39 | 1.01 / 4.33 | - | 0.44 / 2.77 | 1.89 / 2.39 | 7.53 / 7.85 |
| `mag_fallback` C32 (CMSIS-DSP) | 3.26 / 4.27 | 3.26 / 4.27 | 1.76 / 2.40 | - | 1.23 / 1.89 | 2.01 / 3.01 | 8.28 / 10.91 |
| `mag_q15_fallback` (CMSIS-DSP) | 9.76 / 104.46 | 6.76 / 104.46 | 3.64 / 97.46 | - | 1.83 / 97.46 | 4.64 / 77.46 | 12.90 / 208.59 |
| `window_fallback` C32 (CMSIS-DSP) | 3.14 / 3.27 | 3.14 / 3.27 | 1.64 / 1.77 | - | 0.70 / 1.77 | 1.76 / 3.02 | 10.79 / 26.79 |
| `window_cq15_fallback` (CMSIS-DSP) | 7.77 / 8.53 | 5.27 / 5.53 | 2.08 / 2.78 | - | 1.27 / 2.15 | 2.89 / 3.52 | 9.91 / 14.93 |
| `power_db_fallback` f32 (CMSIS-DSP) | 15.27 / 51.65 | 14.76 / 51.65 | 7.39 / 43.00 | - | 5.89 / 43.00 | 15.02 / 53.39 | 28.28 / 77.68 |
| `power_db_q15_fallback` (CMSIS-DSP) | 15.77 / 53.54 | 15.02 / 53.29 | 7.52 / 43.83 | - | 6.02 / 43.83 | 11.77 / 55.65 | 28.28 / 86.83 |
| `agc` C32 | 23.79 / 24.03 | 23.04 / 24.03 | 17.39 / 24.03 | 17.41 | 16.12 / 24.03 | 22.42 / 17.03 | 64.61 / 64.05 |
| `agc` CQ15 | 70.53 / 70.53 | 70.53 / 70.53 | 67.11 / 67.11 | 70.53 | 67.06 / 67.06 | 53.50 / 53.50 | 137.07 / 137.07 |
| `agc_fallback` C32 (GNU Radio) | 28.24 / 24.03 | 28.24 / 24.03 | 23.36 / 24.03 | - | 23.36 / 24.03 | 22.42 / 17.03 | 64.11 / 64.05 |
| `dc_block_real` f32 | 13.66 / 11.51 | 13.03 / 11.51 | 6.61 / 8.76 | 6.81 | 5.49 / 8.77 | 7.40 / 16.01 | 26.81 / 27.54 |
| `dc_block_real` Q15 | 20.01 / 20.01 | 20.01 / 20.01 | 16.01 / 16.01 | 20.01 | 15.01 / 15.01 | 12.01 / 12.01 | 49.03 / 49.03 |
| `deemphasis` f32 | 12.03 / 22.01 | 12.40 / 20.01 | 6.41 / 13.01 | 6.49 | 5.24 / 14.76 | 6.65 / 14.01 | 24.81 / 27.53 |
| `deemphasis` Q15 | 12.51 / 12.51 | 12.51 / 12.51 | 13.52 / 13.52 | 12.51 | 12.77 / 12.77 | 11.01 / 11.01 | 42.03 / 42.03 |
| `dc_block` C32 | 16.28 / 14.01 | 14.78 / 11.51 | 9.16 / 8.77 | 9.80 | 5.35 / 7.76 | 11.03 / 11.01 | 26.53 / 26.53 |
| `dc_block` CQ15 | 29.01 / 29.01 | 29.01 / 29.01 | 37.01 / 37.01 | 29.01 | 26.02 / 26.02 | 23.01 / 23.01 | 53.03 / 53.03 |
| `dc_block_real_fallback` f32 (liquid-dsp) | 11.51 / 102.03 | 11.51 / 102.03 | 8.76 / 102.03 | - | 8.77 / 102.03 | 16.01 / 90.04 | 27.54 / 240.06 |
| `deemphasis_fallback` f32 (GNU Radio) | 22.01 / 60.03 | 20.01 / 60.03 | 13.01 / 60.03 | - | 14.76 / 60.03 | 14.01 / 45.03 | 27.53 / 150.07 |
| `dc_block_fallback` C32 (liquid-dsp) | 14.01 / 121.03 | 11.51 / 121.03 | 8.77 / 121.03 | - | 7.76 / 121.03 | 11.01 / 81.04 | 26.53 / 277.07 |
| `biquad` C32, 2 sections (KFR) | 17.43 / 81.59 | 17.37 / 76.59 | 13.17 / 53.61 | 13.18 | 13.23 / 72.00 | 12.61 / 46.63 | 43.04 / 96.78 |
| `biquad` CQ15, 2 sections (fallback) | 98.08 / 98.08 | 50.56 / 100.07 | 18.30 / 104.07 | 18.07 | 17.73 / 99.08 | 25.16 / 58.06 | 70.53 / 203.11 |
| `biquad_real` f32, 2 sections (KFR) | 18.19 / 40.80 | 18.17 / 38.30 | 14.04 / 26.81 | 14.04 | 13.04 / 36.00 | 13.66 / 23.32 | 38.93 / 48.39 |
| `biquad_real` Q15, 2 sections (fallback) | 46.05 / 46.05 | 21.25 / 46.05 | 18.56 / 46.05 | 17.88 | 17.87 / 45.05 | 11.87 / 26.05 | 41.21 / 96.08 |
| `biquad_fallback` C32 (CMSIS-DSP) | 31.97 / 57.54 | 32.10 / 57.54 | 23.91 / 45.04 | - | 23.78 / 45.04 | 21.71 / 41.53 | 68.47 / 118.83 |
| `biquad_cq15_fallback` (CMSIS-DSP) | 100.07 / 152.09 | 100.07 / 152.09 | 104.07 / 144.09 | - | 99.08 / 140.09 | 58.06 / 100.06 | 203.11 / 272.13 |
| `biquad_real_fallback` f32 (CMSIS-DSP) | 28.78 / 29.04 | 28.33 / 29.04 | 21.38 / 22.54 | - | 19.95 / 22.54 | 21.76 / 22.53 | 50.79 / 57.94 |
| `biquad_real_q15_fallback` (CMSIS-DSP) | 46.05 / 76.04 | 46.05 / 76.04 | 46.05 / 72.04 | - | 45.05 / 70.04 | 26.05 / 50.03 | 96.08 / 136.07 |
| `fir_fallback` C32 (CMSIS-DSP) | 190.82 / 442.96 | 190.82 / 442.96 | 117.56 / 149.90 | - | 116.43 / 150.14 | 153.17 / 301.63 | 509.86 / 676.70 |
| `fir_cq15_fallback` (CMSIS-DSP) | 530.46 / 832.16 | 530.46 / 830.16 | 214.08 / 830.39 | - | 153.06 / 829.89 | 211.54 / 485.63 | 934.94 / 785.69 |
| `fir_decimate_fallback` C32 (CMSIS-DSP) | 59.38 / 87.72 | 59.38 / 87.63 | 58.23 / 58.96 | - | 56.15 / 59.00 | 57.32 / 91.08 | 147.83 / 138.12 |
| `fir_decimate_cq15_fallback` (CMSIS-DSP) | 118.11 / 86.86 | 118.11 / 74.97 | 68.42 / 51.02 | - | 53.59 / 50.02 | 80.46 / 38.56 | 204.82 / 321.21 |
| `fir_interpolate_fallback` C32 (CMSIS-DSP) | 258.72 / 454.71 | 258.72 / 454.71 | 153.84 / 376.18 | - | 151.82 / 375.18 | 187.97 / 361.16 | 603.69 / 592.97 |
| `fir_interpolate_cq15_fallback` (CMSIS-DSP) | 593.52 / 500.74 | 593.50 / 500.74 | 268.38 / 508.82 | - | 205.58 / 508.32 | 259.40 / 256.40 | 638.42 / 1183.72 |
| `resample_fallback` C32 (CMSIS-DSP interpolator) | 328.83 / 451.37 | 328.83 / 451.37 | 325.03 / 372.76 | - | 315.57 / 371.92 | 316.65 / 359.12 | 781.66 / 589.97 |
| `resample_cq15_fallback` (CMSIS-DSP interpolator) | 635.23 / 497.80 | 635.23 / 497.80 | 373.40 / 506.00 | - | 295.89 / 505.50 | 490.10 / 253.10 | 1075.61 / 1180.20 |
| `fir_interpolate_real_fallback` f32 (CMSIS-DSP) | 150.72 / 227.36 | 150.71 / 227.36 | 98.27 / 188.10 | - | 98.27 / 187.60 | 113.76 / 180.58 | 351.14 / 296.50 |
| `fir_interpolate_real_q15_fallback` (CMSIS-DSP) | 321.33 / 250.38 | 321.33 / 250.38 | 125.89 / 254.42 | - | 96.98 / 254.16 | 110.37 / 128.20 | 451.84 / 591.87 |
| `resample_real_fallback` f32 (CMSIS-DSP interpolator) | 324.45 / 225.69 | 277.76 / 225.69 | 306.39 / 186.38 | - | 175.93 / 185.97 | 230.41 / 179.57 | 687.91 / 295.00 |
| `resample_real_q15_fallback` (CMSIS-DSP interpolator) | 423.58 / 248.90 | 423.58 / 248.90 | 371.06 / 253.00 | - | 329.03 / 252.75 | 297.56 / 126.55 | 875.74 / 590.11 |
| `dot` C32, 4096 samples | 2.72 / 2.54 | 2.72 / 2.54 | 0.75 / 1.17 | 0.77 | 0.56 / 0.61 | 1.59 / 1.53 | 7.20 / 29.55 |
| `dot` CQ15, 4096 samples | 3.24 / 9.30 | 2.97 / 6.80 | 1.59 / 2.92 | 1.62 | 0.78 / 2.80 | 2.04 / 2.04 | 6.74 / 17.56 |
| `dot_real` f32, 4096 samples | 1.12 / 1.03 | 1.12 / 1.28 | 0.33 / 0.66 | 0.34 | 0.20 / 0.35 | 0.62 / 0.90 | 3.35 / 6.17 |
| `dot_real` Q15, 4096 samples | 1.40 / 3.88 | 1.39 / 2.79 | 0.70 / 1.43 | 0.71 | 0.64 / 1.30 | 0.72 / 1.16 | 3.23 / 11.55 |
| `scale` C32 | 1.70 / 3.02 | 1.70 / 3.02 | 0.61 / 1.27 | 0.62 | 0.29 / 1.27 | 1.26 / 2.02 | 5.65 / 6.16 |
| `scale` CQ15 | 7.27 / 7.27 | 0.86 / 3.39 | 0.31 / 1.64 | 0.33 | 0.16 / 0.95 | 0.64 / 2.76 | 2.81 / 8.15 |
| `dot_fallback` C32 (CMSIS-DSP) | 6.02 / 7.54 | 5.77 / 8.54 | 4.40 / 5.04 | - | 3.40 / 5.04 | 5.02 / 6.53 | 16.55 / 25.57 |
| `dot_cq15_fallback` (CMSIS-DSP) | 9.30 / 6.30 | 6.80 / 5.30 | 2.92 / 2.68 | - | 2.80 / 2.55 | 2.04 / 2.79 | 17.56 / 25.58 |
| `dot_real_fallback` f32 (CMSIS-DSP) | 3.77 / 3.77 | 3.77 / 4.27 | 2.40 / 2.52 | - | 2.40 / 2.52 | 2.52 / 3.27 | 6.17 / 12.79 |
| `dot_real_q15_fallback` (CMSIS-DSP) | 3.88 / 3.15 | 2.79 / 2.65 | 1.43 / 1.34 | - | 1.30 / 1.28 | 1.16 / 1.40 | 11.55 / 12.79 |
| `scale_fallback` C32 (CMSIS-DSP) | 3.01 / 3.01 | 2.26 / 3.01 | 0.70 / 1.27 | - | 0.45 / 1.27 | 1.26 / 3.01 | 6.16 / 5.41 |
| `scale_cq15_fallback` (CMSIS-DSP) | 7.27 / 10.52 | 3.39 / 5.52 | 1.64 / 2.64 | - | 0.95 / 2.15 | 2.76 / 3.51 | 8.15 / 8.28 |
| `add` C32 (VOLK) | 2.40 / 3.02 | 2.40 / 3.02 | 0.90 / 1.52 | 0.97 | 0.44 / 1.52 | 1.59 / 2.77 | 7.19 / 8.79 |
| `add` CQ15 (VOLK `volk_16i_x2_add_saturated_16i`) | 1.11 / 1.52 | 1.11 / 1.52 | 0.44 / 0.77 | 0.44 | 0.22 / 0.77 | 0.89 / 1.40 | 2.33 / 10.78 |
| `add_real` f32 (VOLK) | 1.21 / 1.52 | 1.21 / 1.52 | 0.46 / 0.77 | 0.50 | 0.23 / 0.40 | 0.80 / 1.40 | 3.63 / 6.04 |
| `add_real` Q15 (VOLK) | 0.56 / 0.77 | 0.56 / 0.77 | 0.23 / 0.40 | 0.23 | 0.12 / 0.40 | 0.45 / 0.71 | 1.19 / 5.41 |
| `mul` C32 (VOLK; fused at `avx2`, `avx512`, `neon` as ours) | 6.27 / 23.02 | 4.72 / 6.02 | 2.12 / 2.77 | 2.13 | 1.06 / 2.77 | 2.59 / 3.52 | 16.79 / 56.03 |
| `mul` CQ15 (our fallback) | 11.52 / 11.52 | 5.36 / 8.77 | 2.32 / 3.64 | 2.32 | 2.20 / 1.89 | 2.55 / 3.14 | 15.17 / 15.17 |
| `mul_real` f32 (VOLK) | 1.21 / 1.52 | 1.21 / 1.52 | 0.46 / 0.77 | 0.51 | 0.23 / 0.40 | 0.80 / 1.40 | 3.61 / 6.04 |
| `mul_real` Q15 (our fallback) | 3.64 / 3.64 | 0.94 / 2.14 | 0.35 / 1.14 | 0.36 | 0.19 / 0.55 | 0.45 / 1.64 | 1.83 / 3.77 |
| `interleave` f32 (VOLK) | 2.28 / 2.52 | 2.28 / 2.52 | 1.07 / 1.40 | 1.08 | 0.48 / 1.40 | 0.89 / 1.77 | 5.89 / 7.79 |
| `interleave` CQ15 (our fallback) | 1.28 / 1.21 | 1.28 / 1.21 | 0.55 / 0.61 | 0.56 | 0.25 / 0.36 | 0.46 / 0.77 | 2.97 / 3.92 |
| `deinterleave` C32 (VOLK) | 2.28 / 2.52 | 2.28 / 2.52 | 1.07 / 1.40 | 1.08 | 0.48 / 1.02 | 0.90 / 1.15 | 5.89 / 7.79 |
| `deinterleave` CQ15 (VOLK) | 1.82 / 3.15 | 1.20 / 1.90 | 0.67 / 0.96 | 0.68 | 0.26 / 0.96 | 0.46 / 0.58 | 2.97 / 3.92 |
| `oscillate` C32 (our fallback) | 9.02 / 25.01 | 8.27 / 25.01 | 4.65 / 11.15 | 4.65 | 1.83 / 3.89 | 6.27 / 21.01 | 22.04 / 28.28 |
| `oscillate` CQ15 (our fallback) | 10.27 / 27.01 | 7.27 / 28.01 | 5.03 / 14.03 | 5.03 | 1.43 / 9.08 | 6.02 / 23.01 | 17.54 / 33.66 |
| `add_fallback` C32 (CMSIS-DSP) | 3.52 / 3.52 | 3.52 / 3.52 | 1.14 / 1.53 | - | 0.77 / 1.53 | 1.51 / 3.52 | 7.79 / 31.54 |
| `add_cq15_fallback` (CMSIS-DSP) | 1.77 / 11.78 | 1.77 / 5.78 | 0.58 / 2.78 | - | 0.39 / 2.40 | 0.77 / 3.27 | 2.33 / 3.62 |
| `add_real_fallback` f32 (CMSIS-DSP) | 1.39 / 1.77 | 1.39 / 1.77 | 0.58 / 0.77 | - | 0.58 / 0.77 | 1.26 / 1.77 | 6.04 / 15.79 |
| `add_real_q15_fallback` (CMSIS-DSP) | 0.71 / 5.90 | 0.71 / 2.90 | 0.30 / 1.40 | - | 0.30 / 1.22 | 0.64 / 1.65 | 1.19 / 1.83 |
| `mul_fallback` C32 (CMSIS-DSP) | 6.27 / 8.27 | 4.89 / 5.27 | 2.46 / 3.02 | - | 2.39 / 3.02 | 2.76 / 5.77 | 15.29 / 18.62 |
| `mul_cq15_fallback` (CMSIS-DSP, truncating Q3.13 product) | 11.52 / 12.78 | 8.77 / 11.28 | 3.64 / 4.53 | - | 1.89 / 3.66 | 3.14 / 5.52 | 15.17 / 20.37 |
| `mul_real_fallback` f32 (CMSIS-DSP) | 1.39 / 1.77 | 1.39 / 1.77 | 0.58 / 0.78 | - | 0.58 / 0.77 | 1.26 / 1.77 | 6.04 / 15.79 |
| `mul_real_q15_fallback` (CMSIS-DSP, truncating) | 3.64 / 4.15 | 2.14 / 2.65 | 1.14 / 1.28 | - | 0.55 / 1.03 | 1.64 / 1.65 | 3.77 / 3.93 |
| `oscillate_fallback` C32 (CMSIS-DSP `arm_sin_cos_f32`) | 25.01 / 101.51 | 25.01 / 97.51 | 11.15 / 87.51 | - | 3.89 / 87.51 | 21.01 / 83.51 | 28.28 / 175.02 |
| `oscillate_cq15_fallback` (CMSIS-DSP `arm_cos_q15`, `arm_sin_q15`) | 27.01 / 59.01 | 28.01 / 59.01 | 14.03 / 59.01 | - | 9.08 / 59.01 | 23.01 / 53.01 | 33.66 / 105.01 |
| `interleave_fallback` f32 (VOLK `generic`) | 2.40 / 2.52 | 2.40 / 2.52 | 1.20 / 1.40 | - | 0.71 / 1.27 | 1.52 / 1.52 | 7.79 / 7.79 |
| `interleave_cq15_fallback` (VOLK's `generic` interleave loop on `int16`) | 1.21 / 1.27 | 1.21 / 1.27 | 0.61 / 0.71 | - | 0.36 / 0.65 | 0.77 / 0.77 | 3.92 / 3.92 |
| `deinterleave_fallback` C32 (VOLK `generic`) | 2.14 / 2.77 | 2.14 / 2.77 | 1.20 / 1.40 | - | 0.95 / 1.15 | 1.26 / 1.52 | 7.79 / 7.80 |
| `deinterleave_cq15_fallback` (VOLK `generic`) | 2.15 / 2.90 | 1.90 / 2.15 | 0.86 / 0.96 | - | 0.36 / 0.59 | 0.64 / 0.77 | 3.91 / 3.92 |
| `fft` C32, N = 64 | 14.95 / 18.88 | 14.95 / 25.30 | 5.84 / 5.88 | 6.28 | 4.80 / 5.53 | 10.42 / 18.28 | 35.50 / 63.23 |
| `fft` C32, N = 1024 | 25.15 / 25.56 | 25.15 / 33.81 | 9.66 / 11.33 | 9.61 | 5.56 / 11.07 | 15.14 / 25.85 | 58.57 / 91.17 |
| `fft` C32, N = 4096 | 28.93 / 31.17 | 28.93 / 39.44 | 10.91 / 13.27 | 10.86 | 5.91 / 11.81 | 17.37 / 30.85 | 67.64 / 105.57 |
| `fft` CQ15, N = 1024, block float | 28.60 / 233.83 | 22.43 / 233.83 | 10.19 / 233.83 | 10.24 | 5.80 / 233.83 | 15.79 / 179.33 | 48.92 / 460.87 |
| `fft` CQ15, N = 1024, fixed scaling | 25.29 / 233.83 | 19.92 / 233.83 | 9.11 / 233.83 | 9.18 | 5.29 / 233.83 | 12.79 / 179.33 | 43.57 / 460.87 |
| `fft` CQ15, N = 1024, fixed scaling, full scale | 26.99 / 233.83 | 21.04 / 233.83 | 9.61 / 233.83 | 9.65 | 5.44 / 233.83 | 13.83 / 179.33 | 46.17 / 460.87 |
| `rfft` f32, N = 1024 | 16.40 / 15.32 | 16.40 / 15.32 | 8.15 / 13.54 | 8.19 | 5.09 / 12.82 | 10.43 / 18.76 | 41.91 / 59.40 |
| `fft_64_fallback` C32 (CMSIS-DSP) | 72.92 / 68.94 | 71.47 / 68.97 | 54.88 / 60.61 | - | 48.95 / 58.08 | 44.09 / 54.44 | 121.02 / 117.59 |
| `fft_1024_fallback` C32 (CMSIS-DSP) | 115.78 / 112.68 | 114.29 / 111.68 | 84.06 / 95.78 | - | 75.22 / 91.86 | 75.65 / 88.38 | 220.35 / 209.86 |
| `fft_4096_fallback` C32 (CMSIS-DSP) | 139.65 / 128.68 | 138.16 / 128.68 | 100.96 / 109.61 | - | 90.84 / 104.62 | 87.67 / 99.62 | 244.92 / 221.28 |
| `fft_cq15_1024_fallback` block float (CMSIS-DSP) | 337.03 / 217.26 | 278.28 / 217.26 | 236.69 / 199.25 | - | 221.86 / 198.93 | 258.90 / 140.66 | 537.22 / 378.19 |
| `fft_cq15_1024_default_fallback` fixed scaling (CMSIS-DSP) | 210.45 / 217.26 | 156.72 / 217.26 | 140.55 / 199.25 | - | 154.80 / 198.93 | 135.19 / 140.66 | 405.94 / 378.19 |
| `rfft_1024_fallback` f32 (CMSIS-DSP) | 63.75 / 58.10 | 63.00 / 56.85 | 44.26 / 45.68 | - | 39.43 / 43.85 | 36.11 / 41.49 | 105.27 / 98.59 |
| `halfband_decimate` C32 (liquid-dsp `resamp2_crcf`) | 28.20 / 121.57 | 28.57 / 122.07 | 12.77 / 104.70 | 12.74 | 7.71 / 99.41 | 20.56 / 134.25 | 92.52 / 440.36 |
| `halfband_decimate_real` f32 (liquid-dsp `resamp2_rrrf`) | 13.53 / 94.91 | 13.53 / 111.91 | 6.21 / 80.56 | 6.15 | 4.68 / 96.22 | 10.72 / 115.23 | 42.94 / 263.59 |
| `real_to_complex` C32 (liquid-dsp `firhilbf`) | 21.63 / 99.08 | 21.37 / 115.58 | 10.82 / 84.33 | 10.95 | 7.57 / 99.99 | 16.47 / 117.39 | 52.87 / 274.99 |
| `real_to_complex_airspy` C32 (libairspy `iqconverter_float`, SSE2, second gate) | 21.63 / 36.15 | - | - | - | - | - | - |
| `halfband_decimate` CQ15 (our fallback) | 59.10 / 174.36 | 47.29 / 78.11 | 34.73 / 57.31 | 34.73 | 26.65 / 46.57 | 35.98 / 52.24 | 120.65 / 134.25 |
| `halfband_decimate_real` Q15 (our fallback) | 23.79 / 94.62 | 19.18 / 61.31 | 13.98 / 51.16 | 13.99 | 9.54 / 45.93 | 14.49 / 42.16 | 54.98 / 100.19 |
| `real_to_complex` CQ15 (libairspy `iqconverter_int16`, provisional) | 34.13 / 67.33 | 27.14 / 45.51 | 18.63 / 34.31 | 18.67 | 12.68 / 34.96 | 21.53 / 35.45 | 67.97 / 126.54 |
| `halfband_decimate_fallback` C32 (liquid-dsp portable) | 65.41 / 209.57 | 66.31 / 209.57 | 45.14 / 209.19 | - | 43.37 / 170.91 | 48.61 / 139.25 | 122.90 / 440.36 |
| `halfband_decimate_real_fallback` f32 (liquid-dsp portable) | 50.32 / 94.91 | 50.32 / 94.91 | 41.87 / 80.56 | - | 41.58 / 79.22 | 40.70 / 92.23 | 86.29 / 263.60 |
| `real_to_complex_fallback` C32 (liquid-dsp portable) | 77.74 / 99.08 | 72.68 / 98.58 | 64.02 / 84.33 | - | 62.19 / 82.99 | 65.23 / 94.39 | 103.61 / 274.99 |
| `halfband_decimate_cq15_fallback` (CMSIS-DSP composition) | 174.36 / 185.99 | 78.11 / 178.05 | 57.31 / 174.69 | - | 46.57 / 173.46 | 52.24 / 104.81 | 134.25 / 188.86 |
| `halfband_decimate_real_q15_fallback` (CMSIS-DSP composition) | 94.62 / 92.92 | 61.31 / 88.98 | 51.16 / 87.33 | - | 45.93 / 86.84 | 42.16 / 52.29 | 100.19 / 94.37 |
| `real_to_complex_cq15_fallback` (libairspy `iqconverter_int16`, provisional) | 117.31 / 67.33 | 80.36 / 45.51 | 68.31 / 34.31 | - | 59.35 / 34.96 | 63.61 / 35.45 | 120.47 / 126.54 |
| `stereo_matrix` C32 (VOLK composition) | 3.90 / 9.59 | 3.89 / 9.59 | 1.58 / 5.09 | 1.59 | 0.77 / 3.59 | 2.01 / 7.09 | 11.53 / 33.63 |
| `stereo_matrix` CQ15 (our fallback) | 2.09 / 4.26 | 2.08 / 3.26 | 0.86 / 1.33 | 0.87 | 0.43 / 0.86 | 1.14 / 1.14 | 5.40 / 5.40 |
| `stereo_matrix_fallback` C32 (CMSIS-DSP planar) | 6.51 / 5.31 | 5.76 / 5.31 | 2.08 / 2.32 | - | 1.33 / 2.32 | 2.01 / 5.30 | 10.78 / 47.34 |
| `stereo_matrix_cq15_fallback` (CMSIS-DSP planar) | 4.26 / 17.70 | 3.26 / 8.70 | 1.33 / 4.20 | - | 0.86 / 3.64 | 1.14 / 4.93 | 5.40 / 5.47 |
| `stereo_pll_fallback` f32 (liquid-dsp `nco_crcf`) | 90.16 / 67.94 | 88.16 / 67.94 | 78.38 / 67.94 | - | 73.38 / 67.94 | 68.78 / 59.92 | 190.06 / 143.86 |
| `stereo_pll_q15_fallback` (CMSIS-DSP oscillator) | 85.03 / 135.02 | 85.03 / 135.02 | 85.03 / 135.02 | - | 85.03 / 135.02 | 58.02 / 113.02 | 183.05 / 228.04 |
| `preemph_limit` f32 (our fallback) | 44.30 / 44.30 | 42.30 / 42.30 | 33.30 / 33.30 | 44.30 | 33.98 / 33.98 | 41.63 / 41.63 | 93.56 / 93.56 |
| `preemph_limit_q15` (our fallback) | 94.87 / 94.87 | 94.87 / 94.87 | 94.50 / 94.50 | 94.87 | 94.50 / 94.50 | 65.96 / 65.96 | 448.58 / 448.58 |
| `noise_blank` C32 (our fallback) | 34.35 / 34.35 | 33.35 / 33.35 | 27.78 / 27.78 | 33.72 | 27.37 / 27.37 | 22.98 / 22.98 | 77.93 / 77.93 |
| `noise_blank_cq15` (our fallback) | 30.71 / 30.71 | 30.71 / 30.71 | 30.33 / 30.33 | 30.36 | 29.97 / 29.97 | 20.89 / 20.89 | 74.15 / 74.15 |
| `auto_notch` f32 (our fallback) | 689.89 / 689.89 | 689.89 / 689.89 | 565.09 / 565.09 | 689.89 | 571.92 / 571.92 | 480.16 / 480.16 | 1700.29 / 1700.29 |
| `am_mod` C32 (our fallback) | 4.52 / 4.52 | 4.52 / 4.52 | 2.02 / 2.02 | 2.03 | 1.24 / 1.24 | 3.52 / 3.52 | 10.54 / 10.54 |
| `am_mod_cq15` (our fallback) | 7.78 / 7.78 | 1.40 / 3.15 | 0.65 / 1.59 | 0.66 | 0.59 / 0.76 | 1.27 / 2.90 | 3.92 / 8.80 |
| `am_mod_fallback` C32 (liquid-dsp `ampmodem`) | 4.52 / 2.78 | 4.52 / 2.78 | 2.02 / 2.78 | - | 1.24 / 2.78 | 3.52 / 1.77 | 10.54 / 46.02 |
| `am_mod_cq15_fallback` (CMSIS-DSP composition) | 7.78 / 16.42 | 3.15 / 13.66 | 1.59 / 11.65 | - | 0.76 / 11.28 | 2.90 / 10.35 | 8.80 / 31.69 |
| `preemph_limit_stereo` C32 (our fallback) | 73.20 / 73.20 | 71.20 / 71.20 | 56.20 / 56.20 | 73.20 | 55.21 / 55.21 | 59.64 / 59.64 | 126.84 / 126.84 |
| `stereo_encode` C32 to f32 (our fallback) | 43.01 / 43.01 | 43.01 / 43.01 | 17.15 / 17.15 | 43.01 | 6.02 / 6.02 | 35.01 / 35.01 | 37.54 / 37.54 |
| `crc16_ccitt_false` (GNU Radio `gr::digital::crc`, per byte) | 8.13 / 10.17 | 8.13 / 10.17 | 6.88 / 10.20 | 8.13 | 6.88 / 10.20 | 7.07 / 9.13 | 21.79 / 26.91 |
| `crc16_x25` (GNU Radio `gr::digital::crc`, per byte) | 7.63 / 8.18 | 7.63 / 8.18 | 6.88 / 8.21 | 7.63 | 6.51 / 8.21 | 7.07 / 7.14 | 19.79 / 22.88 |
| `crc32` (GNU Radio `gr::digital::crc`, per byte) | 6.63 / 8.18 | 6.63 / 8.18 | 5.88 / 8.21 | 6.63 | 5.50 / 8.21 | 7.07 / 7.15 | 19.79 / 22.89 |
| `crc24_adsb` (GNU Radio `gr::digital::crc`, per byte) | 7.63 / 10.17 | 7.63 / 10.17 | 6.88 / 10.20 | 7.63 | 6.88 / 10.20 | 8.06 / 9.13 | 23.79 / 26.91 |
| `frame_decode` (our code, per bit) | 25.28 / 25.28 | 25.28 / 25.28 | 24.85 / 24.85 | 25.07 | 23.79 / 23.79 | 20.23 / 20.23 | 58.70 / 58.70 |
| `frame_encode_block` (our code, per output sample) | 3.08 / 3.08 | 3.08 / 3.08 | 3.11 / 3.11 | 3.08 | 2.98 / 2.98 | 2.63 / 2.63 | 8.97 / 8.97 |
| `frame_decode_block` (our code, per input sample) | 23.28 / 23.28 | 23.28 / 23.28 | 22.86 / 22.86 | 23.07 | 22.79 / 22.79 | 20.23 / 20.23 | 53.71 / 53.71 |
| `symbol_sync` C32 (our fallback) | 69.30 / 69.30 | 65.89 / 65.89 | 60.28 / 60.28 | 69.30 | 56.62 / 56.62 | 46.62 / 46.62 | 138.13 / 138.13 |
| `symbol_sync_real` f32 (our fallback) | 34.16 / 34.16 | 32.79 / 32.79 | 28.66 / 28.66 | 34.16 | 27.29 / 27.29 | 23.41 / 23.41 | 64.45 / 64.45 |
| `costas` C32 (our fallback) | 116.03 / 116.03 | 112.03 / 112.03 | 96.03 / 96.03 | 116.03 | 95.03 / 95.03 | 89.03 / 89.03 | 228.05 / 228.05 |
| `costas_fallback` C32 (GNU Radio `costas_loop_cc`) | 116.03 / 151.96 | 112.03 / 151.96 | 96.03 / 137.67 | - | 95.03 / 137.67 | 89.03 / 105.13 | 228.05 / 255.33 |
| `rds_demod_block` (our code, per input sample) | 138.05 / 138.05 | 137.86 / 137.86 | 46.46 / 46.46 | 48.05 | 41.09 / 41.09 | 78.82 / 78.82 | 340.30 / 340.30 |
| `rds_decode_block` (our code, per input sample) | 39.08 / 39.08 | 38.42 / 38.42 | 37.88 / 37.88 | 38.77 | 37.83 / 37.83 | 32.85 / 32.85 | 84.97 / 84.97 |
| `rds_shape` f32 (our fallback) | 113.16 / 113.16 | 110.16 / 110.16 | 79.16 / 79.16 | 113.16 | 72.16 / 72.16 | 82.13 / 82.13 | 168.41 / 169.41 |
| `stereo_encode_rds` C32 and f32 to f32 (our fallback) | 70.03 / 70.03 | 60.03 / 60.03 | 24.17 / 24.17 | 70.03 | 8.28 / 8.28 | 49.02 / 49.02 | 53.05 / 53.05 |
| `morse_encode_block` (our code, per output sample) | 20.17 / 20.17 | 20.17 / 20.17 | 19.97 / 19.97 | 20.17 | 18.37 / 18.37 | 19.46 / 19.46 | 49.57 / 49.57 |
| `morse_decode_block` (our code, per input sample) | 73.68 / 73.68 | 72.93 / 72.93 | 69.08 / 69.08 | 73.68 | 68.17 / 68.17 | 62.86 / 62.86 | 149.15 / 149.15 |
| `dot_complex` C32 (VOLK `volk_32fc_x2_dot_prod_32fc`) | 5.29 / 8.53 | 5.29 / 6.03 | 2.35 / 3.02 | 2.36 | 2.35 / 3.02 | 3.53 / 4.03 | 13.20 / 40.55 |
| `dot_complex_fallback` C32 (CMSIS-DSP `arm_cmplx_dot_prod_f32`) | 14.41 / 15.02 | 8.66 / 15.02 | 8.41 / 13.02 | - | 6.41 / 13.02 | 9.03 / 11.52 | 13.20 / 35.30 |
| `equalize` C32 (our fallback) | 502.98 / 502.98 | 496.45 / 496.45 | 533.11 / 533.11 | 502.21 | 492.09 / 492.09 | 437.31 / 437.31 | 1027.31 / 1027.31 |
| `equalize_fallback` C32 (GNU Radio `linear_equalizer`) | 502.98 / 722.54 | 496.45 / 722.54 | 533.11 / 729.03 | - | 492.09 / 729.03 | 437.31 / 493.03 | 1027.31 / 1900.06 |
| `channelize` C32 (liquid-dsp `firpfbch2_crcf`, SIMD dot products and FFTW) | 93.20 / 337.96 | 94.69 / 339.96 | 50.42 / 326.64 | 50.03 | 56.63 / 334.08 | 43.43 / 361.58 | 270.93 / 949.71 |
| `channelize_fallback` C32 (liquid-dsp `firpfbch2_crcf`, portable) | 192.43 / 431.21 | 189.18 / 429.58 | 145.84 / 412.39 | - | 143.76 / 398.71 | 151.01 / 297.08 | 436.07 / 949.71 |
| `channelize_block` (our code, per input sample) | 94.70 / 94.70 | 94.67 / 94.67 | 51.04 / 51.04 | 50.65 | 57.12 / 57.12 | 43.68 / 43.68 | 270.93 / 270.93 |
| `channel_select_block` (our code, per output sample) | 3.02 / 3.02 | 3.02 / 3.02 | 2.30 / 2.30 | 3.02 | 1.68 / 1.68 | 2.27 / 2.27 | 10.54 / 10.54 |
| `channel_bank_channelize` (our code, per input sample) | 193.80 / 193.80 | 193.79 / 193.79 | 83.51 / 83.51 | 83.64 | 104.25 / 104.25 | 81.92 / 81.92 | 534.31 / 534.31 |
| `channel_bank_ddc` (our code, per input sample) | 721.18 / 721.18 | 721.77 / 721.77 | 234.33 / 234.33 | 241.94 | 178.06 / 178.06 | 436.60 / 436.60 | 1657.26 / 1657.26 |
| `band_pulse_detect_block` (our code, per lane value) | 121.36 / 121.36 | 120.05 / 120.05 | 112.86 / 112.86 | 119.99 | 112.26 / 112.26 | 110.02 / 110.02 | 266.49 / 266.49 |

Accuracy (SNR in dB against the f64 reference), ours / comparison:

| Kernel | sse2 | sse4.2 | avx2 | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | exact | exact | exact | exact | exact | exact |
| `convert` CS16 to C32 | exact | exact | exact | exact | exact | exact |
| `convert` CU8 to C32 | exact | exact | exact | exact | exact | exact |
| `convert_real` RS16 to f32 | exact | exact | exact | exact | exact | exact |
| `convert` CS8 to CQ15 | exact | exact | exact | exact | exact | exact |
| `convert` CU8 to CQ15 | exact | exact | exact | exact | exact | exact |
| `convert_out` C32 to CS16 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 |
| `convert_out` C32 to CS8 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 |
| `convert_out` C32 to CU8 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 |
| `convert_out_real` f32 to RS16 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 |
| `convert_out_real` f32 to RS8 | 47.2 / 47.2 | 47.2 / 47.2 | 47.2 / 47.2 | 47.2 / 47.2 | 47.2 / 47.2 | 47.2 / 47.2 |
| `convert_out_real` f32 to RU8 | 47.2 / 47.2 | 47.2 / 47.2 | 47.2 / 47.2 | 47.2 / 47.2 | 47.2 / 47.2 | 47.2 / 47.2 |
| `convert_cs8_fallback` (CMSIS-DSP) | exact | exact | exact | exact | exact | exact |
| `convert_cs16_fallback` (CMSIS-DSP) | exact | exact | exact | exact | exact | exact |
| `convert_cu8_fallback` (CMSIS-DSP) | exact | exact | exact | exact | exact | exact |
| `convert_real_rs16_fallback` (CMSIS-DSP) | exact | exact | exact | exact | exact | exact |
| `convert_cq15_cs8_fallback` (CMSIS-DSP) | exact | exact | exact | exact | exact | exact |
| `convert_cq15_cu8_fallback` (CMSIS-DSP) | exact | exact | exact | exact | exact | exact |
| `convert_out_cs16_fallback` (CMSIS-DSP) | 95.5 / 89.4 | 95.5 / 89.4 | 95.5 / 89.4 | 95.5 / 89.4 | 95.5 / 89.4 | 95.5 / 89.4 |
| `convert_out_cs8_fallback` (CMSIS-DSP) | 47.3 / 41.3 | 47.3 / 41.3 | 47.3 / 41.3 | 47.3 / 41.3 | 47.3 / 41.3 | 47.3 / 41.3 |
| `convert_out_cu8_fallback` (CMSIS-DSP) | 47.3 / 41.2 | 47.3 / 41.2 | 47.3 / 41.2 | 47.3 / 41.2 | 47.3 / 41.2 | 47.3 / 41.2 |
| `convert_out_real_rs16_fallback` (CMSIS-DSP) | 95.5 / 89.5 | 95.5 / 89.5 | 95.5 / 89.5 | 95.5 / 89.5 | 95.5 / 89.5 | 95.5 / 89.5 |
| `convert_out_real_rs8_fallback` (CMSIS-DSP) | 47.2 / 41.2 | 47.2 / 41.2 | 47.2 / 41.2 | 47.2 / 41.2 | 47.2 / 41.2 | 47.2 / 41.2 |
| `convert_out_real_ru8_fallback` (CMSIS-DSP) | 47.2 / 41.3 | 47.2 / 41.3 | 47.2 / 41.3 | 47.2 / 41.3 | 47.2 / 41.3 | 47.2 / 41.3 |
| `rotate` C32 | 143.0 / 107.8 | 143.0 / 107.8 | 144.3 / 115.0 | 144.0 / 114.5 | 143.9 / 115.2 | 143.0 / 107.8 |
| `rotate` CQ15 | 91.2 / 91.2 | 91.2 / 91.2 | 91.2 / 91.2 | 91.2 / 91.2 | 91.2 / 91.2 | 91.2 / 91.2 |
| `rotate_fallback` C32 (GNU Radio) | 133.0 / 107.8 | 133.0 / 107.8 | 133.0 / 107.8 | 133.0 / 107.8 | 133.0 / 108.8 | 133.0 / 107.8 |
| `rotate_cq15_fallback` (CMSIS-DSP composition) | 91.2 / 70.9 | 91.2 / 70.9 | 91.2 / 70.9 | 91.2 / 70.9 | 91.2 / 70.9 | 91.2 / 70.9 |
| `fir` C32, 64 taps | 141.7 / 141.7 | 141.7 / 141.7 | 142.1 / 142.3 | 142.1 / 141.9 | 142.1 / 141.0 | 141.3 / 136.4 |
| `fir_decimate` C32, 127 taps, M = 10 | 141.0 / 139.0 | 141.0 / 139.0 | 142.0 / 141.2 | 141.9 / 141.1 | 141.8 / 138.5 | 141.0 / 134.3 |
| `fir_interpolate` C32, L = 2, 128 taps | 141.9 / 141.5 | 141.9 / 141.5 | 141.8 / 142.4 | 141.8 / 141.9 | 141.8 / 141.2 | 141.2 / 136.5 |
| `resample` C32, 3 / 2, 192 taps | 141.6 / 141.4 | 141.6 / 141.4 | 142.2 / 142.2 | 142.3 / 141.9 | 142.2 / 140.8 | 141.6 / 136.3 |
| `fir_interpolate_real` f32, L = 2, 128 taps | 141.8 / 141.4 | 141.8 / 141.5 | 141.8 / 140.7 | 141.8 / 139.4 | 141.8 / 142.3 | 141.4 / 136.4 |
| `resample_real` f32, 3 / 2, 192 taps | 141.8 / 141.3 | 141.8 / 141.4 | 142.3 / 140.6 | 142.3 / 139.6 | 142.3 / 142.1 | 141.8 / 136.7 |
| `fir_interpolate_real` Q15, L = 2, 128 taps | exact | exact | exact | exact | exact | exact |
| `resample_real` Q15, 3 / 2, 192 taps | exact | exact | exact | exact | exact | exact |
| `fir` CQ15, 64 taps (fallback) | exact | exact | exact | exact | exact | exact |
| `fir_decimate` CQ15, 127 taps, M = 10 (fallback) | exact | exact | exact | exact | exact | exact |
| `fir_interpolate` CQ15, L = 2, 128 taps (fallback) | exact | exact | exact | exact | exact | exact |
| `resample` CQ15, 3 / 2, 192 taps (fallback) | exact | exact | exact | exact | exact | exact |
| `mag_sq` C32 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 |
| `mag_sq` CQ15 | 92.8 / 92.8 | 92.8 / 92.8 | 92.8 / 92.8 | 92.8 / 92.8 | 92.8 / 92.8 | 92.8 / 92.8 |
| `quad_demod` C32 | 143.5 / 144.2 | 143.5 / 144.2 | 144.7 / 126.5 | 144.7 / 126.5 | 144.7 / 126.5 | 143.5 / 143.6 |
| `quad_demod` CQ15 | 81.2 / 81.2 | 81.2 / 81.2 | 81.2 / 81.2 | 81.2 / 81.2 | 81.2 / 81.2 | 81.2 / 81.2 |
| `quad_demod_fallback` C32 (GNU Radio) | 143.6 / 114.0 | 143.6 / 114.0 | 143.6 / 114.0 | 143.6 / 114.0 | 143.6 / 114.0 | 143.6 / 114.0 |
| `quad_demod_cq15_fallback` (CMSIS-DSP) | 81.2 / 62.3 | 81.2 / 62.3 | 81.2 / 62.3 | 81.2 / 62.3 | 81.2 / 62.3 | 81.2 / 62.3 |
| `fm_mod` C32 | 133.2 / 96.7 | 133.2 / 96.7 | 133.2 / 96.7 | 133.2 / 96.7 | 133.2 / 94.8 | 133.2 / 96.7 |
| `fm_mod` CQ15 | 89.4 / 89.4 | 89.4 / 89.4 | 89.4 / 89.4 | 89.4 / 89.4 | 89.4 / 89.4 | 89.4 / 89.4 |
| `fm_mod_fallback` C32 (GNU Radio) | 133.2 / 96.7 | 133.2 / 96.7 | 133.2 / 96.7 | 133.2 / 96.7 | 133.2 / 94.8 | 133.2 / 96.7 |
| `fm_mod_cq15_fallback` (CMSIS-DSP composition) | 89.4 / 77.1 | 89.4 / 77.1 | 89.4 / 77.1 | 89.4 / 77.1 | 89.4 / 77.1 | 89.4 / 77.1 |
| `squelch` C32 | exact | exact | exact | exact | exact | exact |
| `squelch` CQ15 | exact | exact | exact | exact | exact | exact |
| `mag` C32 | 150.5 / 150.5 | 150.5 / 150.5 | 150.5 / 150.5 | 150.5 / 150.9 | 150.5 / 150.9 | 150.5 / 150.5 |
| `mag` CQ15 | 96.2 / 96.2 | 96.2 / 96.2 | 96.2 / 96.2 | 96.2 / 96.2 | 96.2 / 96.2 | 96.2 / 96.2 |
| `window` C32 | 151.9 / 151.9 | 151.9 / 151.9 | 151.9 / 151.9 | 151.9 / 151.9 | 151.9 / 151.9 | 151.9 / 151.9 |
| `window` CQ15 | 89.0 / 89.0 | 89.0 / 89.0 | 89.0 / 89.0 | 89.0 / 89.0 | 89.0 / 89.0 | 89.0 / 89.0 |
| `power_db` f32 | 151.9 / 143.0 | 151.9 / 141.5 | 151.9 / 141.6 | 151.9 / 141.6 | 151.9 / 141.6 | 151.9 / 143.0 |
| `power_db` Q15 | 149.9 / 149.9 | 149.9 / 149.9 | 149.9 / 149.9 | 149.9 / 149.9 | 149.9 / 149.9 | 149.9 / 149.9 |
| `mag_sq_fallback` C32 (CMSIS-DSP) | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 |
| `mag_sq_q15_fallback` (CMSIS-DSP) | 92.8 / 74.8 | 92.8 / 74.8 | 92.8 / 74.8 | 92.8 / 74.8 | 92.8 / 74.8 | 92.8 / 74.8 |
| `mag_fallback` C32 (CMSIS-DSP) | 150.5 / 150.5 | 150.5 / 150.5 | 150.5 / 150.5 | 150.5 / 150.5 | 150.5 / 150.5 | 150.5 / 150.5 |
| `mag_q15_fallback` (CMSIS-DSP) | 96.2 / 84.3 | 96.2 / 84.3 | 96.2 / 84.3 | 96.2 / 84.3 | 96.2 / 84.3 | 96.2 / 84.3 |
| `window_fallback` C32 (CMSIS-DSP) | 151.9 / 151.9 | 151.9 / 151.9 | 151.9 / 151.9 | 151.9 / 151.9 | 151.9 / 151.9 | 151.9 / 151.9 |
| `window_cq15_fallback` (CMSIS-DSP) | 89.0 / 83.1 | 89.0 / 83.1 | 89.0 / 83.1 | 89.0 / 83.1 | 89.0 / 83.1 | 89.0 / 83.1 |
| `power_db_fallback` f32 (CMSIS-DSP) | 151.9 / 143.2 | 151.9 / 143.2 | 151.9 / 143.2 | 151.9 / 143.2 | 151.9 / 143.2 | 151.9 / 143.2 |
| `power_db_q15_fallback` (CMSIS-DSP) | 149.9 / 144.0 | 149.9 / 144.0 | 149.9 / 144.0 | 149.9 / 144.0 | 149.9 / 144.0 | 149.9 / 144.0 |
| `agc` C32 | 119.8 / 118.9 | 119.8 / 118.9 | 119.8 / 118.9 | 119.8 / 118.9 | 119.8 / 118.9 | 119.8 / 118.9 |
| `agc` CQ15 | exact | exact | exact | exact | exact | exact |
| `agc_fallback` C32 (GNU Radio) | 119.8 / 118.9 | 119.8 / 118.9 | 119.8 / 118.9 | 119.8 / 118.9 | 119.8 / 118.9 | 119.8 / 118.9 |
| `dc_block_real` f32 | 152.1 / 124.9 | 152.1 / 124.9 | 152.1 / 124.9 | 152.1 / 124.9 | 152.1 / 124.9 | 152.1 / 124.9 |
| `dc_block_real` Q15 | exact | exact | exact | exact | exact | exact |
| `deemphasis` f32 | 152.1 / 145.4 | 152.1 / 145.4 | 152.1 / 145.4 | 152.1 / 145.4 | 152.1 / 145.4 | 152.1 / 145.4 |
| `deemphasis` Q15 | exact | exact | exact | exact | exact | exact |
| `dc_block` C32 | 151.9 / 120.8 | 151.9 / 120.8 | 151.9 / 120.8 | 151.9 / 120.8 | 151.9 / 120.8 | 120.8 / 120.8 |
| `dc_block` CQ15 | exact | exact | exact | exact | exact | exact |
| `dc_block_real_fallback` f32 (liquid-dsp) | 124.9 / 84.1 | 124.9 / 84.1 | 124.9 / 84.1 | 124.9 / 84.1 | 124.9 / 84.1 | 124.9 / 84.1 |
| `deemphasis_fallback` f32 (GNU Radio) | 145.4 / 152.0 | 145.4 / 152.0 | 145.4 / 152.0 | 145.4 / 152.0 | 145.4 / 152.0 | 145.4 / 152.0 |
| `dc_block_fallback` C32 (liquid-dsp) | 120.8 / 85.3 | 120.8 / 85.3 | 120.8 / 85.3 | 120.8 / 85.3 | 120.8 / 85.3 | 120.8 / 85.3 |
| `biquad` C32, 2 sections (KFR) | 136.1 / 136.1 | 136.1 / 136.1 | 136.1 / 136.1 | 136.1 / 136.1 | 136.1 / 136.1 | 136.1 / 136.1 |
| `biquad` CQ15, 2 sections (fallback) | 71.6 / 71.6 | 71.6 / 71.6 | 71.6 / 71.6 | 71.6 / 71.6 | 71.6 / 71.6 | 71.6 / 71.6 |
| `biquad_real` f32, 2 sections (KFR) | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 |
| `biquad_real` Q15, 2 sections (fallback) | 71.5 / 71.5 | 71.5 / 71.5 | 71.5 / 71.5 | 71.5 / 71.5 | 71.5 / 71.5 | 71.5 / 71.5 |
| `biquad_fallback` C32 (CMSIS-DSP) | 136.1 / 134.9 | 136.1 / 134.9 | 136.1 / 134.9 | 136.1 / 134.9 | 136.1 / 134.9 | 136.1 / 134.9 |
| `biquad_cq15_fallback` (CMSIS-DSP) | 71.6 / 56.7 | 71.6 / 56.7 | 71.6 / 56.7 | 71.6 / 56.7 | 71.6 / 56.7 | 71.6 / 56.7 |
| `biquad_real_fallback` f32 (CMSIS-DSP) | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 |
| `biquad_real_q15_fallback` (CMSIS-DSP) | 71.5 / 56.6 | 71.5 / 56.6 | 71.5 / 56.6 | 71.5 / 56.6 | 71.5 / 56.6 | 71.5 / 56.6 |
| `fir_fallback` C32 (CMSIS-DSP) | 136.3 / 136.4 | 136.3 / 136.4 | 136.3 / 136.4 | 136.3 / 136.4 | 136.3 / 136.4 | 136.3 / 136.4 |
| `fir_cq15_fallback` (CMSIS-DSP) | exact / 83.1 | exact / 83.1 | exact / 83.1 | exact / 83.1 | exact / 83.1 | exact / 83.1 |
| `fir_decimate_fallback` C32 (CMSIS-DSP) | 133.3 / 134.3 | 133.3 / 134.3 | 133.3 / 134.3 | 133.3 / 134.3 | 133.3 / 134.3 | 133.3 / 134.3 |
| `fir_decimate_cq15_fallback` (CMSIS-DSP) | exact / 82.2 | exact / 82.2 | exact / 82.2 | exact / 82.2 | exact / 82.2 | exact / 82.2 |
| `fir_interpolate_fallback` C32 (CMSIS-DSP) | 136.3 / 136.5 | 136.3 / 136.5 | 136.3 / 136.5 | 136.3 / 136.5 | 136.3 / 136.5 | 136.3 / 136.5 |
| `fir_interpolate_cq15_fallback` (CMSIS-DSP) | exact / 80.1 | exact / 80.1 | exact / 80.1 | exact / 80.1 | exact / 80.1 | exact / 80.1 |
| `resample_fallback` C32 (CMSIS-DSP interpolator) | 136.5 / 136.4 | 136.5 / 136.4 | 136.5 / 136.4 | 136.5 / 136.4 | 136.5 / 136.4 | 136.5 / 136.4 |
| `resample_cq15_fallback` (CMSIS-DSP interpolator) | exact / 78.2 | exact / 78.2 | exact / 78.2 | exact / 78.2 | exact / 78.2 | exact / 78.2 |
| `fir_interpolate_real_fallback` f32 (CMSIS-DSP) | 136.5 / 136.4 | 136.5 / 136.4 | 136.5 / 136.4 | 136.5 / 136.4 | 136.5 / 136.4 | 136.5 / 136.4 |
| `fir_interpolate_real_q15_fallback` (CMSIS-DSP) | exact / 80.3 | exact / 80.3 | exact / 80.3 | exact / 80.3 | exact / 80.3 | exact / 80.3 |
| `resample_real_fallback` f32 (CMSIS-DSP interpolator) | 136.5 / 136.6 | 136.5 / 136.6 | 136.5 / 136.6 | 136.5 / 136.6 | 136.5 / 136.6 | 136.5 / 136.6 |
| `resample_real_q15_fallback` (CMSIS-DSP interpolator) | exact / 78.0 | exact / 78.0 | exact / 78.0 | exact / 78.0 | exact / 78.0 | exact / 78.0 |
| `dot` C32, 4096 samples | 132.8 / 130.1 | 132.8 / 130.1 | 137.2 / 134.6 | 138.7 / 134.2 | 134.3 / 127.9 | 132.8 / 122.6 |
| `dot` CQ15, 4096 samples | 82.2 / 82.2 | 82.2 / 82.2 | 82.2 / 82.2 | 82.2 / 82.2 | 82.2 / 82.2 | 82.2 / 82.2 |
| `dot_real` f32, 4096 samples | 129.8 / 129.8 | 129.8 / 122.5 | 131.4 / 126.4 | 135.0 / 127.6 | 129.1 / 127.3 | 129.8 / 123.1 |
| `dot_real` Q15, 4096 samples | 88.8 / 88.8 | 88.8 / 88.8 | 88.8 / 88.8 | 88.8 / 88.8 | 88.8 / 88.8 | 88.8 / 88.8 |
| `scale` C32 | 151.5 / 151.5 | 151.5 / 151.5 | 151.5 / 151.5 | 151.5 / 151.5 | 151.5 / 151.5 | 151.5 / 151.5 |
| `scale` CQ15 | 93.7 / 93.7 | 93.7 / 93.7 | 93.7 / 93.7 | 93.7 / 93.7 | 93.7 / 93.7 | 93.7 / 93.7 |
| `dot_fallback` C32 (CMSIS-DSP) | 122.6 / 122.6 | 122.6 / 122.6 | 122.6 / 122.6 | 122.6 / 122.6 | 122.6 / 122.6 | 122.6 / 122.6 |
| `dot_cq15_fallback` (CMSIS-DSP) | 82.2 / 82.2 | 82.2 / 82.2 | 82.2 / 82.2 | 82.2 / 82.2 | 82.2 / 82.2 | 82.2 / 82.2 |
| `dot_real_fallback` f32 (CMSIS-DSP) | 123.1 / 123.1 | 123.1 / 123.1 | 123.1 / 123.1 | 123.1 / 123.1 | 123.1 / 123.1 | 123.1 / 123.1 |
| `dot_real_q15_fallback` (CMSIS-DSP) | 88.8 / 88.8 | 88.8 / 88.8 | 88.8 / 88.8 | 88.8 / 88.8 | 88.8 / 88.8 | 88.8 / 88.8 |
| `scale_fallback` C32 (CMSIS-DSP) | 151.5 / 151.5 | 151.5 / 151.5 | 151.5 / 151.5 | 151.5 / 151.5 | 151.5 / 151.5 | 151.5 / 151.5 |
| `scale_cq15_fallback` (CMSIS-DSP) | 93.7 / 87.6 | 93.7 / 87.6 | 93.7 / 87.6 | 93.7 / 87.6 | 93.7 / 87.6 | 93.7 / 87.6 |
| `add` C32 (VOLK) | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 |
| `add` CQ15 (VOLK `volk_16i_x2_add_saturated_16i`) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `add_real` f32 (VOLK) | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 |
| `add_real` Q15 (VOLK) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `mul` C32 (VOLK; fused at `avx2`, `avx512`, `neon` as ours) | 148.5 / 148.5 | 148.5 / 148.5 | 150.1 / 150.1 | 150.1 / 150.1 | 150.1 / 150.1 | 148.5 / 148.5 |
| `mul` CQ15 (our fallback) | 94.4 / 94.4 | 94.4 / 94.4 | 94.4 / 94.4 | 94.4 / 94.4 | 94.4 / 94.4 | 94.4 / 94.4 |
| `mul_real` f32 (VOLK) | 151.8 / 151.8 | 151.8 / 151.8 | 151.8 / 151.8 | 151.8 / 151.8 | 151.8 / 151.8 | 151.8 / 151.8 |
| `mul_real` Q15 (our fallback) | 91.6 / 91.6 | 91.6 / 91.6 | 91.6 / 91.6 | 91.6 / 91.6 | 91.6 / 91.6 | 91.6 / 91.6 |
| `interleave` f32 (VOLK) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `interleave` CQ15 (our fallback) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `deinterleave` C32 (VOLK) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `deinterleave` CQ15 (VOLK) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `oscillate` C32 (our fallback) | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 |
| `oscillate` CQ15 (our fallback) | 97.8 / 97.8 | 97.8 / 97.8 | 97.8 / 97.8 | 97.8 / 97.8 | 97.8 / 97.8 | 97.8 / 97.8 |
| `add_fallback` C32 (CMSIS-DSP) | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 |
| `add_cq15_fallback` (CMSIS-DSP) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `add_real_fallback` f32 (CMSIS-DSP) | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 | 151.0 / 151.0 |
| `add_real_q15_fallback` (CMSIS-DSP) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `mul_fallback` C32 (CMSIS-DSP) | 148.5 / 148.5 | 148.5 / 148.5 | 148.5 / 148.5 | 148.5 / 148.5 | 148.5 / 148.5 | 148.5 / 148.5 |
| `mul_cq15_fallback` (CMSIS-DSP, truncating Q3.13 product) | 94.4 / 19.6 | 94.4 / 19.6 | 94.4 / 19.6 | 94.4 / 19.6 | 94.4 / 19.6 | 94.4 / 19.6 |
| `mul_real_fallback` f32 (CMSIS-DSP) | 151.8 / 151.8 | 151.8 / 151.8 | 151.8 / 151.8 | 151.8 / 151.8 | 151.8 / 151.8 | 151.8 / 151.8 |
| `mul_real_q15_fallback` (CMSIS-DSP, truncating) | 91.6 / 85.6 | 91.6 / 85.6 | 91.6 / 85.6 | 91.6 / 85.6 | 91.6 / 85.6 | 91.6 / 85.6 |
| `oscillate_fallback` C32 (CMSIS-DSP `arm_sin_cos_f32`) | 133.2 / 140.8 | 133.2 / 140.8 | 133.2 / 140.8 | 133.2 / 140.8 | 133.2 / 140.8 | 133.2 / 140.8 |
| `oscillate_cq15_fallback` (CMSIS-DSP `arm_cos_q15`, `arm_sin_q15`) | 97.8 / 75.5 | 97.8 / 75.5 | 97.8 / 75.5 | 97.8 / 75.5 | 97.8 / 75.5 | 97.8 / 75.5 |
| `interleave_fallback` f32 (VOLK `generic`) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `interleave_cq15_fallback` (VOLK's `generic` interleave loop on `int16`) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `deinterleave_fallback` C32 (VOLK `generic`) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `deinterleave_cq15_fallback` (VOLK `generic`) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `fft` C32, N = 64 | 142.1 / 140.2 | 142.1 / 142.1 | 140.7 / 141.4 | 140.7 / 141.4 | 141.7 / 142.3 | 142.5 / 141.6 |
| `fft` C32, N = 1024 | 138.7 / 137.9 | 138.7 / 138.7 | 139.1 / 139.1 | 139.1 / 139.1 | 138.8 / 139.1 | 138.2 / 138.9 |
| `fft` C32, N = 4096 | 137.8 / 137.2 | 137.8 / 137.8 | 138.2 / 137.9 | 138.2 / 137.9 | 138.0 / 138.3 | 137.4 / 137.7 |
| `fft` CQ15, N = 1024, block float | 65.3 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 |
| `fft` CQ15, N = 1024, fixed scaling | 47.9 / 43.5 | 46.7 / 43.5 | 46.7 / 43.5 | 46.7 / 43.5 | 48.0 / 43.5 | 46.7 / 43.5 |
| `fft` CQ15, N = 1024, fixed scaling, full scale | 60.0 / 55.1 | 58.9 / 55.1 | 58.9 / 55.1 | 58.9 / 55.1 | 59.9 / 55.1 | 58.9 / 55.1 |
| `rfft` f32, N = 1024 | 138.6 / 138.5 | 138.6 / 138.5 | 138.7 / 138.1 | 138.3 / 138.1 | 138.5 / 138.4 | 137.8 / 138.5 |
| `fft_64_fallback` C32 (CMSIS-DSP) | 141.0 / 140.0 | 141.0 / 140.0 | 141.0 / 140.0 | 141.0 / 140.0 | 141.0 / 140.0 | 141.0 / 140.0 |
| `fft_1024_fallback` C32 (CMSIS-DSP) | 138.0 / 138.7 | 138.0 / 138.7 | 138.0 / 138.7 | 138.0 / 138.7 | 138.0 / 138.7 | 138.0 / 138.7 |
| `fft_4096_fallback` C32 (CMSIS-DSP) | 137.4 / 137.7 | 137.4 / 137.7 | 137.4 / 137.7 | 137.4 / 137.7 | 137.4 / 137.7 | 137.4 / 137.7 |
| `fft_cq15_1024_fallback` block float (CMSIS-DSP) | 68.0 / 41.8 | 68.0 / 41.8 | 68.0 / 41.8 | 68.0 / 41.8 | 68.0 / 41.8 | 68.0 / 41.8 |
| `fft_cq15_1024_default_fallback` fixed scaling (CMSIS-DSP) | 46.1 / 41.6 | 46.1 / 41.6 | 46.1 / 41.6 | 46.1 / 41.6 | 46.1 / 41.6 | 46.1 / 41.6 |
| `rfft_1024_fallback` f32 (CMSIS-DSP) | 137.9 / 138.3 | 137.9 / 138.3 | 137.9 / 138.3 | 137.9 / 138.3 | 137.9 / 138.3 | 137.9 / 138.3 |
| `halfband_decimate` C32 (liquid-dsp `resamp2_crcf`) | 142.6 / 143.2 | 142.6 / 143.2 | 143.0 / 143.9 | 143.0 / 143.8 | 143.0 / 143.2 | 142.6 / 142.1 |
| `halfband_decimate_real` f32 (liquid-dsp `resamp2_rrrf`) | 142.4 / 142.0 | 142.4 / 142.0 | 142.8 / 143.8 | 142.8 / 142.2 | 142.8 / 142.0 | 142.4 / 142.0 |
| `real_to_complex` C32 (liquid-dsp `firhilbf`) | 148.7 / 143.2 | 148.7 / 143.4 | 150.3 / 145.5 | 150.3 / 143.3 | 150.3 / 143.0 | 148.7 / 143.2 |
| `real_to_complex_airspy` C32 (libairspy `iqconverter_float`, SSE2, second gate) | 148.7 / 148.7 | - | - | - | - | - |
| `halfband_decimate` CQ15 (our fallback) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `halfband_decimate_real` Q15 (our fallback) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `real_to_complex` CQ15 (libairspy `iqconverter_int16`, provisional) | exact / 81.5 | exact / 81.5 | exact / 81.5 | exact / 81.5 | exact / 81.5 | exact / 81.5 |
| `halfband_decimate_fallback` C32 (liquid-dsp portable) | 142.1 / 142.1 | 142.1 / 142.1 | 142.1 / 142.1 | 142.1 / 142.1 | 142.1 / 142.1 | 142.1 / 142.1 |
| `halfband_decimate_real_fallback` f32 (liquid-dsp portable) | 142.5 / 142.0 | 142.5 / 142.0 | 142.5 / 142.0 | 142.5 / 142.0 | 142.5 / 142.0 | 142.5 / 142.0 |
| `real_to_complex_fallback` C32 (liquid-dsp portable) | 143.5 / 143.2 | 143.5 / 143.2 | 143.5 / 143.2 | 143.5 / 143.2 | 143.5 / 143.2 | 143.5 / 143.2 |
| `halfband_decimate_cq15_fallback` (CMSIS-DSP composition) | exact / 82.5 | exact / 82.5 | exact / 82.5 | exact / 82.5 | exact / 82.5 | exact / 82.5 |
| `halfband_decimate_real_q15_fallback` (CMSIS-DSP composition) | exact / 82.6 | exact / 82.6 | exact / 82.6 | exact / 82.6 | exact / 82.6 | exact / 82.6 |
| `real_to_complex_cq15_fallback` (libairspy `iqconverter_int16`, provisional) | exact / 81.5 | exact / 81.5 | exact / 81.5 | exact / 81.5 | exact / 81.5 | exact / 81.5 |
| `stereo_matrix` C32 (VOLK composition) | 151.3 / 151.3 | 151.3 / 151.3 | 151.3 / 151.3 | 151.3 / 151.3 | 151.3 / 151.3 | 151.3 / 151.3 |
| `stereo_matrix` CQ15 (our fallback) | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact | exact / exact |
| `stereo_matrix_fallback` C32 (CMSIS-DSP planar) | 151.3 / 151.3 | 151.3 / 151.3 | 151.3 / 151.3 | 151.3 / 151.3 | 151.3 / 151.3 | 151.3 / 151.3 |
| `stereo_matrix_cq15_fallback` (CMSIS-DSP planar) | exact / 12.2 | exact / 12.2 | exact / 12.2 | exact / 12.2 | exact / 12.2 | exact / 12.2 |
| `stereo_pll_fallback` f32 (liquid-dsp `nco_crcf`) | 57.0 / 9.5 | 57.0 / 9.5 | 57.0 / 9.5 | 57.0 / 9.5 | 57.0 / 23.4 | 57.0 / 23.4 |
| `stereo_pll_q15_fallback` (CMSIS-DSP oscillator) | 53.6 / 53.7 | 53.6 / 53.7 | 53.6 / 53.7 | 53.6 / 53.7 | 53.6 / 53.7 | 53.6 / 53.7 |
| `preemph_limit` f32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `preemph_limit_q15` (our fallback) | exact | exact | exact | exact | exact | exact |
| `noise_blank` C32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `noise_blank_cq15` (our fallback) | exact | exact | exact | exact | exact | exact |
| `auto_notch` f32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `am_mod` C32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `am_mod_cq15` (our fallback) | exact | exact | exact | exact | exact | exact |
| `am_mod_fallback` C32 (liquid-dsp `ampmodem`) | 150.8 / 150.8 | 150.8 / 150.8 | 150.8 / 150.8 | 150.8 / 150.8 | 150.8 / 152.1 | 150.8 / 150.8 |
| `am_mod_cq15_fallback` (CMSIS-DSP composition) | 92.7 / 89.7 | 92.7 / 89.7 | 92.7 / 89.7 | 92.7 / 89.7 | 92.7 / 89.7 | 92.7 / 89.7 |
| `preemph_limit_stereo` C32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `stereo_encode` C32 to f32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `crc16_ccitt_false` (GNU Radio `gr::digital::crc`, per byte) | exact | exact | exact | exact | exact | exact |
| `crc16_x25` (GNU Radio `gr::digital::crc`, per byte) | exact | exact | exact | exact | exact | exact |
| `crc32` (GNU Radio `gr::digital::crc`, per byte) | exact | exact | exact | exact | exact | exact |
| `crc24_adsb` (GNU Radio `gr::digital::crc`, per byte) | exact | exact | exact | exact | exact | exact |
| `frame_decode` (our code, per bit) | exact | exact | exact | exact | exact | exact |
| `frame_encode_block` (our code, per output sample) | exact | exact | exact | exact | exact | exact |
| `frame_decode_block` (our code, per input sample) | exact | exact | exact | exact | exact | exact |
| `symbol_sync` C32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `symbol_sync_real` f32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `costas` C32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `costas_fallback` C32 (GNU Radio `costas_loop_cc`) | exact / 129.2 | exact / 129.2 | exact / 129.2 | exact / 129.2 | exact / 129.1 | exact / 129.2 |
| `rds_demod_block` (our code, per input sample) | exact | exact | exact | exact | exact | exact |
| `rds_decode_block` (our code, per input sample) | exact | exact | exact | exact | exact | exact |
| `rds_shape` f32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `stereo_encode_rds` C32 and f32 to f32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `morse_encode_block` (our code, per output sample) | exact | exact | exact | exact | exact | exact |
| `morse_decode_block` (our code, per input sample) | exact | exact | exact | exact | exact | exact |
| `dot_complex` C32 (VOLK `volk_32fc_x2_dot_prod_32fc`) | 123.5 / 122.7 | 123.5 / 122.7 | 123.5 / 123.2 | 123.5 / 123.2 | 123.5 / 123.2 | 123.5 / 122.7 |
| `dot_complex_fallback` C32 (CMSIS-DSP `arm_cmplx_dot_prod_f32`) | 123.5 / 119.2 | 123.5 / 119.2 | 123.5 / 119.2 | 123.5 / 119.2 | 123.5 / 119.2 | 123.5 / 119.2 |
| `equalize` C32 (our fallback) | exact | exact | exact | exact | exact | exact |
| `equalize_fallback` C32 (GNU Radio `linear_equalizer`) | exact / 127.5 | exact / 127.5 | exact / 127.5 | exact / 127.5 | exact / 127.2 | exact / 127.5 |
| `channelize` C32 (liquid-dsp `firpfbch2_crcf`, SIMD dot products and FFTW) | 140.9 / 141.2 | 140.9 / 141.2 | 141.4 / 141.6 | 141.4 / 141.4 | 141.4 / 141.3 | 140.9 / 140.9 |
| `channelize_fallback` C32 (liquid-dsp `firpfbch2_crcf`, portable) | 140.7 / 140.9 | 140.7 / 140.9 | 140.7 / 140.9 | 140.7 / 140.9 | 140.7 / 140.9 | 140.7 / 140.9 |
| `channelize_block` (our code, per input sample) | 140.8 / 140.8 | 140.8 / 140.8 | 141.3 / 141.3 | 141.3 / 141.3 | 141.3 / 141.3 | 140.8 / 140.8 |
| `channel_select_block` (our code, per output sample) | exact | exact | exact | exact | exact | exact |
| `channel_bank_channelize` (our code, per input sample) | 138.8 / 138.8 | 138.8 / 138.8 | 139.2 / 139.2 | 139.2 / 139.2 | 139.2 / 139.2 | 138.8 / 138.8 |
| `channel_bank_ddc` (our code, per input sample) | 139.0 / 139.0 | 139.0 / 139.0 | 140.5 / 140.5 | 140.3 / 140.3 | 139.4 / 139.4 | 139.0 / 139.0 |
| `band_pulse_detect_block` (our code, per lane value) | exact | exact | exact | exact | exact | exact |

Simulated cycles per sample on every level with a core model (`cargo measure <kernel> --no-count --no-golden`, llvm-mca 23.1.1, mode `whole-loop`; D33, D35, D68): the execution counts come from the counter of the level in the environment of its count (callgrind at `avx2`, Intel SDE at `sse2`, `sse4.2` and `avx512`, QEMU at `neon`), and ours and the comparison of an entry run in the same emulator on the same CPU. Only the ratio ours / comparison on one model means something; the cycles are shown so that the ratios can be checked and are no hardware figures. The ratios against the comparisons are not judged (the gate there stays the instruction ratio above); one rule is gated on these tables (D321): every entry `<k>` with a row `<k>_fallback` on the same level must have at most the cycles of that row in the column "ours" on both models, as published (two decimals); a level delegated to the fallback (comparison `sdrk fallback`, both ratios 1.00) passes, and an exception needs a `cycle_waiver` in `benches/limits.json` naming an open item. `cargo xtask check cycles` (part of `cargo xtask check`, so CI) reads the committed tables, so a merged row that breaks the rule fails CI without running `llvm-mca`; `cargo measure` prints the verdict (`SIMD vs fallback ...`) for every pair it simulates. The tables are written by `python3 -I benches/cycle_tables.py <the JSON files>` from the `cargo measure` documents (`target/measure/<kernel>.json`). The script **merges**: it replaces or inserts the rows of the entries in the given documents at their place in the order of the results tables and keeps every other row, note and level byte for byte, because `target/measure` is not committed and holds only what was measured in this checkout (a partial set never removes rows). `--rebuild` writes the block from the given documents alone and refuses to drop entries of the block unless `--force` is given (every kernel measured, or entries retired); `--doc FILE` writes another file. Its tests: `python3 -I benches/test_cycle_tables.py`. The `endbr64` of the references built with `-fcf-protection` is simulated as the `nop` it executes as (D64, [How the cycles are simulated](#how-the-cycles-are-simulated)). One artifact of the model remains at `avx2`: GNU Radio's `quadrature_demod_cf` zero-fills its temporary with a `rep stosb` that `znver3` rates at 100 and `alderlake` at 8 cycles per byte, so its 865.81 / 131.57 are about 62 / 65 without it; `znver4` rates the same instruction alike at `avx512` (868.77 there), so the ratios of `quad_demod_fallback` at `avx2` and `avx512` on the reference model are artifacts. The `C32` FFT fallbacks above 1.10 at `sse2` and `sse4.2` (radix-2 contract, 16 `xmm` registers; D170 fuses three stages per pass only off x86), the `CQ15` FIR family fallbacks above 1.10 at `sse2`, `sse4.2` and `neon` (`fir_decimate` `CQ15` on `neon` 1.80 / 1.63 after D172, against a CMSIS-DSP that GCC auto-vectorizes with NEON), and the `convert_out` fallbacks on every level, are above CMSIS-DSP in the instructions too.

The M40 entries `costas` and `costas_fallback` were counted on all four platforms (`cargo xtask count --platform <p> --kernel costas --kernel costas_fallback --merge`, 2026-11-05) and measured with `cargo measure costas` and `cargo measure costas_fallback` on every level with a core model (D608). The fallback counts 0.69 to 0.89 of the instructions of GNU Radio's `costas_loop_cc` but simulates 1.03 to 1.53 of its cycles (3.22 on `alderlake` at `avx2`): both loops are bound by the latency from one symbol to the next, and ours has the longer chain (the table lookup with the integer-to-float conversion of the fraction, the derotation, the detector, the flushed and clamped integrator and the exact `f64` phase step of the contract) where GNU Radio's `sincosf` call overlaps with the rest. The cycle ratios against the libraries are not gated. M40.1 (order 8, D645) changed the order dispatch of the fallback (one loop per order 2, 4 and 8); both entries were counted again on all four platforms (2026-11-06) with the same instructions per symbol (116.03 `sse2`, 112.03 `sse4.2`, 96.03 `avx2`, 95.03 `avx512`, 89.03 `neon`, 228.05 `simd128`; GNU Radio 151.96, 151.96, 137.67, 137.67, 105.13, 255.33) and measured on every level with a core model with the same cycles up to 0.01 (`znver2` at `sse2` 99.06, `cortex-a76` at `neon` 77.05). The workload stays QPSK at order 4; order 8 has no entry of its own (GNU Radio's `costas_loop_cc` has another order-8 detector).

The M47 entries `morse_encode_block` and `morse_decode_block` (our blocks, per output and per input sample of an 8 kHz envelope; no library has them, so information against our own code, D910) were counted on all four platforms (`cargo xtask count --platform <p> --kernel morse_encode_block --kernel morse_decode_block --merge`, 2026-11-07) and measured with `cargo measure` on every level with a core model. Cycles per sample, whole-loop, reference / counter model: the encoder 6.60 / 7.58 (`sse2`, `sse4.2`), 4.44 / 4.50 (`avx2`), 4.20 / 4.64 (`avx512`) and 8.56 / 9.45 (`neon`, `cortex-a72` / `cortex-a76`); the decoder 16.81 / 20.53 (`sse2`), 16.62 / 20.31 (`sse4.2`), 13.06 / 16.08 (`avx2`), 13.19 / 17.03 (`avx512`) and 22.26 / 28.43 (`neon`). The decoder had 147.92 instructions per sample at `avx2` before the speed round of D909 and has 69.08 now; the cycles above are those of the final code. Both blocks are scalar code with a dependency from one sample to the next (the smoother, the level trackers, the run counter), so the levels differ only by code generation.

The M45 entries `channelize` and `channelize_fallback` (16 channels, 8 taps per branch, per input sample) were counted on all four platforms (`cargo xtask count --platform <p> --kernel channelize --kernel channelize_fallback --merge`, 2026-11-07, recounted 2026-11-08 after D922 and after D939) and measured with `cargo measure channelize --no-golden` on every level with a core model, against liquid-dsp's `firpfbch2_crcf` (D919); both are `release` on every level. Cycles per sample, whole-loop, reference / counter model, ours against liquid: the SIMD path 25.93 / 27.72 against 171.64 / 176.30 (`sse2`), 21.95 / 28.21 against 171.63 / 176.29 (`sse4.2`), 12.08 / 12.75 against 120.69 / 99.98 (`avx2`), 12.68 / 16.01 against 104.12 / 84.10 (`avx512`) and 30.93 / 22.59 against 302.17 / 240.53 (`neon`; 42.56 / 32.70 before the two-hop branch sums of D939, open item I45-1); the fallback 53.40 / 58.43, 53.83 / 58.12, 34.97 / 32.37, 38.10 / 32.11 and 98.73 / 81.02 against 166.47 / 180.90, 166.73 / 180.51, 149.61 / 131.39, 107.09 / 118.99 and 222.34 / 165.45 (liquid portable). The SIMD path simulates 0.28 to 0.50 of the cycles of its fallback on every level and both models (D321 holds), at `simd128` it counts 270.93 instructions per sample against 436.07 of the fallback. liquid runs a window object and a dot-product object per branch (`windowcf_push`, `dotprod_crcf_execute` through a call each) and scales every output; ours sums the columns of one history in registers. Before the round of D918 and D920 the SIMD path simulated 24.22 / 22.35 at `avx2` and the fallback 38.11 / 34.13; D922 (the fallback's rows as whole chunks of `M`) took the fallback from 101.68 / 86.10 to 98.73 / 81.02 at `neon` and from 34.70 to 38.10 on the `avx512` reference model (ARM preferred).

The M43 entries `dot_complex`, `dot_complex_fallback`, `equalize` and `equalize_fallback` were counted on all four platforms (`cargo xtask count --platform <p> --kernel dot_complex --kernel dot_complex_fallback --kernel equalize --kernel equalize_fallback --merge`, 2026-11-07) and measured with `cargo measure dot_complex` and `cargo measure equalize` on every level with a core model, after the performance round of D840 to D845. All judged entries are within the release limit and are `release`. `dot_complex` simulates 0.24 to 0.86 of the cycles of its fallback and 0.58 (`znver3`) to 1.03 (`cortex-a72`) of VOLK's protokernel without FMA. `equalize` runs the fallback on every backend (D841), so the facade entry equals the fallback (1.00, D321 holds). `equalize_fallback` counts 0.54 to 0.89 of the instructions of GNU Radio's `linear_equalizer` and simulates 0.12 to 0.93 of its cycles on x86 and 0.81 on `cortex-a76`; on `cortex-a72` it simulates 1.14 (262.11 against 229.47), the cost of the phase tracker and the smoothed `mse` that GNU Radio's loop does not have (D845: about 81 cycles per symbol on `cortex-a72`). Before D840 the fallback simulated 1.18 to 2.32 of GNU Radio's cycles where it was not below and the SIMD entry was slower than the fallback at every level. `dot_complex` at `avx2` and `avx512` counts 2.35 (after the two-block loop of `dot_complex_avx2`, 2.29 before; the cycles are unchanged), and at `avx2-runtime` 2.36 against 2.35 of the fixed `avx2` build (1.00, limit 1.02): the run-time build no longer needs a waiver (open item I43-5, closed).

The M30 kernels have cycle rows for `interleave` and `deinterleave` (both sample types) at every level with a core model (D281 to D286, `cargo measure interleave` and `cargo measure deinterleave` on 2026-11-03), for `mul` C32 and `oscillate` C32 at every level with a core model (D287 to D290, `cargo measure mul` and `cargo measure oscillate` on 2026-11-03; only these two entries were merged) and for the `oscillate` fallbacks and `oscillate_cq15` at `avx2` (D278); `mul_fallback`, `mul_cq15_fallback` and `mul` `CQ15` have rows at every level with a core model (D291 to D293, `cargo measure mul` on 2026-11-03; only these three entries were merged, the `C32` `mul` rows above are those of D288; `mul_fallback` at `avx512`, the plain loop unchanged by D291, simulates at 1.42 of CMSIS-DSP on `znver4` and 0.98 on `sapphirerapids`, within the instruction gate at 0.79; tracked in open item I30-1); the `mul` C32 row at `sse4.2` is that of D295 (`cargo measure mul --level sse4.2` on 2026-11-03, only this entry merged). The four fallback entries of `interleave` and `deinterleave` have rows at every level with a core model (D297, D298, `cargo measure interleave` and `cargo measure deinterleave` on all three platforms on 2026-11-03; the `interleave` and `deinterleave` SIMD rows were merged from the same runs, their code unchanged). Before the tuning of D298 they simulated, on the reference / counter model, at 1.12 / 1.10 (`interleave_fallback`) and 1.11 / 1.09 (`interleave_cq15_fallback`) of VOLK's `generic` code at `sse2` and `sse4.2`, `deinterleave_cq15_fallback` at 1.40 / 1.23, 1.50 / 1.07, 1.33 / 1.00 and 1.04 / 1.00 at `sse2`, `sse4.2`, `avx2` and `avx512`, and the two `deinterleave` fallbacks at 2.20 / 1.43 and 4.35 / 2.13 at `neon`; after it every one is at most 1.00 on the reference model except `deinterleave_cq15_fallback` at `avx2` (0.23 against 0.20 on `znver3`, 1.14; 0.86 on `alderlake`; 0.90 in instructions), tracked in open item I30-1. Gate of D321 (2026-11-03): every row of `add`, `mul`, `interleave` and `deinterleave` (both sample types, SIMD and fallback) was measured again on all three platforms (`cargo measure <kernel> --platform x86_64|x86_64-avx512|aarch64 --no-count --no-golden`) and merged, after D322 to D324; before them the SIMD paths of `add`, `add_cq15`, `add_real` and `mul_real` at `avx2` (0.39 / 0.38, 0.20 / 0.19 on `znver3`), `add_cq15` (0.77 / 0.76 on both models), `interleave` (1.08 / 1.02, 0.93 / 0.89) and `interleave_cq15` (0.55 / 0.51, 0.47 / 0.45) at `neon`, `interleave` (0.52 / 0.51), `interleave_cq15` (0.27 / 0.26, 0.33 / 0.31) and `deinterleave` (0.52 / 0.51, 0.58 / 0.54) at `sse2` and `sse4.2`, and `deinterleave_cq15` at `sse2` (0.76 / 0.64 on `znver2`) were slower than their fallbacks (SIMD / fallback); now each is at most the fallback on both models. Two rows got slower within the gate: `interleave` at `avx2` on `alderlake` 0.48 to 0.51 (the fallback 0.51) and at `avx512` on `znver4` 0.13 to 0.14 (the fallback 0.26). `mul` C32 with FMA (D326, `cargo measure mul` on all three platforms on 2026-11-04, only the `mul` rows merged, the fallback rows unchanged): the comparison of the rows at `avx2`, `avx512` and `neon` is VOLK's fused dispatch; ours before / after in simulated cycles per sample 0.43 / 0.45 and 0.51 / 0.51 at `avx2` (`znver3` / `alderlake`), 0.47 / 0.45 and 0.39 / 0.39 at `avx512` (`znver4` / `sapphirerapids`), 3.12 / 2.27 and 2.02 / 1.83 at `neon` (`cortex-a72` / `cortex-a76`), against VOLK's 0.49 / 0.54, 0.49 / 0.54 and 2.78 / 2.27. The `oscillate` row at `avx512` rests on 64-bit gathers on our side and 32-bit gathers in the compiler-vectorised fallback, a blind spot of the model (low confidence): `znver4` rates them far cheaper than `sapphirerapids` (the fallback 0.76 against 3.79), and both models put ours below the fallback. The whole call of `mul` CQ15 and `mul_real` Q15 in the dispatched function (D332, `cargo measure mul` and `cargo measure mul_real` on all three platforms on 2026-11-04, all their rows merged) changed only these rows, each to at most its value before: `mul_cq15` at `sse4.2` 1.43 / 1.51 to 1.42 / 1.48 and at `neon` 1.48 / 1.39 to 1.46 / 1.37, `mul_real_q15` at `avx512` 0.06 / 0.06 to 0.05 / 0.05 and at `neon` 0.44 / 0.44 to 0.41 / 0.41 (reference / counter model); the other rows of both kernels kept their values (within 0.01 of rounding in two fallback rows).

Open item I30-3 (maintainer decisions D333 to D338, 2026-11-04): the rows of the eight entries decided in this round were measured again and merged with their fallbacks (`cargo measure mag`, `biquad_real_q15`, `convert_cu8`, `agc` and `fir_interpolate_cq15 --platform aarch64`; `cargo measure mag_sq --level avx2`; `cargo measure convert_real_rs16` on all three platforms). `mag`, `mag_q15` and `biquad_real_q15` at `neon` kept their values and stay above their fallbacks on `cortex-a72` by 0.2 %, 0.1 % and 0.4 %, accepted ties with a `cycle_waiver` (D333). SIMD before / after, reference model / counter-model: `convert_cu8` at `neon` 2.15 / 1.08 to 1.18 / 0.94 (D334), `agc` at `neon` 47.07 / 22.46 to the fallback's 39.90 / 21.76 (D335, delegated), `fir_interpolate_cq15` at `neon` 145.25 / 123.36 to 75.82 / 76.00 (D336), `mag_sq` at `avx2` 0.26 / 0.38 to 0.22 / 0.26 (D337), `convert_real_rs16` at `avx2` 0.22 / 0.14 to 0.21 / 0.14 (D338: 0.2148 against 0.2132 of the fallback on `znver3`, equal as published, which is what the gate compares); the other levels of `convert_real_rs16` changed with D338 to at most their values before.

Open item I30-3, the seven filter entries (maintainer decision 1b, D340, D341, 2026-11-04): `cargo measure biquad_cq15 --level sse2` and `cargo measure <kernel> --level sse4.2` for `fir_cq15`, `fir_decimate_cq15` and `fir_interpolate_cq15` re-measured the committed rows first (unchanged); after the change the rows of all seven entries and their fallbacks were measured again and merged. SIMD before / after, reference model / counter-model: at `sse4.2` (D340, the outputs of a branch four at a time with taps split for `pmaddwd`) `fir_cq15` 176.75 / 119.09 to 47.80 / 49.03, `fir_decimate_cq15` 39.18 / 27.75 to 8.90 / 9.34, `fir_interpolate_cq15` 249.66 / 144.54 to 51.51 / 53.45, `fir_interpolate_real_q15` 136.89 / 88.20 to 26.86 / 32.49, `resample_cq15` 249.52 / 143.48 to 50.01 / 51.67, `resample_real_q15` 136.55 / 87.57 to 26.11 / 31.59, each now 0.21 to 0.37 of its fallback; at `sse2` `biquad_cq15` 32.45 / 20.44 to the fallback's 24.94 / 31.17 (D341, delegated; the row of `biquad_cq15_fallback` moved from 24.43 / 31.16 to 24.94 / 31.17 with the same source, the code of the new build). Instructions per sample (`cargo xtask count --platform x86_64 --level <level> --kernel <entry> --merge` for the seven entries and `biquad_cq15_fallback`): at `sse4.2` `fir_cq15` 417.99 to 187.33, `fir_decimate_cq15` 87.46 to 35.72, `fir_interpolate_cq15` 510.19 to 201.70, `fir_interpolate_real_q15` 315.00 to 114.51, `resample_cq15` 506.50 to 196.06, `resample_real_q15` 313.14 to 111.86; at `sse2` `biquad_cq15` 51.81 to 98.08 (the fallback's) and `biquad_cq15_fallback` 100.07 to 98.08; `avx2` unchanged and `avx2-runtime` within 0.01 (`biquad_cq15` and `fir_decimate_cq15` 18.06 to 18.07 and 50.14 to 50.15, the test of the level in the entry), `avx512` unchanged in the counts of the six filters, `neon` and `simd128` source-identical (recounted: `fir_cq15` 227.65 and 841.39 as committed).

The M42 entries `rds_shape` and `stereo_encode_rds` were counted on all four platforms (`cargo xtask count --platform <p> --kernel rds_shape --kernel stereo_encode_rds --merge`, 2026-11-06) and measured with `cargo measure rds_shape` and `cargo measure stereo_encode_rds` (`--no-count --no-golden`, also with `--platform x86_64-avx512` and `--platform aarch64`) on every level with a core model (D757). Both are the fallback against itself (ratio 1.00, except 0.99 for `rds_shape` at `simd128`: 168.41 against 169.41 instructions per sample, the backend build against the fallback build of the same code). `rds_shape` costs 72 to 113 instructions and 41 to 84 simulated cycles per sample (`simd128` 168 instructions): one f64 division for the position and four interpolated table reads with their f64 index arithmetic per sample, latency-bound; the fallback takes `floor` of the non-negative positions as a truncating cast and reads the table as a fixed-size array (D758), 296 instructions per sample at `avx2` before. `stereo_encode_rds` costs 8.3 (`avx512`) to 70 instructions and 1.6 to 22 simulated cycles per sample, the cost of `stereo_encode` plus a third table read.

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
| `rotate_cq15` | 3.80 | 12.10 | 0.31 | 3.85 | 12.37 | 0.31 | sdrk fallback |
| `rotate_cq15_fallback` | 12.10 | 36.49 | 0.33 | 12.37 | 34.93 | 0.35 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `rotate_fallback` | 8.64 | 15.45 | 0.56 | 8.88 | 16.79 | 0.53 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `fir` | 33.48 | 77.65 | 0.43 | 29.84 | 78.96 | 0.38 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 142.31 | 142.31 | 1.00 | 188.30 | 188.30 | 1.00 | sdrk fallback |
| `fir_cq15_fallback` | 142.31 | 177.35 | 0.80 | 188.30 | 220.90 | 0.85 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_decimate` | 6.88 | 12.46 | 0.55 | 6.05 | 13.37 | 0.45 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 30.41 | 30.41 | 1.00 | 45.20 | 45.20 | 1.00 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 30.41 | 24.93 | 1.22 | 45.20 | 25.79 | 1.75 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_decimate_fallback` | 15.09 | 38.80 | 0.39 | 15.82 | 23.74 | 0.67 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_fallback` | 42.70 | 132.97 | 0.32 | 51.73 | 111.96 | 0.46 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate` | 36.23 | 86.10 | 0.42 | 34.15 | 88.20 | 0.39 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 137.50 | 137.50 | 1.00 | 207.96 | 207.96 | 1.00 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 137.50 | 137.61 | 1.00 | 207.96 | 164.43 | 1.26 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate_fallback` | 69.81 | 141.75 | 0.49 | 71.88 | 114.81 | 0.63 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate_real` | 19.42 | 79.39 | 0.24 | 18.14 | 77.13 | 0.24 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 42.43 | 70.88 | 0.60 | 44.68 | 57.41 | 0.78 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fir_interpolate_real_q15` | 74.97 | 74.97 | 1.00 | 112.89 | 112.89 | 1.00 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 74.97 | 68.81 | 1.09 | 112.89 | 82.21 | 1.37 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `resample` | 38.71 | 87.37 | 0.44 | 35.17 | 89.05 | 0.39 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 168.56 | 168.56 | 1.00 | 240.71 | 240.71 | 1.00 | sdrk fallback |
| `resample_cq15_fallback` | 168.56 | 136.68 | 1.23 | 240.71 | 163.55 | 1.47 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `resample_fallback` | 84.56 | 141.22 | 0.60 | 86.87 | 113.81 | 0.76 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `resample_real` | 22.43 | 80.59 | 0.28 | 22.04 | 77.71 | 0.28 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 84.62 | 70.61 | 1.20 | 86.80 | 56.90 | 1.53 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `resample_real_q15` | 103.68 | 103.68 | 1.00 | 144.23 | 144.23 | 1.00 | sdrk fallback |
| `resample_real_q15_fallback` | 103.68 | 68.34 | 1.52 | 144.23 | 81.78 | 1.76 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
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
| `dot_cq15` | 1.00 | 2.20 | 0.46 | 0.82 | 2.53 | 0.32 | sdrk fallback |
| `dot_cq15_fallback` | 2.20 | 1.78 | 1.23 | 2.53 | 2.03 | 1.24 | CMSIS-DSP 1.18.0 arm_dot_prod_q15 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `dot_fallback` | 4.03 | 8.07 | 0.50 | 3.03 | 6.06 | 0.50 | CMSIS-DSP 1.18.0 arm_dot_prod_f32 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `dot_real` | 0.27 | 0.28 | 0.97 | 0.29 | 0.27 | 1.06 | volk_32f_x2_dot_prod_32f |
| `dot_real_fallback` | 4.04 | 4.04 | 1.00 | 3.03 | 3.03 | 1.00 | CMSIS-DSP 1.18.0 arm_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `dot_real_q15` | 0.34 | 0.98 | 0.35 | 0.36 | 0.98 | 0.36 | sdrk fallback |
| `dot_real_q15_fallback` | 0.98 | 0.89 | 1.10 | 0.98 | 1.02 | 0.96 | CMSIS-DSP 1.18.0 arm_dot_prod_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `scale` | 0.51 | 0.64 | 0.80 | 0.51 | 0.79 | 0.64 | volk_32f_s32f_multiply_32f |
| `scale_cq15` | 2.03 | 2.03 | 1.00 | 2.11 | 2.11 | 1.00 | sdrk fallback |
| `scale_cq15_fallback` | 2.03 | 3.78 | 0.54 | 2.11 | 3.65 | 0.58 | CMSIS-DSP 1.18.0 arm_scale_q15 (shift 0, truncating) on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `scale_fallback` | 0.63 | 0.63 | 1.00 | 0.79 | 0.79 | 1.00 | CMSIS-DSP 1.18.0 arm_scale_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `add` | 0.51 | 0.70 | 0.72 | 0.56 | 0.79 | 0.70 | volk_32fc_x2_add_32fc |
| `add_cq15` | 0.25 | 0.35 | 0.72 | 0.28 | 0.40 | 0.70 | volk_16i_x2_add_saturated_16i on the 2 n values |
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
| `fft_1024` | 7.20 | 6.91 | 1.04 | 6.37 | 6.78 | 0.94 | FFTW (fftwf_plan_dft_1d, FFTW_ESTIMATE, forward, out of place) |
| `fft_1024_fallback` | 31.46 | 26.16 | 1.20 | 30.31 | 28.91 | 1.05 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_4096` | 8.15 | 10.41 | 0.78 | 7.32 | 8.77 | 0.83 | FFTW (fftwf_plan_dft_1d, FFTW_ESTIMATE, forward, out of place) |
| `fft_4096_fallback` | 37.63 | 30.53 | 1.23 | 36.49 | 32.90 | 1.11 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_64` | 4.46 | 5.86 | 0.76 | 3.95 | 5.51 | 0.72 | FFTW (fftwf_plan_dft_1d, FFTW_ESTIMATE, forward, out of place) |
| `fft_64_fallback` | 20.08 | 15.15 | 1.33 | 19.29 | 17.76 | 1.09 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_cq15_1024` | 8.84 | 52.83 | 0.17 | 9.71 | 63.00 | 0.15 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 7.37 | 52.83 | 0.14 | 7.76 | 63.00 | 0.12 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 50.23 | 44.75 | 1.12 | 54.68 | 56.34 | 0.97 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_cq15_1024_fallback` | 83.13 | 44.75 | 1.86 | 91.71 | 56.34 | 1.63 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `fft_cq15_1024_full_scale` | 7.68 | 52.83 | 0.15 | 8.00 | 63.00 | 0.13 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 4.71 | 5.63 | 0.84 | 4.42 | 4.58 | 0.96 | FFTW (fftwf_plan_dft_r2c_1d, FFTW_ESTIMATE, forward, out of place) |
| `rfft_1024_fallback` | 16.76 | 14.25 | 1.18 | 16.76 | 15.04 | 1.11 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `am_mod` | 1.09 | 1.09 | 1.00 | 1.16 | 1.16 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 2.62 | 2.62 | 1.00 | 3.12 | 3.12 | 1.00 | sdrk fallback |
| `am_mod_cq15_fallback` | 2.62 | 4.17 | 0.63 | 3.12 | 4.64 | 0.67 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `am_mod_fallback` | 1.09 | 0.59 | 1.84 | 1.16 | 0.72 | 1.61 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 233.12 | 233.12 | 1.00 | 207.68 | 207.68 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 56.90 | 56.90 | 1.00 | 56.38 | 56.38 | 1.00 | sdrk fallback |
| `channel_bank_channelize` | 46.06 | 46.06 | 1.00 | 52.54 | 52.54 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 227.45 | 227.45 | 1.00 | 212.36 | 212.36 | 1.00 | sdrk fallback |
| `channel_select_block` | 0.60 | 0.60 | 1.00 | 0.77 | 0.77 | 1.00 | sdrk fallback |
| `channelize` | 25.93 | 171.64 | 0.15 | 27.72 | 176.30 | 0.16 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 26.79 | 26.79 | 1.00 | 27.83 | 27.83 | 1.00 | sdrk fallback |
| `channelize_fallback` | 53.40 | 166.47 | 0.32 | 58.43 | 180.90 | 0.32 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `costas` | 115.06 | 115.06 | 1.00 | 99.06 | 99.06 | 1.00 | sdrk fallback |
| `costas_fallback` | 115.06 | 86.69 | 1.33 | 99.06 | 80.46 | 1.23 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 13.55 | 15.15 | 0.89 | 11.55 | 13.14 | 0.88 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 8.06 | 8.15 | 0.99 | 7.06 | 7.14 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 13.05 | 15.15 | 0.86 | 11.06 | 13.14 | 0.84 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 8.06 | 8.15 | 0.99 | 7.06 | 7.15 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 157.42 | 157.42 | 1.00 | 164.22 | 164.22 | 1.00 | sdrk fallback |
| `equalize_fallback` | 157.42 | 168.47 | 0.93 | 164.22 | 201.30 | 0.82 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 6.18 | 6.18 | 1.00 | 8.41 | 8.41 | 1.00 | sdrk fallback |
| `frame_decode_block` | 6.01 | 6.01 | 1.00 | 7.96 | 7.96 | 1.00 | sdrk fallback |
| `frame_encode_block` | 1.86 | 1.86 | 1.00 | 1.82 | 1.82 | 1.00 | sdrk fallback |
| `halfband_decimate` | 7.01 | 59.03 | 0.12 | 7.93 | 50.49 | 0.16 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its SSE dot product (-O3 -ffp-contract=off, sse2) |
| `halfband_decimate_cq15` | 16.69 | 39.56 | 0.42 | 16.24 | 46.15 | 0.35 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 39.56 | 41.57 | 0.95 | 46.15 | 50.09 | 0.92 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `halfband_decimate_fallback` | 15.16 | 82.14 | 0.18 | 17.48 | 69.57 | 0.25 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `halfband_decimate_real` | 3.68 | 68.61 | 0.05 | 4.01 | 55.10 | 0.07 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `halfband_decimate_real_fallback` | 12.87 | 68.61 | 0.19 | 13.53 | 55.10 | 0.25 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `halfband_decimate_real_q15` | 7.02 | 22.39 | 0.31 | 6.70 | 27.24 | 0.25 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 22.39 | 20.77 | 1.08 | 27.24 | 25.03 | 1.09 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `halfband_ref_fir_decimate` | 13.14 | 29.62 | 0.44 | 12.41 | 29.82 | 0.42 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 60.59 | 60.59 | 1.00 | 86.10 | 86.10 | 1.00 | sdrk fallback |
| `halfband_ref_shifted` | 9.73 | 17.92 | 0.54 | 11.45 | 21.03 | 0.54 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 19.61 | 42.48 | 0.46 | 20.41 | 50.32 | 0.41 | sdrk fallback |
| `morse_decode_block` | 16.81 | 16.81 | 1.00 | 20.51 | 20.51 | 1.00 | sdrk fallback |
| `morse_encode_block` | 6.60 | 6.60 | 1.00 | 7.58 | 7.58 | 1.00 | sdrk fallback |
| `noise_blank` | 15.92 | 15.92 | 1.00 | 11.94 | 11.94 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 9.59 | 9.59 | 1.00 | 10.59 | 10.59 | 1.00 | sdrk fallback |
| `preemph_limit` | 16.10 | 16.10 | 1.00 | 11.93 | 11.93 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 51.64 | 51.64 | 1.00 | 43.84 | 43.84 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 21.13 | 21.13 | 1.00 | 22.70 | 22.70 | 1.00 | sdrk fallback |
| `rds_decode_block` | 9.18 | 9.18 | 1.00 | 11.91 | 11.91 | 1.00 | sdrk fallback |
| `rds_demod_block` | 44.51 | 44.52 | 1.00 | 39.74 | 39.75 | 1.00 | sdrk fallback |
| `rds_shape` | 83.94 | 83.94 | 1.00 | 75.83 | 75.83 | 1.00 | sdrk fallback |
| `real_to_complex` | 5.31 | 70.40 | 0.08 | 5.53 | 57.25 | 0.10 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `real_to_complex_airspy` | 5.31 | 11.28 | 0.47 | 5.53 | 9.37 | 0.59 | libairspy 1.0.10 iqconverter_float, conversion without remove_dc (USE_SSE2) (-O3 -ffp-contract=off, sse2) |
| `real_to_complex_cq15` | 8.98 | 15.69 | 0.57 | 8.96 | 17.83 | 0.50 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, sse2) |
| `real_to_complex_cq15_fallback` | 29.09 | 15.69 | 1.85 | 34.91 | 17.83 | 1.96 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, sse2) |
| `real_to_complex_fallback` | 17.76 | 70.40 | 0.25 | 20.65 | 57.25 | 0.36 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse2) |
| `stereo_encode` | 10.15 | 10.15 | 1.00 | 10.88 | 10.88 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 18.68 | 18.68 | 1.00 | 17.96 | 17.96 | 1.00 | sdrk fallback |
| `stereo_matrix` | 1.02 | 2.10 | 0.49 | 0.99 | 2.48 | 0.40 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.51 | 1.02 | 0.50 | 0.53 | 1.14 | 0.47 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 1.02 | 5.50 | 0.19 | 1.14 | 5.68 | 0.20 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `stereo_matrix_fallback` | 1.55 | 1.07 | 1.45 | 1.67 | 1.38 | 1.21 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `stereo_pll_fallback` | 88.89 | 71.63 | 1.24 | 77.84 | 63.55 | 1.22 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, sse2) |
| `stereo_pll_q15_fallback` | 38.04 | 55.57 | 0.68 | 38.06 | 58.48 | 0.65 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse2) |
| `symbol_sync` | 49.33 | 49.33 | 1.00 | 39.57 | 39.57 | 1.00 | sdrk fallback |
| `symbol_sync_real` | 23.16 | 23.16 | 1.00 | 18.66 | 18.66 | 1.00 | sdrk fallback |

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
| `rotate_cq15` | 2.78 | 9.55 | 0.29 | 2.54 | 10.37 | 0.24 | sdrk fallback |
| `rotate_cq15_fallback` | 9.55 | 34.89 | 0.27 | 10.37 | 33.66 | 0.31 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `rotate_fallback` | 7.15 | 15.45 | 0.46 | 8.62 | 16.79 | 0.51 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `fir` | 33.48 | 86.68 | 0.39 | 29.84 | 88.48 | 0.34 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 47.80 | 142.31 | 0.34 | 49.16 | 188.30 | 0.26 | sdrk fallback |
| `fir_cq15_fallback` | 142.31 | 177.24 | 0.80 | 188.30 | 220.40 | 0.85 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_decimate` | 6.87 | 14.45 | 0.48 | 6.05 | 14.17 | 0.43 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 8.90 | 30.41 | 0.29 | 9.37 | 45.20 | 0.21 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 30.41 | 27.95 | 1.09 | 45.20 | 22.79 | 1.98 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_decimate_fallback` | 15.09 | 38.81 | 0.39 | 15.82 | 23.69 | 0.67 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_fallback` | 42.70 | 132.97 | 0.32 | 51.73 | 111.96 | 0.46 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate` | 36.27 | 95.13 | 0.38 | 34.15 | 97.71 | 0.35 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 51.51 | 137.49 | 0.37 | 53.70 | 207.96 | 0.26 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 137.49 | 137.61 | 1.00 | 207.96 | 164.43 | 1.26 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate_fallback` | 69.81 | 141.75 | 0.49 | 71.88 | 114.81 | 0.63 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate_real` | 19.44 | 72.80 | 0.27 | 18.18 | 62.85 | 0.29 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 42.43 | 70.88 | 0.60 | 44.68 | 57.41 | 0.78 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fir_interpolate_real_q15` | 26.86 | 74.97 | 0.36 | 32.49 | 112.89 | 0.29 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 74.97 | 68.81 | 1.09 | 112.89 | 82.21 | 1.37 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `resample` | 38.79 | 96.79 | 0.40 | 35.04 | 98.80 | 0.35 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 50.01 | 168.56 | 0.30 | 51.87 | 240.71 | 0.22 | sdrk fallback |
| `resample_cq15_fallback` | 168.56 | 136.68 | 1.23 | 240.71 | 163.55 | 1.47 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `resample_fallback` | 84.56 | 141.22 | 0.60 | 86.87 | 113.81 | 0.76 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `resample_real` | 22.42 | 73.99 | 0.30 | 21.79 | 63.40 | 0.34 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 84.65 | 70.61 | 1.20 | 75.11 | 56.90 | 1.32 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `resample_real_q15` | 26.11 | 103.68 | 0.25 | 31.59 | 144.23 | 0.22 | sdrk fallback |
| `resample_real_q15_fallback` | 103.68 | 68.34 | 1.52 | 144.23 | 81.78 | 1.76 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
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
| `dot_cq15` | 0.81 | 2.03 | 0.40 | 0.75 | 1.72 | 0.43 | sdrk fallback |
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
| `add` | 0.51 | 0.70 | 0.72 | 0.56 | 0.79 | 0.70 | volk_32fc_x2_add_32fc |
| `add_cq15` | 0.25 | 0.35 | 0.72 | 0.28 | 0.40 | 0.71 | volk_16i_x2_add_saturated_16i on the 2 n values |
| `add_cq15_fallback` | 0.35 | 2.28 | 0.15 | 0.46 | 1.46 | 0.31 | CMSIS-DSP 1.18.0 arm_add_q15 on the 2 n values, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `add_fallback` | 0.70 | 0.70 | 1.00 | 0.91 | 0.91 | 1.00 | CMSIS-DSP 1.18.0 arm_add_f32 on the 2 n floats, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `add_real` | 0.26 | 0.36 | 0.71 | 0.28 | 0.40 | 0.70 | volk_32f_x2_add_32f |
| `add_real_fallback` | 0.29 | 0.36 | 0.82 | 0.36 | 0.46 | 0.77 | CMSIS-DSP 1.18.0 arm_add_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `add_real_q15` | 0.13 | 0.18 | 0.71 | 0.14 | 0.20 | 0.70 | volk_16i_x2_add_saturated_16i |
| `add_real_q15_fallback` | 0.15 | 1.14 | 0.13 | 0.18 | 0.73 | 0.25 | CMSIS-DSP 1.18.0 arm_add_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `mul` | 1.52 | 1.55 | 0.98 | 1.19 | 1.55 | 0.77 | volk_32fc_x2_multiply_32fc |
| `mul_cq15` | 1.42 | 3.04 | 0.47 | 1.48 | 2.89 | 0.51 | sdrk fallback |
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
| `fft_1024` | 7.20 | 9.32 | 0.77 | 6.37 | 9.55 | 0.67 | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 31.40 | 25.79 | 1.22 | 30.63 | 28.60 | 1.07 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_4096` | 8.15 | 10.68 | 0.76 | 7.32 | 10.77 | 0.68 | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 37.57 | 30.53 | 1.23 | 36.80 | 32.90 | 1.12 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_64` | 4.46 | 8.57 | 0.52 | 3.95 | 8.08 | 0.49 | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 20.02 | 15.17 | 1.32 | 19.61 | 17.78 | 1.10 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_cq15_1024` | 6.27 | 52.83 | 0.12 | 7.29 | 63.00 | 0.12 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 5.39 | 52.83 | 0.10 | 6.16 | 63.00 | 0.10 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 45.87 | 44.80 | 1.02 | 46.95 | 56.34 | 0.83 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_cq15_1024_fallback` | 76.35 | 44.80 | 1.70 | 77.17 | 56.34 | 1.37 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `fft_cq15_1024_full_scale` | 5.83 | 52.83 | 0.11 | 6.27 | 63.00 | 0.10 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 4.71 | 5.63 | 0.84 | 4.42 | 4.58 | 0.96 | FFTW (fftwf_plan_dft_r2c_1d, FFTW_ESTIMATE, forward, out of place) |
| `rfft_1024_fallback` | 16.64 | 13.39 | 1.24 | 16.45 | 14.58 | 1.13 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `am_mod` | 1.09 | 1.09 | 1.00 | 1.16 | 1.16 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 0.31 | 1.23 | 0.25 | 0.37 | 1.15 | 0.32 | sdrk fallback |
| `am_mod_cq15_fallback` | 1.23 | 3.17 | 0.39 | 1.15 | 3.60 | 0.32 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `am_mod_fallback` | 1.09 | 0.59 | 1.84 | 1.16 | 0.72 | 1.61 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 233.12 | 233.12 | 1.00 | 207.68 | 207.68 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 57.28 | 57.28 | 1.00 | 56.56 | 56.56 | 1.00 | sdrk fallback |
| `channel_bank_channelize` | 45.77 | 45.77 | 1.00 | 52.53 | 52.53 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 227.20 | 227.20 | 1.00 | 212.46 | 212.46 | 1.00 | sdrk fallback |
| `channel_select_block` | 0.60 | 0.60 | 1.00 | 0.77 | 0.77 | 1.00 | sdrk fallback |
| `channelize` | 21.95 | 171.63 | 0.13 | 28.21 | 176.29 | 0.16 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 26.51 | 26.51 | 1.00 | 27.84 | 27.84 | 1.00 | sdrk fallback |
| `channelize_fallback` | 53.83 | 166.73 | 0.32 | 58.12 | 180.51 | 0.32 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `costas` | 115.06 | 115.06 | 1.00 | 98.55 | 98.55 | 1.00 | sdrk fallback |
| `costas_fallback` | 115.06 | 86.69 | 1.33 | 98.55 | 80.46 | 1.22 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 13.55 | 15.15 | 0.89 | 11.55 | 13.14 | 0.88 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 8.06 | 8.15 | 0.99 | 7.06 | 7.14 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 13.05 | 15.15 | 0.86 | 11.06 | 13.14 | 0.84 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 8.06 | 8.15 | 0.99 | 7.06 | 7.15 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 146.43 | 146.43 | 1.00 | 157.82 | 157.82 | 1.00 | sdrk fallback |
| `equalize_fallback` | 146.43 | 168.47 | 0.87 | 157.82 | 201.30 | 0.78 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 6.25 | 6.25 | 1.00 | 8.50 | 8.50 | 1.00 | sdrk fallback |
| `frame_decode_block` | 6.06 | 6.06 | 1.00 | 7.96 | 7.96 | 1.00 | sdrk fallback |
| `frame_encode_block` | 1.86 | 1.86 | 1.00 | 1.82 | 1.82 | 1.00 | sdrk fallback |
| `halfband_decimate` | 6.77 | 59.03 | 0.11 | 7.75 | 50.48 | 0.15 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its SSE dot product (-O3 -ffp-contract=off, sse4.2) |
| `halfband_decimate_cq15` | 13.12 | 23.94 | 0.55 | 13.02 | 21.81 | 0.60 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 23.94 | 38.80 | 0.62 | 21.81 | 47.25 | 0.46 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `halfband_decimate_fallback` | 15.73 | 82.14 | 0.19 | 17.79 | 69.57 | 0.26 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, sse4.2) |
| `halfband_decimate_real` | 3.68 | 53.52 | 0.07 | 4.01 | 48.28 | 0.08 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its SSE (SSE3 horizontal add) dot product (-O3 -ffp-contract=off, sse4.2) |
| `halfband_decimate_real_fallback` | 12.87 | 68.61 | 0.19 | 13.53 | 55.10 | 0.25 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse4.2) |
| `halfband_decimate_real_q15` | 6.20 | 16.50 | 0.38 | 5.37 | 16.82 | 0.32 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 16.50 | 19.39 | 0.85 | 16.82 | 23.61 | 0.71 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `halfband_ref_fir_decimate` | 13.07 | 29.62 | 0.44 | 12.41 | 29.82 | 0.42 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 19.30 | 60.59 | 0.32 | 19.67 | 86.10 | 0.23 | sdrk fallback |
| `halfband_ref_shifted` | 9.50 | 18.49 | 0.51 | 11.27 | 21.34 | 0.53 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 16.03 | 26.84 | 0.60 | 17.17 | 25.98 | 0.66 | sdrk fallback |
| `morse_decode_block` | 16.62 | 16.62 | 1.00 | 20.30 | 20.30 | 1.00 | sdrk fallback |
| `morse_encode_block` | 6.60 | 6.60 | 1.00 | 7.58 | 7.58 | 1.00 | sdrk fallback |
| `noise_blank` | 15.92 | 15.92 | 1.00 | 11.93 | 11.93 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 9.59 | 9.59 | 1.00 | 10.59 | 10.59 | 1.00 | sdrk fallback |
| `preemph_limit` | 16.10 | 16.10 | 1.00 | 11.44 | 11.44 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 51.64 | 51.64 | 1.00 | 43.84 | 43.84 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 20.85 | 20.85 | 1.00 | 21.65 | 21.65 | 1.00 | sdrk fallback |
| `rds_decode_block` | 8.74 | 8.74 | 1.00 | 11.50 | 11.50 | 1.00 | sdrk fallback |
| `rds_demod_block` | 44.51 | 44.50 | 1.00 | 39.67 | 39.68 | 1.00 | sdrk fallback |
| `rds_shape` | 78.95 | 78.95 | 1.00 | 71.83 | 71.83 | 1.00 | sdrk fallback |
| `real_to_complex` | 5.34 | 55.36 | 0.10 | 5.44 | 50.25 | 0.11 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its SSE (SSE3 horizontal add) dot product (-O3 -ffp-contract=off, sse4.2) |
| `real_to_complex_cq15` | 7.55 | 10.87 | 0.69 | 6.96 | 13.90 | 0.50 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, sse4.2) |
| `real_to_complex_cq15_fallback` | 22.69 | 10.87 | 2.09 | 23.68 | 13.90 | 1.70 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, sse4.2) |
| `real_to_complex_fallback` | 17.36 | 70.45 | 0.25 | 19.91 | 57.08 | 0.35 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, sse4.2) |
| `stereo_encode` | 10.15 | 10.15 | 1.00 | 10.88 | 10.88 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 14.16 | 14.16 | 1.00 | 15.14 | 15.14 | 1.00 | sdrk fallback |
| `stereo_matrix` | 1.02 | 2.10 | 0.48 | 0.99 | 2.48 | 0.40 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.51 | 1.52 | 0.34 | 0.53 | 0.83 | 0.64 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 1.52 | 3.43 | 0.44 | 0.83 | 2.20 | 0.38 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `stereo_matrix_fallback` | 1.52 | 1.07 | 1.42 | 1.46 | 1.38 | 1.06 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `stereo_pll_fallback` | 88.29 | 71.63 | 1.23 | 77.34 | 63.55 | 1.22 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, sse4.2) |
| `stereo_pll_q15_fallback` | 38.04 | 55.57 | 0.68 | 38.06 | 58.48 | 0.65 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, sse4.2) |
| `symbol_sync` | 49.82 | 49.82 | 1.00 | 38.10 | 38.10 | 1.00 | sdrk fallback |
| `symbol_sync_real` | 23.29 | 23.29 | 1.00 | 18.29 | 18.29 | 1.00 | sdrk fallback |

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
| `convert_out_real_rs8` | 0.23 | 0.25 | 0.89 | 0.29 | 0.26 | 1.10 | volk_32f_s32f_convert_8i |
| `convert_out_real_rs8_fallback` | 0.54 | 0.26 | 2.06 | 0.52 | 0.32 | 1.64 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_out_real_ru8` | 0.28 | 0.28 | 1.00 | 0.34 | 0.30 | 1.13 | volk_32f_s32f_x2_convert_8u |
| `convert_out_real_ru8_fallback` | 0.84 | 0.26 | 3.22 | 0.92 | 0.32 | 2.91 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `convert_real_rs16` | 0.14 | 0.17 | 0.84 | 0.14 | 0.19 | 0.73 | volk_16i_s32f_convert_32f |
| `convert_real_rs16_fallback` | 0.14 | 0.20 | 0.69 | 0.15 | 0.20 | 0.74 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `rotate` | 0.66 | 3.19 | 0.21 | 0.57 | 3.05 | 0.19 | volk_32fc_s32fc_x2_rotator2_32fc |
| `rotate_cq15` | 1.12 | 4.49 | 0.25 | 1.20 | 5.49 | 0.22 | sdrk fallback |
| `rotate_cq15_fallback` | 4.49 | 29.18 | 0.15 | 5.49 | 25.57 | 0.21 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `rotate_fallback` | 2.65 | 10.64 | 0.25 | 4.41 | 11.16 | 0.39 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `fir` | 9.63 | 58.05 | 0.17 | 8.85 | 72.63 | 0.12 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 40.62 | 40.62 | 1.00 | 61.15 | 61.15 | 1.00 | sdrk fallback |
| `fir_cq15_fallback` | 40.62 | 144.83 | 0.28 | 61.15 | 154.65 | 0.40 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_decimate` | 2.41 | 9.29 | 0.26 | 2.09 | 12.60 | 0.17 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 21.62 | 21.62 | 1.00 | 22.06 | 22.06 | 1.00 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 21.62 | 42.02 | 0.51 | 22.06 | 17.52 | 1.26 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_decimate_fallback` | 11.63 | 71.28 | 0.16 | 11.54 | 32.88 | 0.35 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_fallback` | 31.56 | 51.44 | 0.61 | 30.50 | 80.42 | 0.38 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate` | 12.38 | 66.84 | 0.19 | 11.36 | 76.70 | 0.15 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 52.45 | 52.45 | 1.00 | 71.49 | 71.49 | 1.00 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 52.45 | 136.94 | 0.38 | 71.49 | 136.97 | 0.52 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate_fallback` | 41.06 | 101.68 | 0.40 | 38.06 | 101.03 | 0.38 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate_real` | 7.36 | 83.60 | 0.09 | 6.78 | 93.79 | 0.07 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 25.86 | 50.84 | 0.51 | 24.05 | 50.51 | 0.48 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fir_interpolate_real_q15` | 42.12 | 42.12 | 1.00 | 38.06 | 38.06 | 1.00 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 42.12 | 68.47 | 0.62 | 38.06 | 68.49 | 0.56 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `resample` | 14.85 | 68.15 | 0.22 | 14.01 | 77.87 | 0.18 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 117.54 | 117.54 | 1.00 | 113.21 | 113.21 | 1.00 | sdrk fallback |
| `resample_cq15_fallback` | 117.54 | 136.18 | 0.86 | 113.21 | 136.22 | 0.83 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `resample_fallback` | 64.91 | 100.83 | 0.64 | 61.67 | 100.27 | 0.62 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `resample_real` | 9.82 | 84.38 | 0.12 | 9.13 | 94.98 | 0.10 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 61.16 | 50.42 | 1.21 | 69.86 | 50.13 | 1.39 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `resample_real_q15` | 62.01 | 112.75 | 0.55 | 68.67 | 99.55 | 0.69 | sdrk fallback |
| `resample_real_q15_fallback` | 112.75 | 68.09 | 1.66 | 99.55 | 68.11 | 1.46 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mag` | 0.76 | 0.91 | 0.84 | 0.76 | 0.78 | 0.97 | volk_32fc_magnitude_32f |
| `mag_fallback` | 0.90 | 1.15 | 0.78 | 0.78 | 0.83 | 0.94 | CMSIS-DSP 1.18.0 arm_cmplx_mag_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mag_q15` | 0.97 | 1.06 | 0.92 | 0.98 | 0.92 | 1.07 | volk_16ic_magnitude_16i |
| `mag_q15_fallback` | 1.93 | 79.45 | 0.02 | 1.30 | 83.02 | 0.02 | CMSIS-DSP 1.18.0 arm_cmplx_mag_q15 (Q2.14, arm_sqrt_q31), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `mag_sq` | 0.22 | 0.33 | 0.67 | 0.26 | 0.52 | 0.50 | volk_32fc_magnitude_squared_32f |
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
| `power_db_fallback` | 2.76 | 38.52 | 0.07 | 2.08 | 42.51 | 0.05 | CMSIS-DSP 1.18.0 arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `power_db_q15` | 1.89 | 3.07 | 0.61 | 1.89 | 2.46 | 0.77 | sdrk fallback |
| `power_db_q15_fallback` | 3.07 | 39.25 | 0.08 | 2.46 | 43.32 | 0.06 | CMSIS-DSP 1.18.0 arm_q15_to_float + arm_vlog_f32 + arm_scale_f32 (+ offset loop), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
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
| `biquad_cq15` | 9.16 | 21.19 | 0.43 | 10.57 | 21.26 | 0.50 | sdrk fallback |
| `biquad_cq15_fallback` | 21.19 | 54.13 | 0.39 | 21.26 | 56.47 | 0.38 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `biquad_fallback` | 18.46 | 24.04 | 0.77 | 20.84 | 26.04 | 0.80 | CMSIS-DSP 1.18.0 arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `biquad_real` | 9.12 | 10.30 | 0.89 | 10.06 | 9.56 | 1.05 | KFR 7.2.0 kfr::iir, iir_state<float, 2> (avx2, -ffp-contract=off) |
| `biquad_real_fallback` | 10.13 | 18.04 | 0.56 | 11.77 | 20.05 | 0.59 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df2T_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `biquad_real_q15` | 8.18 | 11.11 | 0.74 | 10.21 | 12.14 | 0.84 | sdrk fallback |
| `biquad_real_q15_fallback` | 11.11 | 27.07 | 0.41 | 12.14 | 28.24 | 0.43 | CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot` | 0.26 | 0.27 | 0.97 | 0.26 | 0.52 | 0.50 | volk_32fc_32f_dot_prod_32fc |
| `dot_complex` | 0.46 | 0.78 | 0.59 | 0.58 | 0.78 | 0.74 | VOLK 3.3.0 volk_32fc_x2_dot_prod_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off) |
| `dot_complex_fallback` | 1.94 | 6.03 | 0.32 | 1.68 | 6.03 | 0.28 | CMSIS-DSP 1.18.0 arm_cmplx_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot_cq15` | 0.30 | 1.04 | 0.29 | 0.39 | 0.77 | 0.51 | sdrk fallback |
| `dot_cq15_fallback` | 1.04 | 1.07 | 0.97 | 0.77 | 1.03 | 0.75 | CMSIS-DSP 1.18.0 arm_dot_prod_q15 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot_fallback` | 3.03 | 6.07 | 0.50 | 3.03 | 6.07 | 0.50 | CMSIS-DSP 1.18.0 arm_dot_prod_f32 twice, on the planar re and im, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot_real` | 0.14 | 0.52 | 0.26 | 0.10 | 0.77 | 0.13 | volk_32f_x2_dot_prod_32f |
| `dot_real_fallback` | 3.03 | 3.04 | 1.00 | 3.03 | 3.04 | 1.00 | CMSIS-DSP 1.18.0 arm_dot_prod_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `dot_real_q15` | 0.13 | 0.32 | 0.40 | 0.16 | 0.38 | 0.42 | sdrk fallback |
| `dot_real_q15_fallback` | 0.32 | 0.54 | 0.59 | 0.38 | 0.52 | 0.74 | CMSIS-DSP 1.18.0 arm_dot_prod_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `scale` | 0.26 | 0.27 | 0.94 | 0.19 | 0.29 | 0.68 | volk_32f_s32f_multiply_32f |
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
| `interleave_cq15_fallback` | 0.13 | 0.14 | 0.97 | 0.13 | 0.14 | 0.96 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic on int16 (VOLK has no 16-bit interleave), benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx2) |
| `interleave_fallback` | 0.26 | 0.27 | 0.98 | 0.51 | 0.52 | 0.98 | VOLK 3.3.0 volk_32f_x2_interleave_32fc_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx2) |
| `deinterleave` | 0.26 | 0.39 | 0.66 | 0.26 | 0.40 | 0.65 | volk_32fc_deinterleave_32f_x2 |
| `deinterleave_cq15` | 0.17 | 0.26 | 0.63 | 0.20 | 0.26 | 0.74 | volk_16ic_deinterleave_16i_x2 |
| `deinterleave_cq15_fallback` | 0.23 | 0.20 | 1.14 | 0.23 | 0.26 | 0.86 | VOLK 3.3.0 volk_16ic_deinterleave_16i_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx2) |
| `deinterleave_fallback` | 0.26 | 1.02 | 0.26 | 0.26 | 0.52 | 0.50 | VOLK 3.3.0 volk_32fc_deinterleave_32f_x2_generic, benches/refs/volk-generic.c (-O3 -ffp-contract=off, avx2) |
| `oscillate` | 1.40 | 2.27 | 0.62 | 1.45 | 4.03 | 0.36 | sdrk fallback |
| `oscillate_cq15` | 1.90 | 4.51 | 0.42 | 1.49 | 4.80 | 0.31 | sdrk fallback |
| `oscillate_cq15_fallback` | 4.51 | 16.77 | 0.27 | 4.80 | 16.78 | 0.29 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample (512-entry table, linear interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `oscillate_fallback` | 2.27 | 31.51 | 0.07 | 4.03 | 80.72 | 0.05 | CMSIS-DSP 1.18.0 arm_sin_cos_f32 of the angle in degrees per sample (512-entry sine table, cubic Hermite interpolation), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_1024` | 2.33 | 2.70 | 0.86 | 2.26 | 3.38 | 0.67 | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 18.43 | 20.03 | 0.92 | 18.87 | 19.56 | 0.96 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_4096` | 2.60 | 3.37 | 0.77 | 2.51 | 3.83 | 0.66 | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 21.89 | 23.30 | 0.94 | 22.36 | 22.33 | 1.00 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_64` | 1.65 | 2.04 | 0.81 | 1.82 | 1.89 | 0.96 | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 12.39 | 12.51 | 0.99 | 12.85 | 12.47 | 1.03 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_cq15_1024` | 2.47 | 46.26 | 0.05 | 2.79 | 48.58 | 0.06 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 2.31 | 46.26 | 0.05 | 2.34 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 28.16 | 37.69 | 0.75 | 30.91 | 38.63 | 0.80 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_cq15_1024_fallback` | 46.25 | 37.69 | 1.23 | 51.39 | 38.63 | 1.33 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `fft_cq15_1024_full_scale` | 2.44 | 46.26 | 0.05 | 2.45 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 2.10 | 2.71 | 0.78 | 2.08 | 2.82 | 0.74 | realfft on rustfft (RealFftPlanner, forward, out of place) |
| `rfft_1024_fallback` | 9.71 | 9.77 | 0.99 | 9.95 | 9.84 | 1.01 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `am_mod` | 0.65 | 0.65 | 1.00 | 0.47 | 0.47 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 0.20 | 0.43 | 0.47 | 0.21 | 0.56 | 0.37 | sdrk fallback |
| `am_mod_cq15_fallback` | 0.43 | 2.91 | 0.15 | 0.56 | 2.30 | 0.24 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `am_mod_fallback` | 0.65 | 0.53 | 1.22 | 0.47 | 0.54 | 0.87 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 164.98 | 164.98 | 1.00 | 164.85 | 164.85 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 46.83 | 46.83 | 1.00 | 42.65 | 42.65 | 1.00 | sdrk fallback |
| `channel_bank_channelize` | 21.00 | 21.00 | 1.00 | 25.61 | 25.61 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 72.75 | 72.75 | 1.00 | 65.45 | 65.45 | 1.00 | sdrk fallback |
| `channel_select_block` | 0.76 | 0.76 | 1.00 | 0.52 | 0.52 | 1.00 | sdrk fallback |
| `channelize` | 12.08 | 120.69 | 0.10 | 12.75 | 99.98 | 0.13 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 12.21 | 12.21 | 1.00 | 12.87 | 12.87 | 1.00 | sdrk fallback |
| `channelize_fallback` | 34.97 | 149.61 | 0.23 | 32.37 | 131.39 | 0.25 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `costas` | 81.08 | 81.08 | 1.00 | 104.05 | 104.05 | 1.00 | sdrk fallback |
| `costas_fallback` | 81.08 | 54.40 | 1.49 | 104.05 | 32.47 | 3.20 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 11.81 | 13.17 | 0.90 | 12.80 | 16.19 | 0.79 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 7.05 | 10.39 | 0.68 | 8.05 | 9.40 | 0.86 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 11.04 | 13.17 | 0.84 | 12.04 | 16.19 | 0.74 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 7.05 | 10.39 | 0.68 | 8.05 | 9.40 | 0.86 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 114.56 | 114.56 | 1.00 | 115.63 | 115.63 | 1.00 | sdrk fallback |
| `equalize_fallback` | 114.56 | 930.66 | 0.12 | 115.63 | 220.13 | 0.53 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 5.56 | 5.56 | 1.00 | 6.17 | 6.17 | 1.00 | sdrk fallback |
| `frame_decode_block` | 6.89 | 6.89 | 1.00 | 5.78 | 5.78 | 1.00 | sdrk fallback |
| `frame_encode_block` | 1.64 | 1.64 | 1.00 | 1.72 | 1.72 | 1.00 | sdrk fallback |
| `halfband_decimate` | 3.40 | 87.24 | 0.04 | 2.97 | 34.43 | 0.09 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its AVX dot product (-O3 -ffp-contract=off, avx2) |
| `halfband_decimate_cq15` | 6.55 | 12.86 | 0.51 | 7.12 | 13.27 | 0.54 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 12.86 | 30.91 | 0.42 | 13.27 | 33.32 | 0.40 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `halfband_decimate_fallback` | 61.26 | 68.43 | 0.90 | 13.84 | 62.30 | 0.22 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, avx2) |
| `halfband_decimate_real` | 2.04 | 65.21 | 0.03 | 1.38 | 32.32 | 0.04 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its AVX dot product (-O3 -ffp-contract=off, avx2) |
| `halfband_decimate_real_fallback` | 9.11 | 80.05 | 0.11 | 8.76 | 50.71 | 0.17 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, avx2) |
| `halfband_decimate_real_q15` | 2.95 | 11.33 | 0.26 | 3.83 | 10.11 | 0.38 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 11.33 | 15.45 | 0.73 | 10.11 | 16.66 | 0.61 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `halfband_ref_fir_decimate` | 5.53 | 21.86 | 0.25 | 5.48 | 21.37 | 0.26 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 41.00 | 41.00 | 1.00 | 39.65 | 39.65 | 1.00 | sdrk fallback |
| `halfband_ref_shifted` | 5.76 | 63.70 | 0.09 | 5.43 | 16.31 | 0.33 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 9.33 | 15.65 | 0.60 | 10.20 | 16.35 | 0.62 | sdrk fallback |
| `morse_decode_block` | 12.78 | 12.78 | 1.00 | 16.09 | 16.09 | 1.00 | sdrk fallback |
| `morse_encode_block` | 4.44 | 4.44 | 1.00 | 4.49 | 4.49 | 1.00 | sdrk fallback |
| `noise_blank` | 11.58 | 11.58 | 1.00 | 11.37 | 11.37 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 8.48 | 8.48 | 1.00 | 9.30 | 9.30 | 1.00 | sdrk fallback |
| `preemph_limit` | 15.37 | 15.37 | 1.00 | 13.18 | 13.18 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 27.62 | 27.62 | 1.00 | 18.69 | 18.69 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 16.71 | 16.71 | 1.00 | 22.32 | 22.32 | 1.00 | sdrk fallback |
| `rds_decode_block` | 7.28 | 7.28 | 1.00 | 7.98 | 7.98 | 1.00 | sdrk fallback |
| `rds_demod_block` | 16.68 | 16.68 | 1.00 | 15.21 | 15.21 | 1.00 | sdrk fallback |
| `rds_shape` | 50.41 | 50.41 | 1.00 | 78.60 | 78.60 | 1.00 | sdrk fallback |
| `real_to_complex` | 2.50 | 91.94 | 0.03 | 2.29 | 35.30 | 0.06 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its AVX dot product (-O3 -ffp-contract=off, avx2) |
| `real_to_complex_cq15` | 3.60 | 7.27 | 0.50 | 4.87 | 8.83 | 0.55 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, avx2) |
| `real_to_complex_cq15_fallback` | 14.91 | 7.27 | 2.05 | 14.50 | 8.83 | 1.64 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, avx2) |
| `real_to_complex_fallback` | 12.67 | 106.82 | 0.12 | 12.75 | 53.74 | 0.24 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, avx2) |
| `stereo_encode` | 4.47 | 4.47 | 1.00 | 5.92 | 5.92 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 7.17 | 7.17 | 1.00 | 8.91 | 8.91 | 1.00 | sdrk fallback |
| `stereo_matrix` | 0.32 | 1.26 | 0.26 | 0.45 | 1.45 | 0.31 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.15 | 0.39 | 0.39 | 0.20 | 0.54 | 0.37 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 0.39 | 1.55 | 0.25 | 0.54 | 1.55 | 0.35 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `stereo_matrix_fallback` | 0.63 | 0.60 | 1.06 | 0.86 | 0.53 | 1.61 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `stereo_pll_fallback` | 63.32 | 57.06 | 1.11 | 79.41 | 78.06 | 1.02 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, avx2) |
| `stereo_pll_q15_fallback` | 37.04 | 51.46 | 0.72 | 38.06 | 34.89 | 1.09 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx2) |
| `symbol_sync` | 31.52 | 31.52 | 1.00 | 43.74 | 43.74 | 1.00 | sdrk fallback |
| `symbol_sync_real` | 15.66 | 15.66 | 1.00 | 21.81 | 21.81 | 1.00 | sdrk fallback |

- Not measured at this level: `real_to_complex_airspy`.

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
| `convert_out_real_ru8` | 0.32 | 0.26 | 1.21 | 0.27 | 0.30 | 0.91 | volk_32f_s32f_x2_convert_8u |
| `convert_out_real_ru8_fallback` | 0.61 | 0.19 | 3.17 | 0.67 | 0.31 | 2.17 | CMSIS-DSP 1.18.0 arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `convert_real_rs16` | 0.13 | 0.13 | 1.01 | 0.11 | 0.11 | 0.96 | volk_16i_s32f_convert_32f |
| `convert_real_rs16_fallback` | 0.14 | 0.15 | 0.94 | 0.15 | 0.20 | 0.74 | CMSIS-DSP 1.18.0 arm_q15_to_float, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `rotate` | 0.71 | 3.89 | 0.18 | 0.44 | 3.36 | 0.13 | volk_32fc_s32fc_x2_rotator2_32fc |
| `rotate_cq15` | 0.79 | 4.59 | 0.17 | 3.22 | 5.64 | 0.57 | sdrk fallback |
| `rotate_cq15_fallback` | 4.59 | 27.49 | 0.17 | 5.64 | 25.81 | 0.22 | CMSIS-DSP 1.18.0 arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `rotate_fallback` | 1.05 | 10.63 | 0.10 | 9.30 | 11.17 | 0.83 | GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags) |
| `fir` | 8.65 | 63.03 | 0.14 | 4.82 | 97.97 | 0.05 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 40.60 | 40.60 | 1.00 | 45.49 | 45.49 | 1.00 | sdrk fallback |
| `fir_cq15_fallback` | 40.60 | 144.59 | 0.28 | 45.49 | 154.66 | 0.29 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_decimate` | 2.05 | 8.83 | 0.23 | 1.61 | 15.25 | 0.11 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 11.57 | 16.80 | 0.69 | 11.17 | 18.22 | 0.61 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 16.80 | 37.21 | 0.45 | 18.22 | 20.44 | 0.89 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_decimate_fallback` | 11.13 | 71.32 | 0.16 | 10.28 | 38.89 | 0.26 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_fallback` | 31.12 | 51.41 | 0.61 | 30.50 | 80.41 | 0.38 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate` | 11.12 | 71.37 | 0.16 | 7.49 | 102.01 | 0.07 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 54.36 | 54.36 | 1.00 | 63.17 | 63.17 | 1.00 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 54.36 | 136.84 | 0.40 | 63.17 | 136.97 | 0.46 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate_fallback` | 39.38 | 101.17 | 0.39 | 38.06 | 100.78 | 0.38 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate_real` | 6.69 | 81.93 | 0.08 | 4.79 | 118.66 | 0.04 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 25.50 | 50.59 | 0.50 | 24.05 | 50.39 | 0.48 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fir_interpolate_real_q15` | 22.72 | 22.72 | 1.00 | 29.26 | 29.26 | 1.00 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 22.72 | 68.42 | 0.33 | 29.26 | 68.49 | 0.43 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `resample` | 13.69 | 71.87 | 0.19 | 12.54 | 102.76 | 0.12 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 67.78 | 84.07 | 0.81 | 69.75 | 89.12 | 0.78 | sdrk fallback |
| `resample_cq15_fallback` | 84.07 | 136.08 | 0.62 | 89.12 | 136.22 | 0.65 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `resample_fallback` | 60.87 | 100.42 | 0.61 | 55.22 | 100.10 | 0.55 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `resample_real` | 9.17 | 82.64 | 0.11 | 8.89 | 119.58 | 0.07 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 40.74 | 50.21 | 0.81 | 243.32 | 50.05 | 4.86 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `resample_real_q15` | 32.80 | 101.47 | 0.32 | 36.74 | 85.53 | 0.43 | sdrk fallback |
| `resample_real_q15_fallback` | 101.47 | 68.04 | 1.49 | 85.53 | 68.11 | 1.26 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
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
| `mul_real` | 0.10 | 0.12 | 0.83 | 0.07 | 0.09 | 0.71 | volk_32f_x2_multiply_32f |
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
| `fft_1024` | 2.44 | 2.62 | 0.93 | 1.80 | 3.27 | 0.55 | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 17.03 | 17.82 | 0.96 | 16.75 | 18.53 | 0.90 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_4096` | 2.70 | 3.03 | 0.89 | 1.93 | 3.71 | 0.52 | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 20.57 | 20.57 | 1.00 | 20.23 | 21.21 | 0.95 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_64` | 1.58 | 1.82 | 0.87 | 1.63 | 1.82 | 0.89 | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 10.91 | 11.76 | 0.93 | 10.73 | 11.39 | 0.94 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_cq15_1024` | 2.32 | 45.37 | 0.05 | 2.65 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 2.08 | 45.37 | 0.05 | 2.34 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 28.66 | 37.40 | 0.77 | 31.56 | 39.49 | 0.80 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_cq15_1024_fallback` | 42.83 | 37.40 | 1.15 | 50.08 | 39.49 | 1.27 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `fft_cq15_1024_full_scale` | 2.14 | 45.37 | 0.05 | 2.40 | 48.58 | 0.05 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 1.93 | 2.57 | 0.75 | 1.61 | 2.83 | 0.57 | realfft on rustfft (RealFftPlanner, forward, out of place) |
| `rfft_1024_fallback` | 9.02 | 8.63 | 1.05 | 8.87 | 9.37 | 0.95 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `am_mod` | 0.38 | 0.38 | 1.00 | 0.30 | 0.30 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 0.14 | 0.32 | 0.43 | 0.20 | 0.48 | 0.43 | sdrk fallback |
| `am_mod_cq15_fallback` | 0.32 | 2.82 | 0.11 | 0.48 | 2.37 | 0.20 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `am_mod_fallback` | 0.38 | 0.53 | 0.73 | 0.30 | 0.54 | 0.56 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 161.09 | 161.09 | 1.00 | 163.36 | 163.36 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 52.29 | 52.29 | 1.00 | 47.86 | 47.86 | 1.00 | sdrk fallback |
| `channel_bank_channelize` | 792.49 | 792.49 | 1.00 | 97.78 | 97.78 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 516.23 | 516.13 | 1.00 | 96.78 | 96.79 | 1.00 | sdrk fallback |
| `channel_select_block` | 0.28 | 0.28 | 1.00 | 1.83 | 1.83 | 1.00 | sdrk fallback |
| `channelize` | 12.68 | 104.12 | 0.12 | 16.01 | 84.10 | 0.19 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 12.69 | 12.69 | 1.00 | 16.13 | 16.13 | 1.00 | sdrk fallback |
| `channelize_fallback` | 38.10 | 107.09 | 0.36 | 32.11 | 118.99 | 0.27 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `costas` | 84.05 | 84.05 | 1.00 | 102.05 | 102.05 | 1.00 | sdrk fallback |
| `costas_fallback` | 84.05 | 68.92 | 1.22 | 102.05 | 66.44 | 1.54 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 11.81 | 13.17 | 0.90 | 12.80 | 16.19 | 0.79 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 7.05 | 10.38 | 0.68 | 8.04 | 9.40 | 0.86 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 11.04 | 13.17 | 0.84 | 12.04 | 16.19 | 0.74 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 7.05 | 10.38 | 0.68 | 8.04 | 9.40 | 0.86 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 107.26 | 107.26 | 1.00 | 113.93 | 113.93 | 1.00 | sdrk fallback |
| `equalize_fallback` | 107.26 | 929.60 | 0.12 | 113.93 | 220.19 | 0.52 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 5.68 | 5.68 | 1.00 | 6.17 | 6.17 | 1.00 | sdrk fallback |
| `frame_decode_block` | 6.88 | 6.88 | 1.00 | 5.78 | 5.78 | 1.00 | sdrk fallback |
| `frame_encode_block` | 1.60 | 1.60 | 1.00 | 1.72 | 1.72 | 1.00 | sdrk fallback |
| `halfband_decimate` | 2.59 | 107.90 | 0.02 | 3.45 | 34.07 | 0.10 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its AVX-512F dot product (-O3 -ffp-contract=off, avx512) |
| `halfband_decimate_cq15` | 7.14 | 11.94 | 0.60 | 8.54 | 13.27 | 0.64 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 11.94 | 30.34 | 0.39 | 13.27 | 33.28 | 0.40 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `halfband_decimate_fallback` | 60.73 | 118.25 | 0.51 | 12.98 | 65.16 | 0.20 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, avx512) |
| `halfband_decimate_real` | 1.64 | 27.50 | 0.06 | 1.08 | 20.10 | 0.05 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its AVX-512F dot product (-O3 -ffp-contract=off, avx512) |
| `halfband_decimate_real_fallback` | 8.91 | 78.71 | 0.11 | 8.78 | 52.09 | 0.17 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, avx512) |
| `halfband_decimate_real_q15` | 3.82 | 11.68 | 0.33 | 4.51 | 12.40 | 0.36 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 11.68 | 15.19 | 0.77 | 12.40 | 16.66 | 0.74 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `halfband_ref_fir_decimate` | 7.56 | 20.89 | 0.36 | 7.77 | 18.96 | 0.41 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 30.90 | 27.20 | 1.14 | 28.35 | 28.95 | 0.98 | sdrk fallback |
| `halfband_ref_shifted` | 4.96 | 63.16 | 0.08 | 5.90 | 15.45 | 0.38 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 9.92 | 14.74 | 0.67 | 11.62 | 16.34 | 0.71 | sdrk fallback |
| `morse_decode_block` | 12.75 | 12.75 | 1.00 | 17.03 | 17.03 | 1.00 | sdrk fallback |
| `morse_encode_block` | 4.20 | 4.20 | 1.00 | 4.64 | 4.64 | 1.00 | sdrk fallback |
| `noise_blank` | 11.45 | 11.45 | 1.00 | 11.33 | 11.33 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 8.41 | 8.41 | 1.00 | 9.25 | 9.25 | 1.00 | sdrk fallback |
| `preemph_limit` | 19.13 | 19.13 | 1.00 | 12.17 | 12.17 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 26.71 | 26.71 | 1.00 | 18.69 | 18.69 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 15.64 | 15.64 | 1.00 | 13.07 | 13.07 | 1.00 | sdrk fallback |
| `rds_decode_block` | 7.20 | 7.20 | 1.00 | 7.96 | 7.96 | 1.00 | sdrk fallback |
| `rds_demod_block` | 481.99 | 481.54 | 1.00 | 51.33 | 51.32 | 1.00 | sdrk fallback |
| `rds_shape` | 42.06 | 42.06 | 1.00 | 39.42 | 39.42 | 1.00 | sdrk fallback |
| `real_to_complex` | 1.99 | 53.98 | 0.04 | 1.57 | 23.02 | 0.07 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its AVX-512F dot product (-O3 -ffp-contract=off, avx512) |
| `real_to_complex_cq15` | 4.28 | 220.09 | 0.02 | 5.03 | 25.92 | 0.19 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, avx512) |
| `real_to_complex_cq15_fallback` | 14.45 | 220.92 | 0.07 | 15.62 | 25.98 | 0.60 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, avx512) |
| `real_to_complex_fallback` | 12.02 | 105.23 | 0.11 | 12.12 | 55.01 | 0.22 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, avx512) |
| `stereo_encode` | 1.15 | 1.15 | 1.00 | 5.53 | 5.53 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 1.59 | 1.59 | 1.00 | 5.54 | 5.54 | 1.00 | sdrk fallback |
| `stereo_matrix` | 0.35 | 0.85 | 0.42 | 0.32 | 1.07 | 0.30 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.20 | 0.39 | 0.51 | 0.13 | 0.58 | 0.23 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 0.39 | 0.79 | 0.49 | 0.58 | 1.55 | 0.37 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `stereo_matrix_fallback` | 0.48 | 0.60 | 0.80 | 0.54 | 0.53 | 1.02 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `stereo_pll_fallback` | 61.34 | 56.30 | 1.09 | 74.40 | 78.06 | 0.95 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, avx512) |
| `stereo_pll_q15_fallback` | 37.04 | 45.61 | 0.81 | 38.05 | 34.89 | 1.09 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, avx512) |
| `symbol_sync` | 28.82 | 28.82 | 1.00 | 39.81 | 39.81 | 1.00 | sdrk fallback |
| `symbol_sync_real` | 14.40 | 14.40 | 1.00 | 19.92 | 19.92 | 1.00 | sdrk fallback |

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
| `fir` | 37.96 | 126.85 | 0.30 | 24.33 | 99.19 | 0.25 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it |
| `fir_cq15` | 106.10 | 107.72 | 0.98 | 92.51 | 98.30 | 0.94 | sdrk fallback |
| `fir_cq15_fallback` | 107.72 | 180.05 | 0.60 | 98.30 | 181.88 | 0.54 | CMSIS-DSP 1.18.0 arm_fir_q15, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_decimate` | 9.89 | 20.50 | 0.48 | 9.08 | 16.88 | 0.54 | volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it |
| `fir_decimate_cq15` | 26.40 | 29.57 | 0.89 | 23.88 | 27.63 | 0.86 | sdrk fallback |
| `fir_decimate_cq15_fallback` | 29.57 | 16.40 | 1.80 | 27.63 | 16.90 | 1.63 | CMSIS-DSP 1.18.0 arm_fir_decimate_q15 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_decimate_fallback` | 32.75 | 34.10 | 0.96 | 25.70 | 41.23 | 0.62 | CMSIS-DSP 1.18.0 arm_fir_decimate_f32 (block 4090 of 4096), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_fallback` | 124.44 | 132.43 | 0.94 | 84.01 | 133.90 | 0.63 | CMSIS-DSP 1.18.0 arm_fir_f32, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate` | 42.06 | 138.57 | 0.30 | 28.58 | 113.64 | 0.25 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it |
| `fir_interpolate_cq15` | 75.80 | 127.87 | 0.59 | 76.00 | 145.98 | 0.52 | sdrk fallback |
| `fir_interpolate_cq15_fallback` | 127.87 | 141.59 | 0.90 | 145.98 | 141.30 | 1.03 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate_fallback` | 143.96 | 135.30 | 1.06 | 102.07 | 152.44 | 0.67 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate_real` | 23.09 | 107.89 | 0.21 | 23.33 | 85.10 | 0.27 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it |
| `fir_interpolate_real_fallback` | 82.78 | 67.65 | 1.22 | 62.36 | 76.22 | 0.82 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fir_interpolate_real_q15` | 73.09 | 73.09 | 1.00 | 73.96 | 73.96 | 1.00 | sdrk fallback |
| `fir_interpolate_real_q15_fallback` | 73.09 | 70.80 | 1.03 | 73.96 | 70.65 | 1.05 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 2, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `resample` | 55.28 | 140.12 | 0.39 | 50.99 | 114.43 | 0.45 | volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it |
| `resample_cq15` | 144.07 | 193.79 | 0.74 | 122.69 | 184.92 | 0.66 | sdrk fallback |
| `resample_cq15_fallback` | 193.79 | 140.28 | 1.38 | 184.92 | 139.74 | 1.32 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `resample_fallback` | 175.90 | 134.66 | 1.31 | 142.04 | 151.50 | 0.94 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `resample_real` | 32.35 | 109.41 | 0.30 | 29.36 | 85.87 | 0.34 | volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it |
| `resample_real_fallback` | 139.69 | 67.33 | 2.07 | 121.24 | 75.75 | 1.60 | CMSIS-DSP 1.18.0 arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `resample_real_q15` | 97.68 | 113.70 | 0.86 | 78.50 | 107.71 | 0.73 | sdrk fallback |
| `resample_real_q15_fallback` | 113.70 | 70.14 | 1.62 | 107.71 | 69.87 | 1.54 | CMSIS-DSP 1.18.0 arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
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
| `fft_1024` | 12.84 | 17.22 | 0.75 | 8.55 | 13.55 | 0.63 | rustfft (FftPlanner, forward, in place) |
| `fft_1024_fallback` | 33.24 | 38.54 | 0.86 | 34.77 | 39.16 | 0.89 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_4096` | 15.06 | 20.37 | 0.74 | 10.02 | 16.16 | 0.62 | rustfft (FftPlanner, forward, in place) |
| `fft_4096_fallback` | 37.38 | 39.40 | 0.95 | 39.15 | 43.25 | 0.91 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_64` | 8.85 | 13.02 | 0.68 | 7.05 | 10.68 | 0.66 | rustfft (FftPlanner, forward, in place) |
| `fft_64_fallback` | 21.69 | 22.44 | 0.97 | 20.29 | 24.31 | 0.83 | CMSIS-DSP 1.18.0 arm_cfft_f32 (forward, in place, bit reversal), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_cq15_1024` | 11.77 | 67.26 | 0.17 | 10.10 | 77.63 | 0.13 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default` | 10.18 | 67.26 | 0.15 | 8.48 | 77.63 | 0.11 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `fft_cq15_1024_default_fallback` | 50.87 | 64.42 | 0.79 | 51.01 | 52.33 | 0.97 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_cq15_1024_fallback` | 120.91 | 64.42 | 1.88 | 100.37 | 52.33 | 1.92 | CMSIS-DSP 1.18.0 arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `fft_cq15_1024_full_scale` | 10.93 | 67.26 | 0.16 | 9.01 | 77.63 | 0.12 | KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place) |
| `rfft_1024` | 8.81 | 10.88 | 0.81 | 5.72 | 10.04 | 0.57 | realfft on rustfft (RealFftPlanner, forward, out of place) |
| `rfft_1024_fallback` | 18.28 | 18.65 | 0.98 | 17.01 | 18.61 | 0.91 | CMSIS-DSP 1.18.0 arm_rfft_fast_f32 (forward, packed output), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `am_mod` | 2.91 | 2.91 | 1.00 | 1.52 | 1.52 | 1.00 | sdrk fallback |
| `am_mod_cq15` | 0.90 | 1.72 | 0.52 | 0.77 | 1.33 | 0.58 | sdrk fallback |
| `am_mod_cq15_fallback` | 1.72 | 4.30 | 0.40 | 1.33 | 4.49 | 0.30 | CMSIS-DSP 1.18.0 arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `am_mod_fallback` | 2.91 | 1.28 | 2.27 | 1.52 | 1.03 | 1.48 | liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3) |
| `auto_notch` | 263.70 | 263.70 | 1.00 | 208.69 | 208.69 | 1.00 | sdrk fallback |
| `band_pulse_detect_block` | 88.06 | 88.06 | 1.00 | 70.36 | 70.36 | 1.00 | sdrk fallback |
| `channel_bank_channelize` | 60.07 | 60.07 | 1.00 | 42.71 | 42.71 | 1.00 | sdrk fallback |
| `channel_bank_ddc` | 308.18 | 308.18 | 1.00 | 263.36 | 263.36 | 1.00 | sdrk fallback |
| `channel_select_block` | 1.27 | 1.27 | 1.00 | 1.27 | 1.27 | 1.00 | sdrk fallback |
| `channelize` | 30.93 | 302.17 | 0.10 | 22.59 | 240.53 | 0.09 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10 |
| `channelize_block` | 30.99 | 30.99 | 1.00 | 22.71 | 22.71 | 1.00 | sdrk fallback |
| `channelize_fallback` | 98.73 | 222.34 | 0.44 | 81.02 | 165.45 | 0.49 | liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT |
| `costas` | 117.06 | 117.06 | 1.00 | 77.05 | 77.05 | 1.00 | sdrk fallback |
| `costas_fallback` | 117.06 | 82.40 | 1.42 | 77.05 | 75.07 | 1.03 | GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags) |
| `crc16_ccitt_false` | 10.13 | 10.16 | 1.00 | 7.10 | 9.15 | 0.78 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc16_x25` | 8.11 | 8.17 | 0.99 | 7.10 | 7.15 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc24_adsb` | 8.10 | 10.16 | 0.80 | 7.09 | 9.15 | 0.78 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `crc32` | 8.11 | 8.17 | 0.99 | 7.10 | 7.16 | 0.99 | GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags) |
| `equalize` | 247.72 | 247.72 | 1.00 | 191.23 | 191.23 | 1.00 | sdrk fallback |
| `equalize_fallback` | 247.72 | 228.84 | 1.08 | 191.23 | 236.11 | 0.81 | GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags) |
| `frame_decode` | 8.83 | 8.83 | 1.00 | 10.38 | 10.38 | 1.00 | sdrk fallback |
| `frame_decode_block` | 8.83 | 8.83 | 1.00 | 9.55 | 9.55 | 1.00 | sdrk fallback |
| `frame_encode_block` | 2.10 | 2.10 | 1.00 | 1.96 | 1.96 | 1.00 | sdrk fallback |
| `halfband_decimate` | 13.17 | 84.75 | 0.16 | 12.58 | 83.63 | 0.15 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its NEON dot product (-O3 -ffp-contract=off, neon) |
| `halfband_decimate_cq15` | 16.35 | 29.04 | 0.56 | 15.22 | 25.78 | 0.59 | sdrk fallback |
| `halfband_decimate_cq15_fallback` | 29.04 | 39.92 | 0.73 | 25.78 | 40.69 | 0.63 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `halfband_decimate_fallback` | 35.63 | 93.29 | 0.38 | 26.81 | 74.24 | 0.36 | liquid-dsp 1.8.3 resamp2_crcf_decim_execute (dotprod_crcf) with its portable dot product (-O3 -ffp-contract=off, neon) |
| `halfband_decimate_real` | 7.75 | 84.61 | 0.09 | 6.79 | 75.02 | 0.09 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its NEON dot product (-O3 -ffp-contract=off, neon) |
| `halfband_decimate_real_fallback` | 25.66 | 78.52 | 0.33 | 21.30 | 58.40 | 0.36 | liquid-dsp 1.8.3 resamp2_rrrf_decim_execute (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, neon) |
| `halfband_decimate_real_q15` | 7.67 | 22.82 | 0.34 | 6.49 | 21.15 | 0.31 | sdrk fallback |
| `halfband_decimate_real_q15_fallback` | 22.82 | 19.75 | 1.16 | 21.15 | 20.11 | 1.05 | CMSIS-DSP 1.18.0 arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `halfband_ref_fir_decimate` | 19.74 | 63.98 | 0.31 | 17.93 | 50.98 | 0.35 | sdrk fallback |
| `halfband_ref_fir_decimate_cq15` | 78.40 | 60.69 | 1.29 | 75.20 | 56.59 | 1.33 | sdrk fallback |
| `halfband_ref_shifted` | 18.84 | 41.31 | 0.46 | 19.08 | 33.38 | 0.57 | sdrk fallback |
| `halfband_ref_shifted_cq15` | 23.39 | 36.10 | 0.65 | 21.74 | 32.32 | 0.67 | sdrk fallback |
| `morse_decode_block` | 22.03 | 22.03 | 1.00 | 28.42 | 28.42 | 1.00 | sdrk fallback |
| `morse_encode_block` | 8.35 | 8.35 | 1.00 | 9.45 | 9.45 | 1.00 | sdrk fallback |
| `noise_blank` | 14.61 | 14.61 | 1.00 | 8.80 | 8.80 | 1.00 | sdrk fallback |
| `noise_blank_cq15` | 9.51 | 9.51 | 1.00 | 10.53 | 10.53 | 1.00 | sdrk fallback |
| `preemph_limit` | 15.43 | 15.43 | 1.00 | 15.41 | 15.41 | 1.00 | sdrk fallback |
| `preemph_limit_q15` | 55.11 | 55.11 | 1.00 | 40.70 | 40.70 | 1.00 | sdrk fallback |
| `preemph_limit_stereo` | 24.12 | 24.12 | 1.00 | 22.90 | 22.90 | 1.00 | sdrk fallback |
| `rds_decode_block` | 12.14 | 12.14 | 1.00 | 13.67 | 13.67 | 1.00 | sdrk fallback |
| `rds_demod_block` | 55.36 | 55.36 | 1.00 | 49.36 | 49.36 | 1.00 | sdrk fallback |
| `rds_shape` | 53.93 | 53.93 | 1.00 | 47.22 | 47.22 | 1.00 | sdrk fallback |
| `real_to_complex` | 10.15 | 86.75 | 0.12 | 8.59 | 76.87 | 0.11 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its NEON dot product (-O3 -ffp-contract=off, neon) |
| `real_to_complex_cq15` | 10.69 | 17.90 | 0.60 | 8.74 | 17.14 | 0.51 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, neon) |
| `real_to_complex_cq15_fallback` | 33.36 | 17.90 | 1.86 | 31.07 | 17.14 | 1.81 | libairspy 1.0.10 iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved), conversion without remove_dc (-O3 -ffp-contract=off, neon) |
| `real_to_complex_fallback` | 31.17 | 80.68 | 0.39 | 30.71 | 60.25 | 0.51 | liquid-dsp 1.8.3 firhilbf_decim_execute_block (dotprod_rrrf) with its portable dot product (-O3 -ffp-contract=off, neon) |
| `stereo_encode` | 15.13 | 15.13 | 1.00 | 16.08 | 16.08 | 1.00 | sdrk fallback |
| `stereo_encode_rds` | 21.13 | 21.13 | 1.00 | 22.07 | 22.07 | 1.00 | sdrk fallback |
| `stereo_matrix` | 1.90 | 5.74 | 0.33 | 1.39 | 5.23 | 0.27 | VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory) |
| `stereo_matrix_cq15` | 0.77 | 0.77 | 1.00 | 0.76 | 0.76 | 1.00 | sdrk fallback |
| `stereo_matrix_cq15_fallback` | 0.77 | 2.70 | 0.28 | 0.76 | 2.31 | 0.33 | CMSIS-DSP 1.18.0 arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `stereo_matrix_fallback` | 1.90 | 3.83 | 0.50 | 1.39 | 3.07 | 0.45 | CMSIS-DSP 1.18.0 arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data, generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `stereo_pll_fallback` | 84.51 | 86.93 | 0.97 | 56.44 | 57.23 | 0.99 | liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, neon) |
| `stereo_pll_q15_fallback` | 41.07 | 100.31 | 0.41 | 39.08 | 88.00 | 0.44 | CMSIS-DSP 1.18.0 arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL), generic C (-O3 -ffp-contract=off, LOOPUNROLL, neon) |
| `symbol_sync` | 47.13 | 47.13 | 1.00 | 28.58 | 28.58 | 1.00 | sdrk fallback |
| `symbol_sync_real` | 20.41 | 20.41 | 1.00 | 12.03 | 12.03 | 1.00 | sdrk fallback |

- Not measured at this level: `real_to_complex_airspy`.

<!-- cycle-tables:end -->

Notes.

- **Comparisons per level.** The `sse2` FFT is judged against FFTW out of place (D3). In place, FFTW needs 31.5 / 35.1 / 40.7 instructions per sample at N = 64 / 1024 / 4096 (ours 0.48 / 0.72 / 0.71 of that), and rustfft, which has no planner below SSE4.1, is scalar there (56.8 / 93.1 / 112.2). On `neon` FFTW (`--enable-neon`, out of place) is information: 15.9 / 26.8 / 39.7. The `CQ15` FFT is judged against KissFFT, whose fixed-point code is scalar on every platform; Ne10 int16 on `neon` needs 19.8 instructions per sample at 44.6 dB (ours 0.80 with block float, 0.65 with fixed scaling). The cheapest VOLK implementation on `neon` is the default dispatch for every kernel but `convert_cs16` (`neon`, 2.52), `fir`, `fir_decimate`, `fir_interpolate` and `resample` (`neon_unroll`, 150.0, 26.7, 169.5 and 171.4), `mag_sq` (`neon`, 1.52), `quad_demod` (19.3) and `convert_out` to `CU8` (`neon`, 4.02); ours is below all of them (`convert_out` to `CU8` 0.99 of the cheapest).
- **Run-time detection** (`avx2-runtime`) costs at most 0.02 times the instructions of the fixed `avx2` build for every accelerated kernel but `fft` C32 at N = 64 (1.07, 6.28 against 5.84 instructions per sample: about 28 instructions more per call, 18 of them in `level::dispatch_fn!`; information under D68; against rustfft it is 1.07 as well, 5.88) and the short kernels of issue #20 with a fixed dispatch cost of a few dozen instructions per call: `dot_real` f32 1.03, `dot` CQ15 1.02 and `scale` CQ15 1.04 (`dev`), `window` CQ15 1.04 (`info`, 0.78 against 0.76), and `fir` CQ15 1.04 (`dev`, the one-output-at-a-time branch of the fallback round, 237.74 against 228.84). The `f64` block scans of `dc_block_real` f32, `deemphasis` f32 and `dc_block` C32 (D53) are dispatched at run time like the other SIMD kernels: 1.03, 1.01 and 1.07 (`info`; before, as fallbacks, 1.31, 1.69 and 1.60). The kernels that stay with the fallback run its baseline code in the run-time build and the AVX2 build of the same code at `avx2` (`info`): `squelch` C32 1.07 and CQ15 1.06, `agc` CQ15 1.05, `dc_block_real` Q15 1.25, `deemphasis` Q15 0.93, `dc_block` CQ15 0.78 (the short forms of the recursions of D40 are selected per target feature, so the baseline build may run the other form).
- **`convert` to `CQ15` and `convert_real`.** `CS8`/`CU8` to `CQ15` widen and shift in the generic code, four vectors per iteration (two before: 0.72 / 0.84 instructions per sample at `avx2`, 0.49 / 0.62 now); `convert_real` of `RS16` (and of real `U8`/`I8` formats) converts the values in pairs through `convert` to `C32`/`CQ15`. VOLK has no unsigned 8-bit input, so `convert_cq15_cu8` is compared with `volk_8i_convert_16i` on the flipped bytes, and since the fallback round the wasm32 build of VOLK has `8i_convert_16i` too (its `generic` protokernel), so both `CQ15` entries are judged against it at `simd128` as well (0.59 and 0.71). The generic code cuts input and output to their common length first (the remainders of the two chunk iterators then start at the same value; with unequal lengths, which the trait calls never pass, the vectors after the main loop went to the wrong place), which also lets the compiler drop a bound: 0.46 / 0.58 instead of 0.49 / 0.62 instructions per sample at `avx2`, 0.59 / 0.84 instead of 0.65 / 0.90 on NEON, 2.84 / 3.37 instead of 3.06 / 3.59 at `simd128`, with the same simulated cycles at `avx2`. The fallback's direct loops for the byte and 16-bit formats (see [kernels.md](kernels.md#convert-and-convert_real)) replace the value-by-value decoder for these formats: `CU8` to `C32` at `avx2` from 88.04 to 1.47 instructions per sample. The entries `convert*_fallback` (D16) compare these loops with CMSIS-DSP's `arm_q7_to_float`, `arm_q15_to_float` and `arm_q7_to_q15` (`CU8` on the flipped bytes, as for VOLK); all are bit-exact and within 1.07 at every level, except `convert_cq15_cu8_fallback` on NEON (0.92 against 0.76 instructions per sample, 1.20, `dev`: the offset of `(2u - 255) << 7` is one vector operation more than `x << 8`). At `avx2` they need 0.42 / 0.42 / 0.51 / 0.21 / 0.28 / 0.26 `znver3` cycles per sample for `CS8` / `CS16` / `CU8` / `RS16` / `CS8` and `CU8` to `CQ15` against 0.58 / 0.39 / 0.58 / 0.20 / 0.27 / 0.27 of CMSIS-DSP (`alderlake` 0.29 / 0.29 / 0.37 / 0.15 / 0.14 / 0.15 against 0.58 / 0.40 / 0.58 / 0.20 / 0.20 / 0.20); the gaps of at most 6 % on `znver3` against 26 % to 31 % ahead on `alderlake` were left as they are.
- **`convert_out`.** Rounds to nearest even like VOLK (decision D7) and keeps NaN as zero; that costs one compare and one AND per vector at SSE/AVX2 (VOLK does not look at NaN), and nothing on NEON and SIMD128. The `U8` encoding reproduces the f64 rounding exactly, which costs one byte bound per value more than VOLK at `sse2`, `sse4.2`, `avx2` and `neon` (a pack and two minimums on x86, a compare and a narrowing on NEON) and a floor and a compare at `avx512`. At `sse2` that is 1.35 times VOLK (the same 1.36 for `convert_out_real` to real `U8`, also `dev`; the cycles and the rounding trade-off are in issue #71); at `sse4.2` and `avx2` 1.05 and 1.06, within the release limit. VOLK's `volk_32f_s32f_x2_convert_8u` does less: it adds the bias in `f32` and rounds that sum, which is not the exact rule (D944: on 10^7 uniform values in [-1, 1) 48 bytes differ, on values within 2^-20 of a multiple of 1/128 4.2 %, every tiny negative value gives 128 instead of 127), and it has no NaN rule, so its cycles are a lower bound, not an equal reference. At `avx2` the `I16` and `I8` paths compare the input with itself for NaN beside the product and mask the converted integers, and cap with `vminps(scale, v)`, which keeps NaN: one load more per vector than the form that clears NaN from the product first (instructions 0.80 / 0.84 to 0.92 / 0.97 of VOLK), but the compare leaves the dependency chain, and the simulated cycles per sample fall from 0.53 to 0.45 (`CS16`) and from 0.52 to 0.43 (`CS8`) on `znver3`, from 0.57 to 0.55 and 0.56 on `alderlake` (ratios to VOLK 1.16 / 1.07 to 0.99 / 1.04 and 1.04 / 1.10 to 0.87 / 1.07). `convert_out_real` (`f32` and `Q15` to `RS16` and the real 8-bit formats) runs the same vector code as `convert_out`; its entries compare with the same VOLK kernels on N values. The fallback writes each container and layout in its own loop and rounds by adding `1.5 * 2^52` after the clamp: at `avx2` it vectorises and needs 9.9 / 9.1 / 9.0 instructions per sample for `CS16` / `CS8` / `CU8` (142.0 / 134.1 / 134.0 before) and 3.35 / 3.03 / 3.03 simulated cycles on `znver3` (40.9 / 37.5 / 26.4 before), 3.89 / 3.20 / 3.20 on `alderlake` (43.1 / 41.6 / 29.1). Since the fallback round the formats with one value per byte or 16-bit word go through direct loops in `f32` (see [kernels.md](kernels.md#convert_out-and-convert_out_real)): `x * 2^7` and `x * 2^15` are exact in `f32`, so NaN is cleared, the value clamped and rounded by adding `1.5 * 2^23`, whose low bits are the word; `U8` takes `floor` of `x * 2^7 + 2^-47` with the even neighbour below at integers (one compare against `0` or `2^-126` by the parity). All of them equal the `f64` formula for every `f32`. At `avx2` (whole-loop, `znver3` / `alderlake`) `CS16` / `CS8` / `CU8` from 3.06 / 3.09 / 3.09 to 0.77 / 1.06 / 1.67 cycles per sample and from 3.22 / 3.23 / 3.23 to 0.73 / 1.03 / 1.82 (2.48 / 3.79 / 5.79 instructions, 9.56 / 9.18 / 9.18 before), the real formats `RS16` / `RS8` / `RU8` from 1.25 / 1.27 / 1.27 to 0.39 / 0.54 / 0.84 and from 1.35 / 1.49 / 1.49 to 0.37 / 0.52 / 0.92. Spot checks (per block): `sse2` `CS16` 7.01 / 5.89 to 2.05 / 2.04 (`skylake` / `znver2`), `CU8` 7.59 / 6.19 to 5.43 / 5.74, `RU8` 3.09 / 2.80 to 2.73 / 2.88 (`znver2` 3 % slower); `avx512` `CU8` 1.65 / 1.85 to 1.31 / 1.33 (`znver4` / `sapphirerapids`) with more instructions (4.18 to 4.79); NEON `CS16` 14.95 / 7.85 to 4.09 / 2.56 (`cortex-a72` / `cortex-a76`), `CU8` 14.66 / 8.44 to 7.14 / 4.35; `simd128` `CS16` / `CU8` 39.35 / 37.59 to 17.57 / 26.57 instructions per sample. The clamp uses `f32::max` and `f32::min` on x86 and AArch64 and a NaN-free sum of the clamped positive and negative parts elsewhere (`f32::max` on wasm32 follows NaN and took 69.07 instead of 17.57 instructions for `CS16`). On bare-metal Arm and RISC-V the integer path stays. The entries `convert_out*_fallback` compare with CMSIS-DSP's `arm_float_to_q15` / `arm_float_to_q7`, which truncate (89.4 / 41.3 dB against 95.5 / 47.3 dB of ours) and for `CU8` have no offset: `info`. The ratio is 1.23 / 1.82 / 2.78 instructions and 1.65 / 2.06 / 3.24 `znver3` cycles at `avx2` for `CS16` / `CS8` / `CU8`; the rounding to even, the NaN rule and the exact `U8` offset are the difference.
- **`rotate`.** The block oscillator needs 2.37 instructions per sample at `avx2` against 5.28 for VOLK 3.3 (and 3.07 for the VOLK 3.1 of Ubuntu 24.04) and is more accurate (142.7 dB against 115.0 dB). Its per-call cost (about 450 instructions at `avx2`) shows in short calls: 9.1 instructions per sample at n = 64 and 32.6 at n = 16. The vectors of a chain are `A W^k` times the powers `E^r` (rounded once from `f64`) instead of steps `E v` from vector to vector, except on AArch64: simulated cycles per sample (whole-loop), before / after, `avx2` 0.69 / 0.67 (`znver3`, now throughput-bound, efficiency 0.81) and 0.56 / 0.57 (`alderlake`) against VOLK's 3.57 / 3.06; `sse2` 2.30 / 2.09 (`skylake`) and 2.27 / 1.97 (`znver2`); `avx512` 0.66 / 0.67 (`znver4`) and 0.56 / 0.56 (`sapphirerapids`); on the Cortex-A76 the powers took 2.12 instead of 1.97, so NEON keeps the steps. Golden SNR 142.3 to 145.0 dB before, 144.9 to 145.2 dB after. The fallback (entry `rotate_fallback`) is gated against GNU Radio's scalar rotator: 13.40 against 61.10 instructions per sample at `avx2` (GNU Radio's complex products follow the C99 rules with the NaN checks), 2.65 / 4.41 against 10.64 / 11.16 cycles; on NEON 26.01 against 28.09 instructions (0.93).
- **`rotate` `CQ15`.** No established library has a fixed-point rotator (VOLK and GNU Radio rotate in float only, CMSIS-DSP has no NCO, IPP is not available), so every level is `info` against our fallback and judged by its efficiency (D32). Each level has its own path (with `nco-tables`; SIMD128 also without): the complex product with `pmaddwd` (`i32x4.dot_i16x8_s`, NEON `smull`/`smlsl`/`smlal` with `sqrshrn`) instead of four 32-bit multiplications and the clamp, and at `sse2`, `sse4.2` and `avx2` the entry offsets through one store and scalar loads instead of `movd`/`pextrd` (on Zen two micro-ops each on the ports of the vector loads, the bound of the loop). Simulated cycles per sample, ours / fallback: `avx2` 1.12 / 6.96 (`znver3`) and 1.20 / 4.78 (`alderlake`), efficiency 0.87 / 0.89; `sse2` 3.81 / 12.10 (`skylake`) and 3.89 / 12.38 (`znver2`); `sse4.2` 2.78 / 9.56 and 2.54 / 10.38; `avx512` 0.75 / 7.59 (`znver4`) and 3.22 / 4.23 (`sapphirerapids`); `neon` 4.50 / 14.12 (`cortex-a72`) and 3.66 / 15.09 (`cortex-a76`). At `avx512` the entries still come from two 64-bit gathers: the 4096-entry table does not fit in registers, and the gather-free form (two AVX2 lookups) takes 1.17 instead of 3.22 cycles on `sapphirerapids` but 1.41 instead of 0.75 on `znver4`, the reference model. The fallback computes the interpolation and the complex product in `i32` (exact, see `fallback/mix.rs`): 15.85 instead of 26.53 instructions per sample at `avx2`. It is gated (entry `rotate_cq15_fallback`) against CMSIS-DSP's building blocks as a CMSIS-DSP user composes them (no NCO there): `arm_cos_q15`/`arm_sin_q15` per sample, `arm_cmplx_mult_cmplx_q15` and the shift back to `Q15` per 64 samples (a 512-entry table, 70.9 dB against our 91.2 dB): 15.85 against 101.60 instructions per sample at `avx2`, 6.96 / 4.78 against 29.23 / 26.70 cycles (`znver3` / `alderlake`; 121.3 / 27.0 before D64, which rated the `endbr64` at the entry of every called library function at about 51 cycles on `znver3`). The same composition is shown as information beside the SIMD entries. At `sse2` the fallback's output clamp compiles to two branches per component (LLVM's cmov conversion, not at `avx2`, where the loop is vectorized); with 12 % saturating samples in the measurement they mispredict on roughly every eighth sample. A branch-free clamp (`sat32(2 w) >> 16` with `cmovo`) simulates at 12.47 / 13.43 instead of 12.10 / 12.12 cycles (`skylake` / `znver2`, which do not model mispredictions); it pays off above about 2 % (`skylake`) and 8 % (`znver2`) saturating samples, so the branches stay.
- **`fir_interpolate`, `resample`.** Every branch of the polyphase filter runs as a contiguous filter in the forms of `fir` (`M = 1`) and `fir_decimate`, with its outputs written at the stride `L`; the comparison makes one VOLK dot product per output with the taps of its branch, as GNU Radio does. The counts are per output sample.
- **`fir_interpolate_real`, `resample_real`.** The same polyphase code as the complex kernels with one component per sample (the `f32` forms hold twice as many samples per vector; AVX-512 runs the chunk form of `resample_real` on 8-lane vectors). The measurements use the taps and ratios of the complex entries (L = 2 with 128 taps; 3 / 2 with 192 taps). `f32` is compared with `volk_32f_x2_dot_prod_32f_a` per output as GNU Radio's `interp_fir_filter_fff` and `rational_resampler_fff` call it; on `simd128` the comparison is the portable `generic` `volk_32f_x2_dot_prod_32f` of the wasm32 build of VOLK (`info`, D12; round 2: 147.91 against 947.03 fuel per sample for `fir_interpolate_real`, 179.99 against 949.03 for `resample_real`). `Q15` has no established library (VOLK has no 16-bit real dot product): the comparison is our fallback at every level (`info`), and the branch computes one output at a time with exact `i64` sums, bit-exact with the fallback.
- **`rfft`.** The inner complex FFT of `N / 2` points is the backend's. The post-processing forms 4 bins and their mirrors per vector at `sse2`, `sse4.2`, `neon` and `simd128`; at `avx2` and `avx512` the fallback's loop, compiled at the level, is cheaper. At `sse2` and `sse4.2` the comparison is FFTW's real-input transform (15.32 instructions per sample at both levels): ours is 1.07 times that (1.09 before the fused radix-2 pass of D49), within the release limit (the fallback's loop was 1.22 times). rustfft's real planner needs 50.96 at `sse2` (scalar) and 24.54 at `sse4.2`.
- **`fir`, `fir_decimate`.** The comparison is GNU Radio's `fir_filter_ccf` (one aligned VOLK dot product per output, D6). Ours is at most 0.2 dB below the comparison (`fir` at `avx2`, 142.1 against 142.3 dB) and above it elsewhere, within the 1 dB rule.
- **FIR family fallbacks against CMSIS-DSP (issue #20, round 2).** Every fallback of the family has an entry `<kernel>_fallback` against CMSIS-DSP 1.18.0 generic C compiled for the level (`arm_fir_*`, D16), and the `CQ15` kernels have the entries `fir_cq15`, `fir_decimate_cq15`, `fir_interpolate_cq15` and `resample_cq15` (`info` against our fallback: no SIMD library has a `CQ15` FIR with real `Q15` taps). The work of CMSIS-DSP per output sample: complex signals run its real filter twice, on the planar `re` and `im` of the same input (deinterleaved once outside the count, its best case); `fir_decimate` by 10 runs on 4090 of the 4096 samples (a block must be a multiple of the factor); `fir_interpolate` by 2 runs `arm_fir_interpolate_*` on 2048 samples (4096 outputs, 64 taps each). CMSIS-DSP has no rational resampler: the `resample` entries count `arm_fir_interpolate_*` by 3 with the same 192 taps per output sample (each output a 64-tap dot product, as ours), but its outputs of one phase have consecutive windows while those of the 3 / 2 resampler are 2 samples apart, so these entries are `info`. The `Q15` functions of CMSIS-DSP truncate (`>> 15`, about 80 dB against the f64 model) where ours round exactly (bit-exact with the reference). Their sums are as exact as ours (`q63_t`, one saturation per output); the rounding costs one add per output, so it does not explain an instruction gap.
  - **The fallback engine** (`sdrk-core/src/fallback/decim.rs`, shared by all of them): the outputs of one polyphase branch whose windows lie in the input are computed in groups of 8 (16 for `f32`/`Q15` with consecutive windows), each output with its own accumulator, taps ascending, no FMA, so the bits are those of the one-at-a-time sum (unit tests against the definition for every split, phase, tail and the `Q15` extremes); with consecutive windows the products of a tap are an element-wise product with consecutive samples that the compiler vectorizes, with windows `M` samples apart one slice per output. The three forms (contiguous taps, polyphase rows, strided windows) are separate functions: inlined into one, they changed each other's register allocation (`fir` 32 against 50 cycles per sample at `avx2`). The old fallback summed one output at a time through chained `StepBy` iterators: 740 to 990 instructions per output at 64 taps per branch. The per-tap step of the strided form is a separate `#[inline(always)]` function: as a closure the compiler kept it out of line at every level, one call per tap (inlined, `fir_decimate` CQ15 at `sse2` 178.89 to 118.11 instructions per sample; per-block cycles on `skylake` / `znver2` 39.73 / 54.35 to 30.71 / 45.22, `resample` CQ15 219.85 / 290.87 to 177.08 / 245.07; on `neon` 39.50 / 35.40 to 37.85 / 34.16 on `cortex-a72` / `cortex-a76` with more instructions, 83.33 to 92.54).
  - **`avx2`, whole-loop cycles per sample `znver3` / `alderlake`, ours before and after against CMSIS-DSP:** `fir` C32 197.9 / 197.4 and 31.5 / 30.3 against 51.4 / 80.4; `fir` CQ15 267.4 / 541.6 and 66.0 / 60.7 against 145.4 / 154.8; `fir_decimate` C32 40.6 / 41.1 and 11.6 / 11.6 against 71.3 / 32.9; `fir_decimate` CQ15 53.8 / 107.2 and 20.1 / 17.5 against 42.0 / 17.5; `fir_interpolate` C32 202.0 / 201.7 and 41.1 / 38.1 against 102.0 / 101.0; `fir_interpolate` CQ15 269.4 / 522.2 and 79.4 / 73.5 against 137.9 / 152.9; `fir_interpolate_real` f32 201.8 / 201.7 and 25.9 / 24.1 against 51.0 / 50.5; `fir_interpolate_real` Q15 235.9 / 392.0 and 47.6 / 41.9 against 68.9 / 76.5; `resample` C32 202.4 / 202.1 and 64.9 / 61.7 against 101.1 / 100.3; `resample` CQ15 269.6 / 523.4 and 109.9 / 91.0 against 136.8 / 152.2; `resample_real` f32 202.3 / 202.2 and 61.2 / 74.5 against 50.5 / 50.1; `resample_real` Q15 236.0 / 392.9 and 112.8 / 91.4 against 68.4 / 76.1. Spot checks of the shared code (per block): `sse2` `fir` C32 391.9 / 329.0 to 44.0 / 52.8 cycles on `skylake` / `znver2` (CMSIS-DSP 133.0 / 111.0), `fir_interpolate_real` 365.1 / 303.6 to 48.1 / 46.0 (71.3 / 57.3); `neon` `fir` C32 328.0 / 186.6 to 123.7 / 90.1 on `cortex-a72` / `cortex-a76` (133.5 / 134.1), `fir_interpolate_real` 465.6 / 402.3 to 84.4 / 63.2 (84.0 / 76.4), `fir` CQ15 373.6 / 371.1 to 108.7 / 98.5 (180.8 / 181.4).
  - **`fir_decimate` `C32` accuracy.** 133.3 dB against 134.3 dB for `arm_fir_decimate_f32` on the count input at every level, more than the 1 dB of the rule, so the entry has an accuracy waiver: both sum each output in one `f32` accumulator without FMA, CMSIS-DSP from the oldest sample on (taps descending in SDRk's order), ours in the ascending order the numeric contract fixes; the grouped fallback has the bits of the one-at-a-time sum, so the fallback before it had the same 133.3 dB. The f64 golden vectors: 143.3 dB (`m10_n63`, optimum 152.1 dB) and 144.3 dB (`m3_n31`, optimum 151.9 dB).
  - **`Q15` fallbacks above CMSIS-DSP's instructions** (`dev` with a ratio waiver at `sse2`, `sse4.2` and `neon`, `waived` where above 1.50x, the comparison is fair, so these levels stay judged; the gaps in simulated cycles are tracked in issue #71). The cause is our code generation, not the accuracy class. `fir_interpolate` CQ15 and `fir_interpolate_real` Q15 (1.52 to 1.69 times): CMSIS-DSP's loop computes two outputs at a time and keeps the samples of four taps in registers, so each loaded tap and sample serves two multiply-adds; our group loads every sample again per tap and output and compiles to scalar code at these levels (x86-64: `movsx`, `imul`, `add` with copies and spills of 16 `i64` accumulators; `neon`: `ldrsh` and `smaddl`). `fir_decimate` CQ15: CMSIS-DSP's tap loop per output is a reduction the C compiler vectorizes (`pmullw`/`pmulhw`, widening, `paddq`; widening multiply-adds on `neon`), ours (windows `M` apart) stays scalar. At `avx2` and `avx512` all of them are within the limits, with fewer whole-loop `avx2` cycles than CMSIS-DSP.
  - **Open.** The resampler's windows `M` apart are slower than the contiguous form (`resample_real` above CMSIS-DSP's interpolator by 3); a `Q15` form for `sse2`, `sse4.2` and `neon` that reuses loaded samples across taps (interpolation) or vectorizes the strided group (decimation); the `Q15`/`CQ15` SIMD branches (one output at a time, latency-bound by the horizontal reduction of every output) take more cycles than the grouped fallback on both core models at `sse2` (all six), `avx2` (all but `resample_real`) and `avx512` (`fir`, `fir_interpolate`, `fir_interpolate_real`), and more instructions at `simd128` (`fir_interpolate_real`): these combinations run the fallback until a grouped SIMD branch replaces them (hotfix D65, table `runs_fallback` in `sdrk-backend-simd/src/fir.rs`; their entries read 1.00). The switch needs the fallback compiled for the level, so the run-time-detection build keeps the SIMD branch at `avx2` (`avx2-runtime` `dev` for `fir` CQ15, `fir_interpolate` CQ15 and `fir_interpolate_real` Q15).
- **AVX-512 `fir_interpolate_real`.** The transposed form of `f32x16` ran 14 vectors 16 samples apart per pass (224 outputs), more than the scratch window of 192 outputs that a run of `drive_strided` takes, so it never ran a whole pass and computed every vector alone. It now runs 12 vectors per pass (192 outputs): 27.45 to 19.00 instructions per sample, 7.78 / 7.13 to 6.96 / 6.67 cycles on `znver4` / `sapphirerapids` (SDE, per block; VOLK 131.9 / 97.1).
- **`quad_demod`.** At `sse2` and `sse4.2` VOLK has no SSE `atan2`; the comparison is the correctly rounded `generic` code (144.2 dB), 0.7 dB above ours, which is within the rule (decision D9). `CQ15` has no established reference (the entries compare with our fallback); since issue #20 it uses the single-division `atan2` of `C32` on every level but AVX-512 (`avx2` 4.89 / 4.73 to 4.40 / 4.01 cycles per sample on `znver3` / `alderlake`, the same golden SNR of 77.9 dB, 0.25 dB below the optimum). Since round 2 the `CQ15` SIMD entries are `info` (D32: no established library has a fixed-point discriminator in SIMD, and our fallback is no fair gate for a SIMD branch); efficiency verdict at `avx2`: 0.41 / 0.67 of the port bound (`znver3` / `alderlake`, latency-bound by the chain of one sample: the exact products, two conversions, the division and the polynomial).
  - **Fallback entries (round 2).** `quad_demod_fallback` judges the `C32` fallback against GNU Radio's scalar `quadrature_demod_cf` (`fast_atan2f`, 114.0 dB against our 143.6 dB; CMSIS-DSP's `arm_atan2_f32` chain, 136.6 dB, as information), `quad_demod_cq15_fallback` the `CQ15` fallback against CMSIS-DSP's chain of `arm_cmplx_mult_cmplx_q15`, `arm_atan2_q15` and `arm_scale_q15` (62.3 against our 81.2 dB). Instructions per sample, `C32`: 105.17 (1.32 of GNU Radio) at `sse2`, 104.48 (1.31) at `sse4.2`, 98.86 (1.14) at `avx2`, 97.28 (1.12) at `avx512`, 110.52 (2.14) at `neon`, 188.90 (1.22) on `simd128`, so the entry is `info` at every level for the accuracy class (reaching `fast_atan2f`'s count would mean giving up the contract's `libm::atan2f`, D34; CMSIS-DSP 1.65 to 1.77 on x86-64, 1.28 at `neon`, 1.38 at `simd128`, information); `CQ15`: 170.85 (0.64), 171.16 (0.65), 162.72 (0.64), 163.66 (0.65), 173.61 (0.70), 332.25 (0.55), `release` at every level. Simulated cycles at `avx2` (whole-loop, `znver3` / `alderlake`): `C32` 39.05 / 32.76, `CQ15` 71.20 / 66.50 (per block 68.61 / 77.25) against CMSIS-DSP's 85.94 / 64.56 (ratio 0.83 / 1.03, per block 0.80 / 1.20). Four blocks per sample of CMSIS-DSP start with an `endbr64` (`arm_atan2_q15`, the `arm_divide_q15` it calls, and `arm_abs_q15`, which `arm_divide_q15` calls twice with a block of one sample; the library is built with `-fcf-protection`); until D64 `znver3` rated each of these blocks at about 51 cycles (`arm_divide_q15` 52.20, `arm_atan2_q15` 51.09, `arm_abs_q15` 51.09 twice; with the marker as `nop` 13.51, 2.04 and 0.85), so CMSIS-DSP read 275.53 / 66.94 and ours 0.26 / 0.99. Cross-check: uiCA rates the same blocks with and without `endbr64` at 12.94 / 11.94, 2.09 / 2.00 and 1.25 / 1.00 cycles (Skylake; Tiger Lake 7.96 / 8.04, 2.06 / 2.00, 1.25 / 1.00), and OSACA's `ZEN3` model has no entry for `endbr64`, so on no other model does the instruction cost more than about one cycle. In simulated speed the `CQ15` fallback is about at parity with CMSIS-DSP; the gate is on instructions (0.55 to 0.70, `release`). The fallback code is unchanged in round 2. Rejected: an inline, branch-free form of `libm::atan2f` and `libm::atan2` (libm's five argument reductions as one form `(a t - b) / (a + b t)` with table constants, the same bits) simulated at 28.23 / 22.81 (`C32`) and 48.98 / 46.03 (`CQ15`, in three loops per block) and met GNU Radio's count, but it always pays the reduction division, which libm skips below `7/16`; libm's interval branches predict well on any real FM signal and mispredict only on a phase close to white noise, which is what the measurement input is near. Timed on the host (an AMD EPYC, 4096 samples, baseline build), the `C32` fallback went from 8.3 to 14.8 ns per sample on narrowband FM (steps up to 0.05 rad), from 9.4 to 15.0 on smooth FM and from 10.9 to 15.0 on wideband FM, and was faster only on random phase (24.0 to 15.4); `CQ15` was about 7 % slower on narrowband input in the baseline build, and the three-loop `CQ15` form simulated 2.1 / 1.9 times the per-block cycles of the one-loop libm form at `neon` (`cortex-a72` / `cortex-a76`). GNU Radio's simulated cycles (865.81 / 131.57 at `avx2`; 902.50 on `znver3` before D64, which counted the `endbr64` at the entry of `fast_atan2f` at 51 cycles) are an artifact of `llvm-mca` 23.1.1: the zero fill of its temporary is a `rep stosb` that `znver3` rates at 100 and `alderlake` at 8 cycles per byte (8 bytes per sample); without it about 62 / 65 cycles per sample.
- **FFT.** `C32` reads its twiddles from static per-span tables on x86-64 and AArch64 (feature `fft-tables`, [backends.md](backends.md)); wasm32 is measured without them (with them: 31.0 / 53.1 / 61.8 fuel per sample). The accuracy of the single-vector transform at N = 64 scatters by about 1.9 dB between levels (140.7 to 142.5 dB), with the comparisons between 140.2 and 142.3 dB. `CQ15`: the radix-4 transform is not bit-identical with the fallback (decision D10); on the bench input it is far more accurate than KissFFT (which rounds every stage of its fixed-point code). Where a radix-2 stage is left over (AVX2 at N = 256, 1024, 4096, the 512 points inside `rfft` 1024 at SSE and AVX-512), it runs in one pass with the first radix-4 stage, bit for bit as the two passes (AVX2 simulated cycles per sample on znver3 / alderlake: N = 1024 2.86 / 2.61 to 2.74 / 2.48, N = 4096 3.04 / 2.80 to 2.97 / 2.73). The fallback checks its twiddle table once per stage instead of once per butterfly and runs the `CQ15` stages with the output shift as a constant and a branch-free clamp, with the same results (simulated at `avx2`: `C32` N = 1024 34.7 / 36.3 to 31.4 / 32.6 cycles per sample; `CQ15` block float 148.4 / 167.3 to 103.0 / 91.3 and fixed scaling 112.1 / 111.5 to 104.7 / 88.1, against 50.5 for KissFFT, the scalar reference of the fallback). At AVX2 the blocks between the passes keep the lane order of the in-lane shuffles (0, 1, 4, 5, 2, 3, 6, 7) with the twiddles in that order (a second copy of the span tables, 40 832 bytes on x86-64): the last pass stores with two unpacks per block instead of four shuffles, bit for bit the same result (whole-loop cycles per sample on znver3 / alderlake: N = 1024 2.55 / 2.39 to 2.35 / 2.26, N = 4096 2.81 / 2.65 to 2.60 / 2.52, `rfft` 1024 2.22 / 2.15 to 2.12 / 2.09; N = 64 unchanged).
- **`fm_mod`.** On wasm32 it is measured without the NCO tables (feature `nco-tables`, off there by default): 34.78 fuel per sample, 27.77 with the tables. At `avx512` it uses gathers (see the caveat at the top). No established library has an FM modulator in SIMD (GNU Radio's `frequency_modulator_fc` is a scalar loop with `fmod` and the 10-bit fixed-point sine table of `gr::fxpt`, a different accuracy class: 96.7 dB against our 133.2 dB) or a fixed-point one, so both SIMD entries are `info` (against GNU Radio and our fallback for `C32`, against our fallback and CMSIS-DSP's `arm_cos_q15`/`arm_sin_q15` for `CQ15`) and judged by their efficiency. The fallbacks are gated: `fm_mod_fallback` against GNU Radio's block, `fm_mod_cq15_fallback` against CMSIS-DSP's sine and cosine of an integer phase accumulator. The fallback computes the phase steps of a run without a branch (vectorized), with a flag for values outside the fast range, and interpolates the `Q15` table in `i32`: instructions per sample at `avx2` 32.49 / 42.54 to 20.15 / 21.78 (`C32` / `CQ15`), whole-loop cycles `znver3` 6.78 / 11.85 to 5.34 / 8.71 and `alderlake` 8.75 / 11.51 to 6.42 / 6.06 (GNU Radio 105.8 / 25.5, CMSIS-DSP 112.2 / 20.9); `sse2` `skylake` / `znver2` 11.95 / 12.62 to 8.95 / 10.90 (`C32`) and 27.17 / 27.19 to 10.76 / 11.73 (`CQ15`); `avx512` `znver4` / `sapphirerapids` 4.41 / 8.60 to 2.58 / 5.79 and 12.53 / 9.24 to 10.23 / 6.19; NEON A72 / A76 18.44 / 17.35 to 18.66 / 16.38 and 19.46 / 19.35 to 18.82 / 15.70. The `avx2` path runs a block in two passes (phases, then lookups with scalar offset loads, [backends.md](backends.md)): one loop with both halves filled the vector scheduler (`llvm-mca` `znver3`), so the simulated cycles were 2.7 times the port bound (33.6 against 12.5 per vector of eight samples). Gather-free lookups by permutes (issue #20) do not fit: the entry tables have 4096 entries of 16 (`C32`) and 8 bytes (`CQ15`), far more than registers hold.
- **`squelch`.** Not accelerated (M19; [backends.md](backends.md)): the average power is a first-order recursion and the gate decision depends on it, so every level runs the fallback, bit-identical to `Reference` (accuracy `exact`). `C32` is gated against the `pwr_squelch_cc` loop of GNU Radio (`benches/refs/gr-squelch.c`: 55.39 instructions per sample on x86-64 at every level, 44.21 on aarch64, 98.71 fuel on wasm32; two virtual calls per sample and the average in double precision through memory): 22.80 (0.41) at `sse2`, 23.25 (0.42) at `sse4.2`, 21.25 (0.38) at `avx2`, 20.25 (0.37) at `avx512`, 18.25 (0.41) at `neon` and 63.60 (0.64) on `simd128`, all `release`; `avx2-runtime` runs the baseline build of the fallback and stays `info` (1.07 against `avx2`). Simulated cycles at `avx2` (`znver3` / `alderlake`, whole-loop since round 2): 9.13 / 10.13 against 20.51 / 15.33 of GNU Radio, 0.45 / 0.66 (per block 10.70 / 11.48 against 17.63 / 22.62). The two virtual calls of GNU Radio's loop (`update_state`, `mute`, compiled with `-fcf-protection`) start with an `endbr64` each, which `llvm-mca` 23.1.1 rated at 51 cycles on `znver3` until D64 (GNU Radio 118.10 / 16.53 then, per block 115.95 / 23.83; the ratio 0.08 to 0.09 of rounds 1 and 2 was this artifact). uiCA gives the two calls alone 13.0 (Skylake) and 13.2 (Tiger Lake) cycles per sample: GNU Radio's average goes through memory (`d_prev_output` stored and loaded per sample), a chain that `llvm-mca` does not model, so its real cost is higher than both `llvm-mca` figures. `CQ15` has no counterpart in GNU Radio and stays `info` against the fallback. How (issue #20): the `C32` loop is split into one loop per gate state, each left through a predicted branch at the first sample that may change the gate, needs the flush to zero or has a non-finite power, so the chain from one average to the next is only the subtraction, multiplication and addition of the contract (before: plus the compare and select of the flush, and the gate and output decided per sample); the `CQ15` loop carries `p + 2^30` so that `round_shift(p, 31)` is one shift, and adds the low 31 bits beside the multiplication of the high part by `2^31 - alpha` (before: the saturating addition of `round_shift`, the subtraction and the multiplication in a row). Before / after, instructions per sample: `C32` 27.35 / 22.80 (`sse2`), 26.35 / 23.25 (`sse4.2`), 21.84 / 21.25 (`avx2`), 21.80 / 20.25 (`avx512`), 23.73 / 18.25 (`neon`), 67.06 / 63.60 (`simd128`); `CQ15` 28.03 / 27.03 (`sse2`, `sse4.2`), 28.03 / 25.53 (`avx2`), 28.53 / 25.53 (`avx512`), 21.02 / 21.02 (`neon`), 71.05 / 67.05 (`simd128`). Simulated cycles, before / after: `avx2` (`znver3` / `alderlake`) `C32` 15.14 / 19.78 to 10.70 / 11.48, `CQ15` 6.04 / 15.39 to 5.56 / 6.95; `avx512` (`znver4` / `sapphirerapids`, SDE counts) `C32` 16.67 / 18.85 to 10.81 / 11.48, `CQ15` 8.54 / 10.07 to 5.56 / 6.95; `neon` (`cortex-a72` / `cortex-a76`, QEMU counts, per block) `C32` 22.78 / 12.72 to 21.93 / 13.25 (round 2: the 4 % of `cortex-a76` is an artifact of the per-block simulation, which runs the four short blocks of each gate loop as loops of their own; simulated as one body, the open loop takes 15.08 / 7.06 and the closed loop 15.08 / 8.05 cycles per sample, against the bound of 7 cycles that the chain of the contract sets on the `cortex-a76`, `fsub`, `fmul` and `fadd`: nothing to recover), `CQ15` 13.09 / 10.59 to 11.08 / 9.12. Input of the measurement: noise with a burst in the middle third, so the gate opens and closes once per call; 4096 samples at 48 kHz, 2 ms time constant, -40 dB open level and 3 dB hysteresis (GNU Radio: -40 dB threshold, the same coefficient, no ramp, no gating).
- **`mag`.** `C32` is the loop of `mag_sq` with a vector `sqrt` on the sums and needs 1.04 instructions per sample at `avx2` against 1.64 for `volk_32fc_magnitude_32f`; the generic code is the path on every level, and it is bit-exact (IEEE `sqrt` is correctly rounded). `CQ15` computes the rounded integer root from a single-precision estimate and one correction step in each direction, with the products from `pmaddwd` (`dot_i16x8` on SIMD128, `smlal` on NEON), and needs 5.09 / 5.09 / 2.25 / 1.22 / 3.26 / 9.03 instructions per sample at `sse2` / `sse4.2` / `avx2` / `avx512` / `neon` / `simd128`; `volk_16ic_magnitude_16i` needs 21.27 / 11.02 / 2.64 / 2.64 / 4.52 / 18.28 (on `neon` the cheapest combination of VOLK, the auto-vectorised `generic` code, needs 3.64, which is information: 0.90). The NEON path takes one correction step instead of two (a biased single-precision estimate that is never above the root, decisions log of M20; 4.39 before). VOLK scales to float and back and rounds differently from the integer rule: both are 96.2 dB against the unrounded magnitude. The entries are `release` at all six levels (every ratio is below 1.00; open item M20-3 closed). The result is bit-exact with the fallback on every level (the sweep over all 2^32 raw pairs, [testing.md](testing.md)).
- **Spectrum fallbacks (`mag_sq`, `mag`, `window`, `power_db`; fallback round).** The fallback entries compare our fallback with CMSIS-DSP's generic C. Changes, all with the bits of the contract: the `CQ15` `mag_sq` and `window` in 32-bit lanes instead of `i64` (`avx2`, `znver3` / `alderlake`, whole-loop: 1.64 / 1.80 to 0.20 / 0.23 and 2.48 / 2.89 to 0.56 / 0.56 cycles per sample, CMSIS-DSP 2.17 / 1.84 and 0.77 / 0.82), the `CQ15` `mag` from the single-precision root with one correction step instead of `isqrt_u64` (15.22 / 14.45 to 7.93 / 10.05, CMSIS-DSP 79.45 / 83.53; 129.29 on `znver3` before D64), `power_db` as the steps of `log10f` without its special cases, so that the loop vectorizes without branches (`f32` 3.45 / 2.88 to 2.76 / 2.08, `Q15` 4.10 / 3.44 to 3.05 / 2.46), and the `C32` `window` with the weights duplicated into a small array on x86 up to AVX2, which LLVM compiles to one `vpermps` per vector of weights instead of splitting and re-interleaving the parts (0.61 / 0.77 to 0.39 / 0.36, CMSIS-DSP 0.41 / 0.38; 4.02 to 3.14 instructions per sample at `sse2`). Spot checks of the shared code (per block, cycles per sample): `sse2` `skylake` / `znver2` faster for every changed fallback (`power_db` `f32` 12.12 / 14.98 to 8.91 / 11.13); `neon` `cortex-a72` / `cortex-a76` faster for every changed fallback but the `f32` `power_db`, whose branch-free steps simulated slower there (21.88 / 9.94 against 19.63 / 8.81 with `libm::log10f`), so the `f32` loop keeps the `libm::log10f` call on AArch64 (target-dependent form as in D40; the same code as before; `Q15` 18.07 / 8.41 to 14.96 / 6.66). `mag_fallback` stayed `info` everywhere but `simd128` in this round: `libm::sqrtf` is inline assembly on x86-64 and AArch64, which the compiler cannot vectorize; since D67 the square root of `sdrk_core::num::sqrt_f32` comes from `f32::sqrt` with the feature `std` of `sdrk-core` (on in the measurement binary, through `sdrk/std` for users), which LLVM vectorizes, and the entry is `release` on every level (next item). The SIMD `power_db` runs blocks of 32 samples in three passes over stack buffers (reduction and division; polynomial; sum), which cuts the latency chain of `log10f` into three: `f32` 2.76 / 2.12 to 2.17 / 1.90, `Q15` 3.09 / 2.46 to 1.98 / 1.95 at `avx2`; the passes are used where they simulated faster (8 lanes, `Q15` at AVX-512), elsewhere one pass; on NEON they do not pay off (`f32`, `cortex-a72` / `cortex-a76`, per block: 15.80 / 8.38 in passes against 15.91 / 6.84 in one pass, the figures of the code before). The `mag_sq` `CQ15` SIMD entries are `info` with the efficiency verdict (0.96 / 0.73 at `avx2`).
- **Vectorizable square root of the fallbacks (D67).** With the feature `std` of `sdrk-core` (on in the measurement binary on every level; for users through `sdrk/std`) `sdrk_core::num::sqrt_f32` and `sqrt_f64` are `f32::sqrt` and `f64::sqrt`, LLVM's own square root, the same bits as `libm` (IEEE 754 rounds it correctly). The loops of `mag_fallback` and `mag_q15_fallback` vectorize (`vsqrtps`, `fsqrt .4s`): at `avx2` (`znver3` / `alderlake`, whole-loop) `mag_fallback` 5.15 / 1.71 to 0.90 / 0.38 cycles per sample against 1.15 / 0.59 of CMSIS-DSP, `mag_q15_fallback` 7.93 / 10.05 to 1.93 / 1.28; instructions per sample of `mag_fallback` 10.01 / 9.01 / 9.01 / 9.01 / 7.01 to 3.26 / 3.26 / 1.76 / 1.23 / 2.01 at `sse2` / `sse4.2` / `avx2` / `avx512` / `neon` (CMSIS-DSP 4.27 / 4.27 / 2.40 / 1.89 / 3.01), `simd128` unchanged (8.28; its `f32x4_sqrt` lane was vectorizable before). `mag_fallback` is `release` on every level. The scalar tail of the SIMD `agc` magnitudes keeps `libm::sqrtf` outside wasm32: vectorized, it cost spills in the recursion of the same function (13.18 / 14.47 to 14.22 / 16.24 cycles per sample). Without `std` (Cortex-M, RISC-V, `no_std` builds on x86-64 and AArch64) the square root stays `libm`: `core::f32::math::sqrt` is unstable, and the `core::arch` intrinsics would need `unsafe`, which the core forbids.
- **`agc`, `dc_block_real`.** The gain of `agc` follows the output of the previous sample, and `dc_block_real` is a first-order recursion on the previous output (M20; [backends.md](backends.md)). The `CQ15` `agc` and the `Q15`/`CQ15` recursions are not accelerated and are `info` against the fallback at every level; the `f32`/`C32` `dc_block_real`, `dc_block` and `deemphasis` run the `f64` block scan of the SIMD backend since the maintainer decision of 2026-10-29 (below). The `C32` `agc` keeps the recursion scalar and computes the rest in vectors (next item). The input of `agc` is noise whose amplitude steps from 0.01 to 0.3 to 0.003 in thirds (the gain moves in both directions within a call; about 22 % of the samples are attacks, 37 % releases at or above `reference / 4` and 41 % releases below it), that of `dc_block_real` noise of 0.2 with an offset of 0.3. `deemphasis` and `dc_block` (issue #20) are counted the same way: `deemphasis` filters noise of 0.5 with 75 us at 48 kHz (`alpha` about 0.24), `dc_block` complex noise of 0.2 with the offset (0.3, -0.2), `r = 0.999`. The recursions bound their speed: the fallback has a second form of each with a shorter chain from one output to the next and the same bits (the `f32` flush beside the arithmetic, the Q15 input terms and rounding constant added before the product), used per kernel on the targets where it simulated faster on both core models of every level ([backends.md](backends.md)). At `avx2` (`znver3` / `alderlake`, cycles per sample) it takes `dc_block_real` and `dc_block` `f32`/`C32` from 9.03 / 13.04 to 7.03 / 9.04, `dc_block_real` `Q15` from 7.02 / 6.64 to 5.53 / 5.56, `dc_block` `CQ15` from 11.07 / 11.10 to 6.30 / 6.12, `deemphasis` `f32` from 12.03 / 16.05 to 9.54 / 11.57 and `deemphasis` `Q15` from 7.04 / 7.05 to 5.53 / 5.53. Round 2 (maintainer decision of 2026-10-29): the `f32`/`C32` variants are tolerance kernels of the SIMD backend and run an `f64` block scan ([backends.md](backends.md)); there is no SIMD library with a block-scan form of these recursions, so their entries stay `info` against the fallback and are judged by simulated cycles: at `avx2` (whole-loop, `znver3` / `alderlake`) `dc_block_real` 2.12 / 1.88, `dc_block` `C32` 2.95 / 2.76 (the fallback 7.03 / 9.04 for both) and `deemphasis` 2.09 / 1.84 (the fallback 9.55 / 11.58); at `sse2` (per block, `skylake` / `znver2`) 4.63 / 3.80, 5.71 / 4.79 and 4.23 / 3.59 against 10.03 / 7.04, 10.07 / 7.08 and 15.13 / 12.12; at `avx512` (`znver4` / `sapphirerapids`) 2.23 / 2.62, 2.65 / 3.34 and 2.09 / 2.35 against 7.04 / 8.05, 9.09 / 11.12 and 10.22 / 11.13; at `neon` (`cortex-a72` / `cortex-a76`) 6.74 / 3.34, 10.20 / 4.79 and 6.03 / 3.09 against 13.07 / 8.06, 16.04 / 8.04 and 18.09 / 10.07. At `simd128` (instructions) `dc_block_real` and `deemphasis` take 26.81 and 24.81 instructions against 27.54 and 27.53; `dc_block` `C32` would take 36.56 against 26.53 and runs the fallback there. The accuracy against the f64 recursion rises from 124.9, 120.8 and 145.4 dB to the optimum, about 152 dB. The fallbacks have entries of their own against the scalar references of the round: `dc_block_real_fallback` and `dc_block_fallback` against liquid-dsp's DC blocker (`iirfilt_*_create_dc_blocker`, direct form II with liquid's portable dot product: 102 and 121 instructions per sample at `avx2`, 229 / 36 and 240 / 41 cycles; its SNR is 84 to 85 dB against the f64 recursion), `deemphasis_fallback` against the IIR loop of GNU Radio's `fm_deemph` (60 instructions, 10.81 / 23.15 cycles at `avx2`; another filter with one more tap, computed in double precision, 152 dB against its own f64 model, hence an accuracy waiver). All three are `release` at every level with 0.06 to 0.37 of the reference's instructions.
- **`biquad`, `biquad_real` against KFR and CMSIS-DSP (D15 to D20).** The measured filter is the fourth-order Butterworth low-pass at 0.1 of the sample rate (two sections) on noise of the amplitude 0.3; the counts are per sample for both sections. On the maintainer's direction the SIMD entries are judged against KFR 7.2.0 (f32, C32; at `simd128` since D29, before that CMSIS-DSP) or our fallback (Q15, CQ15, which no library has in SIMD on our levels), and four fallback entries judge our fallback against CMSIS-DSP 1.18.0 generic C compiled for the same level. The six entries against a reference library are `release` and pass at every level; the two Q15 SIMD entries are `info`, because our own fallback is no fair gate for a SIMD branch (D21; the fallbacks themselves are gated against CMSIS-DSP). All of them are at or below the goal at every level (`biquad_real_fallback`, open item M22-9 until 2026-10-28, is now 0.88 to 0.99 of CMSIS-DSP's `df2T_f32`). The accuracy of every biquad entry is the SNR against the f64 model of the cascade (direct form I in `f64`, the coefficients as the kernels see them, no rounding between the sections), the same yardstick for both sides.
  - **Vector paths (D18).** All four types run a wavefront over the sections: two consecutive sections in the lanes of one vector, the second one sample behind the first, with the operations of the fallback per lane, so the output is the fallback's bit for bit; the `Q15` types compute exactly in `f64` lanes. The same generic `fearless_simd` code on every level, AVX-512 included (the AVX-512 special case of M22 and its 25 % rule are gone). Against KFR: `biquad_real` 0.33 (`avx512`) to 0.55 (`neon`) and 0.83 at `simd128`, `biquad` `C32` 0.18 to 0.27 and 0.44 at `simd128` (before D29 judged against CMSIS-DSP there: 0.69 and 0.36). Against our fallback in instructions: `Q15` 0.40 to 0.46 (1.00 at `sse2`, which runs the fallback), `CQ15` 0.18 to 0.52 (until issue #20 0.40 to 0.68 and 0.19 to 0.59; 0.37 to 0.52 and 0.16 to 0.52 before the fallback got faster, D36; in simulated cycles the `f64` vector path of `Q15` was slower than the fallback, see D35). Since issue #20 the `Q15` and `CQ15` pairs run in 64-bit integer lanes on x86 from SSE4.1 and on NEON (the fallback's own arithmetic, [backends.md](backends.md)); plain SSE2 runs the fallback for `Q15`. No library has a SIMD `Q15` biquad, so the `info` entries are judged by their simulated cycles against the fallback (ours / fallback, cycles per sample of the two models of the level, `llvm-mca`; before the change in brackets): `Q15` `sse4.2` 12.1 / 15.1 and 11.3 / 15.1 (32.5, 17.4), `avx2` 8.2 / 11.1 and 10.2 / 12.1 (21.4, 32.8), `avx512` 9.0 / 14.6 and 10.6 / 13.1 (19.3, 29.4), `neon` 15.2 / 15.1 and 12.9 / 28.1 (36.3, 16.4); `CQ15` `sse4.2` 16.2 / 24.4 and 15.7 / 31.2 (32.5, 21.4), `avx2` 9.2 / 21.2 and 10.6 / 21.3 (17.5, 29.6), `avx512` 9.1 / 21.5 and 10.9 / 20.7 (19.8, 30.0), `neon` 17.3 / 24.1 and 15.4 / 38.1 (37.1, 17.2) (`skylake`/`znver2`, `znver3`/`alderlake`, `znver4`/`sapphirerapids`, `cortex-a72`/`cortex-a76`). Both paths are limited by the chain from one output to the next (a product, one or two sums, the shift, the saturation and the shuffle to the second section); on the Cortex-A72 `Q15` only matches the fallback (`sqxtn` and the shift take 7 of its 15 cycles). `CQ15` at `sse2` keeps the `f64` path: 32.5 / 24.4 on `skylake`, 20.5 / 31.2 on `znver2`. Run-time detection costs at most 0.01 instructions per sample (1.00 at `avx2-runtime` for all four); the entries go through one function per level (`dispatch_fn!`), and the hot loops call methods instead of closures, because with run-time detection a closure is not compiled with the target features of the level (measured before the change: `biquad_real` `Q15` 26.2 against 19.9 at `avx2`).
  - **Fallback (D16, D17).** The `f32` states are flushed on a fixed schedule (once per 16 samples, D17) instead of after every sample, and `biquad_real` `f32` runs two sections per pass (each sample through both); with that the `f32` fallback is within 0.88 to 0.99 of CMSIS-DSP's `df2T_f32` (before: 1.59 to 2.04). In the blocks of sixteen samples of a pair, the flush of the second section (after its sample 14) moves to between the two steps of sample 15, beside the flush of the first section, and in the other fifteen samples the second section stores its output before its state update; both keep every operation and its order (M22-9: at `sse2`, `sse4.2` and `avx2` 30.26, 31.79 and 22.87 before, 28.78, 28.33 and 21.38 after against 29.04, 29.04 and 22.54; `avx512` 21.57 to 19.95; `neon` and `simd128` unchanged). The `C32` fallback is 0.52 to 0.58 of CMSIS-DSP's stereo cascade, the `Q15` fallback 0.52 to 0.71 and the `CQ15` fallback 0.58 to 0.75 of `df1_q15` (D36: two carried partial sums shorten the dependency chain from one output to the next, the `Q15` sections run in pairs, and the history is moved only for the last two samples of a buffer; before 0.67 to 0.81 and 0.66 to 0.90). In simulated cycles at `avx2` (`cargo measure`) the `Q15` fallback is 0.70 / 0.95 of `df1_q15` on `znver3` / `alderlake` (before 0.69, uncertain, / 1.42) and the `CQ15` fallback 0.67 / 0.84 (before 0.80 / 0.99).
  - **Accuracy (D20).** The `f32` association is now the one of KFR and CMSIS-DSP's `df2T_f32`, `s1 = (b1 x + s2) - a1 y`; ours and KFR then give the same SNR (`biquad_real` 136.7 dB, `biquad` 136.1 dB; with the earlier association ours was 1.1 dB lower and failed the 1 dB rule). CMSIS-DSP's stereo cascade keeps the other association (134.9 dB). CMSIS-DSP's Q15 cascade truncates and is 14.9 dB less accurate than ours (56.7 against 71.6 dB).
  - **Notes.** KFR's lane-shifting cascade is slower than CMSIS-DSP's scalar code for two sections at every native level, `avx512` included (`biquad_real`: 40.80 against 29.04 instructions per sample at `sse2`, 26.81 against 22.54 at `avx2`, 36.00 against 22.54 at `avx512`, 23.32 against 22.53 at `neon`; KFR's `avx512` build counts more than its `avx2` build). KFR is the gate on the maintainer's direction (D15: the established SIMD library of the same algorithm class), but CMSIS-DSP is the stronger reference for these entries; every `f32` and `C32` SIMD entry also passes against it, which the results record as information next to KFR (`biquad_real` 0.49 to 0.59, the highest 0.59 at `avx2` and 0.57 at `neon`; `biquad` `C32` 0.29 to 0.30; at `simd128` 0.69 and 0.36). At `simd128` it is the other way round: KFR (48.39 for `biquad_real`, 96.77 for `biquad`) counts fewer than CMSIS-DSP's generic C (57.94, 118.83), because clang turns KFR's vector code into SIMD128 while the scalar cascade gains nothing from it; KFR is the stronger reference there, so the entries got stricter (0.83 and 0.44 instead of 0.69 and 0.36), and the SNR of KFR equals ours there as on the other levels (136.7 dB, 136.1 dB). KFR 7.2.0 loses its state on a call with fewer samples than sections (see the tests of `benches/kernels`); the measured calls have 64 samples and more. The counts were measured with Intel SDE 9.58 for `sse2`, `sse4.2` and `avx512`, callgrind for `avx2`, QEMU 10.2.1 with the `insn` plugin for `neon` and wasmtime fuel for `simd128`, first merged into `benches/results/` with `cargo xtask count --merge`; full runs of `cargo xtask count` on all four platforms (2026-10-27) confirmed every record unchanged and wrote KFR, CMSIS-DSP and `kfr_cxx` into the `references` section of the results files (on `wasm32` CMSIS-DSP only, until the full run of 2026-10-28 with KFR, the wasi-sdk sysroot and Emscripten, D29).
- **`agc` fallbacks (round 2).** `agc_fallback` judges the `C32` fallback alone against the `agc_cc` loop of GNU Radio: 28.24 (1.18) at `sse2` and `sse4.2`, 23.36 (0.97) at `avx2` and `avx512`, 22.42 (1.32) at `neon`, 64.11 (1.00) on `simd128` instructions per sample. GNU Radio's loop does less (see below), so the entry is `dev` where the plain loop needs more instructions (`sse2`, `sse4.2`, `neon`) and `release` elsewhere; in simulated cycles the fallback is below GNU Radio on every model: `avx2` 17.93 / 18.42 against 36.12 / 39.15 (whole-loop, `znver3` / `alderlake`), `sse2` 18.80 / 17.67 against 42.12 / 38.13 (per block, `skylake` / `znver2`), `neon` 39.90 / 21.75 against 49.16 / 30.12 (per block, `cortex-a72` / `cortex-a76`). The `CQ15` fallback has no fair reference (no established fixed-point AGC) and stays `info`; whole-loop at `avx2` it takes 78.32 / 86.42 cycles per sample (`znver3` / `alderlake`, efficiency 0.18 / 0.13: the chain from one gain to the next, the product with the input, the rounded square root in f64, the f64 estimate of the quotient and its exact correction, round 1), the figures that replace the per-block 26.82 / 50.81; no change in round 2.
- **`agc` `C32` against GNU Radio.** The reference is the `agc_cc` loop of GNU Radio (member for member in `benches/refs/gr-agc.c`: the four members live in one object that the output stores may alias, so the gain is loaded and stored in every iteration, `sqrtf` keeps its `errno` rule and `_max_gain > 0.0` is a comparison in double precision, as in `agc.h`). It is compiled once per platform with the flags of GNU Radio's own build, because the loop is header-only code that VOLK's dispatch never touches: 24.03 instructions per sample on x86-64 at every level, 17.03 on aarch64 (GCC fuses the products and sums into `fmadd`, which our kernels do not do, and omits the `errno` branch), 64.05 fuel on wasm32 (clang). Ours: 23.79 (0.99) at `sse2`, 23.04 (0.96) at `sse4.2`, 17.39 (0.72) at `avx2`, 16.12 (0.67) at `avx512`, 22.42 (1.32, `dev`: since D335 `neon` runs the fallback, which is faster there in simulated cycles; 17.61, 1.03, before) at `neon` and 64.61 (1.01) on `simd128`, the others `release` within the limit of 1.10x (open item M20-6 closed; before: 23.84 / 23.84 / 20.84 / 20.84 / 21.84 / 62.21, `neon` `dev` at 1.28). The clamp of the gain to both limits in the attack and in the first release form (a gain at a limit could leave it by one rounding with a clamp on one side) costs 0.48 to 0.64 instructions per sample on x86-64, 0.58 at `neon` and 3.53 fuel on `simd128` (a compare and a select per limit there): 17.03 (1.00) at `neon` and 61.08 (0.95) on `simd128` with the one-sided clamp. How: the output level is taken as `g |x|` instead of `|g x|` (the same value, other roundings), so the magnitudes `|x|` and the products `g x` do not depend on the recursion; the SIMD backend computes them in vectors per block of 256 samples (2.25 and 2.0 instructions per sample at `neon`) and the scalar recursion in between costs a load, a multiplication, the update and an eighth of a store pair (12.5 at `neon` on this input). The release from `reference / 4` up is evaluated as `g + (r_r ref / |x| - r_r g)` (no division by the output level), the release below it as `g + g ((r_r / lo) (ref - m))` (no division), the attack as specified ([kernels.md](kernels.md)); the backend tells per block whether a magnitude is zero or NaN and leaves out the test of the hold otherwise (0.47 instructions per sample less at `neon` on this input, the compare and the `and` per vector included). On NEON the limits are `fminnm`/`fmaxnm` written as `asm!`: with `f32::min`/`max` the compiler adds an `fminnm` of every operand that comes from another basic block, and with the intrinsics `vminnm_f32` it evaluates every branch and selects (31.12 instructions per sample); the recursion runs eight samples per iteration there (four on x86-64, whose 16 registers do not hold eight), written out because the compiler keeps a loop of eight steps (22.01 with that loop). No multiply-add (decision D5). The fallback alone (the information line) is one plain scalar loop for every target since open item M20-7 (no target-specific code in the loop, no `black_box`; the square root `num::sqrt_f32` reaches the hardware instruction through lane 0 of `f32x4_sqrt` on wasm32 with `simd128`, as before the review; four samples per iteration, the limits a compare and a select, the two-sided clamps taking the minimum with `g_max` first so that the three branches do not end in the same operation, which the compiler would otherwise move behind them and turn into a branch of its own on x86-64): 28.24 at `sse2` and `sse4.2`, 23.36 at `avx2` and `avx512`, 22.42 at `neon`, 64.11 fuel on `simd128` and 73.66 on Cortex-M4 (built with `-C target-cpu=cortex-m4` since D30; 76.16 with the generic Armv7E-M tuning, to which the following Cortex-M4 figures refer: before the review 23.84 / 20.84 / 21.84 / 62.21 / 80.50; the review's version with `black_box` 27.87 / 22.73 / 22.92 / 66.89 / 75.52). The level `g |x|` costs the multiplication and, because `x` is now used twice (for `g x` and for its squares), two register copies per sample at `sse2` and two loads at `avx2` that the loads folded into the products avoided before; in the old form the output was squared in place. The loop of GNU Radio does less than ours: no lower gain limit, no NaN or silence rule, no bounded update, and an additive rule that is level dependent (22 dB low after 5 time constants for a step of 30 dB). The accuracy is 119.9 dB against the f64 model of the specification at every level (123.1 dB before; a constant `1 - r_r` rounded to f32 in the release, which would save one instruction, gave 103 dB and was not taken). On wasm32 `libm` computes the square root in software (the instruction of `core` is unstable there), 189.51 fuel per sample before `sdrk_core::num::sqrt_f32` took the `simd128` instruction. The `CQ15` fallback does not divide in integers: each quotient of the update is an f64 estimate followed by one exact correction in 64-bit integers (the remainder `n - k d` as the wrapping difference of the low 64 bits, which bounds on the estimate keep below `2^63`; the argument is in the comments of `fallback/agc.rs`), and the rounded square root is one f64 square root (exact up to `2^32`). That gives 72.86 instructions per sample at `sse2` and `sse4.2`, 69.40 at `avx2`, 70.31 at `avx512`, 54.77 at `neon`, 141.14 fuel on wasm32 and 1336.96 on Cortex-M4 (counted since M20-7; 1333.88 before the workload was built with `-C target-cpu=cortex-m4`, D30), against 124.61 / 124.61 / 125.08 / 110.27 / 111.35 / 620.79 with the 64-bit divisors and the `u128` division of the attack (software `__udivti3`; before them, with `i128` divisions, 190.78 at `neon` and 968.79 fuel). On x86-64 the estimate is turned into an integer by adding `2^52` (the saturating `as` costs about five instructions more there), on AArch64, wasm32 and Cortex-M4 by `as` (one instruction or operator on the first two); the `avx2-runtime` build of this fallback is the baseline code and counts 72.86 against 69.40 of `avx2` (`info`). The values are those of `Reference` bit for bit (unit tests against plain `u128` division at the edges of the domain, at 2 million random points and next to multiples of the divisor, and a kernel test over the parameter extremes). Issue #20 (no established fixed-point AGC exists, so the entry stays `info` and is judged by its efficiency) shortened the path from one gain to the next, which bounds the loop: the candidate `k` comes from one addition of `2^52 - 1/2` instead of `1/2` and then `2^52` on x86-64, the rounded square root is the low 32 bits of the magic sum (no subtraction), the release estimate moves the rate into the divisor (`g e / (d (2^48 / r_r))`, so it does not wait for the product `r_r e`), the divisor `d 2^48` of the release correction is a product with a `black_box` of `2^48` (with the constant the compiler turns `k * (d << 48)` into a shift, a subtraction and a product after `k`), the clamp takes one side per branch, and on x86-64 a saturating output is a branch (12 % of the samples of the benchmark take it; on AArch64 and wasm32 the selects stay, 0.63 instructions and 2.4 cycles per sample fewer at `neon` on the Cortex-A76 model, 16 fuel fewer). Counts: 70.53 at `sse2` and `sse4.2` (72.86 before), 67.11 at `avx2` (69.40), 67.06 at `avx512` (70.31), 53.50 at `neon` (54.77), 137.07 fuel on wasm32 (141.14); Cortex-M4 not counted again. The per-block cycles of `cargo measure` are distorted for this loop (each basic block is simulated as a loop of its own, so a register reused across the blocks of one sample becomes a dependency chain of its own, and llvm-mca's `znver3` and `znver4` models lose the dependency through an eliminated `mov r32, r32`): they move from 79.69 / 87.64 to 26.82 / 50.81 (`znver3` / `alderlake`). The whole path of one sample as one loop body (llvm-mca, register moves written as `lea` against the lost dependency; uiCA for Skylake and Tiger Lake) gives the change of the real chain: a release sample 91.0 -> 78.1 cycles on `znver3`, 100.1 -> 89.1 on `alderlake`, 92.5 -> 80.5 (uiCA Skylake), 93.3 -> 84.3 (Tiger Lake); an attack sample 90.1 -> 82.6, 99.6 -> 93.1, 92.0 -> 83.8, 94.8 -> 86.5; at `avx512` 96.0 -> 80.1 (`znver4`) and 102.1 -> 89.1 (`sapphirerapids`) for a release sample. The update divides by the output level and costs more instructions than the former logistic update: the choice is correctness (a step of 30 dB settles in 5 time constants, the loop cannot collapse, silence holds the gain). At `sse2` and on wasm32 the limits are a compare and a select (`f32::max`/`min` cost 16 instructions per sample more at `sse2` in an isolated copy of the loop, measured with callgrind, and 27.8 more fuel on wasm32).
- **`dot`, `dot_real`, `scale` (issue #20, group C).** One call of 4096 samples per run (the dot products also at n = 64 and 100 as information: the tuned resampler of `sdrk-blocks` calls `dot` twice per output with short windows). The `f32` and `C32` sums use eight vector accumulators: at `avx2` 0.26 / 0.26 simulated cycles per sample for `dot` `C32` against 0.28 / 0.52 for `volk_32fc_32f_dot_prod_32fc` (`znver3` / `alderlake`), 0.14 / 0.10 for `dot_real` against 0.54 / 0.77; four accumulators were latency-bound on `alderlake` (0.58 cycles, the model gives an FMA with a memory operand 12 cycles on the accumulator). At `sse2` and `sse4.2` the generic code without FMA needs 1.07 times VOLK's instructions for `dot` and 1.08 for `dot_real` at `sse2` (cycles at `sse2`: `dot` 0.63 / 0.72 against 0.63 / 0.67, `dot_real` 0.29 / 0.32 against 0.28 / 0.28 on `skylake` / `znver2`). The `Q15` and `CQ15` entries and `scale` `CQ15` are `info` against our fallback (no fair reference); the fallback of `dot` `Q15`/`CQ15` and `scale` `CQ15` was also made vectorizable (plain `i64` sums in runs, `q15_mul` in `i32`), which takes `dot_real` `Q15` at `avx2` from 4.02 / 5.03 to 0.77 / 0.52 cycles per sample and the level paths to 0.13 / 0.16. Run-time detection costs a fixed few dozen instructions per call, which is 2 to 4 % of these cheap kernels at n = 4096: `dot_real`, `dot` `CQ15` and `scale` `CQ15` are `dev` at `avx2-runtime`.
- **`dot`, `dot_real`, `scale` fallbacks against CMSIS-DSP (round 2).** Six entries judge the fallback, compiled for each level, against CMSIS-DSP 1.18.0 generic C of the same level (see the references above): `dot` `C32` / `CQ15` against `arm_dot_prod_f32` / `arm_dot_prod_q15` twice on the planar parts, `dot_real` against the same functions once, `scale` against `arm_scale_f32` / `arm_scale_q15` on the 2N values. Instructions per sample ours / CMSIS-DSP at `sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`: `dot` `C32` 0.80, 0.68, 0.87, 0.67, 0.77, 0.65; `dot` `CQ15` 1.48 (`dev`), 1.28 (`dev`), 1.09, 1.10, 0.73, 0.69; `dot_real` `f32` 1.00, 0.88, 0.95, 0.95, 0.77, 0.48; `dot_real` `Q15` 1.23 (`dev`), 1.05, 1.06, 1.01, 0.83, 0.90; `scale` `C32` 1.00, 0.75, 0.55, 0.36, 0.42, 1.14 (`dev`); `scale` `CQ15` 0.69, 0.61, 0.62, 0.44, 0.79, 0.98. Simulated cycles per sample at `avx2` (`cargo measure`, whole-loop, `znver3` / `alderlake`, ours against CMSIS-DSP): `dot` `C32` 3.03 / 3.03 against 6.07 / 6.07 (both chains latency-bound, ours runs the two components side by side), `dot` `CQ15` 1.04 / 0.77 against 1.07 / 1.03, `dot_real` `f32` 3.03 / 3.03 against 3.04 / 3.04, `dot_real` `Q15` 0.32 / 0.38 against 0.54 / 0.52, `scale` `C32` 0.26 / 0.20 against 0.27 / 0.29, `scale` `CQ15` 0.28 / 0.42 against 0.65 / 0.78. The `f32` and `C32` sums keep the ascending chain of the contract, so they cannot be faster than the latency of one addition per sample; CMSIS-DSP's loops are the same chain.
  - **Changes (D34: every value unchanged, bit for bit).** `dot_real` `Q15` sums the high and low bytes of the samples in blocks of 256 (exact in `i32`), which the compiler turns into `pmaddwd`: from AVX2 one pass per sum, at SSE2 one pass for both, from 256 samples on (shorter slices keep the widening form, which has no fixed cost per block) (`avx2` 0.77 / 0.52 to 0.32 / 0.38 cycles, `avx512` 0.77 / 0.52 to 0.36 / 0.40 on `znver4` / `sapphirerapids`, `sse2` 1.03 / 1.54 to 0.98 / 0.98 on `skylake` / `znver2`; SSE4.2 keeps the widening form). `scale` `CQ15` reads every sample as one 32-bit word and multiplies both halves in its lanes, without widening and repacking (`avx2` 0.67 / 0.57 to 0.28 / 0.42, `sse2` 4.81 / 4.06 to 2.05 / 2.13, `sse4.2` 1.16 / 1.27 to 1.14 / 1.06); AVX-512 keeps the widening form (the word form took 0.43 instead of 0.41 cycles on `sapphirerapids`). `dot_real` `f32` forms the products of eight samples in one vector before the chain adds them, from AVX2, on AArch64 and on wasm32: 3.40 to 2.40 instructions per sample at `avx2` and `avx512` (the gate was 1.35 of CMSIS-DSP), 2.77 to 2.52 at `neon`, 13.80 to 6.17 at `simd128`, the cycles unchanged; at SSE2 and SSE4.2 the cycle tools gave 2 to 4 % more cycles, so the loop stays. `scale` `C32` at `simd128` runs blocks of 16 floats (6.53 to 6.16 instructions per sample).
  - **Not changed.** `dot` `CQ15`: the byte split read as 32-bit words needs `vpmulld` for the signed high bytes and four block sums; it was faster on `znver3` (0.70 against 1.04) but slower on `alderlake` (0.81 against 0.77), slower on `skylake` at `sse2` (2.34 against 2.20) and on `znver2` at `sse4.2` (1.87 against 1.75), so the widening form stays everywhere and the entry is `dev` at `sse2` (1.48) and `sse4.2` (1.28). `dot_real` `Q15` at `sse2` is `dev` (1.23; LLVM vectorizes the byte sums with four lanes there and does not form pairs for `pmaddwd`), `scale` `C32` at `simd128` is `dev` (1.14; the compiled loop computes every address with `i32.add` instead of the `offset` of the load).
  - **SIMD, fixed cost per call.** `dot` `C32` takes its real and imaginary sums from the vector halves and two scalar pairs per half (`PairSum`, the same bits as `unzip` and `reduce_sum`): at `avx2` 3.09 to 2.97 instructions per sample at n = 64 (VOLK 3.84, ratio 0.80 to 0.77), 2.45 to 2.37 at n = 100 and 0.76 to 0.75 at n = 4096, with the measurement harness of this round (its per-call cost is a few instructions above the committed numbers of round 1 on both sides). `dot_real` `f32` at n = 64 stays at 2.17 against 2.73: one block of eight vectors, the tree of the accumulators and the reduction of one vector are the whole fixed cost. Run-time detection still costs about 20 instructions per call (`dispatch_fn!`), so `dot_real`, `dot` `CQ15` and `scale` `CQ15` stay `dev` at `avx2-runtime`.
- **Halfband kernels (M23: `halfband_decimate` C32/CQ15, `halfband_decimate_real` f32/Q15, `real_to_complex` C32/CQ15; D69 to D71, D87 to D100).** The workload is 47 taps (24 non-zero odd-offset taps and the centre; liquid-dsp's `m = 12`, libairspy's 47-tap converter), 4096 input samples per call, with SDRk's taps injected into every reference so that the accuracies are comparable. The gates are external: liquid-dsp 1.8.3 (`resamp2_crcf`/`resamp2_rrrf` for the decimators, `firhilbf` for `real_to_complex`, each with the dot product of the level) for the `f32` types, the portable liquid code for the `f32` fallbacks, and the CMSIS-DSP composition (polyphase split, `arm_fir_q15` with the compacted taps, `arm_scale_q15`, `arm_add_q15`) for the `Q15` fallbacks. Every gated entry is `release` and passes at every level: the `f32` kernels at 0.05 to 0.23 of liquid-dsp's instructions, the `f32` fallbacks at 0.22 to 0.78, the `Q15` fallbacks at 0.27 to 1.06 of the CMSIS-DSP composition (the highest at `simd128` 1.06 and at `sse2` 1.02, limit 1.10), and `avx2-runtime` is within 1.01 of `avx2`. The `Q15`/`CQ15` SIMD entries are `info` (no library; the verdict is the efficiency of `cargo measure`): 0.21 to 0.90 of our fallback. `real_to_complex` `CQ15` is provisionally `info` against libairspy 1.0.10 `iqconverter_int16` (open item I23-1): 0.36 to 0.61 of its instructions at every level, exact against its 81.5 dB; its fallback is 0.95 to 1.99 of libairspy (`info`). The accuracy of every `f32` entry is 142.4 to 150.3 dB against 142.0 to 145.5 dB of liquid-dsp (within the 1 dB rule). Work-saving evidence (information, ours / the dense `fir_decimate` by 2 with the expanded 47 taps, instructions per output sample; the maintainer's target is 0.60): `halfband_decimate` C32 0.59 at `sse2`, 0.60 at `sse4.2`, 0.51 at `avx2`, 0.18 at `avx512`, 0.70 on `neon` and 0.91 on `simd128` (the target is missed on `neon` and `simd128`, where the dense kernel is itself a tight loop; the target is the `own_target` of the entries in `benches/limits.json` and the results record `meets_target` per level, information only, D102); CQ15 0.17 to 0.26; `real_to_complex` against the separate shift followed by the halfband decimation 0.40 to 0.51 (C32) and 0.29 to 0.45 (CQ15). Simulated cycles at `avx2` (table above): the C32 kernel takes 3.78 / 2.99 cycles per output sample (`znver3` / `alderlake`) against 279.4 / 37.5 for liquid-dsp, `halfband_decimate` CQ15 7.60 / 7.14 against 13.07 / 13.27 for the fallback.
  - **How (D94 to D100).** The fallback groups eight outputs (sixteen on wasm32), copies the even-offset samples of the window once (for `real_to_complex` with the sign of the shift) so that every tap multiplies consecutive samples, sums `Q15` products in `i32` when the absolute tap sum is at most 65535 raw (exactly the `i64` sums; with `i32x4.extmul` on wasm32) and adds the centre tap last in `f32` (D94). The SIMD kernels keep complex samples interleaved, copy each sample as one constant-width move, add the symmetric pair of samples before the multiplication with FMA where the level has it, take eight vectors of outputs per pass and size the stack scratch to the filter (D98). Before this round (package C and D) the same counts were: `halfband_decimate` 136.05 (`sse2`) and 102.14 (`avx2`), `halfband_decimate_real` 62.68 and 47.48, `real_to_complex` 75.45 and 57.73; the fallbacks at `neon` 139.4 / 186.7 / 137.5 / 140.5 / 458.0 / 512.5 (`halfband_decimate`, `_cq15`, `_real`, `_real_q15`, `real_to_complex`, `_cq15`), at `simd128` 341.7 / 435.7 / 326.7 / 340.4 / 326.1 / 520.4, which missed their limits at most levels.
- **Efficiency of the `info` entries of M24 (D156 to D160).** Port bound over simulated cycles (`whole-loop`, reference model / counter-model), the verdict required by the status `info` of D111.
  - `stereo_matrix` `CQ15` (SIMD branch): `sse2` 0.98 / 0.98 (`skylake` / `znver2`), `sse4.2` 0.99 / 0.99, `avx2` 0.97 / 0.81 (`znver3` / `alderlake`), `avx512` 0.98 (`znver4`), `neon` 0.93 / 0.83 (`cortex-a72` / `cortex-a76`); bottleneck class throughput at every level. Verdict: the branch runs at the port bound of the instruction set (`pmaddwd`, `packssdw` and the unpacks on x86, `ld2`, the saturating chains and `st2` on NEON); only fewer operations can make it faster. At `neon` it ties the compiled fallback (0.77 / 0.77 cycles, 1.14 / 1.14 instructions) and is kept (D159). On `simd128` there is no SIMD branch for `CQ15` any more: the dot-product form took 5.90 instructions per sample against 5.40 of the fallback, so the kernel runs the fallback there (ratio 1.00, D158); `simd128` has no core model, so it has no efficiency figure.
  - `stereo_pll_q15_fallback` (the loop of the kernel; no SIMD branch exists): efficiency 0.40 / 0.61 at `sse2` and `sse4.2`, 0.39 / 0.40 at `avx2`, 0.39 at `avx512`, 0.50 / 0.59 at `neon`; bottleneck class latency at every level. The loop is a feedback loop: the phase of a sample is formed from the detector output of the previous one, so the dependency chain (phase, three table lookups with interpolation, the two products, the integrator and proportional terms, the new phase) bounds it, not the ports. Dependency-chain bound, taken as the simulated cycles of the same loop body with the dispatch width widened to 64 so that the front end cannot limit it (the cycles did not change): `sse2` 38.0 on `skylake` and 38.0 on `znver2` against 38.04 / 38.05 simulated, `avx2` 37.4 on `alderlake` against 37.39, `neon` 41.0 / 39.0 on `cortex-a72` / `cortex-a76` against 41.07 / 39.08. Verdict: the loop runs at its dependency chain (simulated cycles over chain bound 1.00 on these models); the instructions that are not on the chain are hidden behind it, and only another algorithm (a shorter chain, for example a smaller interpolation depth, which would change the bits) could make it faster. At `avx2` / `avx512` on the reference models (`znver3`, `znver4`) the lea form of the moves gives 37.0 cycles for the same loop; the chain bound was not simulated separately there.
- **`preemph_limit` (M25; D194 to D196).** Not accelerated ([backends.md](backends.md)): the one-pole pre-emphasis recurs on the previous input and output, the envelope on the previous sample, and the gain on the envelope, and the kernel is bit-exact. Both variants are counted through the facade (the fallback) as `info` against our own fallback at every level, with `avx2-runtime`; the entries are `preemph_limit` (`f32`) and `preemph_limit_q15`. Workload: 4096 samples at 48 kHz, pre-emphasis 50 us, limit 0.5, hold 20 ms, release 200 ms, noise of 0.05 with two tone bursts of 0.9 at 3 kHz (attack, hold and release are all visited; the idle state at the start); the `Q15` input is the quantised `f32` signal. Instructions per sample (fuel on `simd128`) at `sse2` / `sse4.2` / `avx2` / `avx512` / `neon` / `simd128`: `f32` 44.30 / 42.30 / 33.30 / 33.98 / 41.63 / 93.56, `Q15` 94.87 / 94.87 / 94.50 / 94.50 / 65.96 / 448.58; the differences between the levels are code generation of the same fallback. `avx2-runtime` runs the baseline build: 44.30 and 94.87. Accuracy `exact` (999 dB) against the `Reference`. Simulated cycles per sample at `avx2` (`znver3` / `alderlake`, whole-loop): `f32` 15.40 / 13.20, `Q15` 27.62 / 18.69; the bottleneck class is latency (efficiency 0.41 / 0.42 and 0.64 / 0.75): the chain of the recursion bounds the loop, not the ports. The `Q15` figure on `simd128` is high (about 4.7 times the `x86_64` count at `sse2` in fuel); no change was made in this round, which only measures.
- **AVX-512** counts come from SDE between the SSC markers and are exact; an earlier version of the method without markers varied by up to 0.07 instructions per sample.
- **Variants.** The misaligned and small-block lines of the results files are information; they show the per-call cost of the conversions (for example `convert` CS16 at `avx512`, n = 64: 2.63 against 2.08 for VOLK).

## Protocol code (`sdrk-proto`, M37)

`sdrk-proto` has no backends (it depends on `sdrk-device` only, R2) and no SIMD (D496): the CRCs are byte-serial table lookups, one 256-entry table per CRC, where each step needs the one before, and the frame decoder is a bit-serial state machine. Ours and the fallback are the same code. The CRC entries are `Family::GnuRadio`: GNU Radio's generic table-driven `gr::digital::crc` (gr-digital, `benches/refs/gr-crc.c`, D527) runs the same messages with the same catalogue parameters and is the gate; `frame_decode` and the two block entries are `Family::Own`, `info` at every level (ratio 1.00), and exist to record the cost. **Units:** a "sample" of the CRC entries is one input **byte** (4096 random bytes per call, as 16 messages of 256 bytes, the largest packet), a "sample" of `frame_decode` one input **bit** (4096 bits per call: eleven frames with a 4-byte preamble and a random 32-byte payload back to back, 344 bits each, then 312 idle zero bits; the decoder keeps its state across calls, `max_sync_errors` 0). The accuracy check compares every CRC with a bit-by-bit model of the catalogue parameters and every decoded payload with the encoded one (999 = all equal).

Simulated cycles per byte (CRCs) and per bit (decoder), `whole-loop`, reference model / counter-model (`cargo measure <entry> --no-golden` on each platform, 2026-11-05, D500):

Since 2026-11-05 the M37 entries are also in `benches/results/*.json` and in the result and cycle tables above, with the comparison side (GNU Radio for the CRCs, our own code for the framing); the figures of this section are ours only and agree with those tables.

| Entry | `sse2` | `sse4.2` | `avx2` | `avx512` | `neon` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crc16_ccitt_false` (per byte) | 13.55 / 11.55 | 13.55 / 11.55 | 11.81 / 12.80 | 11.81 / 12.80 | 10.13 / 7.10 |
| `crc16_x25` (per byte) | 8.06 / 7.06 | 8.06 / 7.06 | 8.05 / 9.05 | 8.05 / 9.04 | 9.12 / 7.10 |
| `crc32` (per byte) | 8.06 / 7.06 | 8.06 / 7.06 | 7.05 / 8.05 | 7.05 / 8.04 | 8.11 / 7.10 |
| `crc24_adsb` (per byte) | 13.05 / 11.06 | 13.05 / 11.06 | 11.04 / 12.04 | 11.04 / 12.04 | 8.10 / 7.09 |
| `frame_decode` (per bit) | 5.81 / 7.68 | 5.81 / 7.68 | 5.46 / 5.68 | 5.47 / 5.67 | 8.28 / 9.46 |
| `frame_encode_block` (per output sample) | 1.86 / 1.78 | 1.86 / 1.78 | 1.63 / 1.71 | 1.57 / 1.68 | 2.09 / 1.93 |
| `frame_decode_block` (per input sample) | 5.64 / 7.68 | 5.71 / 7.77 | 5.21 / 5.19 | 5.20 / 5.18 | 8.27 / 9.37 |

Models as in the [core-model table](#speed-simulated-cycles-per-sample-level-2): `skylake` / `znver2`, `znver3` / `alderlake`, `znver4` / `sapphirerapids`, `cortex-a72` / `cortex-a76`. Bottleneck class: latency for every CRC (efficiency 0.09 to 0.43: the table load and the shift of the previous CRC form the chain), throughput or latency for the decoder (efficiency 0.79 to 0.91).

Instructions per byte (CRCs) and per bit (decoder); `simd128` is wasmtime fuel, the only measure without a core model:

| Entry | `sse2` | `sse4.2` | `avx2` | `avx2-runtime` | `avx512` | `neon` | `simd128` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `crc16_ccitt_false` | 8.13 | 8.13 | 6.88 | 8.13 | 6.88 | 7.07 | 21.79 |
| `crc16_x25` | 7.63 | 7.63 | 6.88 | 7.63 | 6.51 | 7.07 | 19.79 |
| `crc32` | 6.63 | 6.63 | 5.88 | 6.63 | 5.50 | 7.07 | 19.79 |
| `crc24_adsb` | 7.63 | 7.63 | 6.88 | 7.63 | 6.88 | 8.06 | 23.79 |
| `frame_decode` | 25.28 | 25.28 | 24.85 | 25.07 | 23.79 | 20.23 | 58.70 |
| `frame_encode_block` | 3.08 | 3.08 | 3.11 | 3.08 | 2.98 | 2.63 | 8.97 |
| `frame_decode_block` | 23.28 | 23.28 | 22.86 | 23.07 | 22.79 | 20.23 | 53.71 |

**Against GNU Radio** (D527): `gr::digital::crc` of gr-digital (`benches/refs/gr-crc.c`, `-O3`) with the same parameters over the same 256-byte messages; ours divided by GNU Radio, instructions per byte (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`) and simulated cycles per byte (reference model / counter-model at `sse2` = `sse4.2`, `avx2`, `avx512`, `neon`), `cargo measure <entry> --no-golden` on 2026-11-05. Every entry is `release` and within the goal by instructions at every level:

| Entry | Instructions | Cycles |
| --- | --- | --- |
| `crc16_ccitt_false` | 0.80, 0.80, 0.67, 0.67, 0.77, 0.81 | 0.89 / 0.88, 0.90 / 0.79, 0.90 / 0.79, 1.00 / 0.78 |
| `crc16_x25` | 0.93, 0.93, 0.84, 0.79, 0.99, 0.86 | 0.99 / 0.99, 0.78 / 0.96, 0.78 / 0.96, 1.12 / 0.99 |
| `crc32` | 0.81, 0.81, 0.72, 0.67, 0.99, 0.86 | 0.99 / 0.99, 0.68 / 0.86, 0.68 / 0.86, 0.99 / 0.99 |
| `crc24_adsb` | 0.75, 0.75, 0.67, 0.67, 0.88, 0.88 | 0.86 / 0.84, 0.84 / 0.74, 0.84 / 0.74, 0.80 / 0.78 |

Before D527 `crc16_x25` kept its register in a `u16` (16-bit operations in the chain): 1.13 of GNU Radio's instructions at `neon` (above the release limit) and 1.30 / 1.27 of its cycles at `sse2`; the register is now a `u32` with the same 16-bit table and the same bits (the rows above are the new figures). The one cycle ratio above 1.00, `crc16_x25` on `cortex-a72` (9.12 against 8.17), is the half-word table load (`ldrh`) in the chain; GNU Radio loads 64-bit entries from a table four times as large, and the counter-model `cortex-a76` gives 0.99.

For scale: a byte is 8 bits, and a demodulator delivers a bit from several samples (for example 5 samples per bit at 9600 baud and 48 kHz), so a CRC costs well under one cycle per signal sample and the decoder about one. `frame_decode` was 59.40 instructions and 31.22 cycles per bit at `sse2` before the hot path was tightened and inlined (D497). The committed results files get these entries at the next complete count run (D500).

**The packet blocks** (M37, D509): `frame_encode_block` is the block `frame_encode` writing 4096 output samples per call (one `±1.0` per bit; frames of a 4-byte preamble and a random 32-byte payload, 344 samples each, the frame cut at the end of a call continued by the next), `frame_decode_block` the block `frame_decode` reading the 4096 bits of the `frame_decode` workload as `±1.0` soft bits; the units are signal samples, one bit each. Both are `Family::Own` and `info`. Measured with `cargo measure <entry> --no-golden` on all platforms on 2026-11-05. The encoder builds each frame once as packed bytes and copies four samples per nibble from a table: before that (the bit iterator `FrameBits`) it took 35.50 instructions and 7.89 / 8.64 cycles per sample at `avx2`. The decoder block costs what the bare decoder costs once `push_bit` is inlined in both (`#[inline(always)]`): with both callers in one binary LLVM had kept an out-of-line copy, 32.65 instructions and 12.47 / 9.49 cycles per sample.

## RDS blocks (M41)

`rds_demod` and `rds_decode` are composite blocks without a kernel of their own; no library has them, so both count entries are `Family::Own` and `info` at every level (ratio 1.00 against our own code, D678). Counted on all four platforms on 2026-11-06 (`cargo xtask count --platform <p> --kernel rds_demod_block --kernel rds_decode_block --merge`) and their cycles simulated with `cargo measure <entry> --no-count --no-golden --platform x86_64 --platform x86_64-avx512 --platform aarch64` (rows in the cycle tables above).

**Workloads.** `rds_demod_block` reads 4096 samples per call of a 240 kHz multiplex (1 and 2 kHz tones at 0.45, the pilot at 0.1, the RDS subcarrier at 4 % with the biphase pulse of IEC 62106 and random groups 0A), taken cyclically from a buffer of 0.5 s, with the block state kept across calls; the accuracy check decodes the whole buffer from a reset without a bit error after the first 100 bits. `rds_decode_block` reads 4096 `+-1.0` soft bits per call of a period of 40 groups, continued across calls, so it stays synchronized and outputs a group every 104 samples; the accuracy check wants every group of one call with `CRC_OK` at its index. A "sample" is an input sample: a multiplex sample for the demodulator, one soft bit for the decoder.

Instructions per input sample (`simd128`: wasmtime fuel) and simulated cycles per input sample (reference model / counter-model):

| Entry | `sse2` | `sse4.2` | `avx2` | `avx2-runtime` | `avx512` | `neon` | `simd128` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rds_demod_block`, instructions | 138.05 | 137.86 | 46.46 | 48.05 | 41.09 | 78.82 | 340.30 |
| `rds_demod_block`, cycles | 44.52 / 38.82 | 44.50 / 38.75 | 16.82 / 15.22 | - | 477.42 / 53.46 | 55.78 / 49.36 | - |
| `rds_decode_block`, instructions | 39.08 | 38.42 | 37.88 | 38.77 | 37.83 | 32.85 | 84.97 |
| `rds_decode_block`, cycles | 9.18 / 11.80 | 8.74 / 11.42 | 7.39 / 7.97 | - | 7.31 / 7.96 | 12.14 / 13.67 | - |

Models as in the [core-model table](#speed-simulated-cycles-per-sample-level-2). At 240 kHz `rds_demod` costs about 17 cycles per input sample at `avx2` (4 Mcycles per second of signal); the channel filter (583 taps, decimation 12, about 49 real-by-complex products per input sample in `fir_decimate`) is most of it, then `rotate` and `interleave`; `symbol_sync` and `costas` run at 20 kHz and 2375 Hz and the biphase decoding at 2375 chips per second. The `avx512` row of the reference model `znver4` (477 cycles, port-bound 128 cycles per sample, efficiency 0.27) is far above the counter-model `sapphirerapids` (53) and above `znver3` at `avx2` (17) for fewer instructions (41 against 46); the same kind of gap appears for `halfband_decimate_fallback` at `avx512` (61 against 13). It is a property of the `znver4` model for some instruction of that build, not of the block; the instruction counts are the robust figure there. `rds_decode` costs 32 to 39 instructions and 7 to 12 cycles per soft bit (at 1187.5 bit/s about 10 kcycles per second). Before the per-bit path was split from the block ends and the search and inlined (D678) it took 58.6 instructions per bit at `sse2`.

**SIMD (D678).** Neither block gains from a SIMD branch of its own. The time of `rds_demod` is in `fir_decimate`, `rotate` and `interleave`, which already have SIMD paths at every level (46 instructions per sample at `avx2` against 138 at `sse2`); the scalar part is `symbol_sync` and `costas` (feedback loops, not accelerated anywhere) and the biphase step at about 1 % of the input rate. `rds_decode` is a bit-serial recurrence over GF(2) at 1187.5 bit/s: each syndrome depends on the one before, and the bit rate is 200 times lower than the multiplex rate of the demodulator in front of it.

## Channelizer blocks and channel banks (M45)

The blocks `channelize` and `channel_select` and the two banks of M45 test 9 are our own code, information only (`Family::Own`, `info` on every level, ours and fallback the same block code; D928, D930). The library comparison of the channelizer's work is the kernel entry `channelize` against liquid-dsp's `firpfbch2_crcf` (D919, the results tables above). Counted on all four platforms on 2026-11-08 (`cargo xtask count --platform <p> --kernel channelize_block --kernel channel_select_block --kernel channel_bank_channelize --kernel channel_bank_ddc --merge`, the levels of x86_64 one at a time), with their cycles simulated by `cargo measure <entry> --no-count --no-golden --platform x86_64 --platform x86_64-avx512 --platform aarch64` (the rows of the cycle tables above).

**Workloads.**
- `channelize_block`: the default block (16 channels, 8 taps per branch, 0.5 / 1.0), 4096 input samples of noise per call, per input sample.
- `channel_select_block`: channel -1 of 16 lanes, 4096 steps per call, per output sample.
- `channel_bank_channelize`: the bank of the flow `pmr446_monitor`. One `ChannelizeBlock` (16 channels, 24 taps per branch, 0.4 / 0.6) and eight `ChannelSelectBlock` (lanes -8 to -1, PMR446 channels 1 to 8), per input sample at 200 kS/s.
- `channel_bank_ddc`: eight `DdcBlock` for the same eight channels from the same 200 kS/s (decimation 8, the -6 dB point at 6.25 kHz with a transition of 2.5 kHz, 60 dB; flat to 0.4 and 60 dB down from 0.6 of the spacing like the channelizer, 290 taps against its 291 Kaiser taps), per input sample.

Both banks give eight channels at 25 kS/s, and both use 4096 samples of noise per call.

Simulated cycles per sample, reference model (whole-loop):

| Entry | `sse2` (`skylake`) | `sse4.2` (`skylake`) | `avx2` (`znver3`) | `avx512` (`znver4`) | `neon` (`cortex-a72`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `channelize_block` | 26.79 | 26.51 | 12.21 | 12.69 | 30.99 |
| `channel_select_block` | 0.60 | 0.60 | 0.76 | 0.28 | 1.27 |
| `channel_bank_channelize` | 46.06 | 45.77 | 21.00 | 792.49 | 60.07 |
| `channel_bank_ddc` | 227.45 | 227.20 | 72.75 | 516.23 | 308.18 |

Simulated cycles per sample, counter-model (whole-loop):

| Entry | `sse2` (`znver2`) | `sse4.2` (`znver2`) | `avx2` (`alderlake`) | `avx512` (`sapphirerapids`) | `neon` (`cortex-a76`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `channelize_block` | 27.83 | 27.84 | 12.87 | 16.13 | 22.71 |
| `channel_select_block` | 0.77 | 0.77 | 0.52 | 1.83 | 1.27 |
| `channel_bank_channelize` | 52.54 | 52.53 | 25.61 | 97.78 | 42.71 |
| `channel_bank_ddc` | 212.36 | 212.46 | 65.45 | 96.78 | 263.36 |

Instructions per sample (`simd128`: wasmtime fuel, no core model, so no cycles):

| Entry | `sse2` | `sse4.2` | `avx2` | `avx2-runtime` | `avx512` | `neon` | `simd128` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `channelize_block` | 94.70 | 94.67 | 51.04 | 50.65 | 57.12 | 43.68 | 270.93 |
| `channel_select_block` | 3.02 | 3.02 | 2.30 | 3.02 | 1.68 | 2.27 | 10.54 |
| `channel_bank_channelize` | 193.80 | 193.79 | 83.51 | 83.64 | 104.25 | 81.92 | 534.31 |
| `channel_bank_ddc` | 721.18 | 721.77 | 234.33 | 241.94 | 178.06 | 436.60 | 1657.26 |

Report cost of the two banks as pipeline graphs (`sdrk-pipeline/tests/pmr446.rs`, `the_cost_of_one_channelizer_against_eight_ddc`), per input sample:

| Graph from 200 kS/s | Report cost per input sample |
| --- | ---: |
| one `channelize` (16 channels, 24 taps per branch) and eight `channel_select` | 35 |
| eight `ddc` (decimation 8, cutoff 6.25 kHz, transition 2.5 kHz) | 1204 |

Remarks:
- **Test 9: the channelizer bank against the `ddc` bank.** Leaving out `avx512` (next remark), the channelizer bank needs 0.16 to 0.39 of the cycles of the `ddc` bank. On the reference model it is 46.06 against 227.45 at `sse2` (0.20), 21.00 against 72.75 at `avx2` (0.29) and 60.07 against 308.18 at `neon` (0.19; 92.68 before the two-hop branch sums of D939). On the counter-model it is 52.54 against 212.36 (0.25), 25.61 against 65.45 (0.39) and 42.71 against 263.36 (0.16; 70.61 before D939). In instructions it is 0.27 (`sse2`), 0.36 (`avx2`), 0.19 (`neon`) and 0.32 (`simd128`). The report cost is a model of the arithmetic, not of the code. It puts the channelizer bank at 0.03 of the `ddc` bank: 24 + 10 for the channelizer plus 1/8 per `channel_select`, against 150.5 per `ddc`.
- **Artifact at `avx512`.** The `avx512` values of both banks are an artifact of the simulation, like the `rep stosb` note of the cycle tables. Both blocks shift their history with `copy_within`. Under SDE's `-spr` CPU, glibc does these copies with `rep movsb`, and SDE counts every byte as one execution of the instruction. `znver4` rates each execution at 100 cycles and `sapphirerapids` at 8.
  - `channel_bank_channelize` copies 502336 bytes in 65536 samples. That is 770 of its 792 cycles on `znver4` and about 61 of its 98 on `sapphirerapids`, so without the copies it is about 22 and 37.
  - `channel_bank_ddc` copies 294400 bytes. That is about 454 of its 516 cycles and about 36 of its 97, so without the copies it is about 62 and 61.
  - `channelize_block` (12.69 on `znver4`, its history of 8 taps per branch) is not affected.
- **The readers.** One `channel_select` costs 0.28 to 1.83 cycles per step, and a step comes every 8 input samples. The eight readers of the bank add about 0.3 to 1.8 cycles per input sample together.
- **D935.** The step slices of `channel_select` against the former `skip` / `step_by` loop, simulated cycles per output sample of `channel_select_block`, whole-loop (the before values simulated on 2026-11-08 with the old loop put back for the measurement):

  | Level | Before, reference model | After, reference model | Before, counter-model | After, counter-model |
  | --- | ---: | ---: | ---: | ---: |
  | `sse2` | 2.28 | 0.79 | 2.80 | 0.91 |
  | `sse4.2` | 2.28 | 1.03 | 2.80 | 1.03 |
  | `avx2` | 1.89 | 1.24 | 1.61 | 2.28 |
  | `avx512` | 1.89 | 0.55 | 1.61 | 3.67 |
  | `neon` | 5.02 | 2.04 | 4.04 | 2.03 |

  Faster in 8 of the 10 cells; the counter models of `avx2` (`alderlake`) and `avx512` (`sapphirerapids`) are slower. `simd128` has no core model: 13.30 instructions per output sample after (no count before). The cycles of `channel_bank_channelize` before D935 were never simulated; only its instructions were counted (201.67 at `sse2` and 91.89 at `avx2`, 194.26 and 84.59 after D935).
- **D941 (I45-2).** `channel_select` copies the lane of each step as a one-element slice (`step.get(lane..=lane)` and `copy_from_slice`, a 64-bit copy; D940 measured it, the maintainer adopted it). With the step slices of D935, LLVM had split the copy into two strided `f32` streams (18 shuffle uops per 8 steps at `avx2`, two serialized `vgatherdps` at `avx512`); the 64-bit copy is pairs of `vmovsd` with `vmovlhps` up to `avx2` and one `vpgatherdq` at `avx512`. Simulated cycles per output sample of `channel_select_block`, whole-loop:

  | Level | Before D935, reference model | D935, reference model | D941, reference model | Before D935, counter-model | D935, counter-model | D941, counter-model |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `sse2` | 2.28 | 0.79 | 0.60 | 2.80 | 0.91 | 0.77 |
  | `sse4.2` | 2.28 | 1.03 | 0.60 | 2.80 | 1.03 | 0.77 |
  | `avx2` | 1.89 | 1.24 | 0.76 | 1.61 | 2.28 | 0.52 |
  | `avx512` | 1.89 | 0.55 | 0.28 | 1.61 | 3.67 | 1.83 |
  | `neon` | 5.02 | 2.04 | 1.27 | 4.04 | 2.03 | 1.27 |

  Faster than D935 in all 10 cells and faster than the loop before D935 in 9 of 10. The exception is the counter-model of `avx512` (`sapphirerapids`), 1.83 against 1.61 (14 % above), where the model gives the gather a latency of 29 cycles and the gathers of successive iterations run one after the other; the maintainer accepts it (D941). The bank of test 9 (`channel_bank_channelize`, cycles per input sample, reference / counter model) goes from 46.24 / 52.66 to 46.06 / 52.54 (`sse2`), 46.19 / 52.78 to 45.77 / 52.53 (`sse4.2`), 21.44 / 27.30 to 21.00 / 25.61 (`avx2`), 791.29 / 99.43 to 792.49 / 97.78 (`avx512`, the `rep movsb` artifact above) and 60.82 / 43.46 to 60.07 / 42.71 (`neon`). The values are bit-identical. `simd128` has no core model: 10.54 instead of 13.30 instructions per output sample. At `avx2-runtime` the block is the baseline x86-64 code (3.02 instructions per output sample, as at `sse2`) against 2.30 of the `avx2` build, a difference of code generation only (information, the waiver of `benches/limits.json`).

## Band pulse detector (M45.1)

The block `band_pulse_detect` is our own code, information only (`Family::Own`, `info` on every level, ours and fallback the same block code; D962): the natural like-for-like reference, `rtl_433`'s `pulse_detect_package`, is GPL and is not used or linked, and no other library has a pulse detector on the lanes of a channelizer, so the entry is judged by its own cycles. Counted on all four platforms on 2026-11-08 (`cargo xtask count --platform <p> --kernel band_pulse_detect_block --merge`, the levels of `x86_64` one at a time), with its cycles simulated by `cargo measure band_pulse_detect_block --no-count --no-golden --platform x86_64 --platform x86_64-avx512 --platform aarch64` (the rows of the cycle tables above).

**Workload.** `band_pulse_detect_block`: the defaults (`auto`, 9 dB) on 16 lanes at 250 kHz, the lanes of the flow `ism433_pulses` prepared once with `ChannelizeBlock` (16 channels, the defaults) from a 2 MHz band of uniform noise with, in every 40 ms period, two PWM transmissions (24 bits, 200 / 400 us pulses, 200 us gaps) at -310 kHz and +190 kHz and one FSK transmission (24 bits of 100 us, 30 kHz deviation) at +560 kHz, all at about 34 dB above the noise in a channel spacing. 256 steps (4096 lane values) per call, read cyclically; per lane value. On average 4.1 of the 16 lanes are busy per step (the three transmissions, their neighbours that see them in the transition band of the channel filter, and the waits after each burst).

Simulated cycles per lane value, whole-loop:

| Model | `sse2` | `sse4.2` | `avx2` | `avx512` | `neon` |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference model | 56.90 (`skylake`) | 57.28 (`skylake`) | 46.83 (`znver3`) | 52.29 (`znver4`) | 88.06 (`cortex-a72`) |
| counter-model | 56.38 (`znver2`) | 56.56 (`znver2`) | 42.65 (`alderlake`) | 47.86 (`sapphirerapids`) | 70.36 (`cortex-a76`) |

Instructions per lane value (`simd128`: wasmtime fuel, no core model, so no cycles):

| `sse2` | `sse4.2` | `avx2` | `avx2-runtime` | `avx512` | `neon` | `simd128` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 121.36 | 120.05 | 112.86 | 119.99 | 112.26 | 110.02 | 266.49 |

Remarks:
- **Where the instructions go.** Profiled with callgrind at `avx2` (200 calls, before the two changes of D963): of the instructions of the calls, about 73 % are the detectors of the busy lanes (the run machines of the envelope and of the presence, the frequency slicer with one `atan2` per sample of a frame and the merge decisions; about 300 instructions per busy lane value), about 26 % the loop over the lanes of each step (the envelope low-pass, the noise level, the trigger and the choice of the lanes to run) and about 1 % `mag`. The efficiency of 0.50 to 0.75 and the bottleneck class `latency` on every model come from the chain of one sample of a lane to the next in the detectors.
- **The report cost** of the kind is 12 per lane value (spec §3); the instructions are about 9 times that, almost all of it in the busy lanes, so a quieter band costs less: with no busy lane a step is the vector loop over the lanes and `mag` only.
- **D970.** The merge decision (`decide`, run once per burst) is kept out of line (`#[inline(never)]`): inlined into the step loop it took registers from the per-sample detectors; simulated cycles per lane value before -> after (reference / counter model, with the merging rules of D964 to D966 in both): no figure before, the instructions were 160.00 -> 121.36 (`sse2`), 149.75 -> 112.86 (`avx2`); against the entry of D963 (before D964 to D966) the cycles fell from 63.49 / 62.76 to 56.90 / 56.38 (`sse2`), 53.91 / 48.75 to 46.83 / 42.65 (`avx2`), 59.64 / 54.07 to 52.29 / 47.86 (`avx512`) and 105.23 / 86.11 to 88.06 / 70.36 (`neon`).
- **D963.** The frequency of every sample of a frame uses a polynomial `atan2` (within 2e-5 rad of `libm::atan2f`), and each step runs only the lanes of a bitmask of busy lanes (plus the lanes that trigger); simulated cycles per lane value after these two changes (reference / counter model): 63.49 / 62.76 at `sse2`, 64.19 / 63.18 at `sse4.2`, 53.91 / 48.75 at `avx2`, 59.64 / 54.07 at `avx512`, 105.23 / 86.11 at `neon` (`simd128` 301.18 instructions, no core model); no cycles were simulated before the changes, so the step is known in instructions only: 145.15 -> 138.17 per lane value at `avx2`.

## Pipeline benchmark

`cargo xtask count --pipeline` runs `examples/flows/fm_receiver.json` (a `cu8` input at 2.4 MHz, the receiver of the examples with its spectrum output) in blocks of 4096 input samples under callgrind (x86-64 with AVX2, differential method) and writes `benches/results/pipeline.json`:

| Measure | Value |
| --- | --- |
| instructions per input sample | 461.2 (545 before the SIMD optimization) |
| latency of `audio` (`Pipeline::latency`) | 30.14 samples of 48 kHz, 627.9 µs (group delay 27.14 samples, up to 3 samples buffered) |

At 2.4 Msps this is about 1.1 G instructions per second of signal.

The same command also counts the SSB chain of M21 (`ddc` at +300 kHz with decimation 50, 4 kHz cutoff and 2 kHz transition, then `agc` with its defaults, then `ssb_demod` USB with its defaults: the receiver of `examples/flows/ssb_receiver.json` without the squelch, at the rates of the end-to-end SSB test), in the same blocks of 4096 input samples; the line is in `chains.ssb` of `benches/results/pipeline.json` (information, no limit):

| Chain | Instructions per input sample | Latency of `audio` |
| --- | --- | --- |
| `ddc` + `agc` + `ssb_demod` | 631.4 | 481.52 samples of 48 kHz, 10031.7 µs (group delay 479.52 samples: 436 from the 873-tap audio filter, the rest from the channel filter; up to 2 samples buffered) |

`ssb_demod` adds no kernel. It calls `rotate`, `fir_decimate` and a second `rotate` once per piece of 256 samples, never per sample, so every call goes through the facade to the vector paths of the backend (the check is `the_block_equals_the_kernel_chain` in `crates/blocks/sdrk-blocks/tests/ssb_demod.rs`, which compares the block with the three kernel calls bit for bit; the differential count above includes the block, and the drift of the FM line from 461.2 to 461.0 is the toolchain of the day). The channel filter of the chain is the same `ddc` as in the FM line, so the difference of the two lines (170 instructions per input sample) is the cost of `agc` and `ssb_demod` at 48 kHz amortised over the 50 input samples per audio sample.

The same command counts the whole blocks `fm_demod` and `fm_stereo_demod` of M24 at 480 kHz (information, no limit; `benches/pipeline`, chains `fm480` and `stereo480`): a `cf32` input at 480 kHz (the E2E-STEREO multiplex signal of 75 kHz deviation, left 1 kHz, right 3 kHz, pilot 0.1) through the block alone, 48 kHz output (`fm_demod` with `audio_decimation` 10, `fm_stereo_demod` with its defaults), in the same blocks of 4096 input samples. Each chain runs twice: with the vector paths (x86-64 with AVX2, the best level) and with every kernel on the fallback (`--cfg sdrk_fallback_only`, the same `target-cpu`); the lines are in `chains.fm_demod_480k` and `chains.fm_stereo_480k` of `benches/results/pipeline.json` (D149):

| Block at 480 kHz | Instructions per input sample, AVX2 | Instructions per input sample, fallback | Latency of `audio` |
| --- | --- | --- | --- |
| `fm_demod` (`audio_decimation` 10, mono) | 1204.0 | 1829.6 | 25.35 samples of 48 kHz, 528.1 us |
| `fm_stereo_demod` (stereo, left in `re`, right in `im`) | 1233.6 | 1731.3 | 25.35 samples of 48 kHz, 528.1 us |

Stereo costs 2.5 percent more than mono with the vector paths and 5.4 percent less on the fallback; the split of either total between the audio filter, `quad_demod`, `stereo_pll` and the matrix was not measured here (the per-kernel entries above are the kernel costs). The GNU Radio `wfm_rcv_pll` chain (pilot band-pass, `pll_refout_cc`, multiplication, low-pass, matrix) that M24 section 8 names as optional information was not measured against it.

## Running the measurements

```sh
cargo xtask count --platform x86_64          # needs valgrind and Intel SDE (SDRK_SDE=<path to sde64>)
cargo xtask count --platform x86_64-avx512   # needs Intel SDE
cargo xtask count --platform aarch64         # needs a QEMU user-mode build with plugins and the aarch64 cross compilers
cargo xtask count --platform wasm32          # needs wasmtime and clang with the wasi-libc headers
cargo xtask count --pipeline                 # needs valgrind
cargo xtask count --platform cortex-m4       # needs qemu-system-arm with plugins, arm-none-eabi-gcc and the newlib headers
```

**One kernel at a glance.** `cargo measure <kernel>` (alias of `cargo xtask measure`) runs every measurement of the [measurement policy](#measurement-policy) that exists for one kernel and prints it as tables; `target/measure/<kernel>.json` gets the same as JSON (`--json` prints it instead of the tables). The name selects every count entry and golden directory that is the name or starts with it and `_` (`biquad_real` selects `biquad_real`, `biquad_real_q15`, `biquad_real_fallback`, `biquad_real_q15_fallback` and the golden cases of `biquad_real`).

```sh
cargo measure biquad_real                          # the platform of the host, every level
cargo measure biquad_real --level avx2             # one level (repeatable)
cargo measure fir --platform x86_64-avx512         # another platform (repeatable)
cargo measure fft --no-count                       # only the golden cases (--no-golden: only the counts)
cargo measure fft --no-count --no-golden           # only the cycles of every level
python3 -I benches/cycle_tables.py target/measure/fft.json   # merge those rows into the cycle tables below
```

- **Counts:** the judged records of `cargo xtask count` for the selected entries (instructions of ours and of the comparison, the ratio, the status and verdict of `benches/limits.json`, the SNR of both against the f64 model). It needs the counters of the level, like `cargo xtask count`; a platform that cannot be counted is reported as not measured. `benches/results` is not written, and the informational variants are not run.
- **Golden cases:** the golden tests of `sdrk-testkit` (the host build, which detects the level at run time) with the report of `sdrk_testkit::golden` (`SDRK_GOLDEN_REPORT`, [tests/golden/README.md](../tests/golden/README.md)). Per case: the contract floor, the measured SNR (the lowest of all comparisons of the case), the achievable optimum (the expected values rounded once to the output format: the sample type, or for a block-floating-point `CQ15` output a Q15 mantissa with the block exponent the expected values need) and the distance to it ([kernels.md](kernels.md#accuracy-policy)).
- **Cycles** (level 2, report only; `--no-cycles` skips them): for every level with a core model (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`; D68), the execution counts of ours and of the judged comparison from the counter of the level in the environment of its count, simulated by `llvm-mca` on the reference model and the counter-model of the level ([how](#how-the-cycles-are-simulated)). Per entry, cycle mode (`whole-loop` and `per-block`, the one of `--cycle-mode` first; default `whole-loop`) and model: the cycles per sample of both sides, their ratio, our efficiency, the bottleneck class and the coverage. The JSON of every entry names the counter (`block_counts`); an entry that cannot be simulated gets a `note` (in the table too), an entry that is not measured at the level the note `not measured at the level`. `simd128` is listed as not simulated. `--cycle-counter sde` takes the counts of `avx2` from SDE (`-hsw`) instead of callgrind (validation, D68). Only the ratios mean something; the cycles are no hardware figures.

The command fails (exit code 1) when an entry is judged `fail`, a platform cannot be counted or a golden case fails.

Options of `cargo xtask count`: `--level L` and `--kernel NAME` (repeatable) restrict the run (the results file is written only for a complete run), `--no-write` keeps `benches/results` unchanged, `--no-variants` skips the informational misaligned and small-block runs, and `--strict` fails for every entry that is not a clean `release` entry (the release check of open item M13-4).

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

The count jobs run in a pinned container image (`ubuntu:26.04` by digest) with the packages of a pinned apt snapshot, so that the compilers of the C references do not change under the counts; the references are cached by the build script, the image and the snapshot. That includes the nightly Cortex-M4 job `count_m4_qemu` (since 2026-10-28; before, it ran on the bare runner with that runner's unpinned `gcc-arm-none-eabi`): the snapshot has `gcc-arm-none-eabi` 15:14.2.rel1-1, the compiler of the committed CMSIS-DSP numbers.

- The instruction counts run in the nightly and manual runs only (D175). A pull request that touches kernels, backends, `benches`, `xtask` or the toolchain only checks that the measurement harness builds (`cargo check --manifest-path benches/Cargo.toml --workspace --all-targets` in `check`, without the C references); a branch that needs the counts before its merge runs `gh workflow run nightly.yml --ref <branch>`. The misaligned and small-block variants are information and are not judged.
- In the nightly run `count_x64` measures `x86_64` with the variants and the pipeline benchmark, and `count_x64_linux_sde_avx512`, `count_arm64_linux` and `count_wasm32` the platforms `x86_64-avx512`, `aarch64` and `wasm32` (`tree.yml`, [testing.md](testing.md)). They fail only when `cargo xtask count` fails (a limit, an accuracy check or a measurement error); the difference of the fresh results against the committed `benches/results` is shown for information (`git diff`) and the fresh files are kept as an artifact (`count-results-<job>`). The counts drift with the runner (the CPU name in `host_cpu`, and for `x86_64` the counts themselves, because the pool mixes EPYC 7763, 9V45 and 9V74) and with the `stable` toolchain of the day (for example 0.01 instructions per sample in wasm32), so a byte-for-byte comparison fails without a change of the code. Instruction counts are deterministic for one machine and one toolchain. `xtask` removes every `QEMU_*` variable from the environment of `qemu-aarch64`, because qemu-user reads them as options (`QEMU_VERSION` is `-version`).
- wasmtime is installed from its GitHub release tarball, pinned by version and SHA-256 (`WASMTIME_SHA256`).

## Cortex-M4 (fallback, report only)

`cargo xtask count --platform cortex-m4` counts nineteen kernels of the portable fallback on a Cortex-M4F in `qemu-system-arm` (the `insn` plugin) and writes `benches/results/cortex-m4.json`; the nightly job `count_m4_qemu` runs it. The method is the differential one above, with the workload binary `count` of `targets/qemu-cortex-m4`. The report has no limits and no verdicts. Details in [embedded.md](embedded.md#instruction-counts-cortex-m4).

Since M22 (open item M22-10, D27) the four biquad kernels are in the report next to CMSIS-DSP 1.18.0 built for the M4F with the DSP extension and the FPU (the tarball and SHA-256 of the host build, plus the CMSIS-Core headers of CMSIS 6.3.0; `benches/refs/build.sh cortex-m4`). Both sides filter the input of the host counts (the generator, seeds, filter and 4096 samples per call of `benches/kernels`) and are compared on the same input with the f64 model of `benches/kernels`; our SNRs are those of the host counts to the last digit, which confirms the input. Our side is called through a function that is never inlined, with the coefficients behind `black_box`, as on the host (CMSIS-DSP sees them at run time too). Both sides are compiled for the same core (D30): the workload with `-C target-cpu=cortex-m4` (`targets/qemu-cortex-m4/.cargo/config.toml`, recorded as `build.target_cpu` in the report), CMSIS-DSP with `-mcpu=cortex-m4`. Without the flag LLVM tunes for a generic Armv7E-M and turns every non-fused multiply-add of the `f32` kernels into one `vmla.f32`/`vmls.f32` (3 cycles on the M4 against 1 + 1 for `vmul.f32` and `vadd.f32`, which GCC's Cortex-M4 tuning selects), so the instruction counts compared instruction selections, not work: until 2026-10-28 the report had 36.77 (0.82), 73.29 (1.11), 13.88 (0.62) and 33.53 (1.02) in the order of the table.

| Kernel (2 sections) | Ours | CMSIS-DSP | Ratio | SNR ours / CMSIS-DSP (dB) |
| --- | --- | --- | --- | --- |
| `biquad` C32 (CMSIS-DSP `arm_biquad_cascade_stereo_df2T_f32`, re and im as the two channels) | 50.27 | 45.00 | 1.12 | 136.06 / 134.90 |
| `biquad` CQ15 (`arm_biquad_cascade_df1_q15`, postShift 1, two calls on planar re and im, deinterleaving not counted) | 82.05 | 66.05 | 1.24 | 71.59 / 56.73 |
| `biquad_real` f32 (`arm_biquad_cascade_df2T_f32`) | 21.88 | 22.50 | 0.97 | 136.72 / 136.72 |
| `biquad_real` Q15 (`arm_biquad_cascade_df1_q15`, postShift 1) | 37.78 | 33.03 | 1.14 | 71.45 / 56.63 |

The report has no limits, so these ratios get no verdict; measured against the goal of 1.00 that the host entries have, `biquad` C32 (1.12), `biquad` CQ15 (1.24) and `biquad_real` Q15 (1.14) miss it and `biquad_real` f32 (0.97) meets it. The gap is expected and reported as information (D31): the Cortex-M4 has no SIMD or DSP-extension backend yet (M18), so the portable scalar fallback is compared with CMSIS-DSP code that uses the DSP extension. Whether the comparison is like with like: it is the same algorithm (transposed direct form II in `f32`, direct form I in Q15 with a 64-bit accumulator), the same core and the same instruction set (`thumbv7em-none-eabihf` has the DSP extension and the FPU, so both compilers may use `smlal`, `ssat` and the FPU), and CMSIS-DSP is the established library for it; but CMSIS-DSP's Q15 cascade is hand-written for the DSP extension (`__SMLALD`, two 16 x 16 bit products per instruction on packed sample pairs), an optimization that our portable fallback by design does not contain (no target-specific code, one `smlal` per product). The like-with-like counterpart of that path is a biquad of the `arm-dsp` backend (milestone [M18](milestones/M18-arm-dsp-backend.md), not implemented yet; its kernel list has no biquad), not the fallback; the Q15 rows therefore compare the portable fallback with the best established implementation for the core and are information about how far the fallback is from it. CMSIS-DSP truncates where ours rounds (14.8 to 14.9 dB more SNR for ours). On 32-bit targets other than wasm32 our Q15 cascade keeps the state in `i32` and runs the two components of `CQ15` in two passes (D28); with the `i64` state of the 64-bit targets it counts 322.06 (`CQ15`) and 103.28 (`Q15`) instructions per sample here (326.06 and 103.78 with the generic tuning; the 64-bit products of `i64` operands are full 64 x 64 bit multiplications on the M4).

Note that the M4F has a single-precision FPU only: `convert_out_cs16` rounds in `f64` (R7) and so counts about a thousand instructions per sample in software floating point.
