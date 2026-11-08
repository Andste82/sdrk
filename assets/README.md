# Assets

## `llvm-mca-23.1.1-sdrk-sched.patch`

Corrections of the scheduling models of LLVM 23.1.1 that SDRk's cycle simulation uses
(`cargo measure`, [docs/performance.md](../docs/performance.md#speed-simulated-cycles-per-sample-level-2),
decision D942 in [docs/decisions-log.md](../docs/decisions-log.md)). The development container
applies it when it builds `llvm-mca` (`.devcontainer/Dockerfile`): it builds the release source
once, installs that tool as `/opt/llvm-mca-23.1.1-upstream/bin/llvm-mca`, applies the patch,
rebuilds and installs the corrected tool as `/opt/llvm-mca-23.1.1/bin/llvm-mca` (`SDRK_LLVM_MCA`).

To apply it by hand: `patch -p1 < assets/llvm-mca-23.1.1-sdrk-sched.patch` in the root of
`llvm-project-23.1.1.src`.

What it changes, and only in the models SDRk simulates (`skylake`, `znver2`, `znver3`,
`alderlake`, `znver4`, `sapphirerapids`, `cortex-a72`, `cortex-a76`):

- numbers of existing scheduling definitions, and per-opcode definitions (`InstRW`) for the
  instruction forms of the simulated loops whose throughput or latency differed clearly from
  the measurement;
- two resources the models lacked to express a measured limit: one store port in the Znver2
  model (Znver3 and Znver4 have it) and one FP divider in the Alder Lake P and Sapphire Rapids
  models (the Skylake model has it).

Sources, named in an `SDRk:` comment at every change:

| Models | Source |
| --- | --- |
| x86 | uops.info `instructions.xml` of 2026-03-29, SHA-256 `e5e702caaf04c4fc1f75192a9bbe4d8554b3e174f7933a14d124cc33ca677ab3` (for `sapphirerapids` the values of EMR) |
| `cortex-a72` (LLVM's A57 model) | Arm Cortex-A72 Software Optimization Guide, ARM UAN 0016A |
| `cortex-a76` (LLVM's Neoverse N1 model) | Arm Neoverse N1 Software Optimization Guide, PJDOC-466751330-9707, issue 4.0 |

Memory forms keep the latency of the model (uops.info gives only upper bounds for memory
operands); register forms take the measured latency. Data-dependent latencies and throughputs
(divide, square root) take the slowest value of the source.
