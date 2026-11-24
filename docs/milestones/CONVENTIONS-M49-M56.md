# Shared conventions for M49 to M56 (revision 2026-11-17)

These are the definitions that the revised specs M49 to M56 share. They come from the maintainer decision of 2026-11-17, which accepted all nine recommendations of the algorithm review. Each revised spec states the parts it uses in the same words, using the canonical sentences at the end of each section. The milestone that introduces a definition (table in §0) also writes its full text into `docs/kernels.md`, and later milestones refer to that section and to the introducing milestone instead of repeating the details.

Measured facts from the review (NumPy checks during the review) that motivate the rules, cited as measured:
- `COS_Q15[0] = 32767`, so a `CQ15` scramble round trip with the `rotate` product loses up to 2 LSB at `b = 1, 2, 3`.
- A `+1` chip of 32767 changes 32768 of the 65536 `Q15` inputs.
- The `CQ15` inverse `fwht` with saturation per stage and `/N` at the end has errors up to 3643 LSB at `N = 64` and amplitude 4095, and is exact at amplitude 511.

## 0. Where each item is introduced

| Item | Introduced in | Used by (reference only) | Code home |
| --- | --- | --- | --- |
| §A exact `±1`/`±j` operations | **M49** (`scramble`, `b ≤ 2` and the quadrant part of `b ≥ 3`) | M51 (binary chips), M52 (binary chips in the sum), M54 (lag sums) | `sdrk_core::num` (scalar), `sdrk-backend-simd` module `sign` (vector helpers) |
| §B bit-packed code format | **M51** (`spread`) | M52, M53 (kernel constructor and block), M54; M50 uses only §B.1 (chip order and mapping), not the type | `sdrk_core::kernels::code` (type), `sdrk_proto::prbs` (generator into words) |
| §C.1 accumulation output shift | **M52** (`despread`) | M54 (a later `CQ15` variant, backlog), M50 (backlog note) | `DespreadState` |
| §C.2 per-stage transform shift | **M56** (`fwht`), taken over from M07 (`fft` `CQ15`) | the M50 and M53 backlog notes (`CQ15` via the M07 block-floating-point FFT) | `kernels::fwht` |
| §D SIMD rule | the maintainer rule of 2026-11-17, first written into M48 §5 | every milestone M49 to M56, each with its own SIMD plan in its cost section | — |

Dependency lines that follow from this table:
- M51 already depends on M49, and M52 and M54 depend on M51 through M52.
- **M53's Depends line becomes M07, M51** (the code format of §B). M50 is removed from it (decision 7).
- M50, M55 and M56 need no new dependency.

## A. Exact `±1` and `±j` operations

A unit phasor or chip in `{+1, −1, +j, −j}` is applied by sign flips and re/im swaps, **never by a complex product**. In `CQ15`, `+1` cannot be represented (a product with 32767 scales by `1 − 2⁻¹⁵`). In `C32`, a product with a zero component turns `±∞` into NaN (`∞ · 0`) and changes signed zeros.

### A.1 `CQ15`

- **Saturating negation:** `q15_neg_sat(v) = (v == −32768) ? 32767 : −v`, which is Rust's `i16::saturating_neg`. This is the existing rule of `real_to_complex` (`−(−32768) = 32767`).
- **Quadrant map** `cq15_quadrant(x, q)`: multiplication by `j^q`, `q` taken modulo 4. Each `−` is `q15_neg_sat`.

  | `q` | factor | result `(re, im)` |
  | --- | --- | --- |
  | 0 | `+1` | `(x.re, x.im)` |
  | 1 | `+j` | `(−x.im, x.re)` |
  | 2 | `−1` | `(−x.re, −x.im)` |
  | 3 | `−j` | `(x.im, −x.re)` |

- **Inverse (conjugate):** `q' = (4 − q) & 3`. The round trip `cq15_quadrant(cq15_quadrant(x, q), q')` returns `x` exactly, with one exception: a component equal to −32768 that is negated on the way returns as −32767 (1 LSB, and only at that value). The negated components are `re` for `q ∈ {2, 3}` and `im` for `q ∈ {1, 2}`; each such component is negated twice (once in each direction), and the first saturating negation already turns −32768 into 32767. `q = 0` changes nothing; `q = 2` (also the chip `b = 1` applied twice) affects both components, `q = 1` only `im`, `q = 3` only `re`.
- **Chip:** `cq15_chip(x, b)` applies the chip value `1 − 2b` (§B.1): `b = 0` gives `x`, `b = 1` gives `cq15_quadrant(x, 2)`.
- **Inside a wide accumulation** (§C.1) the sign is plain integer negation in `i32`/`i64`, without saturation. `−(−32768)` is `+32768` there, because the value is not stored as `Q15`. `q15_neg_sat` applies only where a result is **stored** as `Q15` (`scramble`, `spread`).
- **SIMD forms (same bits):**
  - Unconditional negation: x86 `psubsw(0, x)`, NEON `vqnegq_s16`, SIMD128 `i16x8.sub_sat_s(0, x)`.
  - Conditional negation by a lane mask `m ∈ {0, −1}`: `subs(xor(x, m), m)`. With `m = −1` this is `~x + 1` saturated, which gives 32767 for −32768.
  - Swap re/im: a 16-bit shuffle within each 32-bit pair.
  - Not allowed: `psignw`/`vpsignw` (it wraps −32768 to −32768).

### A.2 `C32`

- **Negation** is IEEE negation (`-x` in Rust): only the sign bit flips. It is exact for every value: `±0 → ∓0`, `±∞ → ∓∞`, and NaN stays NaN with its payload and a flipped sign. Not allowed: `x * −1.0` and a complex product with `(−1, 0)`.
- **Quadrant map** `c32_quadrant(x, q)`: the table of §A.1 with IEEE negation. A component that is only swapped keeps its bits.
- **Chip:** `c32_chip(x, b)` gives `x` for `b = 0` and `(−x.re, −x.im)` for `b = 1`.
- **Inside a sum:** `acc + (−x)` and `acc − x` give the same bits for every non-NaN result (IEEE 754 defines subtraction as addition of the negation). They differ for NaN: on x86 `a − NaN` returns the NaN unchanged, `a + (−NaN)` returns it with the sign flipped (measured with NumPy in the M51 revision check). A sum or product kernel therefore follows D296 (NaN sign and payload are not part of the contract, NaN lanes compared by class); only the pure sign and swap operations below are compared by `to_bits()` with NaN included.
- **SIMD:** XOR with `0x8000_0000` (masked per lane for a conditional sign), and a 32-bit shuffle for the swap. Tests of the pure sign and swap operations compare `to_bits()`, NaN included. The operation is fully determined, so the D296 relaxation is not needed for them; it applies wherever a kernel adds or multiplies (sums of M52 and M54, products of `Code::Complex`, the product of `scramble` at `b ≥ 3` (M49.1), `fwht` `C32`).
- **Infinity behaviour, stated in every contract that uses §A:** a binary chip or quadrant phasor applied to `±∞` gives `±∞` (sign per the table), never NaN. The general complex-code path (§B.4) keeps the product behaviour (`∞ · 0 = NaN`). Its results for a code `(±1, 0)` can therefore differ from the binary path in non-finite inputs and signed zeros, and in `CQ15` in every value whose magnitude is above about 16384.

### A.3 Scalar API (introduced by M49, `pub`, documented "for backend crates and tests")

```rust
pub const fn q15_neg_sat(x: Q15) -> Q15;
pub fn cq15_quadrant(x: CQ15, q: u32) -> CQ15;  // multiply by j^(q & 3), §A.1
pub fn c32_quadrant(x: C32, q: u32) -> C32;     // multiply by j^(q & 3), §A.2
pub fn cq15_chip(x: CQ15, b: u32) -> CQ15;      // chip value 1 - 2(b & 1)
pub fn c32_chip(x: C32, b: u32) -> C32;
```

All of them are in `sdrk_core::num`. The vector helpers go into `sdrk-backend-simd`, module `sign`, written once against `fearless_simd`, with level paths only where the portable operations are not enough (D1).

### A.4 How M49 uses §A (for reference by the other specs)

- Index `k` with `b` phase bits; `inverse` replaces `k` by `(−k) & (2^b − 1)` and applies the same rule.
- **The quadrant split applies to `b ≤ 2` only** (revised by M49.1, maintainer decision 2026-11-18): `q = 2k` for `b = 1`, `q = k` for `b = 2`, and the output is `quadrant(x, q)`: exact, no multiplication; the round trip is exact except for −32768 (§A.1).
- `b ≥ 3` (M49.1): one table phasor and one product per sample, no quadrant map: the `rotate` product of `x` with the table phasor at the full phase `k << (32 − b)` (`COS[k << (12 − b)]`, `COS[((k << (12 − b)) + 3072) & 4095]`); the output equals one `rotate` step at that phase bit for bit and keeps the `rotate` accuracy for every `k`, `k = 0` included (which multiplies by `(32767, 0)` in `CQ15`). The `CQ15` round trip is within 2 LSB per component for `|x| ≤ 1` (measured).
- (M49 as first merged split every `b ≥ 2` into a quadrant and a residue and selected the input where the residue was 0; M49.1 dropped that for `b ≥ 3`.)

**Canonical sentence:** "`±1` and `±j` are applied by the exact sign and swap operations of the shared conventions §A (`q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, introduced in M49), never by a product. In `CQ15` a stored negation saturates (`−(−32768) = 32767`); inside a wide accumulation it is plain integer negation."

## B. Bit-packed code format

### B.1 Chip order and value (applies to every code of M50 to M56)

- **Value:** chip bit `b ∈ {0, 1}` has the chip value `1 − 2b`: `0 → +1`, `1 → −1`. The specs M50, M51, M53 and M54 already write `1 − 2b`, so this is unchanged. It is the usual DSSS/GNSS convention: the product of two chips is the XOR of their bits.
  - It differs on purpose from the data mapping of `psk_mod` and `symbol_map` (BPSK `2b − 1`: bit 1 → `+1`), which stays unchanged. A code is not data.
- **Order:** chip `i` (0-based) of a PRBS code is the `i`-th value returned by `sdrk_proto::prbs::Prbs::next_bit` after `Prbs::new(poly, seed)`. The PRBS register convention is that of `prbs.rs`: bit `k` of the state is stage `k + 1`, and the output becomes stage 1.

### B.2 Packed form

- A code of `len` chips is stored in `n_words = ceil(len / 32)` words of `u32`.
- Chip `i` is bit `i & 31` (LSB first) of word `i >> 5`: `chip(i) = (words[i >> 5] >> (i & 31)) & 1`. Packing: `words[i >> 5] |= b_i << (i & 31)`.
- Because of R6.6 (little-endian), the words as bytes are LSB-first chips in byte order: byte `j` holds chips `8j … 8j + 7`, LSB first.
- **Partial last word:** the bits `len & 31 … 31` of the last word (when `len` is not a multiple of 32) must be 0.
  - `CodeBits::new` checks this (`InvalidParameter`, nothing built), so the packed form of a code is unique.
  - Kernels never use those bits as chips. A SIMD path may load whole words, but it masks or stops at `len`.
- **Size:** `prbs15` (32767 chips) takes 1024 words = 4 KiB, against 256 KiB as `C32` and 128 KiB as `CQ15`.
- **Golden cases:** they store a packed code as the array `code_words` (`u32` little-endian `.bin`) and the JSON field `code_len`. The Python generator packs with the rule above.

### B.3 Length, period and position

- `len` is 1 to `u32::MAX`. `words.len()` must equal `ceil(len / 32)` (`LengthMismatch`).
- The code is periodic with the period `len`. Short and long indexing are those of M51 §1.
- The state keeps `code_pos = chip_index mod len` as a `u32` and advances it with compare-and-subtract at the period end. A `u64` modulo runs only in `new` and `seek`, never per chip (also not in `Fallback`).
- The word and bit cursor are `code_pos >> 5` and `code_pos & 31`. When `len` is not a multiple of 32, the period seam is not word-aligned. Fallback and SIMD paths process runs that end at the period seam and at symbol boundaries, and a run crossing the seam continues at chip 0 of word 0.

### B.4 Types (introduced by M51 in `sdrk_core::kernels::code`)

```rust
/// A binary code: `len` chips packed LSB first into `u32` words (§B.2), chip value 1 − 2b.
#[derive(Clone, Copy, Debug)]
pub struct CodeBits<'a> { /* words: &'a [u32], len: u32 */ }
impl<'a> CodeBits<'a> {
    pub fn new(words: &'a [u32], len: u32) -> Result<Self>; // len >= 1; LengthMismatch; unused bits 0
    pub fn len(&self) -> u32;
    pub fn words(&self) -> &'a [u32];
    pub fn chip(&self, i: u32) -> u32;                      // 0 or 1, i < len
}
/// The two forms of a code; every kernel with a code takes this.
#[derive(Clone, Copy, Debug)]
pub enum Code<'a, T> { Bits(CodeBits<'a>), Complex(&'a [T]) }
impl<T> Code<'_, T> { pub fn len(&self) -> u32; }
```

- **`Code::Bits` (binary form):** each chip is applied by §A (`c32_chip`, `cq15_chip`; in sums by the sign of §A.1 and §A.2). This form is exact and symmetric, and it needs no multiplication.
- **`Code::Complex` (general form):** any complex chips (phasor codes, I/Q codes, the QPSK case of M51). It keeps the `mul` product of the kernel (for despreading the product with `conj(code)`).
  - The two forms are separate contracts. A complex code `(±1, 0)` equals the binary form for finite `C32` input apart from signed zeros, and it does not equal it in `CQ15` (§A.2).
  - The blocks use `Bits`. The `Complex` form is for callers of the Rust API (`SpreadBlock::with_code` and similar).
- **Generator (M51, `sdrk-proto`):** `Prbs::fill_words(&mut self, words: &mut [u32], chips: u32) -> Result<()>` writes the next `chips` bits in the order of §B.1 and sets the unused bits to 0. `words.len()` must be `ceil(chips / 32)` (`InvalidArgument` otherwise). A word-parallel LFSR is allowed if it gives the same words. `sdrk-proto` cannot depend on `sdrk-core` (R2), so it writes plain `u32` words, and the block wraps them in `CodeBits::new`.
- **M50 and M53 blocks:**
  - M50 (`correlate`) keeps no code after building its taps. It only follows §B.1, so it needs neither the type nor M51.
  - M53 (`code_search`) takes the code as `Code<'_, C32>` in its constructor, which builds the code spectrum once. Its block generates `prbs` into words with `fill_words`.

**Canonical sentence:** "The code is a `Code<'_, T>` (shared conventions §B, introduced in M51): `Code::Bits` is the bit-packed binary form (chip `i` = bit `i & 31` of word `i >> 5`, chip value `1 − 2b`, chip `i` = the `i`-th `Prbs::next_bit` of a PRBS), applied by the exact sign operations of §A; `Code::Complex` is the general form with the `mul` product."

## C. `CQ15` growth and scaling

`CQ15` values that grow by a known factor are scaled by a shift with R7 rounding: `round_shift(v, s) = (v + 2^(s−1)) >> s` (round half up, `sdrk_core::num::round_shift`), then saturated **once** with `sat16`. Nothing saturates before the scaling.

### C.1 Accumulations: output shift (introduced by M52; `despread`)

- The sum over `sf` chips is accumulated **exactly** in `i64`, as the `Acc64` rule of R7:
  - binary chips (`Code::Bits`): each term is `±x` in `Q15` units, using the integer sign of §A.1;
  - complex chips: each term is the `Q30` product `x · conj(c)` per component, formed as in `dot` `CQ15`.
- Output per component: `sat16(round_shift(acc, f + shift))`, with `f = 0` for binary terms and `f = 15` for `Q30` products. The value is `(Σ terms) / 2^shift`.
- **`shift`:** a parameter of the state, 0 to 16 (`InvalidParameter` otherwise). The default is `ceil(log2 sf)` (`sf` 2 → 1, 15 → 4, 16 → 4, 31 → 5, 1023 → 10, 65536 → 16).
  - With the default and inputs of complex magnitude at most 1 (per-chip term at most 1 per component), the output cannot overflow. The only exception is the exact value `+32768`, which saturates to 32767.
  - The state reports the resulting gain `sf / 2^shift`. For a power-of-two `sf` the default gives 1, so `spread` → `despread` returns the symbols up to the rounding. For `sf = 15` it gives `15/16`.
  - `C32` has no shift and returns the plain sum.
- **Order:** integer sums are exact, so every summation order gives the same bits as long as no intermediate overflows. A SIMD path may keep per-lane partial sums in `i32` only where a bound shows they cannot overflow, and widens to `i64` otherwise:
  - binary terms: `|x| ≤ 32768`, so an `i32` lane holds 65535 terms;
  - `Q30` products: `|xr·cr + xi·ci| ≤ 2^31`, so they need `i64`, or the `pmaddwd` split of `dot` `CQ15`.
- The state holds the running accumulators (two `i64` for `CQ15`; for `C32` the eight partial sums per component of §D.2) and the chip count in the window, not a chip history. Partial symbols carry over between calls, and `min_input` is 1 (decision of the review, M52).

### C.2 Transforms: per-stage shift (introduced by M56; the M07 FFT rule)

- A butterfly output is `sat16(round_shift(a ± b, s))`, computed in `i32`, with `s ∈ {0, 1}` per stage. This is the rule of the M07 `CQ15` FFT (`kernels.md`, "Fallback order of operations" (4)), without twiddles.
- **Schedules:**
  - *none*: `s = 0` in every stage, so the result is the unnormalised transform. It saturates when the growth exceeds the range (documented).
  - *fixed*: `s = 1` in every stage, so the result is the transform divided by `N`.
  - *block floating point*: before each stage, `m` is the largest absolute component of the whole vector (`|−32768| = 32768`), and `s = 1` for `m ≥ 16384`, else 0. A Walsh–Hadamard butterfly at most doubles `m`, and there is no twiddle, so one bit per stage is enough. This differs from the FFT's 0, 1 or 2.
- **Exponent:** `E = Σ s`, returned as `FftScale { exponent }` with the FFT meaning: `output · 2^E` equals the unnormalised transform `H·x` up to the per-stage rounding.
- **Saturation:** with `s = 1` the only saturating case is `a − b = 65535` (`a = 32767`, `b = −32768`), which gives 32767. Every other sum and difference fits after the shift.
- **Rounding:** a stage with `s = 1` adds an error in `[−1/2, +1/2]` LSB (closed interval): round half up gives exactly `+1/2` for every odd `a ± b`, and `−1/2` occurs only in the saturating corner `a − b = 65535`. A stage with `s = 0` adds no error. A later shifted stage halves the sum of two errors, a later unshifted stage doubles it at most.
  - *fixed*: the error is at most `log2(N) / 2` LSB. The bound is tight: the review's NumPy check reached exactly `log2(N) / 2`.
  - *block floating point*: the error is at most `Σ_i 2^(u_i) / 2` LSB, summed over the shifted stages `i`, where `u_i` is the number of unshifted stages after stage `i`. With every stage shifted this is the *fixed* bound. The review's NumPy check (`N` = 2 to 1024, amplitudes up to full scale) found no violation; the largest error was 12.4 LSB.
  - *none*: no rounding; the result is exact unless it saturates.
- **Inverse:** `H_N⁻¹ = H_N / N` is exactly the *fixed* schedule, so the `CQ15` inverse of M56 is the fixed schedule (exponent `log2 N`). It replaces "saturate every stage, divide by `N` at the end".
  - Forward *none* followed by the inverse returns `x` exactly (bit for bit) when the forward did not saturate. Reason: the first `k` stages of the inverse form `I_{N/2^k} ⊗ H_{2^k}`, and `H_{2^k}² = 2^k I`, so the value before each shift is an even integer and every rounding shift is exact. The review's NumPy check confirmed it on 1200 random unsaturated round trips (`N` = 2 to 4096), all exact.
  - The forward offers *none* (the current unnormalised forward, still the default), *fixed* and *block floating point*. The M56 spec fixes the name of the parameter. The default is not hidden in the direction.
- **Cortex-M4 note:** `SHADD16` truncates (rounds down), so the rounding shift needs `QADD16`/`QSUB16` plus a correction, a few instructions more (the review's estimate, to be confirmed).

### C.3 Overlap-save and PCPS in `CQ15` (backlog of M50 and M53, not built now)

Plan recorded in both backlogs:
- the M07 block-floating-point FFT forward and inverse;
- the code or tap spectrum stored as `CQ15` with its own exponent;
- the block exponents added up and applied to a caller-chosen fixed output scale with one saturating rounding shift (§C);
- tolerance class.

The same pieces exist for a `CQ15` `code_search` (`rotate` `CQ15`, the block-floating-point FFT, `mag_sq` into a grid with a per-row exponent).

**Canonical sentence:** "`CQ15` growth follows the shared conventions §C: sums are exact in `i64` and leave through one rounding shift `round_shift(acc, f + shift)` with one saturation (default `shift = ceil(log2 sf)`, introduced in M52); transforms shift by 0 or 1 per stage with R7 rounding and one saturation, the exponent returned as `FftScale` (the M07 FFT rule, introduced for `fwht` in M56)."

## D. SIMD rule (maintainer rule, revised 2026-11-17)

### D.1 The rule

1. **SIMD wherever it is algorithmically feasible.** A well-made SIMD path is always at least as good as the scalar fallback. Every kernel of M49 to M56 whose algorithm allows lane-parallel work gets SIMD paths **in its own milestone** (in scope, not "allowed later").
   - The paths are in `sdrk-backend-simd` on every level: `sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`.
   - Each path is **bit-identical to the fallback**, and `backend_for` reports the SIMD level.
2. **Contracts give SIMD the fallback bits.** Each contract fixes an element-wise formula or a fixed reduction order (§D.2), so that bit identity follows from the arithmetic and does not depend on luck.
3. **Not feasible means a stated reason.** Where SIMD is genuinely not algorithmically feasible, the spec states the concrete reason (for example, a strictly serial recursion with no lane-parallel part). The lane-parallel parts of the same kernel still get SIMD. "Not accelerated" without a reason is not allowed.
4. **D321 is a quality gate, not a reason to drop a path.**
   - A SIMD path slower than its fallback in simulated cycles per sample, on the reference model or the counter model, is improved, not removed.
   - Only after several measured failed attempts (each recorded with its numbers in `docs/decisions-log.md`) may that level dispatch to the fallback (`backend_for` reports `fallback` there), with an open item in `docs/open-items.md`.
   - A delegated level passes `cargo xtask check cycles`, and a `cycle_waiver` names the open item where one is needed.
5. **Measurement:**
   - `cargo measure` gives the simulated cycles per sample (per output sample, chip or element, as the entry defines) on every level with a core model, on the reference and the counter model.
   - Each SIMD entry `<k>` is compared with its own `<k>_fallback` (the D321 floor) and with the **fair reference** of D32 at the same level (same work and algorithm class). A library row that does different or less work is marked so ("reference does different work", "reference does less work"), as in M48 §5, and is information, not a gap.
   - Cycles are the only speed measure on the out-of-order levels (M13.1, D2700): no instruction counts are taken there. `cargo xtask count --platform wasm32 --merge` gives the instructions at `simd128` and `cargo xtask count --platform cortex-m4` the Cortex-M report; instruction counts stand only where no core model exists.
   - Numbers go into `docs/performance.md`. Estimates in the specs are marked "(to be confirmed)".
6. **Tests** (`sdrk-backend-simd/tests/<kernel>.rs`; pattern of M48 test 9):
   - every level with a path equals `Fallback` bit for bit, on lengths 0 to 67 and 4096 and on the special values of the kernel;
   - the seams of the kernel: the period seam of a code (§B.3), word boundaries, symbol and call boundaries, the vector tail;
   - the state after each call;
   - the golden cases through each level.

### D.2 Contract patterns that keep SIMD bit-identical

- **Element-wise:** one fixed formula per element (operand order, rounding points), **no FMA** (R6.5).
  - A bit-exact kernel must not call a backend entry that is tolerance at some level. Examples: the `mul` `C32` SIMD entry with FMA at `avx2`, `avx512` and `neon` (D326), the `dot` (real) SIMD sums, the `C32` `fft`.
  - It writes the product of its contract itself; the fallback product of `mul` is two multiplications and one subtraction or addition per part.
- **`f32` reductions:** the `dot_complex` order of M43 (D801).
  - Eight partial sums per component, each starting at `+0.0`; term `i` goes to partial sum `i mod 8`.
  - The fixed tree: `s[k] = acc[k] + acc[k+4]`, `t[k] = s[k] + s[k+2]`, result `t[0] + t[1]`.
  - The term index `i` counts from the start of the reduction window (symbol, period, pilot list), never from the start of a call, so chunking does not change the bits.
  - Padding a partial vector with `+0` terms is allowed only where the kernel shows that it gives the same bits, as `dot_complex` does; otherwise the tail runs the scalar code.
- **Integer reductions:** exact, so any order is allowed, with a width bound (§C.1).
- **Transforms with fixed operands per output** (`fwht`): every output of a stage is one addition or subtraction of two fixed operands. Every lane schedule, stage fusion and cache blocking therefore gives the same bits, and the spec says so.
- **No running increments across elements** unless each lane can recompute the same value exactly. Example: interpolation weights in M55 are computed per carrier from `p0`, the reciprocal step and `k − p0`, not accumulated.
- **Table lookups by index** (gathers or in-register permutes) equal scalar loads. The remainders call the public scalar helpers of `sdrk-core`, as `normal_q17` in M48.
- **State after a call** equals the fallback's state (positions, accumulators, code cursor).

### D.3 Tolerance-class compositions (M50, M53)

- `xcorr_fft` and `code_search` are tolerance kernels built from backend kernels (`fft`, the spectral product, `mag_sq`). Their SIMD path is the fallback's driver compiled at the level with the backend kernels of that level, as `channelize` (D918).
- Where a composed backend kernel is itself tolerance at that level (`fft` `C32`; `mul` `C32` with FMA, D326), bit identity with the fallback is replaced by:
  - bit identity per backend and level, including chunk invariance (`==`; decision 7 for M50);
  - the SNR floor of the contract against the reference.
- Every part outside the composed kernels keeps §D.2: the history copy, the Doppler bin shift (an index rotation, no arithmetic), the accumulation of the grid and the peak search.
- Each spec states this exception in its contract row.

### D.4 Application per milestone

| Milestone | Lane-parallel part (SIMD in scope, every level) | Scalar part and the reason | Fair reference rows (D32; from the review) |
| --- | --- | --- | --- |
| M49 `scramble` | everything: sign/swap of §A for `b ≤ 2`; for `b ≥ 3` the residual phasor by an in-register table permute (small `b`) or a gather, then the `rotate` product without FMA and the quadrant map | none | phasor lookup + `volk_32fc_x2_multiply_32fc` (`C32`), + `arm_cmplx_mult_cmplx_q15` (`CQ15`); GNU Radio `dvbs2_physical_cc` for `b = 2` as information |
| M50 `xcorr_fft` | the composition of §D.3; spectral product, history copy | none in the kernel; the tap spectrum is built once in `new` | liquid `fftfilt_cccf` in two builds (its own FFT against the fallback, FFTW against the SIMD levels); GNU Radio `fft_filter_ccc` does the same work and is noted, not a separate row |
| M51 `spread` | symbol broadcast over a run of chips; binary chips by the lane-mask negation of §A; complex chips by the product without FMA | none (the PRBS generation in the block is a serial LFSR; `fill_words` runs once at construction) | VOLK `volk_32fc_x2_multiply_32fc` (reference does less work), CMSIS-DSP `arm_cmplx_mult_cmplx_q15` (different work: 3.13 output), `arm_negate_q15` (less work) |
| M52 `despread` | chip terms (sign by §A or product) into the eight partial sums (`C32`) or exact integer lanes (`CQ15`) | the dump once per symbol (the tree and the shift of §C.1) | `volk_32fc_x2_conjugate_dot_prod_32fc`, CMSIS-DSP `arm_cmplx_dot_prod_f32`/`_q15`, liquid `dotprod_cccf` |
| M53 `code_search` | the composition of §D.3 (one forward FFT per dwell, the Doppler bins by bin shift, product, inverse FFT, `mag_sq` accumulate) | combining the `D` row results and the two statistics: `O(D)` work once per row or call, no lane-parallel part worth a vector; the code spectrum is built once in `new` | GNSS-SDR `pcps_acquisition` as information (reference does different work: `2·D` FFTs per dwell against `1 + D`) |
| M54 `code_track` | the integer-lag sums (5 to 12 lags of `±x` for a binary code, the product for a complex code, §D.2 order); the per-period combination (the lag-sum trees and the three cubic combinations of the M38 formula, lane-parallel across lags, arms and components) | the discriminator, the loop filter and the position update: a serial recursion once per code period (the next period's positions depend on this period's error) with no lane-parallel part | `volk_gnsssdr` `xn_resampler` + `dot_prod_xn` |
| M55 `pilot_equalize` | per-carrier interpolation, smoothing, reciprocal, equalization and CPE factor; the CPE sum in the §D.2 order over the pilot list | the per-symbol CPE normalization (one `sqrtf`, one reciprocal per symbol) and the per-segment setup of the interpolation | a VOLK composition (`volk_32fc_x2_divide_32fc` and others) and a CMSIS-DSP composition for Cortex-M (to be confirmed in the implementation, marked where they do less work); GNU Radio OFDM equalizers as information |
| M56 `fwht` | every stage (any schedule, §D.2); `CQ15` with the rounding shift of §C.2 | none | FFHT (FALCONN, MIT licence) for `C32` on x86, layout difference marked |

**Canonical sentence:** "SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked."
