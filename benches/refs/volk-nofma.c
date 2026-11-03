/*
 * VOLK's complex product without fused multiply-add: the protokernel of
 * `volk_32fc_x2_multiply_32fc` that multiplies and adds in separate operations, included from
 * the pinned VOLK release unchanged and compiled with `-ffp-contract=off` and the instruction
 * set of the measured level (benches/refs/build.sh, one library per level).
 *
 * - x86-64 (`avx2`, `avx512`): `volk_32fc_x2_multiply_32fc_u_avx` (`_mm256_complexmul_ps`:
 *   `vmovsldup`, `vmovshdup`, `vpermilps`, two `vmulps` and `vaddsubps` per four samples). VOLK
 *   has no AVX-512 implementation of this kernel; on an AVX-512 machine its dispatcher takes
 *   `a_avx2_fma`, the same width as this one.
 * - AArch64 (`neon`): `volk_32fc_x2_multiply_32fc_neon` (`ld2`, four `fmul`, `fsub`, `fadd`,
 *   `st2` per four samples). VOLK's own build compiles it with `-funsafe-math-optimizations`
 *   and the default contraction of GCC, which fuses the products into `fmls` and `fmla`, as in
 *   `neonv8`; VOLK has no NEON kernel without fused operations in that build.
 *
 * The products of SDRk's `mul` must not be fused (D5): these are the same work in the same
 * accuracy class (D32), the gate of `mul` at these levels (D287). VOLK's dispatched kernel
 * (fused) stays as information.
 *
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#define LV_HAVE_GENERIC 1
#if defined(__AVX__)
#define LV_HAVE_AVX 1
#endif
#if defined(__ARM_NEON)
#define LV_HAVE_NEON 1
#endif

/* volk_common.h declares AVX unions where the level has AVX. */
#if defined(__AVX__)
#include <immintrin.h>
#endif
#include <volk/volk_common.h>
#include <volk/volk_complex.h>
#include <volk/volk_32fc_x2_multiply_32fc.h>

#if !defined(__AVX__) && !defined(__ARM_NEON)
#error "volk-nofma.c is built for the levels avx2, avx512 and neon only"
#endif

/* `c[i] = a[i] * b[i]` for `n` samples. */
void sdrk_volk_nofma_32fc_x2_multiply_32fc(lv_32fc_t* c,
                                           const lv_32fc_t* a,
                                           const lv_32fc_t* b,
                                           unsigned int n)
{
#if defined(__AVX__)
    volk_32fc_x2_multiply_32fc_u_avx(c, a, b, n);
#else
    volk_32fc_x2_multiply_32fc_neon(c, a, b, n);
#endif
}

/* The name of the protokernel that `sdrk_volk_nofma_32fc_x2_multiply_32fc` runs. */
const char* sdrk_volk_nofma_32fc_x2_multiply_32fc_name(void)
{
#if defined(__AVX__)
    return "u_avx";
#else
    return "neon";
#endif
}
