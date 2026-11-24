/*
 * VOLK's portable `generic` protokernels of the interleave and deinterleave kernels, the
 * scalar references of SDRk's `interleave` and `deinterleave` fallbacks (D32: a fallback is
 * compared with established scalar code). Included from the pinned VOLK release unchanged and
 * compiled with the flags of the other per-level scalar references (`-O3 -ffp-contract=off`
 * and the instruction set of the measured level, benches/refs/build.sh, one library per
 * level), so the compiler may vectorize them for the level as it may vectorize our fallback.
 *
 * - `volk_32f_x2_interleave_32fc_generic`: two float buffers to complex floats.
 * - `volk_32fc_deinterleave_32f_x2_generic`: complex floats to two float buffers.
 * - `volk_16ic_deinterleave_16i_x2_generic`: complex 16-bit integers to two 16-bit buffers.
 * - `volk_32fc_x2_multiply_32fc_generic`: the complex product of two buffers (`*c = *a * *b` in
 *   C99 complex arithmetic, not fused under `-ffp-contract=off`), the second half of the
 *   reference of the `C32` `scramble` fallback (M49 §5: the phasor looked up into a buffer, then
 *   this product).
 * - `volk_32fc_32f_multiply_32fc_generic`: complex samples times a real sequence (`*c = *a *
 *   *b`, two products per sample), the second half of the reference of the `C32` `scramble`
 *   fallback at `bits = 1` (M49.1 §5: a `±1.0` sequence formed from the index bits into a
 *   buffer, then this product, the form of srsRAN's `srsran_scrambling_c`).
 * - `volk_32f_s32f_convert_16i_generic`, `volk_32f_s32f_convert_8i_generic`,
 *   `volk_32f_s32f_x2_convert_8u_generic`: floats times a scale (plus a bias for the unsigned
 *   bytes), clamped to the range of the integer type, then `rintf` (round to the nearest integer,
 *   ties to even): the scalar references of the `convert_out` fallbacks (D2905).
 * - `volk_32fc_x2_multiply_conjugate_32fc_generic` and `volk_32fc_s32f_atan2_32f_generic` (libm's
 *   `atan2f` per sample): the scalar reference of the `quad_demod` fallback (D2906).
 *
 * VOLK has no 16-bit interleave (the closest kernel, `volk_32f_x2_s32f_interleave_16ic`,
 * converts scaled floats to 16-bit integers, a conversion with a multiply and a rounding per
 * value, not the same work). `sdrk_volk_generic_16i_x2_interleave_16ic` is the loop of
 * `volk_32f_x2_interleave_32fc_generic` with the element type `int16_t` instead of `float`
 * (the mirror of `volk_16ic_deinterleave_16i_x2_generic`), the reference of the `CQ15`
 * interleave fallback.
 *
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#define LV_HAVE_GENERIC 1

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <volk/volk_common.h>
#include <volk/volk_complex.h>
#include <volk/volk_32f_x2_interleave_32fc.h>
#include <volk/volk_32fc_deinterleave_32f_x2.h>
#include <volk/volk_16ic_deinterleave_16i_x2.h>
#include <volk/volk_32fc_x2_multiply_32fc.h>
#include <volk/volk_32fc_32f_multiply_32fc.h>
#include <volk/volk_32f_s32f_convert_16i.h>
#include <volk/volk_32f_s32f_convert_8i.h>
#include <volk/volk_32f_s32f_x2_convert_8u.h>
#include <volk/volk_32fc_x2_multiply_conjugate_32fc.h>
#include <volk/volk_32fc_s32f_atan2_32f.h>

/* `c[i] = (re[i], im[i])` for `n` samples. */
void sdrk_volk_generic_32f_x2_interleave_32fc(lv_32fc_t* c,
                                              const float* re,
                                              const float* im,
                                              unsigned int n)
{
    volk_32f_x2_interleave_32fc_generic(c, re, im, n);
}

/* `re[i] = c[i].re`, `im[i] = c[i].im` for `n` samples. */
void sdrk_volk_generic_32fc_deinterleave_32f_x2(float* re,
                                                float* im,
                                                const lv_32fc_t* c,
                                                unsigned int n)
{
    volk_32fc_deinterleave_32f_x2_generic(re, im, c, n);
}

/* `re[i] = c[i].re`, `im[i] = c[i].im` for `n` 16-bit complex samples. */
void sdrk_volk_generic_16ic_deinterleave_16i_x2(int16_t* re,
                                                int16_t* im,
                                                const lv_16sc_t* c,
                                                unsigned int n)
{
    volk_16ic_deinterleave_16i_x2_generic(re, im, c, n);
}

/* `c[i] = a[i] * b[i]` (complex) for `n` samples. */
void sdrk_volk_generic_32fc_x2_multiply_32fc(lv_32fc_t* c,
                                             const lv_32fc_t* a,
                                             const lv_32fc_t* b,
                                             unsigned int n)
{
    volk_32fc_x2_multiply_32fc_generic(c, a, b, n);
}

/* `c[i] = a[i] * b[i]` (complex times real) for `n` samples. */
void sdrk_volk_generic_32fc_32f_multiply_32fc(lv_32fc_t* c,
                                              const lv_32fc_t* a,
                                              const float* b,
                                              unsigned int n)
{
    volk_32fc_32f_multiply_32fc_generic(c, a, b, n);
}

/* `out[i] = rintf(clamp(in[i] * scale))` as `int16_t` for `n` floats. */
void sdrk_volk_generic_32f_s32f_convert_16i(int16_t* out,
                                            const float* in,
                                            float scale,
                                            unsigned int n)
{
    volk_32f_s32f_convert_16i_generic(out, in, scale, n);
}

/* `out[i] = rintf(clamp(in[i] * scale))` as `int8_t` for `n` floats. */
void sdrk_volk_generic_32f_s32f_convert_8i(int8_t* out,
                                           const float* in,
                                           float scale,
                                           unsigned int n)
{
    volk_32f_s32f_convert_8i_generic(out, in, scale, n);
}

/* `out[i] = rintf(clamp(in[i] * scale + bias))` as `uint8_t` for `n` floats. */
void sdrk_volk_generic_32f_s32f_x2_convert_8u(uint8_t* out,
                                              const float* in,
                                              float scale,
                                              float bias,
                                              unsigned int n)
{
    volk_32f_s32f_x2_convert_8u_generic(out, in, scale, bias, n);
}

/* `c[i] = a[i] * conj(b[i])` (complex) for `n` samples. */
void sdrk_volk_generic_32fc_x2_multiply_conjugate_32fc(lv_32fc_t* c,
                                                       const lv_32fc_t* a,
                                                       const lv_32fc_t* b,
                                                       unsigned int n)
{
    volk_32fc_x2_multiply_conjugate_32fc_generic(c, a, b, n);
}

/* `out[i] = atan2f(c[i].im, c[i].re) / normalize` for `n` samples. */
void sdrk_volk_generic_32fc_s32f_atan2_32f(float* out,
                                           const lv_32fc_t* c,
                                           float normalize,
                                           unsigned int n)
{
    volk_32fc_s32f_atan2_32f_generic(out, c, normalize, n);
}

/*
 * `c[i] = (re[i], im[i])` for `n` 16-bit complex samples: the loop of
 * `volk_32f_x2_interleave_32fc_generic` with `int16_t` elements (VOLK has no 16-bit
 * interleave, see above).
 */
void sdrk_volk_generic_16i_x2_interleave_16ic(lv_16sc_t* complexVector,
                                              const int16_t* iBuffer,
                                              const int16_t* qBuffer,
                                              unsigned int num_points)
{
    int16_t* complexVectorPtr = (int16_t*)complexVector;
    const int16_t* iBufferPtr = iBuffer;
    const int16_t* qBufferPtr = qBuffer;
    unsigned int number;

    for (number = 0; number < num_points; number++) {
        *complexVectorPtr++ = *iBufferPtr++;
        *complexVectorPtr++ = *qBufferPtr++;
    }
}
