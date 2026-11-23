/*
 * The FIR filter `firfilt_crcf` and the polyphase interpolator `firinterp_crcf` of liquid-dsp
 * v1.8.3 (MIT), the references of `pulse_match_block` and `pulse_shape_block` (issue #169): a
 * real-tap filter over complex samples, as the root-raised-cosine pulse filters of SDRk.
 *
 * The file is a translation unit of liquid, as filter_crcf.c is: it defines the type macros of
 * liquid and includes the unpacked sources `firfilt.proto.c`, `firinterp.proto.c` and
 * `firpfb.proto.c` of the pinned release (benches/refs/build.sh puts the release's
 * src/filter/src on the include path), so the code that runs per sample is liquid's own: the
 * window buffer, the `firpfb_crcf` of `firinterp_crcf` and the dot products. The dot products
 * and the window are liquid's objects, compiled from the same release.
 *
 * What differs from liquid's own use:
 *
 *  - The execution method of the dot products is chosen by the caller, as in
 *    liquid-halfband.c: `method` 0 is liquid's portable loop (LIQUID_RUNTIME_PORT), 1 the SIMD
 *    method of the level (SDRK_LIQUID_SIMD, chosen by build.sh). liquid selects it at run time
 *    from the CPU; the measured CPUs are emulated or pinned, so the method is fixed.
 *  - The block loop: `sdrk_liquid_*_execute_block` calls liquid's `_execute_block` (a push and
 *    an execute per sample; for the interpolator one push and `M` branch dot products).
 *  - The prototype is the caller's (SDRk's taps, so that the accuracy of both sides is
 *    comparable), given to liquid's `_create`; its design functions are not linked.
 *
 * Copyright (c) 2007 - 2026 Joseph Gaeddert
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "liquid.internal.h"

#ifndef SDRK_LIQUID_KIND
#error "SDRK_LIQUID_KIND must be 1 (firinterp_crcf) or 2 (firfilt_crcf)"
#endif
#ifndef SDRK_LIQUID_SIMD
#error "SDRK_LIQUID_SIMD must be the LIQUID_RUNTIME_* value of the level"
#endif

/* The design functions of liquid that the unused `_create_*` functions of the prototype files
 * call. They are not linked from liquid (see above) and never run; weak, because
 * liquid-halfband.c and the second object of this file define them as well. */
__attribute__((weak)) float kaiser_beta_As(float _as) { abort(); (void)_as; }
__attribute__((weak)) float liquid_kaiser(unsigned int _i, unsigned int _n, float _beta) { abort(); (void)_i; (void)_n; (void)_beta; }
__attribute__((weak)) float sincf(float _x) { abort(); (void)_x; }
__attribute__((weak)) int liquid_firdes_kaiser(unsigned int _n, float _fc, float _as, float _mu, float* _h)
{ abort(); (void)_n; (void)_fc; (void)_as; (void)_mu; (void)_h; }
__attribute__((weak)) int liquid_firdes_prototype(liquid_firfilt_type _type, unsigned int _k, unsigned int _m, float _beta, float _dt, float* _h)
{ abort(); (void)_type; (void)_k; (void)_m; (void)_beta; (void)_dt; (void)_h; }
__attribute__((weak)) unsigned int estimate_req_filter_len(float _df, float _as) { abort(); (void)_df; (void)_as; }
__attribute__((weak)) int firdespm_lowpass(unsigned int _n, float _fc, float _as, float _mu, float* _h)
{ abort(); (void)_n; (void)_fc; (void)_as; (void)_mu; (void)_h; }
__attribute__((weak)) int liquid_firdes_notch(unsigned int _m, float _f0, float _as, float* _h)
{ abort(); (void)_m; (void)_f0; (void)_as; (void)_h; }
__attribute__((weak)) float fir_group_delay(float* _h, unsigned int _n, float _fc) { abort(); (void)_h; (void)_n; (void)_fc; }
__attribute__((weak)) int liquid_freqrespf(float* _h, unsigned int _h_len, float _fc, liquid_float_complex* _H)
{ abort(); (void)_h; (void)_h_len; (void)_fc; (void)_H; }

#define EXTENSION_SHORT "f"
#define EXTENSION_FULL "crcf"
#define FIRFILT(name) LIQUID_CONCAT(firfilt_crcf, name)
#define FIRINTERP(name) LIQUID_CONCAT(firinterp_crcf, name)
#define FIRPFB(name) LIQUID_CONCAT(firpfb_crcf, name)
#define T float complex
#define TO float complex
#define TC float
#define TI float complex
#define WINDOW(name) LIQUID_CONCAT(windowcf, name)
#define DOTPROD(name) LIQUID_CONCAT(dotprod_crcf, name)
#define TO_COMPLEX 1
#define TC_COMPLEX 0
#define TI_COMPLEX 1
#define PRINTVAL_TO(X, F) PRINTVAL_CFLOAT(X, F)
#define PRINTVAL_TC(X, F) PRINTVAL_FLOAT(X, F)
#define PRINTVAL_TI(X, F) PRINTVAL_CFLOAT(X, F)

#if SDRK_LIQUID_KIND == 1

#include "firpfb.proto.c"
#include "firinterp.proto.c"

/* `h`: the prototype of `h_len` taps, `m`: the interpolation factor. */
void* sdrk_liquid_firinterp_crcf_create(const float* h, unsigned int h_len, unsigned int m, int method)
{
    FIRINTERP() q = FIRINTERP(_create)(m, (float*)h, h_len);
    if (q == NULL) {
        return NULL;
    }
    for (unsigned int i = 0; i < q->filterbank->num_filters; i++) {
        DOTPROD(_runtime_select)(q->filterbank->dp[i], method ? SDRK_LIQUID_SIMD : LIQUID_RUNTIME_PORT);
    }
    return q;
}

void sdrk_liquid_firinterp_crcf_reset(void* q) { FIRINTERP(_reset)((FIRINTERP()) q); }
void sdrk_liquid_firinterp_crcf_destroy(void* q) { FIRINTERP(_destroy)((FIRINTERP()) q); }

/* `n` inputs, `n * m` outputs. */
void sdrk_liquid_firinterp_crcf_execute_block(void* q, float complex* x, unsigned int n, float complex* y)
{
    FIRINTERP(_execute_block)((FIRINTERP()) q, x, n, y);
}

#else

#include "firfilt.proto.c"

/* `h`: the filter of `h_len` taps. */
void* sdrk_liquid_firfilt_crcf_create(const float* h, unsigned int h_len, int method)
{
    FIRFILT() q = FIRFILT(_create)((float*)h, h_len);
    if (q == NULL) {
        return NULL;
    }
    DOTPROD(_runtime_select)(q->dp, method ? SDRK_LIQUID_SIMD : LIQUID_RUNTIME_PORT);
    return q;
}

void sdrk_liquid_firfilt_crcf_reset(void* q) { FIRFILT(_reset)((FIRFILT()) q); }
void sdrk_liquid_firfilt_crcf_destroy(void* q) { FIRFILT(_destroy)((FIRFILT()) q); }

/* `n` inputs, `n` outputs. */
void sdrk_liquid_firfilt_crcf_execute_block(void* q, float complex* x, unsigned int n, float complex* y)
{
    FIRFILT(_execute_block)((FIRFILT()) q, x, n, y);
}

#endif
