/*
 * The halfband decimators of liquid-dsp v1.8.3 (MIT) with liquid's own dot products, the
 * references of `halfband_decimate`, `halfband_decimate_real` and `real_to_complex`:
 *
 *   SDRK_LIQUID_KIND 1  resamp2_crcf_decim_execute   (complex float in and out, float taps)
 *   SDRK_LIQUID_KIND 2  resamp2_rrrf_decim_execute   (float in, out and taps)
 *   SDRK_LIQUID_KIND 3  firhilbf_decim_execute       (real float in, complex float out)
 *
 * The file is a translation unit of liquid, as filter_crcf.c and filter_rrrf.c are: it defines
 * the type macros of liquid and includes the unpacked sources `resamp2.proto.c` and
 * `firhilb.proto.c` of the pinned release (benches/refs/build.sh puts the release's src/filter/src
 * on the include path), so that the code that runs per sample is liquid's own, not a copy. The
 * dot products, the window buffers and the error functions are liquid's objects, compiled from
 * the same release.
 *
 * What differs from liquid's own use:
 *
 *  - The filter is not designed. liquid's `_create` designs a Kaiser halfband filter and its
 *    design code (`liquid_firdes_kaiser`, ...) is not linked; `sdrk_liquid_*_create` builds the
 *    same object from a prototype given by the caller (SDRk's taps, so that the accuracy of both
 *    sides is comparable), with the code of liquid's `_create` that follows the design: the
 *    odd-offset taps, reversed, into the dot product (`h1` of resamp2, `hq` of firhilb).
 *  - The execution method of the dot product is chosen by the caller: `method` 0 is liquid's
 *    portable loop (`LIQUID_RUNTIME_PORT`, the reference of the fallback entries), 1 the SIMD
 *    method of the level (SDRK_LIQUID_SIMD, a LIQUID_RUNTIME_* value chosen by build.sh:
 *    SSE for sse2 and sse4.2, AVX for avx2, AVX-512F for avx512, NEON for neon, the portable
 *    loop for simd128). liquid selects the method at run time from the CPU; the measured CPUs
 *    are emulated or pinned, so the method is fixed.
 *  - The block loop. liquid's `resamp2_*_decim_execute` and `firhilbf_decim_execute` take one
 *    output's worth of input; `firhilbf_decim_execute_block` exists, resamp2's does not, so the
 *    loop of `sdrk_liquid_*_decim_block` is the loop of `firhilb`'s block function.
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
#error "SDRK_LIQUID_KIND must be 1 (resamp2_crcf), 2 (resamp2_rrrf) or 3 (firhilbf)"
#endif
#ifndef SDRK_LIQUID_SIMD
#error "SDRK_LIQUID_SIMD must be the LIQUID_RUNTIME_* value of the level"
#endif

#if SDRK_LIQUID_KIND == 1
/* The design functions of liquid that the unused `_create` and `_recreate` of the prototype
 * files call. They are not linked from liquid (see above) and never run. */
float kaiser_beta_As(float _as) { abort(); (void)_as; }
float liquid_kaiser(unsigned int _i, unsigned int _n, float _beta) { abort(); (void)_i; (void)_n; (void)_beta; }
float sincf(float _x) { abort(); (void)_x; }
int liquid_firdes_kaiser(unsigned int _n, float _fc, float _as, float _mu, float* _h)
{ abort(); (void)_n; (void)_fc; (void)_as; (void)_mu; (void)_h; }
#endif

#define EXTENSION_SHORT "f"

#if SDRK_LIQUID_KIND == 1

#define EXTENSION_FULL "crcf"
#define RESAMP2(name) LIQUID_CONCAT(resamp2_crcf, name)
#define T float complex
#define TO float complex
#define TC float
#define TI float complex
#define WINDOW(name) LIQUID_CONCAT(windowcf, name)
#define DOTPROD(name) LIQUID_CONCAT(dotprod_crcf, name)
#define TC_COMPLEX 0
#include "resamp2.proto.c"

/* `h`: the prototype of 4m + 1 taps; `scale`: liquid's output scale. */
void* sdrk_liquid_resamp2_crcf_create(const float* h, unsigned int m, float scale, int method)
{
    RESAMP2() q = (RESAMP2()) malloc(sizeof(struct RESAMP2(_s)));
    q->m = m;
    q->f0 = 0.0f;
    q->as = 0.0f;
    q->h_len = 4 * m + 1;
    q->h = (TC*) malloc(q->h_len * sizeof(TC));
    memcpy(q->h, h, q->h_len * sizeof(TC));
    q->h1_len = 2 * m;
    q->h1 = (TC*) malloc(q->h1_len * sizeof(TC));
    /* resample, alternate sign, [reverse direction]: as liquid's `_create` */
    unsigned int i, j = 0;
    for (i = 1; i < q->h_len; i += 2)
        q->h1[j++] = q->h[q->h_len - i - 1];
    q->dp = DOTPROD(_create)(q->h1, 2 * q->m);
    DOTPROD(_runtime_select)(q->dp, method ? SDRK_LIQUID_SIMD : LIQUID_RUNTIME_PORT);
    q->w0 = WINDOW(_create)(2 * q->m);
    q->w1 = WINDOW(_create)(2 * q->m);
    RESAMP2(_reset)(q);
    RESAMP2(_set_scale)(q, scale);
    return q;
}

void sdrk_liquid_resamp2_crcf_reset(void* q) { RESAMP2(_reset)((RESAMP2()) q); }
void sdrk_liquid_resamp2_crcf_destroy(void* q) { RESAMP2(_destroy)((RESAMP2()) q); }

/* `n` outputs from `2 n` inputs. */
void sdrk_liquid_resamp2_crcf_decim_block(void* q, float complex* x, unsigned int n, float complex* y)
{
    for (unsigned int i = 0; i < n; i++)
        RESAMP2(_decim_execute)((RESAMP2()) q, &x[2 * i], &y[i]);
}

#elif SDRK_LIQUID_KIND == 2

#define EXTENSION_FULL "rrrf"
#define RESAMP2(name) LIQUID_CONCAT(resamp2_rrrf, name)
#define T float
#define TO float
#define TC float
#define TI float
#define WINDOW(name) LIQUID_CONCAT(windowf, name)
#define DOTPROD(name) LIQUID_CONCAT(dotprod_rrrf, name)
#define TC_COMPLEX 0
#include "resamp2.proto.c"

void* sdrk_liquid_resamp2_rrrf_create(const float* h, unsigned int m, float scale, int method)
{
    RESAMP2() q = (RESAMP2()) malloc(sizeof(struct RESAMP2(_s)));
    q->m = m;
    q->f0 = 0.0f;
    q->as = 0.0f;
    q->h_len = 4 * m + 1;
    q->h = (TC*) malloc(q->h_len * sizeof(TC));
    memcpy(q->h, h, q->h_len * sizeof(TC));
    q->h1_len = 2 * m;
    q->h1 = (TC*) malloc(q->h1_len * sizeof(TC));
    unsigned int i, j = 0;
    for (i = 1; i < q->h_len; i += 2)
        q->h1[j++] = q->h[q->h_len - i - 1];
    q->dp = DOTPROD(_create)(q->h1, 2 * q->m);
    DOTPROD(_runtime_select)(q->dp, method ? SDRK_LIQUID_SIMD : LIQUID_RUNTIME_PORT);
    q->w0 = WINDOW(_create)(2 * q->m);
    q->w1 = WINDOW(_create)(2 * q->m);
    RESAMP2(_reset)(q);
    RESAMP2(_set_scale)(q, scale);
    return q;
}

void sdrk_liquid_resamp2_rrrf_reset(void* q) { RESAMP2(_reset)((RESAMP2()) q); }
void sdrk_liquid_resamp2_rrrf_destroy(void* q) { RESAMP2(_destroy)((RESAMP2()) q); }

void sdrk_liquid_resamp2_rrrf_decim_block(void* q, float* x, unsigned int n, float* y)
{
    for (unsigned int i = 0; i < n; i++)
        RESAMP2(_decim_execute)((RESAMP2()) q, &x[2 * i], &y[i]);
}

#elif SDRK_LIQUID_KIND == 3

#define FIRHILB(name) LIQUID_CONCAT(firhilbf, name)
#define T float
#define WINDOW(name) LIQUID_CONCAT(windowf, name)
#define DOTPROD(name) LIQUID_CONCAT(dotprod_rrrf, name)
#include "firhilb.proto.c"

/* `h`: the prototype of 4m + 1 taps. liquid's `_create` derives the quadrature taps from the
 * prototype with `h[i] * sin(pi/2 t)`, `t` the offset from the centre: at the odd offsets that
 * matter, +-1; `hq` here is the same, with the signs written out. */
void* sdrk_liquid_firhilbf_create(const float* h, unsigned int m, int method)
{
    FIRHILB() q = (FIRHILB()) malloc(sizeof(struct FIRHILB(_s)));
    q->m = m;
    q->as = 0.0f;
    q->h_len = 4 * m + 1;
    q->h = (T*) malloc(q->h_len * sizeof(T));
    q->hc = (T complex*) malloc(q->h_len * sizeof(T complex));
    q->hq_len = 2 * m;
    q->hq = (T*) malloc(q->hq_len * sizeof(T));
    unsigned int i;
    for (i = 0; i < q->h_len; i++) {
        float t = (float) i - (float) (q->h_len - 1) / 2.0f;
        q->hc[i] = h[i] * cexpf(_Complex_I * 0.5f * (float) M_PI * t);
        q->h[i] = cimagf(q->hc[i]);
    }
    unsigned int j = 0;
    for (i = 1; i < q->h_len; i += 2)
        q->hq[j++] = q->h[q->h_len - i - 1];
    q->w0 = WINDOW(_create)(2 * q->m);
    q->w1 = WINDOW(_create)(2 * q->m);
    q->w2 = WINDOW(_create)(2 * q->m);
    q->w3 = WINDOW(_create)(2 * q->m);
    q->dpq = DOTPROD(_create)(q->hq, q->hq_len);
    DOTPROD(_runtime_select)(q->dpq, method ? SDRK_LIQUID_SIMD : LIQUID_RUNTIME_PORT);
    FIRHILB(_reset)(q);
    return q;
}

void sdrk_liquid_firhilbf_reset(void* q) { FIRHILB(_reset)((FIRHILB()) q); }
void sdrk_liquid_firhilbf_destroy(void* q) { FIRHILB(_destroy)((FIRHILB()) q); }

/* `n` outputs from `2 n` inputs: liquid's `firhilbf_decim_execute_block`. */
void sdrk_liquid_firhilbf_decim_block(void* q, float* x, unsigned int n, float complex* y)
{
    FIRHILB(_decim_execute_block)((FIRHILB()) q, x, n, y);
}

#else
#error "unknown SDRK_LIQUID_KIND"
#endif
