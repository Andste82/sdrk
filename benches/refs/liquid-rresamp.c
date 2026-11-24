/*
 * The rational-rate resamplers `rresamp_crcf` and `rresamp_rrrf` of liquid-dsp v1.8.3 (MIT), the
 * scalar references of `resample_fallback` and `resample_real_fallback` (D2904): `P / Q` with
 * the filter bank `firpfb` of `P` branches of `2 m` taps over liquid's window buffer, `Q` inputs
 * to `P` outputs per primitive block. For the 3 / 2 resampler of the measurement,
 * `rresamp_*_create(3, 2, 32, h)` takes SDRk's 192 taps (`2 P m`), so the branch dot products
 * are the 64-tap dot products of ours.
 *
 * The file is a translation unit of liquid, as filter_crcf.c and filter_rrrf.c are: it defines
 * the type macros of liquid and includes the unpacked sources `firpfb.proto.c` and
 * `rresamp.proto.c` of the pinned release (benches/refs/build.sh puts the release's
 * src/filter/src on the include path), so the code that runs per sample is liquid's own. The
 * dot products and the window buffers are liquid's objects, compiled from the same release.
 *
 * What differs from liquid's own use:
 *
 *  - The dot products run liquid's portable loop (`LIQUID_RUNTIME_PORT`, the reference of a
 *    fallback entry: portable C, compiled for the level as our fallback is); liquid selects the
 *    method at run time from the CPU.
 *  - The filter bank of this file is named `sdrk_rr_firpfb_*` (declared with liquid's own
 *    `LIQUID_FIRPFB_DEFINE_API`): liquid-fir.c includes `firpfb.proto.c` for `firinterp_crcf`
 *    into the same library, and the two copies must not collide; the code is the same.
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
#error "SDRK_LIQUID_KIND must be 1 (rresamp_crcf) or 2 (rresamp_rrrf)"
#endif

/* The design and support functions of liquid that the unused `_create_*` functions of the
 * prototype files call. They are not linked from liquid (see above) and never run; weak,
 * because liquid-fir.c, liquid-halfband.c and the second object of this file define some of
 * them as well. */
__attribute__((weak)) float kaiser_beta_As(float _as) { abort(); (void)_as; }
__attribute__((weak)) float liquid_kaiser(unsigned int _i, unsigned int _n, float _beta) { abort(); (void)_i; (void)_n; (void)_beta; }
__attribute__((weak)) float sincf(float _x) { abort(); (void)_x; }
__attribute__((weak)) int liquid_firdes_kaiser(unsigned int _n, float _fc, float _as, float _mu, float* _h)
{ abort(); (void)_n; (void)_fc; (void)_as; (void)_mu; (void)_h; }
__attribute__((weak)) int liquid_firdes_prototype(liquid_firfilt_type _type, unsigned int _k, unsigned int _m, float _beta, float _dt, float* _h)
{ abort(); (void)_type; (void)_k; (void)_m; (void)_beta; (void)_dt; (void)_h; }
__attribute__((weak)) unsigned int liquid_gcd(unsigned int _p, unsigned int _q) { abort(); (void)_p; (void)_q; }

#define EXTENSION_SHORT "f"

#if SDRK_LIQUID_KIND == 1

#define EXTENSION_FULL "crcf"
#define RRESAMP(name) LIQUID_CONCAT(rresamp_crcf, name)
#define FIRPFB(name) LIQUID_CONCAT(sdrk_rr_firpfb_crcf, name)
LIQUID_FIRPFB_DEFINE_API(FIRPFB, liquid_float_complex, float, liquid_float_complex)
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
#define SDRK_RRESAMP(name) sdrk_liquid_rresamp_crcf##name

#else

#define EXTENSION_FULL "rrrf"
#define RRESAMP(name) LIQUID_CONCAT(rresamp_rrrf, name)
#define FIRPFB(name) LIQUID_CONCAT(sdrk_rr_firpfb_rrrf, name)
LIQUID_FIRPFB_DEFINE_API(FIRPFB, float, float, float)
#define T float
#define TO float
#define TC float
#define TI float
#define WINDOW(name) LIQUID_CONCAT(windowf, name)
#define DOTPROD(name) LIQUID_CONCAT(dotprod_rrrf, name)
#define TO_COMPLEX 0
#define TC_COMPLEX 0
#define TI_COMPLEX 0
#define PRINTVAL_TO(X, F) PRINTVAL_FLOAT(X, F)
#define PRINTVAL_TC(X, F) PRINTVAL_FLOAT(X, F)
#define PRINTVAL_TI(X, F) PRINTVAL_FLOAT(X, F)
#define SDRK_RRESAMP(name) sdrk_liquid_rresamp_rrrf##name

#endif

#include "firpfb.proto.c"
#include "rresamp.proto.c"

/* `h`: the prototype of `2 p m` taps; `p / q` the rate. */
void* SDRK_RRESAMP(_create)(const float* h, unsigned int p, unsigned int q, unsigned int m)
{
    RRESAMP() r = RRESAMP(_create)(p, q, m, (TC*)h);
    if (r == NULL) {
        return NULL;
    }
    for (unsigned int i = 0; i < r->pfb->num_filters; i++) {
        DOTPROD(_runtime_select)(r->pfb->dp[i], LIQUID_RUNTIME_PORT);
    }
    return r;
}

void SDRK_RRESAMP(_reset)(void* r) { RRESAMP(_reset)((RRESAMP()) r); }
void SDRK_RRESAMP(_destroy)(void* r) { RRESAMP(_destroy)((RRESAMP()) r); }

/* `n` primitive blocks: `q n` inputs, `p n` outputs (liquid's `_execute_block`). */
void SDRK_RRESAMP(_execute_block)(void* r, T* x, unsigned int n, T* y)
{
    RRESAMP(_execute_block)((RRESAMP()) r, x, n, y);
}
