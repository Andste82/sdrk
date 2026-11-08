/*
 * The analysis filter bank `firpfbch2_crcf` of liquid-dsp v1.8.3 (MIT), the reference of
 * `channelize` (M45): the polyphase channelizer with the output rate 2 Fs / M (one output vector
 * of M channels per M / 2 input samples), the same work as SDRk's kernel: per hop M / 2 window
 * pushes, M real-tap dot products of L = 2 m taps over complex windows, one M-point transform
 * and the scaling by 1 / M of liquid.
 *
 * The file is a translation unit of liquid, as firpfbch_crcf.c is: it defines the type macros of
 * liquid and includes the unpacked source `firpfbch2.proto.c` of the pinned release
 * (benches/refs/build.sh puts the release's src/multichannel/src on the include path), so the
 * code that runs per hop is liquid's own. The dot products, the window buffers, the internal FFT
 * and the error functions are liquid's objects, compiled from the same release.
 *
 *   SDRK_LIQUID_FFTW 1  the transform is FFTW's (`fftwf_plan_dft_1d`, FFTW_ESTIMATE): liquid's
 *                       build with FFTW, its default where FFTW is found (FIND_FFTW ON in its
 *                       CMakeLists.txt); the functions are named sdrk_liquid_firpfbch2_fftw_*
 *   SDRK_LIQUID_FFTW 0  liquid's own FFT (`fft_create_plan`, portable C), its build without
 *                       FFTW; the functions are named sdrk_liquid_firpfbch2_port_*
 *
 * What differs from liquid's own use:
 *
 *  - The execution method of the dot products is chosen by the caller, as in
 *    liquid-halfband.c: `method` 0 is liquid's portable loop (LIQUID_RUNTIME_PORT), 1 the SIMD
 *    method of the level (SDRK_LIQUID_SIMD, chosen by build.sh). liquid selects it at run time
 *    from the CPU; the measured CPUs are emulated or pinned, so the method is fixed.
 *  - The block loop: liquid's `firpfbch2_crcf_execute` takes one hop (M / 2 samples in, M
 *    out); `sdrk_liquid_firpfbch2_*_execute_block` calls it for `n` hops in a row.
 *  - The prototype is the caller's (SDRk's, so that the accuracy of both sides is comparable),
 *    given to liquid's `firpfbch2_crcf_create`; its design functions are not linked.
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

#ifndef SDRK_LIQUID_FFTW
#error "SDRK_LIQUID_FFTW must be 1 (FFTW) or 0 (liquid's own FFT)"
#endif
#ifndef SDRK_LIQUID_SIMD
#error "SDRK_LIQUID_SIMD must be the LIQUID_RUNTIME_* value of the level"
#endif

#if SDRK_LIQUID_FFTW
/* liquid.internal.h selects FFTW for the transforms with this macro (its CMake build defines it
 * when find_package(fftw3f) succeeds). */
#define fftw3f_FOUND 1
#include <fftw3.h>
#define SDRK_V(name) sdrk_firpfbch2_fftw##name
#define SDRK_API(name) sdrk_liquid_firpfbch2_fftw##name
#else
#define SDRK_V(name) sdrk_firpfbch2_port##name
#define SDRK_API(name) sdrk_liquid_firpfbch2_port##name
#endif

/* Both variants link into one library: liquid's names of the object, in its declarations
 * (liquid.h) and its definitions (the prototype file) alike, get the name of the variant. */
#define firpfbch2_crcf_s SDRK_V(_s)
#define firpfbch2_crcf SDRK_V(_crcf)
#define firpfbch2_crcf_create SDRK_V(_create)
#define firpfbch2_crcf_create_kaiser SDRK_V(_create_kaiser)
#define firpfbch2_crcf_copy SDRK_V(_copy)
#define firpfbch2_crcf_destroy SDRK_V(_destroy)
#define firpfbch2_crcf_reset SDRK_V(_reset)
#define firpfbch2_crcf_print SDRK_V(_print)
#define firpfbch2_crcf_get_type SDRK_V(_get_type)
#define firpfbch2_crcf_get_M SDRK_V(_get_M)
#define firpfbch2_crcf_get_m SDRK_V(_get_m)
#define firpfbch2_crcf_execute SDRK_V(_execute)
#define firpfbch2_crcf_execute_analyzer SDRK_V(_execute_analyzer)
#define firpfbch2_crcf_execute_synthesizer SDRK_V(_execute_synthesizer)

#include "liquid.internal.h"

#define EXTENSION_SHORT "f"
#define EXTENSION_FULL "crcf"

#define FIRPFBCH2(name) LIQUID_CONCAT(firpfbch2_crcf, name)
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


#include "firpfbch2.proto.c"

/* `h`: the prototype of 2 M m taps in time order (liquid's `_create` reverses the branches). */
void* SDRK_API(_create)(const float* h, unsigned int M, unsigned int m, int method)
{
    FIRPFBCH2() q = FIRPFBCH2(_create)(LIQUID_ANALYZER, M, m, (float*) h);
    if (q == NULL)
        return NULL;
    for (unsigned int i = 0; i < M; i++)
        DOTPROD(_runtime_select)(q->dp[i], method ? SDRK_LIQUID_SIMD : LIQUID_RUNTIME_PORT);
    return q;
}

void SDRK_API(_reset)(void* q) { FIRPFBCH2(_reset)((FIRPFBCH2()) q); }
void SDRK_API(_destroy)(void* q) { FIRPFBCH2(_destroy)((FIRPFBCH2()) q); }

/* `n` hops: `n M / 2` samples of `x` in, `n M` channel samples to `y`. */
void SDRK_API(_execute_block)(void* q, float complex* x, unsigned int n, float complex* y)
{
    FIRPFBCH2() p = (FIRPFBCH2()) q;
    for (unsigned int i = 0; i < n; i++)
        FIRPFBCH2(_execute_analyzer)(p, &x[i * p->M2], &y[i * p->M]);
}
