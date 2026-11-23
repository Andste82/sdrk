/*
 * The FFT-based FIR filter `fftfilt_cccf` of liquid-dsp v1.8.3 (MIT), the reference of
 * `xcorr_fft` (M50 §5): the streaming fast convolution with complex taps in f32, the same work
 * per output sample as SDRk's kernel (one forward and one inverse transform per block and the
 * spectral product). liquid uses overlap-add with a transform of 2 n points per block of n
 * outputs, SDRk overlap-save with N points per block of N - L + 1 outputs.
 *
 * The file is a translation unit of liquid, as filter_cccf.c is: it defines the type macros of
 * liquid and includes the unpacked source `fftfilt.proto.c` of the pinned release
 * (benches/refs/build.sh puts the release's src/filter/src on the include path), so the code
 * that runs per block is liquid's own. The internal FFT and the error functions are liquid's
 * objects, compiled from the same release into the same library.
 *
 *   SDRK_LIQUID_FFTW 1  the transforms are FFTW's (`fftwf_plan_dft_1d`, FFTW_ESTIMATE): liquid's
 *                       build with FFTW, its default where FFTW is found (FIND_FFTW ON in its
 *                       CMakeLists.txt); the functions are named sdrk_liquid_fftfilt_fftw_*
 *   SDRK_LIQUID_FFTW 0  liquid's own FFT (`fft_create_plan`, portable C), its build without
 *                       FFTW; the functions are named sdrk_liquid_fftfilt_port_*
 *
 * What differs from liquid's own use:
 *
 *  - The block loop: liquid's `fftfilt_cccf_execute` takes one block (n samples in, n out);
 *    `sdrk_liquid_fftfilt_*_execute` takes any number of samples, calls it for every whole block
 *    in place and keeps a partial block in a staging buffer of n samples until it is complete
 *    (the carry of a streaming caller whose call length is no multiple of n). The outputs of the
 *    completed blocks are written in order. The staging copies element by element (no memcpy,
 *    whose `rep movsb` the counters count once per byte, D2609) and touches at most n samples
 *    per call.
 *  - The taps are the caller's (SDRk's, so that the accuracy of both sides is comparable).
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

#if SDRK_LIQUID_FFTW
/* liquid.internal.h selects FFTW for the transforms with this macro (its CMake build defines it
 * when find_package(fftw3f) succeeds). */
#define fftw3f_FOUND 1
#include <fftw3.h>
#define SDRK_V(name) sdrk_fftfilt_fftw##name
#define SDRK_API(name) sdrk_liquid_fftfilt_fftw##name
#else
#define SDRK_V(name) sdrk_fftfilt_port##name
#define SDRK_API(name) sdrk_liquid_fftfilt_port##name
#endif

/* Both variants link into one library: liquid's names of the object, in its declarations
 * (liquid.h) and its definitions (the prototype file) alike, get the name of the variant. */
#define fftfilt_cccf_s SDRK_V(_s)
#define fftfilt_cccf SDRK_V(_cccf)
#define fftfilt_cccf_create SDRK_V(_create)
#define fftfilt_cccf_copy SDRK_V(_copy)
#define fftfilt_cccf_destroy SDRK_V(_destroy)
#define fftfilt_cccf_reset SDRK_V(_reset)
#define fftfilt_cccf_print SDRK_V(_print)
#define fftfilt_cccf_set_scale SDRK_V(_set_scale)
#define fftfilt_cccf_get_scale SDRK_V(_get_scale)
#define fftfilt_cccf_execute SDRK_V(_execute)
#define fftfilt_cccf_get_length SDRK_V(_get_length)

#include "liquid.internal.h"

#define EXTENSION_SHORT "f"
#define EXTENSION_FULL "cccf"

#define FFTFILT(name) LIQUID_CONCAT(fftfilt_cccf, name)
#define T float complex
#define TO float complex
#define TC float complex
#define TI float complex

#define TO_COMPLEX 1
#define TC_COMPLEX 1
#define TI_COMPLEX 1

#define PRINTVAL_TO(X, F) PRINTVAL_CFLOAT(X, F)
#define PRINTVAL_TC(X, F) PRINTVAL_CFLOAT(X, F)
#define PRINTVAL_TI(X, F) PRINTVAL_CFLOAT(X, F)

#include "fftfilt.proto.c"

/* The filter with its staging buffer for a partial block. */
struct SDRK_V(_stream) {
    FFTFILT() q;
    unsigned int n;
    unsigned int fill;
    float complex* stage;
};

/* `h`: the `h_len` complex taps; `n`: liquid's block length (n >= h_len - 1, a transform of 2 n
 * points). NULL when liquid refuses the configuration. */
void* SDRK_API(_create)(const float complex* h, unsigned int h_len, unsigned int n)
{
    struct SDRK_V(_stream)* s = malloc(sizeof(*s));
    if (s == NULL)
        return NULL;
    s->q = FFTFILT(_create)((float complex*) h, h_len, n);
    s->stage = malloc(n * sizeof(float complex));
    if (s->q == NULL || s->stage == NULL) {
        if (s->q != NULL)
            FFTFILT(_destroy)(s->q);
        free(s->stage);
        free(s);
        return NULL;
    }
    s->n = n;
    s->fill = 0;
    return s;
}

void SDRK_API(_reset)(void* p)
{
    struct SDRK_V(_stream)* s = p;
    FFTFILT(_reset)(s->q);
    s->fill = 0;
}

void SDRK_API(_destroy)(void* p)
{
    struct SDRK_V(_stream)* s = p;
    FFTFILT(_destroy)(s->q);
    free(s->stage);
    free(s);
}

/* One element-wise copy (kept out of the memcpy idiom, see the file header). */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-loop-distribute-patterns")))
#endif
static void SDRK_V(_copy_samples)(float complex* dst, const float complex* src, unsigned int len)
{
    for (unsigned int i = 0; i < len; i++)
        dst[i] = src[i];
}

/* `len` samples of `x` in; the n outputs of every block completed by them to `y`, in order (`y`
 * holds at least len + n - 1 samples). Returns the number of outputs. */
unsigned int SDRK_API(_execute)(void* p, float complex* x, unsigned int len, float complex* y)
{
    struct SDRK_V(_stream)* s = p;
    unsigned int n = s->n;
    unsigned int i = 0;
    unsigned int out = 0;
    if (s->fill > 0) {
        unsigned int take = n - s->fill;
        if (take > len)
            take = len;
        SDRK_V(_copy_samples)(&s->stage[s->fill], x, take);
        s->fill += take;
        i = take;
        if (s->fill < n)
            return 0;
        FFTFILT(_execute)(s->q, s->stage, y);
        out = n;
        s->fill = 0;
    }
    for (; i + n <= len; i += n, out += n)
        FFTFILT(_execute)(s->q, &x[i], &y[out]);
    if (i < len) {
        SDRK_V(_copy_samples)(s->stage, &x[i], len - i);
        s->fill = len - i;
    }
    return out;
}
