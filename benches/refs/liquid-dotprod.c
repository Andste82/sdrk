/*
 * The portable structured dot products of liquid-dsp v1.8.3 (src/dotprod/src/dotprod.proto.c,
 * instantiated by dotprod_rrrf.c and dotprod_crcf.c) that iirfilt_rrrf and iirfilt_crcf call:
 * `_create` (coefficients copied into a 64-byte aligned buffer, duplicated for complex input),
 * `_execute` (the call through the run-time selected function pointer) and `_execute_port`
 * (the portable loop), with the execution method fixed to `LIQUID_RUNTIME_PORT`, which liquid's
 * public `dotprod_*_runtime_select` offers; liquid's run-time detection would pick its SSE, AVX
 * or NEON loops, which are not the scalar reference. Error logging, `_print`, `_copy` and the
 * `_run` variants are left out. liquid compiles the dot products in other translation units
 * than the filters, so liquid-dcblock.c calls these functions out of line.
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
#include <stdlib.h>

#define LIQUID_OK 0

typedef struct dotprod_rrrf_s* dotprod_rrrf;
typedef struct dotprod_crcf_s* dotprod_crcf;

/* liquid_aligned_alloc and liquid_aligned_free of src/utility/src/memory.c. */
static void* liquid_aligned_alloc(size_t _alignment, size_t _size)
{
    void* ptr = NULL;
    if (posix_memalign(&ptr, _alignment, _size) != 0)
        return NULL;
    return ptr;
}

static int liquid_aligned_free(void* _p)
{
    free(_p);
    return LIQUID_OK;
}

/* ---- dotprod_rrrf: T = TO = TC = TI = float ---- */

struct dotprod_rrrf_s {
    float* h;       /* real-only coefficients */
    unsigned int n; /* number of coefficients */
    int (*execute)(dotprod_rrrf _q, float* _x, float* _y);
};

static int dotprod_rrrf_execute_port(dotprod_rrrf _q, float* _x, float* _y)
{
    unsigned int i;
    // accumulate
    float r = 0;
    for (i = 0; i < _q->n; i++)
        r += _q->h[i] * _x[i];

    // return result
    *_y = r;
    return LIQUID_OK;
}

dotprod_rrrf dotprod_rrrf_create(float* _h, unsigned int _n)
{
    dotprod_rrrf q = (dotprod_rrrf)malloc(sizeof(struct dotprod_rrrf_s));
    q->n = _n;

    // real-only coefficients
    q->h = (float*)liquid_aligned_alloc(64, (q->n) * sizeof(float));

    // copy coefficients
    unsigned int i;
    for (i = 0; i < q->n; i++)
        q->h[i] = _h[i];

    q->execute = &dotprod_rrrf_execute_port;
    return q;
}

int dotprod_rrrf_destroy(dotprod_rrrf _q)
{
    liquid_aligned_free(_q->h);
    free(_q);
    return LIQUID_OK;
}

int dotprod_rrrf_execute(dotprod_rrrf _q, float* _x, float* _y)
{
    // invoke runtime-specific method
    return _q->execute(_q, _x, _y);
}

/* ---- dotprod_crcf: T = TC = float, TO = TI = float complex ---- */

struct dotprod_crcf_s {
    float* h;       /* real-only coefficients (double length for SIMD) */
    unsigned int n; /* number of coefficients */
    int (*execute)(dotprod_crcf _q, float complex* _x, float complex* _y);
};

static int dotprod_crcf_execute_port(dotprod_crcf _q, float complex* _x, float complex* _y)
{
    unsigned int i;
    // accumulate (stride coefficients by 2)
    float complex r = 0;
    for (i = 0; i < _q->n; i++)
        r += _q->h[2 * i + 0] * _x[i];

    // return result
    *_y = r;
    return LIQUID_OK;
}

dotprod_crcf dotprod_crcf_create(float* _h, unsigned int _n)
{
    dotprod_crcf q = (dotprod_crcf)malloc(sizeof(struct dotprod_crcf_s));
    q->n = _n;

    // real-only coefficients (double length for SIMD)
    q->h = (float*)liquid_aligned_alloc(64, 2 * (q->n) * sizeof(float));

    // copy coefficients
    unsigned int i;
    for (i = 0; i < q->n; i++) {
        q->h[2 * i + 0] = _h[i];
        q->h[2 * i + 1] = _h[i];
    }

    q->execute = &dotprod_crcf_execute_port;
    return q;
}

int dotprod_crcf_destroy(dotprod_crcf _q)
{
    liquid_aligned_free(_q->h);
    free(_q);
    return LIQUID_OK;
}

int dotprod_crcf_execute(dotprod_crcf _q, float complex* _x, float complex* _y)
{
    // invoke runtime-specific method
    return _q->execute(_q, _x, _y);
}
