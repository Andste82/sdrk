/*
 * The DC-blocking IIR filter of liquid-dsp v1.8.3: `iirfilt_rrrf_create_dc_blocker` and
 * `iirfilt_crcf_create_dc_blocker` with the code they run per sample, `_execute_block`,
 * `_execute`, `_execute_norm` (with LIQUID_IIRFILT_USE_DOTPROD, liquid's default) of
 * src/filter/src/iirfilt.proto.c, instantiated as liquid's filter_rrrf.c and filter_crcf.c do
 * (rrrf: float in, out and coefficients; crcf: complex float in and out, float coefficients).
 * Left out: error logging, the second-order-section form (a DC blocker is created in the
 * normal form), the frequency response, printing and copying.
 *
 * The filter is H(z) = sqrt(1 - alpha) (1 - z^-1) / (1 - (1 - alpha) z^-1): the same pole and
 * zero as SDRk's `dc_block` / `dc_block_real` with r = 1 - alpha, but in liquid's direct form
 * II (one state buffer v: v0 = x - a1 v1, y = (b0 v0 + b1 v1) * scale, both sums through the
 * dot products of liquid-dotprod.c, which are called out of line through their function
 * pointer) and with the output gain sqrt(1 - alpha) that `_create_dc_blocker` sets. liquid's
 * public `_set_scale` sets another gain (1 gives SDRk's gain).
 *
 * Copyright (c) 2007 - 2024 Joseph Gaeddert
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

#define LIQUID_OK 0

/* The dot products of liquid-dotprod.c. */
typedef struct dotprod_rrrf_s* dotprod_rrrf;
typedef struct dotprod_crcf_s* dotprod_crcf;
dotprod_rrrf dotprod_rrrf_create(float* _h, unsigned int _n);
int dotprod_rrrf_destroy(dotprod_rrrf _q);
int dotprod_rrrf_execute(dotprod_rrrf _q, float* _x, float* _y);
dotprod_crcf dotprod_crcf_create(float* _h, unsigned int _n);
int dotprod_crcf_destroy(dotprod_crcf _q);
int dotprod_crcf_execute(dotprod_crcf _q, float complex* _x, float complex* _y);

enum { IIRFILT_TYPE_NORM = 0, IIRFILT_TYPE_SOS };

/* ---- iirfilt_rrrf: TO = TC = TI = float ---- */

struct iirfilt_rrrf_s {
    float* b;       /* numerator (feed-forward coefficients) */
    float* a;       /* denominator (feed-back coefficients) */
    float* v;       /* internal filter state (buffer) */
    unsigned int n; /* filter length (order+1) */

    unsigned int nb; /* numerator length */
    unsigned int na; /* denominator length */

    int type; /* filter structure type */

    dotprod_rrrf dpb; /* numerator dot product */
    dotprod_rrrf dpa; /* denominator dot product */

    void* qsos;        /* second-order sections filters (unused here) */
    unsigned int nsos; /* number of second-order sections */

    float scale; /* output scaling factor */
};
typedef struct iirfilt_rrrf_s* iirfilt_rrrf;

static int iirfilt_rrrf_reset(iirfilt_rrrf _q)
{
    unsigned int i;
    // set internal buffer to zero
    for (i = 0; i < _q->n; i++)
        _q->v[i] = 0;
    return LIQUID_OK;
}

int sdrk_liquid_iirfilt_rrrf_set_scale(iirfilt_rrrf _q, float _scale)
{
    _q->scale = _scale;
    return LIQUID_OK;
}

static iirfilt_rrrf iirfilt_rrrf_create(float* _b, unsigned int _nb, float* _a, unsigned int _na)
{
    // validate input
    if (_nb == 0 || _na == 0)
        return NULL;

    // create structure and initialize
    iirfilt_rrrf q = (iirfilt_rrrf)malloc(sizeof(struct iirfilt_rrrf_s));
    q->b = NULL;
    q->a = NULL;
    q->v = NULL;
    q->qsos = NULL;
    q->nsos = 0;
    q->dpb = NULL;
    q->dpa = NULL;
    q->nb = _nb;
    q->na = _na;
    q->n = (q->na > q->nb) ? q->na : q->nb;
    q->type = IIRFILT_TYPE_NORM;

    // allocate memory for numerator, denominator
    q->a = (float*)malloc((q->na) * sizeof(float));
    q->b = (float*)malloc((q->nb) * sizeof(float));

    // normalize coefficients to _a[0]
    float a0 = _a[0];

    unsigned int i;
    for (i = 0; i < q->nb; i++)
        q->b[i] = _b[i] / a0;

    for (i = 0; i < q->na; i++)
        q->a[i] = _a[i] / a0;

    // create buffer and initialize
    q->v = (float*)malloc((q->n) * sizeof(float));

    q->dpa = dotprod_rrrf_create(q->a + 1, q->na - 1);
    q->dpb = dotprod_rrrf_create(q->b, q->nb);

    // reset internal state
    iirfilt_rrrf_reset(q);

    // set scale and return
    sdrk_liquid_iirfilt_rrrf_set_scale(q, 1);
    return q;
}

// create DC-blocking filter
//
//          1 -          z^-1
//  H(z) = ------------------
//          1 - (1-alpha)z^-1
iirfilt_rrrf sdrk_liquid_iirfilt_rrrf_create_dc_blocker(float _alpha)
{
    // validate input
    if (_alpha <= 0.0f)
        return NULL;

    // compute DC-blocking filter coefficients
    float bf[2] = { 1.0f, -1.0f };
    float af[2] = { 1.0f, -1.0f + _alpha };

    // convert to type-specific array
    float b[2] = { (float)bf[0], (float)bf[1] };
    float a[2] = { (float)af[0], (float)af[1] };
    iirfilt_rrrf q = iirfilt_rrrf_create(b, 2, a, 2);

    // adjust scale so maintain consistent gain across the band
    sdrk_liquid_iirfilt_rrrf_set_scale(q, sqrt(1 - _alpha));
    return q;
}

void sdrk_liquid_iirfilt_rrrf_destroy(iirfilt_rrrf _q)
{
    dotprod_rrrf_destroy(_q->dpa);
    dotprod_rrrf_destroy(_q->dpb);
    free(_q->b);
    free(_q->a);
    free(_q->v);
    free(_q);
}

static int iirfilt_rrrf_execute_norm(iirfilt_rrrf _q, float _x, float* _y)
{
    unsigned int i;

    // advance buffer
    for (i = _q->n - 1; i > 0; i--)
        _q->v[i] = _q->v[i - 1];

    // compute new v
    float v0;
    dotprod_rrrf_execute(_q->dpa, _q->v + 1, &v0);
    v0 = _x - v0;
    _q->v[0] = v0;

    // compute new y
    dotprod_rrrf_execute(_q->dpb, _q->v, _y);

    // apply scaling
    *_y *= _q->scale;
    return LIQUID_OK;
}

static int iirfilt_rrrf_execute_sos(iirfilt_rrrf _q, float _x, float* _y)
{
    (void)_q;
    (void)_x;
    *_y = 0; /* never reached: a DC blocker is created in the normal form */
    return -1;
}

static int iirfilt_rrrf_execute(iirfilt_rrrf _q, float _x, float* _y)
{
    if (_q->type == IIRFILT_TYPE_NORM) {
        return iirfilt_rrrf_execute_norm(_q, _x, _y);
    } else {
        return iirfilt_rrrf_execute_sos(_q, _x, _y);
    }
}

// execute the filter on a block of input samples; the
// input and output buffers may be the same
int sdrk_liquid_iirfilt_rrrf_execute_block(iirfilt_rrrf _q, float* _x, unsigned int _n, float* _y)
{
    unsigned int i;
    for (i = 0; i < _n; i++) {
        // compute output sample
        iirfilt_rrrf_execute(_q, _x[i], &_y[i]);
    }
    return LIQUID_OK;
}

/* ---- iirfilt_crcf: TO = TI = float complex, TC = float ---- */

struct iirfilt_crcf_s {
    float* b;         /* numerator (feed-forward coefficients) */
    float* a;         /* denominator (feed-back coefficients) */
    float complex* v; /* internal filter state (buffer) */
    unsigned int n;   /* filter length (order+1) */

    unsigned int nb; /* numerator length */
    unsigned int na; /* denominator length */

    int type; /* filter structure type */

    dotprod_crcf dpb; /* numerator dot product */
    dotprod_crcf dpa; /* denominator dot product */

    void* qsos;        /* second-order sections filters (unused here) */
    unsigned int nsos; /* number of second-order sections */

    float scale; /* output scaling factor */
};
typedef struct iirfilt_crcf_s* iirfilt_crcf;

static int iirfilt_crcf_reset(iirfilt_crcf _q)
{
    unsigned int i;
    // set internal buffer to zero
    for (i = 0; i < _q->n; i++)
        _q->v[i] = 0;
    return LIQUID_OK;
}

int sdrk_liquid_iirfilt_crcf_set_scale(iirfilt_crcf _q, float _scale)
{
    _q->scale = _scale;
    return LIQUID_OK;
}

static iirfilt_crcf iirfilt_crcf_create(float* _b, unsigned int _nb, float* _a, unsigned int _na)
{
    // validate input
    if (_nb == 0 || _na == 0)
        return NULL;

    // create structure and initialize
    iirfilt_crcf q = (iirfilt_crcf)malloc(sizeof(struct iirfilt_crcf_s));
    q->b = NULL;
    q->a = NULL;
    q->v = NULL;
    q->qsos = NULL;
    q->nsos = 0;
    q->dpb = NULL;
    q->dpa = NULL;
    q->nb = _nb;
    q->na = _na;
    q->n = (q->na > q->nb) ? q->na : q->nb;
    q->type = IIRFILT_TYPE_NORM;

    // allocate memory for numerator, denominator
    q->a = (float*)malloc((q->na) * sizeof(float));
    q->b = (float*)malloc((q->nb) * sizeof(float));

    // normalize coefficients to _a[0]
    float a0 = _a[0];

    unsigned int i;
    for (i = 0; i < q->nb; i++)
        q->b[i] = _b[i] / a0;

    for (i = 0; i < q->na; i++)
        q->a[i] = _a[i] / a0;

    // create buffer and initialize
    q->v = (float complex*)malloc((q->n) * sizeof(float complex));

    q->dpa = dotprod_crcf_create(q->a + 1, q->na - 1);
    q->dpb = dotprod_crcf_create(q->b, q->nb);

    // reset internal state
    iirfilt_crcf_reset(q);

    // set scale and return
    sdrk_liquid_iirfilt_crcf_set_scale(q, 1);
    return q;
}

// create DC-blocking filter
//
//          1 -          z^-1
//  H(z) = ------------------
//          1 - (1-alpha)z^-1
iirfilt_crcf sdrk_liquid_iirfilt_crcf_create_dc_blocker(float _alpha)
{
    // validate input
    if (_alpha <= 0.0f)
        return NULL;

    // compute DC-blocking filter coefficients
    float bf[2] = { 1.0f, -1.0f };
    float af[2] = { 1.0f, -1.0f + _alpha };

    // convert to type-specific array
    float b[2] = { (float)bf[0], (float)bf[1] };
    float a[2] = { (float)af[0], (float)af[1] };
    iirfilt_crcf q = iirfilt_crcf_create(b, 2, a, 2);

    // adjust scale so maintain consistent gain across the band
    sdrk_liquid_iirfilt_crcf_set_scale(q, sqrt(1 - _alpha));
    return q;
}

void sdrk_liquid_iirfilt_crcf_destroy(iirfilt_crcf _q)
{
    dotprod_crcf_destroy(_q->dpa);
    dotprod_crcf_destroy(_q->dpb);
    free(_q->b);
    free(_q->a);
    free(_q->v);
    free(_q);
}

static int iirfilt_crcf_execute_norm(iirfilt_crcf _q, float complex _x, float complex* _y)
{
    unsigned int i;

    // advance buffer
    for (i = _q->n - 1; i > 0; i--)
        _q->v[i] = _q->v[i - 1];

    // compute new v
    float complex v0;
    dotprod_crcf_execute(_q->dpa, _q->v + 1, &v0);
    v0 = _x - v0;
    _q->v[0] = v0;

    // compute new y
    dotprod_crcf_execute(_q->dpb, _q->v, _y);

    // apply scaling
    *_y *= _q->scale;
    return LIQUID_OK;
}

static int iirfilt_crcf_execute_sos(iirfilt_crcf _q, float complex _x, float complex* _y)
{
    (void)_q;
    (void)_x;
    *_y = 0; /* never reached: a DC blocker is created in the normal form */
    return -1;
}

static int iirfilt_crcf_execute(iirfilt_crcf _q, float complex _x, float complex* _y)
{
    if (_q->type == IIRFILT_TYPE_NORM) {
        return iirfilt_crcf_execute_norm(_q, _x, _y);
    } else {
        return iirfilt_crcf_execute_sos(_q, _x, _y);
    }
}

// execute the filter on a block of input samples; the
// input and output buffers may be the same
int sdrk_liquid_iirfilt_crcf_execute_block(iirfilt_crcf _q, float complex* _x, unsigned int _n, float complex* _y)
{
    unsigned int i;
    for (i = 0; i < _n; i++) {
        // compute output sample
        iirfilt_crcf_execute(_q, _x[i], &_y[i]);
    }
    return LIQUID_OK;
}
