/*
 * GNU Radio's soft decoder `constellation_soft_decoder_cf` (GNU Radio main at ec1bf15)
 * translated from C++ to C, member for member, for a one-dimensional constellation without
 * pre-differential code:
 *
 * - `constellation::calc_soft_dec` (gr-digital/lib/constellation.cc): per symbol the sum over
 *   all `M` points of `exp(-|x - p|^2 / npwr)` into the bit buckets of the point's index
 *   (`argmin` -86: the argument below it gives `3.84745e-36 / -arg`), then per bit
 *   `log(one) - log(zero)` with the floor 1e-45: the exact log-likelihood ratio (log-sum), not
 *   the max-log value;
 * - the look-up table of `gen_soft_dec_lut` (the grid of `2^precision` by `2^precision` cells over
 *   `+-padding * maxamp`, the edge cells repeated) and its use in `soft_decision_maker`
 *   (`branchless_clip` of the axes, the cell index), where the table is set (`set_npwr`);
 * - `constellation_soft_decoder_cf_impl::work` (gr-digital/lib/constellation_soft_decoder_cf_impl.cc):
 *   `bits_per_symbol` soft values per input symbol, the first the most significant bit.
 *
 * The C++ code returns a `std::vector<float>` per symbol, which this translation does not
 * allocate (the work is the same, the allocation of the library's code is not counted: the
 * reference is faster than GNU Radio's own object code).
 *
 * `branchless_clip` of gnuradio/math.h is written out. The libm `expf` and `logf` are the
 * functions GNU Radio calls.
 *
 * Compiled once per platform with the generic flags of the other GNU Radio references
 * (gr-agc.c).
 *
 * Copyright 2013,2014 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float re;
    float im;
} sdrk_gr_complex;

typedef struct {
    unsigned int m;        /* the points */
    unsigned int k;        /* bits per symbol */
    float npwr;            /* d_npwr */
    sdrk_gr_complex* pts;  /* d_constellation */
    int precision;         /* d_lut_precision (0: no table) */
    float lut_scale;       /* d_lut_scale */
    float maxamp;          /* d_maxamp */
    float padding;         /* d_padding */
    float* lut;            /* d_soft_dec_lut, `lut_scale^2` rows of `k` floats */
} sdrk_gr_softdec;

void sdrk_gr_softdec_free(sdrk_gr_softdec* self)
{
    if (self != NULL) {
        free(self->pts);
        free(self->lut);
        free(self);
    }
}

/* constellation::calc_soft_dec, the result into `s[k]` */
static void calc_soft_dec(const sdrk_gr_softdec* self, sdrk_gr_complex sample, float npwr, float* s)
{
    int M = (int)self->m;
    int k = (int)self->k;
    float tmp[2 * 16];
    float smallestnum = 1e-45f;
    for (int i = 0; i < 2 * k; i++) {
        tmp[i] = 0;
    }
    for (int i = 0; i < M; i++) {
        float zr = sample.re - self->pts[i].re;
        float zi = sample.im - self->pts[i].im;
        float dist = zr * zr + zi * zi;
        float arg = -dist / (npwr * 1.0f);
        float argmin = -86;
        float d = 0;
        if (arg < argmin) {
            d = (3.84745e-36f) / -arg;
        } else {
            d = expf(arg);
        }
        int v = i;
        for (int j = 0; j < k; j++) {
            int mask = 1 << j;
            int bit = (v & mask) >> j;
            if (bit == 0) {
                tmp[2 * j + 0] += d;
            } else {
                tmp[2 * j + 1] += d;
            }
        }
    }
    for (int i = 0; i < k; i++) {
        float one = tmp[2 * i + 1];
        float zero = tmp[2 * i + 0];
        if (one < smallestnum) {
            one = smallestnum;
        }
        if (zero < smallestnum) {
            zero = smallestnum;
        }
        s[k - 1 - i] = (logf(one) - logf(zero));
    }
}

/* constellation::gen_soft_dec_lut */
static void gen_soft_dec_lut(sdrk_gr_softdec* self, int precision, float npwr)
{
    unsigned int k = self->k;
    size_t n = 0;
    self->lut_scale = powf(2.0f, (float)precision);
    float border = 1.0f / self->lut_scale;
    float maxd = (self->maxamp * self->padding) - border;
    float step = (2.0f * maxd) / (self->lut_scale - 2.0f);
    int y = 0;
    int endstop = (int)(self->lut_scale - 2);
    float start = -maxd + step / 2;
#define PUSH(X, Y) \
    do { \
        sdrk_gr_complex c = { start + (X) * step, start + (Y) * step }; \
        calc_soft_dec(self, c, npwr, &self->lut[n * k]); \
        n++; \
    } while (0)
    while (y < endstop) {
        int x = 0;
        if (y == 0) {
            while (x < endstop) {
                if (x == 0 || x == endstop - 1) {
                    PUSH(x, y);
                }
                PUSH(x, y);
                x += 1;
            }
            x = 0;
        }
        while (x < endstop) {
            if (x == 0 || x == endstop - 1) {
                PUSH(x, y);
            }
            PUSH(x, y);
            x += 1;
        }
        x = 0;
        if (y == endstop - 1) {
            while (x < endstop) {
                if (x == 0 || x == endstop - 1) {
                    PUSH(x, y);
                }
                PUSH(x, y);
                x += 1;
            }
        }
        y += 1;
    }
#undef PUSH
    self->precision = precision;
}

/*
 * `constellation_soft_decoder_cf(constellation, npwr)` on the `m` points `pts` (pairs, symbol
 * order) with `k` bits per symbol; `precision` 0: `calc_soft_dec` per symbol, above 0: the table
 * of `gen_soft_dec_lut(precision, npwr)` (`set_soft_dec_lut` of the constellation, `d_padding`
 * 2, `d_maxamp` the largest axis value of the points as `max_min_axes` sets it). NULL when an
 * allocation fails.
 */
sdrk_gr_softdec* sdrk_gr_softdec_new(const float* pts, unsigned int m, unsigned int k, float npwr, int precision)
{
    if (k == 0 || k > 16 || (1u << k) != m) {
        return NULL;
    }
    sdrk_gr_softdec* self = calloc(1, sizeof(*self));
    if (self == NULL) {
        return NULL;
    }
    self->m = m;
    self->k = k;
    self->npwr = npwr;
    self->padding = 2.0f;
    self->pts = malloc((size_t)m * sizeof(sdrk_gr_complex));
    if (self->pts == NULL) {
        sdrk_gr_softdec_free(self);
        return NULL;
    }
    float maxamp = 0.0f;
    for (unsigned int i = 0; i < m; i++) {
        self->pts[i].re = pts[2 * i];
        self->pts[i].im = pts[2 * i + 1];
        /* max_min_axes: d_maxamp = max(re_max, -re_min, im_max, -im_min) */
        float a[4] = { self->pts[i].re, -self->pts[i].re, self->pts[i].im, -self->pts[i].im };
        for (int j = 0; j < 4; j++) {
            if (a[j] > maxamp) {
                maxamp = a[j];
            }
        }
    }
    self->maxamp = maxamp;
    if (precision > 0) {
        size_t cells = (size_t)1 << (2 * precision);
        self->lut = malloc(cells * k * sizeof(float));
        if (self->lut == NULL) {
            sdrk_gr_softdec_free(self);
            return NULL;
        }
        gen_soft_dec_lut(self, precision, npwr);
    }
    return self;
}

/* gr::branchless_clip */
static float branchless_clip(float x, float clip)
{
    float x1 = fabsf(x + clip);
    float x2 = fabsf(x - clip);
    x1 -= x2;
    return 0.5f * x1;
}

/* `n` symbols in, `n * k` soft values out (soft_decision_maker per symbol) */
void sdrk_gr_softdec_work(sdrk_gr_softdec* self, float* out, const sdrk_gr_complex* in, unsigned int n)
{
    unsigned int k = self->k;
    for (unsigned int i = 0; i < n; i++) {
        if (self->precision > 0) {
            float maxd = (self->padding * self->maxamp);
            float border = 1.0f / self->lut_scale;
            float limit = maxd - border;
            float xre = branchless_clip(in[i].re, limit) / maxd;
            float xim = branchless_clip(in[i].im, limit) / maxd;
            float scale = (self->lut_scale - 2.0f) / (2.0f);
            xre = floorf((1.0f + xre) * scale) + 1;
            xim = floorf((1.0f + xim) * scale) + 1;
            int index = (int)(self->lut_scale * xim + xre);
            int max_index = (int)(self->lut_scale * self->lut_scale);
            while (index >= max_index) {
                index -= (int)self->lut_scale;
            }
            while (index < 0) {
                index += (int)self->lut_scale;
            }
            memcpy(&out[(size_t)k * i], &self->lut[(size_t)index * k], k * sizeof(float));
        } else {
            calc_soft_dec(self, in[i], self->npwr, &out[(size_t)k * i]);
        }
    }
}
