/*
 * GNU Radio's square-QAM receivers (GNU Radio main at ec1bf15), translated from C++ to C, member
 * for member: the decision of `constellation_rect`, the carrier loop of `constellation_receiver_cb`
 * and the decision-directed `linear_equalizer` with `adaptive_algorithm_nlms` on such a
 * constellation.
 *
 * - `gr::digital::constellation_rect` (gr-digital/lib/constellation.cc) with dimensionality 1:
 *   `constellation_sector::decision_maker` is the sector value of `get_sector` (the sector of each
 *   axis `int(x / width + n / 2.0)` in double, clamped to `0 .. n - 1`); the sector values come
 *   from `find_sector_values` (`calc_sector_center`, `get_closest_point`, `get_distance`), computed
 *   once at construction. The points are given by the caller (in symbol order, with the sectors
 *   per axis and their widths) and taken without normalization (`NO_NORMALIZATION`, scale factor
 *   1), so the decisions are those of SDRk's unit-power `qam16` grid; GNU Radio's own
 *   `qam_constellation()` builds the same `constellation_rect` with `AMPLITUDE_NORMALIZATION`.
 * - `gr::digital::constellation_receiver_cb_impl::general_work` and `phase_error_tracking`
 *   (gr-digital/lib/constellation_receiver_cb_impl.cc) without tags, with one complex output: the
 *   derotated sample, which the block writes to its fifth output (`out_symbol`); the symbol,
 *   error, phase and frequency outputs are left out, the decision is still made (the phase error
 *   needs it). `constellation::decision_maker_pe` for dimensionality 1:
 *   `phase_error = -arg(sample conj(point))`, `arg` of `std::complex<float>` being `cargf`. The
 *   members of `gr::blocks::control_loop` (gr-blocks/lib/control_loop.cc,
 *   gr-blocks/include/gnuradio/blocks/control_loop.h) as in gr-costas.c: `advance_loop`,
 *   `phase_wrap` (the double constant `2 * GR_M_PI`) and `frequency_limit`. `gr_expj` is `sincosf`
 *   (the configuration of GNU Radio with `HAVE_SINCOSF`).
 * - `gr::digital::linear_equalizer_impl::equalize` with `adaptive_algorithm_nlms`, exactly as
 *   gr-equalizer.c (`sps` 1, no training sequence, decision-directed from the start, one output,
 *   VOLK's generic dot products, one aligned copy of the taps), with `error_dd` deciding through
 *   the `constellation_rect` above and `map_to_points`.
 *
 * The products of `std::complex<float>` are C99 `float complex` products with the same rules.
 * The states are allocated by the `new` functions (outside the counted calls).
 *
 * Compiled once per platform with the generic flags of the other GNU Radio references
 * (gr-agc.c).
 *
 * Copyright 2011,2012,2013,2018,2020 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#define _GNU_SOURCE
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LV_HAVE_GENERIC 1
#include <volk/volk_complex.h>
#include <volk/volk_32fc_x2_conjugate_dot_prod_32fc.h>
#include <volk/volk_32fc_x2_dot_prod_32fc.h>

#ifndef CMPLXF
#define CMPLXF(x, y) __builtin_complex((float)(x), (float)(y))
#endif

#define SDRK_GR_M_PI 3.14159265358979323846 /* GR_M_PI of gnuradio/math.h */
#define SDRK_GR_MAX_POINTS 64

/* ------------------------------------------------------------------------------------------ */
/* constellation_rect                                                                          */

typedef struct {
    unsigned int d_arity;
    unsigned int n_real_sectors, n_imag_sectors;
    float d_width_real_sectors, d_width_imag_sectors;
    lv_32fc_t d_constellation[SDRK_GR_MAX_POINTS];
    unsigned int sector_values[SDRK_GR_MAX_POINTS];
} sdrk_gr_constellation_rect;

/* constellation::get_distance (dimensionality 1) */
static float get_distance(const sdrk_gr_constellation_rect* self, unsigned int index, const lv_32fc_t* sample)
{
    float dist = 0;
    lv_32fc_t d = sample[0] - self->d_constellation[index];
    dist += lv_creal(d) * lv_creal(d) + lv_cimag(d) * lv_cimag(d); /* std::norm */
    return dist;
}

/* constellation::get_closest_point */
static unsigned int get_closest_point(const sdrk_gr_constellation_rect* self, const lv_32fc_t* sample)
{
    float min_euclid_dist = get_distance(self, 0, sample);
    unsigned int min_index = 0;
    for (unsigned int j = 1; j < self->d_arity; j++) {
        float euclid_dist = get_distance(self, j, sample);
        if (euclid_dist < min_euclid_dist) {
            min_euclid_dist = euclid_dist;
            min_index = j;
        }
    }
    return min_index;
}

/* constellation_rect::calc_sector_center */
static lv_32fc_t calc_sector_center(const sdrk_gr_constellation_rect* self, unsigned int sector)
{
    unsigned int real_sector, imag_sector;
    real_sector = (float)sector / self->n_imag_sectors;
    imag_sector = sector - real_sector * self->n_imag_sectors;
    return CMPLXF((real_sector + 0.5 - self->n_real_sectors / 2.0) * self->d_width_real_sectors,
                  (imag_sector + 0.5 - self->n_imag_sectors / 2.0) * self->d_width_imag_sectors);
}

/* constellation_rect(points, {}, 4, real_sectors, imag_sectors, width_real, width_imag,
 * NO_NORMALIZATION) with find_sector_values; 0, or -1 for too many points or sectors */
static int rect_init(sdrk_gr_constellation_rect* self, const lv_32fc_t* points, unsigned int arity, unsigned int real_sectors,
                     unsigned int imag_sectors, float width_real_sectors, float width_imag_sectors)
{
    if (arity == 0 || arity > SDRK_GR_MAX_POINTS || real_sectors * imag_sectors > SDRK_GR_MAX_POINTS) {
        return -1;
    }
    self->d_arity = arity;
    for (unsigned int i = 0; i < arity; i++) {
        self->d_constellation[i] = points[i];
    }
    self->n_real_sectors = real_sectors;
    self->n_imag_sectors = imag_sectors;
    self->d_width_real_sectors = width_real_sectors;
    self->d_width_imag_sectors = width_imag_sectors;
    for (unsigned int i = 0; i < real_sectors * imag_sectors; i++) {
        lv_32fc_t sector_center = calc_sector_center(self, i);
        self->sector_values[i] = get_closest_point(self, &sector_center);
    }
    return 0;
}

/* constellation_rect::get_sector */
static unsigned int get_sector(const sdrk_gr_constellation_rect* self, const lv_32fc_t* sample)
{
    int real_sector, imag_sector;
    unsigned int sector;

    real_sector = (int)(lv_creal(*sample) / self->d_width_real_sectors + self->n_real_sectors / 2.0);
    if (real_sector < 0)
        real_sector = 0;
    if (real_sector >= (int)self->n_real_sectors)
        real_sector = self->n_real_sectors - 1;

    imag_sector = (int)(lv_cimag(*sample) / self->d_width_imag_sectors + self->n_imag_sectors / 2.0);
    if (imag_sector < 0)
        imag_sector = 0;
    if (imag_sector >= (int)self->n_imag_sectors)
        imag_sector = self->n_imag_sectors - 1;

    sector = real_sector * self->n_imag_sectors + imag_sector;
    return sector;
}

/* constellation_sector::decision_maker */
static unsigned int decision_maker(const sdrk_gr_constellation_rect* self, const lv_32fc_t* sample)
{
    unsigned int sector;
    sector = get_sector(self, sample);
    return self->sector_values[sector];
}

/* ------------------------------------------------------------------------------------------ */
/* constellation_receiver_cb                                                                   */

typedef struct {
    /* gr::blocks::control_loop */
    float d_phase, d_freq;
    float d_max_freq, d_min_freq;
    float d_damping, d_loop_bw;
    float d_alpha, d_beta;
    /* d_constellation */
    sdrk_gr_constellation_rect constellation;
} sdrk_gr_constellation_receiver_cb;

/* control_loop::update_gains */
static void update_gains(sdrk_gr_constellation_receiver_cb* self)
{
    float denom = (1.0 + 2.0 * self->d_damping * self->d_loop_bw + self->d_loop_bw * self->d_loop_bw);
    self->d_alpha = (4 * self->d_damping * self->d_loop_bw) / denom;
    self->d_beta = (4 * self->d_loop_bw * self->d_loop_bw) / denom;
}

/*
 * `constellation_receiver_cb(constellation_rect(...), loop_bw, fmin, fmax)`; NULL when the
 * allocation fails or the constellation is too large.
 */
sdrk_gr_constellation_receiver_cb* sdrk_gr_constellation_receiver_cb_new(const lv_32fc_t* points, unsigned int arity,
                                                                         unsigned int real_sectors, unsigned int imag_sectors,
                                                                         float width_real_sectors, float width_imag_sectors,
                                                                         float loop_bw, float fmin, float fmax)
{
    sdrk_gr_constellation_receiver_cb* self = calloc(1, sizeof(*self));
    if (self == NULL) {
        return NULL;
    }
    if (rect_init(&self->constellation, points, arity, real_sectors, imag_sectors, width_real_sectors, width_imag_sectors) != 0) {
        free(self);
        return NULL;
    }
    /* control_loop(loop_bw, fmax, fmin) */
    self->d_phase = 0;
    self->d_freq = 0;
    self->d_max_freq = fmax;
    self->d_min_freq = fmin;
    self->d_damping = sqrtf(2.0f) / 2.0f;
    self->d_loop_bw = loop_bw;
    update_gains(self);
    return self;
}

void sdrk_gr_constellation_receiver_cb_free(sdrk_gr_constellation_receiver_cb* self) { free(self); }

float sdrk_gr_constellation_receiver_cb_frequency(const sdrk_gr_constellation_receiver_cb* self) { return self->d_freq; }

float sdrk_gr_constellation_receiver_cb_alpha(const sdrk_gr_constellation_receiver_cb* self) { return self->d_alpha; }

float sdrk_gr_constellation_receiver_cb_beta(const sdrk_gr_constellation_receiver_cb* self) { return self->d_beta; }

/* constellation::decision_maker_pe (dimensionality 1) */
static unsigned int decision_maker_pe(const sdrk_gr_constellation_rect* c, const lv_32fc_t* sample, float* phase_error)
{
    unsigned int index = decision_maker(c, sample);
    *phase_error = 0;
    *phase_error += -cargf(sample[0] * conjf(c->d_constellation[index]));
    return index;
}

/* control_loop::advance_loop */
static inline void advance_loop(sdrk_gr_constellation_receiver_cb* self, float error)
{
    self->d_freq = self->d_freq + self->d_beta * error;
    self->d_phase = self->d_phase + self->d_freq + self->d_alpha * error;
}

/* control_loop::phase_wrap */
static inline void phase_wrap(sdrk_gr_constellation_receiver_cb* self)
{
    while (self->d_phase > (2 * SDRK_GR_M_PI))
        self->d_phase -= 2 * SDRK_GR_M_PI;
    while (self->d_phase < (-2 * SDRK_GR_M_PI))
        self->d_phase += 2 * SDRK_GR_M_PI;
}

/* control_loop::frequency_limit */
static inline void frequency_limit(sdrk_gr_constellation_receiver_cb* self)
{
    if (self->d_freq > self->d_max_freq)
        self->d_freq = self->d_max_freq;
    else if (self->d_freq < self->d_min_freq)
        self->d_freq = self->d_min_freq;
}

/* constellation_receiver_cb_impl::phase_error_tracking */
static void phase_error_tracking(sdrk_gr_constellation_receiver_cb* self, float phase_error)
{
    advance_loop(self, phase_error);
    phase_wrap(self);
    frequency_limit(self);
}

/*
 * `constellation_receiver_cb_impl::general_work` without tags over `n` samples; `out_symbol`
 * receives the derotated samples.
 */
void sdrk_gr_constellation_receiver_cb_work(sdrk_gr_constellation_receiver_cb* self, lv_32fc_t out_symbol[], const lv_32fc_t in[],
                                            unsigned int n)
{
    for (unsigned int idx = 0; idx < n; ++idx) {
        float s, c;
        sincosf(self->d_phase, &s, &c); /* gr_expj(d_phase) */
        lv_32fc_t nco = CMPLXF(c, s);   /* NCO value for derotating the current sample */
        lv_32fc_t sample = in[idx] * nco; /* downconverted symbol */

        float phase_error;
        unsigned int sym_value = decision_maker_pe(&self->constellation, &sample, &phase_error);
        (void)sym_value;
        phase_error_tracking(self, phase_error); /* corrects phase and frequency offsets */
        out_symbol[idx] = sample;
    }
}

/* ------------------------------------------------------------------------------------------ */
/* linear_equalizer with adaptive_algorithm_nlms on a constellation_rect                       */

typedef struct {
    unsigned int num_taps; /* d_num_taps */
    float step_size;       /* d_step_size of adaptive_algorithm_nlms */
    lv_32fc_t* taps;       /* d_taps of fir_filter_ccc (reversed) */
    lv_32fc_t* aligned;    /* d_aligned_taps[0] */
    lv_32fc_t* buffer;     /* the history, then the input of a call */
    unsigned int capacity; /* the most input samples per call */
    lv_32fc_t error;       /* d_error */
    lv_32fc_t decision;    /* d_decision */
    sdrk_gr_constellation_rect constellation;
} sdrk_gr_linear_equalizer_rect;

void sdrk_gr_linear_equalizer_rect_free(sdrk_gr_linear_equalizer_rect* self)
{
    if (self != NULL) {
        free(self->taps);
        free(self->aligned);
        free(self->buffer);
        free(self);
    }
}

/*
 * `linear_equalizer(num_taps, 1, adaptive_algorithm_nlms(constellation_rect(...), step_size))`
 * followed by `set_taps(taps)` (natural order, `num_taps` values) for at most `capacity` input
 * samples per call; NULL when an allocation fails or the constellation is too large.
 */
sdrk_gr_linear_equalizer_rect* sdrk_gr_linear_equalizer_rect_new(unsigned int num_taps, float step_size, const lv_32fc_t* taps,
                                                                 unsigned int capacity, const lv_32fc_t* points, unsigned int arity,
                                                                 unsigned int real_sectors, unsigned int imag_sectors,
                                                                 float width_real_sectors, float width_imag_sectors)
{
    sdrk_gr_linear_equalizer_rect* self = calloc(1, sizeof(*self));
    if (self == NULL) {
        return NULL;
    }
    self->num_taps = num_taps;
    self->step_size = step_size;
    self->capacity = capacity;
    self->taps = calloc(num_taps, sizeof(lv_32fc_t));
    self->aligned = calloc(num_taps, sizeof(lv_32fc_t));
    self->buffer = calloc((size_t)num_taps - 1 + capacity, sizeof(lv_32fc_t));
    if (self->taps == NULL || self->aligned == NULL || self->buffer == NULL ||
        rect_init(&self->constellation, points, arity, real_sectors, imag_sectors, width_real_sectors, width_imag_sectors) != 0) {
        sdrk_gr_linear_equalizer_rect_free(self);
        return NULL;
    }
    /* fir_filter::set_taps: the taps reversed, one aligned copy */
    for (unsigned int j = 0; j < num_taps; j++) {
        self->taps[j] = taps[num_taps - 1 - j];
        self->aligned[j] = self->taps[j];
    }
    self->error = lv_cmake(0.0f, 0.0f);
    self->decision = lv_cmake(0.0f, 0.0f);
    return self;
}

/* adaptive_algorithm::error_dd with constellation::map_to_points (dimensionality 1) */
static lv_32fc_t error_dd(sdrk_gr_linear_equalizer_rect* self, lv_32fc_t wu, lv_32fc_t* decision)
{
    *decision = self->constellation.d_constellation[decision_maker(&self->constellation, &wu)];
    return *decision - wu;
}

/* adaptive_algorithm_nlms::update_taps */
static void update_taps(sdrk_gr_linear_equalizer_rect* self, lv_32fc_t* taps, const lv_32fc_t* in, lv_32fc_t error,
                        unsigned int num_taps)
{
    lv_32fc_t dp;
    volk_32fc_x2_conjugate_dot_prod_32fc_generic(&dp, in, in, num_taps);
    float magsq = lv_creal(dp);
    float norm_step_size = self->step_size / magsq;

    for (unsigned i = 0; i < num_taps; i++) {
        taps[i] = lv_conj(lv_conj(taps[i]) + norm_step_size * in[i] * lv_conj(error));
    }
}

/* fir_filter_ccc::filter (alignment of one sample: `al` 0) */
static lv_32fc_t filter(const sdrk_gr_linear_equalizer_rect* self, const lv_32fc_t input[])
{
    lv_32fc_t output;
    volk_32fc_x2_dot_prod_32fc_generic(&output, input, self->aligned, self->num_taps);
    return output;
}

/*
 * `linear_equalizer_impl::equalize` over `n` input samples (at most the capacity of the new
 * function): one output per input.
 */
void sdrk_gr_linear_equalizer_rect_work(sdrk_gr_linear_equalizer_rect* self, lv_32fc_t output_symbols[], const lv_32fc_t input[],
                                        unsigned int n)
{
    unsigned int h = self->num_taps - 1;
    if (n > self->capacity) {
        n = self->capacity;
    }
    memcpy(self->buffer + h, input, (size_t)n * sizeof(lv_32fc_t));
    const lv_32fc_t* samples = self->buffer;

    unsigned int j = 0; /* pre-decimated input buffer index */
    for (unsigned i = 0; i < n; i++) {
        output_symbols[i] = filter(self, &samples[j]);

        /* decision-directed mode: error_dd, then update_taps and update_tap */
        self->error = error_dd(self, output_symbols[i], &self->decision);
        update_taps(self, self->taps, &samples[j], self->error, self->num_taps);
        for (unsigned k = 0; k < self->num_taps; k++) {
            /* fir_filter::update_tap: d_taps[k] and its aligned copy */
            self->aligned[k] = self->taps[k];
        }

        j += 1;
    }
    /* the history of the next call */
    memmove(self->buffer + 0, self->buffer + n, (size_t)h * sizeof(lv_32fc_t));
}
