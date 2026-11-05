/*
 * The loop of GNU Radio's `costas_loop_cc` (`gr::digital::costas_loop_cc_impl::work` of
 * gr-digital/lib/costas_loop_cc_impl.cc with `phase_detector_2` and `phase_detector_4` of
 * costas_loop_cc_impl.h, and the members of `gr::blocks::control_loop`, gr-blocks/lib/
 * control_loop.cc and gr-blocks/include/gnuradio/blocks/control_loop.h, of GNU Radio main at
 * ec1bf15) translated from C++ to C, member for member: one output (no frequency, phase or
 * error outputs), no `phase_est` tags and `use_snr` false, the case the block runs as
 * `digital.costas_loop_cc(loop_bw, order)`. `gr_expj` is `sincosf` (the configuration of
 * GNU Radio with `HAVE_SINCOSF`), `fast_cc_multiply` and `branchless_clip` of
 * gnuradio/math.h are written out, and the double constants of `phase_wrap` (`2 * GR_M_PI`)
 * and `branchless_clip` (`0.5`) keep their double arithmetic as in the C++ source.
 *
 * GNU Radio compiles this code with the generic flags of its own build (no `-march`), so
 * build.sh compiles this file once per platform with the flags of `gr-agc.c`.
 *
 * Copyright 2006,2010-2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#define _GNU_SOURCE
#include <math.h>

#define SDRK_GR_M_PI 3.14159265358979323846 /* GR_M_PI of gnuradio/math.h */

typedef struct {
    /* gr::blocks::control_loop */
    float d_phase, d_freq;
    float d_max_freq, d_min_freq;
    float d_damping, d_loop_bw;
    float d_alpha, d_beta;
    /* gr::digital::costas_loop_cc_impl */
    float d_error;
    float d_noise;
    int d_use_snr;
    unsigned int d_order;
} sdrk_gr_costas_cc;

typedef struct {
    float re;
    float im;
} sdrk_gr_complex;

/* control_loop::update_gains */
static void update_gains(sdrk_gr_costas_cc* self)
{
    float denom = (1.0 + 2.0 * self->d_damping * self->d_loop_bw + self->d_loop_bw * self->d_loop_bw);
    self->d_alpha = (4 * self->d_damping * self->d_loop_bw) / denom;
    self->d_beta = (4 * self->d_loop_bw * self->d_loop_bw) / denom;
}

/* costas_loop_cc_impl(loop_bw, order, false): control_loop(loop_bw, 1.0, -1.0) */
void sdrk_gr_costas_cc_init(sdrk_gr_costas_cc* self, float loop_bw, unsigned int order)
{
    self->d_phase = 0;
    self->d_freq = 0;
    self->d_max_freq = 1.0;
    self->d_min_freq = -1.0;
    self->d_damping = sqrtf(2.0f) / 2.0f;
    self->d_loop_bw = loop_bw;
    update_gains(self);
    self->d_error = 0;
    self->d_noise = 1.0;
    self->d_use_snr = 0;
    self->d_order = order;
}

float sdrk_gr_costas_cc_frequency(const sdrk_gr_costas_cc* self) { return self->d_freq; }

static inline float phase_detector_4(sdrk_gr_complex sample)
{
    return ((sample.re > 0.0f ? 1.0f : -1.0f) * sample.im - (sample.im > 0.0f ? 1.0f : -1.0f) * sample.re);
}

static inline float phase_detector_2(sdrk_gr_complex sample) { return (sample.re * sample.im); }

static inline float branchless_clip(float x, float clip) { return 0.5 * (fabsf(x + clip) - fabsf(x - clip)); }

static inline void advance_loop(sdrk_gr_costas_cc* self, float error)
{
    self->d_freq = self->d_freq + self->d_beta * error;
    self->d_phase = self->d_phase + self->d_freq + self->d_alpha * error;
}

static inline void phase_wrap(sdrk_gr_costas_cc* self)
{
    while (self->d_phase > (2 * SDRK_GR_M_PI))
        self->d_phase -= 2 * SDRK_GR_M_PI;
    while (self->d_phase < (-2 * SDRK_GR_M_PI))
        self->d_phase += 2 * SDRK_GR_M_PI;
}

static inline void frequency_limit(sdrk_gr_costas_cc* self)
{
    if (self->d_freq > self->d_max_freq)
        self->d_freq = self->d_max_freq;
    else if (self->d_freq < self->d_min_freq)
        self->d_freq = self->d_min_freq;
}

/* costas_loop_cc_impl::work with one output and no tags */
void sdrk_gr_costas_cc_work(sdrk_gr_costas_cc* self, sdrk_gr_complex optr[], const sdrk_gr_complex iptr[], unsigned noutput_items)
{
    for (unsigned i = 0; i < noutput_items; i++) {
        sdrk_gr_complex nco_out;
        sincosf(-self->d_phase, &nco_out.im, &nco_out.re);

        /* gr::fast_cc_multiply(optr[i], iptr[i], nco_out) */
        float o_r = (iptr[i].re * nco_out.re) - (iptr[i].im * nco_out.im);
        float o_i = (iptr[i].re * nco_out.im) + (iptr[i].im * nco_out.re);
        optr[i].re = o_r;
        optr[i].im = o_i;

        switch (self->d_order) {
        case 2:
            self->d_error = phase_detector_2(optr[i]);
            break;
        case 4:
            self->d_error = phase_detector_4(optr[i]);
            break;
        }
        self->d_error = branchless_clip(self->d_error, 1.0);

        advance_loop(self, self->d_error);
        phase_wrap(self);
        frequency_limit(self);
    }
}
