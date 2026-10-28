/*
 * The loop of GNU Radio's `pwr_squelch_cc` (gr-analog/lib/squelch_base_cc_impl.cc,
 * `squelch_base_cc_impl::general_work`, with `pwr_squelch_cc_impl::update_state` and `mute`
 * of gr-analog/lib/pwr_squelch_cc_impl.{h,cc} and `filter::single_pole_iir<double, double,
 * double>::filter` of gr-filter/include/gnuradio/filter/single_pole_iir.h, GNU Radio main at
 * ec1bf15, which the block `analog.pwr_squelch_cc` runs unchanged) translated from C++ to C,
 * member for member:
 *
 * - `update_state` and `mute` are virtual functions of `squelch_base_cc_impl`, called through
 *   the object's table of virtual functions in every iteration (`general_work` is compiled in
 *   another translation unit than the overrides of `pwr_squelch_cc_impl`, so the C++ compiler
 *   cannot bind them statically either); the members live in the object and are reloaded after
 *   each call;
 * - the average is `d_alpha * x + d_one_minus_alpha * d_prev_output` in double precision, its
 *   input the power `re * re + im * im` computed in float and widened;
 * - the output is `in[i] * gr_complex(d_envelope, 0.0)`, a complex product of `float` with the
 *   C99/C++ rules for infinities and NaNs (the run-time helper `__mulsc3` when the result is
 *   NaN), or zero while muted when the block does not gate.
 *
 * Left out: the scoped lock of `d_setlock` (once per call) and the stream tags `squelch_sob` and
 * `squelch_eob` that `add_item_tag` writes at the transitions of the gate (twice per call of
 * the measurement); the flag `d_tag_next_unmuted` is kept.
 *
 * Copyright 2004,2006,2010,2012,2018 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#define GR_M_PI 3.14159265358979323846

enum squelch_state { ST_MUTED, ST_ATTACK, ST_UNMUTED, ST_DECAY };

typedef struct sdrk_gr_pwr_squelch_cc sdrk_gr_pwr_squelch_cc;

/* The virtual functions of squelch_base_cc_impl that general_work calls. */
typedef struct {
    void (*update_state)(sdrk_gr_pwr_squelch_cc* self, const float _Complex* in);
    bool (*mute)(const sdrk_gr_pwr_squelch_cc* self);
} squelch_vtable;

struct sdrk_gr_pwr_squelch_cc {
    const squelch_vtable* vptr;
    /* squelch_base_cc_impl */
    bool d_tag_next_unmuted;
    int d_ramp;
    int d_ramped;
    bool d_gate;
    double d_envelope;
    enum squelch_state d_state;
    /* pwr_squelch_cc_impl */
    double d_threshold;
    double d_pwr;
    /* filter::single_pole_iir<double, double, double> d_iir */
    double d_alpha;
    double d_one_minus_alpha;
    double d_prev_output;
};

static double iir_filter(sdrk_gr_pwr_squelch_cc* self, double input)
{
    double output = self->d_alpha * input + self->d_one_minus_alpha * self->d_prev_output;
    self->d_prev_output = output;
    return output;
}

static void pwr_update_state(sdrk_gr_pwr_squelch_cc* self, const float _Complex* in)
{
    self->d_pwr = iir_filter(self, crealf(*in) * crealf(*in) + cimagf(*in) * cimagf(*in));
}

static bool pwr_mute(const sdrk_gr_pwr_squelch_cc* self) { return self->d_pwr < self->d_threshold; }

static const squelch_vtable pwr_squelch_vtable = { pwr_update_state, pwr_mute };

size_t sdrk_gr_pwr_squelch_cc_size(void) { return sizeof(sdrk_gr_pwr_squelch_cc); }

/* The constructors of squelch_base_cc_impl, pwr_squelch_cc_impl and single_pole_iir. */
void sdrk_gr_pwr_squelch_cc_init(sdrk_gr_pwr_squelch_cc* self, double db, double alpha, int ramp, bool gate)
{
    self->vptr = &pwr_squelch_vtable;
    self->d_tag_next_unmuted = true;
    self->d_ramp = ramp;
    self->d_gate = gate;
    self->d_state = ST_MUTED;
    self->d_envelope = self->d_ramp ? 0.0 : 1.0;
    self->d_ramped = 0;
    self->d_pwr = 0;
    self->d_prev_output = 0;
    self->d_alpha = alpha;
    self->d_one_minus_alpha = 1.0 - alpha;
    self->d_threshold = pow(10.0, db / 10);
}

int sdrk_gr_pwr_squelch_cc_work(sdrk_gr_pwr_squelch_cc* self, int noutput_items, const float _Complex* in, float _Complex* out)
{
    int j = 0;

    for (int i = 0; i < noutput_items; i++) {
        self->vptr->update_state(self, &in[i]);

        switch (self->d_state) {
        case ST_MUTED:
            if (!self->vptr->mute(self)) {
                self->d_state = self->d_ramp ? ST_ATTACK : ST_UNMUTED;
                if (self->d_state == ST_UNMUTED)
                    self->d_tag_next_unmuted = true;
            }
            break;

        case ST_UNMUTED:
            if (self->d_tag_next_unmuted) {
                self->d_tag_next_unmuted = false;
            }
            if (self->vptr->mute(self)) {
                self->d_state = self->d_ramp ? ST_DECAY : ST_MUTED;
            }
            break;

        case ST_ATTACK:
            self->d_envelope = 0.5 - cos(GR_M_PI * (++self->d_ramped) / self->d_ramp) / 2.0;
            if (self->d_ramped >= self->d_ramp) {
                self->d_state = ST_UNMUTED;
                self->d_tag_next_unmuted = true;
                self->d_envelope = 1.0;
            }
            break;

        case ST_DECAY:
            self->d_envelope = 0.5 - cos(GR_M_PI * (--self->d_ramped) / self->d_ramp) / 2.0;
            if (self->d_ramped == 0.0) {
                self->d_state = ST_MUTED;
            }
            break;
        }

        if (self->d_state != ST_MUTED) {
            out[j++] = in[i] * __builtin_complex((float)self->d_envelope, 0.0f);
        } else {
            if (!self->d_gate) {
                out[j++] = 0.0f;
            }
        }
    }

    return j;
}
