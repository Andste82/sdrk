/*
 * The loop of GNU Radio's `agc_cc` (`gr::analog::kernel::agc_cc::scale` and `scaleN`,
 * gr-analog/include/gnuradio/analog/agc.h of GNU Radio main at ec1bf15, which the block
 * `analog.agc_cc` runs unchanged) translated from C++ to C, member for member: the four
 * members live in one object that the output stores may alias, so the compiler reloads and
 * stores the gain in every iteration exactly as it does for the C++ class, and `std::sqrt`
 * becomes `sqrtf` with the `errno` rule of the default compiler flags.
 *
 * GNU Radio compiles this header-only code with the generic flags of its own build (no
 * `-march`), whatever level VOLK picks at run time, so build.sh compiles this file once per
 * platform with the flags of the other references and the measurement uses that one object at
 * every level of the platform.
 *
 * Copyright 2006,2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <math.h>

typedef struct {
    float rate;      /* adjustment rate */
    float reference; /* reference value */
    float gain;      /* current gain */
    float max_gain;  /* max allowable gain */
} sdrk_gr_agc_cc;

typedef struct {
    float re;
    float im;
} sdrk_gr_complex;

static inline sdrk_gr_complex scale(sdrk_gr_agc_cc* self, sdrk_gr_complex input)
{
    sdrk_gr_complex output = { input.re * self->gain, input.im * self->gain };
    self->gain += self->rate * (self->reference - sqrtf(output.re * output.re + output.im * output.im));
    if (self->max_gain > 0.0 && self->gain > self->max_gain) {
        self->gain = self->max_gain;
    }
    return output;
}

void sdrk_gr_agc_cc_scale_n(sdrk_gr_agc_cc* self, sdrk_gr_complex output[], const sdrk_gr_complex input[], unsigned n)
{
    for (unsigned i = 0; i < n; i++) {
        output[i] = scale(self, input[i]);
    }
}
