/*
 * The scalar rotator of GNU Radio's gr-blocks (`gr::blocks::rotator::rotate`,
 * gr-blocks/include/gnuradio/blocks/rotator.h of GNU Radio main at ec1bf15) translated from
 * C++ to C, member for member: per sample the input times the phase, the phase times the
 * increment (two complex products of `float` with the C99/C++ rules for infinities and NaNs,
 * the run-time helper `__mulsc3` when a result is NaN) and every 512th sample the phase divided
 * by its magnitude (`std::abs` of a `std::complex<float>` is `cabsf`). The members live in one
 * object that the output stores may alias, as in the C++ class.
 *
 * The constructor state is that of `rotator()` followed by `set_phase` and `set_phase_incr`,
 * which normalize their argument. (`rotateN` calls VOLK's `volk_32fc_s32fc_x2_rotator2_32fc`,
 * which benches/volk-sys declares; this file is the scalar method of the same class.)
 *
 * Compiled once per platform with the generic flags of the other GNU Radio references
 * (gr-agc.c).
 *
 * Copyright 2003,2008,2013 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <complex.h>

typedef struct {
    float _Complex phase;      /* d_phase */
    float _Complex phase_incr; /* d_phase_incr */
    unsigned int counter;      /* d_counter */
} sdrk_gr_rotator;

/* rotator() : d_phase(1), d_phase_incr(1), d_counter(0), then set_phase(phase) and
   set_phase_incr(incr). */
void sdrk_gr_rotator_init(sdrk_gr_rotator* self, const float _Complex* phase, const float _Complex* incr)
{
    self->phase = 1.0f;
    self->phase_incr = 1.0f;
    self->counter = 0;
    self->phase = *phase / cabsf(*phase);
    self->phase_incr = *incr / cabsf(*incr);
}

static inline float _Complex rotate(sdrk_gr_rotator* self, float _Complex in)
{
    self->counter++;

    float _Complex z = in * self->phase; /* rotate in by phase */
    self->phase *= self->phase_incr;     /* incr our phase (complex mult == add phases) */

    if ((self->counter % 512) == 0)
        self->phase /= cabsf(self->phase); /* Normalize to ensure multiplication is rotation */

    return z;
}

/* The loop of a caller of `rotate`: `out[i] = rotate(in[i])` for `n` samples. */
void sdrk_gr_rotator_rotate_n(sdrk_gr_rotator* self, float _Complex* out, const float _Complex* in, unsigned n)
{
    for (unsigned i = 0; i < n; i++)
        out[i] = rotate(self, in[i]);
}

/* The current phase (`phase()`). */
void sdrk_gr_rotator_phase(const sdrk_gr_rotator* self, float _Complex* out)
{
    *out = self->phase;
}
