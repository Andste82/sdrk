/*
 * The `work` of GNU Radio's `quadrature_demod_cf` (gr-analog/lib/quadrature_demod_cf_impl.cc,
 * GNU Radio main at ec1bf15, which the block `analog.quadrature_demod_cf` runs unchanged)
 * translated from C++ to C:
 *
 * - the block has a history of two samples: `in` holds `noutput_items + 1` samples, `in[0]`
 *   the last one of the previous call;
 * - `std::vector<gr_complex> tmp(noutput_items)` is a zero-filled heap buffer per call
 *   (`calloc`, freed at the end);
 * - `volk_32fc_x2_multiply_conjugate_32fc(tmp, &in[1], &in[0], n)` is VOLK's `generic`
 *   protokernel, included from the pinned VOLK release (the scalar form of the dispatched call,
 *   `tmp[i] = in[i + 1] * conj(in[i])`, a complex product of `float` with the C99 rules for
 *   infinities and NaNs);
 * - `out[i] = d_gain * gr::fast_atan2f(imag(tmp[i]), real(tmp[i]))` calls the function of
 *   libgnuradio-runtime out of line (gr-fast-atan2f.c, another object file).
 *
 * Compiled once per platform with the generic flags of the other GNU Radio references
 * (gr-agc.c).
 *
 * Copyright 2004,2005,2010,2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <stdlib.h>

#define LV_HAVE_GENERIC 1
#include <volk/volk_complex.h>
#include <volk/volk_32fc_x2_multiply_conjugate_32fc.h>

float sdrk_gr_fast_atan2f(float y, float x);

typedef struct {
    float gain; /* d_gain */
} sdrk_gr_quadrature_demod_cf;

void sdrk_gr_quadrature_demod_cf_init(sdrk_gr_quadrature_demod_cf* self, float gain)
{
    self->gain = gain;
}

/* `in` holds `noutput_items + 1` samples (the history of two), `out` `noutput_items`. Returns
   `noutput_items`, or -1 when the buffer cannot be allocated. */
int sdrk_gr_quadrature_demod_cf_work(sdrk_gr_quadrature_demod_cf* self, int noutput_items, const lv_32fc_t* in, float* out)
{
    lv_32fc_t* tmp = calloc(noutput_items > 0 ? (size_t)noutput_items : 1u, sizeof(lv_32fc_t));
    if (tmp == NULL)
        return -1;
    volk_32fc_x2_multiply_conjugate_32fc_generic(&tmp[0], &in[1], &in[0], noutput_items);
    for (int i = 0; i < noutput_items; i++) {
        out[i] = self->gain * sdrk_gr_fast_atan2f(lv_cimag(tmp[i]), lv_creal(tmp[i]));
    }
    free(tmp);
    return noutput_items;
}
