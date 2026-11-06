/*
 * The loop of GNU Radio's `linear_equalizer` with the normalized LMS of `adaptive_algorithm_nlms`
 * (GNU Radio main at ec1bf15), translated from C++ to C:
 *
 * - `gr::digital::linear_equalizer_impl::equalize` (gr-digital/lib/linear_equalizer_impl.cc)
 *   with `sps` 1, no training sequence and no tags, so the block runs in decision-directed mode
 *   from the start (`d_training_state` `DD`, the constructor's rule for an empty sequence), one
 *   output, no tap or state outputs;
 * - `adaptive_algorithm::error_dd` (gr-digital/include/gnuradio/digital/adaptive_algorithm.h)
 *   with the decision of `constellation_qpsk` (gr-digital/lib/constellation.cc:
 *   `decision_maker` `2 (imag > 0) + (real > 0)`, `map_to_points` of the points
 *   `(+-GR_M_SQRT2, +-GR_M_SQRT2)` of its constructor, `GR_M_SQRT2` of gnuradio/math.h), and
 *   `adaptive_algorithm_nlms::update_taps` (adaptive_algorithm_nlms.h): `magsq` from
 *   `volk_32fc_x2_conjugate_dot_prod_32fc(in, in)`, then
 *   `taps[i] = conj(conj(taps[i]) + (mu / magsq) in[i] conj(error))`;
 * - the members of `gr::filter::kernel::fir_filter_ccc` the equalizer inherits
 *   (gr-filter/lib/fir_filter.cc): the taps are stored reversed (`set_taps`), `filter` is
 *   `volk_32fc_x2_dot_prod_32fc` of the window of the last `num_taps` samples (oldest first)
 *   with the reversed taps, and `update_tap` writes the tap into every aligned copy. VOLK's
 *   generic machine has an alignment of one sample, so there is one copy (`d_naligned` 1) and
 *   no padding;
 * - the VOLK calls are VOLK's `generic` protokernels, included from the pinned VOLK release (the
 *   scalar form of the dispatched calls, as in gr-quad-demod.c); `std::complex<float>` is C99
 *   `float complex` with the same rules for the products.
 *
 * The block's history of `num_taps - 1` samples, which the GNU Radio scheduler keeps in front
 * of the input buffer, is kept in the state and copied in front of the input of every call into
 * a buffer allocated once by the init function.
 *
 * Compiled once per platform with the generic flags of the other GNU Radio references
 * (gr-agc.c).
 *
 * Copyright 2020 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <stdlib.h>
#include <string.h>

#define LV_HAVE_GENERIC 1
#include <volk/volk_complex.h>
#include <volk/volk_32fc_x2_conjugate_dot_prod_32fc.h>
#include <volk/volk_32fc_x2_dot_prod_32fc.h>

#define SDRK_GR_M_SQRT2 1.41421356237309504880 /* GR_M_SQRT2 of gnuradio/math.h */

typedef struct {
    unsigned int num_taps; /* d_num_taps */
    float step_size;       /* d_step_size of adaptive_algorithm_nlms */
    lv_32fc_t* taps;       /* d_taps of fir_filter_ccc (reversed) */
    lv_32fc_t* aligned;    /* d_aligned_taps[0] */
    lv_32fc_t* buffer;     /* the history, then the input of a call */
    unsigned int capacity; /* the most input samples per call */
    lv_32fc_t error;       /* d_error */
    lv_32fc_t decision;    /* d_decision */
    lv_32fc_t points[4];   /* d_constellation of constellation_qpsk */
} sdrk_gr_linear_equalizer;

/*
 * `linear_equalizer(num_taps, 1, adaptive_algorithm_nlms(qpsk, step_size))` followed by
 * `set_taps(taps)` (natural order, `num_taps` values) for at most `capacity` input samples per
 * call. Returns 0, or -1 when an allocation fails.
 */
int sdrk_gr_linear_equalizer_init(sdrk_gr_linear_equalizer* self, unsigned int num_taps, float step_size, const lv_32fc_t* taps,
                                  unsigned int capacity)
{
    self->num_taps = num_taps;
    self->step_size = step_size;
    self->capacity = capacity;
    self->taps = calloc(num_taps, sizeof(lv_32fc_t));
    self->aligned = calloc(num_taps, sizeof(lv_32fc_t));
    self->buffer = calloc((size_t)num_taps - 1 + capacity, sizeof(lv_32fc_t));
    if (self->taps == NULL || self->aligned == NULL || self->buffer == NULL) {
        free(self->taps);
        free(self->aligned);
        free(self->buffer);
        return -1;
    }
    /* fir_filter::set_taps: the taps reversed, one aligned copy */
    for (unsigned int j = 0; j < num_taps; j++) {
        self->taps[j] = taps[num_taps - 1 - j];
        self->aligned[j] = self->taps[j];
    }
    self->error = lv_cmake(0.0f, 0.0f);
    self->decision = lv_cmake(0.0f, 0.0f);
    /* constellation_qpsk(): Gray-coded */
    self->points[0] = lv_cmake((float)-SDRK_GR_M_SQRT2, (float)-SDRK_GR_M_SQRT2);
    self->points[1] = lv_cmake((float)SDRK_GR_M_SQRT2, (float)-SDRK_GR_M_SQRT2);
    self->points[2] = lv_cmake((float)-SDRK_GR_M_SQRT2, (float)SDRK_GR_M_SQRT2);
    self->points[3] = lv_cmake((float)SDRK_GR_M_SQRT2, (float)SDRK_GR_M_SQRT2);
    return 0;
}

void sdrk_gr_linear_equalizer_free(sdrk_gr_linear_equalizer* self)
{
    free(self->taps);
    free(self->aligned);
    free(self->buffer);
    self->taps = NULL;
    self->aligned = NULL;
    self->buffer = NULL;
}

/* constellation_qpsk::decision_maker */
static unsigned int decision_maker(const lv_32fc_t* sample) { return 2 * (lv_cimag(*sample) > 0) + (lv_creal(*sample) > 0); }

/* adaptive_algorithm::error_dd with constellation::map_to_points (dimensionality 1) */
static lv_32fc_t error_dd(sdrk_gr_linear_equalizer* self, lv_32fc_t wu, lv_32fc_t* decision)
{
    *decision = self->points[decision_maker(&wu)];
    return *decision - wu;
}

/* adaptive_algorithm_nlms::update_taps */
static void update_taps(sdrk_gr_linear_equalizer* self, lv_32fc_t* taps, const lv_32fc_t* in, lv_32fc_t error, unsigned int num_taps)
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
static lv_32fc_t filter(const sdrk_gr_linear_equalizer* self, const lv_32fc_t input[])
{
    lv_32fc_t output;
    volk_32fc_x2_dot_prod_32fc_generic(&output, input, self->aligned, self->num_taps);
    return output;
}

/*
 * `linear_equalizer_impl::equalize` over `n` input samples (at most the capacity of the init
 * function): one output per input.
 */
void sdrk_gr_linear_equalizer_work(sdrk_gr_linear_equalizer* self, lv_32fc_t output_symbols[], const lv_32fc_t input[], unsigned int n)
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
    memmove(self->buffer, self->buffer + n, (size_t)h * sizeof(lv_32fc_t));
}
