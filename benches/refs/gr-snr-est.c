/*
 * GNU Radio's SNR estimator block `gr::digital::mpsk_snr_est_cc` with its M2M4 and SVR
 * estimators (gr-digital/lib/mpsk_snr_est_cc_impl.cc, `mpsk_snr_est_cc_impl::work`, and
 * gr-digital/lib/mpsk_snr_est.cc, `mpsk_snr_est_m2m4` and `mpsk_snr_est_svr`, GNU Radio main at
 * ec1bf15), translated from C++ to C member for member:
 *
 * - the members of `mpsk_snr_est` (`d_alpha`, `d_beta`, `d_signal`, `d_noise`) and of the two
 *   estimators (`d_y1`, `d_y2`);
 * - `update` and `snr` are virtual functions of `mpsk_snr_est`, called through the object's
 *   table of virtual functions once per piece of the work loop (`d_snr_est->update(x, &in[index])`)
 *   and once per tag (`d_snr_est->snr()`); the members live in the object;
 * - `abs(input[i])` of a `gr_complex` is `std::abs` of `std::complex<float>`, which GCC and Clang
 *   compile to `__builtin_cabsf` (the library `cabsf`, a `hypotf`): one call per `abs` as written,
 *   since `cabsf` may set `errno` and is not merged; the products and the averages are in double;
 * - the work function updates the estimator up to every `d_nsamples`-th item and computes
 *   `snr()` there, and updates it with the rest; the block has a history of 2 (`set_history(2)`): `in` points to the item
 *   before the first new one, which SVR reads as `input[i]` next to `input[i + 1]` (M2M4 reads
 *   `input[i]`, so it lags by one item, as in GNU Radio).
 *
 * Left out: the copy of the input to the output at the start of `work` (`memcpy`, the
 * pass-through; a library copy that both sides would make, which the instruction counts of a
 * `rep movsb` per byte overstate, so the work of the estimate is compared without it: ours is
 * measured by `SnrEstimateBlock::observe`, D2504), the stream tag of every estimate (`add_item_tag` with `pmt::from_double(snr)`, once per
 * `d_nsamples` items; the estimate is kept in `last_snr` instead), the estimators SIMPLE and SKEW,
 * and the GNU Radio block machinery.
 *
 * Copyright 2011,2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <complex.h>
#include <math.h>
#include <stddef.h>

typedef struct sdrk_gr_mpsk_snr_est_cc sdrk_gr_mpsk_snr_est_cc;

/* The virtual functions of mpsk_snr_est that the block calls. */
typedef struct {
    int (*update)(sdrk_gr_mpsk_snr_est_cc* self, int noutput_items, const float _Complex* input);
    double (*snr)(sdrk_gr_mpsk_snr_est_cc* self);
} snr_est_vtable;

struct sdrk_gr_mpsk_snr_est_cc {
    /* mpsk_snr_est */
    const snr_est_vtable* vptr;
    double d_alpha;
    double d_beta;
    double d_signal;
    double d_noise;
    /* mpsk_snr_est_m2m4, mpsk_snr_est_svr */
    double d_y1;
    double d_y2;
    /* mpsk_snr_est_cc_impl */
    int d_nsamples;
    int d_count;
    double last_snr;
};

static int m2m4_update(sdrk_gr_mpsk_snr_est_cc* self, int noutput_items, const float _Complex* input)
{
    for (int i = 0; i < noutput_items; i++) {
        double y1 = cabsf(input[i]) * cabsf(input[i]);
        self->d_y1 = self->d_alpha * y1 + self->d_beta * self->d_y1;

        double y2 = cabsf(input[i]) * cabsf(input[i]) * cabsf(input[i]) * cabsf(input[i]);
        self->d_y2 = self->d_alpha * y2 + self->d_beta * self->d_y2;
    }
    return noutput_items;
}

static double m2m4_snr(sdrk_gr_mpsk_snr_est_cc* self)
{
    double y1_2 = self->d_y1 * self->d_y1;
    self->d_signal = sqrt(2 * y1_2 - self->d_y2);
    self->d_noise = self->d_y1 - sqrt(2 * y1_2 - self->d_y2);
    return 10.0 * log10(self->d_signal / self->d_noise);
}

static int svr_update(sdrk_gr_mpsk_snr_est_cc* self, int noutput_items, const float _Complex* input)
{
    for (int i = 0; i < noutput_items; i++) {
        double x = cabsf(input[i + 1]);
        double x1 = cabsf(input[i]);
        double y1 = (x * x) * (x1 * x1);
        self->d_y1 = self->d_alpha * y1 + self->d_beta * self->d_y1;

        double y2 = x * x * x * x;
        self->d_y2 = self->d_alpha * y2 + self->d_beta * self->d_y2;
    }
    return noutput_items;
}

static double svr_snr(sdrk_gr_mpsk_snr_est_cc* self)
{
    double x = self->d_y1 / (self->d_y2 - self->d_y1);
    return 10.0 * log10(x - 1 + sqrt(x * (x - 1)));
}

static const snr_est_vtable m2m4_vtable = { m2m4_update, m2m4_snr };
static const snr_est_vtable svr_vtable = { svr_update, svr_snr };

size_t sdrk_gr_mpsk_snr_est_cc_size(void) { return sizeof(sdrk_gr_mpsk_snr_est_cc); }

/* The constructors of mpsk_snr_est_cc_impl (set_alpha, set_type) and of the estimator:
 * `svr` 0 is SNR_EST_M2M4, 1 is SNR_EST_SVR. */
void sdrk_gr_mpsk_snr_est_cc_init(sdrk_gr_mpsk_snr_est_cc* self, int svr, int tag_nsamples, double alpha)
{
    self->vptr = svr ? &svr_vtable : &m2m4_vtable;
    self->d_signal = 0;
    self->d_noise = 0;
    self->d_alpha = alpha;
    self->d_beta = 1.0 - alpha;
    self->d_y1 = 0;
    self->d_y2 = 0;
    self->d_nsamples = tag_nsamples;
    self->d_count = 0;
    self->last_snr = 0;
}

/* mpsk_snr_est_cc_impl::work: `in` holds the history item and then `noutput_items` new items. */
int sdrk_gr_mpsk_snr_est_cc_work(sdrk_gr_mpsk_snr_est_cc* self, int noutput_items, const float _Complex* in)
{
    // Update, calculate, and issue an SNR tag every d_nsamples
    int index = 0, x = 0;
    while (index + (self->d_nsamples - self->d_count) < noutput_items) {
        x = self->d_nsamples - self->d_count;

        // Update the SNR estimate registers from the current input
        self->vptr->update(self, x, &in[index]);

        // Issue a tag with the SNR data
        self->last_snr = self->vptr->snr(self);

        index += x;
        self->d_count = 0;
    }

    // Keep track of remaining items and update estimators
    x = noutput_items - index;
    self->d_count += x;
    self->vptr->update(self, x, &in[index]);

    return noutput_items;
}

/* The estimate of the last tag. */
double sdrk_gr_mpsk_snr_est_cc_last_snr(const sdrk_gr_mpsk_snr_est_cc* self) { return self->last_snr; }
