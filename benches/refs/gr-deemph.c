/*
 * The filter of GNU Radio's `fm_deemph` (gr-analog/python/analog/fm_emph.py, GNU Radio main at
 * ec1bf15): a hierarchical block around `filter.iir_filter_ffd(btaps, ataps, False)` with the
 * first-order taps it designs (bilinear transform of 1 / (1 + s tau) with a prewarped corner,
 * computed in double precision as in Python) and the loop that block runs,
 * `iir_filter_ffd_impl::work` (gr-filter/lib/iir_filter_ffd_impl.cc) calling
 * `iir_filter<float, float, double, double>::filter_n` and `filter`
 * (gr-filter/include/gnuradio/filter/iir_filter.h), translated from C++ to C member for member:
 *
 * - the taps are `std::vector<double>`, the history of the inputs a `std::vector<float>` and
 *   that of the outputs a `std::vector<double>`, each stored twice (`2 * ntaps`) so that the
 *   circular index needs no wrap-around inside the sums; the sizes are read from the vectors in
 *   every call of `filter`;
 * - `acc = fftaps[0] * input + sum fftaps[i] * prev_input[latest_n + i] + sum fbtaps[i] *
 *   prev_output[latest_m + i]` in double, the feedback taps negated once by the new-style
 *   constructor (`oldstyle = false`), the output `(float)acc`.
 *
 * Left out: the check of `d_updated` in `work` (set_taps is never called here) and the
 * exceptions of `set_taps` (the taps are always valid).
 *
 * Copyright 2002,2004,2005,2007,2010,2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

/* The three pointers of a std::vector. */
typedef struct {
    double* begin;
    double* end;
    double* cap;
} vector_double;

typedef struct {
    float* begin;
    float* end;
    float* cap;
} vector_float;

/* iir_filter<float, float, double, double> */
typedef struct {
    bool oldstyle;              /* d_oldstyle */
    vector_double fftaps;       /* d_fftaps */
    vector_double fbtaps;       /* d_fbtaps */
    int latest_n;               /* d_latest_n */
    int latest_m;               /* d_latest_m */
    vector_double prev_output;  /* d_prev_output */
    vector_float prev_input;    /* d_prev_input */
} sdrk_gr_iir_filter_ffd;

static vector_double vector_double_new(size_t n)
{
    double* p = calloc(n > 0 ? n : 1u, sizeof(double));
    vector_double v = { p, p + n, p + n };
    return v;
}

static vector_float vector_float_new(size_t n)
{
    float* p = calloc(n > 0 ? n : 1u, sizeof(float));
    vector_float v = { p, p + n, p + n };
    return v;
}

static unsigned ntaps_ff(const sdrk_gr_iir_filter_ffd* self) { return self->fftaps.end - self->fftaps.begin; }
static unsigned ntaps_fb(const sdrk_gr_iir_filter_ffd* self) { return self->fbtaps.end - self->fbtaps.begin; }

/* iir_filter(fftaps, fbtaps, oldstyle) with set_taps. */
static int iir_filter_init(sdrk_gr_iir_filter_ffd* self, const double* fftaps, size_t n_ff, const double* fbtaps, size_t n_fb, bool oldstyle)
{
    self->oldstyle = oldstyle;
    self->latest_n = 0;
    self->latest_m = 0;
    self->fftaps = vector_double_new(n_ff);
    for (size_t i = 0; i < n_ff; i++)
        self->fftaps.begin[i] = fftaps[i];

    self->fbtaps = vector_double_new(n_fb);
    if (self->oldstyle) {
        for (size_t i = 0; i < n_fb; i++)
            self->fbtaps.begin[i] = fbtaps[i];
    } else {
        // New style negates taps a[1:N-1] to fit with most IIR
        // tap generating programs.
        self->fbtaps.begin[0] = fbtaps[0];
        for (size_t i = 1; i < n_fb; i++) {
            self->fbtaps.begin[i] = -fbtaps[i];
        }
    }

    int n = n_ff;
    int m = n_fb;
    self->prev_input = vector_float_new(2 * n);
    self->prev_output = vector_double_new(2 * m);
    if (self->fftaps.begin == NULL || self->fbtaps.begin == NULL || self->prev_input.begin == NULL || self->prev_output.begin == NULL)
        return -1;
    return 0;
}

/* fm_deemph(fs, tau): the taps of fm_emph.py, then iir_filter_ffd(btaps, ataps, False).
   Returns 0, or -1 when the storage cannot be allocated. */
int sdrk_gr_fm_deemph_init(sdrk_gr_iir_filter_ffd* self, double fs, double tau)
{
    double w_c = 1.0 / tau;

    double w_ca = 2.0 * fs * tan(w_c / (2.0 * fs));

    double k = -w_ca / (2.0 * fs);
    double z1 = -1.0;
    double p1 = (1.0 + k) / (1.0 - k);
    double b0 = -k / (1.0 - k);

    double btaps[2] = { b0 * 1.0, b0 * -z1 };
    double ataps[2] = { 1.0, -p1 };

    return iir_filter_init(self, btaps, 2, ataps, 2, false);
}

void sdrk_gr_fm_deemph_free(sdrk_gr_iir_filter_ffd* self)
{
    free(self->fftaps.begin);
    free(self->fbtaps.begin);
    free(self->prev_output.begin);
    free(self->prev_input.begin);
}

static inline float filter(sdrk_gr_iir_filter_ffd* self, const float input)
{
    double acc;
    unsigned i = 0;
    unsigned n = ntaps_ff(self);
    unsigned m = ntaps_fb(self);

    if (n == 0)
        return (float)0;

    int latest_n = self->latest_n;
    int latest_m = self->latest_m;

    acc = self->fftaps.begin[0] * input;
    for (i = 1; i < n; i++)
        acc += (self->fftaps.begin[i] * self->prev_input.begin[latest_n + i]);
    for (i = 1; i < m; i++)
        acc += (self->fbtaps.begin[i] * self->prev_output.begin[latest_m + i]);

    // store the values twice to avoid having to handle wrap-around in the loop
    self->prev_output.begin[latest_m] = acc;
    self->prev_output.begin[latest_m + m] = acc;
    self->prev_input.begin[latest_n] = input;
    self->prev_input.begin[latest_n + n] = input;

    latest_n--;
    latest_m--;
    if (latest_n < 0)
        latest_n += n;
    if (latest_m < 0)
        latest_m += m;

    self->latest_m = latest_m;
    self->latest_n = latest_n;
    return (float)acc;
}

static void filter_n(sdrk_gr_iir_filter_ffd* self, float output[], const float input[], long n)
{
    for (int i = 0; i < n; i++)
        output[i] = filter(self, input[i]);
}

/* iir_filter_ffd_impl::work */
int sdrk_gr_fm_deemph_work(sdrk_gr_iir_filter_ffd* self, int noutput_items, const float* in, float* out)
{
    filter_n(self, out, in, noutput_items);
    return noutput_items;
}

/* The taps after the constructor: b0, b1 and the stored (negated) feedback tap a1. */
void sdrk_gr_fm_deemph_taps(const sdrk_gr_iir_filter_ffd* self, double taps[3])
{
    taps[0] = self->fftaps.begin[0];
    taps[1] = self->fftaps.begin[1];
    taps[2] = self->fbtaps.begin[1];
}

/* The size of the object, for the storage of the Rust wrapper. */
unsigned long sdrk_gr_fm_deemph_size(void) { return sizeof(sdrk_gr_iir_filter_ffd); }
