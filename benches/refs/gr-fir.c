/*
 * GNU Radio's `fir_filter_ccf` and `interp_fir_filter_ccf` (complex input, real taps; GNU Radio
 * main at ec1bf15) translated from C++ to C, member for member:
 *
 * - `gr::filter::interp_fir_filter_impl` (gr-filter/lib/interp_fir_filter_impl.cc): the taps
 *   are padded with zeros to a multiple of the interpolation `M`, tap `i` goes to the branch
 *   `i % M` at the place `i / M` (`install_taps`), the history is `nt = taps / M` samples, and
 *   `work` gives, per input sample, `out[nf] = d_firs[nf].filter(&in[i])` for the `M` branches.
 *   `M` = 1 is `fir_filter_ccf` (`fir_filter_ccf_impl`, decimation 1: one branch of all taps,
 *   `out[i] = filter(&in[i])`);
 * - `gr::filter::kernel::fir_filter<gr_complex, gr_complex, float>` (gr-filter/lib/
 *   fir_filter.cc): the taps are stored reversed (`set_taps`), and `filter` is
 *   `volk_32fc_32f_dot_prod_32fc` over the window of the last `nt` samples (oldest first).
 *
 * Two builds of this file (benches/refs/build.sh):
 *
 *  - generic: VOLK's `generic` protokernel, one copy of the taps (a machine without SIMD has an
 *    alignment of one sample): the scalar reference, in the archive of gr-symbol-sync.c;
 *  - -DSDRK_GR_VOLK: the form of `fir_filter` on a machine with SIMD (gr-fir-volk.h): per branch
 *    `d_naligned` copies of the taps shifted by one sample each, the input pointer rounded down
 *    to the alignment and `volk_32fc_32f_dot_prod_32fc_a`, VOLK's dispatcher, which runs the
 *    implementation of the machine the pinned `volk_config` of the level selects; every exported
 *    name is prefixed `sdrk_gr_volk_` (libgrvolk.a, linked to libvolk).
 *
 * The block's history of `nt - 1` samples, which the GNU Radio scheduler keeps in front of the
 * input buffer, is kept in the state and copied in front of the input of every call into a
 * buffer allocated once by the new function (aligned to VOLK's alignment in the second build).
 *
 * Compiled with the generic flags of the other GNU Radio references (gr-agc.c).
 *
 * Copyright 2002,2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <stdlib.h>
#include <string.h>

#define LV_HAVE_GENERIC 1
#include <volk/volk_complex.h>
#ifdef SDRK_GR_VOLK
#include "gr-fir-volk.h"
#else
#include <volk/volk_32fc_32f_dot_prod_32fc.h>
#endif

typedef struct {
    unsigned int interp;   /* the interpolation M (1: fir_filter_ccf) */
    unsigned int nt;       /* the taps of a branch, also the history + 1 */
    unsigned int capacity; /* the most input samples per call */
    size_t block;          /* floats between the branches */
    size_t stride;         /* floats between two aligned copies of a branch */
    float* taps;           /* the reversed taps of the branches (and their aligned copies) */
    lv_32fc_t* buffer;     /* the history, then the input of a call */
} sdrk_gr_fir_bank;

#ifdef SDRK_GR_VOLK
/* d_naligned copies, each a volk::vector of `nt + d_naligned - 1` floats starting aligned */
static size_t fir_stride(unsigned int nt)
{
    size_t per_vec = fv_align() / sizeof(float);
    size_t len = (size_t)nt + fv_naligned() - 1;
    return (len + per_vec - 1) / per_vec * per_vec;
}
#endif

void sdrk_gr_fir_bank_free(sdrk_gr_fir_bank* self)
{
    if (self == NULL) {
        return;
    }
#ifdef SDRK_GR_VOLK
    volk_free(self->taps);
    volk_free(self->buffer);
#else
    free(self->taps);
    free(self->buffer);
#endif
    free(self);
}

/*
 * `interp_fir_filter_ccf(interp, taps)` (`fir_filter_ccf(1, taps)` for `interp` 1) for at most
 * `capacity` input samples per call; NULL when an allocation fails.
 */
sdrk_gr_fir_bank* sdrk_gr_fir_bank_new(const float* taps, unsigned int ntaps, unsigned int interp, unsigned int capacity)
{
    if (ntaps == 0 || interp == 0) {
        return NULL;
    }
    sdrk_gr_fir_bank* self = calloc(1, sizeof(*self));
    if (self == NULL) {
        return NULL;
    }
    self->interp = interp;
    /* set_taps: zeros to a multiple of the interpolation; install_taps: nt taps per branch */
    self->nt = (ntaps + interp - 1) / interp;
    self->capacity = capacity;
#ifdef SDRK_GR_VOLK
    unsigned int na = fv_naligned();
    self->stride = fir_stride(self->nt);
    self->block = (size_t)na * self->stride;
    self->taps = volk_malloc(interp * self->block * sizeof(float), fv_align());
    self->buffer = volk_malloc(((size_t)self->nt - 1 + capacity) * sizeof(lv_32fc_t), fv_align());
    if (self->taps != NULL) {
        memset(self->taps, 0, interp * self->block * sizeof(float));
    }
    if (self->buffer != NULL) {
        memset(self->buffer, 0, ((size_t)self->nt - 1 + capacity) * sizeof(lv_32fc_t));
    }
#else
    self->stride = self->nt;
    self->block = self->nt;
    self->taps = calloc((size_t)interp * self->nt, sizeof(float));
    self->buffer = calloc((size_t)self->nt - 1 + capacity, sizeof(lv_32fc_t));
#endif
    if (self->taps == NULL || self->buffer == NULL) {
        sdrk_gr_fir_bank_free(self);
        return NULL;
    }
    for (unsigned int n = 0; n < interp; n++) {
        for (unsigned int k = 0; k < self->nt; k++) {
            /* xtaps[n][j] = taps[j * M + n], then fir_filter::set_taps reverses a branch */
            unsigned int j = self->nt - 1 - k;
            size_t src = (size_t)j * interp + n;
            float t = src < ntaps ? taps[src] : 0.0f;
#ifdef SDRK_GR_VOLK
            for (unsigned int i = 0; i < na; i++) {
                self->taps[n * self->block + i * self->stride + i + k] = t;
            }
#else
            self->taps[n * self->block + k] = t;
#endif
        }
    }
    return self;
}

/* fir_filter::filter of branch `nf` */
static lv_32fc_t filter(const sdrk_gr_fir_bank* self, unsigned int nf, const lv_32fc_t* input)
{
    lv_32fc_t out;
#ifdef SDRK_GR_VOLK
    const lv_32fc_t* ar = (const lv_32fc_t*)((uintptr_t)input & ~(uintptr_t)(fv_align() - 1));
    unsigned int al = (unsigned int)(input - ar);
    volk_32fc_32f_dot_prod_32fc_a(&out, ar, &self->taps[nf * self->block + (size_t)al * self->stride], self->nt + al);
#else
    volk_32fc_32f_dot_prod_32fc_generic(&out, input, &self->taps[nf * self->block], self->nt);
#endif
    return out;
}

/* `work` over `n` input samples (at most the capacity): `n * interp` outputs */
void sdrk_gr_fir_bank_work(sdrk_gr_fir_bank* self, lv_32fc_t* output, const lv_32fc_t* input, unsigned int n)
{
    unsigned int h = self->nt - 1;
    if (n > self->capacity) {
        n = self->capacity;
    }
    memcpy(self->buffer + h, input, (size_t)n * sizeof(lv_32fc_t));
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int nf = 0; nf < self->interp; nf++) {
            output[(size_t)i * self->interp + nf] = filter(self, nf, &self->buffer[i]);
        }
    }
    /* the history of the next call */
    memmove(self->buffer, self->buffer + n, (size_t)h * sizeof(lv_32fc_t));
}
