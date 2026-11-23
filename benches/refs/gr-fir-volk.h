/*
 * The aligned tap copies and the dispatched dot product of GNU Radio's
 * `gr::filter::kernel::fir_filter<gr_complex, gr_complex, gr_complex>` (gr-filter/lib/
 * fir_filter.cc, GNU Radio main at ec1bf15), the form it has on a machine with SIMD:
 *
 * - `d_align` is `volk_get_alignment()` bytes and `d_naligned` is `max(1, d_align / 8)` copies of
 *   the (reversed) taps, copy `i` shifted by `i` samples (`d_aligned_taps[i][i + j]`, the vector
 *   of `d_ntaps + d_naligned - 1` samples, each vector aligned);
 * - `filter` rounds the input pointer down to the alignment (`ar`), takes the offset `al` in
 *   samples and calls `volk_32fc_x2_dot_prod_32fc_a(ar, d_aligned_taps[al], d_ntaps + al)`:
 *   VOLK's dispatcher, which runs the implementation of the machine it selects (the pinned
 *   `volk_config` of the level);
 * - `update_tap` writes the tap into every copy.
 *
 * Used by gr-equalizer.c and gr-constellation.c compiled with -DSDRK_GR_VOLK (the dispatched
 * variants, `libgrvolk.a`); the other build keeps VOLK's `generic` protokernel.
 *
 * Copyright 2002,2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef SDRK_GR_FIR_VOLK_H
#define SDRK_GR_FIR_VOLK_H

#include <stdint.h>
#include <string.h>
#include <volk/volk.h>

/* `d_align` and `d_naligned` of fir_filter are members set once in the constructor: cached here
 * (the layout of the state is fixed by the Rust mirror, so a static) instead of a call into
 * libvolk per tap and sample */
static size_t fv_align_cache;
static unsigned int fv_naligned_cache;

static inline size_t fv_align(void)
{
    if (fv_align_cache == 0) {
        size_t n = volk_get_alignment() / sizeof(lv_32fc_t);
        fv_naligned_cache = n < 1 ? 1u : (unsigned int)n;
        fv_align_cache = volk_get_alignment();
    }
    return fv_align_cache;
}

static inline unsigned int fv_naligned(void)
{
    (void)fv_align();
    return fv_naligned_cache;
}

/* The distance of two copies in samples: the vector of `d_ntaps + d_naligned - 1` samples,
 * rounded up to the alignment (every `volk::vector` of GNU Radio starts aligned, so the
 * aligned protokernels may load from it) */
static inline size_t fv_stride(unsigned int ntaps)
{
    size_t na = fv_naligned(); /* a power of two: the alignment of VOLK in samples */
    size_t len = (size_t)ntaps + na - 1;
    return (len + na - 1) & ~(na - 1);
}

/* The `naligned` zeroed copies (one block, `volk_free` frees it), NULL when it fails. */
static inline lv_32fc_t* fv_alloc(unsigned int ntaps)
{
    size_t n = (size_t)fv_naligned() * fv_stride(ntaps);
    lv_32fc_t* p = (lv_32fc_t*)volk_malloc(n * sizeof(lv_32fc_t), fv_align());
    if (p != NULL) {
        memset(p, 0, n * sizeof(lv_32fc_t));
    }
    return p;
}

/* fir_filter::update_tap (and, for every index, set_taps): `d_aligned_taps[i][i + index] = t` */
static inline void fv_update_tap(lv_32fc_t* copies, unsigned int ntaps, unsigned int index, lv_32fc_t t)
{
    unsigned int na = fv_naligned();
    size_t stride = fv_stride(ntaps);
    for (unsigned int i = 0; i < na; i++) {
        copies[i * stride + i + index] = t;
    }
}

/* fir_filter::filter */
static inline lv_32fc_t fv_filter(const lv_32fc_t* copies, unsigned int ntaps, const lv_32fc_t* input)
{
    size_t align = fv_align();
    const lv_32fc_t* ar = (const lv_32fc_t*)((uintptr_t)input & ~(uintptr_t)(align - 1));
    unsigned int al = (unsigned int)(input - ar);
    lv_32fc_t out;
    volk_32fc_x2_dot_prod_32fc_a(&out, ar, &copies[(size_t)al * fv_stride(ntaps)], ntaps + al);
    return out;
}

#endif
