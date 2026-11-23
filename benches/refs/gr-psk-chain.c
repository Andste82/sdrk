/*
 * The two blocks of GNU Radio's QPSK receiver chain that no other file of this directory holds,
 * translated from C++ to C, member for member (GNU Radio main at ec1bf15):
 *
 * - `fir_filter_ccf` (`gr::filter::kernel::fir_filter<gr_complex, gr_complex, float>` of
 *   gr-filter/lib/fir_filter.cc, decimation 1): the taps are stored reversed (`set_taps`), and
 *   `filter` is VOLK's `generic` `volk_32fc_32f_dot_prod_32fc` over the window of the last
 *   `ntaps` samples (a machine without SIMD has an alignment of one sample, so one copy of the
 *   taps and no offset);
 * - `constellation_decoder_cb` of gr-digital/lib/constellation_decoder_cb_impl.cc with
 *   `constellation_qpsk` (`decision_maker` of constellation.cc: `2 * (imag > 0) + (real > 0)`).
 *
 * `psk_demod_block` is counted against the chain `fir_filter_ccf` + `symbol_sync_cc`
 * (gr-symbol-sync.c) + `costas_loop_cc` (gr-costas.c) + `constellation_decoder_cb`, the sum of
 * the blocks (the receiver flow `qpsk_receiver`).
 *
 * GNU Radio compiles this code with the generic flags of its own build (no `-march`), so
 * build.sh compiles this file once per platform with the flags of `gr-agc.c`.
 *
 * Copyright 2004,2010,2012,2018 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <stddef.h>

typedef struct {
    float re;
    float im;
} sdrk_gr_complex;

/* fir_filter::filter for gr_complex input and float taps: the window `input[0 .. ntaps)` against
 * the reversed taps, volk_32fc_32f_dot_prod_32fc_generic */
static sdrk_gr_complex fir_filter(const sdrk_gr_complex* input, const float* rev_taps,
                                  unsigned ntaps)
{
    float re = 0.0f, im = 0.0f;
    sdrk_gr_complex r;
    for (unsigned j = 0; j < ntaps; j++) {
        re += input[j].re * rev_taps[j];
        im += input[j].im * rev_taps[j];
    }
    r.re = re;
    r.im = im;
    return r;
}

/* fir_filter_ccf::filterN(output, input, n): `input` holds `n + ntaps - 1` samples (the history
 * `ntaps - 1` first), `taps` the taps in their natural order (reversed here as `set_taps`
 * does, into `rev_taps` of `ntaps` floats) */
void sdrk_gr_fir_ccf_set_taps(float* rev_taps, const float* taps, unsigned ntaps)
{
    for (unsigned k = 0; k < ntaps; k++)
        rev_taps[k] = taps[ntaps - 1 - k];
}

void sdrk_gr_fir_ccf_work(sdrk_gr_complex* output, const sdrk_gr_complex* input,
                          const float* rev_taps, unsigned ntaps, unsigned n)
{
    for (unsigned i = 0; i < n; i++)
        output[i] = fir_filter(&input[i], rev_taps, ntaps);
}

/* constellation_decoder_cb::general_work with constellation_qpsk (dimensionality 1) */
void sdrk_gr_constellation_decoder_qpsk(unsigned char* out, const sdrk_gr_complex* in,
                                        unsigned n)
{
    for (unsigned i = 0; i < n; i++)
        out[i] = (unsigned char)(2 * (in[i].im > 0) + (in[i].re > 0));
}
