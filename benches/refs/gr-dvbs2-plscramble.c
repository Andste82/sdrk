/*
 * The per-symbol PL scrambling switch of GNU Radio's `dvbs2_physical_cc`
 * (gr-dtv/lib/dvbs2/dvbs2_physical_cc_impl.cc, `general_work`, the payload loop of the
 * `VLSNR_OFF` branch, GNU Radio main at ec1bf15) translated from C++ to C:
 *
 *     tempin = in[consumed++];
 *     switch (m_cscram[n++]) {
 *     case 0: tempout = tempin; break;
 *     case 1: tempout = gr_complex(-tempin.imag(), tempin.real()); break;
 *     case 2: tempout = -tempin; break;
 *     case 3: tempout = gr_complex(tempin.imag(), -tempin.real()); break;
 *     }
 *     out[produced++] = tempout;
 *
 * `gr_complex` (`std::complex<float>`) is a struct of two floats here, `m_cscram` the `int`
 * array of the sequence `R_n` in 0..3. Left out: the PL header and the pilot blocks, the zero
 * sample that the block writes after every symbol (its output runs at twice the symbol rate),
 * the slot counting and the generation of `R_n` from the two m-sequences (`build_symbols`, once
 * per object in GNU Radio; here the caller passes the sequence).
 *
 * The information reference of SDRk's `scramble` at `bits = 2` (M49 §5), marked "reference does
 * different work": it scrambles by `j^{R_n}` with a switch per symbol on a precomputed `int`
 * sequence of a fixed frame layout, where the kernel takes any `bits`-bit phase byte per sample
 * and maps the quadrant without branches. The row shows a switch-based quadrant loop.
 *
 * Copyright 2015,2016,2018 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */

typedef struct {
    float re;
    float im;
} gr_complex;

/* `out[i] = in[i] * j^{rn[i]}` for `n` symbols, `rn[i]` in 0..3, by the switch of
 * `dvbs2_physical_cc_impl::general_work`. */
void sdrk_gr_dvbs2_plscramble(const gr_complex* in, gr_complex* out, const int* m_cscram, int n)
{
    int consumed = 0;
    int produced = 0;
    gr_complex tempin, tempout;

    tempout.re = 0.0f;
    tempout.im = 0.0f;
    for (int k = 0; k < n; k++) {
        tempin = in[consumed++];
        switch (m_cscram[k]) {
        case 0:
            tempout = tempin;
            break;
        case 1:
            tempout.re = -tempin.im;
            tempout.im = tempin.re;
            break;
        case 2:
            tempout.re = -tempin.re;
            tempout.im = -tempin.im;
            break;
        case 3:
            tempout.re = tempin.im;
            tempout.im = -tempin.re;
            break;
        }
        out[produced++] = tempout;
    }
}
