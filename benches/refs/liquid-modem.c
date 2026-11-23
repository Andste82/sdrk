/*
 * The `modemcf` of liquid-dsp v1.8.3 (MIT) as a symbol mapper, the reference of
 * `symbol_map_block` (issue #169): `modemcf_modulate`, the lookup `symbol -> point` of liquid's
 * arbitrary-constellation modem (`modemcf_create_arbitrary`), which takes the points in the
 * order of the caller.
 *
 * `modemcf_create_arbitrary` prints the table to stdout in liquid 1.8.3, which the measurement
 * binary cannot have (it writes JSON): the wrapper makes the object with the two calls of that
 * function, `_create_arb` and `_arb_init`, without the print.
 *
 * The file is a translation unit of liquid, as modemcf.c is: it defines the type macros of
 * liquid and includes the unpacked sources `modem_common.proto.c`, the modulation families and
 * `modem_arb.proto.c` of the pinned release (benches/refs/build.sh puts the release's
 * src/modem/src on the include path), so the code that runs per symbol is liquid's own; the
 * continuous-phase modems and the frequency modulator of modemcf.c are not included (the mapper
 * does not use them). The modulation types, the APSK and arbitrary-constellation tables and the
 * gray code are liquid's objects, compiled from the same release.
 *
 * What differs from liquid's own use:
 *
 *  - The constellation is the caller's (SDRk's points in its symbol order, so that the labelling
 *    and the accuracy of both sides are comparable). liquid's own QPSK, 16-QAM and 64-QAM
 *    modulators compute the points (`modem_modulate_qam` and the others) with a labelling of
 *    their own, which is not that of SDRk; the arbitrary-constellation modem is liquid's lookup
 *    mapper, the form that GNU Radio's `chunks_to_symbols_bc` has as well.
 *  - The block loop and the packing of the bits: `sdrk_liquid_modem_map_block` packs `k` bits
 *    (one per byte, 0 or 1, the first one the most significant) to a symbol and calls
 *    `modemcf_modulate` for every symbol; liquid has no packing call in its modem.
 *
 * Copyright (c) 2007 - 2026 Joseph Gaeddert
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "liquid.internal.h"

/* `liquid_nextpow2` of liquid's math library (src/math/src/math.c, the file is not linked: it
 * pulls in the special functions of liquid). `modemcf_create_arbitrary` calls it for the bits per
 * symbol, so the function is liquid's own code; weak, in case the real object is linked. */
__attribute__((weak)) unsigned int liquid_nextpow2(unsigned int _x)
{
    if (_x == 0) {
        liquid_error(LIQUID_EICONFIG, "liquid_nextpow2(), input must be greater than zero");
        return 0;
    }

    _x--;
    unsigned int n = 0;
    while (_x > 0) {
        _x >>= 1;
        n++;
    }
    return n;
}

#define MODEM(name) LIQUID_CONCAT(modemcf, name)
#define T float
#define TC float complex
#define PRINTVAL_T(X, F) PRINTVAL_FLOAT(X, F)
#define PRINTVAL_TC(X, F) PRINTVAL_CFLOAT(X, F)
#define EXTENSION "cf"

#include "modem_common.proto.c"

#include "modem_psk.proto.c"
#include "modem_dpsk.proto.c"
#include "modem_ask.proto.c"
#include "modem_qam.proto.c"
#include "modem_apsk.proto.c"

#include "modem_bpsk.proto.c"
#include "modem_qpsk.proto.c"
#include "modem_ook.proto.c"
#include "modem_pi4dqpsk.proto.c"
#include "modem_sqam32.proto.c"
#include "modem_sqam128.proto.c"

#include "modem_arb.proto.c"

/* The arbitrary-constellation modem of `m` points (`pts`, pairs, symbol order). */
void* sdrk_liquid_modem_create(const float* pts, unsigned int m)
{
    float complex* table = malloc((size_t)m * sizeof(float complex));
    if (table == NULL) {
        return NULL;
    }
    for (unsigned int i = 0; i < m; i++) {
        table[i] = pts[2 * i] + _Complex_I * pts[2 * i + 1];
    }
    /* `modemcf_create_arbitrary` of liquid 1.8.3 without its debug print of the table (it writes
     * to stdout unconditionally): the same calls, `_create_arb` and `_arb_init` */
    unsigned int bits = liquid_nextpow2(m);
    MODEM() q = NULL;
    if ((1U << bits) == m) {
        q = MODEM(_create_arb)(bits);
        MODEM(_arb_init)(q, table, m);
    }
    free(table);
    return q;
}

void sdrk_liquid_modem_destroy(void* q) { MODEM(_destroy)((MODEM()) q); }

/* `n` symbols of `k` bits each (one bit per byte, the first one the most significant). */
void sdrk_liquid_modem_map_block(void* q, const unsigned char* bits, unsigned int k, unsigned int n, float complex* y)
{
    for (unsigned int i = 0; i < n; i++) {
        unsigned int sym = 0;
        for (unsigned int j = 0; j < k; j++) {
            sym = (sym << 1) | (bits[(size_t)i * k + j] & 1u);
        }
        MODEM(_modulate)((MODEM()) q, sym, &y[i]);
    }
}
