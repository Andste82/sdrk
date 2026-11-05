/*
 * The amplitude modulator of liquid-dsp v1.8.3, `ampmodem` of src/modem/src/ampmodem.c, with
 * the code that `ampmodem_modulate_block` runs per sample: `ampmodem_modulate` for
 * `LIQUID_AMPMODEM_DSB` with the carrier (`suppressed_carrier` 0), `y = x * mod_index + 1` as a
 * complex value. Left out: error logging, printing, the demodulators and the objects that only
 * they and the single-sideband modulator use (`nco_crcf`, `firfilt_rrrf`, `firhilbf`,
 * `firfilt_crcf`, `wdelaycf`), so `_create` accepts `LIQUID_AMPMODEM_DSB` only; the `_type`
 * test and the `_suppressed_carrier` select of `_modulate` stay as liquid runs them.
 *
 * The scalar reference of SDRk's `am_mod` `C32` fallback (D440): liquid's DSB modulator has the
 * carrier amplitude 1 and neither clamps its input nor scales by a carrier, SDRk's kernel
 * computes `carrier * (1 + depth * clamp(x, -1, 1))`; liquid does the same per-sample work
 * without the clamp and the carrier product.
 *
 * Copyright (c) 2007 - 2021 Joseph Gaeddert
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
#include <stdlib.h>

#define LIQUID_OK 0

typedef enum {
    LIQUID_AMPMODEM_DSB = 0, // double side-band
    LIQUID_AMPMODEM_USB,     // single side-band (upper)
    LIQUID_AMPMODEM_LSB      // single side-band (lower)
} liquid_ampmodem_type;

struct ampmodem_s {
    // modulation index
    float mod_index;

    // modulation type (e.g. LIQUID_AMPMODEM_DSB)
    liquid_ampmodem_type type;

    // suppressed carrier flag
    int suppressed_carrier;
};
typedef struct ampmodem_s* ampmodem;

// create ampmodem object (DSB only, see above)
//  _mod_index          :   modulation index
//  _suppressed_carrier :   carrier suppression flag
ampmodem sdrk_liquid_ampmodem_create_dsb(float _mod_index, int _suppressed_carrier)
{
    ampmodem q = (ampmodem)malloc(sizeof(struct ampmodem_s));
    if (q == NULL)
        return NULL;

    // set internal properties
    q->type = LIQUID_AMPMODEM_DSB;
    q->mod_index = _mod_index;
    q->suppressed_carrier = (_suppressed_carrier != 0);
    return q;
}

void sdrk_liquid_ampmodem_destroy(ampmodem _q)
{
    free(_q);
}

int sdrk_liquid_ampmodem_modulate(ampmodem _q, float _x, float complex* _y)
{
    float complex x_hat = 0.0f;

    if (_q->type == LIQUID_AMPMODEM_DSB) {
        x_hat = _x;
    } else {
        // the single-sideband branch (firhilbf_r2c_execute) is left out; _create gives DSB only
        return LIQUID_OK;
    }

    // save result
    *_y = x_hat * _q->mod_index + (_q->suppressed_carrier ? 0.0f : 1.0f);
    return LIQUID_OK;
}

// modulate block of samples
//  _q      :   ampmodem object
//  _m      :   message signal m(t), [size: _n x 1]
//  _n      :   number of input, output samples
//  _s      :   complex baseband signal s(t) [size: _n x 1]
int sdrk_liquid_ampmodem_modulate_block(ampmodem _q, float* _m, unsigned int _n, float complex* _s)
{
    // TODO: implement more efficient method
    unsigned int i;
    for (i = 0; i < _n; i++)
        sdrk_liquid_ampmodem_modulate(_q, _m[i], &_s[i]);
    return LIQUID_OK;
}
