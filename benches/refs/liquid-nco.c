/*
 * The numerically controlled oscillator of liquid-dsp v1.8.3 in its `LIQUID_VCO_INTERP` form,
 * `nco_crcf` (T = float) of src/nco/src/nco.proto.c, with the code the stereo pilot loop runs
 * per sample: the interpolated sine table of `nco_crcf_create`, `_fp_sin`, `_reset`,
 * `_set_frequency`, `_set_phase`, `_adjust_frequency`, `_adjust_phase`, `_step`, `_sincos`
 * (the `LIQUID_VCO_INTERP` branch), `_static_index` (the branch of that type) and `_constrain`
 * (the `#else` branch: the phase in radians is reduced by two `while` loops of float
 * subtractions, promoted to double, and converted to the 32-bit phase word).
 * Left out: the `LIQUID_NCO` and `LIQUID_VCO_DIRECT` types, error logging, the mixing
 * functions, copying, printing, and `nco_crcf_pll_step` with `_pll_set_bandwidth` (the block
 * function below applies the loop gains of SDRk's `stereo_pll` through `_adjust_frequency` and
 * `_adjust_phase`, the two calls that liquid's PLL step is made of).
 *
 * `sdrk_liquid_nco_pll_block` is the multiplying real phase detector of liquid's
 * examples/nco_pll_real_example.c around those calls, with the outputs of `stereo_pll`:
 * per sample `sin(theta)` and `cos(theta)` from the table, `sin(2 theta)` from the same table
 * (the phase word shifted left by one bit), `e = pilot * cos(theta)`, the output pair
 * `(mpx, mpx * sin(2 theta))`, then `adjust_frequency(ki e)`, `adjust_phase(kp e)` (radians)
 * and `step`. liquid has no lock detector and no integrator limit.
 *
 * Copyright (c) 2007 - 2024 Joseph Gaeddert
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
 *
 * SPDX-License-Identifier: MIT
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NCO_STATIC_LUT_WORDBITS 32
#define NCO_STATIC_LUT_NBITS 10
#define NCO_STATIC_LUT_SIZE (1LLU << NCO_STATIC_LUT_NBITS)
#define NCO_STATIC_LUT_HSIZE (NCO_STATIC_LUT_SIZE >> 1)
#define NCO_STATIC_LUT_QSIZE (NCO_STATIC_LUT_SIZE >> 2)
#define NCO_STATIC_LUT_INDEX_SHIFTED_PI2(index) (((index) + NCO_STATIC_LUT_QSIZE) & (NCO_STATIC_LUT_SIZE - 1))
#define NCO_STATIC_LUT_THETA_SHIFTED_PI2(theta) ((theta) + (1LLU << (NCO_STATIC_LUT_WORDBITS - 2)))
#define NCO_STATIC_LUT_THETA_ACCUM(theta) \
    ((uint32_t)(theta & ((1LLU << (NCO_STATIC_LUT_WORDBITS - NCO_STATIC_LUT_NBITS)) - 1)))

struct vco_tab_e_s {
    float value;
    float skew;
};
typedef struct vco_tab_e_s vco_tab_e;

struct nco_crcf_s {
    uint32_t theta;         // phase
    uint32_t d_theta;       // frequency (phase step)
    vco_tab_e* vcoi_sintab; // sine interpolated look-up table
};
typedef struct nco_crcf_s* nco_crcf;

static float nco_crcf_fp_sin(float x)
{
    return sinf(x * (float)(M_PI) / ((uint32_t)(INT32_MAX) + 1));
}

// constrain phase (or frequency) and convert to fixed-point
static uint32_t nco_crcf_constrain(float _theta)
{
    while (_theta >= 2 * M_PI)
        _theta -= 2 * M_PI;
    while (_theta < 0)
        _theta += 2 * M_PI;

    return (uint32_t)((_theta / (2 * M_PI)) * 0xffffffff);
}

static int nco_crcf_reset(nco_crcf _q)
{
    _q->theta = 0;
    _q->d_theta = 0;
    return 0;
}

static int nco_crcf_set_frequency(nco_crcf _q, float _dtheta)
{
    _q->d_theta = nco_crcf_constrain(_dtheta);
    return 0;
}

static int nco_crcf_set_phase(nco_crcf _q, float _phi)
{
    _q->theta = nco_crcf_constrain(_phi);
    return 0;
}

static int nco_crcf_adjust_frequency(nco_crcf _q, float _df)
{
    _q->d_theta += nco_crcf_constrain(_df);
    return 0;
}

static int nco_crcf_adjust_phase(nco_crcf _q, float _dphi)
{
    _q->theta += nco_crcf_constrain(_dphi);
    return 0;
}

static int nco_crcf_step(nco_crcf _q)
{
    _q->theta += _q->d_theta;
    return 0;
}

// create nco/vco object (type LIQUID_VCO_INTERP; the PLL bandwidth is not set)
nco_crcf sdrk_liquid_nco_crcf_create_vco_interp(void)
{
    nco_crcf q = (nco_crcf)malloc(sizeof(struct nco_crcf_s));
    if (q == NULL)
        return NULL;
    q->vcoi_sintab = (vco_tab_e*)malloc(NCO_STATIC_LUT_SIZE * sizeof(vco_tab_e));
    if (q->vcoi_sintab == NULL) {
        free(q);
        return NULL;
    }
    /* Calculate only [0, PI/2] range and mirror it to [PI, 3*PI/2] range
     * (in single forward iteration, i.e. "sine" direction)
     */
    uint32_t theta = 0;
    const int32_t d_theta = (uint32_t)(UINT32_MAX) / NCO_STATIC_LUT_SIZE;
    unsigned int i;
    for (i = 0; i < NCO_STATIC_LUT_QSIZE; i++) {
        float value = nco_crcf_fp_sin((uint32_t)(theta));
        float next_value = nco_crcf_fp_sin((uint32_t)(theta + d_theta));
        float skew = (next_value - value) / (float)(d_theta);
        q->vcoi_sintab[i].value = value;
        q->vcoi_sintab[i].skew = skew;
        q->vcoi_sintab[i + NCO_STATIC_LUT_HSIZE].value = -value;
        q->vcoi_sintab[i + NCO_STATIC_LUT_HSIZE].skew = -skew;
        theta += d_theta;
    }
    q->vcoi_sintab[NCO_STATIC_LUT_QSIZE].value = 1.0f;
    q->vcoi_sintab[NCO_STATIC_LUT_QSIZE].skew = -q->vcoi_sintab[NCO_STATIC_LUT_QSIZE - 1].skew;
    q->vcoi_sintab[NCO_STATIC_LUT_QSIZE + NCO_STATIC_LUT_HSIZE].value = -q->vcoi_sintab[NCO_STATIC_LUT_QSIZE].value;
    q->vcoi_sintab[NCO_STATIC_LUT_QSIZE + NCO_STATIC_LUT_HSIZE].skew = q->vcoi_sintab[NCO_STATIC_LUT_QSIZE - 1].skew;
    /* Mirror [0, PI/2], [PI, 3*PI/2] ranges to [PI/2, PI], [3*PI/2, 2*PI]
     * (in single backward iteration, i.e. "cosine" direction)
     */
    for (i = 1; i < NCO_STATIC_LUT_QSIZE; i++) {
        unsigned int i_pi2 = i + NCO_STATIC_LUT_QSIZE;
        float value = q->vcoi_sintab[NCO_STATIC_LUT_QSIZE - i].value;
        float skew = q->vcoi_sintab[NCO_STATIC_LUT_QSIZE - i - 1].skew;
        q->vcoi_sintab[i_pi2].value = value;
        q->vcoi_sintab[i_pi2].skew = -skew;
        q->vcoi_sintab[i_pi2 + NCO_STATIC_LUT_HSIZE].value = -value;
        q->vcoi_sintab[i_pi2 + NCO_STATIC_LUT_HSIZE].skew = skew;
    }

    nco_crcf_reset(q);
    return q;
}

// compute index for sine/cosine look-up table (the branch of type LIQUID_VCO_INTERP)
static unsigned int nco_crcf_static_index(nco_crcf _q)
{
    return (_q->theta >> (NCO_STATIC_LUT_WORDBITS - NCO_STATIC_LUT_NBITS)) & (NCO_STATIC_LUT_SIZE - 1);
}

// compute sin, cos of internal phase (the branch of type LIQUID_VCO_INTERP)
static int nco_crcf_sincos(nco_crcf _q, float* _s, float* _c)
{
    unsigned int index = nco_crcf_static_index(_q);
    unsigned int index_pi2 = NCO_STATIC_LUT_INDEX_SHIFTED_PI2(index);
    uint32_t theta_pi2 = NCO_STATIC_LUT_THETA_SHIFTED_PI2(_q->theta);
    *_s = _q->vcoi_sintab[index].value + NCO_STATIC_LUT_THETA_ACCUM(_q->theta) * _q->vcoi_sintab[index].skew;
    *_c = _q->vcoi_sintab[index_pi2].value + NCO_STATIC_LUT_THETA_ACCUM(theta_pi2) * _q->vcoi_sintab[index_pi2].skew;
    return 0;
}

// the same table interpolation for a phase word that is not the object's own (2 theta)
static void sdrk_liquid_vco_sincos(const vco_tab_e* tab, uint32_t theta, float* _s, float* _c)
{
    unsigned int index = (theta >> (NCO_STATIC_LUT_WORDBITS - NCO_STATIC_LUT_NBITS)) & (NCO_STATIC_LUT_SIZE - 1);
    unsigned int index_pi2 = NCO_STATIC_LUT_INDEX_SHIFTED_PI2(index);
    uint32_t theta_pi2 = NCO_STATIC_LUT_THETA_SHIFTED_PI2(theta);
    *_s = tab[index].value + NCO_STATIC_LUT_THETA_ACCUM(theta) * tab[index].skew;
    *_c = tab[index_pi2].value + NCO_STATIC_LUT_THETA_ACCUM(theta_pi2) * tab[index_pi2].skew;
}

void sdrk_liquid_nco_crcf_set_frequency(nco_crcf _q, float _dtheta) { nco_crcf_set_frequency(_q, _dtheta); }
void sdrk_liquid_nco_crcf_set_phase(nco_crcf _q, float _phi) { nco_crcf_set_phase(_q, _phi); }
uint32_t sdrk_liquid_nco_crcf_get_phase_word(const nco_crcf _q) { return _q->theta; }
uint32_t sdrk_liquid_nco_crcf_get_frequency_word(const nco_crcf _q) { return _q->d_theta; }

// the table (value, skew) pairs, for the tests: 1024 entries
const float* sdrk_liquid_nco_crcf_table(const nco_crcf _q) { return (const float*)_q->vcoi_sintab; }

// sin and cos of the object's phase
void sdrk_liquid_nco_crcf_sincos(nco_crcf _q, float* _s, float* _c) { nco_crcf_sincos(_q, _s, _c); }

void sdrk_liquid_nco_crcf_destroy(nco_crcf _q)
{
    if (_q == NULL)
        return;
    free(_q->vcoi_sintab);
    free(_q);
}

// The pilot loop of `stereo_pll` around the calls above: `_kp` and `_ki` are in radians per
// unit of the detector output. `_out` holds `_n` pairs (mpx, mpx * sin(2 theta)).
void sdrk_liquid_nco_pll_block(nco_crcf _q, const float* _mpx, const float* _pilot, float* _out, unsigned int _n,
    float _kp, float _ki)
{
    unsigned int i;
    for (i = 0; i < _n; i++) {
        float s, c, s2, c2;
        nco_crcf_sincos(_q, &s, &c);
        sdrk_liquid_vco_sincos(_q->vcoi_sintab, _q->theta << 1, &s2, &c2);
        (void)s;
        (void)c2;
        float e = _pilot[i] * c;
        _out[2 * i] = _mpx[i];
        _out[2 * i + 1] = _mpx[i] * s2;
        nco_crcf_adjust_frequency(_q, _ki * e);
        nco_crcf_adjust_phase(_q, _kp * e);
        nco_crcf_step(_q);
    }
}
