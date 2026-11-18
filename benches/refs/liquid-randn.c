/*
 * The Gaussian generators of liquid-dsp v1.8.3: `randnf` and `crandnf` of
 * src/random/src/randn.c with the uniform generator `randf` of src/random/src/rand.c (the macro
 * `randf_inline` of include/liquid.internal.h, `(float) rand() / (float) RAND_MAX`). Left out:
 * the probability and distribution functions, `awgn`, `cawgn` and the other generators of the
 * module. Added: two plain loops, the ones a caller writes to fill a block, that scale each value
 * by `sigma` (`randnf` and each component of `crandnf` have variance 1, so `sigma` is the
 * standard deviation per real component, SDRk's convention).
 *
 * The information reference of SDRk's `gauss_noise` (M48 §5), marked "reference does different
 * work": liquid draws from libc `rand()` (glibc's `random_r` under a lock on x86_64 and aarch64,
 * wasi-libc's generator on wasm32) and evaluates Box-Muller in single precision (`sqrtf`,
 * `logf`, the sine branch with `sinf` for `randnf`, `cexpf` for `crandnf`), with a rejection
 * loop for `u1 == 0`, no random access and no result that is the same on every platform; SDRk's
 * kernel runs PCG32 with one draw per real value and the inverse normal CDF of a 1024-entry
 * table in integers, bit-exact on every target and seekable.
 *
 * Copyright (c) 2007 - 2020 Joseph Gaeddert
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
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// uniform random number generator (include/liquid.internal.h)
#define randf_inline() ((float) rand() / (float) RAND_MAX)

// uniform random number generator
static float randf() {
    return randf_inline();
}

// Gauss
static float randnf()
{
    // generate two uniform random numbers
    float u1, u2;

    // ensure u1 does not equal zero
    do {
        u1 = randf();
    } while (u1 == 0.0f);

    u2 = randf();

    return sqrtf(-2*logf(u1)) * sinf(2*M_PI*u2);
}

// Complex Gauss
static void crandnf(float complex * _y)
{
    // generate two uniform random numbers
    float u1, u2;

    // ensure u1 does not equal zero
    do {
        u1 = randf();
    } while (u1 == 0.0f);

    u2 = randf();

    *_y = sqrtf(-2*logf(u1)) * cexpf(_Complex_I*2*M_PI*u2);
}

// Seeds libc's generator (`srand`), so that a run is repeatable on one platform.
void sdrk_liquid_randn_seed(unsigned _seed)
{
    srand(_seed);
}

// Fills `_y` with `_n` values of `randnf() * _sigma`.
void sdrk_liquid_randnf_block(float * _y, unsigned int _n, float _sigma)
{
    unsigned int i;
    for (i = 0; i < _n; i++)
        _y[i] = randnf() * _sigma;
}

// Fills `_y` with `_n` values of `crandnf() * _sigma`.
void sdrk_liquid_crandnf_block(float complex * _y, unsigned int _n, float _sigma)
{
    unsigned int i;
    for (i = 0; i < _n; i++) {
        float complex v;
        crandnf(&v);
        _y[i] = v * _sigma;
    }
}
