/*
 * The work loop of GNU Radio's `fastnoise_source_f` and `fastnoise_source_c`
 * (gr-analog/lib/fastnoise_source_impl.{h,cc}, `fastnoise_source_impl<T>::work` with
 * `sample()`, and `xoroshiro128p_next`, `xoroshiro128p_jump`, `splitmix64_next` and
 * `xoroshiro128p_seed` of gnuradio-runtime/include/gnuradio/xoroshiro128p.h, GNU Radio main at
 * ec1bf15) translated from C++ to C, member for member:
 *
 * - the block draws one xoroshiro128+ value per output sample and copies the pool entry it
 *   selects: `d_samples[xoroshiro128p_next(d_state) & d_bitmask]` when the pool size is a power
 *   of two, `% d_samples.size()` otherwise (the test of `d_bitmask` stays in the loop, as in
 *   `sample()`);
 * - the pool `d_samples` is a `std::vector<T>` (here its begin and end pointers), the state
 *   `d_state[2]` and `d_bitmask` are members of the object;
 * - `fastnoise_source_c` divides its amplitude by `sqrtf(2.0f)` in the constructor, so a pool
 *   entry has the standard deviation `ampl / sqrt(2)` per component.
 *
 * Replaced: `gr::random::gasdev` and `gr::random::rayleigh_complex`, which fill the pool once in
 * the constructor (outside the counted calls of the measurement), by the polar Box-Muller method
 * over a second xoroshiro128+ state seeded with the same seed: Gaussian values with variance 1
 * per component, as those of `gr::random`. The values of the pool are therefore not GNU Radio's;
 * the loop that is measured is. Left out: the scoped lock of `d_setlock` (once per call), the
 * log messages, `set_type`, `set_amplitude`, the uniform, Laplacian and impulse pools and
 * `sample_unbiased` (not called by `work`).
 *
 * The information reference of SDRk's `gauss_noise` (M48 §5), marked "reference does less
 * work": no normal mapping per sample, the output repeats the values of the pool (8192 by
 * default in GNU Radio Companion), its tail ends at the largest value of the pool, and there is
 * no random access. The row shows the floor of what a noise sample can cost.
 *
 * Copyright 2013,2018 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 * (xoroshiro128+ by David Blackman and Sebastiano Vigna, CC0.)
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* gnuradio-runtime/include/gnuradio/xoroshiro128p.h */

static inline uint64_t rotl(const uint64_t x, const int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t xoroshiro128p_next(uint64_t* state)
{
    const uint64_t s0 = state[0];
    uint64_t s1 = state[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    state[1] = rotl(s1, 36);                   // c

    return result;
}

static inline void xoroshiro128p_jump(uint64_t* state)
{
    static const uint64_t JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); ++i) {
        for (unsigned int b = 0; b < 64; ++b) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= state[0];
                s1 ^= state[1];
            }
            xoroshiro128p_next(state);
        }
    }

    state[0] = s0;
    state[1] = s1;
}

static inline uint64_t splitmix64_next(uint64_t* state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

static inline void xoroshiro128p_seed(uint64_t* state, const uint64_t seed)
{
    state[0] = seed;
    state[1] = splitmix64_next(state);
    xoroshiro128p_jump(state);
}

/* The stand-in for gr::random::gasdev (pool generation only, see above). */

static double uniform01(uint64_t* state)
{
    return (double)(xoroshiro128p_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static double gasdev(uint64_t* state)
{
    double x, y, s;
    do {
        x = 2.0 * uniform01(state) - 1.0;
        y = 2.0 * uniform01(state) - 1.0;
        s = x * x + y * y;
    } while (s >= 1.0 || s == 0.0);
    return x * sqrt(-2.0 * log(s) / s);
}

/* fastnoise_source_impl<float> */

typedef struct {
    float d_ampl;
    float* d_samples_begin; /* std::vector<float> d_samples */
    float* d_samples_end;
    uint64_t d_state[2];
    size_t d_bitmask;
} sdrk_gr_fastnoise_source_f;

static int is_pwr_of_two(size_t value) { return value && !(value & (value - 1)); }

size_t sdrk_gr_fastnoise_source_f_size(void) { return sizeof(sdrk_gr_fastnoise_source_f); }

/* The constructor with the type GR_GAUSSIAN; `pool` holds `samples` values of the caller. */
void sdrk_gr_fastnoise_source_f_init(
    sdrk_gr_fastnoise_source_f* self, float ampl, uint64_t seed, float* pool, size_t samples)
{
    uint64_t rng[2];
    self->d_ampl = ampl;
    self->d_samples_begin = pool;
    self->d_samples_end = pool + samples;
    self->d_bitmask = is_pwr_of_two(samples) ? samples - 1 : 0;
    xoroshiro128p_seed(self->d_state, seed);
    xoroshiro128p_seed(rng, seed);
    for (size_t i = 0; i < samples; i++)
        pool[i] = (float)(self->d_ampl * gasdev(rng));
}

static inline float sample_f(sdrk_gr_fastnoise_source_f* self)
{
    size_t idx;
    if (self->d_bitmask) {
        idx = xoroshiro128p_next(self->d_state) & self->d_bitmask;
    } else {
        idx = xoroshiro128p_next(self->d_state) %
              (size_t)(self->d_samples_end - self->d_samples_begin);
    }
    return self->d_samples_begin[idx];
}

int sdrk_gr_fastnoise_source_f_work(
    sdrk_gr_fastnoise_source_f* self, int noutput_items, float* out)
{
    for (int i = 0; i < noutput_items; i++) {
        out[i] = sample_f(self);
    }

    return noutput_items;
}

/* fastnoise_source_impl<gr_complex> */

typedef struct {
    float d_ampl;
    float _Complex* d_samples_begin; /* std::vector<gr_complex> d_samples */
    float _Complex* d_samples_end;
    uint64_t d_state[2];
    size_t d_bitmask;
} sdrk_gr_fastnoise_source_c;

size_t sdrk_gr_fastnoise_source_c_size(void) { return sizeof(sdrk_gr_fastnoise_source_c); }

/* The constructor with the type GR_GAUSSIAN (`d_ampl = ampl / sqrtf(2.0f)`, the pool entries
 * `d_ampl * rayleigh_complex()`); `pool` holds `samples` values of the caller. */
void sdrk_gr_fastnoise_source_c_init(sdrk_gr_fastnoise_source_c* self,
                                     float ampl,
                                     uint64_t seed,
                                     float _Complex* pool,
                                     size_t samples)
{
    uint64_t rng[2];
    self->d_ampl = ampl / sqrtf(2.0f);
    self->d_samples_begin = pool;
    self->d_samples_end = pool + samples;
    self->d_bitmask = is_pwr_of_two(samples) ? samples - 1 : 0;
    xoroshiro128p_seed(self->d_state, seed);
    xoroshiro128p_seed(rng, seed);
    for (size_t i = 0; i < samples; i++) {
        float re = (float)gasdev(rng);
        float im = (float)gasdev(rng);
        pool[i] = __builtin_complex(self->d_ampl * re, self->d_ampl * im);
    }
}

static inline float _Complex sample_c(sdrk_gr_fastnoise_source_c* self)
{
    size_t idx;
    if (self->d_bitmask) {
        idx = xoroshiro128p_next(self->d_state) & self->d_bitmask;
    } else {
        idx = xoroshiro128p_next(self->d_state) %
              (size_t)(self->d_samples_end - self->d_samples_begin);
    }
    return self->d_samples_begin[idx];
}

int sdrk_gr_fastnoise_source_c_work(sdrk_gr_fastnoise_source_c* self,
                                    int noutput_items,
                                    float _Complex* out)
{
    for (int i = 0; i < noutput_items; i++) {
        out[i] = sample_c(self);
    }

    return noutput_items;
}
