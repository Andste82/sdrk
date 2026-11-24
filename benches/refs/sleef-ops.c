/*
 * SLEEF 3.9.0's vectorized single-precision functions (BSL-1.0) behind a C interface for the
 * cycle measurements of `cargo measure` (benches/cref-sys), the SIMD references of the kernels
 * that no other SIMD library computes in their accuracy class (D2903). benches/refs/build.sh
 * builds SLEEF once per SIMD level with CMake (the flags of the level, libsleef-<level>.a) and
 * compiles this file with the flags of the same level (libsleefops-<level>.a); the level selects
 * SLEEF's vector extension of the level:
 *
 *   sse2    Sleef_*f4_u10sse2       sse4.2  Sleef_*f4_u10sse4
 *   avx2    Sleef_*f8_u10avx2       avx512  Sleef_*f16_u10avx512f
 *   neon    Sleef_*f4_u10advsimd
 *
 * - `sdrk_sleef_power_db`: `out = 10 Sleef_log10f(in) + offset` (the work of `power_db`).
 * - `sdrk_sleef_atan2_32fc`: `out = Sleef_atan2f(im, re) * gain` of interleaved complex floats
 *   (the second half of `quad_demod`, after the conjugate product of another library).
 * - `sdrk_sleef_oscillate_32fc`: `out[i] = cos(a) + j sin(a)` with `Sleef_sincosf` of the angle
 *   `a = (int32_t) p * 2 pi / 2^32` of the phase `p = phase + i * increment` of a `uint32_t`
 *   counter (the work of `oscillate` `C32`).
 *
 * The code around the SLEEF calls (loads, the split of the interleaved input, the counter, the
 * interleaved stores) is written with GCC's vector extensions, so that it is the same plain
 * vector code at every level; a tail shorter than one vector runs through a zero-padded vector.
 *
 * SPDX-License-Identifier: BSL-1.0 (SLEEF); this file: the licence of SDRk.
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <sleef.h>

#if defined(__AVX512F__)
#define W 16
#define SLEEF_LEVEL "avx512f"
#define VLOG10(x) Sleef_log10f16_u10avx512f(x)
#define VATAN2(y, x) Sleef_atan2f16_u10avx512f(y, x)
#define VSINCOS(x) Sleef_sincosf16_u10avx512f(x)
typedef __m512 sleef_vf;
typedef Sleef___m512_2 sleef_vf2;
#elif defined(__AVX2__)
#define W 8
#define SLEEF_LEVEL "avx2"
#define VLOG10(x) Sleef_log10f8_u10avx2(x)
#define VATAN2(y, x) Sleef_atan2f8_u10avx2(y, x)
#define VSINCOS(x) Sleef_sincosf8_u10avx2(x)
typedef __m256 sleef_vf;
typedef Sleef___m256_2 sleef_vf2;
#elif defined(__SSE4_1__)
#define W 4
#define SLEEF_LEVEL "sse4"
#define VLOG10(x) Sleef_log10f4_u10sse4(x)
#define VATAN2(y, x) Sleef_atan2f4_u10sse4(y, x)
#define VSINCOS(x) Sleef_sincosf4_u10sse4(x)
typedef __m128 sleef_vf;
typedef Sleef___m128_2 sleef_vf2;
#elif defined(__SSE2__)
#define W 4
#define SLEEF_LEVEL "sse2"
#define VLOG10(x) Sleef_log10f4_u10sse2(x)
#define VATAN2(y, x) Sleef_atan2f4_u10sse2(y, x)
#define VSINCOS(x) Sleef_sincosf4_u10sse2(x)
typedef __m128 sleef_vf;
typedef Sleef___m128_2 sleef_vf2;
#elif defined(__aarch64__)
#define W 4
#define SLEEF_LEVEL "advsimd"
#define VLOG10(x) Sleef_log10f4_u10advsimd(x)
#define VATAN2(y, x) Sleef_atan2f4_u10advsimd(y, x)
#define VSINCOS(x) Sleef_sincosf4_u10advsimd(x)
typedef float32x4_t sleef_vf;
typedef Sleef_float32x4_t_2 sleef_vf2;
#else
#error "no SLEEF vector extension for this level"
#endif

typedef float vf __attribute__((vector_size(W * 4)));
typedef int32_t vi __attribute__((vector_size(W * 4)));
typedef uint32_t vu __attribute__((vector_size(W * 4)));

#if W == 4
#define EVEN 0, 2, 4, 6
#define ODD 1, 3, 5, 7
#define LO 0, 4, 1, 5
#define HI 2, 6, 3, 7
#define LANES 0, 1, 2, 3
#elif W == 8
#define EVEN 0, 2, 4, 6, 8, 10, 12, 14
#define ODD 1, 3, 5, 7, 9, 11, 13, 15
#define LO 0, 8, 1, 9, 2, 10, 3, 11
#define HI 4, 12, 5, 13, 6, 14, 7, 15
#define LANES 0, 1, 2, 3, 4, 5, 6, 7
#else
#define EVEN 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
#define ODD 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
#define LO 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
#define HI 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
#define LANES 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#endif

static inline vf load(const float* p)
{
    vf v;
    memcpy(&v, p, sizeof v);
    return v;
}

static inline void store(float* p, vf v) { memcpy(p, &v, sizeof v); }

static inline vf from_sleef(sleef_vf x)
{
    vf v;
    memcpy(&v, &x, sizeof v);
    return v;
}

static inline sleef_vf to_sleef(vf x)
{
    sleef_vf v;
    memcpy(&v, &x, sizeof v);
    return v;
}

/* The SLEEF vector extension of this build. */
const char* sdrk_sleef_level(void) { return SLEEF_LEVEL; }

/* `out[i] = 10 log10(in[i]) + offset` for `n` floats. */
void sdrk_sleef_power_db(const float* in, float* out, size_t n, float offset)
{
    size_t i = 0;
    for (; i + W <= n; i += W)
        store(out + i, from_sleef(VLOG10(to_sleef(load(in + i)))) * 10.0f + offset);
    if (i < n) {
        float t[W] = { 0 };
        memcpy(t, in + i, (n - i) * sizeof(float));
        store(t, from_sleef(VLOG10(to_sleef(load(t)))) * 10.0f + offset);
        memcpy(out + i, t, (n - i) * sizeof(float));
    }
}

/* `out[i] = atan2(c[i].im, c[i].re) * gain` for `n` interleaved complex floats `c`. */
void sdrk_sleef_atan2_32fc(const float* c, float* out, size_t n, float gain)
{
    size_t i = 0;
    for (; i + W <= n; i += W) {
        const vf a = load(c + 2 * i), b = load(c + 2 * i + W);
        const vf re = __builtin_shufflevector(a, b, EVEN);
        const vf im = __builtin_shufflevector(a, b, ODD);
        store(out + i, from_sleef(VATAN2(to_sleef(im), to_sleef(re))) * gain);
    }
    if (i < n) {
        float t[2 * W] = { 0 }, o[W];
        memcpy(t, c + 2 * i, 2 * (n - i) * sizeof(float));
        const vf a = load(t), b = load(t + W);
        const vf re = __builtin_shufflevector(a, b, EVEN);
        const vf im = __builtin_shufflevector(a, b, ODD);
        store(o, from_sleef(VATAN2(to_sleef(im), to_sleef(re))) * gain);
        memcpy(out + i, o, (n - i) * sizeof(float));
    }
}

/* `out[i] = (cos a, sin a)`, `a = (int32_t) (phase + i increment) * 2 pi / 2^32`, for `n`
 * interleaved complex floats; returns the phase after the last sample. */
uint32_t sdrk_sleef_oscillate_32fc(float* out, size_t n, uint32_t phase, uint32_t increment)
{
    const vu lanes = { LANES };
    vu p = phase + lanes * increment;
    const uint32_t step = (uint32_t) W * increment;
    const float scale = (float) (6.283185307179586476925286766559 / 4294967296.0);
    size_t i = 0;
    for (; i < n; i += W) {
        const vf a = __builtin_convertvector((vi) p, vf) * scale;
        const sleef_vf2 sc = VSINCOS(to_sleef(a));
        const vf s = from_sleef(sc.x), co = from_sleef(sc.y);
        const vf lo = __builtin_shufflevector(co, s, LO);
        const vf hi = __builtin_shufflevector(co, s, HI);
        if (i + W <= n) {
            store(out + 2 * i, lo);
            store(out + 2 * i + W, hi);
        } else {
            float t[2 * W];
            store(t, lo);
            store(t + W, hi);
            memcpy(out + 2 * i, t, 2 * (n - i) * sizeof(float));
        }
        p += step;
    }
    return phase + (uint32_t) n * increment;
}
