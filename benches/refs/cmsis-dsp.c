/*
 * Handles and loops around CMSIS-DSP 1.18.0 functions (generic C, built with __GNUC_PYTHON__
 * and ARM_MATH_LOOPUNROLL like cmsis-biquad.c) for the scalar references of the fallback
 * round (benches/cref-sys, module `cmsis_dsp`). benches/refs/build.sh compiles this file and
 * the CMSIS-DSP sources once per SIMD level with the flags of that level into
 * libcmsisdsp-<level>.a. The vector functions of CMSIS-DSP (conversions, products, magnitudes,
 * arm_vlog_f32, ...) need no handle: cref-sys declares and calls them directly.
 *
 * - FIR filters (arm_fir_*, arm_fir_decimate_*, arm_fir_interpolate_*): one malloc block with
 *   the instance, the coefficients and the state. The taps are given in SDRk's order
 *   (`y[n] = sum h[k] x[n - k]`); CMSIS-DSP wants them time-reversed, which the constructors
 *   do. The state holds `numTaps + 2 * max_block` values, enough for every variant; a call
 *   must not pass more than `max_block` input samples. The Q15 FIR with loop unrolling needs
 *   an even tap count: an odd one gets a zero tap in front of the reversed taps (the padding
 *   CMSIS-DSP documents). A constructor returns null when CMSIS-DSP rejects the arguments.
 * - Transforms: the constant instances of arm_const_structs.c for 64, 1024 and 4096 points
 *   (null for other lengths) and an arm_rfft_fast_instance_f32 handle.
 * - Scalar functions (arm_sin_f32, arm_atan2_f32, ...): a loop over a buffer per function, so
 *   that a call covers a block as for the vector functions.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arm_math.h"
#include "arm_const_structs.h"

/* ---- FIR filters ---- */

typedef struct
{
    arm_fir_instance_f32 instance;
    uint32_t max_block;
} fir_f32;

typedef struct
{
    arm_fir_instance_q15 instance;
    uint32_t max_block;
} fir_q15;

typedef struct
{
    arm_fir_decimate_instance_f32 instance;
    uint32_t max_block;
} fir_decimate_f32;

typedef struct
{
    arm_fir_decimate_instance_q15 instance;
    uint32_t max_block;
} fir_decimate_q15;

typedef struct
{
    arm_fir_interpolate_instance_f32 instance;
    uint32_t max_block;
} fir_interpolate_f32;

typedef struct
{
    arm_fir_interpolate_instance_q15 instance;
    uint32_t max_block;
} fir_interpolate_q15;

/* CMSIS-DSP counts taps in uint16_t. */
static int taps_ok(uint32_t taps, uint32_t max_block)
{
    return taps >= 1u && taps <= 65534u && max_block >= 1u && max_block <= (1u << 24);
}

/* Copies `n` taps reversed to `to + pad` and zeroes the `pad` values before them. */
static void reverse_f32(float32_t* to, const float* from, uint32_t n, uint32_t pad)
{
    for (uint32_t k = 0; k < pad; k++)
        to[k] = 0.0f;
    for (uint32_t k = 0; k < n; k++)
        to[pad + k] = from[n - 1u - k];
}

static void reverse_q15(q15_t* to, const int16_t* from, uint32_t n, uint32_t pad)
{
    for (uint32_t k = 0; k < pad; k++)
        to[k] = 0;
    for (uint32_t k = 0; k < n; k++)
        to[pad + k] = from[n - 1u - k];
}

void* sdrk_cmsis_fir_f32_new(const float* taps, uint32_t n_taps, uint32_t max_block)
{
    if (!taps_ok(n_taps, max_block))
        return NULL;
    fir_f32* h = malloc(sizeof(fir_f32) + sizeof(float32_t) * (2u * n_taps + 2u * max_block));
    if (h == NULL)
        return NULL;
    float32_t* coefs = (float32_t*)(h + 1);
    float32_t* state = coefs + n_taps;
    reverse_f32(coefs, taps, n_taps, 0);
    arm_fir_init_f32(&h->instance, (uint16_t)n_taps, coefs, state, max_block);
    h->max_block = max_block;
    return h;
}

void sdrk_cmsis_fir_f32_run(void* handle, const float* input, float* output, uint32_t n)
{
    fir_f32* h = handle;
    arm_fir_f32(&h->instance, input, output, n);
}

void* sdrk_cmsis_fir_q15_new(const int16_t* taps, uint32_t n_taps, uint32_t max_block)
{
    if (!taps_ok(n_taps, max_block))
        return NULL;
    uint32_t pad = n_taps & 1u;
    uint32_t total = n_taps + pad;
    if (total < 4u)
    {
        pad += 4u - total;
        total = 4u;
    }
    fir_q15* h = malloc(sizeof(fir_q15) + sizeof(q15_t) * (2u * total + 2u * max_block));
    if (h == NULL)
        return NULL;
    q15_t* coefs = (q15_t*)(h + 1);
    q15_t* state = coefs + total;
    reverse_q15(coefs, taps, n_taps, pad);
    if (arm_fir_init_q15(&h->instance, (uint16_t)total, coefs, state, max_block) != ARM_MATH_SUCCESS)
    {
        free(h);
        return NULL;
    }
    h->max_block = max_block;
    return h;
}

void sdrk_cmsis_fir_q15_run(void* handle, const int16_t* input, int16_t* output, uint32_t n)
{
    fir_q15* h = handle;
    arm_fir_q15(&h->instance, input, output, n);
}

/* `max_block` input samples per call at most; it must be a multiple of the factor, and so must
   be the `n` of every call. */
void* sdrk_cmsis_fir_decimate_f32_new(const float* taps, uint32_t n_taps, uint32_t factor, uint32_t max_block)
{
    if (!taps_ok(n_taps, max_block) || factor < 1u || factor > 255u)
        return NULL;
    fir_decimate_f32* h = malloc(sizeof(fir_decimate_f32) + sizeof(float32_t) * (2u * n_taps + 2u * max_block));
    if (h == NULL)
        return NULL;
    float32_t* coefs = (float32_t*)(h + 1);
    float32_t* state = coefs + n_taps;
    reverse_f32(coefs, taps, n_taps, 0);
    if (arm_fir_decimate_init_f32(&h->instance, (uint16_t)n_taps, (uint8_t)factor, coefs, state, max_block) != ARM_MATH_SUCCESS)
    {
        free(h);
        return NULL;
    }
    h->max_block = max_block;
    return h;
}

/* `n` input samples, `n / factor` outputs. */
void sdrk_cmsis_fir_decimate_f32_run(void* handle, const float* input, float* output, uint32_t n)
{
    fir_decimate_f32* h = handle;
    arm_fir_decimate_f32(&h->instance, input, output, n);
}

void* sdrk_cmsis_fir_decimate_q15_new(const int16_t* taps, uint32_t n_taps, uint32_t factor, uint32_t max_block)
{
    if (!taps_ok(n_taps, max_block) || factor < 1u || factor > 255u)
        return NULL;
    fir_decimate_q15* h = malloc(sizeof(fir_decimate_q15) + sizeof(q15_t) * (2u * n_taps + 2u * max_block));
    if (h == NULL)
        return NULL;
    q15_t* coefs = (q15_t*)(h + 1);
    q15_t* state = coefs + n_taps;
    reverse_q15(coefs, taps, n_taps, 0);
    if (arm_fir_decimate_init_q15(&h->instance, (uint16_t)n_taps, (uint8_t)factor, coefs, state, max_block) != ARM_MATH_SUCCESS)
    {
        free(h);
        return NULL;
    }
    h->max_block = max_block;
    return h;
}

void sdrk_cmsis_fir_decimate_q15_run(void* handle, const int16_t* input, int16_t* output, uint32_t n)
{
    fir_decimate_q15* h = handle;
    arm_fir_decimate_q15(&h->instance, input, output, n);
}

/* The tap count must be a multiple of the factor. `n` input samples, `n * factor` outputs. */
void* sdrk_cmsis_fir_interpolate_f32_new(const float* taps, uint32_t n_taps, uint32_t factor, uint32_t max_block)
{
    if (!taps_ok(n_taps, max_block) || factor < 1u || factor > 255u)
        return NULL;
    fir_interpolate_f32* h = malloc(sizeof(fir_interpolate_f32) + sizeof(float32_t) * (2u * n_taps + 2u * max_block));
    if (h == NULL)
        return NULL;
    float32_t* coefs = (float32_t*)(h + 1);
    float32_t* state = coefs + n_taps;
    reverse_f32(coefs, taps, n_taps, 0);
    if (arm_fir_interpolate_init_f32(&h->instance, (uint8_t)factor, (uint16_t)n_taps, coefs, state, max_block) != ARM_MATH_SUCCESS)
    {
        free(h);
        return NULL;
    }
    h->max_block = max_block;
    return h;
}

void sdrk_cmsis_fir_interpolate_f32_run(void* handle, const float* input, float* output, uint32_t n)
{
    fir_interpolate_f32* h = handle;
    arm_fir_interpolate_f32(&h->instance, input, output, n);
}

void* sdrk_cmsis_fir_interpolate_q15_new(const int16_t* taps, uint32_t n_taps, uint32_t factor, uint32_t max_block)
{
    if (!taps_ok(n_taps, max_block) || factor < 1u || factor > 255u)
        return NULL;
    fir_interpolate_q15* h = malloc(sizeof(fir_interpolate_q15) + sizeof(q15_t) * (2u * n_taps + 2u * max_block));
    if (h == NULL)
        return NULL;
    q15_t* coefs = (q15_t*)(h + 1);
    q15_t* state = coefs + n_taps;
    reverse_q15(coefs, taps, n_taps, 0);
    if (arm_fir_interpolate_init_q15(&h->instance, (uint8_t)factor, (uint16_t)n_taps, coefs, state, max_block) != ARM_MATH_SUCCESS)
    {
        free(h);
        return NULL;
    }
    h->max_block = max_block;
    return h;
}

void sdrk_cmsis_fir_interpolate_q15_run(void* handle, const int16_t* input, int16_t* output, uint32_t n)
{
    fir_interpolate_q15* h = handle;
    arm_fir_interpolate_q15(&h->instance, input, output, n);
}

/* Frees a handle of any constructor of this file. */
void sdrk_cmsis_dsp_free(void* handle)
{
    free(handle);
}

/* ---- Transforms ---- */

/* The constant instance of arm_const_structs.c for `len` points, null for other lengths. */
const void* sdrk_cmsis_cfft_f32_instance(uint32_t len)
{
    switch (len)
    {
    case 64u:
        return &arm_cfft_sR_f32_len64;
    case 1024u:
        return &arm_cfft_sR_f32_len1024;
    case 4096u:
        return &arm_cfft_sR_f32_len4096;
    default:
        return NULL;
    }
}

const void* sdrk_cmsis_cfft_q15_instance(uint32_t len)
{
    switch (len)
    {
    case 64u:
        return &arm_cfft_sR_q15_len64;
    case 1024u:
        return &arm_cfft_sR_q15_len1024;
    case 4096u:
        return &arm_cfft_sR_q15_len4096;
    default:
        return NULL;
    }
}

/* arm_rfft_fast_init_f32 for `len` points (32 to 4096, a power of two); null otherwise. */
void* sdrk_cmsis_rfft_fast_f32_new(uint32_t len)
{
    if (len > 4096u)
        return NULL;
    arm_rfft_fast_instance_f32* h = malloc(sizeof(arm_rfft_fast_instance_f32));
    if (h == NULL)
        return NULL;
    if (arm_rfft_fast_init_f32(h, (uint16_t)len) != ARM_MATH_SUCCESS)
    {
        free(h);
        return NULL;
    }
    return h;
}

/* ---- Scalar functions over a buffer ---- */

/* `s[i] = sin(theta[i])`, `c[i] = cos(theta[i])` with theta in degrees (arm_sin_cos_f32). */
void sdrk_cmsis_sin_cos_f32_n(const float* theta, float* s, float* c, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        arm_sin_cos_f32(theta[i], &s[i], &c[i]);
}

/* arm_sin_cos_q31: theta in [-1, 1) for [-180, 180) degrees, results in Q31. */
void sdrk_cmsis_sin_cos_q31_n(const int32_t* theta, int32_t* s, int32_t* c, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        arm_sin_cos_q31(theta[i], &s[i], &c[i]);
}

/* arm_sin_f32 and arm_cos_f32: x in radians. */
void sdrk_cmsis_sin_f32_n(const float* x, float* y, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        y[i] = arm_sin_f32(x[i]);
}

void sdrk_cmsis_cos_f32_n(const float* x, float* y, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        y[i] = arm_cos_f32(x[i]);
}

/* arm_sin_q15 and arm_cos_q15: x in [0, 1) for [0, 2 pi), results in Q15. */
void sdrk_cmsis_sin_q15_n(const int16_t* x, int16_t* y, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        y[i] = arm_sin_q15(x[i]);
}

void sdrk_cmsis_cos_q15_n(const int16_t* x, int16_t* y, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        y[i] = arm_cos_q15(x[i]);
}

/* `out[i] = atan2(iq[2i + 1], iq[2i])` of interleaved complex samples (arm_atan2_f32,
   radians). */
void sdrk_cmsis_atan2_f32_n(const float* iq, float* out, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        arm_atan2_f32(iq[2u * i + 1u], iq[2u * i], &out[i]);
}

/* arm_atan2_q15 of interleaved Q15 samples: the angle in Q2.13 radians. */
void sdrk_cmsis_atan2_q15_n(const int16_t* iq, int16_t* out, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        arm_atan2_q15(iq[2u * i + 1u], iq[2u * i], &out[i]);
}
