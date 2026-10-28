/*
 * The biquad cascades of CMSIS-DSP (generic C, built with __GNUC_PYTHON__ and
 * ARM_MATH_LOOPUNROLL, the defaults of its own CMake build for a host), behind owning handles
 * for the instruction counts of `cargo xtask count` (benches/cref-sys). benches/refs/build.sh
 * compiles this file and the CMSIS-DSP sources it calls once per SIMD level with the flags of
 * that level, into the library of that level (libcmsisbq-<level>.a). For the Cortex-M4 report
 * the same file is built with CMSIS-Core instead of __GNUC_PYTHON__ (the DSP extension and the
 * FPU of the M4F), with malloc from cmsis-m4-heap.c, and called from
 * targets/qemu-cortex-m4/src/bin/count.rs.
 *
 * The coefficients are given in SDRk's order and sign, `[b0, b1, b2, a1, a2]` per section with
 * `y = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2`; CMSIS-DSP adds the feedback terms, so the handles
 * store `-a1` and `-a2`.
 *
 * - df2t_f32: arm_biquad_cascade_df2T_f32, one real channel.
 * - stereo_df2t_f32: arm_biquad_cascade_stereo_df2T_f32, two interleaved channels (the re/im
 *   layout of an SDRk C32 buffer).
 * - df1_q15: arm_biquad_cascade_df1_q15 with postShift = 1, so that the raw Q15 coefficients
 *   are SDRk's Q2.14 values; a 64-bit accumulator, truncated by 14 bits and saturated per
 *   section. A coefficient a = -2.0 (raw -32768) cannot be negated: the constructor then
 *   returns null.
 *
 * Every handle is one malloc block: the instance, the coefficients and the state.
 */
#include <stdint.h>
#include <stdlib.h>

#include "arm_math.h"

typedef struct
{
    arm_biquad_cascade_df2T_instance_f32 instance;
    float32_t* coefs;
    float32_t* state;
} df2t_f32;

typedef struct
{
    arm_biquad_cascade_stereo_df2T_instance_f32 instance;
    float32_t* coefs;
    float32_t* state;
} stereo_df2t_f32;

typedef struct
{
    arm_biquad_casd_df1_inst_q15 instance;
    q15_t* coefs;
    q15_t* state;
} df1_q15;

/* The section count must fit CMSIS-DSP's uint8_t. */
static int stages_ok(uint32_t stages)
{
    return stages >= 1u && stages <= 255u;
}

/* CMSIS order and sign of one f32 section: b0, b1, b2, -a1, -a2. */
static void f32_section(float32_t* to, const float* from)
{
    to[0] = from[0];
    to[1] = from[1];
    to[2] = from[2];
    to[3] = -from[3];
    to[4] = -from[4];
}

void* sdrk_cmsis_df2t_f32_new(const float* coefs, uint32_t stages)
{
    if (!stages_ok(stages))
        return NULL;
    df2t_f32* h = malloc(sizeof(df2t_f32) + sizeof(float32_t) * 7u * stages);
    if (h == NULL)
        return NULL;
    h->coefs = (float32_t*)(h + 1);
    h->state = h->coefs + 5u * stages;
    for (uint32_t k = 0; k < stages; k++)
        f32_section(h->coefs + 5u * k, coefs + 5u * k);
    arm_biquad_cascade_df2T_init_f32(&h->instance, (uint8_t)stages, h->coefs, h->state);
    return h;
}

void sdrk_cmsis_df2t_f32_run(void* handle, const float* input, float* output, uint32_t n)
{
    df2t_f32* h = handle;
    arm_biquad_cascade_df2T_f32(&h->instance, input, output, n);
}

void* sdrk_cmsis_stereo_df2t_f32_new(const float* coefs, uint32_t stages)
{
    if (!stages_ok(stages))
        return NULL;
    stereo_df2t_f32* h = malloc(sizeof(stereo_df2t_f32) + sizeof(float32_t) * 9u * stages);
    if (h == NULL)
        return NULL;
    h->coefs = (float32_t*)(h + 1);
    h->state = h->coefs + 5u * stages;
    for (uint32_t k = 0; k < stages; k++)
        f32_section(h->coefs + 5u * k, coefs + 5u * k);
    arm_biquad_cascade_stereo_df2T_init_f32(&h->instance, (uint8_t)stages, h->coefs, h->state);
    return h;
}

/* `n` frames of two interleaved channels. */
void sdrk_cmsis_stereo_df2t_f32_run(void* handle, const float* input, float* output, uint32_t n)
{
    stereo_df2t_f32* h = handle;
    arm_biquad_cascade_stereo_df2T_f32(&h->instance, input, output, n);
}

void* sdrk_cmsis_df1_q15_new(const int16_t* coefs, uint32_t stages)
{
    if (!stages_ok(stages))
        return NULL;
    for (uint32_t k = 0; k < stages; k++)
    {
        if (coefs[5u * k + 3u] == INT16_MIN || coefs[5u * k + 4u] == INT16_MIN)
            return NULL;
    }
    df1_q15* h = malloc(sizeof(df1_q15) + sizeof(q15_t) * 10u * stages);
    if (h == NULL)
        return NULL;
    h->coefs = (q15_t*)(h + 1);
    h->state = h->coefs + 6u * stages;
    for (uint32_t k = 0; k < stages; k++)
    {
        const int16_t* from = coefs + 5u * k;
        q15_t* to = h->coefs + 6u * k;
        /* b0, 0, b1, b2, -a1, -a2: the zero is the padding of the SIMD variants. */
        to[0] = from[0];
        to[1] = 0;
        to[2] = from[1];
        to[3] = from[2];
        to[4] = (q15_t)-from[3];
        to[5] = (q15_t)-from[4];
    }
    arm_biquad_cascade_df1_init_q15(&h->instance, (uint8_t)stages, h->coefs, h->state, 1);
    return h;
}

void sdrk_cmsis_df1_q15_run(void* handle, const int16_t* input, int16_t* output, uint32_t n)
{
    df1_q15* h = handle;
    arm_biquad_cascade_df1_q15(&h->instance, input, output, n);
}

/* Frees a handle of any of the constructors above. */
void sdrk_cmsis_biquad_free(void* handle)
{
    free(handle);
}
