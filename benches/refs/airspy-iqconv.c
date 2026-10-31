/*
 * The real-to-complex converters of libairspy v1.0.10 (MIT, the last release under the MIT
 * licence): `iqconverter_float` and `iqconverter_int16`, the reference of `real_to_complex`.
 * The converters turn the real samples of the Airspy's ADC, sampled at fs, into complex
 * samples at fs / 2: a shift by fs / 4, a 47-tap halfband filter (the real part through the
 * filter, the imaginary part through the delay line, the centre tap 0.5 being a shift or a
 * multiplication) and the decimation by two, in place on interleaved I/Q.
 *
 * The file includes the pinned release's `iqconverter_float.c` or `iqconverter_int16.c`
 * (benches/refs/build.sh puts libairspy/src on the include path), so that the code that
 * runs is libairspy's own:
 *
 *   SDRK_AIRSPY_KIND 1  iqconverter_float, the generic code (no SSE2 intrinsics)
 *   SDRK_AIRSPY_KIND 2  iqconverter_float with USE_SSE2, which libairspy enables for FreeBSD
 *                       only: the file is compiled with __FreeBSD__ (the SSE2 branch of
 *                       translate_fs_4 and the intrinsics of the generic FIR loop; the 47-tap
 *                       kernel takes the unrolled 24-tap loop, which has no intrinsics)
 *   SDRK_AIRSPY_KIND 3  iqconverter_int16 (a plain C loop that the compiler vectorizes for the
 *                       level it is built for)
 *
 * What differs from libairspy's own use: the kernel is SDRk's (`create` takes the 47 taps, as
 * `airspy_set_hb_kernel`... does, libairspy's own `HB_KERNEL_*` are not used), the conversion
 * runs without `remove_dc` (libairspy's `iqconverter_*_process` is remove_dc followed by the
 * conversion; the conversion alone is the work of `real_to_complex`), and the conversion is
 * given an input and an output buffer (libairspy converts in place: the input is copied to the
 * output buffer first, a memcpy of `len` samples).
 *
 * Copyright (C) 2014, Youssef Touil <youssef@airspy.com>
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
#ifndef SDRK_AIRSPY_KIND
#error "SDRK_AIRSPY_KIND must be 1 (float, generic), 2 (float, SSE2) or 3 (int16)"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if SDRK_AIRSPY_KIND == 2
#include <immintrin.h>
#endif
#ifdef __wasm__
/* wasi-libc declares no memalign, which libairspy's GNU branch uses for its aligned buffers. */
static void* sdrk_memalign(size_t alignment, size_t size)
{
    void* p = NULL;
    return posix_memalign(&p, alignment, size) == 0 ? p : NULL;
}
#define memalign sdrk_memalign
#endif

#if SDRK_AIRSPY_KIND == 1 || SDRK_AIRSPY_KIND == 2

#if SDRK_AIRSPY_KIND == 2
#define SDRK_VARIANT sse2
#else
#define SDRK_VARIANT generic
#endif
#define SDRK_CAT2(a, b, c) a##b##c
#define SDRK_CAT(a, b, c) SDRK_CAT2(a, b, c)
/* Both variants may be linked into one binary: the symbols of libairspy's file get the
 * variant in their names. */
#define iqconverter_float_create SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _float_create)
#define iqconverter_float_free SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _float_free)
#define iqconverter_float_reset SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _float_reset)
#define iqconverter_float_process SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _float_process)
#if SDRK_AIRSPY_KIND == 2
/* The FreeBSD branch of the file defines a function named _aligned_malloc. The macro is set
 * after the system headers (glibc's headers do not accept it). */
#define _aligned_malloc SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _aligned_malloc)
#define __FreeBSD__ 1
#endif

#include "iqconverter_float.c"

#if SDRK_AIRSPY_KIND == 2
#undef __FreeBSD__
#endif

/* `taps`: the 47 taps of the halfband filter (libairspy's `hb_kernel`). */
void* SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _float_new)(const float* taps, int len)
{
    return iqconverter_float_create(taps, len);
}

void SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _float_delete)(void* cnv) { iqconverter_float_free(cnv); }
void SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _float_restart)(void* cnv) { iqconverter_float_reset(cnv); }

/* `len` real samples in, `len` floats out (`len / 2` complex samples, interleaved): the
 * conversion of `iqconverter_float_process` without `remove_dc`. */
void SDRK_CAT(sdrk_airspy_, SDRK_VARIANT, _float_convert)(void* cnv, const float* in, float* out, int len)
{
    memcpy(out, in, (size_t) len * sizeof(float));
    translate_fs_4((iqconverter_float_t*) cnv, out, len);
}

#else /* int16 */

#include "iqconverter_int16.c"

/* `iqconverter_int16_reset` of v1.0.10 clears only half of the delay line (`len * sizeof(int16_t)
 * / 4` bytes for `len / 2` samples of 2 bytes) and half of the FIR queue (`sizeof(int16_t)` for
 * entries of `int32_t`), so the first outputs of a fresh converter hold whatever the allocator
 * returned. The wrapper clears both completely after the reset; the rest is libairspy's. */
static void clear_all(iqconverter_int16_t* cnv)
{
    memset(cnv->delay_line, 0, (size_t) (cnv->len >> 1) * sizeof(int16_t));
    memset(cnv->fir_queue, 0, (size_t) cnv->len * sizeof(int32_t) * SIZE_FACTOR);
}

void* sdrk_airspy_int16_new(const int16_t* taps, int len)
{
    iqconverter_int16_t* cnv = iqconverter_int16_create(taps, len);
    clear_all(cnv);
    return cnv;
}

void sdrk_airspy_int16_delete(void* cnv) { iqconverter_int16_free(cnv); }

void sdrk_airspy_int16_restart(void* cnv)
{
    iqconverter_int16_reset(cnv);
    clear_all(cnv);
}

/* `len` real samples in, `len` int16 out (`len / 2` complex samples, interleaved). */
void sdrk_airspy_int16_convert(void* cnv, const int16_t* in, int16_t* out, int len)
{
    memcpy(out, in, (size_t) len * sizeof(int16_t));
    translate_fs_4((iqconverter_int16_t*) cnv, out, len);
}

#endif
