// Element-wise operations of KFR (its expressions over `univector_ref`, which KFR evaluates with
// the vectors of the architecture it is compiled for), behind a C interface for the cycle
// measurements of `cargo measure` (benches/cref-sys). benches/refs/build.sh compiles this file
// once per SIMD level with the flags of that level (x86-64, aarch64 and wasm32 in the Emscripten
// configuration of kfr-biquad.cpp), into a library of its
// own (libkfrops-<level>.a). They are the SIMD references where VOLK's machine of the level runs
// a `generic` protokernel (`sse2`, the maintainer's decision on I13.1-11, D2785) or a protokernel
// of a lower level (`avx512`, D2900), where no other SIMD library has the operation (D2901), and
// at `simd128`, where VOLK's wasm32 build has only its `generic` protokernels (D12) and KFR runs
// in its Emscripten configuration, `sse41` over Emscripten's SSE headers written with SIMD128
// (the maintainer's decision of 2026-11-24 on item 9 of the optimisation round O1, D2850):
//
// - `sdrk_kfr_i8_to_f32`: `out = in * scale` from signed bytes, the work of
//   `volk_8i_s32f_convert_32f` with `1 / scale`.
// - `sdrk_kfr_i8_to_i16`: `out = in << 8` from signed bytes, the work of `volk_8i_convert_16i`.
// - `sdrk_kfr_complex_mul`: the complex product `out = a b`, the work of
//   `volk_32fc_x2_multiply_32fc`.
// - `sdrk_kfr_power_db`: `out = 10 log10(in) + offset`, the work of VOLK's `log2`, scale and add
//   chain of `power_db`.
// - `sdrk_kfr_add_f32`: `out = a + b` on floats (the complex sum on the `2 n` floats), the work
//   of `volk_32fc_x2_add_32fc`.
// - `sdrk_kfr_satadd_i16`: `out = satadd(a, b)` on `int16_t`, the work of
//   `volk_16i_x2_add_saturated_16i`.
// - `sdrk_kfr_f32_to_u8`: the `CU8` output of SDRk's contract, `out = clamp(round(x * scale +
//   bias), 0, 255)` in `double` with `x` the input and a NaN as 0 (a `select` on `in == in`),
//   the same work as SDRk's `convert_out` `CU8` (the NaN rule and the `f64` formula; O2 group 1,
//   D2955) and more than `volk_32f_s32f_x2_convert_8u` (NaN as 255, the `f32` formula).
//   KFR's `round` rounds ties to even from SSE4.1 (`roundpd`), ties away from zero at `sse2`
//   (its portable form) and on AArch64 (`frinta`): there an input on an exact multiple of
//   `1 / 128` (a tie of the formula) gives the byte above where the even one is below, the
//   rounding of the library; the work per value is the same.
// - `sdrk_kfr_deinterleave_i16`, `sdrk_kfr_interleave_i16`, `sdrk_kfr_interleave_f32`: `real` /
//   `imag` of `complex<int16_t>` and `make_complex` of `int16_t` and `float`, the work of
//   `volk_16ic_deinterleave_16i_x2` and `volk_32f_x2_interleave_32fc` (and of the 16-bit
//   interleave that VOLK lacks).
// - `sdrk_kfr_mag_i16`: `round(sqrt(re^2 + im^2))` in `float` of `complex<int16_t>`, saturated to
//   `int16_t`, the work of `volk_16ic_magnitude_16i`.
// - `sdrk_kfr_mag_sq`: `cabssqr`, the work of `volk_32fc_magnitude_squared_32f`.
// - `sdrk_kfr_quad_demod`: `carg(a cconj(b)) * gain` (KFR's `atan2`), the work of
//   `volk_32fc_x2_multiply_conjugate_32fc` and `volk_32fc_s32f_atan2_32f`.
// - `sdrk_kfr_multiply_conjugate`: `a cconj(b)`, the work of
//   `volk_32fc_x2_multiply_conjugate_32fc`.
// - `sdrk_kfr_scale_f32`: `out = in * scale`, the work of `volk_32f_s32f_multiply_32f`.
// - `sdrk_kfr_complex_mul_real`: complex samples times a real sequence, the work of
//   `volk_32fc_32f_multiply_32fc`.
// - `sdrk_kfr_dot_complex`: `dotproduct` of two complex vectors (no conjugate), the work of
//   `volk_32fc_x2_dot_prod_32fc`.
// - `sdrk_kfr_stereo_matrix_f32`, `sdrk_kfr_stereo_matrix_i16`: the sum/difference matrix
//   `l = s + 2 d`, `r = s - 2 d` of interleaved `(s, d)` into interleaved `(l, r)` as one
//   expression (`real`, `imag`, `+` / `-` and `make_complex`; on `int16_t` with `satadd` and
//   `satsub`: `2 d` saturated, then the saturated sum and difference).
// - `sdrk_kfr_i16_to_f32`: `out = in * scale` from `int16_t`, the work of
//   `volk_16i_s32f_convert_32f` with `1 / scale`.
// - `sdrk_kfr_mul_f32`: `out = a * b` on floats, the work of `volk_32f_x2_multiply_32f`.
// - `sdrk_kfr_dot_f32`, `sdrk_kfr_dot_complex_real`: `dotproduct` of two real vectors and of a
//   complex and a real vector, the work of `volk_32f_x2_dot_prod_32f` and
//   `volk_32fc_32f_dot_prod_32fc`.
// - `sdrk_kfr_mag`: `cabs`, the work of `volk_32fc_magnitude_32f`.
// - `sdrk_kfr_deinterleave_f32`: `real` / `imag` of `complex<float>` (one expression per output),
//   the work of `volk_32fc_deinterleave_32f_x2`.
//
// KFR's own sample conversion `kfr::convert` is not used: it has no 8-bit type, clamps every
// sample and converts one sample at a time (a scalar loop), while these expressions convert
// whole vectors. The file needs no C++ run-time library (no allocation, no exceptions).
#include <kfr/base.hpp>
#include <kfr/math.hpp>
#include <kfr/simd.hpp>

#include <cstddef>
#include <cstdint>

namespace
{
using c32  = kfr::complex<float>;
using ci16 = kfr::complex<std::int16_t>;

template <typename T>
kfr::univector_ref<T> uv(T* p, std::size_t n)
{
    return kfr::make_univector(p, n);
}
template <typename T>
kfr::univector_ref<const T> uv(const T* p, std::size_t n)
{
    return kfr::make_univector(p, n);
}
} // namespace

extern "C"
{
    void sdrk_kfr_i8_to_f32(const std::int8_t* in, float* out, std::size_t n, float scale)
    {
        uv(out, n) = kfr::cast<float>(uv(in, n)) * scale;
    }

    void sdrk_kfr_i8_to_i16(const std::int8_t* in, std::int16_t* out, std::size_t n)
    {
        uv(out, n) = kfr::cast<std::int16_t>(uv(in, n)) << 8;
    }

    void sdrk_kfr_complex_mul(const float* a, const float* b, float* out, std::size_t n)
    {
        uv(reinterpret_cast<c32*>(out), n) =
            uv(reinterpret_cast<const c32*>(a), n) * uv(reinterpret_cast<const c32*>(b), n);
    }

    void sdrk_kfr_power_db(const float* in, float* out, std::size_t n, float offset)
    {
        uv(out, n) = 10.0f * kfr::log10(uv(in, n)) + offset;
    }

    void sdrk_kfr_add_f32(const float* a, const float* b, float* out, std::size_t n)
    {
        uv(out, n) = uv(a, n) + uv(b, n);
    }

    void sdrk_kfr_satadd_i16(const std::int16_t* a, const std::int16_t* b, std::int16_t* out,
                             std::size_t n)
    {
        uv(out, n) = kfr::satadd(uv(a, n), uv(b, n));
    }

    void sdrk_kfr_f32_to_u8(const float* in, std::uint8_t* out, std::size_t n, float scale,
                            float bias)
    {
        const auto x = uv(in, n);
        // `x == x` is false for NaN only; KFR's `==` of expressions is behind
        // KFR_ENABLE_EXPR_CMP, so the comparison is built as the expression it would return.
        const auto ordered =
            kfr::expression_make_function<kfr::fn::equal, decltype(x), decltype(x)>{
                kfr::fn::equal(), decltype(x)(x), decltype(x)(x)
            };
        const auto a = kfr::cast<double>(kfr::select(ordered, x, 0.0f));
        uv(out, n)   = kfr::cast<std::uint8_t>(kfr::cast<std::int32_t>(
            kfr::clamp(kfr::round(a * double(scale) + double(bias)), 0.0, 255.0)));
    }

    void sdrk_kfr_deinterleave_i16(const std::int16_t* c, std::int16_t* re, std::int16_t* im,
                                   std::size_t n)
    {
        const auto x = uv(reinterpret_cast<const ci16*>(c), n);
        uv(re, n)    = kfr::real(x);
        uv(im, n)    = kfr::imag(x);
    }

    void sdrk_kfr_interleave_i16(const std::int16_t* re, const std::int16_t* im, std::int16_t* c,
                                 std::size_t n)
    {
        uv(reinterpret_cast<ci16*>(c), n) = kfr::make_complex(uv(re, n), uv(im, n));
    }

    void sdrk_kfr_interleave_f32(const float* re, const float* im, float* c, std::size_t n)
    {
        uv(reinterpret_cast<c32*>(c), n) = kfr::make_complex(uv(re, n), uv(im, n));
    }

    void sdrk_kfr_mag_i16(const std::int16_t* c, std::int16_t* out, std::size_t n)
    {
        const auto x  = uv(reinterpret_cast<const ci16*>(c), n);
        const auto re = kfr::cast<float>(kfr::real(x));
        const auto im = kfr::cast<float>(kfr::imag(x));
        uv(out, n)    = kfr::cast<std::int16_t>(kfr::cast<std::int32_t>(
            kfr::min(kfr::round(kfr::sqrt(re * re + im * im)), 32767.0f)));
    }

    void sdrk_kfr_mag_sq(const float* c, float* out, std::size_t n)
    {
        uv(out, n) = kfr::cabssqr(uv(reinterpret_cast<const c32*>(c), n));
    }

    void sdrk_kfr_quad_demod(const float* a, const float* b, float* out, std::size_t n, float gain)
    {
        uv(out, n) = kfr::carg(uv(reinterpret_cast<const c32*>(a), n) *
                               kfr::cconj(uv(reinterpret_cast<const c32*>(b), n))) *
                     gain;
    }

    void sdrk_kfr_multiply_conjugate(const float* a, const float* b, float* out, std::size_t n)
    {
        uv(reinterpret_cast<c32*>(out), n) =
            uv(reinterpret_cast<const c32*>(a), n) * kfr::cconj(uv(reinterpret_cast<const c32*>(b), n));
    }

    void sdrk_kfr_scale_f32(const float* in, float* out, std::size_t n, float scale)
    {
        uv(out, n) = uv(in, n) * scale;
    }

    void sdrk_kfr_complex_mul_real(const float* a, const float* b, float* out, std::size_t n)
    {
        uv(reinterpret_cast<c32*>(out), n) = uv(reinterpret_cast<const c32*>(a), n) * uv(b, n);
    }

    void sdrk_kfr_dot_complex(const float* a, const float* b, std::size_t n, float* out)
    {
        const c32 r = kfr::dotproduct(uv(reinterpret_cast<const c32*>(a), n),
                                      uv(reinterpret_cast<const c32*>(b), n));
        out[0]      = r.real();
        out[1]      = r.imag();
    }

    void sdrk_kfr_stereo_matrix_f32(const float* in, float* out, std::size_t n)
    {
        const auto x = uv(reinterpret_cast<const c32*>(in), n);
        const auto s = kfr::real(x);
        const auto d = kfr::imag(x);
        uv(reinterpret_cast<c32*>(out), n) = kfr::make_complex(s + (d + d), s - (d + d));
    }

    void sdrk_kfr_stereo_matrix_i16(const std::int16_t* in, std::int16_t* out, std::size_t n)
    {
        const auto x  = uv(reinterpret_cast<const ci16*>(in), n);
        const auto s  = kfr::real(x);
        const auto d2 = kfr::satadd(kfr::imag(x), kfr::imag(x));
        uv(reinterpret_cast<ci16*>(out), n) = kfr::make_complex(kfr::satadd(s, d2), kfr::satsub(s, d2));
    }

    void sdrk_kfr_i16_to_f32(const std::int16_t* in, float* out, std::size_t n, float scale)
    {
        uv(out, n) = kfr::cast<float>(uv(in, n)) * scale;
    }

    void sdrk_kfr_mul_f32(const float* a, const float* b, float* out, std::size_t n)
    {
        uv(out, n) = uv(a, n) * uv(b, n);
    }

    float sdrk_kfr_dot_f32(const float* a, const float* b, std::size_t n)
    {
        return kfr::dotproduct(uv(a, n), uv(b, n));
    }

    void sdrk_kfr_dot_complex_real(const float* a, const float* b, std::size_t n, float* out)
    {
        const c32 r = kfr::dotproduct(uv(reinterpret_cast<const c32*>(a), n), uv(b, n));
        out[0]      = r.real();
        out[1]      = r.imag();
    }

    void sdrk_kfr_mag(const float* c, float* out, std::size_t n)
    {
        uv(out, n) = kfr::cabs(uv(reinterpret_cast<const c32*>(c), n));
    }

    void sdrk_kfr_deinterleave_f32(const float* c, float* re, float* im, std::size_t n)
    {
        const auto x = uv(reinterpret_cast<const c32*>(c), n);
        uv(re, n)    = kfr::real(x);
        uv(im, n)    = kfr::imag(x);
    }
}
