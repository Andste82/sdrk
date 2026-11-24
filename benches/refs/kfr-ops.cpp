// Element-wise operations of KFR (its expressions over `univector_ref`, which KFR evaluates with
// the vectors of the architecture it is compiled for), behind a C interface for the cycle
// measurements of `cargo measure` (benches/cref-sys). benches/refs/build.sh compiles this file
// once per SIMD level with the flags of that level (x86-64, aarch64 and wasm32 in the Emscripten
// configuration of kfr-biquad.cpp), into a library of its
// own (libkfrops-<level>.a). They are the SIMD references where VOLK's machine of the level runs
// a `generic` protokernel (`sse2`, the maintainer's decision on I13.1-11, D2785) or a protokernel
// of a lower level (`avx512`, D2900), and where no other SIMD library has the operation (D2901):
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
// - `sdrk_kfr_f32_to_u8`: `out = clamp(round(in * scale + bias), 0, 255)` (round to the nearest
//   integer, ties to even), the work of `volk_32f_s32f_x2_convert_8u`.
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
        uv(out, n) = kfr::cast<std::uint8_t>(
            kfr::cast<std::int32_t>(kfr::round(kfr::clamp(uv(in, n) * scale + bias, 0.0f, 255.0f))));
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
}
