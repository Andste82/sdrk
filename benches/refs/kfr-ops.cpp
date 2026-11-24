// Element-wise operations of KFR (its expressions over `univector_ref`, which KFR evaluates with
// the vectors of the architecture it is compiled for), behind a C interface for the cycle
// measurements of `cargo measure` (benches/cref-sys). benches/refs/build.sh compiles this file
// once per SIMD level with the flags of that level, into the library of the KFR biquad of the
// level (libkfrbq-<level>.a). They are the references at `sse2` where VOLK's machine of the level
// runs a `generic` protokernel (the maintainer's decision on I13.1-11, D2785):
//
// - `sdrk_kfr_i8_to_f32`: `out = in * scale` from signed bytes, the work of
//   `volk_8i_s32f_convert_32f` with `1 / scale`.
// - `sdrk_kfr_i8_to_i16`: `out = in << 8` from signed bytes, the work of `volk_8i_convert_16i`.
// - `sdrk_kfr_complex_mul`: the complex product `out = a b`, the work of
//   `volk_32fc_x2_multiply_32fc`.
// - `sdrk_kfr_power_db`: `out = 10 log10(in) + offset`, the work of VOLK's `log2`, scale and add
//   chain of `power_db`.
//
// KFR's own sample conversion `kfr::convert` is not used: it has no 8-bit type, clamps every
// sample and converts one sample at a time (a scalar loop), while these expressions convert
// whole vectors. The file needs no C++ run-time library (no allocation, no exceptions).
#include <kfr/base.hpp>
#include <kfr/math.hpp>
#include <kfr/simd.hpp>

#include <cstddef>
#include <cstdint>

extern "C"
{
    void sdrk_kfr_i8_to_f32(const std::int8_t* in, float* out, std::size_t n, float scale)
    {
        kfr::make_univector(out, n) = kfr::cast<float>(kfr::make_univector(in, n)) * scale;
    }

    void sdrk_kfr_i8_to_i16(const std::int8_t* in, std::int16_t* out, std::size_t n)
    {
        kfr::make_univector(out, n) = kfr::cast<std::int16_t>(kfr::make_univector(in, n)) << 8;
    }

    void sdrk_kfr_complex_mul(const float* a, const float* b, float* out, std::size_t n)
    {
        using c32 = kfr::complex<float>;
        kfr::make_univector(reinterpret_cast<c32*>(out), n) =
            kfr::make_univector(reinterpret_cast<const c32*>(a), n) *
            kfr::make_univector(reinterpret_cast<const c32*>(b), n);
    }

    void sdrk_kfr_power_db(const float* in, float* out, std::size_t n, float offset)
    {
        kfr::make_univector(out, n) = 10.0f * kfr::log10(kfr::make_univector(in, n)) + offset;
    }
}
