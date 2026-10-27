// The biquad cascade of KFR (`kfr::iir` on an `iir_state<float, 2>`: the sections in the lanes
// of one vector, transposed direct form II), behind a C interface for the instruction counts of
// `cargo xtask count` (benches/cref-sys). benches/refs/build.sh compiles this file once per
// SIMD level with the flags of that level, into a library of its own (libkfrbq-<level>.a);
// benches/cref-sys links the library of the level the measurement binary is built for.
//
// One handle filters one real channel: `y = b0 x + s1`, `s1 = s2 + b1 x - a1 y`,
// `s2 = b2 x - a2 y` per section. Consecutive calls continue the stream: KFR keeps the state of
// the sections, including the look-ahead of its cascade, in the handle.
//
// The file is compiled without exceptions and RTTI, and allocates with aligned_alloc and
// placement new, so it needs no C++ run-time library (the state is a plain aggregate of
// vectors). The filter loop itself throws nothing, so this does not change its code.
#include <kfr/dsp/biquad.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <new>

namespace
{
constexpr std::size_t sections = 2;
using state_t                  = kfr::iir_state<float, sections>;
constexpr std::size_t align    = alignof(state_t) < 64 ? 64 : alignof(state_t);
} // namespace

extern "C"
{
    // A handle for the two sections of `coefs` (`[b0, b1, b2, a1, a2]` per section, a0 = 1), or
    // null if `count` is not 2 or the allocation fails.
    void* sdrk_kfr_biquad_f32_new(const float* coefs, std::uint32_t count)
    {
        if (count != sections)
            return nullptr;
        kfr::biquad_section<float> bq[sections];
        for (std::size_t k = 0; k < sections; ++k)
        {
            const float* c = coefs + 5 * k;
            bq[k]          = kfr::biquad_section<float>(1.0f, c[3], c[4], c[0], c[1], c[2]);
        }
        void* memory = std::aligned_alloc(align, (sizeof(state_t) + align - 1) / align * align);
        if (memory == nullptr)
            return nullptr;
        return new (memory) state_t(kfr::iir_params<float, sections>(bq, sections));
    }

    // Filters `n` samples of `input` into `output` (distinct buffers).
    void sdrk_kfr_biquad_f32_run(void* handle, const float* input, float* output, std::size_t n)
    {
        state_t& state                 = *static_cast<state_t*>(handle);
        kfr::make_univector(output, n) = kfr::iir(kfr::make_univector(input, n), std::ref(state));
    }

    void sdrk_kfr_biquad_f32_free(void* handle)
    {
        if (handle == nullptr)
            return;
        static_cast<state_t*>(handle)->~state_t();
        std::free(handle);
    }

    // The SIMD architecture KFR was compiled for (`sse2`, `avx2`, `neon64`, ...).
    const char* sdrk_kfr_biquad_arch(void) { return KFR_STRINGIFY(KFR_ARCH_NAME); }
}
