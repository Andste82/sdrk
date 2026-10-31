#!/usr/bin/env bash
# Builds the C reference libraries of `cargo xtask count` from pinned release tarballs.
#
#   benches/refs/build.sh <x86_64|aarch64|wasm32|cortex-m4> <prefix>
#
# xtask runs this script before it builds the measurement binary of a platform; it returns at
# once when <prefix> already holds a build of this script with the same compilers. CI caches
# <prefix>. Every tarball is checked against its SHA-256 before it is unpacked.
#
# | Library | Version | License | Platforms | Use |
# | --- | --- | --- | --- | --- |
# | VOLK | 3.3.0 | LGPL-3.0-or-later | x86_64, aarch64 (shared, CMake Release, no ORC); wasm32 (the `generic` protokernels, static, see volk-wasm.c; since the fallback round with the multiply, log2, scalar multiply and add, dot product and 8i-to-16i kernels) | the reference of the streaming kernels; VOLK's generic `multiply_conjugate` in gr-quad-demod.c |
# | FFTW | 3.3.10 | GPL-2.0-or-later | x86_64 (`--enable-sse2`), aarch64 (`--enable-neon`); static, single precision | the C32 FFT reference at `sse2`, information on `neon` |
# | KissFFT | 131.1.0 | BSD-3-Clause | all (static, `FIXED_POINT=16`) | the CQ15 FFT reference |
# | Ne10 | 1f059a7 (last commit of the archived repository) | BSD-3-Clause | aarch64 (static, DSP module) | information next to KissFFT on `neon` |
# | GNU Radio `agc_cc` | main at ec1bf15 (agc.h of gr-analog), see gr-agc.c | GPL-3.0-or-later | all (static, one object file) | the reference of the `C32` `agc` |
# | GNU Radio `pwr_squelch_cc` | main at ec1bf15 (squelch_base_cc_impl.cc and pwr_squelch_cc_impl of gr-analog), see gr-squelch.c | GPL-3.0-or-later | all (static, one object file) | the reference of the `C32` `squelch` |
# | GNU Radio `rotator` | main at ec1bf15 (rotator.h of gr-blocks, the scalar `rotate`), see gr-rotator.c | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, one object file) | the scalar reference of the `C32` `rotate` fallback |
# | GNU Radio `frequency_modulator_fc` | main at ec1bf15 (frequency_modulator_fc_impl.cc of gr-analog with `gr::fxpt` and its sine table), see gr-fm-mod.c | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, one object file) | the scalar reference of `fm_mod` |
# | GNU Radio `quadrature_demod_cf` | main at ec1bf15 (quadrature_demod_cf_impl.cc of gr-analog, `fast_atan2f` of gnuradio-runtime, VOLK's generic `multiply_conjugate`), see gr-quad-demod.c and gr-fast-atan2f.c | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, two object files) | the scalar reference of `quad_demod` |
# | GNU Radio `fm_deemph` | main at ec1bf15 (fm_emph.py of gr-analog, `iir_filter_ffd` of gr-filter), see gr-deemph.c | GPL-3.0-or-later | x86_64, aarch64, wasm32 (static, one object file) | the scalar reference of `deemphasis` |
# | liquid-dsp `iirfilt` DC blocker | 1.8.3 (iirfilt.proto.c and dotprod.proto.c, portable dot products), see liquid-dcblock.c and liquid-dotprod.c | MIT | x86_64, aarch64, wasm32 (static, two object files) | the scalar reference of `dc_block` and `dc_block_real` |
# | liquid-dsp halfband decimators | 1.8.3 (the release tarball: `resamp2.proto.c`, `firhilb.proto.c`, the dot products `dotprod_crcf.c` and `dotprod_rrrf.c` with their SSE, AVX, AVX-512F and NEON loops, the window buffers and the support files), see liquid-halfband.c | MIT | x86_64, aarch64, wasm32 (static, one library per level, built from the unpacked release) | the reference of `halfband_decimate` (`resamp2_crcf`), `halfband_decimate_real` (`resamp2_rrrf`) and `real_to_complex` (`firhilbf`) with the SIMD dot product of the level (the portable loop at `simd128`), and the scalar reference of their fallbacks (the portable dot product) |
# | libairspy | 1.0.10 (iqconverter_float.c and iqconverter_int16.c; the last release under the MIT licence, later ones are under a restrictive licence and must not be used) | MIT | x86_64, aarch64, wasm32 (static, one library per level), see airspy-iqconv.c | the SSE2 reference of the `C32` `real_to_complex` and information next to the fallback; the `CQ15` `real_to_complex` (`iqconverter_int16`, a lower accuracy class), conversion only, without `remove_dc` |
# | KFR | 7.2.0 | GPL-2.0-or-later | x86_64 (one library per level), aarch64 (clang++), wasm32 (clang++, KFR's Emscripten configuration: `sse41` over the SSE headers of Emscripten); the header-only `kfr::iir`, see kfr-biquad.cpp | the SIMD reference of `biquad` and `biquad_real` (D15) |
# | CMSIS-DSP | 1.18.0 | Apache-2.0 | x86_64, aarch64, wasm32 (one library per level), generic C; cortex-m4 with the DSP extension and the FPU (biquads only); see cmsis-biquad.c and cmsis-dsp.c | the scalar reference of the biquad fallbacks (D16); the gate of `biquad` and `biquad_real` at `simd128` (D15); the comparison of the Cortex-M4 report (D27); the scalar references of the fallback round (conversions, products, magnitudes, sine and arc tangent, logarithm, FIR filters, FFTs: libcmsisdsp-<level>.a) |
# | wasi-sdk sysroot | 34.0 | Apache-2.0 WITH LLVM-exception (libc++), the licenses of wasi-libc | wasm32 (headers only: libc++ and wasi-libc for the KFR shim, which needs no C++ run-time library) | KFR needs a C++ standard library |
# | Emscripten | 6.0.12 | MIT or NCSA | wasm32 (headers only: the `*intrin.h` of system/include/compat, SSE written with SIMD128) | KFR's x86 code on wasm32, as emcc builds it with `-msse4.1 -msimd128` |
# | CMSIS 6 (CMSIS-Core) | 6.3.0 | Apache-2.0 | cortex-m4 (headers only: `cmsis_compiler.h` and the intrinsics such as `__SMLALD` that CMSIS-DSP uses with the DSP extension) | needed by CMSIS-DSP outside its host configuration |
#
# The libraries are linked only into the measurement binary of `benches/kernels`, which is
# built on the measuring machine and never distributed; none of them is a dependency of a
# library crate of SDRk.
#
# Compilers: x86_64 `${CC:-gcc}`/`${CXX:-g++}`, aarch64 `aarch64-linux-gnu-gcc`/`-g++` (KFR with
# `${KFR_AARCH64_CXX:-clang++} --target=aarch64-linux-gnu`: GCC rejects KFR's NEON code), wasm32
# `${WASM_CC:-clang}` with the wasi-libc headers (KFR with `${WASM_CXX:-clang++}` and the wasi-sdk
# sysroot above), cortex-m4 `${M4_CC:-arm-none-eabi-gcc}` with the
# newlib headers (`libnewlib-arm-none-eabi`: CMSIS-DSP includes `string.h` and `math.h`; the
# biquad sources need only `memset`, which `compiler_builtins` of the Rust workload provides).
# Their versions are part of the stamp and of <prefix>/refs.json, which xtask copies into the
# results; the count jobs of CI pin them with a container image and an apt snapshot (see
# .github/workflows/ci.yml).
#
# Needs curl, sha256sum, tar, make, cmake, ninja and python3 with mako (VOLK), and for the
# platforms aarch64, wasm32 and cortex-m4 the cross compilers above. Environment:
# SDRK_REFS_DOWNLOADS (download cache, default <prefix>/../downloads), SDRK_REFS_JOBS (default 2).
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <x86_64|aarch64|wasm32|cortex-m4> <prefix>" >&2
    exit 2
fi
platform=$1
mkdir -p "$2"
prefix=$(cd "$2" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
downloads=${SDRK_REFS_DOWNLOADS:-$prefix/../downloads}
jobs=${SDRK_REFS_JOBS:-2}

VOLK_VERSION=3.3.0
VOLK_URL=https://github.com/gnuradio/volk/releases/download/v$VOLK_VERSION/volk-$VOLK_VERSION.tar.gz
VOLK_SHA256=89d11c8c8d4213b1b780354cfdbda1fed0c0b65c82847e710638eb3e21418628
FFTW_VERSION=3.3.10
FFTW_URL=https://www.fftw.org/fftw-$FFTW_VERSION.tar.gz
FFTW_SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
KISSFFT_VERSION=131.1.0
KISSFFT_URL=https://github.com/mborgerding/kissfft/archive/refs/tags/$KISSFFT_VERSION.tar.gz
KISSFFT_SHA256=76c1aac87ddb7258f34b08a13f0eebf9e53afa299857568346aa5c82bcafaf1a
NE10_COMMIT=1f059a764d0e1bc2481c0055c0e71538470baa83
NE10_URL=https://github.com/projectNe10/Ne10/archive/$NE10_COMMIT.tar.gz
NE10_SHA256=faf28b2555105224eb76968bad555f7df32cb5b5580b40ed974bbaa2e997f480
KFR_VERSION=7.2.0
KFR_URL=https://github.com/kfrlib/kfr/archive/refs/tags/$KFR_VERSION.tar.gz
KFR_SHA256=4f1603d5b6e609c25d25cff58fe7154a371583fd6d379615b342e44223be0763
CMSISDSP_VERSION=1.18.0
CMSISDSP_URL=https://github.com/ARM-software/CMSIS-DSP/archive/refs/tags/v$CMSISDSP_VERSION.tar.gz
CMSISDSP_SHA256=b60944180f51f9bb50df1a8cccad19d7c1ef8c14ed13da50b9b3f141032c3f56
# The CMSIS release that was current when CMSIS-DSP 1.18.0 came out (2026-09-10).
CMSISCORE_VERSION=6.3.0
CMSISCORE_URL=https://github.com/ARM-software/CMSIS_6/archive/refs/tags/v$CMSISCORE_VERSION.tar.gz
CMSISCORE_SHA256=331df74000876b8fb07933658cbf402c4d9f831b429258792d2c34ab5a6c5bf7
# KFR on wasm32: the C++ standard library of wasi-sdk and the SSE headers of Emscripten, the
# releases that were current when KFR 7.2.0 came out (2026-10-12).
WASISYSROOT_VERSION=34.0
WASISYSROOT_URL=https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sysroot-$WASISYSROOT_VERSION.tar.gz
WASISYSROOT_SHA256=9d813544eeebe38b7b8f2244ed591de46b6db812c6dd1a257ff9f0d2a905a2be
EMSCRIPTEN_VERSION=6.0.12
EMSCRIPTEN_URL=https://github.com/emscripten-core/emscripten/archive/refs/tags/$EMSCRIPTEN_VERSION.tar.gz
EMSCRIPTEN_SHA256=795ef929e4dee52bf03de3742c4252bb121e742afac896378d969b5da2db86eb
# liquid-dsp (the halfband decimators; the DC blocker below is vendored from the same release).
LIQUID_VERSION=1.8.3
LIQUID_URL=https://github.com/jgaeddert/liquid-dsp/archive/refs/tags/v$LIQUID_VERSION.tar.gz
LIQUID_SHA256=18fa83b73db8bb6fe6ea0376e4b5aecf8645970f4604d10d9dadbf609f3f95e2
# libairspy: the last release under the MIT licence.
AIRSPY_VERSION=1.0.10
AIRSPY_URL=https://github.com/airspy/airspyone_host/archive/refs/tags/v$AIRSPY_VERSION.tar.gz
AIRSPY_SHA256=fcca23911c9a9da71cebeffeba708c59d1d6401eec6eb2dd73cae35b8ea3c613

# The optimisation flags of the libraries built here without their own build system or with a
# build system that would tune for the build machine (FFTW adds -mtune=native otherwise).
OPT_FLAGS="-O3 -fomit-frame-pointer -fstrict-aliasing"

case $platform in
    x86_64)
        cc=${CC:-gcc}
        cxx=${CXX:-g++}
        kfr_cxx=$cxx
        ar=${AR:-ar}
        ;;
    aarch64)
        cc=aarch64-linux-gnu-gcc
        cxx=aarch64-linux-gnu-g++
        kfr_cxx="${KFR_AARCH64_CXX:-clang++} --target=aarch64-linux-gnu"
        ar=aarch64-linux-gnu-ar
        ;;
    wasm32)
        cc="${WASM_CC:-clang} --target=wasm32-wasip1"
        cxx=""
        kfr_cxx="${WASM_CXX:-clang++} --target=wasm32-wasip1"
        ar=${WASM_AR:-llvm-ar}
        ;;
    cortex-m4)
        cc=${M4_CC:-arm-none-eabi-gcc}
        cxx=""
        kfr_cxx=""
        ar=${M4_AR:-arm-none-eabi-ar}
        ;;
    *)
        echo "unknown platform $platform" >&2
        exit 2
        ;;
esac

version_of() { $1 --version 2>/dev/null | head -n 1; }
cc_version=$(version_of "${cc%% *}")
cxx_version=""
if [ -n "$cxx" ]; then
    cxx_version=$(version_of "$cxx")
fi
kfr_cxx_version=""
if [ -n "$kfr_cxx" ]; then
    kfr_cxx_version=$(version_of "${kfr_cxx%% *}")
fi

stamp="$prefix/refs.stamp"
want=$( (cat "${BASH_SOURCE[0]}" "$here/volk-wasm.c" "$here/gr-agc.c" "$here/gr-squelch.c" "$here/kfr-biquad.cpp" "$here/cmsis-biquad.c" "$here/cmsis-m4-heap.c" \
    "$here/cmsis-dsp.c" "$here/gr-rotator.c" "$here/gr-fm-mod.c" "$here/gr-quad-demod.c" "$here/gr-fast-atan2f.c" "$here/gr-deemph.c" \
    "$here/liquid-dotprod.c" "$here/liquid-dcblock.c" "$here/liquid-halfband.c" "$here/airspy-iqconv.c"; echo "$platform|$cc_version|$cxx_version|$kfr_cxx_version") | sha256sum | cut -d ' ' -f 1)
if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
    exit 0
fi
echo "==> building the C references for $platform into $prefix"
rm -rf "$prefix/lib" "$prefix/include" "$prefix/refs.json" "$stamp"
work="$prefix/build"
rm -rf "$work"
mkdir -p "$work" "$downloads" "$prefix/lib/pkgconfig" "$prefix/include"

# fetch <url> <sha256> <file name>: downloads into the cache once, then checks the checksum.
fetch() {
    local file="$downloads/$3"
    if [ ! -f "$file" ]; then
        curl -fsSL --retry 3 "$1" -o "$file.part"
        mv "$file.part" "$file"
    fi
    echo "$2  $file" | sha256sum -c --quiet -
}

# unpack <file name> <directory>: the tarball's single top-level directory becomes <directory>.
unpack() {
    mkdir -p "$work/$2"
    tar -xzf "$downloads/$1" -C "$work/$2" --strip-components=1
}

build_volk() {
    fetch "$VOLK_URL" "$VOLK_SHA256" "volk-$VOLK_VERSION.tar.gz"
    unpack "volk-$VOLK_VERSION.tar.gz" volk
    local toolchain=()
    if [ "$platform" = aarch64 ]; then
        toolchain=(-DCMAKE_TOOLCHAIN_FILE="$work/volk/cmake/Toolchains/aarch64-linux-gnu.cmake")
    else
        toolchain=(-DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx")
    fi
    cmake -S "$work/volk" -B "$work/volk-build" -G Ninja "${toolchain[@]}" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib \
        -DENABLE_TESTING=OFF -DENABLE_MODTOOL=OFF -DENABLE_ORC=OFF >"$work/volk-configure.log"
    grep -m 1 'Available machines' "$work/volk-configure.log" || true
    cmake --build "$work/volk-build" -j "$jobs" >"$work/volk-build.log"
    cmake --install "$work/volk-build" >/dev/null
}

# The generic protokernels of VOLK for wasm32, where VOLK has no machine of its own.
build_volk_wasm() {
    fetch "$VOLK_URL" "$VOLK_SHA256" "volk-$VOLK_VERSION.tar.gz"
    unpack "volk-$VOLK_VERSION.tar.gz" volk
    local flags="-O3 -msimd128"
    $cc $flags -DSDRK_VOLK_FLAGS="\"$cc_version: $flags\"" -DSDRK_VOLK_VERSION="\"$VOLK_VERSION\"" \
        -I "$work/volk/include" -I "$work/volk/kernels" -c "$here/volk-wasm.c" -o "$work/volk-wasm.o"
    $ar rcs "$prefix/lib/libvolk.a" "$work/volk-wasm.o"
    cat >"$prefix/lib/pkgconfig/volk.pc" <<EOF
prefix=$prefix
libdir=\${prefix}/lib

Name: volk
Description: The generic protokernels of VOLK $VOLK_VERSION for wasm32 (benches/refs/volk-wasm.c)
Version: $VOLK_VERSION
Libs: -L\${libdir} -lvolk
EOF
}

build_fftw() {
    fetch "$FFTW_URL" "$FFTW_SHA256" "fftw-$FFTW_VERSION.tar.gz"
    unpack "fftw-$FFTW_VERSION.tar.gz" fftw
    local simd=(--enable-sse2)
    local host=()
    if [ "$platform" = aarch64 ]; then
        simd=(--enable-neon)
        host=(--host=aarch64-linux-gnu)
    fi
    (cd "$work/fftw" && ./configure "${host[@]}" --prefix="$prefix" --libdir="$prefix/lib" \
        --enable-float "${simd[@]}" --enable-static --disable-shared --disable-fortran \
        --disable-doc --disable-threads CC="$cc" CFLAGS="$OPT_FLAGS" >"$work/fftw-configure.log")
    make -C "$work/fftw" -j "$jobs" >"$work/fftw-build.log"
    make -C "$work/fftw" install >/dev/null
}

build_kissfft() {
    fetch "$KISSFFT_URL" "$KISSFFT_SHA256" "kissfft-$KISSFFT_VERSION.tar.gz"
    unpack "kissfft-$KISSFFT_VERSION.tar.gz" kissfft
    local flags="-O3"
    if [ "$platform" = wasm32 ]; then
        flags="-O3 -msimd128"
    fi
    $cc $flags -DFIXED_POINT=16 -I "$work/kissfft" -c "$work/kissfft/kiss_fft.c" -o "$work/kiss_fft.o"
    $ar rcs "$prefix/lib/libkissfft16.a" "$work/kiss_fft.o"
    cp "$work/kissfft/kiss_fft.h" "$prefix/include/"
    KISSFFT_FLAGS="$flags -DFIXED_POINT=16"
}

build_ne10() {
    fetch "$NE10_URL" "$NE10_SHA256" "Ne10-$NE10_COMMIT.tar.gz"
    unpack "Ne10-$NE10_COMMIT.tar.gz" ne10
    NE10_LINUX_TARGET_ARCH=aarch64 cmake -S "$work/ne10" -B "$work/ne10-build" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$work/ne10/GNUlinux_config.cmake" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_BUILD_TYPE=Release -DGNULINUX_PLATFORM=ON -DNE10_LINUX_TARGET_ARCH=aarch64 \
        -DNE10_ENABLE_DSP=ON -DNE10_ENABLE_MATH=OFF -DNE10_ENABLE_IMGPROC=OFF \
        -DNE10_ENABLE_PHYSICS=OFF -DNE10_BUILD_UNIT_TEST=OFF -DNE10_BUILD_EXAMPLES=OFF \
        -DNE10_BUILD_SHARED=OFF >"$work/ne10-configure.log"
    cmake --build "$work/ne10-build" -j "$jobs" >"$work/ne10-build.log"
    cp "$(find "$work/ne10-build" -name libNE10.a | head -n 1)" "$prefix/lib/"
}

# The loop of GNU Radio's agc_cc, compiled with the generic flags of GNU Radio's own build.
build_gr_agc() {
    GR_AGC_FLAGS="-O3"
    if [ "$platform" = wasm32 ]; then
        GR_AGC_FLAGS="-O3 -msimd128"
    fi
    $cc $GR_AGC_FLAGS -c "$here/gr-agc.c" -o "$work/gr-agc.o"
    $ar rcs "$prefix/lib/libgragc.a" "$work/gr-agc.o"
}

# The loop of GNU Radio's pwr_squelch_cc, compiled with the same generic flags.
build_gr_squelch() {
    $cc $GR_AGC_FLAGS -c "$here/gr-squelch.c" -o "$work/gr-squelch.o"
    $ar rcs "$prefix/lib/libgrsquelch.a" "$work/gr-squelch.o"
}

# The scalar references of the fallback round from GNU Radio, compiled with the same generic
# flags as agc_cc: the scalar rotator, frequency_modulator_fc, quadrature_demod_cf (with
# fast_atan2f in an object file of its own, as in libgnuradio-runtime, and VOLK's generic
# multiply_conjugate from the unpacked VOLK release, so it runs after build_volk or
# build_volk_wasm) and the IIR filter of fm_deemph. One library per block.
build_gr_fallback() {
    local volk_inc=(-I "$work/volk/include" -I "$work/volk/kernels")
    $cc $GR_AGC_FLAGS -c "$here/gr-rotator.c" -o "$work/gr-rotator.o"
    $ar rcs "$prefix/lib/libgrrotator.a" "$work/gr-rotator.o"
    $cc $GR_AGC_FLAGS -c "$here/gr-fm-mod.c" -o "$work/gr-fm-mod.o"
    $ar rcs "$prefix/lib/libgrfmmod.a" "$work/gr-fm-mod.o"
    $cc $GR_AGC_FLAGS "${volk_inc[@]}" -c "$here/gr-quad-demod.c" -o "$work/gr-quad-demod.o"
    $cc $GR_AGC_FLAGS -c "$here/gr-fast-atan2f.c" -o "$work/gr-fast-atan2f.o"
    $ar rcs "$prefix/lib/libgrquaddemod.a" "$work/gr-quad-demod.o" "$work/gr-fast-atan2f.o"
    $cc $GR_AGC_FLAGS -c "$here/gr-deemph.c" -o "$work/gr-deemph.o"
    $ar rcs "$prefix/lib/libgrdeemph.a" "$work/gr-deemph.o"
}

# liquid-dsp's DC blocker (iirfilt with its portable dot products, liquid-dcblock.c and
# liquid-dotprod.c), with liquid's own optimisation level (-O3 in its CMakeLists.txt), the
# generic flags of the GNU Radio references.
build_liquid_dcblock() {
    $cc $GR_AGC_FLAGS -c "$here/liquid-dotprod.c" -o "$work/liquid-dotprod.o"
    $cc $GR_AGC_FLAGS -c "$here/liquid-dcblock.c" -o "$work/liquid-dcblock.o"
    $ar rcs "$prefix/lib/libliquiddc.a" "$work/liquid-dotprod.o" "$work/liquid-dcblock.o"
}

# The levels of the biquad references with their instruction-set flags: those of the measured
# SDRk level (the `-C target-cpu` of xtask's plan), so that a reference uses what the level
# allows and nothing above it. A comma stands for a space (the list is split at spaces); the
# Cortex-M4 entry is the M4F of the report with its DSP extension and its single-precision FPU.
biquad_levels() {
    case $platform in
        x86_64) echo "sse2:-march=x86-64 sse4.2:-march=x86-64-v2 avx2:-march=x86-64-v3 avx512:-march=sapphirerapids" ;;
        aarch64) echo "neon:-march=armv8-a" ;;
        wasm32) echo "simd128:-msimd128" ;;
        cortex-m4) echo "cortex-m4:-mcpu=cortex-m4,-mfpu=fpv4-sp-d16,-mfloat-abi=hard,-mthumb" ;;
    esac
}

# The common flags of the biquad references: no contraction into FMA (D5 forbids it on our
# side, so neither side may fuse).
BIQUAD_OPT="-O3 -ffp-contract=off"
# KFR without exceptions, RTTI and thread-safe statics: the shim then needs no C++ run-time
# library (its filter loop throws nothing).
KFR_CXX_FLAGS="-std=c++20 -fno-exceptions -fno-rtti -fno-threadsafe-statics"
CMSISDSP_SOURCES="arm_biquad_cascade_df2T_f32 arm_biquad_cascade_df2T_init_f32 arm_biquad_cascade_stereo_df2T_f32 arm_biquad_cascade_stereo_df2T_init_f32 arm_biquad_cascade_df1_q15 arm_biquad_cascade_df1_init_q15"
# The configuration of CMSIS-DSP's own CMake build for a host (its Python wrapper): no CMSIS
# Core, generic C, loop unrolling on (the default of its option LOOPUNROLL).
CMSISDSP_DEFINES="-D__GNUC_PYTHON__ -DARM_MATH_LOOPUNROLL"
# On the Cortex-M4 the configuration of a device build: CMSIS-Core instead of __GNUC_PYTHON__,
# so that arm_math_types.h sets ARM_MATH_DSP from __ARM_FEATURE_DSP and the Q15 cascade takes its
# DSP-extension path (__SMLAD, __SMLALD, __PKHBT on two samples per word); loop unrolling on.
if [ "$platform" = cortex-m4 ]; then
    CMSISDSP_DEFINES="-DARM_MATH_LOOPUNROLL"
fi

# CMSIS-DSP's biquad cascades (no CMake: the six sources and the handles of cmsis-biquad.c),
# one library per level: libcmsisbq-<level>.a. Generic C on the hosts; on the Cortex-M4 with
# CMSIS-Core and the bump allocator of cmsis-m4-heap.c (there is no C library on the target).
build_cmsis_biquad() {
    fetch "$CMSISDSP_URL" "$CMSISDSP_SHA256" "CMSIS-DSP-$CMSISDSP_VERSION.tar.gz"
    mkdir -p "$work/cmsis"
    tar -xzf "$downloads/CMSIS-DSP-$CMSISDSP_VERSION.tar.gz" -C "$work/cmsis" --strip-components=1 \
        "CMSIS-DSP-$CMSISDSP_VERSION/Include" "CMSIS-DSP-$CMSISDSP_VERSION/PrivateInclude" \
        "CMSIS-DSP-$CMSISDSP_VERSION/Source/FilteringFunctions"
    local inc=(-I "$work/cmsis/Include" -I "$work/cmsis/PrivateInclude")
    if [ "$platform" = cortex-m4 ]; then
        fetch "$CMSISCORE_URL" "$CMSISCORE_SHA256" "CMSIS_6-$CMSISCORE_VERSION.tar.gz"
        mkdir -p "$work/cmsis-core"
        tar -xzf "$downloads/CMSIS_6-$CMSISCORE_VERSION.tar.gz" -C "$work/cmsis-core" --strip-components=4 \
            "CMSIS_6-$CMSISCORE_VERSION/CMSIS/Core/Include"
        inc+=(-I "$work/cmsis-core")
    fi
    local entry level march dir src objects
    for entry in $(biquad_levels); do
        level=${entry%%:*}
        march=${entry#*:}
        march=${march//,/ }
        dir="$work/cmsis-$level"
        objects=()
        mkdir -p "$dir"
        for src in $CMSISDSP_SOURCES; do
            $cc $BIQUAD_OPT $march $CMSISDSP_DEFINES "${inc[@]}" \
                -c "$work/cmsis/Source/FilteringFunctions/$src.c" -o "$dir/$src.o"
            objects+=("$dir/$src.o")
        done
        if [ "$platform" = cortex-m4 ]; then
            $cc $BIQUAD_OPT $march -c "$here/cmsis-m4-heap.c" -o "$dir/heap.o"
            objects+=("$dir/heap.o")
        fi
        $cc $BIQUAD_OPT $march $CMSISDSP_DEFINES "${inc[@]}" -c "$here/cmsis-biquad.c" -o "$dir/handles.o"
        $ar rcs "$prefix/lib/libcmsisbq-$level.a" "${objects[@]}" "$dir/handles.o"
    done
}

# The CMSIS-DSP sources of the scalar references of the fallback round (below Source/), with
# the sources they need (arm_sqrt_q31, arm_divide_q15, arm_abs_q15, the radix kernels, the
# bit reversals and the common tables and constant structures of the transforms).
CMSISDSP_FALLBACK_SOURCES="SupportFunctions/arm_q7_to_float SupportFunctions/arm_q15_to_float SupportFunctions/arm_q7_to_q15
    SupportFunctions/arm_float_to_q15 SupportFunctions/arm_float_to_q7 SupportFunctions/arm_q15_to_q7
    BasicMathFunctions/arm_dot_prod_f32 BasicMathFunctions/arm_dot_prod_q15 BasicMathFunctions/arm_scale_f32
    BasicMathFunctions/arm_scale_q15 BasicMathFunctions/arm_mult_f32 BasicMathFunctions/arm_mult_q15
    BasicMathFunctions/arm_abs_q15 BasicMathFunctions/arm_add_q15
    ComplexMathFunctions/arm_cmplx_dot_prod_f32 ComplexMathFunctions/arm_cmplx_dot_prod_q15
    ComplexMathFunctions/arm_cmplx_mult_real_f32 ComplexMathFunctions/arm_cmplx_mult_real_q15
    ComplexMathFunctions/arm_cmplx_mult_cmplx_f32 ComplexMathFunctions/arm_cmplx_mult_cmplx_q15
    ComplexMathFunctions/arm_cmplx_conj_f32 ComplexMathFunctions/arm_cmplx_mag_squared_f32
    ComplexMathFunctions/arm_cmplx_mag_squared_q15 ComplexMathFunctions/arm_cmplx_mag_f32 ComplexMathFunctions/arm_cmplx_mag_q15
    ControllerFunctions/arm_sin_cos_f32 ControllerFunctions/arm_sin_cos_q31
    FastMathFunctions/arm_sin_f32 FastMathFunctions/arm_cos_f32 FastMathFunctions/arm_sin_q15 FastMathFunctions/arm_cos_q15
    FastMathFunctions/arm_atan2_f32 FastMathFunctions/arm_atan2_q15 FastMathFunctions/arm_vlog_f32
    FastMathFunctions/arm_sqrt_q31 FastMathFunctions/arm_divide_q15
    FilteringFunctions/arm_fir_f32 FilteringFunctions/arm_fir_q15 FilteringFunctions/arm_fir_init_f32 FilteringFunctions/arm_fir_init_q15
    FilteringFunctions/arm_fir_decimate_f32 FilteringFunctions/arm_fir_decimate_q15
    FilteringFunctions/arm_fir_decimate_init_f32 FilteringFunctions/arm_fir_decimate_init_q15
    FilteringFunctions/arm_fir_interpolate_f32 FilteringFunctions/arm_fir_interpolate_q15
    FilteringFunctions/arm_fir_interpolate_init_f32 FilteringFunctions/arm_fir_interpolate_init_q15
    TransformFunctions/arm_cfft_f32 TransformFunctions/arm_cfft_q15 TransformFunctions/arm_cfft_radix8_f32
    TransformFunctions/arm_cfft_radix4_q15 TransformFunctions/arm_bitreversal TransformFunctions/arm_bitreversal2
    TransformFunctions/arm_cfft_init_f32 TransformFunctions/arm_rfft_fast_f32 TransformFunctions/arm_rfft_fast_init_f32
    CommonTables/arm_common_tables CommonTables/arm_const_structs"

# The scalar references of the fallback round from CMSIS-DSP (generic C with the defines and
# flags of the biquad references, D16), one library per level: libcmsisdsp-<level>.a with the
# sources above and the handles of cmsis-dsp.c. Not on the Cortex-M4 (its report compares the
# biquads only). Runs after build_cmsis_biquad, which unpacks the headers.
build_cmsis_dsp() {
    local dirs=() src
    for src in $CMSISDSP_FALLBACK_SOURCES; do
        dirs+=("CMSIS-DSP-$CMSISDSP_VERSION/Source/${src%%/*}")
    done
    tar -xzf "$downloads/CMSIS-DSP-$CMSISDSP_VERSION.tar.gz" -C "$work/cmsis" --strip-components=1 \
        $(printf '%s\n' "${dirs[@]}" | sort -u)
    local inc=(-I "$work/cmsis/Include" -I "$work/cmsis/PrivateInclude")
    local entry level march dir objects pids
    for entry in $(biquad_levels); do
        level=${entry%%:*}
        march=${entry#*:}
        march=${march//,/ }
        dir="$work/cmsisdsp-$level"
        mkdir -p "$dir"
        objects=()
        pids=()
        # The sources are independent: compile them $jobs at a time.
        for src in $CMSISDSP_FALLBACK_SOURCES; do
            $cc $BIQUAD_OPT $march $CMSISDSP_DEFINES "${inc[@]}" \
                -c "$work/cmsis/Source/$src.c" -o "$dir/${src##*/}.o" &
            pids+=($!)
            objects+=("$dir/${src##*/}.o")
            if [ ${#pids[@]} -ge "$jobs" ]; then
                wait "${pids[0]}"
                pids=("${pids[@]:1}")
            fi
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        $cc $BIQUAD_OPT $march $CMSISDSP_DEFINES "${inc[@]}" -c "$here/cmsis-dsp.c" -o "$dir/handles.o"
        $ar rcs "$prefix/lib/libcmsisdsp-$level.a" "${objects[@]}" "$dir/handles.o"
    done
}

# The flags of KFR on wasm32 beyond those of the level. KFR has no wasm32 code of its own: its
# CMake build for Emscripten sets KFR_ARCH=sse41 and compiles with `-msse4.1 -msimd128`, which
# emcc turns into the macros below and Emscripten's SSE headers (SSE intrinsics written with
# SIMD128). KFR_WASM_FLAGS repeats that with the headers of the pinned Emscripten release and the
# libc++ and wasi-libc headers of the pinned wasi-sdk sysroot. Without exceptions and RTTI the
# shim needs no library of either: it calls aligned_alloc and free of the Rust target's wasi-libc.
# KFR's generic architecture is no alternative: it does not compile in 7.2.0 (bitshuffle.hpp
# names an intr::bitpermute_pair that only the x86 and ARM code define).
KFR_WASM_FLAGS=""
prepare_kfr_wasm() {
    fetch "$WASISYSROOT_URL" "$WASISYSROOT_SHA256" "wasi-sysroot-$WASISYSROOT_VERSION.tar.gz"
    fetch "$EMSCRIPTEN_URL" "$EMSCRIPTEN_SHA256" "emscripten-$EMSCRIPTEN_VERSION.tar.gz"
    local sysroot="$work/wasi-sysroot" sse="$work/emscripten-sse"
    mkdir -p "$sysroot" "$sse"
    tar -xzf "$downloads/wasi-sysroot-$WASISYSROOT_VERSION.tar.gz" -C "$sysroot" --strip-components=1 \
        "wasi-sysroot-$WASISYSROOT_VERSION/include/wasm32-wasip1" \
        "wasi-sysroot-$WASISYSROOT_VERSION/include/c++"
    tar -xzf "$downloads/emscripten-$EMSCRIPTEN_VERSION.tar.gz" -C "$sse" --strip-components=4 \
        --wildcards "emscripten-$EMSCRIPTEN_VERSION/system/include/compat/*intrin.h"
    KFR_WASM_FLAGS="--sysroot=$sysroot -nostdinc++ -isystem $sysroot/include/wasm32-wasip1/noeh/c++/v1"
    KFR_WASM_FLAGS+=" -isystem $sysroot/include/c++/v1 -isystem $sse"
    KFR_WASM_FLAGS+=" -D__SSE__=1 -D__SSE2__=1 -D__SSE3__=1 -D__SSSE3__=1 -D__SSE4_1__=1"
}

# KFR's `iir` (header-only, see kfr-biquad.cpp), one library per level: libkfrbq-<level>.a.
build_kfr_biquad() {
    fetch "$KFR_URL" "$KFR_SHA256" "kfr-$KFR_VERSION.tar.gz"
    mkdir -p "$work/kfr"
    tar -xzf "$downloads/kfr-$KFR_VERSION.tar.gz" -C "$work/kfr" --strip-components=1 \
        "kfr-$KFR_VERSION/include"
    if [ "$platform" = wasm32 ]; then
        prepare_kfr_wasm
    fi
    local entry level march
    for entry in $(biquad_levels); do
        level=${entry%%:*}
        march=${entry#*:}
        $kfr_cxx $KFR_CXX_FLAGS $BIQUAD_OPT $march $KFR_WASM_FLAGS -I "$work/kfr/include" \
            -c "$here/kfr-biquad.cpp" -o "$work/kfr-$level.o"
        $ar rcs "$prefix/lib/libkfrbq-$level.a" "$work/kfr-$level.o"
    done
}

# liquid-dsp's halfband decimators (liquid-halfband.c) from the unpacked release, one library per
# level: libliquidhb-<level>.a. The level selects the dot product that the wrapper's `method` 1
# runs (SSE for sse2 and sse4.2, AVX for avx2, AVX-512F for avx512, NEON for neon; liquid has no
# SIMD loop for wasm32, there the portable loop compiled with -msimd128 is the reference) and
# the instruction set the library is compiled for. The portable method (the reference of the
# fallback entries) is compiled for the same level, with the same flags as the SIMD loops, and
# without contraction into FMA (BIQUAD_OPT: D5 forbids it on our side, so neither side fuses).
# liquid's own build selects the dot product at run time from the CPU; the measured CPUs are
# pinned, so the method is fixed by the level. The SSE loop of `dotprod_rrrf` (`resamp2_rrrf`
# and `firhilbf`) needs SSE3 for its horizontal add (HAVE_SSE3, which liquid's autotools build
# defines where the compiler has SSE3): it is the reference from sse4.2 (compiled with
# -DHAVE_SSE3=1) and at sse2 the portable loop is (maintainer decision of 2026-10-31, D68).
# `liquid_simd <level> <crcf|rrrf>` is the method of the wrapper's SIMD choice.
liquid_simd() {
    case $1 in
        sse2) if [ "$2" = rrrf ]; then echo LIQUID_RUNTIME_PORT; else echo LIQUID_RUNTIME_SSE; fi ;;
        sse4.2) echo LIQUID_RUNTIME_SSE ;;
        avx2) echo LIQUID_RUNTIME_AVX ;;
        avx512) echo LIQUID_RUNTIME_AVX512 ;;
        neon) echo LIQUID_RUNTIME_NEON ;;
        *) echo LIQUID_RUNTIME_PORT ;;
    esac
}

# The vendored portable dot products of the DC blocker (liquid-dotprod.c) define the functions
# below under liquid's names; the halfband library renames its own (the macros apply to the
# whole translation unit, definitions and uses alike) so that both link into one binary.
LIQUID_HB_RENAME=""
for _f in crcf rrrf; do
    for _g in create destroy execute; do
        LIQUID_HB_RENAME+=" -Ddotprod_${_f}_${_g}=sdrk_hb_dotprod_${_f}_${_g}"
    done
done

build_liquid_halfband() {
    fetch "$LIQUID_URL" "$LIQUID_SHA256" "liquid-dsp-$LIQUID_VERSION.tar.gz"
    unpack "liquid-dsp-$LIQUID_VERSION.tar.gz" liquid
    local src="$work/liquid" entry level march dir objects f simd build sse3
    for entry in $(biquad_levels); do
        level=${entry%%:*}
        march=${entry#*:}
        march=${march//,/ }
        dir="$work/liquidhb-$level"
        mkdir -p "$dir/cfg"
        build=""
        case $platform in
            x86_64) build="BUILD_SSE BUILD_AVX BUILD_AVX512" ;;
            aarch64) build="BUILD_NEON" ;;
        esac
        cp "$src/scripts/liquid.config.h" "$dir/cfg/liquid.config.h"
        for f in $build; do
            sed -i "s/^#define $f  *0/#define $f 1/" "$dir/cfg/liquid.config.h"
        done
        sse3=""
        if [ "$level" != sse2 ] && [ "$platform" = x86_64 ]; then
            sse3="-DHAVE_SSE3=1"
        fi
        local inc=(-I "$dir/cfg" -I "$src/include")
        objects=()
        for f in dotprod/src/dotprod_crcf dotprod/src/dotprod_rrrf buffer/src/buffercf buffer/src/bufferf \
            utility/src/msb_index utility/src/memory core/src/error core/src/logging core/src/runtime; do
            $cc $BIQUAD_OPT $march "${inc[@]}" -I "$src/src/buffer/src" -I "$src/src/dotprod/src" \
                -Wno-unused-result $LIQUID_HB_RENAME $([ "$f" = dotprod/src/dotprod_rrrf ] && echo "$sse3") \
                -c "$src/src/$f.c" -o "$dir/${f##*/}.o"
            objects+=("$dir/${f##*/}.o")
        done
        $cc $BIQUAD_OPT $march "${inc[@]}" -I "$src/src" -c "$src/src/libliquid.c" -o "$dir/libliquid.o"
        objects+=("$dir/libliquid.o")
        for f in 1 2 3; do
            if [ "$f" = 1 ]; then
                simd=$(liquid_simd "$level" crcf)
            else
                simd=$(liquid_simd "$level" rrrf)
            fi
            $cc $BIQUAD_OPT $march "${inc[@]}" -I "$src/src/filter/src" -DSDRK_LIQUID_KIND=$f \
                -DSDRK_LIQUID_SIMD=$simd $LIQUID_HB_RENAME -c "$here/liquid-halfband.c" -o "$dir/halfband$f.o"
            objects+=("$dir/halfband$f.o")
        done
        $ar rcs "$prefix/lib/libliquidhb-$level.a" "${objects[@]}"
    done
}

# libairspy's real-to-complex converters (airspy-iqconv.c) from the unpacked release, one
# library per level: libairspy-<level>.a with the generic float converter, the SSE2 float
# converter (x86_64 only) and the int16 converter, compiled for the level.
build_airspy() {
    fetch "$AIRSPY_URL" "$AIRSPY_SHA256" "airspy-$AIRSPY_VERSION.tar.gz"
    mkdir -p "$work/airspy"
    tar -xzf "$downloads/airspy-$AIRSPY_VERSION.tar.gz" -C "$work/airspy" --strip-components=3 \
        "airspyone_host-$AIRSPY_VERSION/libairspy/src/iqconverter_float.c" \
        "airspyone_host-$AIRSPY_VERSION/libairspy/src/iqconverter_float.h" \
        "airspyone_host-$AIRSPY_VERSION/libairspy/src/iqconverter_int16.c" \
        "airspyone_host-$AIRSPY_VERSION/libairspy/src/iqconverter_int16.h"
    local entry level march dir kinds kind objects
    kinds="1 3"
    if [ "$platform" = x86_64 ]; then
        kinds="1 2 3"
    fi
    for entry in $(biquad_levels); do
        level=${entry%%:*}
        march=${entry#*:}
        march=${march//,/ }
        dir="$work/airspy-$level"
        mkdir -p "$dir"
        objects=()
        for kind in $kinds; do
            $cc $BIQUAD_OPT $march -I "$work/airspy" -DSDRK_AIRSPY_KIND=$kind \
                -c "$here/airspy-iqconv.c" -o "$dir/iqconv$kind.o"
            objects+=("$dir/iqconv$kind.o")
        done
        $ar rcs "$prefix/lib/libairspy-$level.a" "${objects[@]}"
    done
}

KISSFFT_FLAGS=""
GR_AGC_FLAGS=""
libraries=()
case $platform in
    x86_64)
        build_volk
        build_fftw
        build_kissfft
        build_gr_agc
        build_gr_squelch
        build_gr_fallback
        build_liquid_dcblock
        build_kfr_biquad
        build_cmsis_biquad
        build_cmsis_dsp
        build_liquid_halfband
        build_airspy
        libraries=(volk fftw kissfft gragc grsquelch grfallback liquiddc kfr cmsisdsp liquidhb airspy)
        ;;
    aarch64)
        build_volk
        build_fftw
        build_kissfft
        build_ne10
        build_gr_agc
        build_gr_squelch
        build_gr_fallback
        build_liquid_dcblock
        build_kfr_biquad
        build_cmsis_biquad
        build_cmsis_dsp
        build_liquid_halfband
        build_airspy
        libraries=(volk fftw kissfft ne10 gragc grsquelch grfallback liquiddc kfr cmsisdsp liquidhb airspy)
        ;;
    wasm32)
        build_volk_wasm
        build_kissfft
        build_gr_agc
        build_gr_squelch
        build_gr_fallback
        build_liquid_dcblock
        build_kfr_biquad
        build_cmsis_biquad
        build_cmsis_dsp
        build_liquid_halfband
        build_airspy
        libraries=(volk kissfft gragc grsquelch grfallback liquiddc kfr wasisysroot emscripten cmsisdsp liquidhb airspy)
        ;;
    cortex-m4)
        build_cmsis_biquad
        libraries=(cmsisdsp cmsiscore)
        ;;
esac

# The manifest that xtask records in benches/results/<platform>.json.
{
    echo "{"
    echo "  \"platform\": \"$platform\","
    echo "  \"cc\": \"$cc_version\","
    echo "  \"cxx\": \"$cxx_version\","
    echo "  \"kfr_cxx\": \"$kfr_cxx_version\","
    echo "  \"libraries\": {"
    sep=""
    for lib in "${libraries[@]}"; do
        case $lib in
            volk) entry="\"volk\": { \"version\": \"$VOLK_VERSION\", \"sha256\": \"$VOLK_SHA256\", \"build\": \"$([ "$platform" = wasm32 ] && echo "generic protokernels, -O3 -msimd128" || echo "CMake Release, ENABLE_ORC=OFF")\" }" ;;
            fftw) entry="\"fftw\": { \"version\": \"$FFTW_VERSION\", \"sha256\": \"$FFTW_SHA256\", \"build\": \"--enable-float $([ "$platform" = aarch64 ] && echo --enable-neon || echo --enable-sse2), CFLAGS=$OPT_FLAGS\" }" ;;
            kissfft) entry="\"kissfft\": { \"version\": \"$KISSFFT_VERSION\", \"sha256\": \"$KISSFFT_SHA256\", \"build\": \"$KISSFFT_FLAGS\" }" ;;
            gragc) entry="\"gnuradio_agc_cc\": { \"version\": \"main at ec1bf15\", \"sha256\": \"$(sha256sum "$here/gr-agc.c" | cut -d ' ' -f 1)\", \"build\": \"$GR_AGC_FLAGS\" }" ;;
            grsquelch) entry="\"gnuradio_pwr_squelch_cc\": { \"version\": \"main at ec1bf15\", \"sha256\": \"$(sha256sum "$here/gr-squelch.c" | cut -d ' ' -f 1)\", \"build\": \"$GR_AGC_FLAGS\" }" ;;
            grfallback) entry="\"gnuradio_fallback_references\": { \"version\": \"main at ec1bf15\", \"sha256\": \"$(cat "$here/gr-rotator.c" "$here/gr-fm-mod.c" "$here/gr-quad-demod.c" "$here/gr-fast-atan2f.c" "$here/gr-deemph.c" | sha256sum | cut -d ' ' -f 1)\", \"build\": \"gr-rotator.c gr-fm-mod.c gr-quad-demod.c gr-fast-atan2f.c gr-deemph.c (VOLK $VOLK_VERSION generic multiply_conjugate): $GR_AGC_FLAGS\" }" ;;
            liquiddc) entry="\"liquid_dsp_dc_blocker\": { \"version\": \"$LIQUID_VERSION\", \"sha256\": \"$(cat "$here/liquid-dotprod.c" "$here/liquid-dcblock.c" | sha256sum | cut -d ' ' -f 1)\", \"build\": \"liquid-dotprod.c liquid-dcblock.c (portable dot products): $GR_AGC_FLAGS\" }" ;;
            liquidhb) entry="\"liquid_dsp_halfband\": { \"version\": \"$LIQUID_VERSION\", \"sha256\": \"$LIQUID_SHA256\", \"build\": \"liquid-halfband.c over resamp2.proto.c, firhilb.proto.c and the dot products of the release: $BIQUAD_OPT, per level $(biquad_levels | tr , ' '), dot products (crcf / rrrf) $(for e in $(biquad_levels); do printf '%s=%s/%s ' "${e%%:*}" "$(liquid_simd "${e%%:*}" crcf)" "$(liquid_simd "${e%%:*}" rrrf)"; done)\" }" ;;
            airspy) entry="\"libairspy\": { \"version\": \"$AIRSPY_VERSION\", \"sha256\": \"$AIRSPY_SHA256\", \"build\": \"airspy-iqconv.c over iqconverter_float.c and iqconverter_int16.c (conversion without remove_dc; the SSE2 converter with the FreeBSD branch, x86_64 only): $BIQUAD_OPT, per level $(biquad_levels | tr , ' ')\" }" ;;
            kfr) entry="\"kfr\": { \"version\": \"$KFR_VERSION\", \"sha256\": \"$KFR_SHA256\", \"build\": \"kfr-biquad.cpp: ${kfr_cxx} $KFR_CXX_FLAGS $BIQUAD_OPT$([ "$platform" = wasm32 ] && echo ", KFR_ARCH sse41 as emcc -msse4.1 (wasi-sdk sysroot and the SSE headers of Emscripten)"), per level $(biquad_levels)\" }" ;;
            wasisysroot) entry="\"wasi_sysroot\": { \"version\": \"$WASISYSROOT_VERSION\", \"sha256\": \"$WASISYSROOT_SHA256\", \"build\": \"headers of libc++ and wasi-libc for KFR\" }" ;;
            emscripten) entry="\"emscripten\": { \"version\": \"$EMSCRIPTEN_VERSION\", \"sha256\": \"$EMSCRIPTEN_SHA256\", \"build\": \"SSE headers of system/include/compat for KFR\" }" ;;
            cmsisdsp) entry="\"cmsis_dsp\": { \"version\": \"$CMSISDSP_VERSION\", \"sha256\": \"$CMSISDSP_SHA256\", \"build\": \"$([ "$platform" = cortex-m4 ] && echo "CMSIS-Core, DSP extension and FPU, cmsis-biquad.c and cmsis-m4-heap.c" || echo "generic C with cmsis-biquad.c and cmsis-dsp.c"): $CMSISDSP_DEFINES $BIQUAD_OPT, per level $(biquad_levels | tr , ' ')\" }" ;;
            cmsiscore) entry="\"cmsis_core\": { \"version\": \"$CMSISCORE_VERSION\", \"sha256\": \"$CMSISCORE_SHA256\", \"build\": \"headers of CMSIS/Core/Include\" }" ;;
            ne10) entry="\"ne10\": { \"version\": \"$NE10_COMMIT\", \"sha256\": \"$NE10_SHA256\", \"build\": \"CMake Release, DSP module\" }" ;;
        esac
        printf '%s    %s' "$sep" "$entry"
        sep=$',\n'
    done
    echo
    echo "  }"
    echo "}"
} >"$prefix/refs.json"
rm -rf "$work"
echo "$want" >"$stamp"
echo "==> C references for $platform ready"
