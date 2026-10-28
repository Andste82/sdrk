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
# | VOLK | 3.3.0 | LGPL-3.0-or-later | x86_64, aarch64 (shared, CMake Release, no ORC); wasm32 (the `generic` protokernels, static, see volk-wasm.c) | the reference of the streaming kernels |
# | FFTW | 3.3.10 | GPL-2.0-or-later | x86_64 (`--enable-sse2`), aarch64 (`--enable-neon`); static, single precision | the C32 FFT reference at `sse2`, information on `neon` |
# | KissFFT | 131.1.0 | BSD-3-Clause | all (static, `FIXED_POINT=16`) | the CQ15 FFT reference |
# | Ne10 | 1f059a7 (last commit of the archived repository) | BSD-3-Clause | aarch64 (static, DSP module) | information next to KissFFT on `neon` |
# | GNU Radio `agc_cc` | main at ec1bf15 (agc.h of gr-analog), see gr-agc.c | GPL-3.0-or-later | all (static, one object file) | the reference of the `C32` `agc` |
# | KFR | 7.2.0 | GPL-2.0-or-later | x86_64 (one library per level), aarch64 (clang++); the header-only `kfr::iir`, see kfr-biquad.cpp | the SIMD reference of `biquad` and `biquad_real` (D15) |
# | CMSIS-DSP | 1.18.0 | Apache-2.0 | x86_64, aarch64, wasm32 (one library per level), generic C; cortex-m4 with the DSP extension and the FPU; see cmsis-biquad.c | the scalar reference of the biquad fallbacks (D16); the gate of `biquad` and `biquad_real` at `simd128` (D15); the comparison of the Cortex-M4 report (D27) |
# | CMSIS 6 (CMSIS-Core) | 6.3.0 | Apache-2.0 | cortex-m4 (headers only: `cmsis_compiler.h` and the intrinsics such as `__SMLALD` that CMSIS-DSP uses with the DSP extension) | needed by CMSIS-DSP outside its host configuration |
#
# The libraries are linked only into the measurement binary of `benches/kernels`, which is
# built on the measuring machine and never distributed; none of them is a dependency of a
# library crate of SDRk.
#
# Compilers: x86_64 `${CC:-gcc}`/`${CXX:-g++}`, aarch64 `aarch64-linux-gnu-gcc`/`-g++` (KFR with
# `${KFR_AARCH64_CXX:-clang++} --target=aarch64-linux-gnu`: GCC rejects KFR's NEON code), wasm32
# `${WASM_CC:-clang}` with the wasi-libc headers, cortex-m4 `${M4_CC:-arm-none-eabi-gcc}` with the
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
        kfr_cxx=""
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
want=$( (cat "${BASH_SOURCE[0]}" "$here/volk-wasm.c" "$here/gr-agc.c" "$here/kfr-biquad.cpp" "$here/cmsis-biquad.c" "$here/cmsis-m4-heap.c"; echo "$platform|$cc_version|$cxx_version|$kfr_cxx_version") | sha256sum | cut -d ' ' -f 1)
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

# KFR's `iir` (header-only, see kfr-biquad.cpp), one library per level: libkfrbq-<level>.a.
build_kfr_biquad() {
    fetch "$KFR_URL" "$KFR_SHA256" "kfr-$KFR_VERSION.tar.gz"
    mkdir -p "$work/kfr"
    tar -xzf "$downloads/kfr-$KFR_VERSION.tar.gz" -C "$work/kfr" --strip-components=1 \
        "kfr-$KFR_VERSION/include"
    local entry level march
    for entry in $(biquad_levels); do
        level=${entry%%:*}
        march=${entry#*:}
        $kfr_cxx $KFR_CXX_FLAGS $BIQUAD_OPT $march -I "$work/kfr/include" \
            -c "$here/kfr-biquad.cpp" -o "$work/kfr-$level.o"
        $ar rcs "$prefix/lib/libkfrbq-$level.a" "$work/kfr-$level.o"
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
        build_kfr_biquad
        build_cmsis_biquad
        libraries=(volk fftw kissfft gragc kfr cmsisdsp)
        ;;
    aarch64)
        build_volk
        build_fftw
        build_kissfft
        build_ne10
        build_gr_agc
        build_kfr_biquad
        build_cmsis_biquad
        libraries=(volk fftw kissfft ne10 gragc kfr cmsisdsp)
        ;;
    wasm32)
        build_volk_wasm
        build_kissfft
        build_gr_agc
        build_cmsis_biquad
        libraries=(volk kissfft gragc cmsisdsp)
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
            kfr) entry="\"kfr\": { \"version\": \"$KFR_VERSION\", \"sha256\": \"$KFR_SHA256\", \"build\": \"kfr-biquad.cpp: ${kfr_cxx} $KFR_CXX_FLAGS $BIQUAD_OPT, per level $(biquad_levels)\" }" ;;
            cmsisdsp) entry="\"cmsis_dsp\": { \"version\": \"$CMSISDSP_VERSION\", \"sha256\": \"$CMSISDSP_SHA256\", \"build\": \"$([ "$platform" = cortex-m4 ] && echo "CMSIS-Core, DSP extension and FPU, cmsis-biquad.c and cmsis-m4-heap.c" || echo "generic C with cmsis-biquad.c"): $CMSISDSP_DEFINES $BIQUAD_OPT, per level $(biquad_levels | tr , ' ')\" }" ;;
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
