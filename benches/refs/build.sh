#!/usr/bin/env bash
# Builds the C reference libraries of `cargo xtask count` from pinned release tarballs.
#
#   benches/refs/build.sh <x86_64|aarch64|wasm32> <prefix>
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
#
# The libraries are linked only into the measurement binary of `benches/kernels`, which is
# built on the measuring machine and never distributed; none of them is a dependency of a
# library crate of SDRk.
#
# Compilers: x86_64 `${CC:-gcc}`/`${CXX:-g++}`, aarch64 `aarch64-linux-gnu-gcc`/`-g++`, wasm32
# `${WASM_CC:-clang}` with the wasi-libc headers. Their versions are part of the stamp and of
# <prefix>/refs.json, which xtask copies into the results; the count jobs of CI pin them with
# a container image and an apt snapshot (see .github/workflows/ci.yml).
#
# Needs curl, sha256sum, tar, make, cmake, ninja and python3 with mako (VOLK), and for the
# platforms aarch64 and wasm32 the cross compilers above. Environment: SDRK_REFS_DOWNLOADS
# (download cache, default <prefix>/../downloads), SDRK_REFS_JOBS (default 2).
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <x86_64|aarch64|wasm32> <prefix>" >&2
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

# The optimisation flags of the libraries built here without their own build system or with a
# build system that would tune for the build machine (FFTW adds -mtune=native otherwise).
OPT_FLAGS="-O3 -fomit-frame-pointer -fstrict-aliasing"

case $platform in
    x86_64)
        cc=${CC:-gcc}
        cxx=${CXX:-g++}
        ar=${AR:-ar}
        ;;
    aarch64)
        cc=aarch64-linux-gnu-gcc
        cxx=aarch64-linux-gnu-g++
        ar=aarch64-linux-gnu-ar
        ;;
    wasm32)
        cc="${WASM_CC:-clang} --target=wasm32-wasip1"
        cxx=""
        ar=${WASM_AR:-llvm-ar}
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

stamp="$prefix/refs.stamp"
want=$( (cat "${BASH_SOURCE[0]}" "$here/volk-wasm.c"; echo "$platform|$cc_version|$cxx_version") | sha256sum | cut -d ' ' -f 1)
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

KISSFFT_FLAGS=""
libraries=()
case $platform in
    x86_64)
        build_volk
        build_fftw
        build_kissfft
        libraries=(volk fftw kissfft)
        ;;
    aarch64)
        build_volk
        build_fftw
        build_kissfft
        build_ne10
        libraries=(volk fftw kissfft ne10)
        ;;
    wasm32)
        build_volk_wasm
        build_kissfft
        libraries=(volk kissfft)
        ;;
esac

# The manifest that xtask records in benches/results/<platform>.json.
{
    echo "{"
    echo "  \"platform\": \"$platform\","
    echo "  \"cc\": \"$cc_version\","
    echo "  \"cxx\": \"$cxx_version\","
    echo "  \"libraries\": {"
    sep=""
    for lib in "${libraries[@]}"; do
        case $lib in
            volk) entry="\"volk\": { \"version\": \"$VOLK_VERSION\", \"sha256\": \"$VOLK_SHA256\", \"build\": \"$([ "$platform" = wasm32 ] && echo "generic protokernels, -O3 -msimd128" || echo "CMake Release, ENABLE_ORC=OFF")\" }" ;;
            fftw) entry="\"fftw\": { \"version\": \"$FFTW_VERSION\", \"sha256\": \"$FFTW_SHA256\", \"build\": \"--enable-float $([ "$platform" = aarch64 ] && echo --enable-neon || echo --enable-sse2), CFLAGS=$OPT_FLAGS\" }" ;;
            kissfft) entry="\"kissfft\": { \"version\": \"$KISSFFT_VERSION\", \"sha256\": \"$KISSFFT_SHA256\", \"build\": \"$KISSFFT_FLAGS\" }" ;;
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
