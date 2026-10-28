# Development image for SDRk: everything the milestones M00-M16 need on a Linux
# host (Rust with all targets, Python golden generator, emulators, instruction
# counting, cycle simulation, WASM tooling). Build context: the repository root.
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH

# Base tools, C toolchain, Python for tests/golden/generate.py (M04), emulators
# (M13/M15: QEMU user mode for aarch64/riscv, system mode for Arm and RISC-V;
# since Ubuntu 26.04 qemu-system-riscv32 is in its own package qemu-system-riscv),
# valgrind (callgrind, M13), libvolk (M13 comparison via volk-sys), binaryen
# (wasm-opt, M14), cross tools for objdump/size, and what benches/refs/build.sh
# needs for the C references (g++ and the libc for aarch64, wasi-libc and
# llvm-ar for wasm32, the newlib headers for CMSIS-DSP on the Cortex-M4, mako
# for VOLK).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    openssh-client \
    sudo \
    xz-utils \
    unzip \
    ripgrep \
    fd-find \
    xxd \
    jq \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    ninja-build \
    clang \
    lld \
    gdb-multiarch \
    valgrind \
    libvolk-dev \
    binaryen \
    python3 \
    python3-venv \
    python3-pip \
    qemu-user \
    qemu-system-arm \
    qemu-system-misc \
    qemu-system-riscv \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    wasi-libc \
    llvm \
    python3-mako \
    gcc-arm-none-eabi \
    libnewlib-arm-none-eabi \
    binutils-arm-none-eabi \
    binutils-riscv64-unknown-elf \
    libglib2.0-dev \
    libpixman-1-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust: stable (rust-toolchain.toml selects it), plus nightly for Miri (M13).
ARG RUST_VERSION=stable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain $RUST_VERSION --profile minimal \
    && rustup component add rustfmt clippy rust-src llvm-tools-preview \
    # thumbv7em-none-eabihf: CI bare-metal build and Cortex-M4 (M00, M15)
    # thumbv8m.main-none-eabihf: Cortex-M55 (M15, backlog Helium)
    # riscv32imac/imafc-unknown-none-elf: RISC-V QEMU and ESP32-P4 (M15, M16)
    # wasm32-unknown-unknown / wasm32-wasip1: bindings and wasmtime tests (M13, M14)
    # aarch64-unknown-linux-gnu: NEON under qemu-aarch64 (M13)
    && rustup target add \
        thumbv7em-none-eabihf \
        thumbv8m.main-none-eabihf \
        riscv32imac-unknown-none-elf \
        riscv32imafc-unknown-none-elf \
        wasm32-unknown-unknown \
        wasm32-wasip1 \
        aarch64-unknown-linux-gnu \
    && rustup toolchain install nightly --profile minimal --component miri --component rust-src

# Cargo tools. Pin versions with --version when a milestone fixes them:
# wasm-bindgen-cli must equal the `wasm-bindgen = "=..."` of bindings/wasm (M14),
# cargo-nextest the NEXTEST_VERSION of .github/workflows/tree.yml (the test
# commands of docs/testing.md use it); cargo-fuzz runs the nightly fuzz job.
ARG WASM_BINDGEN_VERSION=0.2.129
ARG NEXTEST_VERSION=0.9.148
RUN cargo install --locked cargo-deny wasmtime-cli cbindgen iai-callgrind-runner cargo-fuzz \
    && cargo install --locked wasm-bindgen-cli --version $WASM_BINDGEN_VERSION \
    && cargo install --locked cargo-nextest --version $NEXTEST_VERSION \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git

# Instruction counting (cargo xtask count, docs/performance.md), pinned to the
# versions and checksums of .github/workflows/tree.yml:
# - Intel SDE for the x86_64 levels sse2, sse4.2 and avx512 (proprietary
#   freeware of Intel, downloaded at build time, not part of the repository).
# - QEMU user mode for aarch64 built with --enable-plugins (the packaged
#   qemu-aarch64 has no plugin support); xtask builds the insn plugin itself.
ARG SDE_URL=https://downloadmirror.intel.com/859732/sde-external-9.58.0-2025-06-16-lin.tar.xz
ARG SDE_SHA256=f849acecad4c9b108259c643b2688fd65c35723cd23368abe5dd64b917cc18c0
ARG SDE_DIR=sde-external-9.58.0-2025-06-16-lin
RUN curl -sfL -o /tmp/sde.tar.xz "$SDE_URL" \
    && echo "$SDE_SHA256  /tmp/sde.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/sde.tar.xz -C /opt \
    && ln -s /opt/$SDE_DIR/sde64 /usr/local/bin/sde64 \
    && rm /tmp/sde.tar.xz

ARG QEMU_PLUGINS_VERSION=10.2.1
ARG QEMU_PLUGINS_SHA256=a3717477d8e2c84d630bfffbc20f6cd3293eb45aa1e6dac6d0cc27689991c9e1
RUN curl -sfL -o /tmp/qemu.tar.xz https://download.qemu.org/qemu-$QEMU_PLUGINS_VERSION.tar.xz \
    && echo "$QEMU_PLUGINS_SHA256  /tmp/qemu.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/qemu.tar.xz -C /tmp \
    && cd /tmp/qemu-$QEMU_PLUGINS_VERSION \
    && ./configure --target-list=aarch64-linux-user --enable-plugins --disable-docs \
    && ninja -C build qemu-aarch64 \
    && install -D build/qemu-aarch64 /opt/qemu-plugins/bin/qemu-aarch64 \
    && cd / && rm -rf /tmp/qemu.tar.xz /tmp/qemu-$QEMU_PLUGINS_VERSION
ENV SDRK_SDE=/usr/local/bin/sde64
ENV SDRK_QEMU_AARCH64=/opt/qemu-plugins/bin/qemu-aarch64

# Golden generator: byte-identical output needs Python 3.12 with the pinned
# NumPy and SciPy of tests/golden/requirements.txt (tests/golden/README.md);
# Ubuntu 26.04 ships a newer Python, so uv installs 3.12 into a venv that
# python3.12 runs. uv is installed here; the venv of the cycle-simulation tools
# below uses it too.
COPY tests/golden/requirements.txt /tmp/golden-requirements.txt
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv --python 3.12 /opt/sdrk-golden-venv \
    && VIRTUAL_ENV=/opt/sdrk-golden-venv uv pip install -r /tmp/golden-requirements.txt \
    && printf '#!/bin/sh\nexec /opt/sdrk-golden-venv/bin/python "$@"\n' > /usr/local/bin/python3.12 \
    && chmod +x /usr/local/bin/python3.12 \
    && rm /tmp/golden-requirements.txt

# Simulated cycles per sample (level 2 of the measurement policy, D33 in
# docs/decisions-log.md, docs/performance.md#measurement-policy):
# - llvm-mca of LLVM 23.1.1 (the LLVM of the Rust toolchain at the time of the
#   decision), built from the release source with the X86 and AArch64 targets
#   only; the Ubuntu llvm package has another version and must not be used for
#   the published ratios. The checksum is the one of the release attestation.
ARG LLVM_MCA_VERSION=23.1.1
ARG LLVM_MCA_SHA256=ebe9be46fe8756d58c5b198ffad0fa2a766257add81a4dc52179bfacc7888ee6
RUN curl -sfL -o /tmp/llvm.tar.xz \
       https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_MCA_VERSION/llvm-project-$LLVM_MCA_VERSION.src.tar.xz \
    && echo "$LLVM_MCA_SHA256  /tmp/llvm.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/llvm.tar.xz -C /tmp \
       llvm-project-$LLVM_MCA_VERSION.src/llvm \
       llvm-project-$LLVM_MCA_VERSION.src/cmake \
       llvm-project-$LLVM_MCA_VERSION.src/third-party \
       llvm-project-$LLVM_MCA_VERSION.src/libc \
    && cmake -S /tmp/llvm-project-$LLVM_MCA_VERSION.src/llvm -B /tmp/llvm-build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
       -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
       -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF \
       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld \
    && ninja -C /tmp/llvm-build llvm-mca \
    && install -D /tmp/llvm-build/bin/llvm-mca /opt/llvm-mca-$LLVM_MCA_VERSION/bin/llvm-mca \
    && rm -rf /tmp/llvm.tar.xz /tmp/llvm-project-$LLVM_MCA_VERSION.src /tmp/llvm-build
ENV SDRK_LLVM_MCA=/opt/llvm-mca-23.1.1/bin/llvm-mca

# - Second opinions where a decision depends on a single result (D33): OSACA
#   (AGPL-3.0; Zen, Intel, Neoverse N1/V2, Cortex-A72) from PyPI, and uiCA
#   (AGPL-3.0; Intel cores) at a pinned commit. Both are development tools that
#   only read assembly; nothing of them is linked into SDRk. uiCA's setup
#   downloads the current instructions.xml of uops.info, which has no version;
#   its SHA-256 is kept next to the data in /opt/uiCA/instructions.xml.sha256.
ARG OSACA_VERSION=0.7.1
ARG UICA_COMMIT=6ebeeab0f8705cae4b17a9362f9c65ef91f5cc44
RUN UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv --python 3.12 /opt/sdrk-perf-venv \
    && VIRTUAL_ENV=/opt/sdrk-perf-venv uv pip install osaca==$OSACA_VERSION plotly setuptools \
    && git clone https://github.com/andreas-abel/uiCA.git /opt/uiCA \
    && cd /opt/uiCA \
    && git checkout $UICA_COMMIT \
    && git submodule update --init \
    && cd XED-to-XML && /opt/sdrk-perf-venv/bin/python ./mfile.py --opt=2 --no-encoder pymodule \
    && cp xed.* .. && cd .. \
    && git submodule deinit -f --all && rm -rf .git/modules/* \
    && curl -sfL -o instructions.xml https://www.uops.info/instructions.xml \
    && sha256sum instructions.xml > instructions.xml.sha256 \
    && /opt/sdrk-perf-venv/bin/python ./convertXML.py instructions.xml \
    && rm instructions.xml \
    && printf '#!/bin/sh\nexec /opt/sdrk-perf-venv/bin/python /opt/uiCA/uiCA.py "$@"\n' > /usr/local/bin/uiCA \
    && chmod +x /usr/local/bin/uiCA \
    && ln -s /opt/sdrk-perf-venv/bin/osaca /usr/local/bin/osaca
ENV SDRK_OSACA=/usr/local/bin/osaca
ENV SDRK_UICA=/usr/local/bin/uiCA

# Not installed here, because the milestones download them pinned and
# checksum-verified through xtask: esp-emulator and the Espressif QEMU fork
# (M15), espup / espflash / ESP-IDF (M15, M16), Chrome and Firefox with their
# drivers for wasm-bindgen-test (M14, CI action).

# Node.js (LTS): WASM package tests (M14) and the coding assistant CLI
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Coding assistant CLI
RUN npm install -g @anthropic-ai/claude-code

# GitHub CLI (gh) from GitHub's official apt repository
RUN mkdir -p -m 755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
