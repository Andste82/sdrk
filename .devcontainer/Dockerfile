# Development image for SDRk: everything the milestones M00-M16 need on a Linux
# host (Rust with all targets, Python golden generator, emulators, instruction
# counting, WASM tooling). Build context: the repository root.
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH

# Base tools, C toolchain, Python for tests/golden/generate.py (M04), emulators
# (M13/M15: QEMU user mode for aarch64/riscv, system mode for Arm and RISC-V),
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

# Cargo tools. Pin versions with --version when a milestone fixes them
# (wasm-bindgen-cli must match the wasm-bindgen crate version of M14).
RUN cargo install --locked cargo-deny wasm-bindgen-cli wasmtime-cli cbindgen iai-callgrind-runner \
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
# python3.12 runs.
COPY tests/golden/requirements.txt /tmp/golden-requirements.txt
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv --python 3.12 /opt/sdrk-golden-venv \
    && VIRTUAL_ENV=/opt/sdrk-golden-venv uv pip install -r /tmp/golden-requirements.txt \
    && printf '#!/bin/sh\nexec /opt/sdrk-golden-venv/bin/python "$@"\n' > /usr/local/bin/python3.12 \
    && chmod +x /usr/local/bin/python3.12 \
    && rm /tmp/golden-requirements.txt

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
