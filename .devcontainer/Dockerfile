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
# (wasm-opt, M14) and cross tools for objdump/size.
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
    gcc-arm-none-eabi \
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

# Not installed here, because the milestones download them pinned and
# checksum-verified through xtask: Intel SDE (M13), esp-emulator and the
# Espressif QEMU fork (M15), espup / espflash / ESP-IDF (M15, M16),
# Chrome and Firefox with their drivers for wasm-bindgen-test (M14, CI action).

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
