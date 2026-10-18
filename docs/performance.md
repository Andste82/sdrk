# Performance

Performance is measured in executed instructions per sample in an emulator and compared with an established library (VOLK, rustfft, CMSIS-DSP, ESP-DSP) at equal accuracy against the f64 reference.

| Instructions per sample vs. reference library | Meaning |
| --- | --- |
| ≤ 1.00× | goal: equal or better |
| ≤ 1.10× | accepted |
| ≤ 1.15× | hard limit; CI fails above it |
| ≤ 1.50× | allowed only while a kernel is under development, never in a release |

Instruction counts are a strong proxy on in-order cores (Cortex-M, ESP32) and a weaker one on out-of-order CPUs and for memory-bound kernels. The measurement setup is filled in by M13.
