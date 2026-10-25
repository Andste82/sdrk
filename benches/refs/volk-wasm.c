/*
 * VOLK for wasm32: the portable `generic` protokernels of VOLK compiled for wasm32 with
 * simd128, exported under the names of VOLK's dispatcher pointers and of the functions of its
 * public API that benches/volk-sys uses, so that the measurement binary links unchanged.
 *
 * VOLK has no wasm32 machine. On a machine without SIMD its dispatcher ranks every kernel's
 * implementations by their dependencies and takes the first one with the highest rank, which
 * is `generic` for every kernel below (atan2 has `polynomial` with the same rank after it), so
 * this file reproduces VOLK's default dispatch there. Built by benches/refs/build.sh.
 */
#define LV_HAVE_GENERIC 1

#include <stdbool.h>
#include <stddef.h>

#include <volk/volk_complex.h>

#include <volk/volk_16i_s32f_convert_32f.h>
#include <volk/volk_16ic_magnitude_16i.h>
#include <volk/volk_32f_s32f_convert_16i.h>
#include <volk/volk_32f_s32f_convert_8i.h>
#include <volk/volk_32f_s32f_x2_convert_8u.h>
#include <volk/volk_32fc_32f_dot_prod_32fc.h>
#include <volk/volk_32fc_magnitude_32f.h>
#include <volk/volk_32fc_magnitude_squared_32f.h>
#include <volk/volk_32fc_s32f_atan2_32f.h>
#include <volk/volk_32fc_s32fc_x2_rotator2_32fc.h>
#include <volk/volk_32fc_x2_multiply_conjugate_32fc.h>
#include <volk/volk_8i_s32f_convert_32f.h>

/* VOLK's `volk_func_desc_t`: the implementations of a kernel on the selected machine. */
typedef struct {
    const char** impl_names;
    const int* impl_deps;
    const bool* impl_alignment;
    size_t n_impls;
} volk_func_desc_t;

static const char* generic_names[] = { "generic" };
static const int generic_deps[] = { 1 };
static const bool generic_alignment[] = { false };

/* The dispatcher pointer `name` and its aligned variant `name_a`, both set to the generic
   implementation, and the description with that single implementation. */
#define GENERIC_KERNEL(name)                                                                  \
    __typeof__(&name##_generic) name = name##_generic;                                        \
    __typeof__(&name##_generic) name##_a = name##_generic;                                    \
    volk_func_desc_t name##_get_func_desc(void)                                               \
    {                                                                                         \
        volk_func_desc_t desc = { generic_names, generic_deps, generic_alignment, 1 };        \
        return desc;                                                                          \
    }

GENERIC_KERNEL(volk_8i_s32f_convert_32f)
GENERIC_KERNEL(volk_16i_s32f_convert_32f)
GENERIC_KERNEL(volk_32f_s32f_convert_16i)
GENERIC_KERNEL(volk_32f_s32f_convert_8i)
GENERIC_KERNEL(volk_32f_s32f_x2_convert_8u)
GENERIC_KERNEL(volk_32fc_s32fc_x2_rotator2_32fc)
GENERIC_KERNEL(volk_32fc_32f_dot_prod_32fc)
GENERIC_KERNEL(volk_32fc_magnitude_32f)
GENERIC_KERNEL(volk_32fc_magnitude_squared_32f)
GENERIC_KERNEL(volk_16ic_magnitude_16i)
GENERIC_KERNEL(volk_32fc_x2_multiply_conjugate_32fc)
GENERIC_KERNEL(volk_32fc_s32f_atan2_32f)

/* The machine `generic` has no SIMD architecture and therefore an alignment of 1. */
size_t volk_get_alignment(void) { return 1; }

const char* volk_get_machine(void) { return "generic (wasm32)"; }

const char* volk_version(void) { return SDRK_VOLK_VERSION; }

const char* volk_compiler_flags(void) { return SDRK_VOLK_FLAGS; }
