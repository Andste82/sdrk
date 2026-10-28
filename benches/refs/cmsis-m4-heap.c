/*
 * The allocator of the CMSIS-DSP handles (cmsis-biquad.c) on the Cortex-M4 of the report, which
 * runs without a C library: malloc hands out pieces of one static arena, free keeps them. The
 * count workload (targets/qemu-cortex-m4, binary `count`) creates at most two handles per run
 * of a few dozen bytes each and the board is reset between runs, so the arena never fills.
 * Only the constructors call it, never the counted filter calls.
 */
#include <stddef.h>
#include <stdint.h>

static uint64_t arena[512];
static size_t used;

void* malloc(size_t size)
{
    size_t words = (size + sizeof(uint64_t) - 1u) / sizeof(uint64_t);
    if (words > sizeof(arena) / sizeof(arena[0]) - used)
        return NULL;
    void* p = &arena[used];
    used += words;
    return p;
}

void free(void* p)
{
    (void)p;
}
