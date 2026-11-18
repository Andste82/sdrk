/*
 * GNU Radio's multiplicative scrambler and descrambler `gr::digital::scrambler_bb` and
 * `gr::digital::descrambler_bb` (gr-digital/lib/scrambler_bb_impl.cc,
 * gr-digital/lib/descrambler_bb_impl.cc and the shift register
 * gr-digital/include/gnuradio/digital/lfsr.h, GNU Radio main at ec1bf15), translated from C++
 * to C member for member:
 *
 * - the members of `gr::digital::lfsr`: `d_shift_register`, `d_mask`, `d_seed` and
 *   `d_shift_register_length` (`uint64_t`, `uint8_t`);
 * - `next_bit_scramble(input)` and `next_bit_descramble(input)` on the path GNU Radio builds
 *   with a compiler that has `__builtin_parityl` (`HAVE_BUILTIN_PARITYL`, set by its CMake for
 *   GCC and Clang);
 * - the `work` loops of `scrambler_bb_impl` and `descrambler_bb_impl`: one call of the shift
 *   register per item of unpacked bits (one bit per byte, the LSB).
 *
 * Left out: the exception of the constructor for a register longer than 63 bits (the init
 * function returns -1 instead), `next_bit`, `pre_shift` and the GNU Radio block machinery.
 *
 * Copyright 2008,2010,2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <stdint.h>

/* gr::digital::lfsr */
typedef struct {
    uint64_t shift_register; /* d_shift_register */
    uint64_t mask;           /* d_mask */
    uint64_t seed;           /* d_seed */
    uint8_t length;          /* d_shift_register_length */
} sdrk_gr_lfsr;

unsigned long sdrk_gr_lfsr_size(void) { return sizeof(sdrk_gr_lfsr); }

/* lfsr::lfsr(mask, seed, reg_len) */
int sdrk_gr_lfsr_init(sdrk_gr_lfsr* self, uint64_t mask, uint64_t seed, uint8_t reg_len)
{
    if (reg_len > 63) {
        return -1;
    }
    self->shift_register = seed;
    self->mask = mask;
    self->seed = seed;
    self->length = reg_len;
    return 0;
}

/* lfsr::reset */
void sdrk_gr_lfsr_reset(sdrk_gr_lfsr* self) { self->shift_register = self->seed; }

/* lfsr::next_bit_scramble */
static inline unsigned char next_bit_scramble(sdrk_gr_lfsr* self, unsigned char input)
{
    unsigned char output = self->shift_register & 1;
    uint64_t newbit;
    newbit = __builtin_parityl(self->shift_register & self->mask) ^ (input & 1);
    self->shift_register = ((self->shift_register >> 1) | (newbit << self->length));
    return output;
}

/* lfsr::next_bit_descramble */
static inline unsigned char next_bit_descramble(sdrk_gr_lfsr* self, unsigned char input)
{
    unsigned char output;
    output = __builtin_parityl(self->shift_register & self->mask) ^ (input & 1);
    uint64_t newbit = input & 1;
    self->shift_register = ((self->shift_register >> 1) | (newbit << self->length));
    return output;
}

/* scrambler_bb_impl::work */
int sdrk_gr_scrambler_bb_work(sdrk_gr_lfsr* self,
                              int noutput_items,
                              const unsigned char* in,
                              unsigned char* out)
{
    for (int i = 0; i < noutput_items; i++) {
        out[i] = next_bit_scramble(self, in[i]);
    }
    return noutput_items;
}

/* descrambler_bb_impl::work */
int sdrk_gr_descrambler_bb_work(sdrk_gr_lfsr* self,
                                int noutput_items,
                                const unsigned char* in,
                                unsigned char* out)
{
    for (int i = 0; i < noutput_items; i++) {
        out[i] = next_bit_descramble(self, in[i]);
    }
    return noutput_items;
}
