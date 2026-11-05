/*
 * GNU Radio's generic CRC `gr::digital::crc` (gr-digital/lib/crc.cc and
 * gr-digital/include/gnuradio/digital/crc.h, GNU Radio main at ec1bf15), translated from C++
 * to C member for member:
 *
 * - the members `d_table` (`std::array<uint64_t, 256>`), `d_num_bits`, `d_mask`,
 *   `d_initial_value`, `d_final_xor`, `d_input_reflected` and `d_result_reflected`;
 * - the constructor builds the 256-entry table of the polynomial (reflected or MSB first) with
 *   the same loops;
 * - `compute(data, len)` is the byte-by-byte table loop with `uint64_t` remainders, the
 *   `reflect` of the result when the input and result reflection differ, then the final xor.
 *
 * Left out: the exception of the constructor for a width outside 8 to 64 bits (the init
 * function returns -1 instead) and the `std::vector` overload of `compute`.
 *
 * Copyright 2022 Daniel Estevez <daniel@destevez.net>
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* gr::digital::crc */
typedef struct {
    uint64_t table[256];      /* d_table */
    unsigned num_bits;        /* d_num_bits */
    uint64_t mask;            /* d_mask */
    uint64_t initial_value;   /* d_initial_value */
    uint64_t final_xor;       /* d_final_xor */
    bool input_reflected;     /* d_input_reflected */
    bool result_reflected;    /* d_result_reflected */
} sdrk_gr_crc;

/* crc::reflect */
static uint64_t reflect(const sdrk_gr_crc* self, uint64_t word)
{
    uint64_t ret;
    ret = word & 1;
    for (unsigned i = 1; i < self->num_bits; ++i) {
        word >>= 1;
        ret = (ret << 1) | (word & 1);
    }
    return ret;
}

unsigned long sdrk_gr_crc_size(void) { return sizeof(sdrk_gr_crc); }

/* crc::crc(num_bits, poly, initial_value, final_xor, input_reflected, result_reflected) */
int sdrk_gr_crc_init(sdrk_gr_crc* self,
                     unsigned num_bits,
                     uint64_t poly,
                     uint64_t initial_value,
                     uint64_t final_xor,
                     bool input_reflected,
                     bool result_reflected)
{
    self->num_bits = num_bits;
    self->mask = num_bits == 64 ? ~(uint64_t)0 : ((uint64_t)1 << num_bits) - 1;
    self->initial_value = initial_value & self->mask;
    self->final_xor = final_xor & self->mask;
    self->input_reflected = input_reflected;
    self->result_reflected = result_reflected;

    if ((num_bits < 8) || (num_bits > 64)) {
        return -1;
    }

    self->table[0] = 0;
    if (self->input_reflected) {
        poly = reflect(self, poly);
        uint64_t crc = 1;
        int i = 128;
        do {
            if (crc & 1) {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
            for (int j = 0; j < 256; j += 2 * i) {
                self->table[i + j] = (crc ^ self->table[j]) & self->mask;
            }
            i >>= 1;
        } while (i > 0);
    } else {
        const uint64_t msb = (uint64_t)1 << (num_bits - 1);
        uint64_t crc = msb;
        int i = 1;
        do {
            if (crc & msb) {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
            for (int j = 0; j < i; ++j) {
                self->table[i + j] = (crc ^ self->table[j]) & self->mask;
            }
            i <<= 1;
        } while (i < 256);
    }
    return 0;
}

/* crc::compute(data, len) */
uint64_t sdrk_gr_crc_compute(sdrk_gr_crc* self, const uint8_t* data, size_t len)
{
    uint64_t rem = self->initial_value;

    if (self->input_reflected) {
        for (size_t i = 0; i < len; ++i) {
            uint8_t byte = data[i];
            uint8_t idx = (rem ^ byte) & 0xff;
            rem = self->table[idx] ^ (rem >> 8);
        }
    } else {
        for (size_t i = 0; i < len; ++i) {
            uint8_t byte = data[i];
            uint8_t idx = ((rem >> (self->num_bits - 8)) ^ byte) & 0xff;
            rem = (self->table[idx] ^ (rem << 8)) & self->mask;
        }
    }

    if (self->input_reflected != self->result_reflected) {
        rem = reflect(self, rem);
    }

    rem = rem ^ self->final_xor;
    return rem;
}
