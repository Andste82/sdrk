/*
 * The two blocks of GNU Radio's mapper flow that turn bits into constellation points,
 * translated from C++ to C, member for member (GNU Radio main at ec1bf15):
 *
 * - `pack_k_bits_bb` (`gr::blocks::kernel::pack_k_bits::pack` of gr-blocks/lib/pack_k_bits.cc):
 *   `k` input bytes of one bit each become one byte, the first bit the most significant;
 * - `chunks_to_symbols_bc` with the dimension `D` = 1 (`chunks_to_symbols_impl::work` of
 *   gr-digital/lib/chunks_to_symbols_impl.cc): `out[i] = symbol_table[in[i]]` for the byte `in[i]`.
 *
 * `symbol_map_block` is counted against the sum of the two on the same bits (the table of
 * `chunks_to_symbols_bc` is the constellation points in the symbol order of SDRk, as
 * `constellation_rect.points()` is for GNU Radio's own modulators). The scheduler's buffer
 * between the blocks is one byte per symbol, allocated once by the new function.
 *
 * Compiled once per platform with the generic flags of the other GNU Radio references
 * (gr-agc.c).
 *
 * Copyright 2004,2010,2012 Free Software Foundation, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Linked only into the measurement binary of benches/kernels, which is never distributed.
 */
#include <stdlib.h>
#include <string.h>

typedef struct {
    float re;
    float im;
} sdrk_gr_complex;

typedef struct {
    unsigned int k;        /* bits per symbol (d_k of pack_k_bits_bb) */
    unsigned int m;        /* the points of the symbol table */
    unsigned int capacity; /* the most symbols per call */
    sdrk_gr_complex* table;
    unsigned char* chunks; /* the output of pack_k_bits_bb, the input of chunks_to_symbols_bc */
} sdrk_gr_chunks;

void sdrk_gr_chunks_free(sdrk_gr_chunks* self)
{
    if (self != NULL) {
        free(self->table);
        free(self->chunks);
        free(self);
    }
}

/* `chunks_to_symbols_bc(symbol_table, 1)` after `pack_k_bits_bb(k)` for at most `capacity`
 * symbols per call; `points` are `m` pairs; NULL when an allocation fails. */
sdrk_gr_chunks* sdrk_gr_chunks_new(const float* points, unsigned int m, unsigned int k, unsigned int capacity)
{
    sdrk_gr_chunks* self = calloc(1, sizeof(*self));
    if (self == NULL) {
        return NULL;
    }
    self->k = k;
    self->m = m;
    self->capacity = capacity;
    self->table = malloc((size_t)m * sizeof(sdrk_gr_complex));
    self->chunks = malloc(capacity);
    if (self->table == NULL || self->chunks == NULL) {
        sdrk_gr_chunks_free(self);
        return NULL;
    }
    for (unsigned int i = 0; i < m; i++) {
        self->table[i].re = points[2 * i];
        self->table[i].im = points[2 * i + 1];
    }
    return self;
}

/* pack_k_bits::pack */
static void pack(unsigned char* bytes, const unsigned char* bits, int nbytes, unsigned int d_k)
{
    for (int i = 0; i < nbytes; i++) {
        bytes[i] = 0x00;
        for (unsigned int j = 0; j < d_k; j++) {
            bytes[i] |= (0x01 & bits[i * d_k + j]) << (d_k - j - 1);
        }
    }
}

/* `n` symbols (at most the capacity) from `n * k` bits */
void sdrk_gr_chunks_work(sdrk_gr_chunks* self, sdrk_gr_complex* out, const unsigned char* bits, unsigned int n)
{
    if (n > self->capacity) {
        n = self->capacity;
    }
    pack(self->chunks, bits, (int)n, self->k);
    /* chunks_to_symbols_impl::work, D = 1 */
    for (const unsigned char* in = self->chunks; in < self->chunks + n; ++in) {
        unsigned int key = (unsigned int)*in;
        *out = self->table[key];
        ++out;
    }
}
