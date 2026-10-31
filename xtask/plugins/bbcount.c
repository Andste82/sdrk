/*
 * Execution counts of the translation blocks of a user-mode QEMU run, for the simulated cycles
 * of `cargo xtask measure` (xtask/src/mca.rs, docs/performance.md#simulated-cycles).
 *
 * xtask builds this plugin against the `qemu-plugin.h` of the pinned QEMU release
 * (xtask/src/count.rs, `qemu_plugin`). It assumes one vCPU (the measurement binary has one
 * thread). At exit it writes, through the plugin log (`-d plugin`):
 *
 *   bbcount 1                                  format version
 *   guest_base <hex>                           host address of guest address 0
 *   map <lo> <hi> <offset> <path>              file mappings of the QEMU process (host addresses)
 *   tb <vaddr> <insns> <bytes> <executions>    every translation block that ran
 *   jump <from> <to> <times>                   control transfers between blocks that are not a
 *                                              fall-through: the address of the last instruction
 *                                              of a block and the start of the next one
 *
 * The guest code is mapped from its files into the QEMU process at `guest_base + vaddr`, so the
 * host mappings give the object and the file offset of every guest address, for the binary and
 * for the shared libraries alike.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <glib.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

/* The successors of a block remembered in the block itself; further ones go to `jumps`. */
#define CACHED 2

typedef struct Tb {
    uint64_t vaddr;
    uint64_t last;
    uint64_t insns;
    uint64_t bytes;
    uint64_t count;
    struct Tb *next[CACHED];
    uint64_t next_count[CACHED];
} Tb;

typedef struct {
    Tb *from;
    Tb *to;
} Pair;

static GPtrArray *tbs;
static GHashTable *jumps;
static Tb *previous;
static uint64_t guest_base;
static gboolean have_base;

static guint pair_hash(gconstpointer p)
{
    const Pair *k = p;
    return g_direct_hash(k->from) ^ (g_direct_hash(k->to) * 31u);
}

static gboolean pair_equal(gconstpointer a, gconstpointer b)
{
    const Pair *x = a, *y = b;
    return x->from == y->from && x->to == y->to;
}

static void transfer(Tb *from, Tb *to)
{
    for (int i = 0; i < CACHED; i++) {
        if (from->next[i] == to) {
            from->next_count[i]++;
            return;
        }
        if (from->next[i] == NULL) {
            from->next[i] = to;
            from->next_count[i] = 1;
            return;
        }
    }
    Pair key = { from, to };
    uint64_t *n = g_hash_table_lookup(jumps, &key);
    if (n == NULL) {
        Pair *k = g_new(Pair, 1);
        *k = key;
        n = g_new0(uint64_t, 1);
        g_hash_table_insert(jumps, k, n);
    }
    (*n)++;
}

static void exec(unsigned int vcpu, void *udata)
{
    Tb *t = udata;
    t->count++;
    if (previous != NULL && previous->vaddr + previous->bytes != t->vaddr) {
        transfer(previous, t);
    }
    previous = t;
}

static void trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    Tb *t = g_new0(Tb, 1);
    size_t n = qemu_plugin_tb_n_insns(tb);
    t->vaddr = qemu_plugin_tb_vaddr(tb);
    t->insns = n;
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        t->bytes += qemu_plugin_insn_size(insn);
        if (i + 1 == n) {
            t->last = qemu_plugin_insn_vaddr(insn);
        }
        if (!have_base) {
            guest_base = (uint64_t)(uintptr_t)qemu_plugin_insn_haddr(insn) -
                         qemu_plugin_insn_vaddr(insn);
            have_base = TRUE;
        }
    }
    g_ptr_array_add(tbs, t);
    qemu_plugin_register_vcpu_tb_exec_cb(tb, exec, QEMU_PLUGIN_CB_NO_REGS, t);
}

static void at_exit(qemu_plugin_id_t id, void *p)
{
    g_autoptr(GString) out = g_string_new("bbcount 1\n");
    g_string_append_printf(out, "guest_base %" PRIx64 "\n", guest_base);
    FILE *maps = fopen("/proc/self/maps", "r");
    if (maps != NULL) {
        char line[4096];
        while (fgets(line, sizeof line, maps) != NULL) {
            uint64_t lo, hi, offset;
            char perms[8], path[4096];
            if (sscanf(line, "%" SCNx64 "-%" SCNx64 " %7s %" SCNx64 " %*s %*s %4095s", &lo, &hi,
                       perms, &offset, path) == 5 &&
                path[0] == '/') {
                g_string_append_printf(out, "map %" PRIx64 " %" PRIx64 " %" PRIx64 " %s\n", lo,
                                       hi, offset, path);
            }
        }
        fclose(maps);
    }
    for (guint i = 0; i < tbs->len; i++) {
        Tb *t = g_ptr_array_index(tbs, i);
        if (t->count) {
            g_string_append_printf(out, "tb %" PRIx64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                                   t->vaddr, t->insns, t->bytes, t->count);
        }
        for (int k = 0; k < CACHED; k++) {
            if (t->next[k] != NULL) {
                g_string_append_printf(out, "jump %" PRIx64 " %" PRIx64 " %" PRIu64 "\n", t->last,
                                       t->next[k]->vaddr, t->next_count[k]);
            }
        }
    }
    GHashTableIter it;
    gpointer key, value;
    g_hash_table_iter_init(&it, jumps);
    while (g_hash_table_iter_next(&it, &key, &value)) {
        const Pair *k = key;
        g_string_append_printf(out, "jump %" PRIx64 " %" PRIx64 " %" PRIu64 "\n", k->from->last,
                               k->to->vaddr, *(uint64_t *)value);
    }
    qemu_plugin_outs(out->str);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info,
                                           int argc, char **argv)
{
    tbs = g_ptr_array_new();
    jumps = g_hash_table_new(pair_hash, pair_equal);
    qemu_plugin_register_vcpu_tb_trans_cb(id, trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
