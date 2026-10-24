/* QEMU virt (RISC-V, 32 bit): RAM starts at 0x80000000 (128 MiB by default); with `-bios none`
   and `-kernel` the first hart starts at the load address of the ELF. Source: QEMU
   documentation, "RISC-V Generic Virtual Platform" (hw/riscv/virt.c memory map). Code, data
   and stack all live in RAM. */
MEMORY
{
  RAM : ORIGIN = 0x80000000, LENGTH = 16M
}

REGION_ALIAS("REGION_TEXT", RAM);
REGION_ALIAS("REGION_RODATA", RAM);
REGION_ALIAS("REGION_DATA", RAM);
REGION_ALIAS("REGION_BSS", RAM);
REGION_ALIAS("REGION_HEAP", RAM);
REGION_ALIAS("REGION_STACK", RAM);
