/* QEMU mps3-an547 (Cortex-M55), secure alias. Source: QEMU documentation, "Arm MPS2 and MPS3
   boards", and `info mtree` of QEMU 10.2.1. The reset vectors are read from address 0, which
   is the 512 KiB ITCM; the 4 MiB SRAM at 0x21000000 (two 2 MiB banks) holds data and stack.
   The suite no longer fits the ITCM, so the code of the test cases (`sdrk_testkit`) goes to
   the 2 MiB code SRAM at 0x01000000, which QEMU loads straight from the ELF file. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 512K
  CODE  : ORIGIN = 0x01000000, LENGTH = 2M
  RAM   : ORIGIN = 0x21000000, LENGTH = 4M
}

/* Placed before `.text`, so these input sections are taken here and not by `.text`. */
SECTIONS
{
  .text.testkit : ALIGN(4)
  {
    *(.text._R*12sdrk_testkit*)
    . = ALIGN(4);
  } > CODE
}
INSERT BEFORE .text;
