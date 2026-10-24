/* QEMU mps3-an547 (Cortex-M55), secure alias. Source: QEMU documentation, "Arm MPS2 and MPS3
   boards", and `info mtree` of QEMU 10.2.1. The reset vectors are read from address 0, which
   is the 512 KiB ITCM; the 4 MiB SRAM at 0x21000000 (two 2 MiB banks) holds data and stack. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 512K
  RAM   : ORIGIN = 0x21000000, LENGTH = 4M
}
