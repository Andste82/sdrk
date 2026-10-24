/* QEMU mps2-an386 (Cortex-M4): the 4 MiB SRAM at 0x00000000 holds the image and the 4 MiB SRAM
   at 0x20000000 holds data and stack. Source: QEMU documentation, "Arm MPS2 and MPS3 boards"
   (the memory map of AN386 as modelled by hw/arm/mps2.c). */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   : ORIGIN = 0x20000000, LENGTH = 4M
}
