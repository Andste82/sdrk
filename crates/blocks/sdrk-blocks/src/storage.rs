//! Storage abstraction: where blocks get their buffers from.
//!
//! Blocks never allocate themselves. They ask a [`Storage`] for zero-initialised buffers
//! during construction. [`ArenaStorage`] carves the buffers out of one caller-provided byte
//! slice (microcontrollers); upper layers implement `Storage` with `Vec`.

use core::mem::{align_of, size_of};

use crate::error::{Error, Result};

/// Alignment that [`storage_bytes`](crate::FirBlock::storage_bytes) assumes for the start of an
/// arena. A slice that is not aligned to this may need up to `ARENA_ALIGN - 1` more bytes.
pub const ARENA_ALIGN: usize = 16;

/// A byte array aligned to [`ARENA_ALIGN`], for arenas on the stack or in statics.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Storage};
/// let mut mem = AlignedBytes([0u8; 64]);
/// let mut arena = ArenaStorage::new(&mut mem.0);
/// let buf = arena.alloc::<f32>(4)?;
/// assert_eq!(buf, &[0.0; 4]);
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct AlignedBytes<const N: usize>(pub [u8; N]);

/// Source of buffers for block construction. Blocks never allocate themselves.
pub trait Storage {
    /// The buffer type handed out for elements of type `T`.
    type Buf<T: bytemuck::Pod + Default + Send>: AsRef<[T]> + AsMut<[T]> + Send;

    /// A zero-initialised buffer of exactly `len` elements.
    ///
    /// # Errors
    /// [`Error::Storage`] if the storage cannot provide the buffer.
    fn alloc<T: bytemuck::Pod + Default + Send>(&mut self, len: usize) -> Result<Self::Buf<T>>;
}

/// Heap-free storage carved from one caller-provided byte slice (bump allocation, never freed).
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Storage};
/// let mut mem = AlignedBytes([0u8; 32]);
/// let mut arena = ArenaStorage::new(&mut mem.0);
/// let a = arena.alloc::<u8>(3)?;
/// assert_eq!(a.len(), 3);
/// let b = arena.alloc::<f32>(2)?; // padded to a multiple of 4
/// assert_eq!(arena.used(), 12);
/// assert_eq!(b.len(), 2);
/// assert!(arena.alloc::<f32>(100).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug)]
pub struct ArenaStorage<'a> {
    rest: &'a mut [u8],
    used: usize,
}

impl<'a> ArenaStorage<'a> {
    /// Creates a storage over `bytes`.
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            rest: bytes,
            used: 0,
        }
    }

    /// Bytes handed out so far, including alignment padding.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Bytes still available (padding of the next buffer not yet subtracted).
    pub fn remaining(&self) -> usize {
        self.rest.len()
    }
}

impl<'a> Storage for ArenaStorage<'a> {
    type Buf<T: bytemuck::Pod + Default + Send> = &'a mut [T];

    fn alloc<T: bytemuck::Pod + Default + Send>(&mut self, len: usize) -> Result<&'a mut [T]> {
        let available = self.rest.len();
        let pad = self.rest.as_ptr().align_offset(align_of::<T>());
        let bytes = len
            .checked_mul(size_of::<T>())
            .and_then(|b| b.checked_add(pad))
            .filter(|&b| pad != usize::MAX && b <= available)
            .ok_or_else(|| Error::Storage {
                needed: len
                    .saturating_mul(size_of::<T>())
                    .saturating_add(if pad == usize::MAX { 0 } else { pad }),
                available,
            })?;
        let rest = core::mem::take(&mut self.rest);
        let internal = Error::Core(sdrk_core::Error::Internal("arena split"));
        let (head, tail) = rest
            .split_at_mut_checked(bytes)
            .ok_or_else(|| internal.clone())?;
        let (_, mem) = head
            .split_at_mut_checked(pad)
            .ok_or_else(|| internal.clone())?;
        mem.fill(0);
        let typed = bytemuck::try_cast_slice_mut::<u8, T>(mem).map_err(|_| internal)?;
        self.rest = tail;
        self.used += bytes;
        Ok(typed)
    }
}

/// Counts the bytes a sequence of allocations needs in an arena that starts at an
/// [`ARENA_ALIGN`]-aligned address.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Plan {
    used: usize,
}

impl Plan {
    /// Adds a buffer of `len` elements of `T`.
    pub(crate) fn add<T: bytemuck::Pod>(&mut self, len: usize) -> Result<()> {
        let align = align_of::<T>();
        let pad = (align - self.used % align) % align;
        let overflow = || Error::InvalidParameter {
            name: "storage",
            reason: "the buffers do not fit into the address space",
        };
        self.used = len
            .checked_mul(size_of::<T>())
            .and_then(|b| b.checked_add(pad))
            .and_then(|b| b.checked_add(self.used))
            .ok_or_else(overflow)?;
        Ok(())
    }

    /// Total bytes.
    pub(crate) fn bytes(self) -> usize {
        self.used
    }
}
