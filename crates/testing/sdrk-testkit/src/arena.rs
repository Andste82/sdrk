//! Bump allocator over a caller-provided buffer (no heap, no `unsafe`).
//!
//! Safe Rust cannot hand out several long-lived slices from `&mut self` and
//! later reclaim the memory with `reset`. The arena therefore offers two
//! ways to take memory:
//!
//! - [`Arena::take`] bumps a persistent offset and returns one slice that
//!   borrows the arena (the previous slice must be dropped first);
//! - [`Arena::carver`] returns a [`Carver`] that hands out any number of
//!   non-overlapping slices; the memory is free again when it is dropped.

use bytemuck::Pod;
use core::mem::{align_of, size_of};

use crate::suite::Failure;

/// Bump allocator over a caller buffer.
///
/// ```
/// use sdrk_core::C32;
/// use sdrk_testkit::Arena;
/// let mut buf = [0u8; 64];
/// let mut arena = Arena::new(&mut buf);
/// let bytes = arena.take::<u8>(3)?;
/// assert_eq!(bytes.len(), 3);
/// let c = arena.take::<C32>(2)?;
/// assert_eq!(c.as_ptr() as usize % core::mem::align_of::<C32>(), 0);
/// # Ok::<(), sdrk_testkit::Failure>(())
/// ```
#[derive(Debug)]
pub struct Arena<'a> {
    buf: &'a mut [u8],
    used: usize,
}

/// Number of padding bytes so that `addr + pad` is a multiple of `align`.
fn padding(addr: usize, align: usize) -> usize {
    (align - addr % align) % align
}

/// Checks that `n` elements of `T` fit into `len` bytes starting at `addr`;
/// returns the padding and the total number of bytes needed.
fn plan<T: Pod>(addr: usize, len: usize, n: usize) -> Result<(usize, usize), Failure> {
    let pad = padding(addr, align_of::<T>());
    let total = n
        .checked_mul(size_of::<T>())
        .and_then(|bytes| bytes.checked_add(pad));
    match total {
        Some(total) if total <= len => Ok((pad, total)),
        _ => Err(
            Failure::new("arena", "out of arena memory").detail(format_args!(
                "need {n} x {} bytes (+{pad} padding), have {len}",
                size_of::<T>()
            )),
        ),
    }
}

/// Splits `total` bytes off the front of `rest`, drops `pad` bytes of padding,
/// zero-fills the rest of them and casts them to `T`. Returns the typed slice
/// and the remainder. `None` only if `plan` was not consulted first.
fn carve<T: Pod>(rest: &mut [u8], pad: usize, total: usize) -> Option<(&mut [T], &mut [u8])> {
    let (head, tail) = rest.split_at_mut_checked(total)?;
    let (_, mem) = head.split_at_mut_checked(pad)?;
    mem.fill(0);
    let typed = bytemuck::try_cast_slice_mut::<u8, T>(mem).ok()?;
    Some((typed, tail))
}

fn internal() -> Failure {
    Failure::new("arena", "internal arena error")
}

impl<'a> Arena<'a> {
    /// Creates an arena over `buf`.
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, used: 0 }
    }

    /// Takes `n` zero-filled elements of `T`, aligned for `T`.
    ///
    /// # Errors
    /// Returns a [`Failure`] if the arena is exhausted.
    pub fn take<T: Pod>(&mut self, n: usize) -> Result<&mut [T], Failure> {
        let (_, rest) = self
            .buf
            .split_at_mut_checked(self.used)
            .ok_or_else(internal)?;
        let (pad, total) = plan::<T>(rest.as_ptr() as usize, rest.len(), n)?;
        let (typed, _) = carve::<T>(rest, pad, total).ok_or_else(internal)?;
        self.used += total;
        Ok(typed)
    }

    /// Starts a scope in which several slices can be alive at once. The
    /// memory taken from the [`Carver`] is released when it is dropped.
    ///
    /// ```
    /// use sdrk_core::C32;
    /// use sdrk_testkit::Arena;
    /// let mut buf = [0u8; 256];
    /// let mut arena = Arena::new(&mut buf);
    /// let mut carver = arena.carver();
    /// let a = carver.take::<C32>(4)?;
    /// let b = carver.take::<u8>(4)?;
    /// a[0].re = 1.0;
    /// b[0] = 1;
    /// # Ok::<(), sdrk_testkit::Failure>(())
    /// ```
    pub fn carver(&mut self) -> Carver<'_> {
        let rest = self.buf.get_mut(self.used..).unwrap_or_default();
        Carver { rest }
    }

    /// Frees everything taken so far.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// Unused bytes (padding not yet accounted for).
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.used)
    }
}

/// A scope of an [`Arena`] that can hand out several slices at once.
#[derive(Debug)]
pub struct Carver<'s> {
    rest: &'s mut [u8],
}

impl<'s> Carver<'s> {
    /// Takes `n` zero-filled elements of `T`, aligned for `T`.
    ///
    /// # Errors
    /// Returns a [`Failure`] if the arena is exhausted; nothing is taken then.
    pub fn take<T: Pod>(&mut self, n: usize) -> Result<&'s mut [T], Failure> {
        let (pad, total) = plan::<T>(self.rest.as_ptr() as usize, self.rest.len(), n)?;
        let rest = core::mem::take(&mut self.rest);
        let (typed, tail) = carve::<T>(rest, pad, total).ok_or_else(internal)?;
        self.rest = tail;
        Ok(typed)
    }

    /// Unused bytes.
    pub fn remaining(&self) -> usize {
        self.rest.len()
    }
}
