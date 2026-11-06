//! Heap-backed storage for the blocks of a pipeline (zero-filled boxed slices).

use sdrk_blocks::{Error, Result, Storage};

/// Allocates `len` zeroed elements; out-of-memory becomes an error instead of an abort.
pub(crate) fn alloc_vec<T: Clone + Default>(len: usize) -> Result<Vec<T>> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(len).map_err(|_| Error::Storage {
        needed: len.saturating_mul(core::mem::size_of::<T>()),
        available: 0,
    })?;
    v.resize(len, T::default());
    Ok(v)
}

/// A [`Storage`] that hands out zero-filled boxed slices.
///
/// The allocation uses `try_reserve_exact`, so a request that cannot be satisfied is an
/// [`Error::Storage`] and not an abort.
///
/// ```
/// use sdrk_blocks::Storage;
/// use sdrk_pipeline::VecStorage;
/// let buf = VecStorage.alloc::<f32>(3)?;
/// assert_eq!(&*buf, &[0.0; 3]);
/// assert!(VecStorage.alloc::<u64>(usize::MAX / 8).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VecStorage;

impl Storage for VecStorage {
    // Boxed slices: a buffer never changes its length, so the capacity of a `Vec` would only add
    // 8 bytes to every buffer handle of a block.
    type Buf<T: bytemuck::Pod + Default + Send> = Box<[T]>;

    fn alloc<T: bytemuck::Pod + Default + Send>(&mut self, len: usize) -> Result<Box<[T]>> {
        alloc_vec(len).map(Vec::into_boxed_slice)
    }
}
