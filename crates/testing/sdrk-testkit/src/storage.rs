//! `Vec`-backed block storage for host tests (feature `std`).
//!
//! The pipeline of a later milestone implements `Storage` the same way; the heap-free blocks
//! themselves only know the trait.

use sdrk_blocks::{Result, Storage};

/// A [`Storage`] that hands out `Vec`s.
///
/// ```
/// use sdrk_blocks::Storage;
/// use sdrk_testkit::VecStorage;
/// let buf = VecStorage.alloc::<f32>(3)?;
/// assert_eq!(buf, vec![0.0; 3]);
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VecStorage;

impl Storage for VecStorage {
    type Buf<T: bytemuck::Pod + Default + Send> = std::vec::Vec<T>;

    fn alloc<T: bytemuck::Pod + Default + Send>(&mut self, len: usize) -> Result<std::vec::Vec<T>> {
        Ok(std::vec![T::default(); len])
    }
}
