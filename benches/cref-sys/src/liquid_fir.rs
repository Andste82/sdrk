//! The references of the pulse filters of the QAM and PSK blocks (issue #169): liquid-dsp 1.8.3
//! (MIT, cfg `has_liquidhb`, `benches/refs/liquid-fir.c`), the FIR filter [`LiquidFirfiltCrcf`]
//! (`firfilt_crcf`) and the polyphase interpolator [`LiquidFirinterpCrcf`] (`firinterp_crcf`):
//! complex samples in and out, real taps, with liquid's own dot product: the SIMD loop of the
//! level (SSE, AVX, AVX-512F, NEON; the portable loop at `simd128`) or, on request, the
//! portable loop. Both are built with the taps of the caller (SDRk's, so that the accuracy of
//! both sides is comparable). The rational-rate resamplers [`LiquidRresampCrcf`] and
//! [`LiquidRresampRrrf`] (`rresamp_crcf`, `rresamp_rrrf`, `benches/refs/liquid-rresamp.c`) run the
//! portable dot product: the scalar references of the resample fallbacks (D2904).

use core::ffi::{c_int, c_uint, c_void};

#[cfg(has_liquidhb)]
mod ffi {
    use core::ffi::{c_int, c_uint, c_void};

    unsafe extern "C" {
        pub fn sdrk_liquid_firinterp_crcf_create(
            h: *const f32,
            h_len: c_uint,
            m: c_uint,
            method: c_int,
        ) -> *mut c_void;
        pub fn sdrk_liquid_firinterp_crcf_reset(q: *mut c_void);
        pub fn sdrk_liquid_firinterp_crcf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_firinterp_crcf_execute_block(
            q: *mut c_void,
            x: *mut [f32; 2],
            n: c_uint,
            y: *mut [f32; 2],
        );
        pub fn sdrk_liquid_rresamp_crcf_create(
            h: *const f32,
            p: c_uint,
            q: c_uint,
            m: c_uint,
        ) -> *mut c_void;
        pub fn sdrk_liquid_rresamp_crcf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_rresamp_crcf_execute_block(
            q: *mut c_void,
            x: *mut [f32; 2],
            n: c_uint,
            y: *mut [f32; 2],
        );
        pub fn sdrk_liquid_rresamp_rrrf_create(
            h: *const f32,
            p: c_uint,
            q: c_uint,
            m: c_uint,
        ) -> *mut c_void;
        pub fn sdrk_liquid_rresamp_rrrf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_rresamp_rrrf_execute_block(
            q: *mut c_void,
            x: *mut f32,
            n: c_uint,
            y: *mut f32,
        );
        pub fn sdrk_liquid_firfilt_crcf_create(
            h: *const f32,
            h_len: c_uint,
            method: c_int,
        ) -> *mut c_void;
        pub fn sdrk_liquid_firfilt_crcf_reset(q: *mut c_void);
        pub fn sdrk_liquid_firfilt_crcf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_modem_create(pts: *const f32, m: c_uint) -> *mut c_void;
        pub fn sdrk_liquid_modem_destroy(q: *mut c_void);
        pub fn sdrk_liquid_modem_map_block(
            q: *mut c_void,
            bits: *const u8,
            k: c_uint,
            n: c_uint,
            y: *mut [f32; 2],
        );
        pub fn sdrk_liquid_modem_std_create(m: c_uint) -> *mut c_void;
        pub fn sdrk_liquid_modem_points(q: *mut c_void, m: c_uint, pts: *mut f32);
        pub fn sdrk_liquid_modem_demod_soft_block(
            q: *mut c_void,
            x: *const [f32; 2],
            n: c_uint,
            k: c_uint,
            soft: *mut u8,
            syms: *mut c_uint,
        );
        pub fn sdrk_liquid_firfilt_crcf_execute_block(
            q: *mut c_void,
            x: *mut [f32; 2],
            n: c_uint,
            y: *mut [f32; 2],
        );
    }
}

/// The block length as liquid's `unsigned int`.
fn count(n: usize) -> c_uint {
    c_uint::try_from(n).expect("block too long")
}

/// liquid's `firfilt_crcf`: `y[n] = sum h[k] x[n - k]`, zero history at the start.
pub struct LiquidFirfiltCrcf {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
}

// SAFETY: the C object owns its buffers and is used by one thread at a time.
unsafe impl Send for LiquidFirfiltCrcf {}

impl LiquidFirfiltCrcf {
    /// The filter with the taps `h`; `simd` selects the SIMD dot product of the level,
    /// otherwise the portable one. `None` without the library or without taps.
    #[must_use]
    pub fn new(h: &[f32], simd: bool) -> Option<Self> {
        if h.is_empty() {
            return None;
        }
        #[cfg(has_liquidhb)]
        {
            // SAFETY: `h` holds `h.len()` values, which the call copies.
            let q = unsafe {
                ffi::sdrk_liquid_firfilt_crcf_create(h.as_ptr(), count(h.len()), c_int::from(simd))
            };
            // Not `then_some`: it would build (and drop, so destroy) a null object.
            if q.is_null() { None } else { Some(Self { q }) }
        }
        #[cfg(not(has_liquidhb))]
        {
            let _ = simd;
            None
        }
    }

    /// Clears the delay line.
    pub fn reset(&mut self) {
        #[cfg(has_liquidhb)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_firfilt_crcf_reset(self.q);
        }
    }

    /// `x.len()` outputs from `x`.
    ///
    /// # Panics
    /// If `y` is not as long as `x`.
    pub fn execute(&mut self, x: &mut [[f32; 2]], y: &mut [[f32; 2]]) {
        assert_eq!(x.len(), y.len(), "liquid firfilt: lengths");
        let n = count(x.len());
        #[cfg(has_liquidhb)]
        // SAFETY: both buffers hold `n` samples (liquid only reads `x`).
        unsafe {
            ffi::sdrk_liquid_firfilt_crcf_execute_block(self.q, x.as_mut_ptr(), n, y.as_mut_ptr());
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidFirfiltCrcf {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_firfilt_crcf_destroy(self.q) }
    }
}

/// liquid's `firinterp_crcf`: `y[k m + i] = sum_j h[i + j m] x[k - j]` (the taps padded with
/// zeros to a multiple of `m`), zero history at the start.
pub struct LiquidFirinterpCrcf {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
    m: usize,
}

// SAFETY: the C object owns its buffers and is used by one thread at a time.
unsafe impl Send for LiquidFirinterpCrcf {}

impl LiquidFirinterpCrcf {
    /// The interpolator by `m` (at least 2) with the prototype `h`; `simd` as for
    /// [`LiquidFirfiltCrcf::new`]. `None` without the library, without taps or for a factor
    /// below 2.
    #[must_use]
    pub fn new(h: &[f32], m: usize, simd: bool) -> Option<Self> {
        if h.len() < m || m < 2 {
            return None;
        }
        #[cfg(has_liquidhb)]
        {
            // SAFETY: `h` holds `h.len()` values, which the call copies.
            let q = unsafe {
                ffi::sdrk_liquid_firinterp_crcf_create(
                    h.as_ptr(),
                    count(h.len()),
                    count(m),
                    c_int::from(simd),
                )
            };
            if q.is_null() {
                None
            } else {
                Some(Self { q, m })
            }
        }
        #[cfg(not(has_liquidhb))]
        {
            let _ = simd;
            None
        }
    }

    /// Clears the delay line.
    pub fn reset(&mut self) {
        #[cfg(has_liquidhb)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_firinterp_crcf_reset(self.q);
        }
    }

    /// `x.len() * m` outputs from `x`.
    ///
    /// # Panics
    /// If `y` is not `x.len() * m` long.
    pub fn execute(&mut self, x: &mut [[f32; 2]], y: &mut [[f32; 2]]) {
        assert_eq!(y.len(), x.len() * self.m, "liquid firinterp: lengths");
        let n = count(x.len());
        #[cfg(has_liquidhb)]
        // SAFETY: `x` holds `n` samples (liquid only reads them) and `y` holds `n m`.
        unsafe {
            ffi::sdrk_liquid_firinterp_crcf_execute_block(
                self.q,
                x.as_mut_ptr(),
                n,
                y.as_mut_ptr(),
            );
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidFirinterpCrcf {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_firinterp_crcf_destroy(self.q) }
    }
}

/// liquid's `modemcf` as a symbol mapper (`benches/refs/liquid-modem.c`): the arbitrary
/// constellation of the caller's points (`modemcf_create_arbitrary`) and `modemcf_modulate`
/// per symbol, with the bits packed to the symbol by the wrapper.
pub struct LiquidModemMap {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
    k: usize,
}

// SAFETY: the C object owns its table and is used by one thread at a time.
unsafe impl Send for LiquidModemMap {}

impl LiquidModemMap {
    /// The mapper for the table `points` (symbol order) of `2^k` entries; `None` without the
    /// library or for another table length.
    #[must_use]
    pub fn new(points: &[[f32; 2]], k: usize) -> Option<Self> {
        if k == 0 || k > 8 || points.len() != 1 << k {
            return None;
        }
        #[cfg(has_liquidhb)]
        {
            // SAFETY: `points` holds `2^k` pairs, which the call copies.
            let q = unsafe {
                ffi::sdrk_liquid_modem_create(points.as_ptr().cast(), count(points.len()))
            };
            if q.is_null() {
                None
            } else {
                Some(Self { q, k })
            }
        }
        #[cfg(not(has_liquidhb))]
        {
            let _ = k;
            None
        }
    }

    /// `y.len()` symbols from `bits` (`y.len() * k` bytes of 0 or 1).
    ///
    /// # Panics
    /// If `bits` is not `y.len() * k` long.
    pub fn map(&mut self, bits: &[u8], y: &mut [[f32; 2]]) {
        assert_eq!(bits.len(), y.len() * self.k, "liquid modem: lengths");
        let n = count(y.len());
        #[cfg(has_liquidhb)]
        // SAFETY: `bits` holds `n * k` bytes and `y` `n` pairs.
        unsafe {
            ffi::sdrk_liquid_modem_map_block(
                self.q,
                bits.as_ptr(),
                count(self.k),
                n,
                y.as_mut_ptr(),
            );
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidModemMap {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_modem_destroy(self.q) }
    }
}

/// liquid's own 16-QAM or 64-QAM modem (`modemcf_create(LIQUID_MODEM_QAM16 / QAM64)`) as a soft
/// demodulator (`modemcf_demodulate_soft`, `benches/refs/liquid-modem.c`): the hard symbol and
/// one unsigned byte per bit (127 is zero, liquid's scale), liquid's symbol labelling (not
/// SDRk's).
pub struct LiquidModemDemod {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
    m: usize,
    k: usize,
}

// SAFETY: the C object owns its tables and is used by one thread at a time.
unsafe impl Send for LiquidModemDemod {}

impl LiquidModemDemod {
    /// The modem with `m` = 16 or 64 points; `None` without the library or for another size.
    #[must_use]
    pub fn new(m: usize) -> Option<Self> {
        if m != 16 && m != 64 {
            return None;
        }
        #[cfg(has_liquidhb)]
        {
            // SAFETY: plain construction.
            let q = unsafe { ffi::sdrk_liquid_modem_std_create(count(m)) };
            if q.is_null() {
                None
            } else {
                Some(Self {
                    q,
                    m,
                    k: m.trailing_zeros() as usize,
                })
            }
        }
        #[cfg(not(has_liquidhb))]
        {
            None
        }
    }

    /// The bits per symbol.
    #[must_use]
    pub fn bits_per_symbol(&self) -> usize {
        self.k
    }

    /// The points of the modem in its symbol order.
    #[must_use]
    pub fn points(&mut self) -> Vec<[f32; 2]> {
        #[allow(unused_mut)]
        let mut pts = vec![[0.0f32; 2]; self.m];
        #[cfg(has_liquidhb)]
        // SAFETY: `pts` holds `m` pairs.
        unsafe {
            ffi::sdrk_liquid_modem_points(self.q, count(self.m), pts.as_mut_ptr().cast());
        }
        pts
    }

    /// `x.len()` symbols: the hard symbols into `syms` and `k` soft bytes per symbol into `soft`.
    ///
    /// # Panics
    /// If `syms` is not as long as `x` or `soft` not `k` times as long.
    pub fn demod_soft(&mut self, x: &[[f32; 2]], soft: &mut [u8], syms: &mut [u32]) {
        assert!(
            syms.len() == x.len() && soft.len() == x.len() * self.k,
            "liquid modem: lengths"
        );
        let n = count(x.len());
        #[cfg(has_liquidhb)]
        // SAFETY: the buffers hold `n`, `n k` and `n` elements.
        unsafe {
            ffi::sdrk_liquid_modem_demod_soft_block(
                self.q,
                x.as_ptr(),
                n,
                count(self.k),
                soft.as_mut_ptr(),
                syms.as_mut_ptr(),
            );
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidModemDemod {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_modem_destroy(self.q) }
    }
}

macro_rules! rresamp {
    ($name:ident, $sample:ty, $create:ident, $destroy:ident, $execute:ident, $what:literal) => {
        #[doc = concat!("liquid's `", $what, "` (`P / Q` over a filter bank of `P` branches of `2 m` taps,")]
        /// the portable dot product): `P` outputs per `Q` inputs, zero history at the start.
        pub struct $name {
            #[cfg_attr(not(has_liquidhb), allow(dead_code))]
            q: *mut c_void,
            p: usize,
            d: usize,
        }

        // SAFETY: the C object owns its buffers and is used by one thread at a time.
        unsafe impl Send for $name {}

        impl $name {
            #[doc = concat!("`", $what, "_create(p, q, m, h)` with the `2 p m` taps `h`; `None` without the")]
            /// library or for taps of another length.
            #[must_use]
            pub fn new(h: &[f32], p: usize, q: usize, m: usize) -> Option<Self> {
                if p == 0 || q == 0 || m == 0 || h.len() != 2 * p * m {
                    return None;
                }
                #[cfg(has_liquidhb)]
                {
                    // SAFETY: `h` holds `2 p m` values, which the call copies.
                    let r = unsafe { ffi::$create(h.as_ptr(), count(p), count(q), count(m)) };
                    (!r.is_null()).then_some(Self { q: r, p, d: q })
                }
                #[cfg(not(has_liquidhb))]
                None
            }

            /// `y.len() / P` primitive blocks: `Q` inputs each from `x`, `P` outputs each into `y`.
            ///
            /// # Panics
            /// If `y` is no multiple of `P` or `x` is shorter than its blocks need.
            pub fn execute(&mut self, x: &mut [$sample], y: &mut [$sample]) {
                assert!(
                    y.len() % self.p == 0 && x.len() >= y.len() / self.p * self.d,
                    "liquid rresamp: lengths"
                );
                let n = count(y.len() / self.p);
                #[cfg(has_liquidhb)]
                // SAFETY: `x` holds at least `Q n` samples (liquid only reads them), `y` `P n`.
                unsafe {
                    ffi::$execute(self.q, x.as_mut_ptr(), n, y.as_mut_ptr());
                }
                #[cfg(not(has_liquidhb))]
                let _ = n;
            }
        }

        #[cfg(has_liquidhb)]
        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: the object came from the constructor and is destroyed once.
                unsafe { ffi::$destroy(self.q) }
            }
        }
    };
}

rresamp!(
    LiquidRresampCrcf,
    [f32; 2],
    sdrk_liquid_rresamp_crcf_create,
    sdrk_liquid_rresamp_crcf_destroy,
    sdrk_liquid_rresamp_crcf_execute_block,
    "rresamp_crcf"
);
rresamp!(
    LiquidRresampRrrf,
    f32,
    sdrk_liquid_rresamp_rrrf_create,
    sdrk_liquid_rresamp_rrrf_destroy,
    sdrk_liquid_rresamp_rrrf_execute_block,
    "rresamp_rrrf"
);
