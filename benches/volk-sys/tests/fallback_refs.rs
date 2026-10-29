//! Smoke tests of the VOLK functions that the fallback round added to the wasm32 build of the
//! generic protokernels (`benches/refs/volk-wasm.c`); on x86-64 and AArch64 they run VOLK's own
//! build. Run through the benchmark workspace with `PKG_CONFIG_PATH` pointing at the build of
//! `benches/refs/build.sh`; without VOLK the tests are skipped.

use volk_sys::lv_32fc_t;

fn c(re: f32, im: f32) -> lv_32fc_t {
    lv_32fc_t { re, im }
}

#[test]
fn round2_kernels_link_and_compute() {
    if !volk_sys::FULL {
        eprintln!("skipped: VOLK is not linked");
        return;
    }
    let a = [
        c(1.0, 2.0),
        c(-0.5, 0.25),
        c(3.0, -1.0),
        c(0.0, 1.0),
        c(2.0, 2.0),
    ];
    let b = [
        c(0.5, -1.0),
        c(2.0, 0.0),
        c(-1.0, 0.5),
        c(1.0, 1.0),
        c(0.25, -0.25),
    ];
    let r = [2.0f32, -1.0, 0.5, 4.0, 1.5];
    let n = 5;
    // SAFETY (all calls below): every buffer holds `n` elements and outputs do not overlap
    // inputs.
    unsafe {
        let mut out = [lv_32fc_t::default(); 5];
        volk_sys::native::volk_32fc_32f_multiply_32fc(out.as_mut_ptr(), a.as_ptr(), r.as_ptr(), n);
        for i in 0..5 {
            assert_eq!(out[i], c(a[i].re * r[i], a[i].im * r[i]));
        }

        let pos = [1.0f32, 2.0, 8.0, 0.5, 1024.0];
        let mut l = [0.0f32; 5];
        volk_sys::native::volk_32f_log2_32f(l.as_mut_ptr(), pos.as_ptr(), n);
        for (o, x) in l.iter().zip(&pos) {
            assert!((o - x.log2()).abs() < 1e-3, "log2({x}) = {o}");
        }

        let mut s = [0.0f32; 5];
        volk_sys::native::volk_32f_s32f_multiply_32f(s.as_mut_ptr(), r.as_ptr(), 3.0, n);
        assert_eq!(s, [6.0, -3.0, 1.5, 12.0, 4.5]);
        volk_sys::native::volk_32f_s32f_add_32f(s.as_mut_ptr(), r.as_ptr(), 1.0, n);
        assert_eq!(s, [3.0, 0.0, 1.5, 5.0, 2.5]);

        let mut dot = 0.0f32;
        volk_sys::volk_32f_x2_dot_prod_32f(&raw mut dot, r.as_ptr(), pos.as_ptr(), n);
        let want: f32 = r.iter().zip(&pos).map(|(x, y)| x * y).sum();
        assert!((dot - want).abs() < 1e-3);

        let mut cdot = lv_32fc_t::default();
        volk_sys::volk_32fc_x2_dot_prod_32fc(&raw mut cdot, a.as_ptr(), b.as_ptr(), n);
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(&b) {
            re += x.re * y.re - x.im * y.im;
            im += x.re * y.im + x.im * y.re;
        }
        assert!((cdot.re - re).abs() < 1e-4 && (cdot.im - im).abs() < 1e-4);

        let bytes = [1i8, -1, 127, -128, 0];
        let mut wide = [0i16; 5];
        volk_sys::volk_8i_convert_16i(wide.as_mut_ptr(), bytes.as_ptr(), n);
        assert_eq!(wide, [256, -256, 32512, -32768, 0]);
    }
    for name in [
        "volk_32fc_32f_multiply_32fc",
        "volk_32f_log2_32f",
        "volk_32f_s32f_multiply_32f",
        "volk_32f_s32f_add_32f",
        "volk_32f_x2_dot_prod_32f",
        "volk_32fc_x2_dot_prod_32fc",
        "volk_8i_convert_16i",
    ] {
        let list = volk_sys::implementations(name).expect("declared");
        assert!(list.iter().any(|i| i.name == "generic"), "{name}: {list:?}");
    }
}
