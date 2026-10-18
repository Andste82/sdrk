//! The `delegate!` macro.

/// Implements kernel traits for [`Sdrk`](crate::Sdrk) by delegation and builds the
/// backend table behind [`backend_for`](crate::backend_for) from the same entries.
///
/// Each entry has the form
/// `Trait<Type>::method(&self, arg: Ty, ...) -> Ret => Backend`, entries are separated
/// by `;`. A generic method lists its generics and bounds in square brackets after the
/// name: `Trait<Type>::method[A: Bound, B: Other](&self, ...) -> Ret => Backend`. `Backend` is a unit struct that implements the trait and
/// [`BackendName`](crate::BackendName). The macro must be invoked exactly once
/// (it also defines `backend_for`); later backends only change the `=> Backend`
/// target of an entry, guarded by `#[cfg(...)]`.
///
/// The trait, the sample type and the result type must be in scope at the call site.
#[macro_export]
macro_rules! delegate {
    (
        $(
            $(#[$meta:meta])*
            $tr:ident < $ty:ident > :: $m:ident $([ $($gen:tt)* ])? ( &self $(, $a:ident : $at:ty)* $(,)? ) -> $ret:ty => $be:ident
        );+ $(;)?
    ) => {
        $(
            $(#[$meta])*
            impl $tr<$ty> for $crate::Sdrk {
                #[inline]
                fn $m $(< $($gen)* >)? (&self $(, $a: $at)*) -> $ret {
                    <$be as $tr<$ty>>::$m(&$be $(, $a)*)
                }
            }
        )+

        /// Name of the implementation the facade uses for `kernel` on sample type `ty`,
        /// or `"none"` for an unknown combination.
        ///
        /// ```
        /// use sdrk::{backend_for, SampleType};
        /// assert_eq!(backend_for("scale", SampleType::C32), "fallback");
        /// assert_eq!(backend_for("nope", SampleType::C32), "none");
        /// ```
        pub fn backend_for(kernel: &'static str, ty: $crate::SampleType) -> &'static str {
            const TABLE: &[(&str, $crate::SampleType, &str)] = &[
                $(
                    $(#[$meta])*
                    (
                        ::core::stringify!($m),
                        <$ty as $crate::Sample>::TYPE,
                        <$be as $crate::BackendName>::NAME,
                    ),
                )+
            ];
            for &(k, t, b) in TABLE {
                if t == ty && k == kernel {
                    return b;
                }
            }
            "none"
        }
    };
}
