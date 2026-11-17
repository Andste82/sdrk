//! `ParamMap` with one flat name for several paths (M44.1.1 §4): `set` checks every path, then
//! writes every path, and reaches all of them or none; `get` reads the first.

#![allow(clippy::unwrap_used)]

use sdrk::{C32, Progress, Rate};
use sdrk_blocks::{
    Block, BlockDescription, BlockTiming, Chain, Error, ParamMap, ParamPaths, ParamValue,
    ScaleBlock, ScaleConfig,
};
use sdrk_testkit::VecStorage;

type Vs = VecStorage;

/// A scale block whose `set_param` refuses a gain above 0 dB, which its `check_param` (the
/// default, from the description) lets through: a write that fails after every check passed.
struct Picky(ScaleBlock<C32, Vs>);

impl Block for Picky {
    type In = C32;
    type Out = C32;
    fn description(&self) -> &'static BlockDescription {
        self.0.description()
    }
    fn process(&mut self, i: &[C32], o: &mut [C32]) -> Result<Progress, Error> {
        self.0.process(i, o)
    }
    fn timing(&self) -> BlockTiming {
        self.0.timing()
    }
    fn reset(&mut self) {
        self.0.reset();
    }
    fn set_param(&mut self, n: &str, v: ParamValue) -> Result<(), Error> {
        if v.as_f64()? > 0.0 {
            return Err(Error::OutOfRange {
                name: "gain_db",
                min: -120.0,
                max: 0.0,
                got: v.as_f64()?,
            });
        }
        self.0.set_param(n, v)
    }
    fn param(&self, n: &str) -> Result<ParamValue, Error> {
        self.0.param(n)
    }
    fn min_input(&self) -> usize {
        self.0.min_input()
    }
}

fn scale() -> ScaleBlock<C32, Vs> {
    ScaleBlock::new(
        &ScaleConfig::default(),
        Rate::hz(48_000).unwrap(),
        &mut VecStorage,
    )
    .unwrap()
}

const MAP: ParamMap = ParamMap::new(&[
    ("gain_db", "0.gain_db"),
    ("other_db", "1.0.gain_db"),
    ("gain_db", "1.1.gain_db"),
]);

type Stages = Chain<ScaleBlock<C32, Vs>, Chain<ScaleBlock<C32, Vs>, Picky, Vs>, Vs>;

fn stages() -> Stages {
    let inner = Chain::new(scale(), Picky(scale()), 16, &mut VecStorage).unwrap();
    Chain::new(scale(), inner, 16, &mut VecStorage).unwrap()
}

#[test]
fn one_flat_name_reaches_every_path_in_the_order_of_the_table() {
    let mut c = stages();
    assert!(MAP.paths("gain_db").eq(["0.gain_db", "1.1.gain_db"]));
    assert_eq!(MAP.path("gain_db"), Some("0.gain_db"));
    assert_eq!(MAP.check(&c, "gain_db", &ParamValue::F64(-6.0)), Ok(()));
    assert_eq!(
        MAP.check(&c, "gain", &ParamValue::F64(-6.0)),
        Err(Error::UnknownParameter)
    );
    MAP.set(&mut c, "gain_db", ParamValue::F64(-6.0)).unwrap();
    for path in MAP.paths("gain_db") {
        assert_eq!(c.path_param(path).unwrap(), ParamValue::F64(-6.0), "{path}");
    }
    // The other name keeps its own path.
    assert_eq!(c.param("1.0.gain_db").unwrap(), ParamValue::F64(0.0));
    assert_eq!(MAP.get(&c, "gain_db").unwrap(), ParamValue::F64(-6.0));
    MAP.set(&mut c, "other_db", ParamValue::F64(-3.0)).unwrap();
    assert_eq!(MAP.get(&c, "other_db").unwrap(), ParamValue::F64(-3.0));
    assert_eq!(MAP.get(&c, "gain_db").unwrap(), ParamValue::F64(-6.0));
}

#[test]
fn a_value_that_one_building_block_refuses_reaches_none_of_them() {
    let mut c = stages();
    MAP.set(&mut c, "gain_db", ParamValue::F64(-6.0)).unwrap();
    // Refused by the check of every path (outside the range of the description): nothing is
    // written.
    assert!(matches!(
        MAP.set(&mut c, "gain_db", ParamValue::F64(-130.0)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(
        MAP.set(&mut c, "gain_db", ParamValue::I64(1)),
        Err(Error::WrongParamType)
    );
    // Refused by the write of the second path after both checks passed: the first path gets its
    // value back.
    assert!(matches!(
        MAP.set(&mut c, "gain_db", ParamValue::F64(3.0)),
        Err(Error::OutOfRange { max, .. }) if max == 0.0
    ));
    for path in MAP.paths("gain_db") {
        assert_eq!(c.path_param(path).unwrap(), ParamValue::F64(-6.0), "{path}");
    }
    assert_eq!(
        MAP.set(&mut c, "gain", ParamValue::F64(0.0)),
        Err(Error::UnknownParameter)
    );
}

#[test]
fn check_param_changes_nothing_and_follows_the_paths() {
    let c = stages();
    assert_eq!(c.check_param("1.1.gain_db", &ParamValue::F64(-1.0)), Ok(()));
    assert!(matches!(
        c.check_param("0.gain_db", &ParamValue::F64(50.0)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(
        c.check_param("2.gain_db", &ParamValue::F64(0.0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(
        c.check_param("0.gain", &ParamValue::F64(0.0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(c.param("0.gain_db").unwrap(), ParamValue::F64(0.0));
}
