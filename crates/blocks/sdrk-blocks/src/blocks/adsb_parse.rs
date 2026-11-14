//! `adsb_parse`: the frames of `adsb_decode` to aircraft messages as JSON text packets. The
//! block reads the extended squitters with [`sdrk_proto::adsb::parse`] and keeps the last even
//! and odd position frame of every aircraft in a small table, so that the global CPR decoding
//! finds a latitude and longitude without a reference position.

use core::fmt::Write as _;

use sdrk::{Delay, Progress, Ratio, RatioKind};
use sdrk_proto::adsb::{self, Body, Horizontal, Message, TargetState, Velocity};
use sdrk_proto::sensors::JsonBuf;
use sdrk_proto::{Packet, PacketFlags};

use super::unknown;
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// Aircraft the table remembers; a new one replaces the one seen longest ago.
pub const ADSB_PARSE_AIRCRAFT: usize = 64;

/// The sample rate of the frame indices (the rate `adsb_decode` accepts).
const RATE_HZ: f64 = 2_000_000.0;
/// Words of one table entry: address and flags, even position, even index, odd position, odd
/// index, index of the last frame.
const WORDS: usize = 6;
/// One aircraft: address and flags, even position, even index, odd position, odd index, index
/// of the last frame.
type Entry = [u64; WORDS];
const USED: u64 = 1 << 32;
const HAS_EVEN: u64 = 1 << 33;
const HAS_ODD: u64 = 1 << 34;
const NON_ICAO: u64 = 1 << 35;
const SURFACE: u64 = 1 << 36;
const FEET_PER_METRE: f64 = 3.280_839_895;

/// Description of the `adsb_parse` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "adsb_parse",
    summary: "Parses the frames of adsb_decode into aircraft messages as JSON text packets.",
    details: "Reads the frames of adsb_decode and decodes the extended squitters whose CRC is \
        correct: DF 17, and DF 18 with the control field 0 or 6 (ICAO address) or 1 or 5 \
        (non-ICAO address: the line has non_icao true). Decoded: the aircraft identification \
        (type codes 1 to 4: flight, category), the airborne position (9 to 18 with the \
        barometric altitude alt_baro in feet, 20 to 22 with the GNSS height alt_geom in feet, \
        0 with the altitude only), the surface position (5 to 8: ground true, gs in knots, \
        track), the airborne velocity (19: gs in knots and track in degrees, or mag_heading and \
        ias or tas; baro_rate or geom_rate in ft/min, geom_delta in ft), the aircraft status \
        (28: emergency and squawk), the target state (29: nav_altitude_mcp or nav_altitude_fms, \
        nav_qnh, nav_heading, nav_modes, nac_p, nic_baro, sil) and the operational status (31: \
        version, nac_p, sil, nic_a, gva). One packet of JSON text per message (fields time, \
        hex, icao, df, tc, then the values; hex is the address in lower case, icao the same as \
        a number; time is the frame index in seconds at 2 MS/s since construction or reset) \
        with the index of the frame. A position message also gets lat and lon: the block keeps \
        the last even and the last odd position frame of the 64 most recently seen aircraft \
        and decodes an airborne pair globally (CPR) when the other frame is at most max_age_s \
        older or newer; without a partner frame, lat and lon come from the local decoding with \
        the reference position (ref_lat, ref_lon; a reference of 0, 0 switches it off), else \
        they are left out. A surface position is ambiguous by quadrant (the coding repeats \
        every 90 degrees) and needs the reference position, within 45 NM, for the pair and for \
        a single frame. Other type codes and subtypes (23 to 27, 30), short frames and other \
        DF 18 control fields give no packet. A line is cut to the 256 bytes of a packet by \
        leaving out its last optional fields. Pitfall: the first airborne position frame of \
        an aircraft has no lat and lon unless a reference position is set. Pitfall: a pair \
        that straddles the border of a longitude zone is refused and waits for the next \
        frame. Cost: 60 per frame.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Packets],
        description: "The frames of adsb_decode (14 bytes of DF 17 and 18; other frames are \
            skipped).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "One packet per decoded message: the JSON text, the index of the frame.",
    }],
    params: &[
        ParamDesc {
            name: "ref_lat",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: -90.0,
                max: 90.0,
                default: Some(0.0),
            },
            typical: "0 (off); the receiver latitude in degrees, north positive",
            mutability: Mutability::Build,
            description: "Latitude of the reference position for the local decoding of a \
                position frame without a partner; the aircraft must be within 180 NM.",
        },
        ParamDesc {
            name: "ref_lon",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: -180.0,
                max: 180.0,
                default: Some(0.0),
            },
            typical: "0 (off); the receiver longitude in degrees, east positive",
            mutability: Mutability::Build,
            description: "Longitude of the reference position (see ref_lat); both 0 switch the \
                local decoding off.",
        },
        ParamDesc {
            name: "max_age_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 0.001,
                max: 3600.0,
                default: Some(10.0),
            },
            typical: "10; an aircraft moves little in that time",
            mutability: Mutability::Build,
            description: "The longest time between an even and an odd frame that are decoded \
                together.",
        },
    ],
    required: &[],
};

/// Configuration of [`AdsbParseBlock`].
///
/// ```
/// use sdrk_blocks::{AdsbParseConfig, ParamValue};
/// let mut cfg = AdsbParseConfig::default();
/// assert_eq!((cfg.ref_lat, cfg.ref_lon, cfg.max_age_s), (0.0, 0.0, 10.0));
/// cfg.set("ref_lat", ParamValue::F64(52.2))?;
/// assert!(cfg.set("ref_lat", ParamValue::F64(91.0)).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdsbParseConfig {
    /// Latitude of the reference position in degrees (0 with `ref_lon` 0: off).
    pub ref_lat: f64,
    /// Longitude of the reference position in degrees.
    pub ref_lon: f64,
    /// The longest time in seconds between the even and the odd frame of a global decoding.
    pub max_age_s: f64,
}

impl Default for AdsbParseConfig {
    fn default() -> Self {
        Self {
            ref_lat: 0.0,
            ref_lon: 0.0,
            max_age_s: 10.0,
        }
    }
}

impl AdsbParseConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` of `ref_lat`, `ref_lon` or `max_age_s`.
    pub fn validate(&self) -> Result<()> {
        for (name, v, min, max) in [
            ("ref_lat", self.ref_lat, -90.0, 90.0),
            ("ref_lon", self.ref_lon, -180.0, 180.0),
            ("max_age_s", self.max_age_s, 0.001, 3600.0),
        ] {
            if !(min..=max).contains(&v) {
                return Err(Error::OutOfRange {
                    name,
                    min,
                    max,
                    got: v,
                });
            }
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        let mut c = *self;
        match p.name {
            "ref_lat" => c.ref_lat = value.as_f64()?,
            "ref_lon" => c.ref_lon = value.as_f64()?,
            "max_age_s" => c.max_age_s = value.as_f64()?,
            _ => return unknown(),
        }
        c.validate()?;
        *self = c;
        Ok(())
    }

    /// Current value of a parameter.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "ref_lat" => Ok(ParamValue::F64(self.ref_lat)),
            "ref_lon" => Ok(ParamValue::F64(self.ref_lon)),
            "max_age_s" => Ok(ParamValue::F64(self.max_age_s)),
            _ => unknown(),
        }
    }
}

/// ADS-B parser: frame [`Packets`] in, [`Packets`] of JSON messages out.
///
/// ```
/// use sdrk_blocks::{AdsbParseBlock, AdsbParseConfig, ArenaStorage, Block, Packet};
/// use sdrk_proto::PacketFlags;
/// let cfg = AdsbParseConfig::default();
/// let mut mem = vec![0u8; AdsbParseBlock::<ArenaStorage<'_>>::storage_bytes(&cfg)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut parse = AdsbParseBlock::new(&cfg, &mut arena)?;
/// let frame = [0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98];
/// let input = [Packet::new(2_000_000, PacketFlags::CRC_OK, &frame)?];
/// let mut out = [Packet::default(); 1];
/// let p = parse.process(&input, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (1, 1));
/// let text = core::str::from_utf8(out[0].data())?;
/// assert!(text.contains("\"flight\":\"KLM1023\"") && text.contains("\"hex\":\"4840d6\""));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AdsbParseBlock<S: Storage> {
    config: AdsbParseConfig,
    /// The aircraft table, `WORDS` words per entry.
    table: S::Buf<Entry>,
    max_age: u64,
    local: bool,
    parsed: u64,
    undecoded: u64,
    ignored: u64,
    positions: u64,
    lost: u64,
}

impl<S: Storage> core::fmt::Debug for AdsbParseBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AdsbParseBlock")
            .field("config", &self.config)
            .field("parsed", &self.parsed)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> AdsbParseBlock<S> {
    /// Creates the block.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration, `Storage` if `storage` is too small.
    pub fn new(config: &AdsbParseConfig, storage: &mut S) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: *config,
            table: storage.alloc::<Entry>(ADSB_PARSE_AIRCRAFT)?,
            max_age: (config.max_age_s * RATE_HZ) as u64,
            local: config.ref_lat != 0.0 || config.ref_lon != 0.0,
            parsed: 0,
            undecoded: 0,
            ignored: 0,
            positions: 0,
            lost: 0,
        })
    }

    /// Exact bytes of storage the block needs.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &AdsbParseConfig) -> Result<usize> {
        config.validate()?;
        let mut plan = Plan::default();
        plan.add::<Entry>(ADSB_PARSE_AIRCRAFT)?;
        Ok(plan.bytes())
    }

    /// Messages written so far.
    pub fn parsed(&self) -> u64 {
        self.parsed
    }

    /// Valid frames with a type code that is not decoded.
    pub fn undecoded(&self) -> u64 {
        self.undecoded
    }

    /// Packets that were no extended squitter with a correct CRC.
    pub fn ignored(&self) -> u64 {
        self.ignored
    }

    /// Position messages that got a latitude and longitude.
    pub fn positions(&self) -> u64 {
        self.positions
    }

    /// Messages whose JSON text did not fit a packet.
    pub fn lost(&self) -> u64 {
        self.lost
    }

    /// The table entry of the address (the one seen longest ago when it is new). A non-ICAO
    /// address is a different aircraft than the same number as an ICAO address.
    fn entry(&mut self, icao: u32, non_icao: bool, index: u64) -> Option<&mut Entry> {
        let key = USED | u64::from(icao) | if non_icao { NON_ICAO } else { 0 };
        let t = self.table.as_mut();
        let slot = t
            .iter()
            .position(|e| e[0] & (USED | NON_ICAO | 0xFF_FFFF) == key)
            .or_else(|| t.iter().position(|e| e[0] & USED == 0))
            .or_else(|| {
                t.iter()
                    .enumerate()
                    .min_by_key(|(_, e)| e[5])
                    .map(|(i, _)| i)
            })?;
        let e = t.get_mut(slot)?;
        if e[0] & (USED | NON_ICAO | 0xFF_FFFF) != key {
            *e = [key, 0, 0, 0, 0, index];
        }
        e[5] = index;
        Some(e)
    }

    /// Stores the position frame (CPR coordinates `cpr` of the format `odd`) and returns the
    /// latitude and longitude when they can be decoded. Airborne frames decode globally from a
    /// pair, else locally with the reference; surface frames need the reference in both cases
    /// (D1433).
    fn locate(
        &mut self,
        m: &Message,
        index: u64,
        cpr: (u32, u32),
        odd: bool,
        surface: bool,
    ) -> Option<(f64, f64)> {
        let (max_age, local) = (self.max_age, self.local);
        let (rlat, rlon) = (self.config.ref_lat, self.config.ref_lon);
        let e = self.entry(m.icao, m.non_icao, index)?;
        let [head, even_pos, even_at, odd_pos, odd_at, _] = e;
        let (mine, mine_at, other, other_at, flag, other_flag) = if odd {
            (odd_pos, odd_at, even_pos, even_at, HAS_ODD, HAS_EVEN)
        } else {
            (even_pos, even_at, odd_pos, odd_at, HAS_EVEN, HAS_ODD)
        };
        // The coordinates of the airborne and the surface coding are not a pair.
        if (*head & SURFACE != 0) != surface {
            *head &= !(HAS_EVEN | HAS_ODD | SURFACE);
            *head |= if surface { SURFACE } else { 0 };
        }
        *mine = u64::from(cpr.0) | u64::from(cpr.1) << 17;
        *mine_at = index;
        *head |= flag;
        let split = |w: u64| ((w & 0x1FFFF) as u32, (w >> 17) as u32);
        let mut pos = None;
        if *head & other_flag != 0 && other_at.abs_diff(index) <= max_age && (local || !surface) {
            let (mine, theirs) = (split(*mine), split(*other));
            let (even, odd_frame) = if odd { (theirs, mine) } else { (mine, theirs) };
            pos = if surface {
                adsb::cpr_surface_global(even, odd_frame, odd, rlat, rlon)
            } else {
                adsb::cpr_global(even, odd_frame, odd)
            };
        }
        if pos.is_none() && local {
            pos = if surface {
                adsb::cpr_surface_local(cpr.0, cpr.1, odd, rlat, rlon)
            } else {
                adsb::cpr_local(cpr.0, cpr.1, odd, rlat, rlon)
            };
        }
        if pos.is_some() {
            self.positions += 1;
        }
        pos
    }

    /// Writes the JSON text of a message into `j`, with the position of a position message;
    /// false when the message has no line.
    fn json(&mut self, m: &Message, index: u64, j: &mut JsonBuf) -> bool {
        let _ = write!(
            j,
            "{{\"time\":{:.3},\"hex\":\"{:06x}\",\"icao\":{},\"df\":{},\"tc\":{}",
            index as f64 / RATE_HZ,
            m.icao,
            m.icao,
            m.df,
            m.type_code
        );
        if m.non_icao {
            let _ = j.write_str(",\"non_icao\":true");
        }
        match &m.body {
            Body::Identification(id) => put(
                j,
                format_args!(
                    ",\"flight\":\"{}\",\"category\":\"{}{}\"",
                    id.callsign_str(),
                    id.category.0 as char,
                    id.category.1
                ),
            ),
            Body::AltitudeOnly(a) => {
                if let Some(a) = a {
                    put(j, format_args!(",\"alt_baro\":{a}"));
                }
            }
            Body::Position(p) => {
                if let Some(a) = p.altitude {
                    if p.gnss {
                        let ft = libm::round(f64::from(a) * FEET_PER_METRE) as i32;
                        put(j, format_args!(",\"alt_geom\":{ft}"));
                    } else {
                        put(j, format_args!(",\"alt_baro\":{a}"));
                    }
                }
                let cpr = (p.cpr_lat, p.cpr_lon);
                if let Some((lat, lon)) = self.locate(m, index, cpr, p.odd, false) {
                    put(j, format_args!(",\"lat\":{lat:.5},\"lon\":{lon:.5}"));
                }
            }
            Body::Surface(s) => {
                put(j, format_args!(",\"ground\":true"));
                if let Some(g) = s.speed_kt {
                    put(j, format_args!(",\"gs\":{g}"));
                }
                if let Some(t) = s.track_deg {
                    put(j, format_args!(",\"track\":{t:.1}"));
                }
                let cpr = (s.cpr_lat, s.cpr_lon);
                if let Some((lat, lon)) = self.locate(m, index, cpr, s.odd, true) {
                    put(j, format_args!(",\"lat\":{lat:.5},\"lon\":{lon:.5}"));
                }
            }
            Body::Velocity(v) => velocity_json(j, v),
            Body::Status(s) => put(
                j,
                format_args!(
                    ",\"emergency\":\"{}\",\"squawk\":\"{:04}\"",
                    s.emergency_name(),
                    s.squawk
                ),
            ),
            Body::Target(t) => target_json(j, t),
            Body::Operational(o) => {
                if !o.airborne {
                    put(j, format_args!(",\"ground\":true"));
                }
                put(
                    j,
                    format_args!(
                        ",\"version\":{},\"nac_p\":{},\"sil\":{},\"nic_a\":{}",
                        o.version,
                        o.nac_p,
                        o.sil,
                        u8::from(o.nic_supplement)
                    ),
                );
                if let Some(g) = o.gva {
                    put(j, format_args!(",\"gva\":{g}"));
                }
            }
            _ => return false,
        }
        j.write_str("}").is_ok()
    }
}

fn velocity_json(j: &mut JsonBuf, v: &Velocity) {
    match v.horizontal {
        Horizontal::Ground { .. } => {
            if let (Some(g), Some(t)) = (v.horizontal.ground_speed_kt(), v.horizontal.track_deg()) {
                put(j, format_args!(",\"gs\":{g:.1},\"track\":{t:.1}"));
            }
        }
        Horizontal::Air {
            heading_deg,
            airspeed_kt,
            true_airspeed,
        } => {
            if let Some(h) = heading_deg {
                put(j, format_args!(",\"mag_heading\":{h:.1}"));
            }
            if let Some(s) = airspeed_kt {
                let key = if true_airspeed { "tas" } else { "ias" };
                put(j, format_args!(",\"{key}\":{s}"));
            }
        }
    }
    if let Some(r) = v.vertical_rate {
        let key = if v.vertical_rate_baro {
            "baro_rate"
        } else {
            "geom_rate"
        };
        put(j, format_args!(",\"{key}\":{r}"));
    }
    if let Some(d) = v.gnss_baro_diff {
        put(j, format_args!(",\"geom_delta\":{d}"));
    }
}

fn target_json(j: &mut JsonBuf, t: &TargetState) {
    if let Some(a) = t.selected_altitude {
        let key = if t.altitude_fms {
            "nav_altitude_fms"
        } else {
            "nav_altitude_mcp"
        };
        put(j, format_args!(",\"{key}\":{a}"));
    }
    if let Some(q) = t.baro_setting {
        put(j, format_args!(",\"nav_qnh\":{q:.1}"));
    }
    if let Some(h) = t.selected_heading {
        put(j, format_args!(",\"nav_heading\":{h:.1}"));
    }
    if let Some(md) = t.modes {
        let mut list = JsonBuf::new();
        for (on, name) in [
            (md.autopilot, "autopilot"),
            (md.vnav, "vnav"),
            (md.alt_hold, "althold"),
            (md.approach, "approach"),
            (md.lnav, "lnav"),
            (md.tcas, "tcas"),
        ] {
            if on {
                let sep = if list.is_empty() { "" } else { "," };
                let _ = write!(list, "{sep}\"{name}\"");
            }
        }
        if let Ok(s) = core::str::from_utf8(list.as_bytes()) {
            put(j, format_args!(",\"nav_modes\":[{s}]"));
        }
    }
    // The accuracy fields come last: they are left out when the line is full.
    put(
        j,
        format_args!(
            ",\"nac_p\":{},\"nic_baro\":{},\"sil\":{}",
            t.nac_p,
            u8::from(t.nic_baro),
            t.sil
        ),
    );
}

/// Appends `args` to the JSON text when it fits with the closing brace; else leaves it out
/// (the text stays valid JSON and under the 256 bytes of a packet).
fn put(j: &mut JsonBuf, args: core::fmt::Arguments<'_>) {
    let mut piece = JsonBuf::new();
    let fits = piece.write_fmt(args).is_ok() && j.len() + piece.len() < JsonBuf::CAPACITY;
    if let (true, Ok(s)) = (fits, core::str::from_utf8(piece.as_bytes())) {
        let _ = j.write_str(s);
    }
}

impl<S: Storage> Block for AdsbParseBlock<S> {
    type In = Packets;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[Packet], output: &mut [Packet]) -> Result<Progress> {
        let mut consumed = 0;
        let mut produced = 0;
        for packet in input {
            if produced == output.len() {
                break;
            }
            consumed += 1;
            let Some(m) = adsb::parse(packet.data()) else {
                self.ignored += 1;
                continue;
            };
            if matches!(m.body, Body::NotDecoded(_)) {
                self.undecoded += 1;
                continue;
            }
            let mut j = JsonBuf::new();
            if !self.json(&m, packet.index, &mut j) {
                self.lost += 1;
                continue;
            }
            match Packet::new(packet.index, PacketFlags::CRC_OK, j.as_bytes()) {
                Ok(p) => {
                    if let Some(slot) = output.get_mut(produced) {
                        *slot = p;
                    }
                    produced += 1;
                    self.parsed += 1;
                }
                Err(_) => self.lost += 1,
            }
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::ONE,
            kind: RatioKind::Packets {
                max_per_input: Ratio::ONE,
            },
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.table.as_mut().fill([0; WORDS]);
        self.parsed = 0;
        self.undecoded = 0;
        self.ignored = 0;
        self.positions = 0;
        self.lost = 0;
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(_) => Err(Error::ReadOnlyParameter),
            None => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        match index {
            0 => Some(("parsed", self.parsed as f64)),
            1 => Some(("undecoded", self.undecoded as f64)),
            2 => Some(("ignored", self.ignored as f64)),
            3 => Some(("positions", self.positions as f64)),
            4 => Some(("lost", self.lost as f64)),
            _ => None,
        }
    }
}
