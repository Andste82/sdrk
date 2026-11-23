//! `psk_ber [symbols [first seed [seeds]]]`: the BER and alignment changes of `psk_demod` against GNU Radio's
//! receiver chain on the same signals (issue #169, §D); see `kernels::psk_ber`.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_sym = args.first().and_then(|a| a.parse().ok()).unwrap_or(200_000);
    let first: u64 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(2601);
    let count: u64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(8);
    let seeds: Vec<u64> = (first..first + count).collect();
    print!("{}", kernels::psk_ber::report(n_sym, &seeds));
}
