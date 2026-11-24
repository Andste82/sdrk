//! `psk_ber [symbols [first seed [seeds]]]` or `psk_ber acquisition [symbols [first seed [seeds
//! [sync_bw...]]]]`: the BER and alignment changes of `psk_demod` against GNU Radio's
//! receiver chain on the same signals (issue #169, §D); see `kernels::psk_ber`.

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("acquisition") {
        // `psk_ber acquisition [symbols [first seed [seeds [sync_bw...]]]]`
        args.remove(0);
        let n_sym = args.first().and_then(|a| a.parse().ok()).unwrap_or(8000);
        let first: u64 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(2701);
        let count: u64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(4);
        let mut bws: Vec<f64> = args.iter().skip(3).filter_map(|a| a.parse().ok()).collect();
        if bws.is_empty() {
            bws = vec![0.01, 0.0075, 0.005];
        }
        let seeds: Vec<u64> = (first..first + count).collect();
        print!(
            "{}",
            kernels::psk_ber::acquisition_report(n_sym, &seeds, &bws)
        );
        return;
    }
    let n_sym = args.first().and_then(|a| a.parse().ok()).unwrap_or(200_000);
    let first: u64 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(2601);
    let count: u64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(8);
    let seeds: Vec<u64> = (first..first + count).collect();
    print!("{}", kernels::psk_ber::report(n_sym, &seeds));
}
