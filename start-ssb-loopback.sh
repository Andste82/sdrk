#!/usr/bin/env bash
# Starts the simulated SSB voice link: microphone -> ssb_transmitter flow -> simulated radio
# link (ssb_channel_receiver flow) -> sound card. No HackRF or RTL-SDR is needed.
#
#   ./start-ssb-loopback.sh                          # speak into the default microphone
#   ./start-ssb-loopback.sh -i test.wav              # a WAV file instead of the microphone
#   ./start-ssb-loopback.sh -i test.wav -o rx.wav    # write the received audio to a file
#
# Ctrl+C stops it. Needs the Linux sdrk-tx binary in the same folder as this script (or in PATH;
# set SDRK_TX to use another one).
set -euo pipefail

input=""
out=""
while getopts "i:o:h" opt; do
    case "$opt" in
        i) input="$OPTARG" ;;
        o) out="$OPTARG" ;;
        h)
            sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) exit 2 ;;
    esac
done

here="$(cd "$(dirname "$0")" && pwd)"
exe="${SDRK_TX:-$here/sdrk-tx}"
if [[ ! -x "$exe" ]]; then
    exe="$(command -v sdrk-tx || true)"
fi
if [[ -z "$exe" || ! -x "$exe" ]]; then
    echo "sdrk-tx not found next to this script or in PATH (set SDRK_TX)." >&2
    exit 1
fi

tx_flow="$here/examples/flows/ssb_transmitter.yaml"
rx_flow="$here/examples/flows/ssb_channel_receiver.yaml"
for f in "$tx_flow" "$rx_flow"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing file: $f" >&2
        exit 1
    fi
done

args=(--loopback --flow "$tx_flow" --rx-flow "$rx_flow")
if [[ -n "$input" ]]; then
    args+=(--input "$input")
else
    args+=(--mic)
fi
if [[ -n "$out" ]]; then
    args+=(--out "$out")
fi

echo "Starting the simulated SSB link (Ctrl+C to stop)..."
exec "$exe" "${args[@]}"
