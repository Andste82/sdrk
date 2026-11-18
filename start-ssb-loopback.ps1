# Starts the simulated SSB voice link: microphone -> ssb_transmitter flow -> simulated radio
# link (ssb_channel_receiver flow) -> sound card. No HackRF or RTL-SDR is needed.
#
#   .\start-ssb-loopback.ps1                  # speak into the default microphone
#   .\start-ssb-loopback.ps1 -Input test.wav  # a WAV file instead of the microphone
#   .\start-ssb-loopback.ps1 -Input test.wav -Out rx.wav   # write the received audio to a file
#
# Ctrl+C stops it. Needs sdrk-tx.exe in the same folder as this script.
param(
    [string]$Input = "",
    [string]$Out = ""
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$exe = Join-Path $here "sdrk-tx.exe"
$txFlow = Join-Path $here "examples\flows\ssb_transmitter.yaml"
$rxFlow = Join-Path $here "examples\flows\ssb_channel_receiver.yaml"

foreach ($f in @($exe, $txFlow, $rxFlow)) {
    if (-not (Test-Path $f)) { throw "Missing file: $f" }
}

$args_ = @("--loopback", "--flow", $txFlow, "--rx-flow", $rxFlow)
if ($Input -ne "") {
    $args_ += @("--input", (Resolve-Path $Input).Path)
} else {
    $args_ += "--mic"
}
if ($Out -ne "") {
    $args_ += @("--out", $Out)
}

Write-Host "Starting the simulated SSB link (Ctrl+C to stop)..."
& $exe @args_
