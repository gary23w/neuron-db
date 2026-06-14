# Start the real neuron-db server and open the demo pointed at it.  Right-click > Run with PowerShell.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$core = "..\..\rust\neuron-core"
$bin  = Join-Path $core "target\release\serve.exe"
if (-not (Test-Path $bin)) {
  if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Write-Host "building serve (first run)..."; Push-Location $core
    cargo build --release --features server --bin serve; Pop-Location
  } else {
    Write-Host "No serve.exe and no cargo. Install Rust (https://rustup.rs), or from the repo root run:  docker compose up -d"
    Read-Host "press enter to exit"; exit 1
  }
}
Write-Host "neuron-db -> http://localhost:8088  (db: $PWD\demo.db)"
Start-Process -FilePath $bin -ArgumentList "$PWD\demo.db","8088"
Start-Sleep -Seconds 1
Start-Process (Join-Path $PSScriptRoot "index.html")
Write-Host "neuron-db is running. Stop the 'serve' process (Task Manager) to end."
