param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"

if (-not (Test-Path ".env")) {
    Copy-Item ".\.env.example" ".\.env"
}

Write-Host "Setup complete."
Write-Host "Edit .env if needed, then run .\scripts\run_server.ps1"
