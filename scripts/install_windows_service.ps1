param(
    [string]$ServiceName = "polyGraphicsBackend",
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$NssmPath = "C:\tools\nssm\nssm.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $NssmPath)) {
    throw "NSSM not found at $NssmPath. Download NSSM and update -NssmPath."
}

$runScript = Join-Path $ProjectRoot "scripts\run_server.ps1"

& $NssmPath install $ServiceName "powershell.exe" "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`" -ProjectRoot `"$ProjectRoot`""
& $NssmPath set $ServiceName AppDirectory $ProjectRoot
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath start $ServiceName

Write-Host "Service '$ServiceName' installed and started."
