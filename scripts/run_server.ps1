param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $kv = $line.Split("=", 2)
            if ($kv.Count -eq 2) {
                [System.Environment]::SetEnvironmentVariable($kv[0].Trim(), $kv[1].Trim())
            }
        }
    }
}

$HostValue = if ($env:APP_HOST) { $env:APP_HOST } else { "0.0.0.0" }
$PortValue = if ($env:APP_PORT) { $env:APP_PORT } else { "8000" }
$ReloadValue = if ($env:APP_RELOAD) { $env:APP_RELOAD } else { "false" }
$WorkersValue = if ($env:APP_WORKERS) { $env:APP_WORKERS } else { "1" }

$args = @(
    "-m", "uvicorn", "app.main:app",
    "--host", $HostValue,
    "--port", $PortValue,
    "--workers", $WorkersValue
)

if ($ReloadValue -eq "true") {
    $args += "--reload"
}

& ".\.venv\Scripts\python.exe" @args
