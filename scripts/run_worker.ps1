param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}

$envFile = if (Test-Path ".\.env.worker") { ".\.env.worker" } elseif (Test-Path ".\.env") { ".\.env" } else { $null }
if ($envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $kv = $line.Split("=", 2)
            if ($kv.Count -eq 2) {
                [System.Environment]::SetEnvironmentVariable($kv[0].Trim(), $kv[1].Trim())
            }
        }
    }
}

& ".\.venv\Scripts\python.exe" -m worker.remote_worker
