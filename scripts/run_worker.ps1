param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
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

# TripoSR must use THIS machine's Python. Copied .env.worker paths from another PC often break.
$tripoPy = [System.Environment]::GetEnvironmentVariable("POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON")
if (-not [string]::IsNullOrWhiteSpace($tripoPy) -and -not (Test-Path $tripoPy)) {
    Write-Host "POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON not found ($tripoPy) - using local venv: $venvPython" -ForegroundColor Yellow
    $env:POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON = $venvPython
    $env:TRIPO_TRIPOSR_PYTHON = $venvPython
}

& $venvPython -m worker.remote_worker
