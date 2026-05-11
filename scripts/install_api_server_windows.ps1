param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$PublicApiBase = "https://agentmanager.easymediasuitecloud.com/polygraph",
    [switch]$RegenerateWorkerToken
)

# Idempotent installer for the HTTPS API host (CPU): venv, pip deps, and production-oriented .env keys
# for remote GPU workers (APP_REMOTE_WORKERS, APP_PUBLIC_BASE_URL, worker token).

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

function Merge-DotEnvKey {
    param(
        [string[]]$Lines,
        [string]$Key,
        [string]$Value
    )
    $pattern = "^\s*" + [regex]::Escape($Key) + "\s*="
    $out = New-Object System.Collections.ArrayList
    $found = $false
    foreach ($line in $Lines) {
        if ($line -match $pattern) {
            [void]$out.Add("${Key}=${Value}")
            $found = $true
        }
        else {
            [void]$out.Add($line)
        }
    }
    if (-not $found) {
        [void]$out.Add("${Key}=${Value}")
    }
    ,@($out.ToArray())
}

function New-WorkerToken {
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $bytes = New-Object byte[] 32
    $rng.GetBytes($bytes)
    [Convert]::ToBase64String($bytes)
}

Write-Host "==> Running scripts\setup_windows.ps1" -ForegroundColor Cyan
& "$PSScriptRoot\setup_windows.ps1" -ProjectRoot $ProjectRoot

$envPath = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") $envPath
}

$lines = @(Get-Content $envPath -ErrorAction SilentlyContinue)
if (-not $lines) { $lines = @() }

$PublicApiBase = $PublicApiBase.Trim().TrimEnd('/')
$existingToken = ""
foreach ($line in $lines) {
    if ($line -match '^\s*APP_WORKER_TOKEN\s*=\s*(.+)\s*$') {
        $existingToken = $matches[1].Trim()
    }
}
$token = if ($RegenerateWorkerToken) {
    New-WorkerToken
} elseif ($existingToken) {
    $existingToken
} else {
    New-WorkerToken
}

$lines = Merge-DotEnvKey $lines "APP_ROOT_PATH" "/polygraph"
$lines = Merge-DotEnvKey $lines "APP_PUBLIC_BASE_URL" $PublicApiBase
$lines = Merge-DotEnvKey $lines "APP_REMOTE_WORKERS" "true"
$lines = Merge-DotEnvKey $lines "APP_WORKER_TOKEN" $token
$lines = Merge-DotEnvKey $lines "APP_MODEL_BASE_URL" "$PublicApiBase/output"
$lines = Merge-DotEnvKey $lines "APP_UPLOADS_BASE_URL" "$PublicApiBase/uploads"

Set-Content -Path $envPath -Value ($lines -join "`r`n") -Encoding UTF8

Write-Host ""
Write-Host "API .env updated at $envPath" -ForegroundColor Green
if ((-not $existingToken) -or $RegenerateWorkerToken) {
    Write-Host "Copy this token to each GPU worker .env.worker as POLYGRAPH_WORKER_TOKEN:" -ForegroundColor Yellow
    Write-Host $token -ForegroundColor Yellow
} else {
    Write-Host "APP_WORKER_TOKEN left unchanged. Use -RegenerateWorkerToken to rotate (then update all workers)." -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "Next: proxy HTTPS so $PublicApiBase/* reaches this app, then run .\scripts\run_server.ps1 or install NSSM (scripts\install_windows_service.ps1)." -ForegroundColor DarkGray
Write-Host "Re-run with -RegenerateWorkerToken to issue a new APP_WORKER_TOKEN (update all workers)." -ForegroundColor DarkGray
