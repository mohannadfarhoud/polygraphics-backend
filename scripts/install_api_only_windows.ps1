param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$PublicApiBase = "https://agentmanager.easymediasuitecloud.com/polygraph",
    [switch]$RegenerateWorkerToken
)

# SQLite + FastAPI + uploads/output storage on this machine only.
# Set APP_REMOTE_WORKERS=true and APP_WORKER_TOKEN for GPU PCs running scripts\run_worker.ps1.
# Does NOT pass -SingleMachine (pipeline never runs in-process on this host).

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "API-only / split deploy - jobs run on a separate GPU worker." -ForegroundColor Cyan
Write-Host "This PC keeps the database and serves HTTPS; copy APP_WORKER_TOKEN to each worker .env.worker." -ForegroundColor DarkGray
Write-Host ""

$installer = Join-Path $PSScriptRoot "install_api_server_windows.ps1"
if ($RegenerateWorkerToken) {
    & $installer -ProjectRoot $ProjectRoot -PublicApiBase $PublicApiBase -RegenerateWorkerToken
} else {
    & $installer -ProjectRoot $ProjectRoot -PublicApiBase $PublicApiBase
}
