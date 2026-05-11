param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [ValidateSet("cu121", "cu124", "cu118")]
    [string]$CudaTorchIndex = "cu121",
    [string]$NssmPath = "C:\tools\nssm\nssm.exe",
    [string]$ServiceName = "polyGraphicsWorker",
    [switch]$InstallService
)

# GPU worker PC (outbound-only): venv, CUDA PyTorch, SAM + DUSt3R via install_ml_windows.ps1 -SkipTorch,
# optional .env.worker from example.

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "==> Running scripts\setup_windows.ps1" -ForegroundColor Cyan
& "$PSScriptRoot\setup_windows.ps1" -ProjectRoot $ProjectRoot

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtualenv missing at $venvPython"
}

Write-Host "==> Installing PyTorch + torchvision ($CudaTorchIndex)" -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaTorchIndex"

Write-Host "==> SAM + DUSt3R stack (skipping duplicate torch install)" -ForegroundColor Cyan
& "$PSScriptRoot\install_ml_windows.ps1" -ProjectRoot $ProjectRoot -SkipTorch

$workerEnv = Join-Path $ProjectRoot ".env.worker"
$example = Join-Path $ProjectRoot ".env.worker.example"
if (-not (Test-Path $workerEnv)) {
    if (Test-Path $example) {
        Copy-Item $example $workerEnv
        Write-Host "Created $workerEnv — set POLYGRAPH_WORKER_TOKEN to match APP_WORKER_TOKEN on the API server." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Worker setup complete." -ForegroundColor Green
Write-Host "1. Edit $workerEnv (POLYGRAPH_API_BASE, POLYGRAPH_WORKER_TOKEN)." -ForegroundColor DarkGray
Write-Host "2. Align checkpoint paths with PUT /settings or POLYGRAPH_OVERRIDE_* vars." -ForegroundColor DarkGray
Write-Host "3. Run: .\scripts\run_worker.ps1" -ForegroundColor DarkGray

if ($InstallService) {
    if (-not (Test-Path $NssmPath)) {
        throw "NSSM not found at $NssmPath. Install NSSM or pass -NssmPath."
    }
    $runScript = Join-Path $ProjectRoot "scripts\run_worker.ps1"
    & $NssmPath install $ServiceName "powershell.exe" "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`""
    & $NssmPath set $ServiceName AppDirectory $ProjectRoot
    & $NssmPath set $ServiceName Start SERVICE_AUTO_START
    & $NssmPath start $ServiceName
    Write-Host "NSSM service '$ServiceName' installed and started." -ForegroundColor Green
}
