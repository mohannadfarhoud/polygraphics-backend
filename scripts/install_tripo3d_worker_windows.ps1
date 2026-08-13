param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$TripoSrRepo = "C:\polyGraphics\third_party\TripoSR",
    [switch]$SkipTripoSrClone
)

# Install Tripo cloud SDK (tripo3d) into the worker venv and optionally clone TripoSR for
# POST /tripo/image-to-model jobs handled by the GPU worker (no cloud API key).

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtualenv missing at $venvPython. Run .\scripts\setup_windows.ps1 first."
}

Write-Host "==> Installing tripo3d (Tripo cloud SDK)" -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install "tripo3d>=0.3.0"

Write-Host "==> Verifying tripo3d import" -ForegroundColor Cyan
& $venvPython -c "from tripo3d import TripoClient; print('tripo3d OK', TripoClient)"

if (-not $SkipTripoSrClone) {
    if (-not (Test-Path $TripoSrRepo)) {
        Write-Host "==> Cloning TripoSR to $TripoSrRepo" -ForegroundColor Cyan
        $parent = Split-Path $TripoSrRepo -Parent
        if (-not (Test-Path $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        git clone https://github.com/VAST-AI-Research/TripoSR $TripoSrRepo
    } else {
        Write-Host "TripoSR repo already exists: $TripoSrRepo" -ForegroundColor DarkGray
    }

    $triposrReq = Join-Path $TripoSrRepo "requirements.txt"
    if (Test-Path $triposrReq) {
        Write-Host "==> Installing TripoSR Python dependencies (may take several minutes)" -ForegroundColor Cyan
        & $venvPython -m pip install -r $triposrReq
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARN: TripoSR requirements failed (often torchmcubes needs Visual Studio C++ build tools)." -ForegroundColor Yellow
            Write-Host "      Installing PyMCubes fallback and patching TripoSR isosurface..." -ForegroundColor Yellow
            & $venvPython -m pip install PyMCubes xatlas
            $iso = Join-Path $TripoSrRepo "tsr\models\isosurface.py"
            if (Test-Path $iso) {
                $src = Get-Content $iso -Raw
                if ($src -match 'from torchmcubes import marching_cubes' -and $src -notmatch 'import mcubes') {
                    $src = $src -replace 'from torchmcubes import marching_cubes', @'
try:
    from torchmcubes import marching_cubes
except ImportError:
    import mcubes

    def marching_cubes(vol, thresh):
        import numpy as np
        import torch
        v_pos, t_pos_idx = mcubes.marching_cubes(vol.detach().cpu().numpy(), float(thresh))
        v_pos = torch.from_numpy(v_pos.astype(np.float32))
        t_pos_idx = torch.from_numpy(t_pos_idx.astype(np.int64))
        return v_pos, t_pos_idx
'@
                    Set-Content -Path $iso -Value $src -NoNewline
                    Write-Host "Patched $iso to use PyMCubes when torchmcubes is missing." -ForegroundColor Green
                }
            }
        }
    }

    $workerEnv = Join-Path $ProjectRoot ".env.worker"
    if (Test-Path $workerEnv) {
        $content = Get-Content $workerEnv -Raw
        if ($content -notmatch 'POLYGRAPH_OVERRIDE_TRIPOSR_REPO=') {
            Add-Content $workerEnv @"

# TripoSR local image-to-3D (worker polls /internal/worker/tripo/next)
# Paths must exist on THIS GPU PC (do not copy another machine's Documents path).
POLYGRAPH_OVERRIDE_TRIPOSR_REPO=$TripoSrRepo
POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON=$venvPython
"@
            Write-Host "Added TripoSR paths to .env.worker" -ForegroundColor Yellow
        } else {
            # Rewrite python path if missing/wrong machine.
            $lines = Get-Content $workerEnv
            $changed = $false
            $newLines = foreach ($line in $lines) {
                if ($line -match '^\s*POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON=(.*)$') {
                    $cur = $Matches[1].Trim()
                    if (-not $cur -or -not (Test-Path $cur)) {
                        $changed = $true
                        "POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON=$venvPython"
                        continue
                    }
                }
                $line
            }
            if ($changed) {
                Set-Content -Path $workerEnv -Value $newLines
                Write-Host "Updated POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON -> $venvPython" -ForegroundColor Yellow
            }
        }
    }
}

Write-Host ""
Write-Host "Tripo worker install complete." -ForegroundColor Green
Write-Host "  tripo3d: installed in .venv (Tripo cloud API; set TRIPO_API_KEY in .env.worker)" -ForegroundColor DarkGray
Write-Host "  TripoSR: $TripoSrRepo (local /tripo jobs without API key)" -ForegroundColor DarkGray
Write-Host "Restart worker: .\scripts\run_worker.ps1" -ForegroundColor DarkGray
