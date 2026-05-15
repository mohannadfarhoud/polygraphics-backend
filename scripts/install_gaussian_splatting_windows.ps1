<#
.SYNOPSIS
  Clone graphdeco-inria/gaussian-splatting (with submodules) and build CUDA extensions
  inside the project virtualenv so the API can run reconstruction_backend=gaussian_splatting.

.REQUIREMENTS (official repo - same as upstream)
  - NVIDIA GPU + driver
  - CUDA Toolkit installed; nvcc must be on PATH (version should match the PyTorch CUDA wheel)
  - Visual Studio 2022 Build Tools with "Desktop development with C++" workload
  - Git

  CPU-only servers cannot build or run official 3DGS training in a practical way.

.PARAMETER TorchCudaIndexUrl
  PyTorch wheel index for CUDA (default cu124). Use cu118 if your toolkit is 11.8.

.PARAMETER PythonExe
  Optional full path to python.exe. Use this to install extensions into a *worker* virtualenv
  when train.py is not run from the backend project's .venv (e.g. polygraph_worker\.venv).
  Combine with -SkipTorchCuda if that venv already has the right CUDA PyTorch.

.PARAMETER TorchCudaArchList
  Optional; sets TORCH_CUDA_ARCH_LIST for the submodule build (e.g. 8.6 for Ampere / RTX 30xx).

.EXAMPLE
  .\scripts\install_gaussian_splatting_windows.ps1

.EXAMPLE
  .\scripts\install_gaussian_splatting_windows.ps1 -TorchCudaIndexUrl "https://download.pytorch.org/whl/cu118"

.EXAMPLE
  # Build only the CUDA wheels into your *worker* venv (API / MapAnything python already OK):
  .\scripts\install_gaussian_splatting_windows.ps1 -SkipTorchCuda `
    -PythonExe "C:\Users\me\polygraph_worker\.venv\Scripts\python.exe"
#>

param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$ThirdPartyRoot = "C:\polyGraphics\third_party",
    [string]$PythonExe = "",
    [switch]$SkipTorchCuda,
    [string]$TorchCudaIndexUrl = "https://download.pytorch.org/whl/cu124",
    # e.g. 8.6 (Ampere RTX30), 8.9 (Ada RTX40), 7.5 (Turing). Omit to let setuptools guess (can fail).
    [string]$TorchCudaArchList = "",
    [switch]$SkipSubmoduleBuild
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if ($PythonExe) {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        Write-Error "PythonExe not found: $PythonExe"
    }
    $venvPython = [string](Resolve-Path -LiteralPath $PythonExe).Path
}
else {
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Error "Virtualenv not found at $venvPython. Run scripts\setup_windows.ps1 first or pass -PythonExe to your worker ``.venv\Scripts\python.exe``."
    }
}

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Run-Pip([string[]]$Args) {
    & $venvPython -m pip @Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$gsRepo = Join-Path $ThirdPartyRoot "gaussian-splatting"

if (-not (Test-Path $ThirdPartyRoot)) {
    New-Item -ItemType Directory -Path $ThirdPartyRoot | Out-Null
}

Step "Cloning Gaussian Splatting at $gsRepo (recursive submodules)"
if (-not (Test-Path $gsRepo)) {
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git $gsRepo
} else {
    Push-Location $gsRepo
    git submodule update --init --recursive
    Pop-Location
}

if (-not $SkipTorchCuda) {
    Step "Installing PyTorch + torchvision with CUDA ($TorchCudaIndexUrl)"
    Run-Pip @("install", "--upgrade", "pip", "setuptools", "wheel")
    Run-Pip @("install", "torch", "torchvision", "--index-url", $TorchCudaIndexUrl)
} else {
    Step "Skipping PyTorch CUDA reinstall (-SkipTorchCuda)"
}

Step "Installing lightweight Python deps for train.py"
$req = Join-Path $gsRepo "requirements.txt"
if (Test-Path $req) {
    Run-Pip @("install", "-r", $req)
} else {
    Run-Pip @("install", "--upgrade", "plyfile", "tqdm")
}

if ($SkipSubmoduleBuild) {
    Write-Host ""
    Write-Host "Skipped building CUDA extensions (-SkipSubmoduleBuild). Build manually:" -ForegroundColor Yellow
    Write-Host "  cd `"$gsRepo\submodules\diff-gaussian-rasterization`"" -ForegroundColor Yellow
    Write-Host "  & `"$venvPython`" -m pip install ." -ForegroundColor Yellow
    Write-Host "  cd `"..\simple-knn`"" -ForegroundColor Yellow
    Write-Host "  & `"$venvPython`" -m pip install ." -ForegroundColor Yellow
} else {
    $dgr = Join-Path $gsRepo "submodules\diff-gaussian-rasterization"
    $skn = Join-Path $gsRepo "submodules\simple-knn"
    if (-not (Test-Path $dgr)) {
        Write-Error "Missing submodule: $dgr - run: cd `"$gsRepo`"; git submodule update --init --recursive"
    }

    # Reuse the active Visual Studio developer prompt environment for setuptools builds.
    $env:DISTUTILS_USE_SDK = "1"

    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        Write-Error @"
cl.exe is not on PATH. GPU extension builds need MSVC (Visual Studio C++ toolset). nvcc will fail with: Cannot find compiler 'cl.exe'.

Option A: Start Menu -> **x64 Native Tools Command Prompt for VS 2022**, cd to this repo, rerun this script.

Option B: From PowerShell, run pip through:
  .\scripts\invoke_vs_build_tools.ps1 "<venv>\Scripts\python.exe" -m pip install --no-build-isolation "<gs>\submodules\diff-gaussian-rasterization"
"@
    }

    Step "Pre-build diagnostics (Torch vs CUDA toolkit — compare major versions)"
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        Write-Host ("nvcc:" + " " + $nvcc.Source) -ForegroundColor Gray
        & nvcc --version
    }
    else {
        Write-Host "WARNING: nvcc not on PATH. Open 'x64 Native Tools Command Prompt for VS 2022' (or VS Developer PowerShell) and rerun." -ForegroundColor Yellow
    }
    Write-Host "--- PyTorch ---" -ForegroundColor Gray
    & $venvPython -c @"
import torch
try:
    v = getattr(torch.version, 'cuda', None)
except Exception:
    v = None
print('torch', torch.__version__, 'torch.version.cuda', v)
print('cuda_available', torch.cuda.is_available())
"@
    if ($TorchCudaArchList) {
        $env:TORCH_CUDA_ARCH_LIST = $TorchCudaArchList
        Write-Host "Using TORCH_CUDA_ARCH_LIST=$($env:TORCH_CUDA_ARCH_LIST)" -ForegroundColor Gray
    }
    elseif ($env:TORCH_CUDA_ARCH_LIST) {
        Write-Host "TORCH_CUDA_ARCH_LIST=$($env:TORCH_CUDA_ARCH_LIST) (inherited from shell)" -ForegroundColor Gray
    }
    else {
        Write-Host "Tip: if nvcc dies with unclear arch errors, pass -TorchCudaArchList (e.g. 8.6 for RTX 30xx)." -ForegroundColor DarkYellow
    }

    Step "Building diff-gaussian-rasterization (CUDA - requires nvcc)"
    Push-Location $dgr
    & $venvPython -m pip install --no-build-isolation .
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Host ""
        Write-Host "BUILD FAILED — scroll PIP OUTPUT ABOVE for the first nvcc/cl error line." -ForegroundColor Red
        Write-Host "Typical fixes (Windows):" -ForegroundColor Yellow
        Write-Host "  1) Run from 'x64 Native Tools Command Prompt for VS 2022' so cl.exe/msvc env is initialized." -ForegroundColor Yellow
        Write-Host "  2) Match PyTorch to your toolkit: reinstall torch from the cu12x/cu118 index that matches CUDA on PATH;" -ForegroundColor Yellow
        Write-Host "     e.g. cu121 wheel + CUDA Toolkit 12.1+; set CUDA_HOME to that toolkit root if needed." -ForegroundColor Yellow
        Write-Host '  3) Try explicit arch: .\scripts\install_gaussian_splatting_windows.ps1 ... -TorchCudaArchList "8.6"' -ForegroundColor Yellow
        Write-Host "  4) If MSVC is 'unsupported' for your CUDA revision, patch CUDA toolkit or upgrade CUDA to match CUDA-MSVS matrix." -ForegroundColor Yellow
        exit $LASTEXITCODE
    }
    Pop-Location

    Step "Building simple-knn (CUDA)"
    Push-Location $skn
    & $venvPython -m pip install --no-build-isolation .
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        exit $LASTEXITCODE
    }
    Pop-Location
}

Step "Verifying torch CUDA inside venv"
& $venvPython -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device')"

Write-Host ""
Step "Done. Example PUT /settings body:"
$jsonPath = $gsRepo.Replace('\', '\\')
Write-Host @"
{
  "reconstruction_backend": "gaussian_splatting",
  "gs_repo_path": "$jsonPath",
  "device": "cuda",
  "allow_placeholder_pipeline": false
}
"@ -ForegroundColor Yellow
Write-Host "Restart the API service after saving settings." -ForegroundColor Yellow
