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

.EXAMPLE
  .\scripts\install_gaussian_splatting_windows.ps1

.EXAMPLE
  .\scripts\install_gaussian_splatting_windows.ps1 -TorchCudaIndexUrl "https://download.pytorch.org/whl/cu118"
#>

param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$ThirdPartyRoot = "C:\polyGraphics\third_party",
    [switch]$SkipTorchCuda,
    [string]$TorchCudaIndexUrl = "https://download.pytorch.org/whl/cu124",
    [switch]$SkipSubmoduleBuild
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtualenv not found at $venvPython. Run scripts\setup_windows.ps1 first."
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

    Step "Building diff-gaussian-rasterization (CUDA - requires nvcc)"
    Push-Location $dgr
    & $venvPython -m pip install --no-build-isolation .
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Host ""
        Write-Host "BUILD FAILED. Typical fixes:" -ForegroundColor Red
        Write-Host "  - Install CUDA Toolkit; open a new shell and run: nvcc --version" -ForegroundColor Yellow
        Write-Host "  - Install VS 2022 Build Tools (C++ workload)" -ForegroundColor Yellow
        Write-Host "  - Use -TorchCudaIndexUrl that matches your CUDA major version" -ForegroundColor Yellow
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
  "colmap_binary_path": "C:\\\\COLMAP\\\\COLMAP.bat",
  "gs_init_source": "colmap",
  "device": "cuda",
  "allow_placeholder_pipeline": false
}
"@ -ForegroundColor Yellow
Write-Host "Restart the API service after saving settings." -ForegroundColor Yellow
