param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [ValidateSet("vit_b", "vit_l", "vit_h")]
    [string]$SamModel = "vit_b",
    [string]$ModelsRoot = "C:\polyGraphics\models",
    [switch]$SkipTorch,
    [switch]$SkipSam,
    [switch]$SkipCheckpoints,
    [switch]$SkipMapAnything
)

# One-shot installer: PyTorch (CPU baseline) + SAM + Meta MapAnything (pip from GitHub) + SAM checkpoint.
# For GPU reconstruction, reinstall torch with your CUDA wheel (see PyTorch homepage) instead of cpu index.
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtualenv not found at $venvPython. Run scripts\setup_windows.ps1 first."
}

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# ---------- 1. PyTorch (CPU) ----------
if (-not $SkipTorch) {
    Step "Installing PyTorch (CPU build) and torchvision"
    & $venvPython -m pip install --upgrade pip setuptools wheel
    & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    Step "Skipping PyTorch install (-SkipTorch)"
}

# ---------- 2. SAM ----------
if (-not $SkipSam) {
    Step "Installing segment-anything (SAM)"
    & $venvPython -m pip install --upgrade git+https://github.com/facebookresearch/segment-anything.git
} else {
    Step "Skipping SAM package install (-SkipSam)"
}

# ---------- 3. MapAnything ----------
if (-not $SkipMapAnything) {
    Step "Installing MapAnything (facebookresearch/map-anything from GitHub; weights load from Hugging Face on first run)"
    & $venvPython -m pip install --upgrade "git+https://github.com/facebookresearch/map-anything.git"
} else {
    Step "Skipping MapAnything (-SkipMapAnything)"
}

# ---------- 4. Model checkpoints ----------
$samCheckpoints = @{
    "vit_b" = @{ name = "sam_vit_b_01ec64.pth"; url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" }
    "vit_l" = @{ name = "sam_vit_l_0b3195.pth"; url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" }
    "vit_h" = @{ name = "sam_vit_h_4b8939.pth"; url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" }
}

$samDir = Join-Path $ModelsRoot "sam"
$samCkpt = Join-Path $samDir $samCheckpoints[$SamModel].name

if (-not $SkipCheckpoints) {
    if (-not (Test-Path $samDir)) { New-Item -ItemType Directory -Path $samDir | Out-Null }

    if (-not (Test-Path $samCkpt)) {
        Step "Downloading SAM checkpoint ($SamModel) -> $samCkpt"
        Invoke-WebRequest -Uri $samCheckpoints[$SamModel].url -OutFile $samCkpt
    } else {
        Step "SAM checkpoint already present: $samCkpt"
    }
} else {
    Step "Skipping checkpoint downloads (-SkipCheckpoints)"
}

# ---------- 5. Sanity check imports ----------
Step "Verifying imports inside the venv"
& $venvPython -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())"
& $venvPython -c "from segment_anything import sam_model_registry; print('segment-anything ok')"
& $venvPython -c "import mapanything; print('mapanything ok')"

Write-Host ""
Step "Done. Apply paths to the API via PUT /settings, e.g.:"
Write-Host @"
  {
    "reconstruction_backend": "mapanything",
    "mapanything_pretrained_id": "facebook/map-anything-apache",
    "device": "cpu",
    "sam_model_type": "$SamModel",
    "sam_checkpoint_path": "$samCkpt",
    "allow_placeholder_pipeline": false,
    "max_image_side": 512
  }
"@ -ForegroundColor Yellow
Write-Host "Then restart the API service." -ForegroundColor Yellow
Write-Host "For GPU reconstruction: reinstall torch torchvision with CUDA, then rerun import check." -ForegroundColor DarkGray
Write-Host 'Optional - Gaussian Splatting (NVIDIA + CUDA + VS Build Tools): .\scripts\install_gaussian_splatting_windows.ps1' -ForegroundColor DarkGray
