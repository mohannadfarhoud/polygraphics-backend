param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [ValidateSet("vit_b", "vit_l", "vit_h")]
    [string]$SamModel = "vit_b",
    [ValidateSet("224_linear", "512_dpt")]
    [string]$Dust3rModel = "224_linear",
    [string]$ModelsRoot = "C:\polyGraphics\models",
    [string]$ThirdPartyRoot = "C:\polyGraphics\third_party",
    [switch]$SkipTorch,
    [switch]$SkipDust3r,
    [switch]$SkipSam,
    [switch]$SkipCheckpoints
)

# One-shot installer for the optional ML stack (PyTorch CPU + SAM + DUSt3R)
# plus their model checkpoints. Idempotent: re-running only does missing steps.
# Run from an Administrator PowerShell with the project's venv in place.

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

# ---------- 3. DUSt3R ----------
$dust3rRepo = Join-Path $ThirdPartyRoot "dust3r"
if (-not $SkipDust3r) {
    Step "Cloning + installing DUSt3R into $dust3rRepo"
    if (-not (Test-Path $ThirdPartyRoot)) { New-Item -ItemType Directory -Path $ThirdPartyRoot | Out-Null }

    if (-not (Test-Path $dust3rRepo)) {
        git clone --recursive https://github.com/naver/dust3r.git $dust3rRepo
    } else {
        Push-Location $dust3rRepo
        git submodule update --init --recursive
        Pop-Location
    }

    Push-Location $dust3rRepo
    if (Test-Path "requirements.txt") {
        & $venvPython -m pip install -r "requirements.txt"
    }
    & $venvPython -m pip install -e .
    Pop-Location
} else {
    Step "Skipping DUSt3R package install (-SkipDust3r)"
}

# ---------- 4. Model checkpoints ----------
$samCheckpoints = @{
    "vit_b" = @{ name = "sam_vit_b_01ec64.pth"; url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" }
    "vit_l" = @{ name = "sam_vit_l_0b3195.pth"; url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" }
    "vit_h" = @{ name = "sam_vit_h_4b8939.pth"; url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" }
}

$dust3rCheckpoints = @{
    "224_linear" = @{ name = "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"; url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth" }
    "512_dpt"    = @{ name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth";    url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" }
}

$samDir = Join-Path $ModelsRoot "sam"
$dust3rDir = Join-Path $ModelsRoot "dust3r"
$samCkpt = Join-Path $samDir $samCheckpoints[$SamModel].name
$dust3rCkpt = Join-Path $dust3rDir $dust3rCheckpoints[$Dust3rModel].name

if (-not $SkipCheckpoints) {
    if (-not (Test-Path $samDir))    { New-Item -ItemType Directory -Path $samDir    | Out-Null }
    if (-not (Test-Path $dust3rDir)) { New-Item -ItemType Directory -Path $dust3rDir | Out-Null }

    if (-not (Test-Path $samCkpt)) {
        Step "Downloading SAM checkpoint ($SamModel) -> $samCkpt"
        Invoke-WebRequest -Uri $samCheckpoints[$SamModel].url -OutFile $samCkpt
    } else {
        Step "SAM checkpoint already present: $samCkpt"
    }

    if (-not (Test-Path $dust3rCkpt)) {
        Step "Downloading DUSt3R checkpoint ($Dust3rModel) -> $dust3rCkpt"
        Invoke-WebRequest -Uri $dust3rCheckpoints[$Dust3rModel].url -OutFile $dust3rCkpt
    } else {
        Step "DUSt3R checkpoint already present: $dust3rCkpt"
    }
} else {
    Step "Skipping checkpoint downloads (-SkipCheckpoints)"
}

# ---------- 5. Sanity check imports ----------
Step "Verifying imports inside the venv"
& $venvPython -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())"
& $venvPython -c "from segment_anything import sam_model_registry; print('segment-anything ok')"
& $venvPython -c "from dust3r.inference import inference; print('dust3r ok')"

Write-Host ""
Step "Done. Apply paths to the API via PUT /settings, e.g.:"
Write-Host @"
  {
    "reconstruction_backend": "dust3r",
    "device": "cpu",
    "sam_model_type": "$SamModel",
    "sam_checkpoint_path": "$samCkpt",
    "dust3r_checkpoint_path": "$dust3rCkpt",
    "allow_placeholder_pipeline": false,
    "max_image_side": 512
  }
"@ -ForegroundColor Yellow
Write-Host "Then: Restart-Service polygraphics" -ForegroundColor Yellow
