#!/usr/bin/env bash
# One-shot installer for the polygraphics-backend ML stack on Linux / Colab.
#
# Usage (Colab):
#   !bash scripts/colab_setup.sh
#
# Idempotent: re-running only does what's missing. Safe to run after kernel restart.
# Designed to NOT clobber Colab's preinstalled CUDA torch (DUSt3R requirements
# can otherwise downgrade you to torch+cpu).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
THIRD_PARTY="${THIRD_PARTY:-$REPO_DIR/.third_party}"
CHECKPOINTS="${CHECKPOINTS:-$REPO_DIR/.checkpoints}"
SAM_MODEL="${SAM_MODEL:-vit_b}"   # vit_b is small (~375 MB) and fast on Colab T4
INSTALL_COLMAP="${INSTALL_COLMAP:-1}" # set 0 to skip (apt is fast on Colab)

echo "==> [1/7] System packages..."
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg wget git build-essential unzip

if [ "$INSTALL_COLMAP" = "1" ]; then
  echo "==> [1b/7] COLMAP (apt) so the 'auto' / 'colmap' backend works out of the box..."
  $SUDO apt-get install -y -qq --no-install-recommends colmap || \
    echo "    (colmap apt install failed; the 'auto' backend will need reconstruction_backend=dust3r instead)"
fi

cd "$REPO_DIR"

echo "==> [2/7] Capture pre-existing torch (so we can restore it if anything overwrites it)..."
PRE_TORCH="$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.__version__)
except Exception:
    pass
PY
)"
echo "    pre-existing torch: '${PRE_TORCH:-<none>}'"

echo "==> [3/7] Python deps from requirements.txt..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "==> [4/7] DUSt3R (clone + register on sys.path)..."
mkdir -p "$THIRD_PARTY"
if [ ! -d "$THIRD_PARTY/dust3r" ]; then
  git clone --recursive https://github.com/naver/dust3r.git "$THIRD_PARTY/dust3r"
else
  (cd "$THIRD_PARTY/dust3r" && git submodule update --init --recursive)
fi
# IMPORTANT: install DUSt3R deps WITHOUT torch / torchvision lines — pip would
# otherwise pull torch+cpu from PyPI and overwrite Colab's CUDA build.
if [ -f "$THIRD_PARTY/dust3r/requirements.txt" ]; then
  TMP_REQ="$(mktemp)"
  grep -Ev '^[[:space:]]*(torch|torchvision)([[:space:]<>=!~].*)?$' \
    "$THIRD_PARTY/dust3r/requirements.txt" > "$TMP_REQ" || true
  python -m pip install --quiet -r "$TMP_REQ" || true
  rm -f "$TMP_REQ"
fi
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$THIRD_PARTY/dust3r" > "$SITE_PACKAGES/dust3r_repo.pth"
echo "    wrote $SITE_PACKAGES/dust3r_repo.pth"

# CroCo CUDA RoPE extension is optional (just speeds DUSt3R up). Best-effort.
if [ -d "$THIRD_PARTY/dust3r/croco/models/curope" ]; then
  (cd "$THIRD_PARTY/dust3r/croco/models/curope" && python setup.py build_ext --inplace 2>/dev/null) \
    || echo "    (curope native build skipped — using slower torch fallback)"
fi

echo "==> [5/7] Segment-Anything (SAM)..."
python -m pip install --quiet "git+https://github.com/facebookresearch/segment-anything.git"

echo "==> [6/7] Restore CUDA torch if anything above swapped it for +cpu..."
NOW_TORCH="$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.__version__)
except Exception:
    pass
PY
)"
echo "    after installs:    '${NOW_TORCH:-<none>}'"

# If we had a CUDA torch before and now we have +cpu (or torch went missing), try to recover.
if [[ -z "$NOW_TORCH" || ( "$NOW_TORCH" == *"+cpu" && "$PRE_TORCH" != *"+cpu" ) ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "    GPU detected — installing torch (CUDA 12.1 wheel) to replace +cpu..."
    python -m pip install --quiet --upgrade --force-reinstall \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121
  else
    echo "    No GPU runtime detected; keeping torch+cpu. (Switch Runtime > T4 GPU and re-run for CUDA.)"
  fi
fi

echo "==> [6b/7] SAM checkpoint ($SAM_MODEL)..."
mkdir -p "$CHECKPOINTS"
case "$SAM_MODEL" in
  vit_b) SAM_FILE=sam_vit_b_01ec64.pth ;;
  vit_l) SAM_FILE=sam_vit_l_0b3195.pth ;;
  vit_h) SAM_FILE=sam_vit_h_4b8939.pth ;;
  *) echo "Unknown SAM_MODEL=$SAM_MODEL"; exit 1 ;;
esac
SAM_CKPT="$CHECKPOINTS/$SAM_FILE"
if [ ! -f "$SAM_CKPT" ]; then
  wget --no-verbose -O "$SAM_CKPT" "https://dl.fbaipublicfiles.com/segment_anything/$SAM_FILE"
fi

# Locate COLMAP if installed.
COLMAP_BIN="$(command -v colmap || true)"

echo "==> [7/7] runtime_settings.json (DUSt3R via Hugging Face on first job)..."
mkdir -p "$REPO_DIR/config"
SAM_CKPT="$SAM_CKPT" SAM_MODEL="$SAM_MODEL" REPO_DIR="$REPO_DIR" COLMAP_BIN="$COLMAP_BIN" python <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["REPO_DIR"]) / "config" / "runtime_settings.json"
existing = {}
if p.is_file():
    try:
        existing = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
existing.update({
    "sam_checkpoint_path": os.environ["SAM_CKPT"],
    "sam_model_type": os.environ["SAM_MODEL"],
    "dust3r_checkpoint_path": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
    "device": "auto",
    # If COLMAP isn't on PATH, default to plain dust3r so the pipeline is
    # ready immediately (auto would otherwise fail readiness without colmap).
    "reconstruction_backend": "auto" if os.environ.get("COLMAP_BIN") else "dust3r",
    "max_images": 100,
    "allow_placeholder_pipeline": False,
})
if os.environ.get("COLMAP_BIN"):
    existing["colmap_binary_path"] = os.environ["COLMAP_BIN"]
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
print(f"    wrote {p}")
PY

echo
echo "==> Verifying imports..."
python - <<'PY'
import importlib, sys
for m in ("torch", "torchvision", "cv2", "numpy", "open3d", "trimesh", "plyfile",
          "fastapi", "uvicorn", "segment_anything", "dust3r.inference"):
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}", file=sys.stderr)
import torch
print(f"  torch                   = {torch.__version__}")
print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
print(f"  torch.cuda.device_name  = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu (no GPU runtime)'}")
PY

echo
echo "==> Setup complete."
echo "    If torch.cuda.is_available is False, set Runtime > Change runtime type > T4 GPU"
echo "    and re-run THIS cell so torch is reinstalled with CUDA wheels."
echo "    Then: bash scripts/colab_serve.sh"
