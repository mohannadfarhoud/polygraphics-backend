#!/usr/bin/env bash
# One-shot installer for the polygraphics-backend ML stack on Linux / Colab.
#
# Usage (Colab):
#   !bash scripts/colab_setup.sh
#
# Idempotent: re-running only does what's missing. Safe to run after kernel restart.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
THIRD_PARTY="${THIRD_PARTY:-$REPO_DIR/.third_party}"
CHECKPOINTS="${CHECKPOINTS:-$REPO_DIR/.checkpoints}"
SAM_MODEL="${SAM_MODEL:-vit_b}"   # vit_b is small (~375 MB) and fast on Colab T4

echo "==> [1/6] System packages (libGL, ffmpeg, build tools)..."
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg wget git build-essential

cd "$REPO_DIR"

echo "==> [2/6] Python deps (uses Colab's preinstalled torch + CUDA)..."
python -m pip install --quiet --upgrade pip
# Install torch only if it's missing (Colab already ships a CUDA-enabled build).
python - <<'PY' || python -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
PY
python -m pip install --quiet -r requirements.txt

echo "==> [3/6] DUSt3R (clone + register on sys.path)..."
mkdir -p "$THIRD_PARTY"
if [ ! -d "$THIRD_PARTY/dust3r" ]; then
  git clone --recursive https://github.com/naver/dust3r.git "$THIRD_PARTY/dust3r"
else
  (cd "$THIRD_PARTY/dust3r" && git submodule update --init --recursive)
fi
if [ -f "$THIRD_PARTY/dust3r/requirements.txt" ]; then
  python -m pip install --quiet -r "$THIRD_PARTY/dust3r/requirements.txt" || true
fi
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$THIRD_PARTY/dust3r" > "$SITE_PACKAGES/dust3r_repo.pth"
echo "    wrote $SITE_PACKAGES/dust3r_repo.pth"

# CroCo CUDA RoPE extension is optional (just speeds DUSt3R up). Best-effort build.
if [ -d "$THIRD_PARTY/dust3r/croco/models/curope" ]; then
  (cd "$THIRD_PARTY/dust3r/croco/models/curope" && python setup.py build_ext --inplace 2>/dev/null) \
    || echo "    (curope native build skipped — using slower torch fallback)"
fi

echo "==> [4/6] Segment-Anything (SAM)..."
python -m pip install --quiet "git+https://github.com/facebookresearch/segment-anything.git"

echo "==> [5/6] SAM checkpoint ($SAM_MODEL)..."
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

echo "==> [6/6] runtime_settings.json (HF download for DUSt3R weights on first job)..."
mkdir -p "$REPO_DIR/config"
python - <<PY
import json, os
from pathlib import Path
p = Path(os.environ.get("REPO_DIR", "$REPO_DIR")) / "config" / "runtime_settings.json"
existing = {}
if p.is_file():
    try: existing = json.loads(p.read_text(encoding="utf-8"))
    except Exception: pass
existing.update({
    "sam_checkpoint_path": "$SAM_CKPT",
    "sam_model_type": "$SAM_MODEL",
    "dust3r_checkpoint_path": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
    "device": "auto",
    "reconstruction_backend": "auto",
    "max_images": 100,
    "allow_placeholder_pipeline": False,
})
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
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}  device = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

echo
echo "==> Setup complete. Next: bash scripts/colab_serve.sh"
