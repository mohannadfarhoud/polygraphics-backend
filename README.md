# polyGraphics backend

Backend pipeline to convert multi-view 2D images into either a meshed `.glb` (**MapAnything**) or a Gaussian Splatting `.ply` (**3DGS**, seeded via MapAnything then `train.py`).

## Free GPU demo on Google Colab

**Open the notebook from GitHub** (always loads latest `main`):  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohannadfarhoud/polygraphics-backend/blob/main/colab_demo.ipynb)

Switch to the **T4 GPU** runtime (`Runtime > Change runtime type > T4 GPU`), then `Runtime > Run all`. The notebook:

1. Downloads this repo as a **ZIP** into the Colab VM (**no `git clone`** — avoids Colab credential / “could not read Username” errors),
2. Runs `scripts/colab_setup.sh` (SAM, deps, Meta **MapAnything** from GitHub, SAM checkpoint — HF pulls MapAnything weights on first reconstruct),
3. Runs `scripts/colab_serve.sh` which starts `uvicorn` and opens a free **Cloudflare Quick Tunnel** (`trycloudflare.com`) — no account, no token.

The last cell prints a public HTTPS URL like `https://random-words-xyz.trycloudflare.com`. Open `<URL>/swagger` to drive it. Limits: ~12 h max session, ~90 min idle disconnect, disk wiped on session end. Perfect for short demos with full-speed GPU **MapAnything** / Gaussian Splatting.

**Colab / GitHub fetch:** The notebook downloads **`main` as a ZIP** (anonymous HTTPS). If you still see old cells mentioning `git clone`, open the notebook via the badge link above — do not use an uploaded `.ipynb` copy from your laptop. If `/content/polygraphics-backend` is half-broken, run `!rm -rf /content/polygraphics-backend` once, then re-run from section 2.

## Pipeline (mesh / `.glb`)

1. SAM segmentation creates binary masks and forces a black background.
2. **MapAnything** (Meta) predicts metric depths, rays, poses, then we merge masked pixels into a unified world-frame point cloud.
3. Open3D statistical outlier removal cleans noise.
4. Poisson meshing + decimation creates a lightweight surface mesh; vertex colours come from the point cloud, then optional **multi-view photo projection** onto vertices (see `mesh_photo_vertex_bake` in `PUT /settings`) for a closer match to the real photos.
5. `.glb` export → `output/<job_id>.glb` (optional Draco-style compression via Open3D when `mesh_glb_draco_compression` is true).

### RTX 3050 (8 GB VRAM) preset

Tune `PUT /settings` roughly like this for mesh + splats on a single GPU:

```json
{
  "device": "cuda",
  "reconstruction_backend": "mapanything",
  "mapanything_memory_efficient_inference": true,
  "mapanything_minibatch_size": 1,
  "sam_use_fp16": true,
  "max_input_image_side": 1600,
  "max_image_side": 1024,
  "gs_iterations": 10000,
  "gs_sh_degree": 2,
  "gs_densify_until_iter": 7000,
  "mesh_glb_draco_compression": true,
  "mesh_photo_vertex_bake": true
}
```

### SAM masks CLI (CUDA)

Batch binary masks outside the API:

```powershell
.\.venv\Scripts\python.exe .\scripts\sam_masks_cuda.py --input-dir .\photos --mask-dir .\masks --checkpoint C:\path\to\sam_vit_b_01ec64.pth --model-type vit_b
```

## Pipeline (Gaussian Splatting / `.ply`)

Set **`reconstruction_backend = "gaussian_splatting"`** in `PUT /settings`, set **`gs_repo_path`** to your local clone of [`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting) (with CUDA extensions built on the worker). Default **`gs_iterations`** is **`10000`**; raise toward **`30000`** for final-quality `.ply` when VRAM/time allow.

1. SAM segmentation (same as above). When **`save_raw_masks = true`** (default), 1‑channel `.png` masks are written to `masks/<job_id>/mask_NNN.png` alongside the masked colour images.
2. **Initial scene**: **MapAnything** runs on the masked images (subprocess-isolated when `gpu_isolate_phases=true` + CUDA). We write a COLMAP-compatible **TEXT** sparse model (`sparse/0/cameras.txt`, `images.txt`, `points3D.txt`) that the upstream **gaussian-splatting** `train.py` can read alongside `scene/images/` (same layout as historical DUSt3R→COLMAP-text seeding).

   **`gs_train_with_original_images`** (default `true`): immediately before **`train.py`**, `scene/images` is rewritten per view using the **same downscaled originals** as the SAM stage (masked filenames unchanged). That way registration still uses masking-friendly views but the **Gaussian photometric loss** is supervised against real colours—avoids systematically **dark/black** optimisation when SAM blacks out backgrounds.
3. **Training**: shells out to `python <gs_repo_path>/train.py` with `--iterations`, `--sh_degree`, `--opacity_reset_interval`, optional `--resolution`, and `--densify_until_iter` when `gs_densify_until_iter > 0` from the official [`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting) repo.
4. The latest `point_cloud/iteration_<N>/point_cloud.ply` is copied to `output/<job_id>.ply`.
5. `model_url` points to that `.ply`; `model_format = "ply"`.

### Job `stage` vocabulary (for the UI)

The canonical mapping (progress ranges + ordered flows) lives in **`ui/job-stages-progress.json`**. The API also exposes it at **`GET /job-stages`** so your frontend can fetch it at runtime and stay aligned with the server.

`GET /jobs/{id}` and `notify_job_update` push a `stage` string (plus a `progress` 0-100). Switch on the **base** of the string (everything before the first space — the suffix in parens is human-readable):

| `stage` base | Approx. progress | Meaning | Emitted on |
|---|---:|---|---|
| `starting` | 5 | Job picked up; settings validated. | always |
| `phase_1_segmentation` | 10 → 40 | SAM masks per image. Suffix is `(i/total)`. | always |
| `phase_2_alignment` | 40 → 60 | **MapAnything** metric reconstruction. | mesh + GS |
| `phase_3_sanitization` | 55 → 70 | Confidence filter + Open3D SOR on mesh path; GS aligns with staged progress. | mesh + GS seed |
| `phase_4_colmap_bridge` | 60 | Write COLMAP-text sparse (`sparse/0/*.txt`) for `train.py` from MapAnything poses/points. | GS path |
| `phase_4_colmap_scene` | — | Unused (legacy COLMAP CLI SfM removed). | — |
| `phase_5_gaussian_splatting` | 65 → 95 | `train.py` running. | GS path |
| `meshing` | 75 | Poisson + decimation. | mesh path only |
| `vertex_color_transfer` | 77 | Transfer nearest cloud colors to mesh vertices. | mesh path only |
| `mesh_cleanup` | 79 | Keep only the largest connected mesh component. | mesh path only |
| `photo_vertex_bake` | 86 | Multi-view projected colours from original images. | mesh path when cameras exist and `mesh_photo_vertex_bake` is true |
| `color_autobalance` | 90 | Auto-lift dark colors and center/scale mesh near origin. | mesh path only |
| `exporting` | 94 | Writing the final `.glb` / `.ply`. | always |
| `completed` | 100 | Job finished, `model_url` is ready. | always |

The list of valid base values is also exported as `app.job_stages.ALL_STAGES`. Example UI mapping:

```ts
const STAGE_LABELS: Record<string, string> = {
  starting: "Preparing job…",
  phase_1_segmentation: "Masking subject (SAM)",
  phase_2_alignment: "MapAnything reconstruction",
  phase_3_sanitization: "Cleaning point cloud",
  phase_4_colmap_bridge: "Writing sparse COLMAP text for GS",
  phase_4_colmap_scene: "(unused legacy stage)",
  phase_5_gaussian_splatting: "Training Gaussian Splatting",
  meshing: "Meshing surface",
  vertex_color_transfer: "Applying point-cloud colors",
  mesh_cleanup: "Removing disconnected fragments",
  photo_vertex_bake: "Projecting photo colors",
  color_autobalance: "Balancing colors + centering object",
  exporting: "Exporting model",
  completed: "Done",
};
const base = (job.stage ?? "").split(" ")[0];
const label = STAGE_LABELS[base] ?? job.stage ?? "Working…";
```

### Pipeline protocol settings (MapAnything + mesh + GS)

These map to tunable fields in `PUT /settings`:

| Setting | Default | Phase | Notes |
|---|---:|---|---|
| `save_raw_masks` | `true` | 1 | Save binary `.png` masks to `masks/<job_id>/`. |
| `max_input_image_side` | `1600` | 0–1 | Longest edge after ingest — downscale before SAM / MapAnything. |
| `reconstruction_backend` | `mapanything` | — | **`mapanything`** for `.glb` mesh; **`gaussian_splatting`** for `.ply`. |
| `mapanything_pretrained_id` | `facebook/map-anything-apache` | 2–4 | Hugging Face hub id passed to `MapAnything.from_pretrained`. |
| `mapanything_memory_efficient_inference` | `true` | 2 | Prefer `True` on 8 GB GPUs. |
| `mapanything_minibatch_size` | `1` | 2 | VRAM-friendly infer minibatch when memory-efficient mode is on. |
| `mapanything_max_input_views` | `48` | 2 | Subsample uniformly when many photos — avoids OOM. |
| `mapanything_apply_confidence_mask` | `false` | 2 | Optional learned-confidence filtering (`model.infer`). |
| `nb_neighbors` | `26` | 3 | Open3D SOR neighbours before Poisson. |
| `std_ratio` | `1.75` | 3 | Open3D SOR std-dev ratio (protocol ~1.5–2.0). |
| `decimation_target_triangles` | `300000` | mesh | GLB triangle budget after Poisson. |
| `mesh_photo_vertex_bake` | `true` | mesh | Sample vertex RGB when poses exist — use **`mesh_photo_vertex_bake_sample_source`** (`masked` = no backdrop smear from full photos; `original` = legacy). |
| `mesh_glb_draco_compression` | `true` | mesh | Compressed GLB when Open3D allows. |
| `sam_use_fp16` | `true` | 1 | CUDA AMP fp16 for SAM. |
| `poisson_depth` | `9` | mesh | Open3D Poisson depth — lower tends to tame noisy neural clouds. |
| `sam_segmentation_mode` | `center_subject_table` | 1 | **`center_subject_table`** (**default**): centre FG + corners BG + bottom-edge negatives (table plane). **`center_subject`**: corners only; **`center_point`** centre tap only; **`auto_masks_center_bias`** when subject is off-centre. |
| `sam_table_edge_negative_points` | `11` | 1 | Used only with **`center_subject_table`**: count of negatives along the bottom strip (0–24); raise if table persists, lower if thin stands/feet vanish. |
| `gs_train_with_original_images` | `true` | 5 | Replace `scene/images` pixels with originals before `train.py` (fixes dark splats). |
| `compare_mesh_preview_with_gs` | `false` | — | When GS: emit **`{job}_compare_mesh.glb`** (MapAnything) before `.ply`. |
| `gs_opacity_reset_interval` | `3000` | 5 | Passed to `--opacity_reset_interval` (worker may lengthen on short runs). |
| `gs_iterations` | `10000` | 5 | Default RTX‑3050 friendly; raise toward `30000` for finals. |
| `gs_densify_until_iter` | `7000` | 5 | Forwarded to `train.py --densify_until_iter`. |

> Real GS training officially needs a CUDA GPU. On CPU it’s impractical (or unsupported, depending on fork).

### Installing Gaussian Splatting on Windows (official repo)

The API runs [`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting) (`train.py`). That repo builds **CUDA extensions** (`diff-gaussian-rasterization`, `simple-knn`). You need:

1. **NVIDIA GPU** + current driver  
2. **CUDA Toolkit** with **`nvcc` on `PATH`** (match the PyTorch CUDA wheel, e.g. cu118 vs cu124)  
3. **Visual Studio 2022 Build Tools** with **Desktop development with C++**  
4. This project’s **`.venv`** (`scripts/setup_windows.ps1`)

From the repo root:

```powershell
.\scripts\install_gaussian_splatting_windows.ps1
```

CUDA 11.8 toolkit example:

```powershell
.\scripts\install_gaussian_splatting_windows.ps1 -TorchCudaIndexUrl "https://download.pytorch.org/whl/cu118"
```

If **MapAnything / PyTorch already live in another venv** (e.g. `polygraph_worker\.venv`) but the backend repo `.venv` does not run `train.py`, build the GS extensions into that interpreter:

```powershell
.\scripts\install_gaussian_splatting_windows.ps1 -SkipTorchCuda `
  -PythonExe "C:\path\to\polygraph_worker\.venv\Scripts\python.exe"
```

Clone only (skip compiling extensions):

```powershell
.\scripts\install_gaussian_splatting_windows.ps1 -SkipSubmoduleBuild
```

If **`pip` reports `nvcc` failed with exit code 1**, open the lines above `nvcc`:

- **`Cannot find compiler 'cl.exe'`** means MSVC was not on `PATH`. Use **`x64 Native Tools Command Prompt for VS 2022`**, **or** from any repo checkout run pip through **`scripts\invoke_vs_build_tools.ps1`** so `vcvars64.bat` initializes the compiler env (same args you would pass to `python.exe`).

  ```powershell
  .\scripts\invoke_vs_build_tools.ps1 `
      "C:\Users\you\polygraph_worker\.venv\Scripts\python.exe" `
      -m pip install --no-build-isolation `
      "C:\polyGraphics\third_party\gaussian-splatting\submodules\diff-gaussian-rasterization"
  ```

- **CUDA vs PyTorch**: align **PyTorch's `torch.version.cuda`** with your **CUDA toolkit** (`nvcc --version`) when possible; set **`CUDA_HOME`** if needed. Pass **`-TorchCudaArchList`** (e.g. **`8.6`**) on `install_gaussian_splatting_windows.ps1` if arch-related errors appear.

- Optional: **`pip install ninja`** in the same venv for faster extension builds (removes the “falling back to distutils” warning).

If **`simple_knn`** fails only with ninja’s **`RuntimeError: Error compiling objects for extension`**, PyTorch stripped the detail—capture a full transcript and align the CUDA toolkit:

1. **Save everything** — search the log for **`error C`**, **`fatal error`**, **`cub`**, **`thrust`**, **`cannot open`**:

   ```powershell
   $env:MAX_JOBS = "1"
   .\scripts\invoke_vs_build_tools.ps1 `
       "C:\Users\you\polygraph_worker\.venv\Scripts\python.exe" `
       -m pip install -vv --no-build-isolation `
       "C:\polyGraphics\third_party\gaussian-splatting\submodules\simple-knn" `
       2>&1 | Tee-Object -FilePath "$env:TEMP\simple_knn_build.log"
   ```

   If the log stays opaque, **`pip uninstall ninja`** in that venv and rerun the same line (**setuptools** / distutils path often prints clearer `cl`/header errors).

2. **`torch.version.cuda` vs CUDA Toolkit**: if **`& "…\python.exe" -c "import torch; print(torch.version.cuda)"`** prints **`12.4`** but **`nvcc --version`** shows **12.1**, install NVIDIA **CUDA Toolkit 12.4**, set **`CUDA_HOME`** to that install, and ensure **`CUDA\v12.4\bin`** appears **before** any older **`CUDA\bin`** on `PATH`. Re-run **`invoke_vs_build_tools.ps1`** and the pip install.

3. **Conda workflows only**: missing **`cub`** headers sometimes shows up against older stacks—see [**gaussian-splatting#1023**](https://github.com/graphdeco-inria/gaussian-splatting/issues/1023) (`conda install cccl`). Pip-only setups usually fix this by fixing toolkit / `CUDA_HOME`.

Then **`PUT /settings`**: `reconstruction_backend`, `gs_repo_path` (default clone: `C:\polyGraphics\third_party\gaussian-splatting`), `device`: `"cuda"`, install **MapAnything** in the same venv (`pip install git+https://github.com/facebookresearch/map-anything.git`), restart the API.

**CPU-only machines:** leave **`gs_allow_cpu_fallback`: `true`** (default). The API still runs **MapAnything** to build COLMAP-text sparse points, then writes a **colored** 3DGS-format `.ply`: one Gaussian per sparse point with RGB in the SH DC bands — **no** official `train.py`, **no** CUDA extensions. **`gs_repo_path` is optional** on CPU when fallback is enabled.

Set **`gs_allow_cpu_fallback`: `false`** only if you install an **NVIDIA GPU**, CUDA, and **graphdeco-inria/gaussian-splatting** as described above — then full training runs.

Use **`reconstruction_backend`: `mapanything`** for `.glb` surfaces without GS.

### Mesh (`.glb`) vs Gaussian Splatting (`.ply`)

The default **`reconstruction_backend` is `mapanything`** (feed-forward metric mesh). **`assert_pipeline_ready`** checks **SAM** + importable **`mapanything`** + PyTorch for both mesh and GS paths; GPU GS additionally needs **`gs_repo_path`** with `train.py` unless CPU fallback is allowed.

To use **Gaussian Splatting**, set **`reconstruction_backend`: `"gaussian_splatting"`** in `PUT /settings`, plus **`gs_repo_path`** and a **CUDA GPU** with extensions built as above. MapAnything still seeds the COLMAP-text scene. The API outputs `.ply` and sets **`model_format`: `"ply"`**.

### SAM: centre the subject and drop the backdrop

`sam_segmentation_mode` defaults to **`center_subject_table`**: Segment Anything gets a **foreground point at the image centre**, **background points at all four corners**, plus **extra negatives along an inset strip near the bottom** (typical tabletop contact line). The mask is then tightened to the **single connected foreground component that touches the centre**. Use **`center_subject`** if that bottom strip wrongly removes thin bases; **`center_point`** is the simpler “one centre tap”; switch to **`auto_masks_center_bias`** when the object is deliberately off-centre; avoid **`auto_masks_largest_area`** unless you know the largest segment is still the subject.

### Improving mesh quality (MapAnything + Open3D)

- Use **≥ 24 recommended, 40+ ideal** overlapping orbit photos around a **fixed object** (`GET /capture-guide` for onboarding copy).
- Default **`poisson_depth`** is **`9`** to limit spike artefacts on noisy clouds; **raise** toward `10–11` only once reconstructions look clean (`PUT /settings`).
- Increase **`decimation_target_triangles`** for heavier meshes (slower / larger `.glb`).
- On CPU, keep **`max_image_side`** moderate (512–768) to avoid OOM; on GPU you can go higher for more detail.

## Real 3D vs demo mode

By default **`allow_placeholder_pipeline` is `false`** (`PUT /settings`). In that mode:

- **`COMPLETED` only happens** after real segmentation + reconstruction (and either real meshing or real GS training) + a non-empty `.glb` / `.ply` export.
- For mesh / GS seed: **`sam_checkpoint_path`** plus install **Meta MapAnything** (`https://github.com/facebookresearch/map-anything`) — default HF id **`facebook/map-anything-apache`**.
- For GPU Gaussian Splatting: **`gs_repo_path`** (clone of `graphdeco-inria/gaussian-splatting`).

If **`allow_placeholder_pipeline` is `true`**, the API uses a **fake center mask** + **fake points** for the mesh path, and writes a **dummy GS-style PLY** for the GS path — **wiring tests only**, not real geometry.

## Current implementation status

- HTTP job API, meshing (Open3D), GLB export, and model URL handling are implemented.
- With **`allow_placeholder_pipeline=false`**, SAM runs via **`segment_anything`** when installed; reconstruction uses **`mapanything`** when installed.

### One-shot ML install (Windows)

After the base `setup_windows.ps1` has created `.venv`, run:

```powershell
.\scripts\install_ml_windows.ps1
```

The script installs PyTorch (**CPU baseline** from the PyTorch CPU index), **`segment-anything`**, **MapAnything (`pip` from GitHub)**, and downloads the SAM checkpoint under `C:\polyGraphics\models\sam`. Upgrade to a **CUDA PyTorch** wheel on GPU workers, then reinstall / verify imports. MapAnything weights download from Hugging Face on the first job (`mapanything_pretrained_id`).

```powershell
Restart-Service polygraphics   # or your service wrapper
```

## AI-prior command backend (worker runbook)

`ai_prior` and `hybrid_prior_refine` can call an external image-to-3D model through a command adapter.

### 1) Configure worker environment

In `.env.worker`:

```env
POLYGRAPH_OVERRIDE_AI_PRIOR_COMMAND=C:\Users\mohannad\polygraph_worker\scripts\ai_prior_command_adapter_windows.bat
POLYGRAPH_AI_PRIOR_UPSTREAM_CMD=C:\path\to\real_ai_prior_runner.bat
# optional:
# POLYGRAPH_AI_PRIOR_UPSTREAM_ARGS_TEMPLATE=--input-manifest {input_manifest} --output {output_mesh}
# AI_PRIOR_API_KEY=...
```

### 2) Configure runtime settings

Use `PUT /settings`:

```json
{
  "reconstruction_backend": "hybrid_prior_refine",
  "ai_prior_provider": "command",
  "ai_prior_command": "C:\\Users\\mohannad\\polygraph_worker\\scripts\\ai_prior_command_adapter_windows.bat",
  "ai_prior_command_args_template": "--input-manifest {input_manifest} --output {output_mesh}",
  "ai_prior_api_key_env": "AI_PRIOR_API_KEY",
  "ai_prior_require_api_key": false
}
```

### 3) Contract required by provider command

- Input manifest path from `--input-manifest`:
  - JSON shape: `{ "masked_images": ["abs_path1", "abs_path2", ...] }`
- Output path from `--output`:
  - should write `.glb` (adapter also accepts `.obj`/`.ply` and converts to `.glb`)

### 4) Run one test job

1. Upload and start a job (or `POST /jobs/reconstruct`).
2. Check `GET /jobs/{id}` fields:
   - `reconstruction_confidence`, `route_taken`, `quality_reason`
3. Inspect reports:
   - `uploads/{job_id}/capture_quality_report.json`
   - `uploads/{job_id}/reconstruction_report.json`

## Tripo API cloud provider (demo/evaluation)

Use this when you want to compare hosted Tripo quality during a trial window.

### 1) Install Tripo SDK on worker

```powershell
pip install tripo3d
```

### 2) Configure worker environment

In `.env.worker`:

```env
POLYGRAPH_OVERRIDE_AI_PRIOR_COMMAND=C:\Users\mohannad\polygraph_worker\scripts\ai_prior_command_adapter_windows.bat
POLYGRAPH_AI_PRIOR_UPSTREAM_CMD=C:\Users\mohannad\polygraph_worker\scripts\tripo_api_upstream_windows.bat
TRIPO_API_KEY=tsk_xxx

# optional quality/cost tuning:
# TRIPO_MODEL_VERSION=v3.1-20260211
# TRIPO_TEXTURE_QUALITY=detailed
# TRIPO_POLL_TIMEOUT_SECONDS=1800
```

### 3) Configure runtime settings (strict Tripo-only route)

Use `PUT /settings`:

```json
{
  "reconstruction_backend": "ai_prior",
  "ai_prior_provider": "command",
  "ai_prior_command": "C:\\Users\\mohannad\\polygraph_worker\\scripts\\ai_prior_command_adapter_windows.bat",
  "ai_prior_command_args_template": "--input-manifest {input_manifest} --output {output_mesh}",
  "ai_prior_api_key_env": "TRIPO_API_KEY",
  "ai_prior_require_api_key": true,
  "ai_prior_force_prior_only": true,
  "ai_prior_timeout_seconds": 1800
}
```

Notes:
- `scripts/tripo_api_upstream.py` receives all masked views, picks the best single view, and submits Tripo `image_to_model`.
- In `command` mode with `ai_prior_force_prior_only=true`, backend minimum input becomes 1 image (useful for quick trials).
- The AI-prior route keeps Tripo output as the final mesh path unless you intentionally switch to hybrid refinement.

## Isolation model: dataset → train → activate → predict

PicPolish uploads **before/after** image pairs; the API learns foreground masks and serves **CPU ONNX** inference for HQ product cutouts (RGBA PNG).

### Flow

1. **Login as admin** — `POST /auth/login` with `{ "email": "admin", "password": "devtek2026" }` → Bearer token.
2. **Create dataset once** — `POST /isolation/datasets` with `{ "name": "..." }` (**Bearer required**). Keep this `dataset_id`.
3. **Upload pairs** — `POST /isolation/datasets/{dataset_id}/pairs` (multipart; **Bearer required**). Append more pairs to the **same** dataset over time.
4. **Train (incremental)** — `POST /isolation/train` with `{ "dataset_id", "grow_active": true }` (**Bearer required**).  
   Defaults: resume from the active model checkpoint, reuse the same `model_id`, auto-activate. Each successful train bumps `generation` (1 → 2 → 3…).
5. **Predict** — `POST /isolation/predict` with multipart `file` (**no auth**). Uses the active growing model. Returns RGBA PNG (or `format=json`).
6. **See active model** — `GET /isolation/models/active` → `model_id`, `metrics.generation`, IoU (**public**).

Training / dataset / model-admin routes require any authenticated user. Predict + health + model list remain public.


Health: `GET /isolation/health` (ONNX Runtime, active model).

All routes are under `APP_ROOT_PATH` (e.g. `/polygraph/isolation/...`). Swagger tag: **isolation**.

### Storage layout

```
datasets/isolation/{dataset_id}/meta.json
datasets/isolation/{dataset_id}/pairs/{i}/before.*, after.*, mask.png
models/isolation/{model_id}/model.onnx, metrics.json
uploads/isolation/predict/{uuid}/isolated.png, mask.png
uploads/isolation/exports/{dataset_id}.zip   # worker download
```

### Training on a GPU worker (production API is CPU-only)

When `APP_REMOTE_WORKERS=true`, train jobs are queued for the remote worker (`GET /internal/worker/isolation/train/next`). The worker downloads the dataset zip, runs training, and uploads `model.onnx`.

**Offline / manual GPU train** (same machine or Colab):

```bash
# Export dataset zip from API disk, or copy datasets/isolation/{id}/
python scripts/isolation_train_worker.py \
  --dataset-dir /path/to/datasets/isolation/{dataset_id} \
  --output-dir /tmp/iso-out \
  --base-model isnet-general-use --epochs 20

# Import ONNX without training (CPU smoke test):
curl -X POST \
  -F "file=@model.onnx" -F "name=imported" -F "activate=true" \
  https://host/polygraph/isolation/models/import
```

**Worker env** (`.env.worker`): same `POLYGRAPH_API_BASE` + `POLYGRAPH_WORKER_TOKEN` as reconstruction jobs; the worker polls isolation train jobs automatically.

### Environment

```env
# APP_FREE_ISOLATION_PER_MONTH=50
# ISOLATION_TRAIN_ON_API=0          # 1 = run train in API process (dev/small sets)
# ISOLATION_TRAIN_DEV_MOCK=0        # 1 = skip real train, export placeholder ONNX (tests)
# ISOLATION_BASE_ONNX_PATH=         # optional fixed ONNX for predict without registry
# ISOLATION_MIN_PAIRS=20            # warn below; block unless force_min_pairs on train request
```

Minimum pairs: warn if < 20; pass `"force_min_pairs": true` on `POST /isolation/train` for small test sets.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Deploy on Windows Server

### 1) Initial setup

Run PowerShell as Administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
cd C:\path\to\polyGraphics-backend
.\scripts\setup_windows.ps1
```

This creates `.venv`, installs dependencies, and creates `.env` from `.env.example`.

### 2) Configure environment

Edit `.env`:

```env
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=false
APP_WORKERS=1
APP_ROOT_DIR=.
APP_CDN_BASE_URL=https://cdn.yoursite.com
```

### 3) Start app manually (test)

```powershell
.\scripts\run_server.ps1
```

Open:
- `http://<server-ip>:8000/swagger`
- `http://<server-ip>:8000/health`

### 4) Install as Windows Service (recommended)

Install [NSSM](https://nssm.cc/download), then:

```powershell
.\scripts\install_windows_service.ps1 -NssmPath "C:\tools\nssm\nssm.exe"
```

Manage service:

```powershell
Get-Service polyGraphicsBackend
Restart-Service polyGraphicsBackend
Stop-Service polyGraphicsBackend
```

## API

- **`POST /jobs`** — multipart **`files`** in the body, optional **`job_id`**, optional **`capture_metadata`** (JSON text). Saves images under `uploads/{job_id}/`, writes metadata to `uploads/{job_id}/capture_metadata.json` when provided, and creates the job as **`PENDING`** (nothing runs until you start). Use this when the UI uploads first and starts processing later.
- **`POST /jobs/{job_id}/start`** — begins the pipeline (**`PENDING` → `QUEUED` → …**). Requires at least **2 photos** from different angles (or a video that yields ≥2 good frames). Optional `ai_prior` + `command` with `ai_prior_force_prior_only=true` still accepts 1 image for external AI adapters only.
- **`POST /jobs/reconstruct`** — convenience: **upload + start in one call** (same multipart fields; supports optional `capture_metadata` JSON text). Minimum images depend on active backend.
- **`PUT /jobs/{job_id}/capture-metadata`** — upsert structured capture metadata as JSON body after a job exists (useful when mobile upload and metadata upload are separate operations).
- **`GET /jobs/{job_id}/capture-metadata`** — fetch the stored capture metadata payload for debugging/analytics.
- After any upload route, poll **`GET /jobs/{job_id}`** for status and **`model_url`** when **`COMPLETED`**.
- `GET /jobs` — list all jobs.
- `GET /jobs/{job_id}` — job status, `model_url`, `error`.
  - also includes metadata-derived fields when present: `capture_metadata_url`, `capture_metadata_version`, `capture_total_frames`, `capture_accepted_frames`, `capture_avg_quality_score`, `capture_orbit_coverage_deg`.
- `POST /jobs/{job_id}/stop` — cooperative cancel (`STOPPED`).
- `POST /jobs/{job_id}/continue` — resume from `STOPPED`, `FAILED`, or `PAUSED` (re-queues; needs `uploads/{job_id}/input_*`).
- `POST /jobs/{job_id}/reprocess` — re-run from saved inputs (including after `COMPLETED`).
- `DELETE /jobs/{job_id}` — delete a gallery/job item (job row + uploaded images + generated model/report artifacts). Active jobs must be stopped first.
- `GET /models` — list `.glb` and `.ply` files under the configured output directory (`url`, **`image_url`** preview of first upload, `size_bytes`, … — no server filesystem paths).
- `GET /settings` / `PUT /settings` — runtime options.
- `GET /job-stages` — JSON for UI progress labels (same as `ui/job-stages-progress.json`).
- `GET /capture-guide` — capture UX + recommended `PUT /settings` field overlays for object/scene photogrammetry (`ui/capture-guide.json`).
- `GET /capture-guide/web` — machine-readable web capture contract (live frame checks, thresholds, upload gates, metadata schema) derived from current runtime settings.
- `GET /server/status` — hardware/software snapshot.

### Android capture metadata payload

`capture_metadata` (multipart text field on upload routes) and `PUT /jobs/{job_id}/capture-metadata` (JSON body) use the same structure:

```json
{
  "schema_version": "1.0",
  "session_id": "session-123",
  "session": {
    "object_label": "shoe",
    "lighting": "indoor",
    "camera_fov_deg": 76.0
  },
  "frames": [
    {
      "file": "IMG_0001.jpg",
      "frame_index": 0,
      "timestamp_ms": 1715321000123,
      "yaw_deg": -20.0,
      "pitch_deg": 8.0,
      "depth_m": 0.42,
      "quality_score": 0.91,
      "accepted": true
    }
  ],
  "summary": {
    "total_frames": 24,
    "accepted_frames": 20,
    "avg_quality_score": 0.84,
    "orbit_coverage_deg": 282.0
  }
}
```

`summary` is optional; server derives it when omitted.

## `model_url` (downloads)

`job.model_url` is built from **`cdn_base_url`** in `PUT /settings` (or **`APP_MODEL_BASE_URL`** / **`APP_CDN_BASE_URL`** in the environment). The **extension matches the chosen backend**:

- mesh path → `output/<job_id>.glb` (`model_format = "glb"`)
- gaussian splatting → `output/<job_id>.ply` (`model_format = "ply"`)

This API serves files at **`GET /output/<job_id>.<ext>`** (same folder as `output_dir_name`, default `output/`). Default base is **`http://127.0.0.1:8000/output`**, so a typical `model_url` is:

- `http://127.0.0.1:8000/output/<job_id>.glb` (mesh)
- `http://127.0.0.1:8000/output/<job_id>.ply` (GS)

Use your real public origin in production (CDN or API host).

## Reverse proxy (Nginx)

When **`APP_ROOT_PATH`** is set (e.g. `/polygraph`), this app registers **`GET`/`HEAD`** routes for **both** `/output/…`, `/uploads/…` **and** `/polygraph/output/…`, `/polygraph/uploads/…`, so downloads work **directly from Uvicorn** without Nginx. The app does **not** set **`FastAPI(root_path=...)`** (that used to make only the prefixed URLs work). Optional snippets are still in **`deploy/`** if you terminate TLS or merge paths at the edge.

Copy-and-paste snippets are in:

- **`deploy/nginx-output-uploads.conf`** — `/output/` and `/uploads/` → `http://127.0.0.1:8000`
- **`deploy/nginx-polygraph-api.conf`** — `/polygraph/` → API (strip prefix)
- **`deploy/README.md`** — setup A vs B (root assets vs everything under `/polygraph`)

## Database (jobs)

- Jobs are stored in **SQLite**: default file **`data/jobs.sqlite`** under the app root (override with env **`APP_DATABASE_PATH`**).
- Each row has **`job_id`**, **`status`**, optional **`model_url`**, **`error`**, **`image_count`**, Unix **`created_at` / `updated_at`**, and API responses also include **`created_at_iso` / `updated_at_iso`** (UTC, `Z`).
- On first start, if legacy **`config/jobs.json`** exists and the DB is empty, it is **imported once** and the JSON file is renamed to **`config/jobs.json.bak`**.

## Virtual try-on config API

Per-model settings for the polyGraphics-frontend earring try-on (camera + face tracking). Stored in the same SQLite DB as jobs (`model_try_on_config` table). Out of scope: video, landmarks, or per-frame poses.

### Coordinate system

All `*_point` fields use **normalized try-on space** (same as frontend `prepareTryOnModel`):

- Model centered at origin; largest bounding-box axis = **1.0** unit
- Each point: `{ "x", "y", "z" }` with components clamped to **[-1, 1]**
- **Hanger** — hook attachment on the mesh (e.g. `{ "x": 0, "y": 0.5, "z": 0 }`)
- **Profile** — second point on the thin/left side; orientation uses the vector **hanger → profile** (optional until set; must be ≥ **0.02** units from hanger when both are present)

### Endpoints

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/models/{job_id}/try-on` | Full config; **404** if never saved |
| `PUT` | `/models/{job_id}/try-on` | Upsert; **partial body merges** with existing; `hanger_point` required on first save |
| `GET` | `/models/{job_id}/hanger-point` | Legacy wrapper — `{ x, y, z }` only |
| `PUT` | `/models/{job_id}/hanger-point` | Updates hanger only; does **not** clear profile/rotation |
| `GET` | `/users/me/try-on-calibration` | Per-user offsets; requires header **`X-User-Id`** |
| `PUT` | `/users/me/try-on-calibration` | Partial merge; requires **`X-User-Id`** |

`job_id` must match a job in the DB or an existing `output/{job_id}.glb` / `.ply` (same as gallery models). Invalid `jewelry_type` or profile too close to hanger → **400**. Unknown model → **404**.

Optional access control: send **`X-User-Id`** on writes; the first writer becomes owner (`APP_TRY_ON_BIND_OWNER=1`, default). Another user id → **403**. Public read/write when no owner is set.

OpenAPI: live at **`GET /openapi.json`** (and under `APP_ROOT_PATH` if set). Regenerate committed copy: `python scripts/export_openapi.py` → `openapi.json`.

### Photo try-on compose (AI)

Static **photorealistic** earring placement on a user-uploaded face photo (frontend `/photo-try-on`). Uses an image-editing model (default **Gemini 2.5 Flash Image**).

| Method | Path | Notes |
|--------|------|--------|
| `POST` | `/try-on/photo-compose` | Multipart: `face_image`, `job_id`, `placement_x/y`, `image_width/height`; returns **202** + `compose_id` |
| `GET` | `/try-on/photo-compose/{compose_id}` | Poll until `status` is `completed` or `failed` |

**Setup (API host):**

```bash
pip install -r requirements-photo-compose.txt
```

Env: `GEMINI_API_KEY`, optional `PHOTO_COMPOSE_MODEL`, `PHOTO_COMPOSE_MAX_MB`, `PHOTO_COMPOSE_UPLOAD_DIR`. For local dev without Gemini: `PHOTO_COMPOSE_DEV_MOCK=1` (copies marked face as result).

Model `job_id` must have a product thumbnail (`image_url` from `GET /models` — first `input_*` under `uploads/{job_id}/`). Result: `result_url` under `/uploads/photo-compose/{compose_id}/result.jpg`.

## Integration points

- To use PostgreSQL/MySQL later, replace the `jobs_db` module (same `JobManager` API) or add a SQLAlchemy layer.
- Set `job_manager.notifier` in `app/main.py` to a real `WebSocketNotifier` for push updates (pipeline status is already reflected via `update_job`).

