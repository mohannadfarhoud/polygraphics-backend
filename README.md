# polyGraphics backend

Backend pipeline to convert multi-view 2D images into either a meshed `.glb` (DUSt3R/COLMAP path) or a Gaussian Splatting `.ply` (3DGS path).

## Free GPU demo on Google Colab

**Open the notebook from GitHub** (always loads latest `main`):  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohannadfarhoud/polygraphics-backend/blob/main/colab_demo.ipynb)

Switch to the **T4 GPU** runtime (`Runtime > Change runtime type > T4 GPU`), then `Runtime > Run all`. The notebook:

1. Downloads this repo as a **ZIP** into the Colab VM (**no `git clone`** — avoids Colab credential / “could not read Username” errors),
2. Runs `scripts/colab_setup.sh` (installs DUSt3R, SAM, deps, downloads the SAM checkpoint, points the API at HF for DUSt3R weights),
3. Runs `scripts/colab_serve.sh` which starts `uvicorn` and opens a free **Cloudflare Quick Tunnel** (`trycloudflare.com`) — no account, no token.

The last cell prints a public HTTPS URL like `https://random-words-xyz.trycloudflare.com`. Open `<URL>/swagger` to drive it. Limits: ~12 h max session, ~90 min idle disconnect, disk wiped on session end. Perfect for short demos with full-speed GPU DUSt3R / Gaussian Splatting.

**Colab / GitHub fetch:** The notebook downloads **`main` as a ZIP** (anonymous HTTPS). If you still see old cells mentioning `git clone`, open the notebook via the badge link above — do not use an uploaded `.ipynb` copy from your laptop. If `/content/polygraphics-backend` is half-broken, run `!rm -rf /content/polygraphics-backend` once, then re-run from section 2.

## Pipeline (mesh / `.glb`)

1. SAM segmentation creates binary masks and forces a black background.
2. DUSt3R (or COLMAP) reconstruction produces aligned 3D points.
3. Open3D statistical outlier removal cleans noise.
4. Poisson meshing + decimation creates a lightweight surface mesh; vertex colours come from the reconstructed point cloud.
5. `.glb` export → `output/<job_id>.glb`.

## Pipeline (Gaussian Splatting / `.ply`)

Set **`reconstruction_backend = "gaussian_splatting"`** in `PUT /settings`.

1. SAM segmentation (same as above). When **`save_raw_masks = true`** (default), 1‑channel `.png` masks are written to `masks/<job_id>/mask_NNN.png` alongside the masked colour images.
2. **Initial scene** (Phase 4 of the pipeline protocol):
   - `gs_init_source = "colmap"` runs COLMAP on the masked images to produce `cameras.bin` / `images.bin` / `points3D.bin`.
   - **`gs_init_source = "dust3r"`** runs DUSt3R + `GlobalAligner` and writes a COLMAP **text** sparse reconstruction (`sparse/0/cameras.txt` / `images.txt` / `points3D.txt`). The 3D-points seed is the confidence-filtered DUSt3R cloud — exactly the “seed” described in the protocol.
3. **Training**: shells out to `python <gs_repo_path>/train.py -s <scene> -m <model> --iterations <gs_iterations> --sh_degree <gs_sh_degree> --opacity_reset_interval <gs_opacity_reset_interval> [--resolution <gs_resolution>]` from the official [`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting) repo.
4. The latest `point_cloud/iteration_<N>/point_cloud.ply` is copied to `output/<job_id>.ply`.
5. `model_url` points to that `.ply`; `model_format = "ply"`.

### Job `stage` vocabulary (for the UI)

The canonical mapping (progress ranges + ordered flows) lives in **`ui/job-stages-progress.json`**. The API also exposes it at **`GET /job-stages`** so your frontend can fetch it at runtime and stay aligned with the server.

`GET /jobs/{id}` and `notify_job_update` push a `stage` string (plus a `progress` 0-100). Switch on the **base** of the string (everything before the first space — the suffix in parens is human-readable):

| `stage` base | Approx. progress | Meaning | Emitted on |
|---|---:|---|---|
| `starting` | 5 | Job picked up; settings validated. | always |
| `phase_1_segmentation` | 10 → 40 | SAM masks per image. Suffix is `(i/total)`. | always |
| `phase_2_alignment` | 40 → 60 | DUSt3R inference + `GlobalAligner` (epochs/lr from settings). | mesh path; GS path with `gs_init_source=dust3r` |
| `phase_3_sanitization` | 55 → 70 | Confidence filter + Open3D SOR. | mesh path; GS path with `gs_init_source=dust3r` |
| `phase_4_colmap_bridge` | 60 | Writing `sparse/0/{cameras,images,points3D}.txt` from DUSt3R seed. | GS path with `gs_init_source=dust3r` |
| `phase_4_colmap_scene` | 50 | Running COLMAP (`feature_extractor` → `exhaustive_matcher` → `mapper`). | GS path with `gs_init_source=colmap` |
| `phase_5_gaussian_splatting` | 65 → 95 | `train.py` running. | GS path |
| `meshing` | 75 | Poisson + decimation. | mesh path only |
| `vertex_color_transfer` | 77 | Transfer nearest cloud colors to mesh vertices. | mesh path only |
| `mesh_cleanup` | 79 | Keep only the largest connected mesh component. | mesh path only |
| `color_autobalance` | 90 | Auto-lift dark colors and center/scale mesh near origin. | mesh path only |
| `exporting` | 94 | Writing the final `.glb` / `.ply`. | always |
| `completed` | 100 | Job finished, `model_url` is ready. | always |

The list of valid base values is also exported as `app.job_stages.ALL_STAGES`. Example UI mapping:

```ts
const STAGE_LABELS: Record<string, string> = {
  starting: "Preparing job…",
  phase_1_segmentation: "Masking subject (SAM)",
  phase_2_alignment: "Aligning views (DUSt3R)",
  phase_3_sanitization: "Cleaning point cloud",
  phase_4_colmap_bridge: "Building COLMAP scene from DUSt3R",
  phase_4_colmap_scene: "Building COLMAP scene",
  phase_5_gaussian_splatting: "Training Gaussian Splatting",
  meshing: "Meshing surface",
  vertex_color_transfer: "Applying point-cloud colors",
  mesh_cleanup: "Removing disconnected fragments",
  color_autobalance: "Balancing colors + centering object",
  exporting: "Exporting model",
  completed: "Done",
};
const base = (job.stage ?? "").split(" ")[0];
const label = STAGE_LABELS[base] ?? job.stage ?? "Working…";
```

### Pipeline protocol settings (DUSt3R + GS)

These map 1‑to‑1 to the user-provided pipeline protocol and are tunable in `PUT /settings`:

| Setting | Default | Phase | Notes |
|---|---:|---|---|
| `save_raw_masks` | `true` | 1 | Save binary `.png` masks to `masks/<job_id>/`. |
| `dust3r_aligner_iters` | `300` | 2 | `niter` for `compute_global_alignment` (≥ 300 for stable floors). |
| `dust3r_aligner_lr` | `0.01` | 2 | Learning rate for the global aligner. |
| `dust3r_confidence_threshold` | `0` | 3 | Drop DUSt3R points below this normalized per-pixel confidence. `0` disables. |
| `nb_neighbors` | `20` | 3 | Open3D SOR neighbours. |
| `std_ratio` | `2.0` | 3 | Open3D SOR std-dev ratio (protocol range: 1.5–2.0). |
| `decimation_target_triangles` | `300000` | mesh | Higher keeps more detail (slower / larger GLB). |
| `gs_opacity_reset_interval` | `3000` | 5 | Forwarded to `train.py --opacity_reset_interval`. |
| `gs_iterations` | `30000` | 5 | `7_000` (Quick) or `30_000` (Dense). |
| `gs_init_source` | `colmap` | 4 | Set to `dust3r` to seed GS from the DUSt3R cloud. |

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

Clone only (skip compiling extensions):

```powershell
.\scripts\install_gaussian_splatting_windows.ps1 -SkipSubmoduleBuild
```

Then **`PUT /settings`**: `reconstruction_backend`, `gs_repo_path` (default clone: `C:\polyGraphics\third_party\gaussian-splatting`), `colmap_binary_path`, `device`: `"cuda"`, restart the API.

**CPU-only machines:** leave **`gs_allow_cpu_fallback`: `true`** (default). The API still builds the COLMAP or DUSt3R scene, then writes a **colored** 3DGS-format `.ply`: one Gaussian per sparse point with RGB in the SH DC bands — **no** official `train.py`, **no** CUDA extensions. This is **not** the same quality as GPU optimization, but **you get real colors** and most splat viewers open the file. You only need **`colmap_binary_path`** (for `gs_init_source="colmap"`) or DUSt3R settings (for `dust3r`); **`gs_repo_path` is optional** on CPU when fallback is enabled.

Set **`gs_allow_cpu_fallback`: `false`** only if you install an **NVIDIA GPU**, CUDA, and **graphdeco-inria/gaussian-splatting** as described above — then full training runs.

Otherwise use mesh backends (`auto` / `dust3r` / `colmap`) for `.glb` surfaces without GS.

### Mesh (`.glb`) vs Gaussian Splatting (`.ply`)

The default **`reconstruction_backend` is `auto`** (DUSt3R for smaller sets, COLMAP at **`auto_dust3r_max_images`** and above). That path produces a **polygon mesh** (`.glb`). If your `model_format` is `glb`, you are on the mesh path.

To use **Gaussian Splatting**, set **`reconstruction_backend`: `"gaussian_splatting"`** in `PUT /settings`, plus **`gs_repo_path`**, **`colmap_binary_path`** (for `gs_init_source="colmap"`), and a **CUDA GPU** with the extensions built as above. The API then outputs `.ply` and sets **`model_format`: `"ply"`**.

### SAM: mask the center subject

`PUT /settings` includes **`sam_segmentation_mode`** (default **`auto_masks_center_bias`**): SAM auto-generates masks and scores them by size × closeness to center. Other modes: **`center_point`** (prompt with a positive point at image center) and **`auto_masks_largest_area`** (legacy: largest mask only — often the background).

### Improving mesh quality (DUSt3R + Open3D)

- Use **20–40+ well-overlapping** photos of the same object, turntable style if possible.
- Raise **`poisson_depth`** (e.g. 10–12) and/or increase **`decimation_target_triangles`** for a denser mesh (larger files, slower).
- On CPU, keep **`max_image_side`** moderate (512–768) to avoid OOM; on GPU you can go higher for more detail.

## Real 3D vs demo mode

By default **`allow_placeholder_pipeline` is `false`** (`PUT /settings`). In that mode:

- **`COMPLETED` only happens** after real segmentation + reconstruction (and either real meshing or real GS training) + a non-empty `.glb` / `.ply` export.
- For DUSt3R: configure **`sam_checkpoint_path`** + **`dust3r_checkpoint_path`** and install **DUSt3R from source** (`https://github.com/naver/dust3r`).
- For Gaussian Splatting: configure **`gs_repo_path`** (clone of `graphdeco-inria/gaussian-splatting`) plus **`colmap_binary_path`** (when `gs_init_source="colmap"`).

If **`allow_placeholder_pipeline` is `true`**, the API uses a **fake center mask** + **fake points** for the mesh path, and writes a **dummy GS-style PLY** for the GS path — **wiring tests only**, not real geometry.

## Current implementation status

- HTTP job API, meshing (Open3D), GLB export, and model URL handling are implemented.
- With **`allow_placeholder_pipeline=false`**, SAM runs via **`segment_anything`** when installed; DUSt3R runs via **`dust3r`** when installed.

### One-shot ML install (Windows)

After the base `setup_windows.ps1` has created `.venv`, run:

```powershell
.\scripts\install_ml_windows.ps1            # SAM vit_b + DUSt3R 224_linear (CPU-friendly)
.\scripts\install_ml_windows.ps1 -SamModel vit_h -Dust3rModel 512_dpt   # GPU-class
```

The script installs PyTorch (CPU build), `segment-anything`, clones and `pip install -e .` DUSt3R into `C:\polyGraphics\third_party\dust3r`, and downloads the checkpoints into `C:\polyGraphics\models\{sam,dust3r}`. It is idempotent — re-running only re-does missing steps. After it finishes, push the suggested paths via `PUT /settings` and `Restart-Service polygraphics`.

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

- **`POST /jobs`** — multipart **`files`** in the body, optional **`job_id`**. Saves images under `uploads/{job_id}/`, creates the job as **`PENDING`** (nothing runs until you start). Use this when the UI uploads first and starts processing later.
- **`POST /jobs/{job_id}/start`** — begins the pipeline (**`PENDING` → `QUEUED` → …**). Requires **at least 2** images saved for that job.
- **`POST /jobs/reconstruct`** — convenience: **upload + start in one call** (same multipart fields). Requires at least 2 images.
- After any upload route, poll **`GET /jobs/{job_id}`** for status and **`model_url`** when **`COMPLETED`**.
- `GET /jobs` — list all jobs (from `config/jobs.json`).
- `GET /jobs/{job_id}` — job status, `model_url`, `error`.
- `POST /jobs/{job_id}/stop` — cooperative cancel (`STOPPED`).
- `POST /jobs/{job_id}/continue` — resume from `STOPPED`, `FAILED`, or `PAUSED` (re-queues; needs `uploads/{job_id}/input_*`).
- `POST /jobs/{job_id}/reprocess` — re-run from saved inputs (including after `COMPLETED`).
- `GET /models` — list `.glb` and `.ply` files under the configured output directory (`url`, **`image_url`** preview of first upload, `size_bytes`, … — no server filesystem paths).
- `GET /settings` / `PUT /settings` — runtime options.
- `GET /server/status` — hardware/software snapshot.

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

## Integration points

- To use PostgreSQL/MySQL later, replace the `jobs_db` module (same `JobManager` API) or add a SQLAlchemy layer.
- Set `job_manager.notifier` in `app/main.py` to a real `WebSocketNotifier` for push updates (pipeline status is already reflected via `update_job`).

