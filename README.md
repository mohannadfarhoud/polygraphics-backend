# polyGraphics backend

Backend pipeline to convert multi-view 2D images into either a meshed `.glb` (DUSt3R/COLMAP path) or a Gaussian Splatting `.ply` (3DGS path).

## Pipeline (mesh / `.glb`)

1. SAM segmentation creates binary masks and forces a black background.
2. DUSt3R (or COLMAP) reconstruction produces aligned 3D points.
3. Open3D statistical outlier removal cleans noise.
4. Poisson meshing + decimation creates a lightweight surface mesh.
5. `.glb` export → `output/<job_id>.glb`.

## Pipeline (Gaussian Splatting / `.ply`)

Set **`reconstruction_backend = "gaussian_splatting"`** in `PUT /settings`.

1. SAM segmentation (same as above).
2. **Initial scene**: `gs_init_source = "colmap"` runs COLMAP on the masked images to produce `cameras.bin` / `images.bin` / `points3D.bin`. (`"dust3r"` init is scaffolded; not implemented yet.)
3. **Training**: shells out to `python <gs_repo_path>/train.py -s <scene> -m <model> --iterations <gs_iterations> --sh_degree <gs_sh_degree> [--resolution <gs_resolution>]` from the official [`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting) repo.
4. The latest `point_cloud/iteration_<N>/point_cloud.ply` is copied to `output/<job_id>.ply`.
5. `model_url` points to that `.ply`; `model_format = "ply"`.

> Real GS training officially needs a CUDA GPU. On CPU it’s impractical (or unsupported, depending on fork).

### Mesh (`.glb`) vs Gaussian Splatting (`.ply`)

The default **`reconstruction_backend` is `dust3r`**, which produces a **polygon mesh** (`.glb`). That pipeline does **not** run 3D Gaussian Splatting. If your `model_format` is `glb`, you are on the mesh path.

To use **Gaussian Splatting**, set **`reconstruction_backend`: `"gaussian_splatting"`** in `PUT /settings`, plus **`gs_repo_path`**, **`colmap_binary_path`** (for `gs_init_source="colmap"`), and a CUDA-capable machine for practical training times. The API then outputs `.ply` and sets **`model_format`: `"ply"`**.

### SAM: mask the center subject

`PUT /settings` includes **`sam_segmentation_mode`** (default **`center_point`**): SAM is prompted with a **positive point at the image center**, which targets the object in the middle of the frame. Other modes: **`auto_masks_center_bias`** (all auto-masks, scored by size × closeness to center) and **`auto_masks_largest_area`** (legacy: largest mask only — often the background).

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
- `GET /models` — list `.glb` and `.ply` files under the configured output directory (`url`, `size_bytes`, … — no server filesystem paths).
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

## Database (jobs)

- Jobs are stored in **SQLite**: default file **`data/jobs.sqlite`** under the app root (override with env **`APP_DATABASE_PATH`**).
- Each row has **`job_id`**, **`status`**, optional **`model_url`**, **`error`**, **`image_count`**, Unix **`created_at` / `updated_at`**, and API responses also include **`created_at_iso` / `updated_at_iso`** (UTC, `Z`).
- On first start, if legacy **`config/jobs.json`** exists and the DB is empty, it is **imported once** and the JSON file is renamed to **`config/jobs.json.bak`**.

## Integration points

- To use PostgreSQL/MySQL later, replace the `jobs_db` module (same `JobManager` API) or add a SQLAlchemy layer.
- Set `job_manager.notifier` in `app/main.py` to a real `WebSocketNotifier` for push updates (pipeline status is already reflected via `update_job`).

