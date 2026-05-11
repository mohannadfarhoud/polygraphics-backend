# Split Windows deploy: HTTPS API (CPU) + outbound GPU worker

The API and SQLite job database run on a machine with a **static IP / DNS** and HTTPS (for example behind IIS or nginx). The heavy reconstruction runs on a **second PC with a GPU** that has **no public IP** and only needs **outbound HTTPS** to the same domain.

Public API prefix in production:

`https://agentmanager.easymediasuitecloud.com/polygraph`

Example status URL:

`https://agentmanager.easymediasuitecloud.com/polygraph/server/status`

## API server (CPU)

1. Clone the repo and run **`scripts\install_api_only_windows.ps1`** (recommended name for “DB + API only”), or **`scripts\install_api_server_windows.ps1`** without `-SingleMachine`. Both set **`APP_REMOTE_WORKERS=true`**, **`APP_WORKER_TOKEN`**, and public URL keys in `.env`. Do **not** use `-SingleMachine` on this host.
2. Configure nginx/IIS so `/polygraph` is forwarded to Uvicorn and **paths** `/polygraph/uploads`, `/polygraph/output`, etc. match what the app exposes (`APP_ROOT_PATH=/polygraph`). For **WebSocket** job notifications (`wss://…/polygraph/internal/worker/ws`), nginx needs upgrade headers, for example:
   ```nginx
   proxy_http_version 1.1;
   proxy_set_header Upgrade $http_upgrade;
   proxy_set_header Connection "upgrade";
   ```
3. Set **`APP_PUBLIC_BASE_URL`** to the full public base **including** `/polygraph` so remote workers receive absolute download URLs for inputs.
4. Set **`APP_REMOTE_WORKERS=true`** and a strong **`APP_WORKER_TOKEN`**. Use the same token on every GPU worker.
5. Set **`APP_MODEL_BASE_URL`** / **`APP_UPLOADS_BASE_URL`** to your public `.../polygraph/output` and `.../polygraph/uploads` if clients must see absolute model/upload URLs (see `.env.example`).
6. Run the service: `scripts\run_server.ps1` or `scripts\install_windows_service.ps1` with NSSM.

## GPU worker

1. Clone the **same** repository on the worker.
2. Run `scripts\install_worker_windows.ps1` (venv, CUDA PyTorch, SAM + DUSt3R stack from `install_ml_windows.ps1 -SkipTorch`, checkpoints).
3. Copy `.env.worker.example` to `.env.worker` and set **`POLYGRAPH_API_BASE`** and **`POLYGRAPH_WORKER_TOKEN`** (must match the API).
4. Ensure **`PUT /settings`** paths (SAM/DUSt3R checkpoints, DUSt3R repo, COLMAP) exist on the worker machine, or use **`POLYGRAPH_OVERRIDE_*`** variables in `.env.worker` (see `.env.worker.example`).
5. Run `scripts\run_worker.ps1` or install an NSSM service that runs that script.

## Firewall

- API machine: allow inbound HTTPS (and SSH/RDP as you prefer).
- Worker: **no inbound** required for job processing; only outbound TCP 443 to your API host.

## Troubleshooting

- **503 on `/internal/worker/next`**: enable **`APP_REMOTE_WORKERS`** on the API or set **`APP_WORKER_TOKEN`** on the API.
- **Worker WebSocket disconnects or never gets jobs**: confirm nginx **Upgrade** headers for `/internal/worker/ws`; worker falls back to **`GET /internal/worker/next`** on **`POLYGRAPH_POLL_SECONDS`** (default 30s).
- **Worker cannot download images**: **`APP_PUBLIC_BASE_URL`** must be reachable from the worker and must include the `/polygraph` prefix.
- **Wrong model paths on GPU**: use **`POLYGRAPH_OVERRIDE_*`** or align **`PUT /settings`** with paths on the worker disk.
