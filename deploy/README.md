# Reverse proxy (Nginx)

The Python app listens on **port 8000** and mounts:

| URL path | Purpose |
|----------|---------|
| `/output/` | Generated `.glb` / `.ply` models |
| `/uploads/` | Per-job uploaded images (previews) |
| `/jobs`, `/settings`, … | REST API |
| `/swagger` | Swagger UI (if not using only `/polygraph`) |

## Two common setups

### A — Root-relative asset URLs (`/output/...`, `/uploads/...`)

Use when **`APP_MODEL_BASE_URL`** / **`APP_UPLOADS_BASE_URL`** are unset (or only relative paths).

You need **both**:

1. **`nginx-output-uploads.conf`** — maps `https://your-domain/output/` and `/uploads/` to Uvicorn.
2. **`nginx-polygraph-api.conf`** — maps `https://your-domain/polygraph/` → API (optional if API is only internal).

### B — Everything under `/polygraph/` (including files)

Set in `.env`:

```env
APP_MODEL_BASE_URL=https://your-domain/polygraph/output
APP_UPLOADS_BASE_URL=https://your-domain/polygraph/uploads
APP_ROOT_PATH=/polygraph
```

Then only the **`nginx-polygraph-api.conf`** block is required for API **and** static files (because `/polygraph/output/...` is stripped and forwarded as `/output/...`).

## Install

1. Open your site’s `server { ... }` for HTTPS (e.g. `/etc/nginx/sites-available/your-site`).
2. Paste the `location` blocks from the chosen `.conf` files **inside** that `server` block.
3. Test and reload:

```bash
sudo nginx -t && sudo nginx -s reload
```

## Windows

If Nginx is on Windows, edit `nginx.conf` or an included `conf`, same syntax, then:

```cmd
nginx -t
nginx -s reload
```

## Verify

```bash
curl -sI "https://your-domain/output/SOME_JOB_ID.glb" | head -5
curl -sI "https://your-domain/uploads/SOME_JOB_ID/input_000.jpg" | head -5
```

Expect `HTTP/1.1 200` (or `304`) if the file exists on the server disk under `polygraphics-backend/output/` and `uploads/`.
