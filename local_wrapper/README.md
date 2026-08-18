# Wan 2.2 self-hosted video generation — usage & integration

Runs Wan 2.2 14B image-to-video on a RunPod GPU. Models live on a network
volume, so workers can be stopped ($0/hr GPU cost) between sessions and recreated
without downloading the weights again. Never commit real pod IDs or worker URLs.

## Day-to-day usage (CLI)

```bash
# full batch: auto-starts the pod, generates, auto-stops
python3 run_batch.py --batch jobs.csv

# fast drafts (720p/16fps, ~5 min/clip) — pick keepers, then re-run finals
python3 run_batch.py --batch jobs.csv --draft --seeds 3

# finals (1080p/32fps with upscale + interpolation, ~8.5 min/clip)
python3 run_batch.py --batch keepers.csv

# single clip against an already-running pod (no start/stop)
python3 wan_client.py --server https://<POD_ID>-8188.proxy.runpod.net \
    --image shot01.jpg --prompt "slow dolly-in, cinematic" --draft
```

Requires in your shell: `RUNPOD_API_KEY` (runpod.io → Settings → API Keys) and
either `RUNPOD_POD_NAME` or `RUNPOD_POD_ID` (run_batch only).

- `workflow_api.json` — FINAL: 6 steps → 2x live-action upscale → 1920×1080 → RIFE → 32fps mp4
- `workflow_draft.json` — DRAFT: 4 steps, native 1280×720 @ 16fps mp4

## Local API server (fal.ai-style, for your own apps)

`api_server.py` wraps everything in a simple localhost API — your apps never
touch Runpod or ComfyUI directly:

```bash
export RUNPOD_API_KEY=...  RUNPOD_POD_NAME=wan22-video-gen
python3 api_server.py        # http://127.0.0.1:8787
```

```bash
# submit (draft=true for fast previews; omit for full 1080p pass)
curl -X POST http://127.0.0.1:8787/generate \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: shot01-draft-v1' \
  -d '{"image_path": "/abs/path/shot01.jpg", "prompt": "slow dolly-in", "draft": true}'
# -> {"job_id": "...", "status_url": "/status/...", "result_url": "/result/..."}

curl http://127.0.0.1:8787/status/<job_id>     # queued|starting_pod|generating|uploading|done|error|cancelled
curl -L -o clip.mp4 http://127.0.0.1:8787/result/<job_id>
```

Retry the same request with the same `Idempotency-Key` to retrieve the existing
job without paying for a duplicate generation. Reusing a key for different
inputs returns HTTP 409.

### Product API (`/v1`)

New applications should use the stable, tenant-scoped resource API. The legacy
routes above remain compatible.

```bash
curl -X POST http://127.0.0.1:8787/v1/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-product-key' \
  -H 'Idempotency-Key: shot01-draft-v1' \
  -d '{"image_b64":"...", "prompt":"slow dolly-in", "model":"wan-2.2-draft"}'

curl -H 'Authorization: Bearer your-product-key' \
  'http://127.0.0.1:8787/v1/generations?limit=20&status=done'
curl -X POST -H 'Authorization: Bearer your-product-key' \
  http://127.0.0.1:8787/v1/generations/<id>/cancel
```

The API also supports `GET /v1/models`, generation get/result, explicit retry,
and deletion. See [`openapi.yaml`](openapi.yaml) for the full contract. Progress
is deliberately coarse stage progress, not a fabricated render percentage.

Configure product keys as a JSON map. Idempotency keys and all reads/mutations
are scoped to `owner_id`; an admin key can inspect all tenants:

```bash
export WAN_API_KEYS_JSON='{
  "replace-with-a-long-random-key":{"owner_id":"user-123","project_id":"default","role":"user"},
  "replace-with-an-admin-key":{"owner_id":"ops","project_id":"default","role":"admin"}
}'
export WAN_CORS_ORIGINS='https://app.example.com,http://localhost:3000'
```

`WAN_API_TOKEN` remains an admin-compatible bootstrap token. When no keys are
configured, anonymous access is permitted only on a loopback-bound development
server. Never ship product API keys in browser bundles; put authentication and
per-user authorization in your application backend.

The model registry currently exposes `wan-2.2-draft`, `wan-2.2-standard`, and
`wan-2.2-hq`. A future Wan 2.7 deployment should be added as another adapter and
model ID, preserving the same generation API and UI. Model cost data appears
only after at least three completed local samples; the service does not publish
fixed marketing estimates as measured costs.

The first job auto-starts the pod (2–4 min); after `WAN_IDLE_STOP_MIN` minutes
(default 10) with no work it auto-stops the pod, so idle GPU billing can't happen.
Without `RUNPOD_API_KEY` it still works against a pod you started manually
(auto start/stop disabled). Each worker runs one job at a time; the pool can process
jobs concurrently when autoscaling adds workers.

## Durable metadata and cheap video storage

Do not put MP4 bytes in Postgres. The production path stores searchable job,
attempt, idempotency, and cost metadata in Postgres and stores the generated
video in a private S3-compatible bucket such as Cloudflare R2 or AWS S3.

```bash
python3 -m pip install -r requirements-api.txt

export DATABASE_URL='postgresql://...'
export WAN_STORAGE_BACKEND=s3
export WAN_S3_BUCKET=wan-videos
export WAN_S3_ENDPOINT_URL='https://<ACCOUNT_ID>.r2.cloudflarestorage.com'
export WAN_S3_REGION=auto
export WAN_S3_ACCESS_KEY_ID='...'
export WAN_S3_SECRET_ACCESS_KEY='...'
python3 api_server.py
```

`GET /result/<job_id>` returns a temporary HTTP 302 to a signed object URL, so
the API server does not pay the memory or bandwidth cost of proxying the MP4.
The default link lifetime is one hour. Upload retries reuse the downloaded local
artifact and never rerun the GPU generation merely because object storage failed.

The bucket must remain private. Configure an object lifecycle policy separately
if free/trial videos should expire after a fixed retention period. Cloudflare's
[R2 boto3 example](https://developers.cloudflare.com/r2/examples/aws/boto3/)
documents the endpoint and signed-URL setup.

Local development remains zero-config: without `DATABASE_URL` and
`WAN_STORAGE_BACKEND=s3`, metadata stays in `outputs/jobs.json` and videos remain
on local disk.


## Remote access (other devices / apps)

The server binds per `WAN_BIND` (default `127.0.0.1`; set `0.0.0.0` to serve the
network) and requires a configured bearer token on every endpoint except
`/health`. It refuses to network-bind without `WAN_API_TOKEN` or
`WAN_API_KEYS_JSON`.
Both values live in `.env` next to `api_server.py`, which is loaded at start:

```bash
python3 api_server.py
```

Remote callers that can't reference this Mac's disk send the image inline:

```json
POST /generate
{"image_b64": "<base64 of jpg/png>", "prompt": "...", "quality": "final", "seconds": 5}
```

(≤25MB body; temporary uploads are removed after the result is durably stored.)
Remote callers cannot
submit `image_path`; local-path reads are accepted only while the server is bound
to loopback. Recommended transport for
other devices: Tailscale — the Mac gets a stable private hostname, no open ports,
and clients use `http://<mac-tailscale-name>:8787` with the same token.

## Calling the pod directly from other apps

The pod is a plain **ComfyUI HTTP API**. ComfyUI does not provide the user and
tenant security required by a public product. Treat the endpoint as private,
never expose it in client responses or logs, and route public callers through
the authenticated wrapper. For a public deployment, add an authenticated worker
gateway or use a RunPod Serverless handler rather than relying on URL secrecy.

Base URL while the pod runs: `https://<POD_ID>-8188.proxy.runpod.net`

**Gotcha:** Runpod's proxy rejects the default Python-urllib user agent with 403.
Send any custom `User-Agent` header.

### Lifecycle (start → generate → stop)

1. `POST https://rest.runpod.io/v1/pods/{POD_ID}/start` with header
   `Authorization: Bearer $RUNPOD_API_KEY`
2. Poll `GET {BASE}/system_stats` until HTTP 200 (2–4 min after start).
3. Upload the source image: `POST {BASE}/upload/image` (multipart form,
   field `image`, optional `overwrite=true`). Response echoes the stored `name`.
4. Load `workflow_api.json` (or `workflow_draft.json`), patch three things:
   - the `LoadImage` node's `inputs.image` → uploaded file name
   - the positive `CLIPTextEncode` node's `inputs.text` → your motion prompt
   - both `KSamplerAdvanced` nodes' `inputs.noise_seed` → any int (vary per take)
   then `POST {BASE}/prompt` with body `{"prompt": <workflow>, "client_id": "<uuid>"}`.
   Response contains `prompt_id`.
5. Poll `GET {BASE}/history/{prompt_id}` until `outputs` is non-empty
   (finals ~8.5 min, drafts ~5 min). Grab `filename`/`subfolder` from the
   `videos`/`images` entries.
6. Download: `GET {BASE}/view?filename=<f>&subfolder=<s>&type=output`
7. `POST https://rest.runpod.io/v1/pods/{POD_ID}/stop` (same auth) — **always**,
   or the provider continues charging the worker's hourly rate.

Steps 3–6 are exactly what `wan_client.py` implements (`run_job()` is importable),
and steps 1/2/7 are `run_batch.py` — crib from them.

### Notes

- One job at a time: the 4090 runs jobs serially; extra POSTs to `/prompt` queue up
  (fine — that's how batching works). For concurrent load from an app with multiple
  users, this single-pod setup isn't enough — the next step up is Runpod Serverless
  (per-second billing, autoscaling workers), a separate build.
- The pod's GPU can be reclaimed by other users while stopped. If start fails with
  "GPUs no longer available", use the console's "Automatically migrate" (keeps
  everything, new pod id — update `RUNPOD_POD_ID`).
- Each attempt reports elapsed time, hourly GPU rate, and estimated GPU cost.
  Use those fields instead of fixed per-clip claims, then reconcile them against
  RunPod billing before setting prices.
