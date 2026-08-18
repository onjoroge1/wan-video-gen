# Wan 2.2 self-hosted video generation — usage & integration

Runs Wan 2.2 14B image-to-video on a Runpod RTX 4090 (pod `wan22-video-gen-migration`,
id `eyph723c5tuta1`, EU-RO-1). Models live on the network volume `wan22-models`, so the
pod can be stopped ($0/hr) between sessions and restarted in ~3 minutes.

## Day-to-day usage (CLI)

```bash
# full batch: auto-starts the pod, generates, auto-stops
python3 run_batch.py --batch jobs.csv

# fast drafts (720p/16fps, ~5 min/clip) — pick keepers, then re-run finals
python3 run_batch.py --batch jobs.csv --draft --seeds 3

# finals (1080p/32fps with upscale + interpolation, ~8.5 min/clip)
python3 run_batch.py --batch keepers.csv

# single clip against an already-running pod (no start/stop)
python3 wan_client.py --server https://eyph723c5tuta1-8188.proxy.runpod.net \
    --image shot01.jpg --prompt "slow dolly-in, cinematic" --draft
```

Requires in your shell: `RUNPOD_API_KEY` (runpod.io → Settings → API Keys) and
`RUNPOD_POD_ID=eyph723c5tuta1` (run_batch only; wan_client needs neither).

- `workflow_api.json` — FINAL: 6 steps → 2x live-action upscale → 1920×1080 → RIFE → 32fps mp4
- `workflow_draft.json` — DRAFT: 4 steps, native 1280×720 @ 16fps mp4

## Local API server (fal.ai-style, for your own apps)

`api_server.py` wraps everything in a simple localhost API — your apps never
touch Runpod or ComfyUI directly:

```bash
export RUNPOD_API_KEY=...  RUNPOD_POD_ID=eyph723c5tuta1
python3 api_server.py        # http://127.0.0.1:8787
```

```bash
# submit (draft=true for fast previews; omit for full 1080p pass)
curl -X POST http://127.0.0.1:8787/generate -H 'Content-Type: application/json' \
  -d '{"image_path": "/abs/path/shot01.jpg", "prompt": "slow dolly-in", "draft": true}'
# -> {"job_id": "...", "status_url": "/status/...", "result_url": "/result/..."}

curl http://127.0.0.1:8787/status/<job_id>     # queued|starting_pod|generating|done|error
curl -o clip.mp4 http://127.0.0.1:8787/result/<job_id>
```

The first job auto-starts the pod (2–4 min); after `WAN_IDLE_STOP_MIN` minutes
(default 10) with no work it auto-stops the pod, so idle GPU billing can't happen.
Without `RUNPOD_API_KEY` it still works against a pod you started manually
(auto start/stop disabled). Jobs run serially — queue as many as you like.


## Remote access (other devices / apps)

The server binds per `WAN_BIND` (default `127.0.0.1`; set `0.0.0.0` to serve the
network) and requires `Authorization: Bearer $WAN_API_TOKEN` on every endpoint
except `/health` when a token is set. It refuses to network-bind without a token.
Both values live in `.env` next to `api_server.py`; inject them at start:

```bash
set -a; source <(grep -E '^(WAN_|RUNPOD_)' .env); set +a; python3 api_server.py
```

Remote callers that can't reference this Mac's disk send the image inline:

```json
POST /generate
{"image_b64": "<base64 of jpg/png>", "prompt": "...", "quality": "final", "seconds": 5}
```

(≤25MB body; uploads persist under `outputs/uploads/`.) Recommended transport for
other devices: Tailscale — the Mac gets a stable private hostname, no open ports,
and clients use `http://<mac-tailscale-name>:8787` with the same token.

## Calling the pod directly from other apps

The pod is a plain **ComfyUI HTTP API** — anything that can make HTTP requests
(n8n, Make, a Next.js backend, another Python service) can drive it. No auth on the
ComfyUI endpoint itself (the URL is the secret — don't publish it).

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
   or you pay $0.74/hr for an idle GPU.

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
- Costs: ~$0.06/draft, ~$0.10/final at $0.75/hr; storage flat $17.50/mo.
