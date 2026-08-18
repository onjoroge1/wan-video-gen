# Zero → running: self-hosted Wan 2.2 video API on Runpod

What you get: a localhost (optionally LAN/remote) HTTP API that turns
`image + prompt → 5s–7.5s video clip`, with pods that start, heal, scale, and
stop themselves. Per-attempt cost is measured from the actual worker rate.

## 1. Runpod account

1. Sign up at runpod.io, load credit (prepaid; $25 is a comfortable start).
2. Settings → API Keys → create one. This goes in `.env` as `RUNPOD_API_KEY`.

## 2. Network volume (your model store — the only permanent thing)

Storage → New Network Volume:
- **Datacenter**: pick one showing high availability for your GPU
  (default config assumes `EU-RO-1`; override with `RUNPOD_DC`).
- **Size**: 250 GB. The Wan 2.2 model set + extras is roughly 150 GB; verify
  current storage pricing in your selected region before provisioning.
- Put its id in `.env` as `RUNPOD_VOLUME_ID`.

Everything else (pods) is disposable; the volume is what you'd cry about.

## 3. Configure + first pod

```bash
cp .env.example .env && $EDITOR .env      # key, volume id, pod name
chmod 600 .env
set -a; source .env; set +a
python3 deploy_pod.py
```

First boot on a fresh volume takes ~20 min: the template (hearmeman v26)
downloads the core Wan 2.2 models to the volume. Later pods boot in ~90s.

## 4. One-time model bootstrap

Open the pod's **JupyterLab** (Runpod console → pod → Connect → port 8888),
open a Terminal, paste the contents of `bootstrap_models.sh`, run it
(~30 GB: fp8 models, fast-LoRAs, upscaler, RIFE). Then restart the pod's
ComfyUI (easiest: Runpod console → pod → Restart). Validate the installed
models and custom nodes before paying for a full generation:

```bash
python3 validate_runtime.py --check-models \
  --server https://<POD_ID>-8188.proxy.runpod.net
```

## 5. Run the API server

Install the lightweight API-only persistence dependencies (separate from the
large Wan model environment):

```bash
python3 -m pip install -r requirements-api.txt
```

For production, set `DATABASE_URL` and the `WAN_S3_*` values shown in
`.env.example`. Keep the bucket private. Postgres stores job metadata; generated
video bytes go to S3-compatible object storage. Without those variables the
wrapper deliberately falls back to `outputs/jobs.json` and local video files.

To import existing JSON history after configuring Postgres:

```bash
python3 migrate_jobs.py --dry-run
python3 migrate_jobs.py
```

```bash
python3 api_server.py
# [api] binding 127.0.0.1:8787
```

Smoke test:

```bash
curl -X POST http://127.0.0.1:8787/generate -H 'Content-Type: application/json' \
  -d '{"image_path": "'$PWD'/your_test.jpg", "prompt": "slow cinematic push-in", "draft": true}'
curl http://127.0.0.1:8787/status/<job_id>     # queued → generating → uploading → done
curl -L -o clip.mp4 http://127.0.0.1:8787/result/<job_id>
```

See `README.md` for the full API contract (quality tiers, seconds, seeds,
batches, remote access with `WAN_BIND` + `WAN_API_TOKEN` + `image_b64`).

For a multi-user app, configure `WAN_API_KEYS_JSON` and use `/v1/generations`.
Set `WAN_CORS_ORIGINS` to the exact UI origins that may call the API. The v1
contract is checked in as `openapi.yaml`; legacy endpoints remain available for
existing scripts.

## Operating notes (learned the expensive way)

- **Stopped pods lose their GPU** in busy datacenters, and Runpod's start
  endpoint then 500s forever. The server handles this: it retries, declares
  the pod unstartable, **creates a replacement on the volume, and terminates
  the corpse** — expect pod IDs to change over time. Address pods by the
  name prefix, never by ID.
- **RAM matters more than VRAM for loading**: the 14B fp16 pair OOMs on
  hosts with ~43 GB container RAM. The deploy spec pins `minRAMPerGPU: 40`
  and the workflows use fp8_scaled weights; don't "upgrade" to fp16 without
  ~80 GB RAM hosts.
- **The watchdog fails toward $0/hr**: if pods run 30 min with zero completed
  jobs, everything is stopped and active jobs are failed loudly. A wedged
  pipeline should never be a silent $18/day.
- **Measure costs from your own jobs**: every attempt records elapsed seconds,
  the pod's hourly rate, and `estimated_gpu_cost`. Cold start, retries, and idle
  time make fixed per-clip estimates misleading; reconcile estimates against
  RunPod billing before setting product prices.
- Never commit `.env`. The `.gitignore` here enforces that; keep it.
