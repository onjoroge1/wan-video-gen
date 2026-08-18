#!/usr/bin/env python3
"""fal.ai-style local API for self-hosted Wan 2.2 pods on Runpod.

Production behavior:
  - Worker POOL: every Runpod pod whose name starts with RUNPOD_POD_NAME gets a
    worker thread; jobs are dispatched to whichever pod is free (deploy a second
    pod with the same name prefix and throughput doubles, no config change).
  - Pods auto-start on demand (with retry/backoff through Runpod 5xx blips) and
    auto-stop after WAN_IDLE_STOP_MIN minutes with no work (default 10).
  - Jobs REQUEUE on infrastructure failures (pod start/boot trouble) up to
    WAN_JOB_RETRIES times before erroring; generation errors don't requeue.
  - Job state persists to Postgres when DATABASE_URL is set, otherwise to the
    local outputs/jobs.json development fallback.
  - Generated videos upload to S3-compatible storage when configured; local
    files remain the zero-dependency development fallback.
  - While jobs are active, macOS sleep is held off via `caffeinate`.

Setup (either):
  RUNPOD_API_KEY=... RUNPOD_POD_NAME=wan22-video-gen python3 api_server.py
  RUNPOD_API_KEY=... RUNPOD_POD_ID=<id>              python3 api_server.py

Endpoints:
  POST /generate   {"image_path": "/abs/path.jpg", "prompt": "...",
                    "draft": true, "seed": 123}       # draft+seed optional
  GET  /status/<job_id> | /result/<job_id> | /jobs | /health
  POST /v1/generations | GET /v1/generations[/<id>] | GET /v1/models
  POST /v1/generations/<id>/{cancel,retry} | DELETE /v1/generations/<id>
"""

import base64
import hashlib
import ipaddress
import json
import math
import os
import shlex

# macOS: urllib's system-proxy autodetection (_scproxy) is not thread-safe and
# raises EDEADLK ("Resource deadlock avoided") in threaded servers under launchd.
# We never need a proxy for runpod.net/localhost — disable detection outright.
os.environ.setdefault("no_proxy", "*")
import queue
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import run_batch
import persistence
import product_api
import storage
import wan_client


def _load_dotenv():
    """Load WAN_/RUNPOD_ settings without requiring a shell-compatible .env."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        env_file = Path.home() / "Documents" / "video" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        allowed = key == "DATABASE_URL" or key.startswith(("RUNPOD_", "WAN_"))
        if not allowed or key in os.environ:
            continue
        values = shlex.split(raw_value, comments=True, posix=True)
        if values:
            os.environ[key] = values[0]


_load_dotenv()

PORT = int(os.environ.get("WAN_API_PORT", "8787"))
BIND = os.environ.get("WAN_BIND", "127.0.0.1")
API_TOKEN = os.environ.get("WAN_API_TOKEN", "")
try:
    API_KEYS = product_api.load_api_keys(os.environ.get("WAN_API_KEYS_JSON", ""))
except ValueError as exc:
    raise RuntimeError(f"invalid WAN_API_KEYS_JSON: {exc}") from exc
CORS_ORIGINS = {
    origin.strip() for origin in os.environ.get("WAN_CORS_ORIGINS", "").split(",")
    if origin.strip()
}
MAX_UPLOAD_MB = int(os.environ.get("WAN_MAX_UPLOAD_MB", "25"))
MAX_PROMPT_CHARS = int(os.environ.get("WAN_MAX_PROMPT_CHARS", "4000"))
IDLE_STOP_MIN = float(os.environ.get("WAN_IDLE_STOP_MIN", "10"))
JOB_RETRIES = int(os.environ.get("WAN_JOB_RETRIES", "3"))
OUT_DIR = Path(__file__).parent / "outputs"
JOBS_FILE = OUT_DIR / "jobs.json"
DATABASE_URL = os.environ.get("DATABASE_URL", "")
SIGNED_URL_SECONDS = max(60, min(
    604800, int(os.environ.get("WAN_SIGNED_URL_SECONDS", "3600"))
))
WORKFLOWS = {
    "draft": Path(__file__).parent / "workflow_draft.json",   # 4-step distill, 720p16, fastest
    "final": Path(__file__).parent / "workflow_api.json",     # 6-step distill + upscale + RIFE
    "hq":    Path(__file__).parent / "workflow_hq.json",      # 20-step cfg 3.5, no distill — realism
}
MODEL_REGISTRY = {
    "wan-2.2-draft": {
        "name": "Wan 2.2 Draft", "family": "wan", "version": "2.2",
        "quality": "draft", "workflow": WORKFLOWS["draft"], "status": "available",
    },
    "wan-2.2-standard": {
        "name": "Wan 2.2 Standard", "family": "wan", "version": "2.2",
        "quality": "final", "workflow": WORKFLOWS["final"], "status": "available",
    },
    "wan-2.2-hq": {
        "name": "Wan 2.2 HQ", "family": "wan", "version": "2.2",
        "quality": "hq", "workflow": WORKFLOWS["hq"], "status": "available",
    },
}
QUALITY_MODELS = {model["quality"]: model_id for model_id, model in MODEL_REGISTRY.items()}
API_KEY = os.environ.get("RUNPOD_API_KEY")
JOB_REPOSITORY = persistence.build_job_repository(DATABASE_URL, JOBS_FILE)
VIDEO_STORAGE = storage.build_video_storage({
    "backend": os.environ.get("WAN_STORAGE_BACKEND", "local"),
    "bucket": os.environ.get("WAN_S3_BUCKET"),
    "endpoint_url": os.environ.get("WAN_S3_ENDPOINT_URL"),
    "region": os.environ.get("WAN_S3_REGION", "auto"),
    "prefix": os.environ.get("WAN_S3_PREFIX", "videos"),
    "access_key": os.environ.get("WAN_S3_ACCESS_KEY_ID"),
    "secret_key": os.environ.get("WAN_S3_SECRET_ACCESS_KEY"),
    "delete_local": os.environ.get("WAN_DELETE_LOCAL_AFTER_UPLOAD", "true").lower()
                    not in ("0", "false", "no"),
})

jobs = {}                 # job_id -> dict
job_queue = queue.Queue()
jobs_lock = threading.RLock()
last_activity = time.time()
last_job_done = time.time()   # money watchdog: last time ANY job completed
WATCHDOG_MIN = float(os.environ.get("WAN_WATCHDOG_MIN", "30"))
caffeinate_proc = None


def log(msg):
    print(f"[api] {msg}", flush=True)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def public_job(job):
    """Return job data without local paths, workflow paths, or provider IDs."""
    hidden = {
        "workflow", "image_path", "pod", "provider_prompt_id", "client_id",
        "idempotency_key", "request_fingerprint", "output", "output_key",
        "pending_output", "managed_input", "input_sha256", "error",
    }
    result = {key: value for key, value in job.items() if key not in hidden}
    result["attempt_history"] = [
        {key: value for key, value in attempt.items() if key not in ("pod", "error")}
        for attempt in job.get("attempt_history", [])
    ]
    return result


def health_payload():
    """Health must not disclose pod IDs that reveal direct ComfyUI endpoints."""
    return {
        "ok": True,
        "workers": {
            "total": len(PODS),
            "running": sum(1 for pod in PODS if pod.running and not getattr(pod, "retired", False)),
            "available": sum(1 for pod in PODS if not getattr(pod, "retired", False)),
        },
        "queued": job_queue.qsize(),
        "persistence": JOB_REPOSITORY.backend,
        "storage": VIDEO_STORAGE.backend,
    }


def is_loopback(address):
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def is_loopback_bind(address):
    return address == "localhost" or is_loopback(address)


def detect_image_extension(raw):
    """Validate supported image bytes before passing them to native decoders."""
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("image must be a JPEG, PNG, or WebP file")


def request_fingerprint(prompt, image_sha256, model_id, frames, requested_seed):
    canonical = json.dumps({
        "prompt": prompt,
        "image_sha256": image_sha256,
        "model": model_id,
        "frames": frames,
        "seed": requested_seed,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def submission_response(job_id, job, duplicate=False):
    return {
        "job_id": job_id,
        "status_url": f"/status/{job_id}",
        "result_url": f"/result/{job_id}",
        "seed": job["seed"],
        "draft": job["draft"],
        "duplicate": duplicate,
    }


def submit_generation(req, idempotency_key="", allow_local_path=True, principal=None,
                      strict=False):
    """Validate and enqueue one request. Returns (HTTP status, response payload)."""
    global last_activity
    prompt = req.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return 400, {"error": "a non-empty prompt is required"}
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        return 400, {"error": f"prompt exceeds {MAX_PROMPT_CHARS} characters"}

    principal = principal or product_api.Principal("local", "admin", "default")
    requested_model = req.get("model")
    if requested_model:
        model = MODEL_REGISTRY.get(requested_model)
        if not model:
            return 400, {"error": f"unknown model; use one of {sorted(MODEL_REGISTRY)}"}
        quality = model["quality"]
        model_id = requested_model
    else:
        quality = req.get("quality") or ("draft" if req.get("draft") else "final")
        model_id = QUALITY_MODELS.get(quality)
    if quality not in WORKFLOWS or not model_id:
        return 400, {"error": f"quality must be one of {sorted(WORKFLOWS)}"}
    try:
        seconds_value = req.get("seconds", 5)
        seconds = float(5 if seconds_value is None else seconds_value)
        if not math.isfinite(seconds):
            raise ValueError
        if strict and not 2 <= seconds <= 7.5:
            return 400, {"error": "seconds must be between 2 and 7.5"}
        frames = max(33, min(121, round(seconds * 16) + 1))
        requested_seed = req.get("seed")
        if requested_seed is not None:
            requested_seed = int(requested_seed)
            if requested_seed < 0 or requested_seed > 2**63 - 1:
                raise ValueError
    except (TypeError, ValueError):
        return 400, {"error": "seconds or seed is invalid"}

    image_path = req.get("image_path", "")
    image_b64 = req.get("image_b64", "")
    if bool(image_path) == bool(image_b64):
        return 400, {"error": "provide exactly one of image_path or image_b64"}

    raw = None
    if image_b64:
        try:
            raw = base64.b64decode(image_b64, validate=True)
            extension = detect_image_extension(raw)
        except Exception as exc:
            return 400, {"error": str(exc) or "image_b64 is invalid"}
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            return 413, {"error": f"decoded image exceeds {MAX_UPLOAD_MB}MB"}
        image_sha256 = hashlib.sha256(raw).hexdigest()
    else:
        if not allow_local_path:
            return 400, {"error": "image_path is allowed only for localhost callers"}
        path = Path(str(image_path)).expanduser().resolve()
        if not path.is_file():
            return 400, {"error": "image file was not found"}
        if path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            return 413, {"error": f"image exceeds {MAX_UPLOAD_MB}MB"}
        raw_for_validation = path.read_bytes()
        try:
            detect_image_extension(raw_for_validation)
        except ValueError as exc:
            return 400, {"error": str(exc)}
        image_sha256 = hashlib.sha256(raw_for_validation).hexdigest()
        image_path = str(path)

    idempotency_key = (idempotency_key or req.get("idempotency_key") or "").strip()
    if idempotency_key and not 8 <= len(idempotency_key) <= 128:
        return 400, {"error": "Idempotency-Key must contain 8 to 128 characters"}
    fingerprint = request_fingerprint(
        prompt, image_sha256, model_id, frames, requested_seed
    )

    with jobs_lock:
        if idempotency_key:
            for existing_id, existing in jobs.items():
                if existing.get("owner_id", "local") != principal.owner_id:
                    continue
                if existing.get("idempotency_key") != idempotency_key:
                    continue
                if existing.get("request_fingerprint") != fingerprint:
                    return 409, {"error": "Idempotency-Key was reused for another request"}
                return 200, submission_response(existing_id, existing, duplicate=True)

        if raw is not None:
            up_dir = OUT_DIR / "uploads"
            up_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(up_dir / f"{uuid.uuid4().hex}{extension}")
            Path(image_path).write_bytes(raw)

        job_id = uuid.uuid4().hex
        seed = requested_seed if requested_seed is not None else random.randint(0, 2**48)
        jobs[job_id] = {
            "status": "queued",
            "image_path": image_path,
            "input_sha256": image_sha256,
            "prompt": prompt,
            "seed": seed,
            "draft": quality == "draft",
            "quality": quality,
            "model_id": model_id,
            "frames": frames,
            "workflow": str(WORKFLOWS[quality]),
            "managed_input": raw is not None,
            "idempotency_key": idempotency_key or None,
            "request_fingerprint": fingerprint,
            "owner_id": principal.owner_id,
            "project_id": principal.project_id,
            "attempt_count": 0,
            "attempt_history": [],
            "created": now_iso(),
        }
        last_activity = time.time()
        if not save_jobs(job_id):
            jobs.pop(job_id, None)
            if raw is not None:
                Path(image_path).unlink(missing_ok=True)
            return 503, {"error": "job could not be durably accepted; retry safely"}
        job_queue.put(job_id)
        return 202, submission_response(job_id, jobs[job_id])


def cancel_generation(job_id, principal):
    """Request idempotent cancellation without interrupting another ComfyUI job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not product_api.owns(principal, job):
            return 404, "not_found", "generation was not found"
        if job.get("status") == "cancelled":
            return 200, None, job
        if job.get("status") in ("done", "error", "uploading"):
            return 409, "invalid_state", f"cannot cancel a {job['status']} generation"
        previous = dict(job)
        job["cancel_requested"] = True
        if job.get("status") == "queued":
            job["status"] = "cancelled"
            job["terminal"] = True
            job["cancelled"] = now_iso()
            job["queue_tombstone_pending"] = True
        if not save_jobs(job_id):
            job.clear()
            job.update(previous)
            return 503, "persistence_unavailable", "cancellation was not durably recorded"
        return 202 if job.get("status") != "cancelled" else 200, None, job


def retry_generation(job_id, principal):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not product_api.owns(principal, job):
            return 404, "not_found", "generation was not found"
        if job.get("status") not in ("error", "cancelled"):
            return 409, "invalid_state", "only failed or cancelled generations can be retried"
        if job.get("error_code") == "generation_error":
            return 409, "invalid_state", "fix the request and submit a new generation"
        recoverable = (
            job.get("pending_output") and os.path.isfile(job["pending_output"])
        ) or os.path.isfile(job.get("image_path", ""))
        if not recoverable:
            return 410, "input_unavailable", "the retained input is no longer available"
        previous = dict(job)
        already_enqueued = job.pop("queue_tombstone_pending", False)
        for key in ("terminal", "cancel_requested", "cancelled", "completed", "error", "error_code",
                    "provider_prompt_id", "client_id", "resume_provider_prompt"):
            job.pop(key, None)
        job["status"] = "queued"
        job["retried"] = now_iso()
        if not save_jobs(job_id):
            job.clear()
            job.update(previous)
            return 503, "persistence_unavailable", "retry was not durably recorded"
        if not already_enqueued:
            job_queue.put(job_id)
        return 202, None, job


def delete_generation(job_id, principal):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not product_api.owns(principal, job):
            return 404, "not_found", "generation was not found"
        if job.get("status") in ("queued", "starting_pod", "generating", "uploading"):
            return 409, "invalid_state", "cancel the active generation before deleting it"
        try:
            if job.get("storage_backend") == "s3" and VIDEO_STORAGE.backend != "s3":
                raise storage.StorageError("server is not configured for the job's S3 storage")
            VIDEO_STORAGE.delete(job)
            for path_key in ("image_path", "pending_output"):
                path = job.get(path_key)
                if path and (path_key != "image_path" or job.get("managed_input")):
                    Path(path).unlink(missing_ok=True)
            removed = jobs.pop(job_id)
            try:
                JOB_REPOSITORY.delete_job(job_id, jobs)
            except Exception:
                jobs[job_id] = removed
                raise
        except (storage.StorageError, persistence.PersistenceError, OSError) as exc:
            return 503, "delete_failed", str(exc)
        return 204, None, None


def save_jobs(job_id=None):
    # Persistence is best-effort: a failed write must NEVER kill a worker thread.
    # (2026-08-13: an EDEADLK here froze a worker mid-job; its jobs stayed
    # "active" forever, which disabled idle-stop and burned $7 of pod time.)
    try:
        with jobs_lock:
            if job_id is None:
                JOB_REPOSITORY.save_all(jobs)
            else:
                JOB_REPOSITORY.save_job(job_id, jobs[job_id], jobs)
        return True
    except Exception as e:
        log(f"save_jobs failed (non-fatal): {e}")
        return False


def load_jobs():
    """Reload persisted jobs; anything that was queued/in-flight goes back on
    the queue so a server restart never orphans work."""
    try:
        JOB_REPOSITORY.initialize()
        persisted = JOB_REPOSITORY.load_all()
        jobs.update(persisted)
    except Exception as e:
        raise RuntimeError(f"could not initialize job persistence: {e}") from e
    if DATABASE_URL and not persisted and JOBS_FILE.exists():
        legacy_jobs = persistence.JsonJobRepository(JOBS_FILE).load_all()
        if legacy_jobs:
            raise RuntimeError(
                "Postgres is empty but legacy outputs/jobs.json contains jobs; "
                "run migrate_jobs.py before accepting traffic"
            )
    requeued = 0
    changed = False
    for jid, j in jobs.items():
        if j.pop("queue_tombstone_pending", None):
            changed = True
        if j["status"] in ("queued", "starting_pod", "generating", "uploading"):
            pending_output = j.get("pending_output")
            if pending_output and os.path.isfile(pending_output):
                j["status"] = "queued"
                job_queue.put(jid)
                requeued += 1
                changed = True
            elif os.path.isfile(j.get("image_path", "")):
                j["resume_provider_prompt"] = bool(
                    j["status"] == "generating" and j.get("provider_prompt_id")
                )
                j["status"] = "queued"
                job_queue.put(jid)
                requeued += 1
                changed = True
            else:
                j["status"] = "error"
                j["error_code"] = "missing_input"
                j["error"] = "input image vanished during server restart"
                changed = True
    if changed:
        save_jobs()
    if requeued:
        log(f"requeued {requeued} in-flight job(s) from previous run")


def caffeinate(active):
    """Hold macOS awake while jobs are active; release when idle."""
    global caffeinate_proc
    if active and caffeinate_proc is None:
        try:
            caffeinate_proc = subprocess.Popen(["caffeinate", "-dims"])
            log("caffeinate on — Mac will not sleep while jobs run")
        except FileNotFoundError:
            caffeinate_proc = None
    elif not active and caffeinate_proc is not None:
        caffeinate_proc.terminate()
        caffeinate_proc = None
        log("caffeinate off")


class Pod:
    """One Runpod pod = one worker."""

    def __init__(self, pod_id, autoscaled=False):
        self.id = pod_id
        self.server = f"https://{pod_id}-8188.proxy.runpod.net"
        self.running = False
        self.hourly_cost = 0.0
        self.cooldown_until = 0.0   # dead pods sit out instead of stealing jobs
        self.autoscaled = autoscaled
        self.retired = False
        self.stop_event = threading.Event()
        self.worker_thread = None

    def reconcile(self):
        if not API_KEY:
            return self.running
        record = run_batch.get_pod(self.id, API_KEY)
        self.running = record.get("desiredStatus") == "RUNNING"
        self.hourly_cost = float(record.get("costPerHr") or self.hourly_cost or 0)
        return self.running

    def start_worker(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=worker, args=(self,), daemon=True, name=f"wan-worker-{self.id}"
        )
        self.worker_thread.start()

    def ensure(self):
        if self.running:
            return
        if not API_KEY:
            try:
                run_batch.wait_for_comfyui(self.server, timeout=30)
                self.running = True
                return
            except Exception:
                raise RuntimeError("pod not running and RUNPOD_API_KEY not set")
        try:
            record = run_batch.start_pod(self.id, API_KEY)  # retries 5xx internally
            self.hourly_cost = float(record.get("costPerHr") or 0)
        except RuntimeError as e:
            # Start exhausted its retries: this host's GPU is gone (EU-RO-1 reclaims
            # stopped pods within the hour). SELF-HEAL: fresh pod on the shared
            # volume, terminate the corpse, take over its identity in the pool.
            log(f"pod {self.id} unstartable ({e}) — self-healing with a fresh pod")
            old = self.id
            # Terminate first. Creating first can orphan a billable replacement
            # if cleanup of the dead provider record subsequently fails.
            run_batch.terminate_pod(old, API_KEY)
            new_id = run_batch.create_pod(API_KEY, os.environ.get(
                "RUNPOD_POD_NAME", "wan22-video-gen") + "-5090")
            self.id = new_id
            self.server = f"https://{new_id}-8188.proxy.runpod.net"
        try:
            run_batch.wait_for_comfyui(self.server, timeout=240)
        except RuntimeError:
            if not API_KEY:
                raise
            # Pod is RUNNING but ComfyUI died inside it (the v26 image launches it
            # once, never restarts — seen after RAM-heavy 121-frame HQ jobs).
            # A container restart re-runs the boot script; models live on the volume.
            log(f"pod {self.id} running but ComfyUI dead — restarting container")
            run_batch.rp("POST", f"/pods/{self.id}/restart", API_KEY)
            run_batch.wait_for_comfyui(self.server, timeout=900)
        self.running = True
        log(f"pod {self.id} ready")

    def stop(self):
        if not API_KEY:
            return False
        try:
            if not self.reconcile():
                return True
        except Exception as e:
            # Provider state reads can fail during an outage. Still issue the stop
            # request; stop_pod performs its own confirmation and fails loudly.
            log(f"could not reconcile worker before stop; attempting stop anyway: {e}")
        run_batch.stop_pod(self.id, API_KEY)
        self.running = False
        return True

    def retire(self, terminate=False):
        """Stop dispatch before removing a worker from the pool."""
        self.retired = True
        self.stop_event.set()
        if self.worker_thread and self.worker_thread is not threading.current_thread():
            self.worker_thread.join(timeout=5)
        if API_KEY:
            if terminate:
                run_batch.terminate_pod(self.id, API_KEY)
                self.running = False
            else:
                self.stop()


PODS = []


def any_active():
    return any(j["status"] in ("queued", "starting_pod", "generating") for j in jobs.values())


def worker(pod):
    global last_activity
    while not pod.stop_event.is_set():
        try:
            try:
                job_id = job_queue.get(timeout=1)
            except queue.Empty:
                continue
            if pod.stop_event.is_set():
                job_queue.put(job_id)
                job_queue.task_done()
                break
            _worker_one(pod, job_id)
        except Exception as e:
            # A worker thread must be unkillable: any escaped exception here would
            # otherwise freeze its job as "active" forever, which paralyzes
            # idle-stop and turns a bug into an unbounded GPU bill.
            log(f"worker {pod.id}: escaped exception (recovered): {type(e).__name__}: {e}")
            time.sleep(10)


def _finish_attempt(job, attempt, status, error=None):
    attempt["status"] = status
    attempt["finished"] = now_iso()
    if "started_epoch" in attempt:
        attempt["elapsed_seconds"] = round(time.time() - attempt["started_epoch"], 3)
        attempt.pop("started_epoch", None)
    if error:
        attempt["error"] = str(error)[:300]
    rate = float(attempt.get("hourly_cost") or 0)
    attempt["estimated_gpu_cost"] = round(attempt["elapsed_seconds"] / 3600 * rate, 6)
    job["estimated_gpu_cost"] = round(
        sum(float(item.get("estimated_gpu_cost") or 0)
            for item in job.get("attempt_history", [])), 6
    )


def _publish_output(job_id, job, attempt, local_path):
    """Store once, persist its address, then optionally remove the local copy."""
    local_path = str(Path(local_path).resolve())
    job["status"] = "uploading"
    job["pending_output"] = local_path
    save_jobs(job_id)
    metadata = VIDEO_STORAGE.store(local_path, job_id)
    job.update(metadata)
    job["status"] = "done"
    job["completed"] = now_iso()
    _finish_attempt(job, attempt, "done")
    if not save_jobs(job_id):
        raise storage.StorageError(
            "video stored but result metadata was not durably persisted"
        )
    VIDEO_STORAGE.cleanup_local(local_path)
    if job.get("managed_input") and job.get("image_path"):
        Path(job["image_path"]).unlink(missing_ok=True)
        job.pop("image_path", None)
        job.pop("managed_input", None)
    job.pop("pending_output", None)
    save_jobs(job_id)


def _worker_one(pod, job_id):
    global last_activity
    job = jobs.get(job_id)
    if job is None:
        job_queue.task_done()
        return
    job.pop("queue_tombstone_pending", None)
    attempt = {
        "number": job.get("attempt_count", 0) + 1,
        "status": "starting_pod",
        "started": now_iso(),
        "started_epoch": time.time(),
    }
    job["attempt_count"] = attempt["number"]
    job.setdefault("attempt_history", []).append(attempt)
    try:
        if job.get("cancel_requested"):
            if job.get("provider_prompt_id"):
                cancel_server = (
                    f"https://{job['pod']}-8188.proxy.runpod.net"
                    if job.get("pod") else pod.server
                )
                wan_client.cancel_prompt(cancel_server, job["provider_prompt_id"])
            job["status"] = "cancelled"
            job["terminal"] = True
            job["cancelled"] = job.get("cancelled", now_iso())
            _finish_attempt(job, attempt, "cancelled", job.get("error"))
            return
        if job.get("terminal"):
            _finish_attempt(job, attempt, "cancelled", job.get("error"))
            return
        if job.get("pending_output") and os.path.isfile(job["pending_output"]):
            _publish_output(job_id, job, attempt, job["pending_output"])
            global last_job_done
            last_job_done = time.time()
            log(f"job {job_id} stored without regenerating")
            return
        # A pod that just failed to start has no GPU right now (reclaimed host).
        # Hand the job straight back for a healthy worker and sit out the cooldown —
        # otherwise a dead pod keeps grabbing jobs and burning their retry budget.
        if time.time() < pod.cooldown_until:
            job["attempt_count"] -= 1
            job["attempt_history"].pop()
            job["status"] = "queued"
            job_queue.put(job_id)
            time.sleep(30)
            return
        job["pod"] = pod.id
        job["status"] = "starting_pod"
        save_jobs(job_id)
        caffeinate(True)
        pod.ensure()
        if job.get("cancel_requested"):
            raise wan_client.GenerationCancelled("generation cancelled by user")
        attempt["pod_ready"] = now_iso()
        attempt["pod"] = pod.id
        attempt["hourly_cost"] = pod.hourly_cost
        job["pod"] = pod.id       # ensure() may have self-healed to a new pod id
        job["status"] = "generating"
        save_jobs(job_id)
        last_activity = time.time()

        if job.pop("resume_provider_prompt", False) and job.get("provider_prompt_id"):
            outputs = wan_client.wait_for_result(
                pod.server, job["provider_prompt_id"],
                should_cancel=lambda: bool(job.get("cancel_requested")),
            )
        else:
            wf = json.loads(Path(job["workflow"]).read_text())
            image_name = wan_client.upload_image(pod.server, job["image_path"])
            if job.get("cancel_requested"):
                raise wan_client.GenerationCancelled("generation cancelled by user")
            wf = wan_client.patch_workflow(wf, image_name, job["prompt"], job["seed"],
                                           frames=job.get("frames"))
            client_id = uuid.uuid4().hex
            resp = wan_client.http_json(
                f"{pod.server}/prompt", {"prompt": wf, "client_id": client_id}
            )
            job["client_id"] = client_id
            job["provider_prompt_id"] = resp["prompt_id"]
            save_jobs(job_id)
            outputs = wan_client.wait_for_result(
                pod.server, resp["prompt_id"],
                should_cancel=lambda: bool(job.get("cancel_requested")),
            )
        # quality+frames in the name: A/B settings must never collide.
        stem = (f"{Path(job['image_path']).stem}_{job['seed']}"
                f"_{job.get('quality', 'final')}_{job.get('frames', 81)}f")
        saved = wan_client.download_outputs(pod.server, outputs, str(OUT_DIR / stem))
        if not saved:
            raise wan_client.GenerationError(
                "generation finished but no output file returned"
            )
        # A watchdog can fail this job while its provider request unwinds. A late
        # response must not resurrect a terminal job.
        if job.get("terminal"):
            _finish_attempt(job, attempt, "cancelled", job.get("error"))
            return
        _publish_output(job_id, job, attempt, saved[0])
        last_job_done = time.time()
        log(f"job {job_id} done and stored")
    except wan_client.GenerationCancelled as e:
        job["status"] = "cancelled"
        job["terminal"] = True
        job["cancelled"] = now_iso()
        job.pop("cancel_requested", None)
        _finish_attempt(job, attempt, "cancelled", e)
        log(f"job {job_id} cancelled")
    except wan_client.CancellationError as e:
        stopped = False
        if API_KEY:
            try:
                stopped = pod.stop()
            except Exception as stop_error:
                log(f"CRITICAL: cancellation failed and pod stop was not confirmed: {stop_error}")
        job["terminal"] = True
        job.pop("cancel_requested", None)
        if stopped:
            job["status"] = "cancelled"
            job["cancelled"] = now_iso()
            _finish_attempt(job, attempt, "cancelled", "provider stopped after cancellation failure")
        else:
            job["status"] = "error"
            job["error_code"] = "cancellation_error"
            job["error"] = "provider cancellation could not be confirmed"
            _finish_attempt(job, attempt, "cancellation_error", e)
    except wan_client.GenerationError as e:
        if job.get("terminal"):
            _finish_attempt(job, attempt, "cancelled", job.get("error"))
            return
        job["status"] = "error"
        job["error_code"] = "generation_error"
        job["error"] = str(e)[:300]
        _finish_attempt(job, attempt, "generation_error", e)
        log(f"job {job_id} rejected without retry: {job['error']}")
    except storage.StorageError as e:
        if job.get("terminal"):
            _finish_attempt(job, attempt, "cancelled", job.get("error"))
            return
        err = str(e)[:300]
        _finish_attempt(job, attempt, "storage_error", e)
        if job["attempt_count"] < JOB_RETRIES and job.get("pending_output"):
            job["status"] = "queued"
            log(f"job {job_id} storage failure ({err}) — retrying without regeneration")
            save_jobs(job_id)
            time.sleep(30)
            job_queue.put(job_id)
        else:
            job["status"] = "error"
            job["error_code"] = "storage_error"
            job["error"] = f"could not store result after {JOB_RETRIES} attempts: {err}"
    except (wan_client.InfrastructureError, RuntimeError, OSError,
            urllib.error.URLError) as e:
        if job.get("terminal"):
            _finish_attempt(job, attempt, "cancelled", job.get("error"))
            return
        pod.cooldown_until = time.time() + 900
        try:
            pod.reconcile()
        except Exception as reconcile_error:
            log(f"could not reconcile pod {pod.id}: {reconcile_error}")
        err = str(e)[:300]
        _finish_attempt(job, attempt, "infrastructure_error", e)
        if job["attempt_count"] < JOB_RETRIES:
            log(f"job {job_id} infra failure ({err}) — requeue "
                f"{job['attempt_count']}/{JOB_RETRIES} in 30s")
            job["status"] = "queued"
            save_jobs(job_id)
            time.sleep(30)
            job_queue.put(job_id)
        else:
            job["status"] = "error"
            job["error_code"] = "infrastructure_error"
            job["error"] = f"gave up after {JOB_RETRIES} attempts: {err}"
            log(f"job {job_id} failed permanently: {err}")
    except Exception as e:
        if job.get("terminal"):
            _finish_attempt(job, attempt, "cancelled", job.get("error"))
            return
        job["status"] = "error"
        job["error_code"] = "internal_error"
        job["error"] = f"{type(e).__name__}: {str(e)[:260]}"
        _finish_attempt(job, attempt, "internal_error", e)
        log(f"job {job_id} failed: {job['error']}")
    finally:
        last_activity = time.time()
        save_jobs(job_id)
        if not any_active():
            caffeinate(False)
        job_queue.task_done()


MAX_PODS = int(os.environ.get("WAN_MAX_PODS", "3"))
SCALE_THRESHOLD = int(os.environ.get("WAN_SCALE_THRESHOLD", "6"))
_scale_cooldown = 0.0


def autoscaler():
    """Scale OUT when the queue is deep (burst = fal-class wall time), scale IN
    by TERMINATING autoscaled pods once the queue drains (models live on the
    shared volume, so a fresh pod is ~90s away — corpses are worthless)."""
    global _scale_cooldown
    while True:
        time.sleep(30)
        try:
            depth = job_queue.qsize()
            if (API_KEY and depth > SCALE_THRESHOLD and len(PODS) < MAX_PODS
                    and time.time() > _scale_cooldown):
                _scale_cooldown = time.time() + 300   # one scale event per 5 min
                name = os.environ.get("RUNPOD_POD_NAME", "wan22-video-gen") + "-scale"
                log(f"autoscale OUT: {depth} queued > {SCALE_THRESHOLD} — creating pod "
                    f"({len(PODS)}/{MAX_PODS} in pool)")
                new_id = None
                try:
                    new_id = run_batch.create_pod(API_KEY, name)
                    pod = Pod(new_id, autoscaled=True)
                    pod.start_worker()
                    PODS.append(pod)
                    log("autoscale OUT: a worker joined the pool")
                except Exception as e:
                    if new_id and not any(p.id == new_id for p in PODS):
                        try:
                            run_batch.terminate_pod(new_id, API_KEY)
                        except Exception as cleanup_error:
                            log(f"CRITICAL: could not clean up failed scale-out: {cleanup_error}")
                    log(f"autoscale OUT failed ({e}) — retrying after cooldown")
            if not any_active():
                for pod in [p for p in PODS if getattr(p, "autoscaled", False)]:
                    log("autoscale IN: queue drained — retiring a burst worker")
                    try:
                        pod.retire(terminate=True)
                        PODS.remove(pod)
                    except Exception as e:
                        log(f"CRITICAL: autoscale IN could not terminate worker: {e}")
        except Exception as e:
            log(f"autoscaler error: {e}")


def idle_watcher():
    global last_job_done
    while True:
        time.sleep(60)
        try:
            if not any_active() and time.time() - last_activity > IDLE_STOP_MIN * 60:
                for pod in PODS:
                    if pod.running:
                        log(f"idle {IDLE_STOP_MIN:.0f} min — stopping pod {pod.id}")
                        pod.stop()
            # MONEY WATCHDOG — the failsafe behind every other failsafe. If any pod
            # has been running for WATCHDOG_MIN minutes without a single completed
            # job, something is wedged (frozen worker, unfixable pod, hung ComfyUI).
            # A wedged system must fail toward $0/hr: stop everything, fail the
            # jobs loudly. (2026-08-13: absence of this guard cost $7 overnight.)
            if any(p.running for p in PODS) and \
                    time.time() - last_job_done > WATCHDOG_MIN * 60:
                log(f"WATCHDOG: no job completed in {WATCHDOG_MIN:.0f} min with pods "
                    f"running — stopping ALL pods and failing active jobs")
                for pod in PODS:
                    try:
                        pod.stop()
                        pod.cooldown_until = time.time() + 3600
                    except Exception as e:
                        log(f"CRITICAL: WATCHDOG could not confirm worker stopped: {e}")
                for j in jobs.values():
                    if j["status"] in ("queued", "starting_pod", "generating"):
                        j["status"] = "error"
                        j["terminal"] = True
                        j["error_code"] = "watchdog"
                        j["error"] = (f"killed by money watchdog: no completion in "
                                      f"{WATCHDOG_MIN:.0f} min — system was wedged")
                while not job_queue.empty():
                    try:
                        job_queue.get_nowait(); job_queue.task_done()
                    except queue.Empty:
                        break
                save_jobs()
                last_job_done = time.time()   # arm for the next cycle, don't refire
        except Exception as e:
            log(f"idle watcher error: {e}")


class Handler(BaseHTTPRequestHandler):
    request_id = None
    principal = None

    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin and origin in CORS_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Idempotency-Key, X-Request-ID")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def _json(self, code, payload):
        body = b"" if code == 204 else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.request_id:
            self.send_header("X-Request-ID", self.request_id)
        self._cors()
        self.end_headers()
        if code != 204:
            self.wfile.write(body)

    def _error(self, code, error_code, message, details=None):
        error = {"code": error_code, "message": message}
        if details is not None:
            error["details"] = details
        self._json(code, {"error": error, "request_id": self.request_id})

    def _prepare(self):
        candidate = self.headers.get("X-Request-ID", "").strip()
        if not candidate or len(candidate) > 128 or not all(
            character.isalnum() or character in "._-" for character in candidate
        ):
            candidate = uuid.uuid4().hex
        self.request_id = candidate

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length < 0:
            raise ValueError
        if length > MAX_UPLOAD_MB * 1024 * 1024:
            raise OverflowError
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError
        return payload

    def log_message(self, fmt, *args):
        pass

    def _authed(self):
        """Resolve a bearer token to a tenant principal."""
        path = urllib.parse.urlsplit(self.path).path
        if path == "/health":
            self.principal = product_api.Principal("public", "user", "default")
            return True
        authorization = self.headers.get("Authorization", "")
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        self.principal = product_api.authenticate(
            token, API_TOKEN, API_KEYS,
            allow_anonymous_local=(
                is_loopback_bind(BIND) and is_loopback(self.client_address[0])
            ),
        )
        if self.principal:
            return True
        if path.startswith("/v1/"):
            self._error(401, "unauthorized", "missing or invalid bearer token")
        else:
            self._json(401, {"error": "missing or bad Authorization bearer token"})
        return False

    def _find_generation(self, job_id):
        job = jobs.get(job_id)
        if not job or not product_api.owns(self.principal, job):
            self._error(404, "not_found", "generation was not found")
            return None
        return job

    def _send_result(self, job, v1=False):
        if job["status"] != "done":
            if v1:
                self._error(409, "not_ready", f"generation is {job['status']}")
            else:
                self._json(404, {"error": f"job not finished (status: {job['status']})"})
            return
        if job.get("storage_backend") == "s3":
            try:
                if VIDEO_STORAGE.backend != "s3":
                    raise storage.StorageError("server is not configured for the job's S3 storage")
                location = VIDEO_STORAGE.download_url(job, SIGNED_URL_SECONDS)
            except storage.StorageError as exc:
                if v1:
                    self._error(503, "storage_unavailable", str(exc))
                else:
                    self._json(503, {"error": str(exc)})
                return
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Request-ID", self.request_id)
            self._cors()
            self.end_headers()
            return
        output = Path(job.get("output", ""))
        if not output.is_file():
            if v1:
                self._error(410, "result_unavailable", "stored result is no longer available")
            else:
                self._json(410, {"error": "stored result is no longer available"})
            return
        self.send_response(200)
        self.send_header("Content-Type", job.get("output_content_type", "video/mp4"))
        self.send_header("Content-Length", str(output.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{job.get("output_filename", "video.mp4")}"')
        self.send_header("X-Request-ID", self.request_id)
        self._cors()
        self.end_headers()
        with output.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_GET(self):
        self._prepare()
        if not self._authed():
            return
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        parts = path.strip("/").split("/")
        if path == "/health":
            self._json(200, health_payload())
        elif path == "/jobs":
            with jobs_lock:
                payload = {
                    jid: public_job(job) for jid, job in jobs.items()
                    if product_api.owns(self.principal, job)
                }
            self._json(200, payload)
        elif path == "/v1/models":
            with jobs_lock:
                catalog = product_api.model_catalog(MODEL_REGISTRY, dict(jobs))
            self._json(200, {"object": "list", "data": catalog})
        elif path == "/v1/openapi.yaml":
            body = Path(__file__).with_name("openapi.yaml").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/yaml")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-ID", self.request_id)
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        elif path == "/v1/generations":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", [20])[0])
                if not 1 <= limit <= 100:
                    raise ValueError("limit must be between 1 and 100")
                status = query.get("status", [None])[0]
                if status and status not in ("queued", "starting_pod", "generating", "uploading", "done", "error", "cancelled"):
                    raise ValueError("status is invalid")
                model_id = query.get("model", [None])[0]
                if model_id and model_id not in MODEL_REGISTRY:
                    raise ValueError("model is invalid")
                with jobs_lock:
                    page, next_cursor = product_api.paginate(
                        dict(jobs), self.principal, limit,
                        query.get("cursor", [None])[0], status, model_id
                    )
            except ValueError as exc:
                self._error(400, "invalid_request", str(exc))
                return
            self._json(200, {
                "object": "list",
                "data": [product_api.generation_view(jid, job) for jid, job in page],
                "has_more": next_cursor is not None,
                "next_cursor": next_cursor,
            })
        elif len(parts) == 3 and parts[:2] == ["v1", "generations"]:
            job = self._find_generation(parts[2])
            if job:
                self._json(200, product_api.generation_view(parts[2], job))
        elif len(parts) == 4 and parts[:2] == ["v1", "generations"] and parts[3] == "result":
            job = self._find_generation(parts[2])
            if job:
                self._send_result(job, v1=True)
        elif len(parts) == 2 and parts[0] == "status":
            job = jobs.get(parts[1])
            if not job or not product_api.owns(self.principal, job):
                self._json(404, {"error": "unknown job id"})
                return
            self._json(200, public_job(job))
        elif len(parts) == 2 and parts[0] == "result":
            job = jobs.get(parts[1])
            if not job or not product_api.owns(self.principal, job):
                self._json(404, {"error": "unknown job id"})
            else:
                self._send_result(job)
        else:
            if path.startswith("/v1/"):
                self._error(404, "not_found", "route was not found")
            else:
                self._json(404, {"error": "not found"})

    def do_POST(self):
        global last_activity
        self._prepare()
        if not self._authed():
            return
        path = urllib.parse.urlsplit(self.path).path
        parts = path.strip("/").split("/")
        if path not in ("/generate", "/v1/generations") and not (
            len(parts) == 4 and parts[:2] == ["v1", "generations"]
            and parts[3] in ("cancel", "retry")
        ):
            if path.startswith("/v1/"):
                self._error(404, "not_found", "route was not found")
            else:
                self._json(404, {"error": "not found"})
            return
        if len(parts) == 4:
            action = cancel_generation if parts[3] == "cancel" else retry_generation
            status, error_code, result = action(parts[2], self.principal)
            if error_code:
                self._error(status, error_code, result)
            else:
                self._json(status, product_api.generation_view(parts[2], result))
            return
        try:
            req = self._body()
        except OverflowError:
            if path.startswith("/v1/"):
                self._error(413, "request_too_large", f"body over {MAX_UPLOAD_MB}MB limit")
            else:
                self._json(413, {"error": f"body over {MAX_UPLOAD_MB}MB limit"})
            return
        except Exception:
            if path.startswith("/v1/"):
                self._error(400, "invalid_json", "body must be a JSON object")
            else:
                self._json(400, {"error": "body must be JSON"})
            return

        status, payload = submit_generation(
            req,
            idempotency_key=self.headers.get("Idempotency-Key", ""),
            allow_local_path=(
                is_loopback_bind(BIND) and
                is_loopback(self.client_address[0])
            ),
            principal=self.principal,
            strict=path.startswith("/v1/"),
        )
        if path == "/generate":
            self._json(status, payload)
        elif status >= 400:
            error_code = {
                409: "idempotency_conflict",
                413: "request_too_large",
                503: "service_unavailable",
            }.get(status, "invalid_request")
            self._error(status, error_code, payload["error"])
        else:
            job_id = payload["job_id"]
            response = product_api.generation_view(job_id, jobs[job_id])
            response["duplicate"] = payload["duplicate"]
            self._json(status, response)

    def do_DELETE(self):
        self._prepare()
        if not self._authed():
            return
        parts = urllib.parse.urlsplit(self.path).path.strip("/").split("/")
        if len(parts) != 3 or parts[:2] != ["v1", "generations"]:
            self._error(404, "not_found", "route was not found")
            return
        status, error_code, message = delete_generation(parts[2], self.principal)
        if error_code:
            self._error(status, error_code, message)
        else:
            self._json(204, {})

    def do_OPTIONS(self):
        self._prepare()
        origin = self.headers.get("Origin", "")
        if origin not in CORS_ORIGINS:
            self._error(403, "origin_not_allowed", "origin is not allowed")
            return
        self.send_response(204)
        self.send_header("X-Request-ID", self.request_id)
        self._cors()
        self.end_headers()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    if not is_loopback_bind(BIND) and not (API_TOKEN or API_KEYS):
        sys.exit("error: refusing to bind beyond localhost without WAN_API_TOKEN or "
                 "WAN_API_KEYS_JSON set — configure auth, or keep WAN_BIND=127.0.0.1")
    try:
        load_jobs()
    except RuntimeError as e:
        sys.exit(f"error: {e}")
    pod_ids = []
    if os.environ.get("RUNPOD_POD_ID"):
        pod_ids = [os.environ["RUNPOD_POD_ID"]]
    elif API_KEY:
        pod_ids = run_batch.resolve_pod_ids(API_KEY)
    if not pod_ids:
        sys.exit("error: set RUNPOD_POD_ID, or RUNPOD_API_KEY + RUNPOD_POD_NAME "
                 "(no pods matched)")
    if not API_KEY:
        log("WARNING: RUNPOD_API_KEY not set — pod auto start/stop disabled")
    for pid in pod_ids:
        pod = Pod(pid)
        if API_KEY:
            try:
                pod.reconcile()
            except Exception as e:
                log(f"worker state reconciliation failed: {e}")
        PODS.append(pod)
        pod.start_worker()
    threading.Thread(target=idle_watcher, daemon=True).start()
    threading.Thread(target=autoscaler, daemon=True).start()
    log(f"starting pool with {len(pod_ids)} worker(s); idle-stop {IDLE_STOP_MIN:.0f} min")
    if API_TOKEN or API_KEYS:
        log("bearer-token auth ENABLED (all endpoints except /health)")
    log(f"binding {BIND}:{PORT}")
    server = ThreadingHTTPServer((BIND, PORT), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        JOB_REPOSITORY.close()


if __name__ == "__main__":
    main()
