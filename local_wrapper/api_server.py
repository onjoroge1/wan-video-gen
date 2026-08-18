#!/usr/bin/env python3
"""fal.ai-style local API for self-hosted Wan 2.2 pods on Runpod.

Production behavior:
  - Worker POOL: every Runpod pod whose name starts with RUNPOD_POD_NAME gets a
    worker thread; jobs are dispatched to whichever pod is free (deploy a second
    pod with the same name prefix and throughput doubles, no config change).
  - Pods auto-start on demand (with retry/backoff through Runpod 5xx blips) and
    auto-stop after WAN_IDLE_STOP_MIN minutes with no work (default 45).
  - Jobs REQUEUE on infrastructure failures (pod start/boot trouble) up to
    WAN_JOB_RETRIES times before erroring; generation errors don't requeue.
  - Job state persists to outputs/jobs.json — a server restart re-queues
    whatever was in flight instead of orphaning it.
  - While jobs are active, macOS sleep is held off via `caffeinate`.

Setup (either):
  RUNPOD_API_KEY=... RUNPOD_POD_NAME=wan22-video-gen python3 api_server.py
  RUNPOD_API_KEY=... RUNPOD_POD_ID=<id>              python3 api_server.py

Endpoints:
  POST /generate   {"image_path": "/abs/path.jpg", "prompt": "...",
                    "draft": true, "seed": 123}       # draft+seed optional
  GET  /status/<job_id> | /result/<job_id> | /jobs | /health
"""

import base64
import json
import os

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
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import run_batch
import wan_client

PORT = int(os.environ.get("WAN_API_PORT", "8787"))
BIND = os.environ.get("WAN_BIND", "127.0.0.1")
API_TOKEN = os.environ.get("WAN_API_TOKEN", "")
MAX_UPLOAD_MB = int(os.environ.get("WAN_MAX_UPLOAD_MB", "25"))
IDLE_STOP_MIN = float(os.environ.get("WAN_IDLE_STOP_MIN", "45"))
JOB_RETRIES = int(os.environ.get("WAN_JOB_RETRIES", "3"))
OUT_DIR = Path(__file__).parent / "outputs"
JOBS_FILE = OUT_DIR / "jobs.json"
WORKFLOWS = {
    "draft": Path(__file__).parent / "workflow_draft.json",   # 4-step distill, 720p16, fastest
    "final": Path(__file__).parent / "workflow_api.json",     # 6-step distill + upscale + RIFE
    "hq":    Path(__file__).parent / "workflow_hq.json",      # 20-step cfg 3.5, no distill — realism
}

def _load_dotenv():
    """Pick up Runpod credentials from REELFORGE's .env when not already set,
    so launchd can run this server directly (no shell wrapper — the .env has
    lines that break shell `source`, and TCC-wise only python needs access)."""
    env_file = Path(__file__).parent / ".env"          # local copy (outside TCC scope)
    if not env_file.exists():
        env_file = Path.home() / "Documents" / "video" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            if k.strip().startswith("RUNPOD_") and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()


_load_dotenv()
API_KEY = os.environ.get("RUNPOD_API_KEY")

jobs = {}                 # job_id -> dict
job_queue = queue.Queue()
jobs_lock = threading.Lock()
last_activity = time.time()
last_job_done = time.time()   # money watchdog: last time ANY job completed
WATCHDOG_MIN = float(os.environ.get("WAN_WATCHDOG_MIN", "60"))
caffeinate_proc = None


def log(msg):
    print(f"[api] {msg}", flush=True)


def save_jobs():
    # Persistence is best-effort: a failed write must NEVER kill a worker thread.
    # (2026-08-13: an EDEADLK here froze a worker mid-job; its jobs stayed
    # "active" forever, which disabled idle-stop and burned $7 of pod time.)
    try:
        with jobs_lock:
            tmp = JOBS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(jobs, indent=1))
            tmp.replace(JOBS_FILE)
    except Exception as e:
        log(f"save_jobs failed (non-fatal): {e}")


def load_jobs():
    """Reload persisted jobs; anything that was queued/in-flight goes back on
    the queue so a server restart never orphans work."""
    if not JOBS_FILE.exists():
        return
    try:
        jobs.update(json.loads(JOBS_FILE.read_text()))
    except Exception as e:
        log(f"could not load {JOBS_FILE}: {e}")
        return
    requeued = 0
    for jid, j in jobs.items():
        if j["status"] in ("queued", "starting_pod", "generating"):
            if os.path.isfile(j["image_path"]):
                j["status"] = "queued"
                job_queue.put(jid)
                requeued += 1
            else:
                j["status"] = "error"
                j["error"] = "input image vanished during server restart"
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

    def __init__(self, pod_id):
        self.id = pod_id
        self.server = f"https://{pod_id}-8188.proxy.runpod.net"
        self.running = False
        self.cooldown_until = 0.0   # dead pods sit out instead of stealing jobs

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
            run_batch.start_pod(self.id, API_KEY)      # retries 5xx internally
        except RuntimeError as e:
            # Start exhausted its retries: this host's GPU is gone (EU-RO-1 reclaims
            # stopped pods within the hour). SELF-HEAL: fresh pod on the shared
            # volume, terminate the corpse, take over its identity in the pool.
            log(f"pod {self.id} unstartable ({e}) — self-healing with a fresh pod")
            old = self.id
            new_id = run_batch.create_pod(API_KEY, os.environ.get(
                "RUNPOD_POD_NAME", "wan22-video-gen") + "-5090")
            run_batch.terminate_pod(old, API_KEY)
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
        if not self.running or not API_KEY:
            return
        run_batch.stop_pod(self.id, API_KEY)
        self.running = False


PODS = []


def any_active():
    return any(j["status"] in ("queued", "starting_pod", "generating") for j in jobs.values())


def worker(pod):
    global last_activity
    while True:
        try:
            _worker_one(pod)
        except Exception as e:
            # A worker thread must be unkillable: any escaped exception here would
            # otherwise freeze its job as "active" forever, which paralyzes
            # idle-stop and turns a bug into an unbounded GPU bill.
            log(f"worker {pod.id}: escaped exception (recovered): {type(e).__name__}: {e}")
            time.sleep(10)


def _worker_one(pod):
    global last_activity
    if True:
        job_id = job_queue.get()
        # A pod that just failed to start has no GPU right now (reclaimed host).
        # Hand the job straight back for a healthy worker and sit out the cooldown —
        # otherwise a dead pod keeps grabbing jobs and burning their retry budget.
        if time.time() < pod.cooldown_until:
            job_queue.put(job_id)
            job_queue.task_done()
            time.sleep(30)
            return
        job = jobs[job_id]
        job["pod"] = pod.id
        try:
            job["status"] = "starting_pod"
            save_jobs()
            caffeinate(True)
            pod.ensure()
            job["pod"] = pod.id       # re-stamp: ensure() may have self-healed to a new pod id
            job["status"] = "generating"
            save_jobs()
            last_activity = time.time()

            wf = json.loads(Path(job["workflow"]).read_text())
            image_name = wan_client.upload_image(pod.server, job["image_path"])
            wf = wan_client.patch_workflow(wf, image_name, job["prompt"], job["seed"],
                                           frames=job.get("frames"))
            resp = wan_client.http_json(f"{pod.server}/prompt",
                                        {"prompt": wf, "client_id": uuid.uuid4().hex})
            outputs = wan_client.wait_for_result(pod.server, resp["prompt_id"])
            # quality+frames in the name: same image+seed at different settings
            # must never collide (an A/B test once produced 3 identical "variants")
            stem = (f"{Path(job['image_path']).stem}_{job['seed']}"
                    f"_{job.get('quality', 'final')}_{job.get('frames', 81)}f")
            saved = wan_client.download_outputs(pod.server, outputs, str(OUT_DIR / stem))
            if not saved:
                raise RuntimeError("generation finished but no output file returned")
            job["output"] = str(saved[0])
            job["status"] = "done"
            global last_job_done
            last_job_done = time.time()
            log(f"job {job_id} done on {pod.id} -> {saved[0]}")
        except (RuntimeError, SystemExit, OSError, urllib.error.URLError) as e:
            # Infrastructure trouble (pod start, boot, network): the job itself
            # is fine — requeue it unless it's out of attempts.
            pod.running = False
            pod.cooldown_until = time.time() + 900   # 15 min bench after infra failure
            job["attempts"] = job.get("attempts", 0) + 1
            err = str(e)[:140]
            if job["attempts"] < JOB_RETRIES:
                log(f"job {job_id} infra failure ({err}) — requeue "
                    f"{job['attempts']}/{JOB_RETRIES} in 30s")
                job["status"] = "queued"
                save_jobs()
                time.sleep(30)
                job_queue.put(job_id)
            else:
                job["status"] = "error"
                job["error"] = f"gave up after {JOB_RETRIES} attempts: {err}"
                log(f"job {job_id} failed permanently: {err}")
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {str(e)[:140]}"
            log(f"job {job_id} failed: {job['error']}")
        finally:
            last_activity = time.time()
            save_jobs()
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
            healthy = [p for p in PODS if time.time() >= p.cooldown_until]
            if (API_KEY and depth > SCALE_THRESHOLD and len(PODS) < MAX_PODS
                    and time.time() > _scale_cooldown):
                _scale_cooldown = time.time() + 300   # one scale event per 5 min
                name = os.environ.get("RUNPOD_POD_NAME", "wan22-video-gen") + "-scale"
                log(f"autoscale OUT: {depth} queued > {SCALE_THRESHOLD} — creating pod "
                    f"({len(PODS)}/{MAX_PODS} in pool)")
                try:
                    new_id = run_batch.create_pod(API_KEY, name)
                    pod = Pod(new_id)
                    pod.autoscaled = True
                    PODS.append(pod)
                    threading.Thread(target=worker, args=(pod,), daemon=True).start()
                    log(f"autoscale OUT: pod {new_id} joined the pool")
                except Exception as e:
                    log(f"autoscale OUT failed ({e}) — retrying after cooldown")
            if not any_active():
                for pod in [p for p in PODS if getattr(p, "autoscaled", False)]:
                    log(f"autoscale IN: queue drained — terminating {pod.id}")
                    try:
                        run_batch.terminate_pod(pod.id, API_KEY)
                    except Exception as e:
                        log(f"autoscale IN: terminate failed ({e}) — idle-stop will catch it")
                    PODS.remove(pod)
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
                        if API_KEY:
                            run_batch.stop_pod(pod.id, API_KEY)
                        pod.running = False
                        pod.cooldown_until = time.time() + 3600
                    except Exception as e:
                        log(f"WATCHDOG: stop {pod.id} failed: {e}")
                for j in jobs.values():
                    if j["status"] in ("queued", "starting_pod", "generating"):
                        j["status"] = "error"
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
    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass

    def _authed(self):
        """Bearer-token check. /health stays open (it leaks only pool status);
        everything else requires the token when one is configured."""
        if not API_TOKEN or self.path == "/health":
            return True
        if self.headers.get("Authorization", "") == f"Bearer {API_TOKEN}":
            return True
        self._json(401, {"error": "missing or bad Authorization bearer token"})
        return False

    def do_GET(self):
        if not self._authed():
            return
        parts = self.path.strip("/").split("/")
        if self.path == "/health":
            self._json(200, {"ok": True, "pods": {p.id: ("running" if p.running else "stopped")
                                                  for p in PODS},
                             "queued": job_queue.qsize()})
        elif self.path == "/jobs":
            self._json(200, {jid: {k: v for k, v in j.items() if k != "workflow"}
                             for jid, j in jobs.items()})
        elif len(parts) == 2 and parts[0] == "status":
            job = jobs.get(parts[1])
            if not job:
                self._json(404, {"error": "unknown job id"})
                return
            self._json(200, {k: v for k, v in job.items() if k != "workflow"})
        elif len(parts) == 2 and parts[0] == "result":
            job = jobs.get(parts[1])
            if not job:
                self._json(404, {"error": "unknown job id"})
            elif job["status"] != "done":
                self._json(404, {"error": f"job not finished (status: {job['status']})"})
            else:
                data = Path(job["output"]).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{Path(job["output"]).name}"')
                self.end_headers()
                self.wfile.write(data)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        global last_activity
        if not self._authed():
            return
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_UPLOAD_MB * 1024 * 1024:
                self._json(413, {"error": f"body over {MAX_UPLOAD_MB}MB limit"})
                return
            req = json.loads(self.rfile.read(length))
        except Exception:
            self._json(400, {"error": "body must be JSON"})
            return

        prompt = req.get("prompt", "")
        image_path = req.get("image_path", "")
        image_b64 = req.get("image_b64", "")
        if not prompt or not (image_path or image_b64):
            self._json(400, {"error": "prompt plus image_path or image_b64 required"})
            return
        if image_b64:
            # Remote callers can't reference this Mac's disk — accept the image
            # inline and persist it under outputs/uploads/.
            try:
                raw = base64.b64decode(image_b64, validate=True)
            except Exception:
                self._json(400, {"error": "image_b64 is not valid base64"})
                return
            ext = {"\xff\xd8": ".jpg", "\x89P": ".png"}.get(raw[:2].decode("latin1"), ".png")
            up_dir = OUT_DIR / "uploads"
            up_dir.mkdir(exist_ok=True)
            image_path = str(up_dir / f"{uuid.uuid4().hex[:12]}{ext}")
            Path(image_path).write_bytes(raw)
        elif not os.path.isfile(image_path):
            self._json(400, {"error": f"image not found: {image_path}"})
            return

        quality = req.get("quality") or ("draft" if req.get("draft") else "final")
        if quality not in WORKFLOWS:
            self._json(400, {"error": f"quality must be one of {sorted(WORKFLOWS)}"})
            return
        seconds = float(req.get("seconds") or 5)
        frames = max(33, min(121, round(seconds * 16) + 1))   # 2s..7.5s native @16fps

        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {
            "status": "queued",
            "image_path": image_path,
            "prompt": prompt,
            "seed": int(req.get("seed", random.randint(0, 2**48))),
            "draft": quality == "draft",
            "quality": quality,
            "frames": frames,
            "workflow": str(WORKFLOWS[quality]),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        last_activity = time.time()
        save_jobs()
        job_queue.put(job_id)
        self._json(202, {"job_id": job_id,
                         "status_url": f"/status/{job_id}",
                         "result_url": f"/result/{job_id}",
                         "seed": jobs[job_id]["seed"], "draft": jobs[job_id]["draft"]})


def main():
    OUT_DIR.mkdir(exist_ok=True)
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
        PODS.append(pod)
        threading.Thread(target=worker, args=(pod,), daemon=True).start()
    load_jobs()
    threading.Thread(target=idle_watcher, daemon=True).start()
    threading.Thread(target=autoscaler, daemon=True).start()
    log(f"listening on http://127.0.0.1:{PORT}  "
        f"(pool: {', '.join(pod_ids)}; idle-stop {IDLE_STOP_MIN:.0f} min)")
    if BIND != "127.0.0.1" and not API_TOKEN:
        sys.exit("error: refusing to bind beyond localhost without WAN_API_TOKEN set "
                 "— generate one and export it, or keep WAN_BIND=127.0.0.1")
    if API_TOKEN:
        log("bearer-token auth ENABLED (all endpoints except /health)")
    log(f"binding {BIND}:{PORT}")
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
