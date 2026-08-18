#!/usr/bin/env python3
"""Start the Runpod pod, run a batch of Wan 2.2 I2V jobs, stop the pod.

The pod is ALWAYS stopped afterwards — on success, on error, and on Ctrl-C —
so idle GPU billing can't happen. Your network volume (models, workflow)
persists across stop/start.

One-time setup:
  export RUNPOD_API_KEY=...   # runpod.io -> Settings -> API Keys
  export RUNPOD_POD_ID=...    # the pod's id from the Runpod console

Usage:
  python run_batch.py --batch jobs.csv
  python run_batch.py --image shot01.png --prompt "slow dolly-in" --seeds 3
  python run_batch.py --batch jobs.csv --keep-running   # skip the auto-stop

All other flags (--workflow, --out, --seeds) are passed through to wan_client.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import wan_client

API = "https://rest.runpod.io/v1"


def rp(method, path, api_key):
    req = urllib.request.Request(
        f"{API}{path}", method=method,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        return json.loads(body) if body else {}


def resolve_pod_ids(api_key, pod_name=None):
    """ALL pod ids whose name starts with pod_name (or RUNPOD_POD_NAME).
    Multiple matches = a deliberate worker pool (e.g. a 4090 + a 5090)."""
    pod_name = pod_name or os.environ.get("RUNPOD_POD_NAME")
    if not pod_name:
        return []
    pods = rp("GET", "/pods", api_key)
    return [p["id"] for p in pods if p.get("name", "").startswith(pod_name)]


def resolve_pod_id(api_key, pod_id=None, pod_name=None):
    """Return a pod id. If pod_name is given (or RUNPOD_POD_NAME), look the pod
    up by name — survives Runpod migrations, which change the id."""
    if pod_id:
        return pod_id
    pod_name = pod_name or os.environ.get("RUNPOD_POD_NAME")
    if not pod_name:
        return None
    # Prefix match: Runpod appends "-migration" to the name when a pod is
    # auto-migrated to a new host, so exact names go stale.
    pods = rp("GET", "/pods", api_key)
    matches = [p for p in pods if p.get("name", "").startswith(pod_name)]
    if not matches:
        sys.exit(f"error: no pod named {pod_name!r}* found on this account")
    if len(matches) > 1:
        names = [f"{p['name']} ({p['id']})" for p in matches]
        sys.exit(f"error: multiple pods match {pod_name!r}*: {names} — "
                 f"terminate duplicates or pass --pod-id explicitly")
    pod_id = matches[0]["id"]
    print(f"resolved pod {pod_name!r} -> {pod_id}")
    return pod_id


POD_TEMPLATE = {
    "imageName": "hearmeman/comfyui-wan-template:v26",
    "gpuTypeIds": ["NVIDIA GeForce RTX 5090"],
    "gpuCount": 1,
    "networkVolumeId": "4tcigizrep",
    "containerDiskInGb": 50,
    "dataCenterIds": ["EU-RO-1"],
    "cloudType": "SECURE",
    "ports": ["8188/http", "8888/http"],
    "env": {"download_wan22": "true"},
    "minRAMPerGPU": 40,
}


def create_pod(api_key, name):
    """Create a FRESH pod on the shared volume. In EU-RO-1 a stopped pod loses its
    GPU within the hour and start returns HTTP 500 forever; creating a new pod finds
    any host with a free GPU instead of begging one specific host."""
    body = json.dumps(dict(POD_TEMPLATE, name=name)).encode()
    req = urllib.request.Request(
        f"{API}/pods", data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        pod = json.loads(r.read())
    print(f"created pod {pod['id']} ({name}) at ${pod.get('costPerHr', '?')}/hr")
    return pod["id"]


def terminate_pod(pod_id, api_key):
    try:
        rp("DELETE", f"/pods/{pod_id}", api_key)
        print(f"terminated pod {pod_id}")
    except Exception as e:
        print(f"WARNING: could not terminate {pod_id}: {e}", file=sys.stderr)


def start_pod(pod_id, api_key):
    """Start a pod, retrying transient Runpod API errors. The start endpoint
    returns 5xx under host contention (observed 10x in a row 2026-08-13); one
    eventually succeeds, so failing on the first is throwing away good jobs."""
    pod = rp("GET", f"/pods/{pod_id}", api_key)
    if pod.get("desiredStatus") == "RUNNING":
        print(f"pod {pod_id} already running")
        return
    print(f"starting pod {pod_id}...")
    last_err = None
    for attempt in range(6):
        if attempt:
            wait = min(15 * (2 ** (attempt - 1)), 120)
            print(f"start attempt {attempt} failed ({last_err}); retrying in {wait}s...")
            time.sleep(wait)
        try:
            rp("POST", f"/pods/{pod_id}/start", api_key)
            break
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code < 500:          # 4xx won't heal by retrying
                raise
        except Exception as e:        # timeouts, connection resets
            last_err = type(e).__name__
    else:
        raise RuntimeError(f"pod start kept failing after 6 attempts ({last_err})")
    for _ in range(60):
        time.sleep(5)
        pod = rp("GET", f"/pods/{pod_id}", api_key)
        if pod.get("desiredStatus") == "RUNNING":
            print("pod is running")
            return
    raise RuntimeError("pod did not reach RUNNING within 5 minutes")


def stop_pod(pod_id, api_key):
    print(f"stopping pod {pod_id}...")
    for attempt in range(3):
        try:
            rp("POST", f"/pods/{pod_id}/stop", api_key)
            print("pod stopped — no further GPU billing")
            return
        except Exception as e:
            print(f"stop attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(10)
    print(f"WARNING: could not stop pod {pod_id} — stop it manually in the "
          f"Runpod console NOW to avoid idle billing!", file=sys.stderr)


def wait_for_comfyui(server, timeout=600):
    """ComfyUI takes a while to boot after pod start; poll until it answers."""
    print(f"waiting for ComfyUI at {server} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{server}/system_stats", timeout=10):
                print("ComfyUI is up")
                return
        except Exception:
            time.sleep(10)
    raise RuntimeError(f"ComfyUI did not come up at {server} within {timeout}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID"))
    ap.add_argument("--workflow", default=str(os.path.join(os.path.dirname(__file__), "workflow_api.json")))
    ap.add_argument("--draft", action="store_true",
                    help="fast preview: 720p/16fps, no upscale or interpolation (workflow_draft.json)")
    ap.add_argument("--image")
    ap.add_argument("--prompt")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch", help="CSV with columns: image,prompt[,seeds]")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--keep-running", action="store_true",
                    help="do not stop the pod when the batch finishes")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("error: set RUNPOD_API_KEY (runpod.io -> Settings -> API Keys)")
    args.pod_id = resolve_pod_id(api_key, args.pod_id)
    if not args.pod_id:
        sys.exit("error: set RUNPOD_POD_ID / RUNPOD_POD_NAME or pass --pod-id")
    if args.draft:
        args.workflow = str(os.path.join(os.path.dirname(__file__), "workflow_draft.json"))
    if not os.path.exists(args.workflow):
        sys.exit(f"error: {args.workflow} not found — export it from ComfyUI with 'Export (API)'")

    jobs = []
    if args.batch:
        import csv
        with open(args.batch) as f:
            for row in csv.DictReader(f):
                jobs.append((row["image"], row["prompt"], int(row.get("seeds") or args.seeds)))
    elif args.image and args.prompt:
        jobs.append((args.image, args.prompt, args.seeds))
    else:
        sys.exit("error: pass --image and --prompt, or --batch jobs.csv")
    missing = [img for img, _, _ in jobs if not os.path.exists(img)]
    if missing:
        sys.exit(f"error: input image(s) not found: {missing}")

    server = f"https://{args.pod_id}-8188.proxy.runpod.net"
    start_pod(args.pod_id, api_key)
    t0 = time.time()
    try:
        wait_for_comfyui(server)
        for image, prompt, seeds in jobs:
            wan_client.run_job(server, args.workflow, image, prompt, seeds, args.out)
        print(f"batch done: {len(jobs)} job(s) in {int((time.time() - t0) / 60)} min")
    finally:
        if args.keep_running:
            print("--keep-running set: pod left running (billing continues!)")
        else:
            stop_pod(args.pod_id, api_key)


if __name__ == "__main__":
    main()
