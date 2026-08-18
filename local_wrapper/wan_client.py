#!/usr/bin/env python3
"""Minimal client for a ComfyUI server running Wan 2.2 I2V (e.g. on a Runpod pod).

One-time setup:
  1. In ComfyUI, build/tune your Wan 2.2 I2V workflow (with the 4-step LoRAs).
  2. Export it via: Workflow menu -> Export (API) -> save as workflow_api.json
     next to this script.
  3. Set your server URL below or via --server / WAN_SERVER env var.
     For Runpod: https://<POD_ID>-8188.proxy.runpod.net

Usage:
  # single clip, 3 seed variants
  python wan_client.py --image shot01.png --prompt "slow dolly-in, ..." --seeds 3

  # batch: CSV with columns image,prompt[,seeds]
  python wan_client.py --batch jobs.csv

Outputs land in ./outputs/ named <image-stem>_<seed>.mp4
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_SERVER = os.environ.get("WAN_SERVER", "http://127.0.0.1:8188")

# Runpod's proxy blocks the default "Python-urllib" user agent with a 403,
# so send a custom one on every request (urlretrieve included).
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", "wan-client/1.0")]
urllib.request.install_opener(_opener)


def http_json(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def upload_image(server, path):
    """Upload an input image via /upload/image (multipart)."""
    boundary = uuid.uuid4().hex
    fname = Path(path).name
    body = b""
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fname}\"\r\n".encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += Path(path).read_bytes()
    body += f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode()
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{server}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["name"]


def patch_workflow(wf, image_name, prompt, seed, frames=None):
    """Patch LoadImage, positive prompt, sampler seeds — and optionally the clip
    length in frames (WanImageToVideo; 81 = 5s @16fps, up to 121 = 7.5s)."""
    patched = {"image": False, "prompt": False, "seed": False}
    for node in wf.values():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})
        title = node.get("_meta", {}).get("title", "").lower()

        if ctype == "LoadImage":
            inputs["image"] = image_name
            patched["image"] = True
        elif ctype == "WanImageToVideo" and frames:
            inputs["length"] = int(frames)
        elif ctype == "CLIPTextEncode" and "negative" not in title:
            inputs["text"] = prompt
            patched["prompt"] = True
        elif ctype in ("KSampler", "KSamplerAdvanced"):
            key = "seed" if "seed" in inputs else "noise_seed"
            inputs[key] = seed
            patched["seed"] = True

    missing = [k for k, ok in patched.items() if not ok]
    if missing:
        sys.exit(f"error: could not find node(s) to patch: {missing}. "
                 f"Check that the workflow was exported in API format and contains "
                 f"LoadImage / CLIPTextEncode / KSampler nodes.")
    return wf


def wait_for_result(server, prompt_id, poll=5, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        hist = http_json(f"{server}/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                sys.exit(f"error: generation failed on server: {json.dumps(status)[:500]}")
            outputs = entry.get("outputs", {})
            if outputs:
                return outputs
        time.sleep(poll)
    sys.exit("error: timed out waiting for generation")


def download_outputs(server, outputs, dest_stem):
    saved = []
    for node_output in outputs.values():
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []):
                if item.get("type") != "output":
                    continue
                q = urllib.parse.urlencode(
                    {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": "output"}
                )
                ext = Path(item["filename"]).suffix or ".mp4"
                dest = Path(f"{dest_stem}{ext}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{server}/view?{q}", dest)
                saved.append(dest)
    return saved


def run_job(server, workflow_path, image, prompt, seeds, outdir):
    wf_template = json.loads(Path(workflow_path).read_text())
    image_name = upload_image(server, image)
    stem = Path(image).stem
    for i in range(seeds):
        seed = random.randint(0, 2**48)
        wf = patch_workflow(json.loads(json.dumps(wf_template)), image_name, prompt, seed)
        resp = http_json(f"{server}/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
        prompt_id = resp["prompt_id"]
        print(f"[{stem}] seed {seed} queued ({i + 1}/{seeds}), waiting...", flush=True)
        outputs = wait_for_result(server, prompt_id)
        saved = download_outputs(server, outputs, f"{outdir}/{stem}_{seed}")
        for p in saved:
            print(f"[{stem}] saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI base URL")
    ap.add_argument("--workflow", default=str(Path(__file__).parent / "workflow_api.json"))
    ap.add_argument("--draft", action="store_true",
                    help="fast preview: 720p/16fps, no upscale or interpolation (workflow_draft.json)")
    ap.add_argument("--image", help="input image for a single job")
    ap.add_argument("--prompt", help="motion/style prompt for a single job")
    ap.add_argument("--seeds", type=int, default=1, help="seed variants per image")
    ap.add_argument("--batch", help="CSV with columns: image,prompt[,seeds]")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.draft:
        args.workflow = str(Path(__file__).parent / "workflow_draft.json")
    if not Path(args.workflow).exists():
        sys.exit(f"error: {args.workflow} not found — export your workflow from ComfyUI "
                 f"with 'Export (API)' first (see docstring).")

    jobs = []
    if args.batch:
        with open(args.batch) as f:
            for row in csv.DictReader(f):
                jobs.append((row["image"], row["prompt"], int(row.get("seeds") or args.seeds)))
    elif args.image and args.prompt:
        jobs.append((args.image, args.prompt, args.seeds))
    else:
        sys.exit("error: pass --image and --prompt, or --batch jobs.csv")

    t0 = time.time()
    for image, prompt, seeds in jobs:
        run_job(args.server, args.workflow, image, prompt, seeds, args.out)
    print(f"done: {len(jobs)} job(s) in {int(time.time() - t0)}s")


if __name__ == "__main__":
    main()
