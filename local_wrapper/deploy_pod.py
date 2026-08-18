#!/usr/bin/env python3
"""Deploy a fresh worker pod onto the shared network volume.

  python3 deploy_pod.py                  # name: $RUNPOD_POD_NAME
  python3 deploy_pod.py my-second-pod    # extra pool member (same prefix = pooled)

Reads RUNPOD_API_KEY / RUNPOD_VOLUME_ID (and optional RUNPOD_GPU / RUNPOD_DC /
RUNPOD_IMAGE) from the environment. Waits until ComfyUI answers, then exits —
from that point the api_server's name-prefix resolution picks the pod up.
"""

import os
import sys

import run_batch


def main():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("error: set RUNPOD_API_KEY")
    name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUNPOD_POD_NAME", "wan22-video-gen")
    pod_id = run_batch.create_pod(api_key, name)
    server = f"https://{pod_id}-8188.proxy.runpod.net"
    print(f"waiting for ComfyUI at {server} (first boot on a fresh volume can take ~20 min "
          f"while the template downloads models; ~90s on a warm volume)...")
    run_batch.wait_for_comfyui(server, timeout=1800)
    print(f"pod {pod_id} ready. If this is a brand-new volume, now run bootstrap_models.sh "
          f"in JupyterLab (port 8888) — see INSTALL.md step 4.")


if __name__ == "__main__":
    main()
