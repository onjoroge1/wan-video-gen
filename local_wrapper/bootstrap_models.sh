#!/bin/bash
# One-time volume bootstrap — run this INSIDE the pod (JupyterLab -> Terminal,
# port 8888 on your first pod). The hearmeman v26 template auto-downloads the
# core Wan 2.2 fp16 set; this adds what the local_wrapper workflows need on top:
#
#   1. fp8_scaled I2V pair   — halves RAM so any >=40GB host runs them (fp16 OOMs
#                              on 43GB cgroup limits mid-job)
#   2. LightX2V 4-step LoRAs — the draft/final fast tiers
#   3. 2x-LiveAction SPAN upscaler — 720p -> 1080p finishing pass
#   4. ComfyUI-Frame-Interpolation (RIFE) — 16fps -> 32fps
#
# Idempotent: wget -c resumes/skips, git pull refreshes. ~30GB total on first run.
set -e
M=/workspace/ComfyUI/models
HF=https://huggingface.co

cd "$M/diffusion_models"
for m in high low; do
  wget -c "$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_${m}_noise_14B_fp8_scaled.safetensors"
done

cd "$M/loras"
for m in high low; do
  wget -c "$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_${m}_noise.safetensors"
done

mkdir -p "$M/upscale_models" && cd "$M/upscale_models"
wget -c -O 2x-LiveAction.pth \
  "$HF/Phips/2xLiveAction_SPAN/resolve/main/2xLiveAction_SPAN.pth" || \
  echo "WARN: upscaler URL moved — grab any 2x SPAN/ESRGAN model and name it 2x-LiveAction.pth"

cd /workspace/ComfyUI/custom_nodes
if [ -d ComfyUI-Frame-Interpolation ]; then
  git -C ComfyUI-Frame-Interpolation pull
else
  git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
fi
pip install -r ComfyUI-Frame-Interpolation/requirements-no-cupy.txt

echo "BOOTSTRAP DONE — restart ComfyUI (or the pod) so new nodes/models load"
