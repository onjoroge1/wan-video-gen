#!/usr/bin/env python3
"""Validate workflow structure and, on a pod, referenced models and node classes."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


MODEL_INPUTS = {
    "unet_name",
    "lora_name",
    "clip_name",
    "vae_name",
    "model_name",
    "ckpt_name",
}
REQUIRED_BINDINGS = {
    "LoadImage",
    "CLIPTextEncode",
    "KSamplerAdvanced",
    "WanImageToVideo",
}


def load_workflow(path):
    try:
        workflow = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"{path.name}: invalid JSON: {exc}") from exc
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError(f"{path.name}: workflow must be a non-empty object")
    return workflow


def inspect_workflow(path):
    workflow = load_workflow(path)
    classes = {node.get("class_type") for node in workflow.values()}
    missing_bindings = REQUIRED_BINDINGS - classes
    if missing_bindings:
        raise ValueError(
            f"{path.name}: missing required nodes: {sorted(missing_bindings)}"
        )
    models = set()
    for node in workflow.values():
        for key, value in node.get("inputs", {}).items():
            if key in MODEL_INPUTS and isinstance(value, str):
                models.add(value)
    return classes, models


def available_files(comfy_root):
    search_roots = [comfy_root / "models", comfy_root / "custom_nodes"]
    available = set()
    for root in search_roots:
        if root.exists():
            available.update(path.name for path in root.rglob("*") if path.is_file())
    return available


def server_node_classes(server):
    request = urllib.request.Request(
        f"{server.rstrip('/')}/object_info",
        headers={"User-Agent": "wan-runtime-validator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return set(json.loads(response.read()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-dir", type=Path, default=Path(__file__).parent,
        help="directory containing workflow_*.json",
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path("/workspace/ComfyUI"),
        help="ComfyUI installation root",
    )
    parser.add_argument("--server", help="optional ComfyUI URL for node validation")
    parser.add_argument(
        "--check-models", action="store_true",
        help="verify every referenced model basename exists under ComfyUI",
    )
    args = parser.parse_args()

    workflows = sorted(args.workflow_dir.glob("workflow_*.json"))
    errors = []
    required_classes = set()
    required_models = set()
    if not workflows:
        errors.append(f"no workflow_*.json files found in {args.workflow_dir}")
    for path in workflows:
        try:
            classes, models = inspect_workflow(path)
            required_classes.update(classes)
            required_models.update(models)
            print(f"OK workflow {path.name}: {len(classes)} node classes, {len(models)} models")
        except ValueError as exc:
            errors.append(str(exc))

    if args.check_models:
        available = available_files(args.comfy_root)
        for model in sorted(required_models - available):
            errors.append(f"missing model file: {model}")
        if not required_models - available:
            print(f"OK models: {len(required_models)} referenced files found")

    if args.server:
        try:
            available_classes = server_node_classes(args.server)
            missing_classes = required_classes - available_classes
            for class_name in sorted(missing_classes):
                errors.append(f"ComfyUI missing node class: {class_name}")
            if not missing_classes:
                print(f"OK nodes: {len(required_classes)} workflow classes available")
        except Exception as exc:
            errors.append(f"could not query ComfyUI object_info: {exc}")

    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 1
    print("runtime validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
