#!/usr/bin/env python3
"""Stable product-facing primitives for the Wan generation API."""

import base64
import hmac
import json
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    owner_id: str
    role: str = "user"
    project_id: str = "default"

    @property
    def is_admin(self):
        return self.role == "admin"


def load_api_keys(raw):
    """Parse WAN_API_KEYS_JSON without ever returning the secret in values."""
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("WAN_API_KEYS_JSON must be a JSON object")
    result = {}
    for token, config in payload.items():
        if not isinstance(token, str) or len(token) < 8 or not isinstance(config, dict):
            raise ValueError("each API key must be at least 8 characters and map to an object")
        owner_id = str(config.get("owner_id", "")).strip()
        role = str(config.get("role", "user"))
        project_id = str(config.get("project_id", "default"))
        if not owner_id or role not in ("user", "admin"):
            raise ValueError("each API key needs owner_id and role user or admin")
        result[token] = Principal(owner_id, role, project_id)
    return result


def authenticate(bearer_token, legacy_token, api_keys, allow_anonymous_local=False):
    if bearer_token:
        for candidate, principal in api_keys.items():
            if hmac.compare_digest(bearer_token, candidate):
                return principal
        if legacy_token and hmac.compare_digest(bearer_token, legacy_token):
            return Principal("local", "admin", "default")
        return None
    if allow_anonymous_local and not legacy_token and not api_keys:
        return Principal("local", "admin", "default")
    return None


def owns(principal, job):
    return principal.is_admin or job.get("owner_id", "local") == principal.owner_id


def encode_cursor(created, job_id):
    raw = json.dumps([created, job_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(value):
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        created, job_id = json.loads(raw)
        if not isinstance(created, str) or not isinstance(job_id, str):
            raise ValueError
        return created, job_id
    except Exception as exc:
        raise ValueError("cursor is invalid") from exc


def paginate(jobs, principal, limit=20, cursor=None, status=None, model_id=None):
    visible = [
        (job_id, job) for job_id, job in jobs.items()
        if owns(principal, job)
        and (not status or job.get("status") == status)
        and (not model_id or job.get("model_id") == model_id)
    ]
    visible.sort(key=lambda item: (item[1].get("created", ""), item[0]), reverse=True)
    if cursor:
        boundary = decode_cursor(cursor)
        visible = [item for item in visible if (item[1].get("created", ""), item[0]) < boundary]
    page = visible[:limit]
    next_cursor = None
    if len(visible) > limit and page:
        last_id, last_job = page[-1]
        next_cursor = encode_cursor(last_job.get("created", ""), last_id)
    return page, next_cursor


STAGE_PROGRESS = {
    "queued": 0,
    "starting_pod": 5,
    "generating": 25,
    "uploading": 90,
    "done": 100,
    "cancelled": 100,
    "error": 100,
}


def generation_view(job_id, job):
    public_fields = {
        "status", "prompt", "seed", "quality", "frames", "owner_id",
        "project_id", "attempt_count", "estimated_gpu_cost", "created",
        "completed", "cancelled", "retried", "error_code", "output_filename",
        "output_sha256", "output_bytes", "output_content_type", "storage_backend",
    }
    data = {key: value for key, value in job.items() if key in public_fields}
    data.update({
        "id": job_id,
        "object": "generation",
        "model": job.get("model_id", "wan-2.2-standard"),
        "progress": {
            "stage": job.get("status", "queued"),
            "percent": STAGE_PROGRESS.get(job.get("status"), 0),
            "estimated": False,
        },
        "result_url": f"/v1/generations/{job_id}/result" if job.get("status") == "done" else None,
    })
    data["attempt_history"] = [
        {key: value for key, value in attempt.items() if key not in ("pod", "error")}
        for attempt in job.get("attempt_history", [])
    ]
    return data


def model_catalog(model_registry, jobs):
    result = []
    for model_id, model in model_registry.items():
        samples = sorted(
            float(job.get("estimated_gpu_cost") or 0)
            for job in jobs.values()
            if job.get("model_id") == model_id and job.get("status") == "done"
            and float(job.get("estimated_gpu_cost") or 0) > 0
        )
        entry = {key: value for key, value in model.items() if key != "workflow"}
        entry.update({"id": model_id, "object": "model", "cost_samples": len(samples)})
        if len(samples) >= 3:
            entry["observed_cost"] = {
                "currency": "USD",
                "median": round(statistics.median(samples), 6),
                "p90": round(samples[min(len(samples) - 1, int(len(samples) * .9))], 6),
            }
        result.append(entry)
    return result
