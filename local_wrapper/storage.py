#!/usr/bin/env python3
"""Generated-video storage backends."""

import hashlib
import mimetypes
from pathlib import Path


class StorageError(RuntimeError):
    """An artifact could not be durably uploaded or delivered."""


def file_digest(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class LocalVideoStorage:
    backend = "local"

    def store(self, path, job_id):
        path = Path(path).resolve()
        digest, size = file_digest(path)
        suffix = path.suffix.lower() or ".mp4"
        return {
            "storage_backend": self.backend,
            "output": str(path),
            "output_filename": f"wan-{job_id[:12]}{suffix}",
            "output_sha256": digest,
            "output_bytes": size,
            "output_content_type": mimetypes.guess_type(path.name)[0] or "video/mp4",
        }

    def download_url(self, job, expires_seconds):
        return None

    def cleanup_local(self, path):
        return None

    def delete(self, job):
        output = job.get("output")
        if output:
            Path(output).unlink(missing_ok=True)


class S3VideoStorage:
    backend = "s3"

    def __init__(self, bucket, endpoint_url=None, region="auto", prefix="videos",
                 access_key=None, secret_key=None, client=None, delete_local=True):
        if not bucket:
            raise ValueError("bucket is required")
        self.bucket = bucket
        self.endpoint_url = endpoint_url or None
        self.region = region or "auto"
        self.prefix = prefix.strip("/")
        self.delete_local = delete_local
        self.client = client or self._build_client(access_key, secret_key)

    def _build_client(self, access_key, secret_key):
        try:
            import boto3
        except ImportError as exc:
            raise StorageError(
                "S3 storage requires: pip install -r requirements-api.txt"
            ) from exc
        kwargs = {
            "service_name": "s3",
            "region_name": self.region,
        }
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if access_key:
            kwargs["aws_access_key_id"] = access_key
        if secret_key:
            kwargs["aws_secret_access_key"] = secret_key
        return boto3.client(**kwargs)

    def object_key(self, path, job_id):
        filename = f"result{Path(path).suffix.lower() or '.mp4'}"
        return "/".join(part for part in (self.prefix, job_id, filename) if part)

    def store(self, path, job_id):
        path = Path(path).resolve()
        digest, size = file_digest(path)
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        key = self.object_key(path, job_id)
        download_name = f"wan-{job_id[:12]}{path.suffix.lower() or '.mp4'}"
        try:
            self.client.upload_file(
                str(path),
                self.bucket,
                key,
                ExtraArgs={
                    "ContentType": content_type,
                    "ContentDisposition": f'attachment; filename="{download_name}"',
                    "Metadata": {"sha256": digest, "job-id": job_id},
                },
            )
        except Exception as exc:
            raise StorageError(f"could not upload generated video: {exc}") from exc
        return {
            "storage_backend": self.backend,
            "output_key": key,
            "output_filename": download_name,
            "output_sha256": digest,
            "output_bytes": size,
            "output_content_type": content_type,
        }

    def cleanup_local(self, path):
        if self.delete_local:
            Path(path).unlink(missing_ok=True)

    def download_url(self, job, expires_seconds):
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": job["output_key"],
                },
                ExpiresIn=expires_seconds,
            )
        except Exception as exc:
            raise StorageError(f"could not sign generated video URL: {exc}") from exc

    def delete(self, job):
        key = job.get("output_key")
        if not key:
            return
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            raise StorageError(f"could not delete generated video: {exc}") from exc


def build_video_storage(config):
    backend = config.get("backend", "local").lower()
    if backend == "local":
        return LocalVideoStorage()
    if backend != "s3":
        raise ValueError("WAN_STORAGE_BACKEND must be local or s3")
    return S3VideoStorage(
        bucket=config.get("bucket"),
        endpoint_url=config.get("endpoint_url"),
        region=config.get("region", "auto"),
        prefix=config.get("prefix", "videos"),
        access_key=config.get("access_key"),
        secret_key=config.get("secret_key"),
        delete_local=config.get("delete_local", True),
    )
