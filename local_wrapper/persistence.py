#!/usr/bin/env python3
"""Durable job repositories for the local API wrapper."""

import json
import threading
from pathlib import Path


SCHEMA_FILE = Path(__file__).with_name("schema.sql")


class PersistenceError(RuntimeError):
    """Job state could not be durably stored or loaded."""


class JsonJobRepository:
    """Atomic single-process repository used for zero-dependency development."""

    backend = "json"

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self):
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text())
        except Exception as exc:
            raise PersistenceError(f"could not read {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PersistenceError(f"{self.path} must contain a JSON object")
        return payload

    def save_all(self, jobs):
        self.initialize()
        try:
            with self.lock:
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(json.dumps(jobs, indent=1))
                temporary.replace(self.path)
        except Exception as exc:
            raise PersistenceError(f"could not write {self.path}: {exc}") from exc

    def save_job(self, job_id, job, all_jobs):
        self.save_all(all_jobs)

    def delete_job(self, job_id, all_jobs):
        self.save_all(all_jobs)

    def close(self):
        return None


class PostgresJobRepository:
    """Thread-safe Postgres JSONB repository backed by a small connection pool."""

    backend = "postgres"

    def __init__(self, database_url, pool=None, jsonb_factory=None):
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self.pool = pool
        self.jsonb_factory = jsonb_factory

    def _jsonb_factory(self):
        if self.jsonb_factory is None:
            from psycopg.types.json import Jsonb
            self.jsonb_factory = Jsonb
        return self.jsonb_factory

    def _ensure_pool(self):
        if self.pool is not None:
            return
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise PersistenceError(
                "Postgres requires: pip install -r requirements-api.txt"
            ) from exc
        self.pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True},
            open=True,
        )

    def initialize(self):
        self._ensure_pool()
        schema = SCHEMA_FILE.read_text()
        try:
            with self.pool.connection() as connection:
                connection.execute(schema)
        except Exception as exc:
            raise PersistenceError(f"could not initialize Postgres schema: {exc}") from exc

    def load_all(self):
        self._ensure_pool()
        try:
            with self.pool.connection() as connection:
                rows = connection.execute(
                    "SELECT id, payload FROM video_jobs ORDER BY created_at"
                ).fetchall()
        except Exception as exc:
            raise PersistenceError(f"could not load jobs from Postgres: {exc}") from exc
        return {str(job_id): payload for job_id, payload in rows}

    def save_all(self, jobs):
        self._ensure_pool()
        try:
            jsonb = self._jsonb_factory()
            with self.pool.connection() as connection:
                with connection.transaction():
                    for job_id, job in jobs.items():
                        self._upsert(connection, jsonb, job_id, job)
        except Exception as exc:
            raise PersistenceError(f"could not save jobs to Postgres: {exc}") from exc

    def save_job(self, job_id, job, all_jobs=None):
        self._ensure_pool()
        try:
            jsonb = self._jsonb_factory()
            with self.pool.connection() as connection:
                self._upsert(connection, jsonb, job_id, job)
        except Exception as exc:
            raise PersistenceError(f"could not save job to Postgres: {exc}") from exc

    def delete_job(self, job_id, all_jobs=None):
        self._ensure_pool()
        try:
            with self.pool.connection() as connection:
                connection.execute("DELETE FROM video_jobs WHERE id = %s", (job_id,))
        except Exception as exc:
            raise PersistenceError(f"could not delete job from Postgres: {exc}") from exc

    @staticmethod
    def _upsert(connection, jsonb, job_id, job):
        connection.execute(
            """
            INSERT INTO video_jobs (
                id, owner_id, project_id, model_id, status, idempotency_key,
                request_fingerprint, quality, seed, estimated_gpu_cost, payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                owner_id = EXCLUDED.owner_id,
                project_id = EXCLUDED.project_id,
                model_id = EXCLUDED.model_id,
                status = EXCLUDED.status,
                idempotency_key = EXCLUDED.idempotency_key,
                request_fingerprint = EXCLUDED.request_fingerprint,
                quality = EXCLUDED.quality,
                seed = EXCLUDED.seed,
                estimated_gpu_cost = EXCLUDED.estimated_gpu_cost,
                payload = EXCLUDED.payload,
                updated_at = now()
            """,
            (
                job_id,
                job.get("owner_id", "local"),
                job.get("project_id", "default"),
                job.get("model_id", "wan-2.2-standard"),
                job.get("status", "error"),
                job.get("idempotency_key"),
                job.get("request_fingerprint"),
                job.get("quality"),
                job.get("seed"),
                job.get("estimated_gpu_cost", 0),
                jsonb(job),
            ),
        )

    def close(self):
        if self.pool is not None:
            self.pool.close()


def build_job_repository(database_url, json_path):
    if database_url:
        return PostgresJobRepository(database_url)
    return JsonJobRepository(json_path)
