import base64
import http.client
import json
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "local_wrapper"
sys.path.insert(0, str(WRAPPER))

import api_server
import migrate_jobs
import persistence
import product_api
import run_batch
import storage
import validate_runtime
import wan_client


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"test-image"


def drain_jobs_queue():
    while True:
        try:
            api_server.job_queue.get_nowait()
            api_server.job_queue.task_done()
        except queue.Empty:
            return


class WrapperStateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_out = api_server.OUT_DIR
        self.old_jobs_file = api_server.JOBS_FILE
        self.old_api_key = api_server.API_KEY
        self.old_repository = api_server.JOB_REPOSITORY
        self.old_video_storage = api_server.VIDEO_STORAGE
        self.old_database_url = api_server.DATABASE_URL
        self.old_api_token = api_server.API_TOKEN
        self.old_api_keys = api_server.API_KEYS
        self.old_cors_origins = api_server.CORS_ORIGINS
        api_server.OUT_DIR = Path(self.temp.name) / "outputs"
        api_server.OUT_DIR.mkdir()
        api_server.JOBS_FILE = api_server.OUT_DIR / "jobs.json"
        api_server.JOB_REPOSITORY = persistence.JsonJobRepository(api_server.JOBS_FILE)
        api_server.VIDEO_STORAGE = storage.LocalVideoStorage()
        api_server.DATABASE_URL = ""
        api_server.API_TOKEN = ""
        api_server.API_KEYS = {}
        api_server.CORS_ORIGINS = set()
        api_server.API_KEY = None
        api_server.jobs.clear()
        api_server.PODS.clear()
        drain_jobs_queue()

    def tearDown(self):
        drain_jobs_queue()
        api_server.jobs.clear()
        api_server.PODS.clear()
        api_server.OUT_DIR = self.old_out
        api_server.JOBS_FILE = self.old_jobs_file
        api_server.API_KEY = self.old_api_key
        api_server.JOB_REPOSITORY = self.old_repository
        api_server.VIDEO_STORAGE = self.old_video_storage
        api_server.DATABASE_URL = self.old_database_url
        api_server.API_TOKEN = self.old_api_token
        api_server.API_KEYS = self.old_api_keys
        api_server.CORS_ORIGINS = self.old_cors_origins
        self.temp.cleanup()

    def request(self, **overrides):
        request = {
            "prompt": "slow cinematic push-in",
            "image_b64": base64.b64encode(PNG_BYTES).decode(),
            "quality": "draft",
        }
        request.update(overrides)
        return request

    def test_idempotent_submission_enqueues_once(self):
        status, first = api_server.submit_generation(
            self.request(), idempotency_key="request-0001"
        )
        duplicate_status, second = api_server.submit_generation(
            self.request(), idempotency_key="request-0001"
        )

        self.assertEqual(202, status)
        self.assertEqual(200, duplicate_status)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(1, api_server.job_queue.qsize())
        self.assertEqual(1, len(api_server.jobs))

    def test_idempotency_key_reuse_with_new_request_is_rejected(self):
        api_server.submit_generation(self.request(), idempotency_key="request-0002")
        status, payload = api_server.submit_generation(
            self.request(prompt="a different shot"), idempotency_key="request-0002"
        )
        self.assertEqual(409, status)
        self.assertIn("reused", payload["error"])
        self.assertEqual(1, api_server.job_queue.qsize())

    def test_idempotency_keys_are_scoped_to_owner(self):
        first = product_api.Principal("owner-a")
        second = product_api.Principal("owner-b")
        status_a, result_a = api_server.submit_generation(
            self.request(), idempotency_key="shared-key", principal=first
        )
        status_b, result_b = api_server.submit_generation(
            self.request(), idempotency_key="shared-key", principal=second
        )
        self.assertEqual((202, 202), (status_a, status_b))
        self.assertNotEqual(result_a["job_id"], result_b["job_id"])

    def test_model_registry_decouples_public_model_from_quality(self):
        status, result = api_server.submit_generation(
            self.request(model="wan-2.2-hq", quality=None)
        )
        job = api_server.jobs[result["job_id"]]
        self.assertEqual(202, status)
        self.assertEqual("wan-2.2-hq", job["model_id"])
        self.assertEqual("hq", job["quality"])

    def test_v1_strict_duration_rejects_out_of_contract_value(self):
        status, payload = api_server.submit_generation(
            self.request(seconds=0), strict=True
        )
        self.assertEqual(400, status)
        self.assertIn("between 2 and 7.5", payload["error"])

    def test_pagination_is_owner_scoped_and_stable(self):
        principal = product_api.Principal("owner-a")
        api_server.jobs.update({
            "a3": {"owner_id": "owner-a", "created": "2026-01-03T00:00:00Z", "status": "done"},
            "a2": {"owner_id": "owner-a", "created": "2026-01-02T00:00:00Z", "status": "done"},
            "a1": {"owner_id": "owner-a", "created": "2026-01-01T00:00:00Z", "status": "error"},
            "b4": {"owner_id": "owner-b", "created": "2026-01-04T00:00:00Z", "status": "done"},
        })
        first, cursor = product_api.paginate(api_server.jobs, principal, limit=1, status="done")
        second, next_cursor = product_api.paginate(
            api_server.jobs, principal, limit=1, cursor=cursor, status="done"
        )
        self.assertEqual(["a3"], [item[0] for item in first])
        self.assertEqual(["a2"], [item[0] for item in second])
        self.assertIsNone(next_cursor)

    def test_queued_cancel_is_idempotent_and_retry_does_not_duplicate_queue_entry(self):
        principal = product_api.Principal("owner-a")
        _, result = api_server.submit_generation(self.request(), principal=principal)
        job_id = result["job_id"]
        status, error, job = api_server.cancel_generation(job_id, principal)
        again, again_error, _ = api_server.cancel_generation(job_id, principal)
        self.assertEqual((200, None, "cancelled"), (status, error, job["status"]))
        self.assertEqual((200, None), (again, again_error))
        retry_status, retry_error, retried = api_server.retry_generation(job_id, principal)
        self.assertEqual((202, None, "queued"), (retry_status, retry_error, retried["status"]))
        self.assertEqual(1, api_server.job_queue.qsize())

    def test_owner_cannot_mutate_another_tenants_generation(self):
        api_server.jobs["private"] = {"owner_id": "owner-a", "status": "queued"}
        status, error, _ = api_server.cancel_generation(
            "private", product_api.Principal("owner-b")
        )
        self.assertEqual((404, "not_found"), (status, error))

    def test_delete_removes_result_and_metadata(self):
        output = Path(self.temp.name) / "result.mp4"
        output.write_bytes(b"video")
        api_server.jobs["done-job"] = {
            "owner_id": "owner-a", "status": "done", "output": str(output),
            "storage_backend": "local",
        }
        status, error, _ = api_server.delete_generation(
            "done-job", product_api.Principal("owner-a")
        )
        self.assertEqual((204, None), (status, error))
        self.assertFalse(output.exists())
        self.assertNotIn("done-job", api_server.jobs)

    def test_v1_errors_have_stable_envelope_and_request_id(self):
        api_server.API_KEYS = {
            "tenant-secret": product_api.Principal("owner-a")
        }
        server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), api_server.Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        connection = http.client.HTTPConnection(*server.server_address)
        try:
            connection.request("GET", "/v1/generations/missing", headers={
                "Authorization": "Bearer tenant-secret", "X-Request-ID": "request-123",
            })
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(404, response.status)
            self.assertEqual("not_found", payload["error"]["code"])
            self.assertEqual("request-123", payload["request_id"])
            self.assertEqual("request-123", response.getheader("X-Request-ID"))
        finally:
            connection.close()
            thread.join(timeout=2)
            server.server_close()

    def test_v1_create_assigns_authenticated_owner(self):
        api_server.API_KEYS = {
            "tenant-secret": product_api.Principal("owner-a", project_id="project-7")
        }
        server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), api_server.Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        connection = http.client.HTTPConnection(*server.server_address)
        try:
            body = json.dumps(self.request(model="wan-2.2-draft"))
            connection.request("POST", "/v1/generations", body=body, headers={
                "Authorization": "Bearer tenant-secret",
                "Content-Type": "application/json",
                "Idempotency-Key": "product-request-1",
            })
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(202, response.status)
            self.assertEqual("generation", payload["object"])
            self.assertEqual("wan-2.2-draft", payload["model"])
            job = api_server.jobs[payload["id"]]
            self.assertEqual("owner-a", job["owner_id"])
            self.assertEqual("project-7", job["project_id"])
        finally:
            connection.close()
            thread.join(timeout=2)
            server.server_close()

    def test_cors_preflight_allows_only_configured_origin(self):
        api_server.CORS_ORIGINS = {"https://app.example.com"}
        server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), api_server.Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        connection = http.client.HTTPConnection(*server.server_address)
        try:
            connection.request("OPTIONS", "/v1/generations", headers={
                "Origin": "https://app.example.com",
            })
            response = connection.getresponse()
            response.read()
            self.assertEqual(204, response.status)
            self.assertEqual("https://app.example.com", response.getheader("Access-Control-Allow-Origin"))
        finally:
            connection.close()
            thread.join(timeout=2)
            server.server_close()

    def test_submission_is_not_queued_when_persistence_fails(self):
        api_server.JOB_REPOSITORY = mock.Mock()
        api_server.JOB_REPOSITORY.save_job.side_effect = persistence.PersistenceError(
            "database offline"
        )
        status, payload = api_server.submit_generation(self.request())
        self.assertEqual(503, status)
        self.assertIn("durably", payload["error"])
        self.assertEqual(0, api_server.job_queue.qsize())
        self.assertEqual({}, api_server.jobs)

    def test_remote_callers_cannot_read_local_paths(self):
        image = Path(self.temp.name) / "private.jpg"
        image.write_bytes(b"\xff\xd8\xff" + b"private")
        status, payload = api_server.submit_generation(
            {"prompt": "test", "image_path": str(image)}, allow_local_path=False
        )
        self.assertEqual(400, status)
        self.assertIn("localhost", payload["error"])

    def test_all_loopback_bind_forms_are_recognized(self):
        for address in ("127.0.0.1", "::1", "localhost"):
            self.assertTrue(api_server.is_loopback_bind(address))
        self.assertFalse(api_server.is_loopback_bind("0.0.0.0"))

    def test_invalid_image_content_is_rejected(self):
        status, payload = api_server.submit_generation(
            self.request(image_b64=base64.b64encode(b"not-an-image").decode())
        )
        self.assertEqual(400, status)
        self.assertIn("JPEG, PNG, or WebP", payload["error"])

    def test_health_does_not_disclose_worker_id(self):
        pod = api_server.Pod("secret-provider-id")
        pod.running = True
        api_server.PODS.append(pod)
        payload = api_server.health_payload()
        self.assertNotIn("secret-provider-id", json.dumps(payload))
        self.assertEqual(1, payload["workers"]["running"])

    def test_public_job_redacts_provider_and_idempotency_data(self):
        public = api_server.public_job({
            "status": "generating",
            "pod": "secret-provider-id",
            "provider_prompt_id": "prompt-secret",
            "idempotency_key": "request-secret",
            "output": "/private/secret-output.mp4",
            "output_key": "private/secret-key.mp4",
            "attempt_history": [{"pod": "secret-provider-id", "status": "active"}],
        })
        serialized = json.dumps(public)
        self.assertNotIn("secret", serialized)
        self.assertIn("active", serialized)

    def test_generating_job_resumes_existing_provider_prompt_after_restart(self):
        image = Path(self.temp.name) / "input.png"
        image.write_bytes(PNG_BYTES)
        api_server.JOBS_FILE.write_text(json.dumps({
            "job-1": {
                "status": "generating",
                "image_path": str(image),
                "provider_prompt_id": "provider-prompt-1",
            }
        }))
        api_server.load_jobs()
        self.assertEqual("queued", api_server.jobs["job-1"]["status"])
        self.assertTrue(api_server.jobs["job-1"]["resume_provider_prompt"])
        self.assertEqual(1, api_server.job_queue.qsize())

    def test_empty_postgres_does_not_silently_ignore_legacy_jobs(self):
        api_server.DATABASE_URL = "postgresql://configured"
        api_server.JOBS_FILE.write_text(json.dumps({
            "legacy-job": {"status": "done"}
        }))
        repository = mock.Mock()
        repository.load_all.return_value = {}
        api_server.JOB_REPOSITORY = repository
        with self.assertRaisesRegex(RuntimeError, "migrate_jobs.py"):
            api_server.load_jobs()

    def test_pod_stop_keeps_running_state_when_shutdown_is_unconfirmed(self):
        api_server.API_KEY = "test-key"
        pod = api_server.Pod("pod-1")
        pod.running = True
        with mock.patch.object(run_batch, "get_pod", return_value={
            "desiredStatus": "RUNNING", "costPerHr": 1.0,
        }), mock.patch.object(run_batch, "stop_pod", side_effect=RuntimeError("no confirmation")):
            with self.assertRaises(RuntimeError):
                pod.stop()
        self.assertTrue(pod.running)

    def test_self_heal_terminates_dead_pod_before_creating_replacement(self):
        api_server.API_KEY = "test-key"
        pod = api_server.Pod("dead-pod")
        events = []

        with mock.patch.object(
            run_batch, "start_pod", side_effect=RuntimeError("unstartable")
        ), mock.patch.object(
            run_batch, "terminate_pod",
            side_effect=lambda pod_id, _key: events.append(("terminate", pod_id)),
        ), mock.patch.object(
            run_batch, "create_pod",
            side_effect=lambda _key, _name: events.append(("create", "new-pod")) or "new-pod",
        ), mock.patch.object(run_batch, "wait_for_comfyui"):
            pod.ensure()

        self.assertEqual([("terminate", "dead-pod"), ("create", "new-pod")], events)
        self.assertEqual("new-pod", pod.id)

    def test_worker_thread_retires(self):
        pod = api_server.Pod("pod-1")
        pod.start_worker()
        pod.retire()
        self.assertTrue(pod.retired)
        self.assertFalse(pod.worker_thread.is_alive())

    def test_comfyui_generation_error_is_not_retried(self):
        image = Path(self.temp.name) / "input.png"
        image.write_bytes(PNG_BYTES)
        job_id = "job-generation-error"
        api_server.jobs[job_id] = {
            "status": "queued",
            "image_path": str(image),
            "prompt": "test",
            "seed": 1,
            "quality": "draft",
            "frames": 33,
            "workflow": str(WRAPPER / "workflow_draft.json"),
            "attempt_count": 0,
            "attempt_history": [],
        }
        api_server.job_queue.put(job_id)
        api_server.job_queue.get_nowait()
        pod = api_server.Pod("pod-1")
        pod.running = True
        with mock.patch.object(wan_client, "upload_image", return_value="input.png"), \
                mock.patch.object(wan_client, "http_json", return_value={"prompt_id": "p1"}), \
                mock.patch.object(wan_client, "wait_for_result",
                                  side_effect=wan_client.GenerationError("bad workflow")), \
                mock.patch.object(api_server, "caffeinate"):
            api_server._worker_one(pod, job_id)

        job = api_server.jobs[job_id]
        self.assertEqual("error", job["status"])
        self.assertEqual("generation_error", job["error_code"])
        self.assertEqual(1, job["attempt_count"])
        self.assertEqual(0, api_server.job_queue.qsize())

    def test_attempt_cost_is_recorded(self):
        job = {"attempt_history": []}
        attempt = {"started_epoch": 640.0, "hourly_cost": 1.0}
        job["attempt_history"].append(attempt)
        with mock.patch.object(api_server.time, "time", return_value=1000.0):
            api_server._finish_attempt(job, attempt, "done")
        self.assertEqual(360.0, attempt["elapsed_seconds"])
        self.assertEqual(0.1, attempt["estimated_gpu_cost"])
        self.assertEqual(0.1, job["estimated_gpu_cost"])

    def test_terminal_watchdog_job_is_not_requeued(self):
        image = Path(self.temp.name) / "input.png"
        image.write_bytes(PNG_BYTES)
        job_id = "job-watchdog-terminal"
        api_server.jobs[job_id] = {
            "status": "error",
            "terminal": True,
            "error_code": "watchdog",
            "error": "killed by money watchdog",
            "image_path": str(image),
            "prompt": "test",
            "seed": 1,
            "quality": "draft",
            "frames": 33,
            "workflow": str(WRAPPER / "workflow_draft.json"),
            "attempt_count": 0,
            "attempt_history": [],
        }
        api_server.job_queue.put(job_id)
        api_server.job_queue.get_nowait()
        pod = api_server.Pod("pod-1")

        with mock.patch.object(api_server, "caffeinate"):
            api_server._worker_one(pod, job_id)

        job = api_server.jobs[job_id]
        self.assertEqual("error", job["status"])
        self.assertEqual("watchdog", job["error_code"])
        self.assertEqual("cancelled", job["attempt_history"][0]["status"])
        self.assertEqual(0, api_server.job_queue.qsize())

    def test_pending_output_is_stored_without_restarting_gpu_generation(self):
        output = Path(self.temp.name) / "already-generated.mp4"
        output.write_bytes(b"video-bytes")
        managed_input = Path(self.temp.name) / "managed-input.png"
        managed_input.write_bytes(PNG_BYTES)
        job_id = "job-storage-retry"
        api_server.jobs[job_id] = {
            "status": "queued",
            "pending_output": str(output),
            "image_path": str(managed_input),
            "managed_input": True,
            "attempt_count": 0,
            "attempt_history": [],
        }
        api_server.job_queue.put(job_id)
        api_server.job_queue.get_nowait()
        pod = api_server.Pod("pod-1")

        with mock.patch.object(pod, "ensure") as ensure, \
                mock.patch.object(api_server, "caffeinate"):
            api_server._worker_one(pod, job_id)

        ensure.assert_not_called()
        self.assertEqual("done", api_server.jobs[job_id]["status"])
        self.assertEqual("local", api_server.jobs[job_id]["storage_backend"])
        self.assertFalse(managed_input.exists())
        self.assertNotIn("image_path", api_server.jobs[job_id])
        self.assertEqual(0, api_server.job_queue.qsize())

    def test_s3_result_endpoint_redirects_to_short_lived_url(self):
        backend = mock.Mock()
        backend.backend = "s3"
        backend.download_url.return_value = "https://signed.example/video"
        api_server.VIDEO_STORAGE = backend
        api_server.jobs["job-1"] = {
            "status": "done",
            "storage_backend": "s3",
            "output_key": "videos/job-1/result.mp4",
        }
        server = api_server.ThreadingHTTPServer(
            ("127.0.0.1", 0), api_server.Handler
        )
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address)
            connection.request("GET", "/result/job-1")
            response = connection.getresponse()
            self.assertEqual(302, response.status)
            self.assertEqual("https://signed.example/video", response.getheader("Location"))
            self.assertEqual("private, no-store", response.getheader("Cache-Control"))
            backend.download_url.assert_called_once()
        finally:
            connection.close()
            thread.join(timeout=2)
            server.server_close()


class PersistenceAndStorageTest(unittest.TestCase):
    def test_json_repository_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = persistence.JsonJobRepository(Path(directory) / "jobs.json")
            repository.save_all({"job-1": {"status": "done", "seed": 4}})
            self.assertEqual(
                {"job-1": {"status": "done", "seed": 4}},
                repository.load_all(),
            )

    def test_schema_enforces_idempotency_and_uses_jsonb(self):
        schema = (WRAPPER / "schema.sql").read_text().lower()
        self.assertIn("on video_jobs (owner_id, idempotency_key)", schema)
        self.assertIn("payload jsonb not null", schema)
        self.assertIn("status, created_at", schema)
        self.assertIn("'cancelled'", schema)

    def test_openapi_contract_contains_product_lifecycle(self):
        contract = (WRAPPER / "openapi.yaml").read_text()
        for path in (
            "/v1/models:", "/v1/generations:",
            "/v1/generations/{generation_id}/cancel:",
            "/v1/generations/{generation_id}/retry:",
            "/v1/generations/{generation_id}/result:",
        ):
            self.assertIn(path, contract)

    def test_postgres_repository_upserts_one_job_not_full_history(self):
        connection = mock.MagicMock()
        connection.__enter__.return_value = connection
        pool = mock.Mock()
        pool.connection.return_value = connection
        repository = persistence.PostgresJobRepository(
            "postgresql://example",
            pool=pool,
            jsonb_factory=lambda value: ("jsonb", value),
        )

        repository.save_job(
            "job-1",
            {"status": "done", "quality": "draft", "seed": 7},
        )

        connection.execute.assert_called_once()
        sql, params = connection.execute.call_args.args
        self.assertIn("on conflict (id) do update", sql.lower())
        self.assertEqual("job-1", params[0])
        self.assertEqual(("jsonb", {"status": "done", "quality": "draft", "seed": 7}),
                         params[-1])

    def test_s3_storage_uploads_private_object_and_signs_download(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.write_bytes(b"video-bytes")
            client = mock.Mock()
            client.generate_presigned_url.return_value = "https://signed.example/video"
            backend = storage.S3VideoStorage(
                bucket="private-videos",
                endpoint_url="https://account.r2.cloudflarestorage.com",
                client=client,
                delete_local=True,
            )

            metadata = backend.store(video, "abcdef1234567890")
            self.assertTrue(video.exists(), "local result stays until metadata is durable")
            client.upload_file.assert_called_once()
            upload = client.upload_file.call_args
            self.assertEqual("private-videos", upload.args[1])
            self.assertEqual("videos/abcdef1234567890/result.mp4", upload.args[2])
            self.assertNotIn("ACL", upload.kwargs["ExtraArgs"])
            self.assertEqual("s3", metadata["storage_backend"])

            url = backend.download_url(metadata, 3600)
            self.assertEqual("https://signed.example/video", url)
            self.assertEqual(3600, client.generate_presigned_url.call_args.kwargs["ExpiresIn"])
            backend.cleanup_local(video)
            self.assertFalse(video.exists())

            backend.delete(metadata)
            client.delete_object.assert_called_once_with(
                Bucket="private-videos", Key="videos/abcdef1234567890/result.mp4"
            )

    def test_failed_s3_upload_keeps_local_recovery_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.write_bytes(b"video-bytes")
            client = mock.Mock()
            client.upload_file.side_effect = OSError("object store offline")
            backend = storage.S3VideoStorage(
                bucket="private-videos", client=client, delete_local=True
            )
            with self.assertRaises(storage.StorageError):
                backend.store(video, "job-1")
            self.assertTrue(video.exists())

    def test_unknown_storage_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            storage.build_video_storage({"backend": "database"})

    def test_json_to_postgres_migration_dry_run_needs_no_database_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "jobs.json"
            source.write_text(json.dumps({"job-1": {"status": "done"}}))
            with mock.patch.object(
                sys, "argv", [
                    "migrate_jobs.py", "--source", str(source),
                    "--database-url", "postgresql://unused", "--dry-run",
                ],
            ), mock.patch.object(
                persistence.PostgresJobRepository, "initialize"
            ) as initialize:
                self.assertEqual(0, migrate_jobs.main())
            initialize.assert_not_called()


class ClientAndRuntimeTest(unittest.TestCase):
    def test_workflow_errors_raise_generation_error_not_system_exit(self):
        with self.assertRaises(wan_client.GenerationError):
            wan_client.patch_workflow({}, "image.png", "prompt", 1)

    def test_server_generation_error_is_typed(self):
        history = {
            "prompt-1": {"status": {"status_str": "error", "messages": []}}
        }
        with mock.patch.object(wan_client, "http_json", return_value=history):
            with self.assertRaises(wan_client.GenerationError):
                wan_client.wait_for_result("http://example", "prompt-1", poll=0, timeout=1)

    def test_waiting_generation_uses_atomic_per_job_cancellation(self):
        with mock.patch.object(wan_client, "cancel_prompt", return_value=True) as cancel:
            with self.assertRaises(wan_client.GenerationCancelled):
                wan_client.wait_for_result(
                    "http://example", "prompt-1", poll=0, timeout=1,
                    should_cancel=lambda: True,
                )
        cancel.assert_called_once_with("http://example", "prompt-1")

    def test_stop_pod_requires_provider_confirmation(self):
        responses = [
            {},
            {"desiredStatus": "RUNNING"},
            {"desiredStatus": "EXITED"},
        ]
        with mock.patch.object(run_batch, "rp", side_effect=responses), \
                mock.patch.object(run_batch.time, "sleep"):
            result = run_batch.stop_pod("pod-1", "key")
        self.assertEqual("EXITED", result["desiredStatus"])

    def test_stop_pod_raises_when_provider_never_confirms(self):
        with mock.patch.object(run_batch, "rp", side_effect=OSError("offline")), \
                mock.patch.object(run_batch.time, "sleep"):
            with self.assertRaises(RuntimeError):
                run_batch.stop_pod("pod-1", "key")

    def test_workflows_are_structurally_valid(self):
        for path in WRAPPER.glob("workflow_*.json"):
            classes, models = validate_runtime.inspect_workflow(path)
            self.assertTrue(classes)
            self.assertTrue(models)

    def test_bootstrap_names_match_fast_workflow_dependencies(self):
        bootstrap = (WRAPPER / "bootstrap_models.sh").read_text()
        expanded_bootstrap = (
            bootstrap.replace("${m}", "high") + "\n" +
            bootstrap.replace("${m}", "low")
        )
        workflow = json.loads((WRAPPER / "workflow_api.json").read_text())
        expected = {
            value
            for node in workflow.values()
            for key, value in node.get("inputs", {}).items()
            if key in {"lora_name", "model_name"}
        }
        for filename in expected:
            self.assertIn(filename, expanded_bootstrap)


if __name__ == "__main__":
    unittest.main()
