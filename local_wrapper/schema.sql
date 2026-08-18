CREATE TABLE IF NOT EXISTS video_jobs (
    id text PRIMARY KEY,
    status text NOT NULL CHECK (
        status IN ('queued', 'starting_pod', 'generating', 'uploading', 'done', 'error', 'cancelled')
    ),
    owner_id text NOT NULL DEFAULT 'local',
    project_id text NOT NULL DEFAULT 'default',
    model_id text NOT NULL DEFAULT 'wan-2.2-standard',
    idempotency_key text,
    request_fingerprint text,
    quality text,
    seed bigint,
    estimated_gpu_cost numeric(14, 6) NOT NULL DEFAULT 0,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE video_jobs ADD COLUMN IF NOT EXISTS owner_id text NOT NULL DEFAULT 'local';
ALTER TABLE video_jobs ADD COLUMN IF NOT EXISTS project_id text NOT NULL DEFAULT 'default';
ALTER TABLE video_jobs ADD COLUMN IF NOT EXISTS model_id text NOT NULL DEFAULT 'wan-2.2-standard';
ALTER TABLE video_jobs DROP CONSTRAINT IF EXISTS video_jobs_idempotency_key_key;
ALTER TABLE video_jobs DROP CONSTRAINT IF EXISTS video_jobs_status_check;
ALTER TABLE video_jobs ADD CONSTRAINT video_jobs_status_check CHECK (
    status IN ('queued', 'starting_pod', 'generating', 'uploading', 'done', 'error', 'cancelled')
);

CREATE UNIQUE INDEX IF NOT EXISTS video_jobs_owner_idempotency_idx
    ON video_jobs (owner_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS video_jobs_status_created_idx
    ON video_jobs (status, created_at);

CREATE INDEX IF NOT EXISTS video_jobs_owner_created_idx
    ON video_jobs (owner_id, created_at DESC);
