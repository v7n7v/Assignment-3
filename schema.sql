-- ============================================================
-- Assignment 3 Schema
-- PostgreSQL + pgvector + Apache AGE
-- ============================================================

-- ---- A2 base tables (detection pipeline) --------------------

CREATE TABLE IF NOT EXISTS detection_events (
    event_id    uuid PRIMARY KEY,
    run_id      uuid NOT NULL,
    robot_id    text NOT NULL,
    sequence    bigint NOT NULL,
    stamp       timestamptz NOT NULL,
    image_frame_id text,
    image_sha256   text,
    width       int,
    height      int,
    encoding    text,
    x           double precision,
    y           double precision,
    yaw         double precision,
    vx          double precision,
    vy          double precision,
    wz          double precision,
    tf_ok       boolean,
    t_base_camera double precision[],
    raw_event   jsonb NOT NULL,
    UNIQUE(run_id, robot_id, sequence)
);

CREATE TABLE IF NOT EXISTS detections (
    det_pk      bigserial PRIMARY KEY,
    event_id    uuid REFERENCES detection_events(event_id) ON DELETE CASCADE,
    det_id      uuid,
    class_id    int,
    class_name  text,
    confidence  double precision,
    x1          double precision,
    y1          double precision,
    x2          double precision,
    y2          double precision,
    UNIQUE(event_id, det_id)
);

-- ---- pgvector embeddings ------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS detection_embeddings (
    det_pk      bigint PRIMARY KEY REFERENCES detections(det_pk),
    model       text DEFAULT 'ViT-B/32',
    embedding   vector(512)
);

-- index for fast KNN
CREATE INDEX IF NOT EXISTS idx_embed_ivfflat
    ON detection_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

-- ---- Apache AGE semantic graph ------------------------------

CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- create the graph (skip if it already exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'semantic_map'
    ) THEN
        PERFORM ag_catalog.create_graph('semantic_map');
    END IF;
END
$$;
