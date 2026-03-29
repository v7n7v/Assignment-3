#!/usr/bin/env python3
"""
Seeds the database with fake detection data for testing.
Generates a trajectory with detections + embeddings + graph
so we can demo relocalization without running the sim.
"""

import os
import sys
import uuid
import math
import time
import json
from datetime import datetime, timezone

import numpy as np
import psycopg2

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "maze_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "postgres")

GRID_SIZE = 1.0
FUSION_DIST = 1.0

# reproducible random
rng = np.random.RandomState(42)

# --- synthetic CLIP embeddings ---
# each object class gets a base vector; instances add small noise
# so KNN will cluster by class (which is the whole point)

CLASSES = [
    "cup", "chair", "book", "bottle", "dining table",
    "potted plant", "couch", "tv", "bed", "clock", "laptop",
]
CLASS_BASES = {}
for c in CLASSES:
    v = rng.randn(512).astype(np.float32)
    v /= np.linalg.norm(v)
    CLASS_BASES[c] = v


def make_embedding(class_name):
    base = CLASS_BASES[class_name]
    noise = rng.randn(512).astype(np.float32) * 0.08
    v = base + noise
    v /= np.linalg.norm(v)
    return v.tolist()


# --- simulated trajectory through a small maze ---
# (x, y, yaw, [(class, confidence), ...])
TRAJECTORY = [
    (0.3, 0.4, 0.0,   [("chair", 0.87), ("cup", 0.92)]),
    (0.8, 0.3, 0.2,   [("cup", 0.88), ("book", 0.79)]),
    (1.4, 0.4, 0.1,   [("bottle", 0.91)]),
    (2.1, 0.3, 0.5,   [("cup", 0.84), ("bottle", 0.86)]),
    (2.4, 0.8, 1.2,   [("chair", 0.93), ("dining table", 0.85)]),
    (2.2, 1.4, 1.8,   [("potted plant", 0.82), ("cup", 0.80)]),
    (1.5, 1.3, 2.5,   [("couch", 0.88), ("tv", 0.91)]),
    (1.0, 1.2, 2.8,   [("book", 0.77), ("laptop", 0.83)]),
    (0.4, 1.3, 3.0,   [("bed", 0.86), ("clock", 0.74)]),
    (0.3, 0.7, -0.5,  [("chair", 0.84), ("cup", 0.90)]),
]


# ---- helpers ----

def get_conn():
    for i in range(15):
        try:
            conn = psycopg2.connect(
                host=DB_HOST, port=DB_PORT,
                dbname=DB_NAME, user=DB_USER, password=DB_PASS,
            )
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError:
            print(f"Waiting for db ({i+1})...")
            time.sleep(2)
    sys.exit("Could not connect to database")


def init_age(cur):
    cur.execute("LOAD 'age';")
    cur.execute('SET search_path = ag_catalog, "$user", public;')


def cy(cur, q):
    cur.execute(f"SELECT * FROM cypher('semantic_map', $$ {q} $$) AS (v agtype);")
    return cur.fetchall()


def place_id(x, y):
    gx = int(math.floor(x / GRID_SIZE))
    gy = int(math.floor(y / GRID_SIZE))
    return f"place_{gx}_{gy}", gx, gy


def main():
    conn = get_conn()
    cur = conn.cursor()
    init_age(cur)

    # ensure helper table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            object_id text PRIMARY KEY,
            class_name text,
            mean_x double precision,
            mean_y double precision,
            obs_count int DEFAULT 1,
            first_seen timestamptz DEFAULT now(),
            last_seen timestamptz DEFAULT now()
        )
    """)
    conn.commit()

    run_id = str(uuid.uuid4())
    base_time = time.time()
    print(f"Seeding run {run_id} with {len(TRAJECTORY)} keyframes ...\n")

    # graph: create run node
    cy(cur, f"MERGE (r:Run {{run_id: '{run_id}'}}) RETURN r")

    for kf_idx, (x, y, yaw, det_list) in enumerate(TRAJECTORY, start=1):
        event_id = str(uuid.uuid4())
        ts = datetime.fromtimestamp(base_time + kf_idx * 2, tz=timezone.utc)

        # --- relational: detection_events ---
        raw = {"seed": True, "kf": kf_idx}
        cur.execute("""
            INSERT INTO detection_events
            (event_id, run_id, robot_id, sequence, stamp,
             width, height, encoding, x, y, yaw, raw_event)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            event_id, run_id, "tb3_sim", kf_idx, ts,
            640, 480, "rgb8", x, y, yaw, json.dumps(raw),
        ))

        # graph: keyframe + pose
        cy(cur, f"""
            MATCH (r:Run {{run_id: '{run_id}'}})
            CREATE (r)-[:HAS_KEYFRAME]->(k:Keyframe {{
                keyframe_id: {kf_idx}, timestamp: '{ts.isoformat()}'
            }})
            RETURN k
        """)

        cy(cur, f"""
            MATCH (k:Keyframe {{keyframe_id: {kf_idx}}})
            CREATE (k)-[:AT_POSE]->(p:Pose {{
                map_x: {x}, map_y: {y}, map_yaw: {yaw}
            }})
            RETURN p
        """)

        # --- per-detection ---
        for cls, conf in det_list:
            det_id = str(uuid.uuid4())
            emb = make_embedding(cls)

            # relational: detections
            cur.execute("""
                INSERT INTO detections
                (event_id, det_id, class_id, class_name, confidence,
                 x1, y1, x2, y2)
                VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s)
                ON CONFLICT DO NOTHING
                RETURNING det_pk
            """, (
                event_id, det_id,
                CLASSES.index(cls), cls, conf,
                100, 100, 200, 200,  # fake bbox
            ))
            row = cur.fetchone()
            if row is None:
                continue
            det_pk = row[0]

            # relational: embedding
            vec_str = "[" + ",".join(str(v) for v in emb) + "]"
            cur.execute("""
                INSERT INTO detection_embeddings (det_pk, model, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT DO NOTHING
            """, (det_pk, "ViT-B/32", vec_str))

            # graph: observation
            obs_id = det_id
            cy(cur, f"""
                MATCH (k:Keyframe {{keyframe_id: {kf_idx}}})
                CREATE (k)-[:HAS_OBSERVATION]->(obs:Observation {{
                    obs_id: '{obs_id}',
                    class_name: '{cls}',
                    confidence: {conf}
                }})
                RETURN obs
            """)

            # object fusion (relational helper)
            cur.execute("""
                SELECT object_id, mean_x, mean_y, obs_count
                FROM objects
                WHERE class_name = %s
                  AND sqrt(power(mean_x - %s,2) + power(mean_y - %s,2)) < %s
                ORDER BY sqrt(power(mean_x - %s,2) + power(mean_y - %s,2))
                LIMIT 1
            """, (cls, x, y, FUSION_DIST, x, y))
            obj_row = cur.fetchone()

            if obj_row:
                oid, mx, my, cnt = obj_row
                nc = cnt + 1
                nmx = (mx * cnt + x) / nc
                nmy = (my * cnt + y) / nc
                cur.execute("""
                    UPDATE objects SET mean_x=%s, mean_y=%s, obs_count=%s, last_seen=now()
                    WHERE object_id=%s
                """, (nmx, nmy, nc, oid))
                cy(cur, f"""
                    MATCH (o:Object {{object_id: '{oid}'}})
                    SET o.mean_x = {nmx}, o.mean_y = {nmy}, o.obs_count = {nc}
                    RETURN o
                """)
            else:
                oid = f"obj_{uuid.uuid4().hex[:8]}"
                cur.execute("""
                    INSERT INTO objects (object_id, class_name, mean_x, mean_y)
                    VALUES (%s,%s,%s,%s)
                """, (oid, cls, x, y))
                cy(cur, f"""
                    CREATE (o:Object {{
                        object_id: '{oid}', class_name: '{cls}',
                        mean_x: {x}, mean_y: {y}, obs_count: 1
                    }}) RETURN o
                """)
                # assign to place
                pid, gx, gy = place_id(x, y)
                cy(cur, f"MERGE (p:Place {{place_id: '{pid}', gx: {gx}, gy: {gy}}}) RETURN p")
                # adjacency
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    npid = f"place_{gx+dx}_{gy+dy}"
                    cy(cur, f"""
                        MATCH (a:Place {{place_id: '{pid}'}}), (b:Place {{place_id: '{npid}'}})
                        MERGE (a)-[:ADJACENT_TO]->(b)
                        MERGE (b)-[:ADJACENT_TO]->(a)
                        RETURN a
                    """)
                cy(cur, f"""
                    MATCH (o:Object {{object_id: '{oid}'}}), (p:Place {{place_id: '{pid}'}})
                    CREATE (o)-[:IN_PLACE]->(p)
                    RETURN o
                """)

            # link observation to object
            cy(cur, f"""
                MATCH (obs:Observation {{obs_id: '{obs_id}'}}),
                      (o:Object {{object_id: '{oid}'}})
                CREATE (obs)-[:OF_OBJECT]->(o)
                RETURN o
            """)

        conn.commit()
        pid_str, _, _ = place_id(x, y)
        print(f"  KF {kf_idx:2d}  pos=({x:.1f},{y:.1f})  dets={len(det_list):2d}  "
              f"place={pid_str}")

    # summary
    cur.execute("SELECT count(*) FROM detection_events")
    ev_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM detections")
    det_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM detection_embeddings")
    emb_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM objects")
    obj_count = cur.fetchone()[0]

    print(f"\nDone!  events={ev_count}  detections={det_count}  "
          f"embeddings={emb_count}  objects={obj_count}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
