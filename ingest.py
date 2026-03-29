#!/usr/bin/env python3
"""
Ingest worker - subscribes to Zenoh detection events and writes
to PostgreSQL (relational + pgvector + AGE graph).
"""

import json
import os
import time
import math
from datetime import datetime, timezone

import psycopg2
import zenoh

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "maze_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "postgres")

GRID_SIZE = 1.0  # place grid cell in meters
FUSION_DIST = 1.0  # max distance to merge same-class objects


def get_conn():
    for attempt in range(30):
        try:
            conn = psycopg2.connect(
                host=DB_HOST, port=DB_PORT,
                dbname=DB_NAME, user=DB_USER, password=DB_PASS,
            )
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError:
            print(f"Waiting for db... ({attempt+1})")
            time.sleep(2)
    raise RuntimeError("Could not connect")


def init_age(cur):
    cur.execute("LOAD 'age';")
    cur.execute('SET search_path = ag_catalog, "$user", public;')


def cypher(cur, query):
    sql = f"SELECT * FROM cypher('semantic_map', $$ {query} $$) AS (v agtype);"
    cur.execute(sql)
    return cur.fetchall()


def place_id_for(x, y):
    gx = int(math.floor(x / GRID_SIZE))
    gy = int(math.floor(y / GRID_SIZE))
    return f"place_{gx}_{gy}", gx, gy


def ensure_place(cur, pid, gx, gy):
    cypher(cur, f"MERGE (p:Place {{place_id: '{pid}', gx: {gx}, gy: {gy}}}) RETURN p")

    # adjacency to 4-connected neighbours
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = gx + dx, gy + dy
        npid = f"place_{nx}_{ny}"
        cypher(cur, f"""
            MATCH (a:Place {{place_id: '{pid}'}}), (b:Place {{place_id: '{npid}'}})
            MERGE (a)-[:ADJACENT_TO]->(b)
            MERGE (b)-[:ADJACENT_TO]->(a)
            RETURN a
        """)


def find_or_create_object(cur, class_name, x, y, det_pk):
    # Object fusion: merge if same class exists nearby, else create new.
    # Using a relational helper table because AGE's agtype parsing
    # is annoying for distance calculations.
    return _fuse_object_relational(cur, class_name, x, y, det_pk)


def _fuse_object_relational(cur, class_name, x, y, det_pk):

    # make sure helper table exists
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

    # find nearest same-class object
    cur.execute("""
        SELECT object_id, mean_x, mean_y, obs_count
        FROM objects
        WHERE class_name = %s
          AND sqrt(power(mean_x - %s, 2) + power(mean_y - %s, 2)) < %s
        ORDER BY sqrt(power(mean_x - %s, 2) + power(mean_y - %s, 2))
        LIMIT 1
    """, (class_name, x, y, FUSION_DIST, x, y))

    row = cur.fetchone()
    if row:
        oid, mx, my, cnt = row
        # running average for position
        new_cnt = cnt + 1
        new_mx = (mx * cnt + x) / new_cnt
        new_my = (my * cnt + y) / new_cnt
        cur.execute("""
            UPDATE objects
            SET mean_x = %s, mean_y = %s, obs_count = %s, last_seen = now()
            WHERE object_id = %s
        """, (new_mx, new_my, new_cnt, oid))

        # update graph node too
        cypher(cur, f"""
            MATCH (o:Object {{object_id: '{oid}'}})
            SET o.mean_x = {new_mx}, o.mean_y = {new_my}, o.obs_count = {new_cnt}
            RETURN o
        """)
        return oid
    else:
        import uuid
        oid = f"obj_{uuid.uuid4().hex[:8]}"
        cur.execute("""
            INSERT INTO objects (object_id, class_name, mean_x, mean_y)
            VALUES (%s, %s, %s, %s)
        """, (oid, class_name, x, y))

        # create graph node
        cypher(cur, f"""
            CREATE (o:Object {{
                object_id: '{oid}',
                class_name: '{class_name}',
                mean_x: {x},
                mean_y: {y},
                obs_count: 1
            }})
            RETURN o
        """)

        # assign to place
        pid, gx, gy = place_id_for(x, y)
        ensure_place(cur, pid, gx, gy)
        cypher(cur, f"""
            MATCH (o:Object {{object_id: '{oid}'}}), (p:Place {{place_id: '{pid}'}})
            CREATE (o)-[:IN_PLACE]->(p)
            RETURN o
        """)

        return oid


def insert_event(conn, event):
    cur = conn.cursor()
    init_age(cur)

    stamp = event["image"]["stamp"]
    ts = datetime.fromtimestamp(
        stamp["sec"] + stamp.get("nanosec", 0) / 1e9, tz=timezone.utc
    )

    try:
        # -- relational tables --
        cur.execute("""
            INSERT INTO detection_events
            (event_id, run_id, robot_id, sequence, stamp,
             image_frame_id, width, height, encoding,
             x, y, yaw, raw_event)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            event["event_id"], event["run_id"], event["robot_id"],
            event.get("keyframe_id", event.get("sequence", 0)), ts,
            event["image"].get("frame_id"),
            event["image"].get("width"), event["image"].get("height"),
            event["image"].get("encoding"),
            event["odometry"]["map_x"], event["odometry"]["map_y"],
            event["odometry"]["map_yaw"],
            json.dumps(event),
        ))

        kf_id = event.get("keyframe_id", event.get("sequence", 0))
        mx = event["odometry"]["map_x"]
        my = event["odometry"]["map_y"]
        myaw = event["odometry"]["map_yaw"]

        # -- graph: run + keyframe + pose --
        cypher(cur, f"MERGE (r:Run {{run_id: '{event['run_id']}'}}) RETURN r")

        cypher(cur, f"""
            MATCH (r:Run {{run_id: '{event['run_id']}'}})
            CREATE (r)-[:HAS_KEYFRAME]->(k:Keyframe {{
                keyframe_id: {kf_id},
                timestamp: '{ts.isoformat()}'
            }})
            RETURN k
        """)

        cypher(cur, f"""
            MATCH (k:Keyframe {{keyframe_id: {kf_id}}})
            CREATE (k)-[:AT_POSE]->(p:Pose {{
                map_x: {mx}, map_y: {my}, map_yaw: {myaw}
            }})
            RETURN p
        """)

        # -- per detection --
        for det in event.get("detections", []):
            bbox = det["bbox_xyxy"]
            cur.execute("""
                INSERT INTO detections
                (event_id, det_id, class_id, class_name, confidence,
                 x1, y1, x2, y2)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
                RETURNING det_pk
            """, (
                event["event_id"], det["det_id"],
                det["class_id"], det["class_name"], det["confidence"],
                bbox[0], bbox[1], bbox[2], bbox[3],
            ))

            row = cur.fetchone()
            if row is None:
                continue
            det_pk = row[0]

            # embedding
            if "embedding" in det:
                vec_str = "[" + ",".join(str(v) for v in det["embedding"]) + "]"
                cur.execute("""
                    INSERT INTO detection_embeddings (det_pk, model, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT DO NOTHING
                """, (det_pk, det.get("embedding_model", "ViT-B/32"), vec_str))

            # observation node
            obs_id = det["det_id"]
            cypher(cur, f"""
                MATCH (k:Keyframe {{keyframe_id: {kf_id}}})
                CREATE (k)-[:HAS_OBSERVATION]->(obs:Observation {{
                    obs_id: '{obs_id}',
                    class_name: '{det['class_name']}',
                    confidence: {det['confidence']}
                }})
                RETURN obs
            """)

            # object fusion
            oid = find_or_create_object(cur, det["class_name"], mx, my, det_pk)

            cypher(cur, f"""
                MATCH (obs:Observation {{obs_id: '{obs_id}'}}),
                      (o:Object {{object_id: '{oid}'}})
                CREATE (obs)-[:OF_OBJECT]->(o)
                RETURN o
            """)

        conn.commit()
        print(f"Ingested kf={kf_id}  dets={len(event.get('detections',[]))}  "
              f"pos=({mx:.1f},{my:.1f})")

    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")


def main():
    conn = get_conn()
    print("Connected to database")

    conf = zenoh.Config()
    session = zenoh.open(conf)
    print("Subscribed to maze/**/detections/v2/*")

    def on_sample(sample):
        try:
            event = json.loads(bytes(sample.payload))
            insert_event(conn, event)
        except Exception as e:
            print(f"Error processing: {e}")

    sub = session.declare_subscriber("maze/**/detections/v2/*", on_sample)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    sub.undeclare()
    session.close()
    conn.close()


if __name__ == "__main__":
    main()
