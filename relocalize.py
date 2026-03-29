#!/usr/bin/env python3
"""
Semantic re-localization using pgvector KNN + AGE graph.
Run with: python relocalize.py [class1] [class2] ...
"""

import os
import sys

import numpy as np
import psycopg2

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "maze_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "postgres")

# same RNG seed + class bases as seed_data.py so our "query" embeddings
# are close (but not identical) to the stored ones
rng = np.random.RandomState(99)  # different seed so it's not exact match

CLASSES = [
    "cup", "chair", "book", "bottle", "dining table",
    "potted plant", "couch", "tv", "bed", "clock", "laptop",
]
# re-create the same class bases (seed=42 in seed_data.py)
_base_rng = np.random.RandomState(42)
CLASS_BASES = {}
for c in CLASSES:
    v = _base_rng.randn(512).astype(np.float32)
    v /= np.linalg.norm(v)
    CLASS_BASES[c] = v


def make_query_embedding(class_name):
    base = CLASS_BASES[class_name]
    noise = rng.randn(512).astype(np.float32) * 0.12  # a bit more noise
    v = base + noise
    v /= np.linalg.norm(v)
    return v


def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS,
    )


def init_age(cur):
    cur.execute("LOAD 'age';")
    cur.execute('SET search_path = ag_catalog, "$user", public;')


def cy(cur, q):
    cur.execute(f"SELECT * FROM cypher('semantic_map', $$ {q} $$) AS (v agtype);")
    return cur.fetchall()




def relocalize(conn, query_classes):
    cur = conn.cursor()
    init_age(cur)

    print("=" * 60)
    print("SEMANTIC RE-LOCALIZATION")
    print("=" * 60)
    print(f"Query objects: {query_classes}\n")

    all_knn_results = []

    for cls in query_classes:
        emb = make_query_embedding(cls)
        vec_str = "[" + ",".join(str(float(v)) for v in emb) + "]"

        # ---- Step 1: pgvector KNN search ----
        print(f"--- KNN search for '{cls}' ---")
        cur.execute("""
            SELECT d.det_pk, d.class_name, d.confidence,
                   de.embedding <=> %s::vector AS cosine_dist
            FROM detection_embeddings de
            JOIN detections d ON d.det_pk = de.det_pk
            ORDER BY de.embedding <=> %s::vector
            LIMIT 5
        """, (vec_str, vec_str))

        rows = cur.fetchall()
        for det_pk, cname, conf, dist in rows:
            print(f"  det_pk={det_pk}  class={cname}  conf={conf:.2f}  "
                  f"cos_dist={dist:.4f}")
            all_knn_results.append((det_pk, cname, dist))

        print()

    if not all_knn_results:
        print("No KNN results found. Is the database seeded?")
        return

    # ---- Step 2: Map KNN hits to places via graph ----
    print("--- Mapping detections -> objects -> places ---")
    place_scores = {}  # place_id -> list of distances

    for det_pk, cname, dist in all_knn_results:
        # get the event for this detection to find the keyframe_id
        cur.execute("""
            SELECT ev.event_id, ev.x, ev.y, ev.yaw, ev.sequence
            FROM detections d
            JOIN detection_events ev ON ev.event_id = d.event_id
            WHERE d.det_pk = %s
        """, (det_pk,))
        ev_row = cur.fetchone()
        if ev_row is None:
            continue
        _, ex, ey, eyaw, seq = ev_row

        # find the place for this position (grid bin)
        import math
        gx = int(math.floor(ex))
        gy = int(math.floor(ey))
        pid = f"place_{gx}_{gy}"

        if pid not in place_scores:
            place_scores[pid] = {"dists": [], "poses": []}
        place_scores[pid]["dists"].append(dist)
        place_scores[pid]["poses"].append((ex, ey, eyaw))

    # ---- Step 3: Rank places ----
    print("\n--- Top-3 candidate places ---")
    ranked = sorted(place_scores.items(), key=lambda x: np.mean(x[1]["dists"]))

    for i, (pid, info) in enumerate(ranked[:3]):
        avg_dist = np.mean(info["dists"])
        hits = len(info["dists"])
        # average pose as hypothesis
        px = np.mean([p[0] for p in info["poses"]])
        py = np.mean([p[1] for p in info["poses"]])
        pyaw = np.mean([p[2] for p in info["poses"]])

        print(f"  #{i+1}  {pid}")
        print(f"       avg_cosine_dist = {avg_dist:.4f}")
        print(f"       hits            = {hits}")
        print(f"       pose_hypothesis = ({px:.2f}, {py:.2f}, yaw={pyaw:.2f})")

        # also show what objects are in this place (graph query)
        try:
            rows = cy(cur, f"""
                MATCH (p:Place {{place_id: '{pid}'}})<-[:IN_PLACE]-(o:Object)
                RETURN o.class_name
            """)
            objs = [str(r[0]).strip('"') for r in rows]
            print(f"       known_objects   = {objs}")
        except Exception:
            pass

        print()

    # ---- Step 4: Graph neighborhood ----
    if ranked:
        best_pid = ranked[0][0]
        print(f"--- Adjacent places to best match ({best_pid}) ---")
        try:
            rows = cy(cur, f"""
                MATCH (a:Place {{place_id: '{best_pid}'}})-[:ADJACENT_TO]->(b:Place)
                RETURN b.place_id
            """)
            for r in rows:
                print(f"  -> {str(r[0]).strip('\"')}")
        except Exception:
            print("  (no adjacency data)")

    print("\n" + "=" * 60)
    print("BEST ESTIMATE:")
    if ranked:
        best = ranked[0]
        px = np.mean([p[0] for p in best[1]["poses"]])
        py = np.mean([p[1] for p in best[1]["poses"]])
        pyaw = np.mean([p[2] for p in best[1]["poses"]])
        print(f"  Place: {best[0]}")
        print(f"  Pose:  x={px:.2f}  y={py:.2f}  yaw={pyaw:.2f} rad")
    print("=" * 60)

    cur.close()


# --- demo queries ---

def run_demo_queries(conn):
    cur = conn.cursor()
    init_age(cur)

    print("\n" + "#" * 60)
    print("# REQUIRED QUERIES")
    print("#" * 60)

    # -- Vector query --
    print("\n[VECTOR] Top-5 nearest neighbours to a 'cup' embedding:")
    emb = make_query_embedding("cup")
    vec_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
    cur.execute("""
        SELECT d.det_pk, d.class_name, d.confidence,
               de.embedding <=> %s::vector AS cosine_dist
        FROM detection_embeddings de
        JOIN detections d ON d.det_pk = de.det_pk
        ORDER BY de.embedding <=> %s::vector
        LIMIT 5
    """, (vec_str, vec_str))
    for row in cur.fetchall():
        print(f"  {row}")

    # -- Graph query: objects in a place --
    print("\n[GRAPH] Objects in place_0_0:")
    try:
        rows = cy(cur, """
            MATCH (p:Place {place_id: 'place_0_0'})<-[:IN_PLACE]-(o:Object)
            RETURN o.class_name, o.mean_x, o.mean_y, o.obs_count
        """)
        for r in rows:
            print(f"  {r[0]}")
    except Exception as e:
        print(f"  (query failed: {e})")

    # -- Graph query: all places --
    print("\n[GRAPH] All places in the map:")
    try:
        rows = cy(cur, "MATCH (p:Place) RETURN p.place_id, p.gx, p.gy")
        for r in rows:
            print(f"  {r[0]}")
    except Exception as e:
        print(f"  (query failed: {e})")

    # -- Graph query: full path Run -> Keyframe -> Object -> Place --
    print("\n[GRAPH] Keyframe -> Observation -> Object chain (first 5):")
    try:
        rows = cy(cur, """
            MATCH (k:Keyframe)-[:HAS_OBSERVATION]->(obs:Observation)
                  -[:OF_OBJECT]->(o:Object)-[:IN_PLACE]->(p:Place)
            RETURN k.keyframe_id, obs.class_name, o.object_id, p.place_id
        """)
        # AGE returns multiple columns as single agtype, let's just print
        for r in rows[:5]:
            print(f"  {r[0]}")
    except Exception as e:
        print(f"  (query failed: {e})")

    cur.close()


def main():
    conn = connect()

    # determine query classes
    if len(sys.argv) > 1:
        query_classes = sys.argv[1:]
    else:
        # default demo: robot sees a cup and a chair
        query_classes = ["cup", "chair"]

    relocalize(conn, query_classes)
    run_demo_queries(conn)

    conn.close()


if __name__ == "__main__":
    main()
