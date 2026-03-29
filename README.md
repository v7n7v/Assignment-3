# Assignment 3 — Semantic Graph + Re-Localization

Extends the A2 detection pipeline with CLIP embeddings (pgvector) and a property graph (Apache AGE) to do semantic re-localization.

## How it works

The detector runs YOLO + CLIP on camera frames, applies keyframe gating (skip if robot hasn't moved enough), and publishes detections with 512-d embeddings to Zenoh. The ingest worker picks those up, stores everything in Postgres (relational tables + pgvector + AGE graph). Then the re-localizer queries the graph to figure out where the robot is based on what it sees.

## Setup

```bash
# start postgres with pgvector + AGE
docker-compose up -d --build

# seed with test data (skip this if running the real sim)
pip install psycopg2-binary numpy
python seed_data.py

# run re-localization
python relocalize.py
python relocalize.py couch tv    # or pick specific objects
```

For the live pipeline with Gazebo:
```bash
pip install -r requirements.txt
python detector.py    # terminal 1
python ingest.py      # terminal 2
```

## Schema

**Relational** (carried over from A2):
- `detection_events` — keyframe metadata + pose
- `detections` — individual YOLO bounding boxes

**pgvector**:
- `detection_embeddings` — 512-dim CLIP vectors, IVFFlat indexed

**Apache AGE graph** (`semantic_map`):
- Nodes: Run, Keyframe, Pose, Place, Object, Observation
- Edges: HAS_KEYFRAME, AT_POSE, HAS_OBSERVATION, OF_OBJECT, IN_PLACE, ADJACENT_TO

## Queries

KNN similarity search:
```sql
SELECT d.det_pk, d.class_name, d.confidence,
       de.embedding <=> '[...]'::vector AS cosine_dist
FROM detection_embeddings de
JOIN detections d ON d.det_pk = de.det_pk
ORDER BY de.embedding <=> '[...]'::vector
LIMIT 5;
```

Objects in a place:
```sql
SELECT * FROM cypher('semantic_map', $$
    MATCH (p:Place {place_id: 'place_0_0'})<-[:IN_PLACE]-(o:Object)
    RETURN o.class_name, o.mean_x, o.mean_y, o.obs_count
$$) AS (class agtype, x agtype, y agtype, cnt agtype);
```

Adjacent places:
```sql
SELECT * FROM cypher('semantic_map', $$
    MATCH (a:Place {place_id: 'place_0_0'})-[:ADJACENT_TO]->(b:Place)
    RETURN b.place_id
$$) AS (pid agtype);
```

The full re-localization pipeline (KNN -> graph join -> place ranking) is implemented in `relocalize.py`.

## Place construction

Simple grid binning — each (x,y) maps to `place_{floor(x)}_{floor(y)}` with 1m cells. Neighbors get ADJACENT_TO edges.

## Object fusion

When we see an object, check if same class exists within 1m. If yes, merge (running average of position). If no, create a new Object node and link it to the grid place.

## What works and what doesn't

KNN does a good job matching same-class objects since CLIP embeddings are semantically meaningful. Grid places give decent spatial grouping. Re-localization works well when the query objects are distinctive to a specific area.

Main issues: grid cells are coarse so objects near boundaries can end up in the wrong place. Object fusion only checks distance, doesn't account for viewing angle. And if the same object is seen from far away in two different runs, it might not get merged.

## Files

- `Dockerfile.db` — postgres + pgvector + AGE
- `docker-compose.yml` — runs the db
- `schema.sql` — all tables + extensions
- `detector.py` — YOLO + CLIP detector with keyframe gating
- `ingest.py` — zenoh subscriber, writes to db + builds graph
- `seed_data.py` — populates db with fake data for testing
- `relocalize.py` — re-localization demo
