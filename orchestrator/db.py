"""Shared metadata store (blueprint section 17.2), SQLite."""
import os
import re
import sqlite3
import uuid

DB_PATH = os.environ.get("DRAGVERSE_DB", "data/dragverse.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams(
    id TEXT PRIMARY KEY, name TEXT, members_json TEXT,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS scans(
    id TEXT PRIMARY KEY, team_id TEXT REFERENCES teams(id),
    device TEXT, capture_frame_json TEXT,
    created_at TEXT DEFAULT (datetime('now')), status TEXT);
CREATE TABLE IF NOT EXISTS meshes(
    id TEXT PRIMARY KEY, scan_id TEXT NOT NULL REFERENCES scans(id),
    mode TEXT CHECK (mode IN ('fast','fidelity')),
    glb_url TEXT, ply_url TEXT,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS semantic_objects(
    id TEXT PRIMARY KEY, mesh_id TEXT NOT NULL REFERENCES meshes(id),
    label TEXT, bbox3d_json TEXT, mask_url TEXT, confidence REAL,
    embedding_id TEXT,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS twins(
    id TEXT PRIMARY KEY, mesh_id TEXT NOT NULL REFERENCES meshes(id),
    unity_scene_url TEXT, navmesh_url TEXT, anchor_transform_json TEXT,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS task_graphs(
    id TEXT PRIMARY KEY, twin_id TEXT NOT NULL REFERENCES twins(id),
    source_text TEXT, lang TEXT,
    provider TEXT CHECK (provider IN ('sarvam','function_gemma','keyword')),
    graph_json TEXT,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS demonstrations(
    id TEXT PRIMARY KEY, twin_id TEXT NOT NULL REFERENCES twins(id),
    task_graph_id TEXT REFERENCES task_graphs(id),
    trajectory_url TEXT, recorded_at TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS policies(
    id TEXT PRIMARY KEY, twin_id TEXT NOT NULL REFERENCES twins(id),
    task_graph_id TEXT REFERENCES task_graphs(id),
    base_checkpoint TEXT, finetuned_ckpt_url TEXT, sim_success_rate REAL,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS artifacts(
    id TEXT PRIMARY KEY, policy_id TEXT NOT NULL REFERENCES policies(id),
    device_label TEXT, precision TEXT, op_coverage_pct REAL,
    est_latency_ms REAL, artifact_url TEXT,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS robots(
    id TEXT PRIMARY KEY, team_id TEXT REFERENCES teams(id),
    kind TEXT CHECK (kind IN ('sim','unoq')), config_json TEXT,
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS deployments(
    id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    robot_id TEXT NOT NULL REFERENCES robots(id), status TEXT,
    deployed_at TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS sync_events(
    id TEXT PRIMARY KEY, twin_id TEXT NOT NULL REFERENCES twins(id),
    new_scan_id TEXT REFERENCES scans(id), diff_summary_json TEXT,
    triggered_by TEXT, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS telemetry(
    id TEXT PRIMARY KEY, deployment_id TEXT REFERENCES deployments(id),
    kind TEXT CHECK (kind IN ('image','log','failure')), payload_url TEXT,
    created_at TEXT DEFAULT (datetime('now')));
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or DB_PATH
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    # CREATE TABLE IF NOT EXISTS never alters an existing table, so a database made before
    # 'keyword' was a valid provider would reject every /plan insert. Refuse it up front.
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'task_graphs'"
    ).fetchone()
    if row and "'keyword'" not in row[0]:
        raise RuntimeError(f"database schema changed: delete {DB_PATH} and restart")
    conn.executescript(SCHEMA)
    conn.commit()


def new_id() -> str:
    return uuid.uuid4().hex


# Table and column names cannot be bound as SQL parameters, so they are
# interpolated. Every identifier is checked against the schema below first —
# callers pass literals today, but this keeps a future caller from routing a
# request field into an identifier position.
TABLES = frozenset(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA))
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_table(table: str) -> str:
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    return table


def _check_columns(cols) -> None:
    bad = [c for c in cols if not _IDENTIFIER.match(c)]
    if bad:
        raise ValueError(f"invalid column name(s): {bad}")


def insert(conn: sqlite3.Connection, table: str, **cols) -> str:
    row_id = cols.pop("id", None) or new_id()
    cols["id"] = row_id
    _check_table(table)
    _check_columns(cols)
    keys = ", ".join(cols)
    marks = ", ".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO {table}({keys}) VALUES ({marks})",  # nosec B608 - identifiers checked
        list(cols.values()),
    )
    conn.commit()
    return row_id


def get(conn: sqlite3.Connection, table: str, row_id: str):
    row = conn.execute(
        f"SELECT * FROM {_check_table(table)} WHERE id = ?",  # nosec B608 - table checked
        (row_id,),
    ).fetchone()
    return dict(row) if row else None
