import sqlite3

import pytest

from orchestrator import db

TABLES = {
    "teams", "scans", "meshes", "semantic_objects", "twins", "task_graphs",
    "demonstrations", "policies", "artifacts", "robots", "deployments",
    "sync_events", "telemetry",
}


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


def test_all_tables_exist(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert TABLES <= {r["name"] for r in rows}


def test_insert_get_roundtrip(conn):
    scan_id = db.insert(conn, "scans", device="oneplus15", status="uploading")
    row = db.get(conn, "scans", scan_id)
    assert row["device"] == "oneplus15"
    assert row["status"] == "uploading"
    assert row["created_at"]


def test_fk_violation_raises(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert(conn, "meshes", scan_id="nope", mode="fast")


def test_schema_tables_match_constant():
    assert db.TABLES == TABLES


def test_identifiers_are_rejected_not_interpolated(conn):
    """Table and column names cannot be bound as parameters, so they are guarded."""
    with pytest.raises(ValueError):
        db.get(conn, "scans; DROP TABLE scans", "x")
    with pytest.raises(ValueError):
        db.insert(conn, "nonexistent", device="x")
    with pytest.raises(ValueError):
        db.insert(conn, "scans", **{"device) VALUES ('x'); --": "x"})
    # the guard did not take the table down with it
    assert db.get(conn, "scans", "missing") is None


def test_task_graph_accepts_the_keyword_provider(conn):
    scan_id = db.insert(conn, "scans", device="test", status="complete")
    mesh_id = db.insert(conn, "meshes", scan_id=scan_id, mode="fast")
    twin_id = db.insert(conn, "twins", mesh_id=mesh_id)
    graph_id = db.insert(conn, "task_graphs", twin_id=twin_id, provider="keyword",
                         source_text="go to the table", lang="en", graph_json="{}")
    assert db.get(conn, "task_graphs", graph_id)["provider"] == "keyword"


def test_stale_task_graphs_schema_is_refused():
    """CREATE TABLE IF NOT EXISTS never alters a table, so an old database would reject
    every 'keyword' insert. Startup must say so instead of failing on the first /plan."""
    c = db.connect(":memory:")
    c.execute("CREATE TABLE task_graphs(id TEXT PRIMARY KEY, "
              "provider TEXT CHECK (provider IN ('sarvam','function_gemma')))")
    with pytest.raises(RuntimeError, match="database schema changed"):
        db.init_db(c)
    c.close()
