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
