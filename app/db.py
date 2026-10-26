"""Prediction/request log store. SQLite by default, PostgreSQL when DATABASE_URL says so.

Every prediction is written as one structured row (spec 3.6): the monitoring dashboard
reads these rows, so without this table there is no monitoring. A separate `requests`
table records every call (endpoint, status, latency) for the operational metrics, and
an `outcomes` table holds ground-truth labels when they arrive 90 days later.
"""
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY {auto},
        ts TEXT NOT NULL,
        request_id TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        model_version TEXT NOT NULL,
        git_sha TEXT NOT NULL,
        features_json TEXT NOT NULL,
        probability REAL NOT NULL,
        decision TEXT NOT NULL,
        latency_ms REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY {auto},
        ts TEXT NOT NULL,
        request_id TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        status_code INTEGER NOT NULL,
        latency_ms REAL NOT NULL,
        meta_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY {auto},
        ts TEXT NOT NULL,
        key TEXT NOT NULL,
        severity TEXT NOT NULL,
        message TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS outcomes (
        request_id TEXT PRIMARY KEY,
        observed_retained INTEGER NOT NULL,
        observed_at TEXT NOT NULL
    )""",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LogStore:
    def __init__(self, url: str):
        self.url = url
        self.is_pg = url.startswith("postgres")
        self._lock = threading.Lock()
        self._conn: Any  # psycopg or sqlite3 connection, depending on the URL
        if self.is_pg:
            import psycopg  # installed in the container image; not needed for SQLite use

            self._conn = psycopg.connect(url, autocommit=True)
            schema = [s.replace("INTEGER PRIMARY KEY {auto}", "INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY") for s in SCHEMA]
        else:
            path = url.replace("sqlite:///", "", 1)
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            schema = [s.replace("{auto}", "AUTOINCREMENT") for s in SCHEMA]
        for stmt in schema:
            self._exec(stmt)

    # -- low level -----------------------------------------------------------
    def _q(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.is_pg else sql

    def _exec(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock:
            if self.is_pg:
                with self._conn.cursor() as cur:
                    cur.execute(self._q(sql), params)
            else:
                self._conn.execute(sql, params)
                self._conn.commit()

    def _all(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            if self.is_pg:
                with self._conn.cursor() as cur:
                    cur.execute(self._q(sql), params)
                    cols = [d.name for d in (cur.description or [])]
                    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def ping(self) -> bool:
        try:
            self._all("SELECT 1 AS ok")
            return True
        except Exception:
            return False

    # -- writes --------------------------------------------------------------
    def log_prediction(self, request_id: str, endpoint: str, model_version: str, git_sha: str,
                       features: Dict[str, Any], probability: float, decision: str, latency_ms: float) -> None:
        self._exec(
            "INSERT INTO predictions (ts, request_id, endpoint, model_version, git_sha, features_json,"
            " probability, decision, latency_ms) VALUES (?,?,?,?,?,?,?,?,?)",
            (utcnow(), request_id, endpoint, model_version, git_sha,
             json.dumps(features, default=str, sort_keys=True), float(probability), decision, float(latency_ms)),
        )

    def log_request(self, request_id: str, endpoint: str, status_code: int, latency_ms: float,
                    meta: Optional[Dict[str, Any]] = None) -> None:
        self._exec(
            "INSERT INTO requests (ts, request_id, endpoint, status_code, latency_ms, meta_json) VALUES (?,?,?,?,?,?)",
            (utcnow(), request_id, endpoint, int(status_code), float(latency_ms),
             json.dumps(meta) if meta else None),
        )

    def log_alert(self, key: str, severity: str, message: str) -> None:
        self._exec("INSERT INTO alerts (ts, key, severity, message) VALUES (?,?,?,?)",
                   (utcnow(), key, severity, message))

    def record_outcome(self, request_id: str, observed_retained: int) -> None:
        self._exec("DELETE FROM outcomes WHERE request_id = ?", (request_id,))
        self._exec("INSERT INTO outcomes (request_id, observed_retained, observed_at) VALUES (?,?,?)",
                   (request_id, int(observed_retained), utcnow()))

    def purge_older_than(self, days: float) -> Dict[str, int]:
        """Retention policy (Law 25 s.23: destroy when the purpose is achieved). Prediction rows hold customer
        feature values, so they are deleted after `days` (default 120 = 90-day label delay + 30 days grace);
        request metadata follows the same limit. Labels carry no features and are kept for accuracy history."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        counts = {}
        for table in ("predictions", "requests"):
            n = self._all(f"SELECT COUNT(*) AS n FROM {table} WHERE ts < ?", (cutoff,))[0]["n"]
            self._exec(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
            counts[table] = int(n)
        return counts

    # -- reads ---------------------------------------------------------------
    @staticmethod
    def _since(hours: float) -> str:
        return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")

    def recent_predictions(self, hours: float = 24, limit: int = 5000) -> List[Dict[str, Any]]:
        return self._all(
            "SELECT * FROM predictions WHERE ts >= ? ORDER BY id DESC LIMIT ?", (self._since(hours), limit))

    def recent_requests(self, hours: float = 24, limit: int = 20000) -> List[Dict[str, Any]]:
        return self._all(
            "SELECT * FROM requests WHERE ts >= ? ORDER BY id DESC LIMIT ?", (self._since(hours), limit))

    def recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))

    def alert_exists_since(self, key: str, hours: float) -> bool:
        return bool(self._all("SELECT 1 AS x FROM alerts WHERE key = ? AND ts >= ? LIMIT 1",
                              (key, self._since(hours))))

    def first_prediction_ts(self) -> Optional[str]:
        rows = self._all("SELECT MIN(ts) AS t FROM predictions")
        return rows[0]["t"] if rows and rows[0]["t"] else None

    def labelled_pairs(self) -> List[Dict[str, Any]]:
        return self._all(
            "SELECT p.request_id, p.probability, p.decision, o.observed_retained FROM predictions p"
            " JOIN outcomes o ON o.request_id = p.request_id")
