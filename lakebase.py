import base64
import os
from contextlib import contextmanager
from urllib.parse import unquote, urlparse

import pg8000.dbapi
from databricks.sdk import WorkspaceClient


_w = WorkspaceClient()

_SECRET_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_SECRET_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")


def _lakebase_url() -> str:
    """Read and decode the Lakebase PostgreSQL URL from Databricks Secrets."""
    secret = _w.secrets.get_secret(
        scope=_SECRET_SCOPE,
        key=_SECRET_KEY,
    )

    if not secret.value:
        raise RuntimeError(
            f"Lakebase secret {_SECRET_SCOPE}/{_SECRET_KEY} is empty."
        )

    return base64.b64decode(secret.value).decode("utf-8")


def _connection_settings() -> dict:
    """Parse the PostgreSQL URL into pg8000 connection settings."""
    parsed = urlparse(_lakebase_url())

    if not parsed.hostname:
        raise RuntimeError("Lakebase URL is missing a hostname.")
    if not parsed.username:
        raise RuntimeError("Lakebase URL is missing a username.")
    if parsed.password is None:
        raise RuntimeError("Lakebase URL is missing a password.")
    if not parsed.path or parsed.path == "/":
        raise RuntimeError("Lakebase URL is missing a database name.")

    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/"),
        "user": unquote(parsed.username),
        "password": unquote(parsed.password),
        "ssl_context": True,
    }


@contextmanager
def get_connection():
    """Yield a pg8000 DB-API connection and always close it."""
    conn = pg8000.dbapi.connect(**_connection_settings())

    try:
        yield conn
    finally:
        conn.close()


def run_query(
    sql: str,
    params: tuple | list | None = None,
) -> list[dict]:
    """Run a SELECT query and return rows as dictionaries."""
    with get_connection() as conn:
        cur = conn.cursor()

        try:
            cur.execute(sql, params or ())
            rows = cur.fetchall()

            if not cur.description:
                return []

            column_names = [
                column[0]
                for column in cur.description
            ]

            return [
                dict(zip(column_names, row))
                for row in rows
            ]
        finally:
            cur.close()


def run_write(
    sql: str,
    params: tuple | list | None = None,
) -> int:
    """Run one INSERT, UPDATE, or DELETE and return the affected row count."""
    with get_connection() as conn:
        cur = conn.cursor()

        try:
            cur.execute(sql, params or ())
            conn.commit()
            return cur.rowcount

        except Exception:
            conn.rollback()
            raise

        finally:
            cur.close()


def run_many(
    sql: str,
    param_sets: list[tuple],
) -> int:
    """Run the same write statement for many parameter sets."""
    if not param_sets:
        return 0

    with get_connection() as conn:
        cur = conn.cursor()

        try:
            cur.executemany(sql, param_sets)
            conn.commit()
            return cur.rowcount

        except Exception:
            conn.rollback()
            raise

        finally:
            cur.close()
