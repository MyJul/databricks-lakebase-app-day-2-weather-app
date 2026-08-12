"""
Databricks Weather Intelligence App.

Endpoints:
    GET  /healthz
    GET  /weather
    POST /weather/sync
    POST /weather/search

Data flow:
    NWS Alerts API
        -> weather_documents
        -> ingest_weather_embeddings.py
        -> weather_embeddings
        -> /weather/search
"""

import json
import logging
import os
from functools import lru_cache
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer
import lakebase
from weather_client import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")
app = Flask(__name__)

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_SEARCH_RESULTS = 20

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})

@app.errorhandler(Exception)
def handle_exception(err):
    """Return JSON for unhandled application errors."""
    logger.exception(
        "Unhandled exception while processing request"
    )
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({
        "error": str(err)
    }), status_code

@lru_cache(maxsize=1)
def _get_embedding_model():
    """Load the embedding model once and reuse it."""
    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL_NAME,
    )
    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

@app.route("/")
def index():
    """Render the app UI if templates/index.html exists."""
    return render_template("index.html")

@app.route("/weather", methods=["GET"])
def list_weather():
    """Return weather documents already stored in Lakebase."""
    try:
        limit = int(
            request.args.get("limit", 50)
        )
    except ValueError:
        return jsonify({
            "error": "limit must be an integer"
        }), 400
    limit = max(1, min(limit, 100))
    location = request.args.get("location")
    source_type = request.args.get("source_type")
    where_clauses = []
    params = []
    if location:
        where_clauses.append(
            "location = %s"
        )
        params.append(location)
    if source_type:
        where_clauses.append(
            "source_type = %s"
        )
        params.append(source_type)
    where_sql = ""
    if where_clauses:
        where_sql = (
            "WHERE "
            + " AND ".join(where_clauses)
        )
    params.append(limit)
    rows = lakebase.run_query(
        f"""
        SELECT
            id,
            location,
            source_type,
            event,
            headline,
            narrative_text,
            severity,
            certainty,
            urgency,
            issued_at,
            effective_at,
            onset_at,
            expires_at,
            sender_name,
            area_desc,
            synced_at,
            updated_at
        FROM {WEATHER_TABLE_NAME}
        {where_sql}
        ORDER BY effective_at DESC NULLS LAST
        LIMIT %s
        """,
        tuple(params),
    )
    return jsonify(rows)

@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Fetch active NWS alerts and upsert them into weather_documents.

    Example body:
    {
        "locations": ["Baltimore, MD", "Washington, DC"],
        "limit": 50
    }
    """
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 400
    body = request.get_json(silent=True) or {}
    locations = (
        body.get("locations")
     
    if (
        not isinstance(locations, list)
        or not locations
    ):
        return jsonify({
            "error": "locations must be a non-empty list"
        }), 400
    try:
        limit = int(
            body.get("limit", 50)
        )
    except (TypeError, ValueError):
        return jsonify({
            "error": "limit must be an integer"
        }), 400
    limit = max(1, min(limit, 100))
    client = WeatherClient()
    total_synced = 0
    locations_processed = []
    for location in locations:
        if (
            not isinstance(location, str)
            or not location.strip()
        ):
            continue
        location = location.strip()
        try:
            documents = (
                client.get_weather_documents(
                    location=location,
                    limit=limit,
                )
            )
            synced = _upsert_weather_batch(
                documents
            )

            total_synced += synced

            locations_processed.append({
                "location": location,
                "documents": synced,
            })
        except Exception as exc:
            logger.exception(
                "Failed to sync weather for %s",
                location,
            )
            locations_processed.append({
                "location": location,
                "documents": 0,
                "error": str(exc),
            })
    return jsonify({
        "synced": total_synced,
        "locations": locations_processed,
    })

@app.route("/weather/search", methods=["POST"])
def weather_search():
    """
    Semantic search over weather_embeddings.

    Example body:
    {
        "query": "flash flood risk this weekend",
        "top_k": 5
    }
    """
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 400

    body = request.get_json(silent=True) or {}
    query = body.get("query", "")
    if not isinstance(query, str):
        return jsonify({
            "error": "query must be a string"
        }), 400
    query = query.strip()
    if not query:
        return jsonify({
            "error": "query is required"
        }), 400
    try:
        top_k = int(
            body.get("top_k", 5)
        )
    except (TypeError, ValueError):
        return jsonify({
            "error": "top_k must be an integer"
        }), 400
    top_k = max(
        1,
        min(top_k, MAX_SEARCH_RESULTS),
    )
    count_rows = lakebase.run_query(
        f"""
        SELECT COUNT(*) AS embedding_count
        FROM {EMBEDDINGS_TABLE_NAME}
        """
    )
    embedding_count = (
        count_rows[0]["embedding_count"]
        if count_rows
        else 0
    )
    if embedding_count == 0:
        return jsonify({
            "query": query,
            "count": 0,
            "results": [],
            "message": (
                "No weather embeddings are available. "
                "Run the embedding pipeline first."
            ),
        })
    model = _get_embedding_model()
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()
    vector_string = (
        "["
        + ",".join(
            str(float(value))
            for value in query_vector
        )
        + "]"
    )
    results = lakebase.run_query(
        f"""
        SELECT
            e.id AS chunk_id,
            e.document_id,
            e.chunk_index,
            e.chunk_text,
            e.model_name,
            d.location,
            d.source_type,
            d.event,
            d.headline,
            d.severity,
            d.certainty,
            d.urgency,
            d.effective_at,
            d.onset_at,
            d.expires_at,
            1 - (
                e.embedding <=> %s::vector
            ) AS similarity
        FROM {EMBEDDINGS_TABLE_NAME} e
        JOIN {WEATHER_TABLE_NAME} d
            ON d.id = e.document_id
        ORDER BY
            e.embedding <=> %s::vector
        LIMIT %s
        """,
        (
            vector_string,
            vector_string,
            top_k,
        ),
    )
    for result in results:
        if result.get("similarity") is not None:
            result["similarity"] = float(
                result["similarity"]
            )
    return jsonify({
        "query": query,
        "count": len(results),
        "results": results,
    })

def _upsert_weather_batch(
    documents: list[dict],
) -> int:
    """Upsert normalized NWS alerts into weather_documents."""
    if not documents:
        return 0
    sql = f"""
        INSERT INTO {WEATHER_TABLE_NAME} (
            id,
            location,
            source_type,
            event,
            headline,
            narrative_text,
            severity,
            certainty,
            urgency,
            issued_at,
            effective_at,
            onset_at,
            expires_at,
            sender_name,
            area_desc,
            payload,
            synced_at,
            updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s::jsonb, NOW(), NOW()
        )
        ON CONFLICT (id) DO UPDATE
        SET
            location = EXCLUDED.location,
            source_type = EXCLUDED.source_type,
            event = EXCLUDED.event,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            severity = EXCLUDED.severity,
            certainty = EXCLUDED.certainty,
            urgency = EXCLUDED.urgency,
            issued_at = EXCLUDED.issued_at,
            effective_at = EXCLUDED.effective_at,
            onset_at = EXCLUDED.onset_at,
            expires_at = EXCLUDED.expires_at,
            sender_name = EXCLUDED.sender_name,
            area_desc = EXCLUDED.area_desc,
            payload = EXCLUDED.payload,
            synced_at = NOW(),
            updated_at = CASE
                WHEN {WEATHER_TABLE_NAME}.narrative_text
                     IS DISTINCT FROM EXCLUDED.narrative_text
                THEN NOW()
                ELSE {WEATHER_TABLE_NAME}.updated_at
            END
    """
    rows = []
    for document in documents:
        rows.append((
            document["id"],
            document["location"],
            document["source_type"],
            document.get("event"),
            document.get("headline"),
            document["narrative_text"],
            document.get("severity"),
            document.get("certainty"),
            document.get("urgency"),
            document.get("issued_at"),
            document.get("effective_at"),
            document.get("onset_at"),
            document.get("expires_at"),
            document.get("sender_name"),
            document.get("area_desc"),
            json.dumps(document["payload"]),
        ))
    lakebase.run_many(
        sql,
        rows,
    )
    return len(rows)

if __name__ == "__main__":
    host = os.getenv(
        "FLASK_RUN_HOST",
        "0.0.0.0",
    )

    port = int(
        os.getenv(
            "FLASK_RUN_PORT",
            "8000",
        )
    )
    logger.info(
        "Starting weather app on http://%s:%s",
        host,
        port,
    )
    app.run(
        debug=False,
        host=host,
        port=port,
    )
