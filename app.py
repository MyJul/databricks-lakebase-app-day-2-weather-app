"""
Databricks Weather Intelligence App
- Serves a Flask REST API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls weather alerts from the National Weather Service API
- Stores normalized weather documents in weather_news
- Performs semantic search against weather_news_chunk_embeddings

Run locally:
    python app.py

Deploy as a Databricks App using app.yaml.
"""

import json
import logging
import os
from functools import lru_cache

from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

import lakebase
from weather_client import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)
_w = WorkspaceClient()

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_news")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME","weather_embeddings")
WEATHER_CHUNK_TABLE_NAME = os.environ.get("WEATHER_CHUNK_TABLE_NAME", "weather_news_chunk_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DEFAULT_LOCATIONS = [
    location.strip()
    for location in os.environ.get(
        "WEATHER_LOCATIONS",
        "Baltimore, MD,Washington, DC",
    ).split(",")
    if location.strip()
]

# Maximum number of search results allowed by the API.
MAX_SEARCH_RESULTS = 20


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})

@app.errorhandler(Exception)
def handle_exception(err):
    """
    Ensure all unhandled errors return JSON instead of an HTML error page.
    """
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code

@lru_cache(maxsize=1)
def _get_embedding_model():
    """
    Load the embedding model once and reuse it.

    This is intentionally cached because loading SentenceTransformer is
    expensive and should not happen on every /weather/search request.
    """
    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL_NAME,
    )

    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@app.route("/")
def index():
    """
    Render the weather application UI.
    """
    return render_template("index.html")

@app.route("/weather", methods=["GET"])
def list_weather():
    """
    Return weather documents already stored in Lakebase.

    Optional query parameters:
        limit=50
        location=Baltimore, MD
        source_type=alert
    """

    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    limit = max(1, min(limit, 100))

    location = request.args.get("location")
    source_type = request.args.get("source_type")
    where_clauses = []
    params = []
    if location:
        where_clauses.append("location = %s")
        params.append(location)
    if source_type:
        where_clauses.append("source_type = %s")
        params.append(source_type)
    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)
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
            effective_at,
            onset_at,
            expires_at,
            sender_name,
            area_desc,
            synced_at
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
    Fetch weather alerts from the National Weather Service and upsert
    normalized documents into weather_news.
    Request body:
    {
        "locations": [
            "Baltimore, MD",
            "Washington, DC"
        ],
        "limit": 50
    }
    The location may also be supplied as a latitude/longitude pair
    depending on the WeatherClient implementation.
    """
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 400
    body = request.json or {}
    locations = body.get("locations") or DEFAULT_LOCATIONS
    if not isinstance(locations, list) or not locations:
        return jsonify({
            "error": "locations must be a non-empty list"
        }), 400
    try:
        limit = int(body.get("limit", 50))
    except (TypeError, ValueError):
        return jsonify({
            "error": "limit must be an integer"
        }), 400
    limit = max(1, min(limit, 100))
    client = WeatherClient()
    total_synced = 0
    locations_processed = []

    for location in locations:
        if not isinstance(location, str) or not location.strip():
            continue
        location = location.strip()
        try:
            documents = client.get_weather_documents(
                location=location,
                limit=limit,
            )
            synced = _upsert_weather_batch(documents)
            total_synced += synced
            locations_processed.append({
                "location": location,
                "documents": synced,
            })
        except Exception:
            logger.exception(
                "Failed to sync weather for location: %s",
                location,
            )
            locations_processed.append({
                "location": location,
                "documents": 0,
                "error": "Failed to retrieve weather data",
            })
    return jsonify({
        "synced": total_synced,
        "locations": locations_processed,
    })

@app.route("/weather/search", methods=["POST"])
def weather_search():
    """
    Perform semantic search over weather narrative chunks.
    Request body:
    {
        "query": "flash flood risk this weekend",
        "top_k": 5
    }
    Returns the most semantically relevant weather documents/chunks.
    """
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 400
    body = request.json or {}
    query = body.get("query", "")
    if not isinstance(query, str):
        return jsonify({
            "error": "query must be a string"
        }), 400
    query = query.strip()
    if not query:
        return jsonify({
            "error": "Query text is required"
        }), 400
    try:
        top_k = int(body.get("top_k", 5))
    except (TypeError, ValueError):
        return jsonify({
            "error": "top_k must be an integer"
        }), 400
         
    top_k = max(1, min(top_k, MAX_SEARCH_RESULTS))

    try:
        model = _get_embedding_model()
        query_vector = model.encode(query).tolist()
        vector_str = (
            "["
            + ",".join(str(float(value)) for value in query_vector)
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

            FROM {WEATHER_CHUNK_TABLE_NAME} e
            JOIN {WEATHER_TABLE_NAME} d
                ON e.document_id = d.id
            ORDER BY
                e.embedding <=> %s::vector
            LIMIT %s
            """,
            (   vector_str,
                vector_str,
                top_k,
            ),
        )

        return jsonify({
            "query": query,
            "count": len(results),
            "results": results,
        })

    except Exception as exc:
        logger.exception("Error during weather semantic search")
        return jsonify({
            "error": str(exc)
        }), 500

def _upsert_weather_batch(documents: list[dict]) -> int:
    """
    Upsert normalized NWS weather documents into weather_news.
    The NWS alert ID is used as the primary key.
    This makes synchronization idempotent:
        same NWS alert ID -> update existing row
        new NWS alert ID  -> insert new row
    """
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for document in documents:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_TABLE_NAME} (
                        id,
                        location,
                        source_type,
                        event,
                        headline,
                        narrative_text,
                        description,
                        instruction,
                        severity,
                        certainty,
                        urgency,
                        effective_at,
                        onset_at,
                        expires_at,
                        sender_name,
                        sender_id,
                        area_desc,
                        geocode,
                        geometry,
                        payload,
                        synced_at,
                        updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        now(), now()
                    )

                    ON CONFLICT (id) DO UPDATE
                    SET
                        location = EXCLUDED.location,
                        source_type = EXCLUDED.source_type,
                        event = EXCLUDED.event,
                        headline = EXCLUDED.headline,
                        narrative_text = EXCLUDED.narrative_text,
                        description = EXCLUDED.description,
                        instruction = EXCLUDED.instruction,
                        severity = EXCLUDED.severity,
                        certainty = EXCLUDED.certainty,
                        urgency = EXCLUDED.urgency,
                        effective_at = EXCLUDED.effective_at,
                        onset_at = EXCLUDED.onset_at,
                        expires_at = EXCLUDED.expires_at,
                        sender_name = EXCLUDED.sender_name,
                        sender_id = EXCLUDED.sender_id,
                        area_desc = EXCLUDED.area_desc,
                        geocode = EXCLUDED.geocode,
                        geometry = EXCLUDED.geometry,
                        payload = EXCLUDED.payload,
                        synced_at = now(),
                        updated_at = now()
                    """,
                    (
                        document["id"],
                        document["location"],
                        document["source_type"],
                        document.get("event"),
                        document.get("headline"),
                        document["narrative_text"],
                        document.get("description"),
                        document.get("instruction"),
                        document.get("severity"),
                        document.get("certainty"),
                        document.get("urgency"),
                        document.get("effective_at"),
                        document.get("onset_at"),
                        document.get("expires_at"),
                        document.get("sender_name"),
                        document.get("sender_id"),
                        document.get("area_desc"),
                        json.dumps(document.get("geocode")),
                        json.dumps(document.get("geometry")),
                        json.dumps(document["payload"]),
                    ),
                )

                count += 1

        conn.commit()

    return count

if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))

    logger.info(
        "Starting weather app on http://%s:%s",
        host,
        port,
    )

    app.run(
        debug=True,
        host=host,
        port=port,
    );
