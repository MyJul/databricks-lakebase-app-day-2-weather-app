from pathlib import Path

notebook = r'''# Databricks notebook source

# MAGIC %md
# MAGIC # Ingest Weather Documents → Vector Embeddings
# MAGIC
# MAGIC This notebook reads weather documents from Lakebase, chunks the
# MAGIC `narrative_text`, generates embeddings, and stores the vectors in
# MAGIC `weather_embeddings`.
# MAGIC
# MAGIC Pipeline:
# MAGIC
# MAGIC `weather_documents`
# MAGIC → find new or updated documents
# MAGIC → chunk `narrative_text`
# MAGIC → `sentence-transformers/all-MiniLM-L6-v2`
# MAGIC → `VECTOR(384)`
# MAGIC → `weather_embeddings`
# MAGIC
# MAGIC This notebook intentionally uses **pg8000**.
# MAGIC It does not use psycopg2 or Spark JDBC.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Packages
# MAGIC
# MAGIC Install the pure-Python PostgreSQL driver and sentence-transformers.

# COMMAND ----------

# MAGIC %pip install -q pg8000 sentence-transformers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import base64
from urllib.parse import unquote, urlparse

import pg8000.dbapi
from sentence_transformers import SentenceTransformer

# Use the same %s placeholders already used throughout this notebook.
pg8000.dbapi.paramstyle = "format"

print("Imports loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Configure the Lakebase tables, Databricks secret, embedding model,
# MAGIC and text chunking parameters.

# COMMAND ----------

dbutils.widgets.text(
    "weather_table_name",
    "weather_documents",
    "Source table (weather documents)"
)

dbutils.widgets.text(
    "embeddings_table_name",
    "weather_embeddings",
    "Destination table (weather vectors)"
)

dbutils.widgets.text(
    "embedding_model",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Embedding model"
)

dbutils.widgets.text(
    "database_secret_scope",
    "database",
    "Lakebase secret scope"
)

dbutils.widgets.text(
    "database_secret_key",
    "lakebase-url",
    "Lakebase secret key"
)

dbutils.widgets.text(
    "chunk_size",
    "800",
    "Weather narrative chunk size (chars)"
)

dbutils.widgets.text(
    "chunk_overlap",
    "100",
    "Weather narrative chunk overlap (chars)"
)

dbutils.widgets.text(
    "batch_size",
    "50",
    "Embedding batch size"
)

WEATHER_TABLE_NAME = dbutils.widgets.get(
    "weather_table_name"
)

EMBEDDINGS_TABLE_NAME = dbutils.widgets.get(
    "embeddings_table_name"
)

EMBEDDING_MODEL_NAME = dbutils.widgets.get(
    "embedding_model"
)

DATABASE_SECRET_SCOPE = dbutils.widgets.get(
    "database_secret_scope"
)

DATABASE_SECRET_KEY = dbutils.widgets.get(
    "database_secret_key"
)

CHUNK_SIZE = int(
    dbutils.widgets.get("chunk_size")
)

CHUNK_OVERLAP = int(
    dbutils.widgets.get("chunk_overlap")
)

BATCH_SIZE = int(
    dbutils.widgets.get("batch_size")
)

# Different embedding models produce different vector dimensions.
# The pgvector VECTOR(N) column must match the selected model.

match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384

    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384

    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768

    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768

    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384

    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768

    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024

    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536

    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072

    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - "
            "add its output dimension to the match/case block "
            "before running this notebook."
        )

if CHUNK_SIZE <= 0:
    raise ValueError("chunk_size must be greater than zero")

if CHUNK_OVERLAP < 0:
    raise ValueError("chunk_overlap cannot be negative")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "chunk_overlap must be smaller than chunk_size"
    )

if BATCH_SIZE <= 0:
    raise ValueError("batch_size must be greater than zero")

print(
    f"Using model {EMBEDDING_MODEL_NAME!r} "
    f"-> {EMBEDDING_DIM}-dim vectors"
)

print(f"Source table:    {WEATHER_TABLE_NAME}")
print(f"Embedding table: {EMBEDDINGS_TABLE_NAME}")
print(f"Chunk size:      {CHUNK_SIZE}")
print(f"Chunk overlap:   {CHUNK_OVERLAP}")
print(f"Batch size:      {BATCH_SIZE}")
print(f"Secret scope:    {DATABASE_SECRET_SCOPE}")
print(f"Secret key:      {DATABASE_SECRET_KEY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakebase Connection
# MAGIC
# MAGIC Read the PostgreSQL connection URL from Databricks Secrets and connect
# MAGIC directly with pg8000.
# MAGIC
# MAGIC The helper accepts either:
# MAGIC
# MAGIC - a Base64-encoded PostgreSQL URL, or
# MAGIC - a plain PostgreSQL URL.

# COMMAND ----------

def get_lakebase_url() -> str:
    """Read the Lakebase PostgreSQL URL from Databricks Secrets."""
    secret_value = dbutils.secrets.get(
        scope=DATABASE_SECRET_SCOPE,
        key=DATABASE_SECRET_KEY,
    )

    if not secret_value:
        raise RuntimeError(
            f"Secret {DATABASE_SECRET_SCOPE}/{DATABASE_SECRET_KEY} is empty."
        )

    # If the secret already contains a normal PostgreSQL URL, use it directly.
    if secret_value.startswith(
        ("postgresql://", "postgres://")
    ):
        return secret_value

    # Otherwise, treat it as the Base64-encoded URL used by this project.
    try:
        decoded_value = base64.b64decode(
            secret_value
        ).decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            "Lakebase secret is neither a PostgreSQL URL nor "
            "a valid Base64-encoded PostgreSQL URL."
        ) from exc

    if not decoded_value.startswith(
        ("postgresql://", "postgres://")
    ):
        raise RuntimeError(
            "Decoded Lakebase secret is not a PostgreSQL URL."
        )

    return decoded_value


def get_connection():
    """Return a new pg8000 DB-API Lakebase connection."""
    parsed = urlparse(
        get_lakebase_url()
    )

    if not parsed.hostname:
        raise RuntimeError(
            "Lakebase URL is missing a hostname."
        )

    if not parsed.username:
        raise RuntimeError(
            "Lakebase URL is missing a username."
        )

    if parsed.password is None:
        raise RuntimeError(
            "Lakebase URL is missing a password."
        )

    if not parsed.path or parsed.path == "/":
        raise RuntimeError(
            "Lakebase URL is missing a database name."
        )

    return pg8000.dbapi.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/"),
        user=unquote(parsed.username),
        password=unquote(parsed.password),
        ssl_context=True,
    )


def query_rows(
    sql: str,
    params: tuple | list | None = None,
) -> list[dict]:
    """Run a SELECT query and return rows as dictionaries."""
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            sql,
            params or (),
        )

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
        conn.close()


print("Lakebase connection helpers created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Lakebase Connection
# MAGIC
# MAGIC Confirm that the notebook can connect to Lakebase before continuing.

# COMMAND ----------

connection_test = query_rows(
    """
    SELECT
        current_database() AS database_name,
        current_user AS database_user
    """
)

if not connection_test:
    raise RuntimeError(
        "Lakebase connection test returned no rows."
    )

print("Lakebase connection successful.")
print(
    "Database:",
    connection_test[0]["database_name"],
)
print(
    "User:",
    connection_test[0]["database_user"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Required Tables
# MAGIC
# MAGIC Confirm that both weather tables exist before creating embeddings.

# COMMAND ----------

required_tables = query_rows(
    """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name IN (%s, %s)
    ORDER BY table_name
    """,
    (
        WEATHER_TABLE_NAME,
        EMBEDDINGS_TABLE_NAME,
    ),
)

found_tables = {
    row["table_name"]
    for row in required_tables
}

missing_tables = {
    WEATHER_TABLE_NAME,
    EMBEDDINGS_TABLE_NAME,
} - found_tables

if missing_tables:
    raise RuntimeError(
        "Missing required Lakebase table(s): "
        + ", ".join(sorted(missing_tables))
        + ". Run the SQL setup scripts first."
    )

print(
    "Required tables found:",
    ", ".join(sorted(found_tables)),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify pgvector Column
# MAGIC
# MAGIC Confirm that `weather_embeddings.embedding` is a pgvector column.

# COMMAND ----------

embedding_column = query_rows(
    """
    SELECT
        column_name,
        data_type,
        udt_name
    FROM information_schema.columns
    WHERE table_name = %s
      AND column_name = 'embedding'
    """,
    (EMBEDDINGS_TABLE_NAME,),
)

if not embedding_column:
    raise RuntimeError(
        f"The {EMBEDDINGS_TABLE_NAME}.embedding column was not found."
    )

print(
    "Embedding data type:",
    embedding_column[0]["data_type"],
)

print(
    "Embedding UDT:",
    embedding_column[0]["udt_name"],
)

if embedding_column[0]["udt_name"] != "vector":
    raise RuntimeError(
        f"{EMBEDDINGS_TABLE_NAME}.embedding is not a pgvector column."
    )

print("pgvector column verified.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Narrative Text
# MAGIC
# MAGIC Weather narratives are split into overlapping character-based chunks.
# MAGIC
# MAGIC Default values:
# MAGIC
# MAGIC - Chunk size: 800 characters
# MAGIC - Chunk overlap: 100 characters

# COMMAND ----------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if not text:
        return []

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks = []

    for start in range(
        0,
        len(text),
        step,
    ):
        chunk = text[
            start:start + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

        if (
            start + chunk_size
            >= len(text)
        ):
            break

    return chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Documents to Embed
# MAGIC
# MAGIC A document is processed when:
# MAGIC
# MAGIC - it has never been embedded, or
# MAGIC - its `updated_at` value is newer than its most recent embedding.

# COMMAND ----------

def get_documents_to_embed() -> list[dict]:
    """
    Return documents that have never been embedded or were updated
    after their most recent embedding was created.
    """
    return query_rows(
        f"""
        SELECT
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.narrative_text,
            d.updated_at
        FROM {WEATHER_TABLE_NAME} d
        WHERE
            d.narrative_text IS NOT NULL
            AND TRIM(d.narrative_text) <> ''
            AND (
                NOT EXISTS (
                    SELECT 1
                    FROM {EMBEDDINGS_TABLE_NAME} e
                    WHERE e.document_id = d.id
                )
                OR d.updated_at > COALESCE(
                    (
                        SELECT MAX(e.created_at)
                        FROM {EMBEDDINGS_TABLE_NAME} e
                        WHERE e.document_id = d.id
                    ),
                    TIMESTAMPTZ '1970-01-01 00:00:00+00'
                )
            )
        ORDER BY d.updated_at ASC
        """
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Chunk Rows
# MAGIC
# MAGIC Create one chunk record for every chunk produced from each weather
# MAGIC document.

# COMMAND ----------

def build_chunk_rows(
    documents: list[dict],
) -> list[dict]:
    """Create chunk records for each weather document."""
    chunk_rows = []

    for document in documents:
        chunks = chunk_text(
            document["narrative_text"]
        )

        for chunk_index, text in enumerate(
            chunks
        ):
            chunk_rows.append({
                "document_id": document["id"],
                "chunk_index": chunk_index,
                "chunk_text": text,
            })

    return chunk_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Embeddings
# MAGIC
# MAGIC Generate embeddings for each weather text chunk using the configured
# MAGIC sentence-transformers model.

# COMMAND ----------

def embed_chunks(
    model: SentenceTransformer,
    chunk_rows: list[dict],
) -> list[dict]:
    """Generate embeddings for all weather chunks."""
    embedded_rows = []

    for start in range(
        0,
        len(chunk_rows),
        BATCH_SIZE,
    ):
        batch = chunk_rows[
            start:start + BATCH_SIZE
        ]

        texts = [
            row["chunk_text"]
            for row in batch
        ]

        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for row, vector in zip(
            batch,
            vectors,
        ):
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {EMBEDDING_DIM} dimensions "
                    f"but received {len(vector)}."
                )

            embedded_rows.append({
                **row,
                "embedding": [
                    float(value)
                    for value in vector
                ],
            })

    return embedded_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Embeddings to PostgreSQL Vector Format
# MAGIC
# MAGIC pgvector accepts vector literals in this form:
# MAGIC
# MAGIC `[0.0123,-0.0345,...]`

# COMMAND ----------

def vector_to_string(
    vector: list[float],
) -> str:
    """Convert a Python vector to PostgreSQL vector literal syntax."""
    return (
        "["
        + ",".join(
            str(float(value))
            for value in vector
        )
        + "]"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Replace Document Embeddings
# MAGIC
# MAGIC Delete stale embeddings for the documents processed in this run and
# MAGIC insert the newly generated chunks and vectors.

# COMMAND ----------

def replace_document_embeddings(
    document_ids: list[str],
    embedded_rows: list[dict],
) -> int:
    """
    Replace embeddings for the documents processed in this run.

    This allows an updated NWS alert to be re-embedded rather than
    leaving stale chunks in the vector table.
    """
    if not document_ids:
        return 0

    conn = get_connection()
    cur = conn.cursor()

    try:
        for document_id in document_ids:
            cur.execute(
                f"""
                DELETE FROM {EMBEDDINGS_TABLE_NAME}
                WHERE document_id = %s
                """,
                (document_id,),
            )

        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id,
                document_id,
                chunk_index,
                chunk_text,
                embedding,
                model_name,
                created_at
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s::vector,
                %s,
                NOW()
            )
            ON CONFLICT (document_id, chunk_index)
            DO UPDATE
            SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                model_name = EXCLUDED.model_name,
                created_at = NOW()
        """

        rows = []

        for row in embedded_rows:
            rows.append((
                (
                    f"{row['document_id']}"
                    f":{row['chunk_index']}"
                ),
                row["document_id"],
                row["chunk_index"],
                row["chunk_text"],
                vector_to_string(
                    row["embedding"]
                ),
                EMBEDDING_MODEL_NAME,
            ))

        if rows:
            cur.executemany(
                insert_sql,
                rows,
            )

        conn.commit()
        return len(rows)

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Stored Embeddings
# MAGIC
# MAGIC Check the number of stored embeddings and verify their vector
# MAGIC dimensions.

# COMMAND ----------

def verify_embeddings():
    """Print a simple verification of the stored vector data."""
    rows = query_rows(
        f"""
        SELECT
            COUNT(*) AS embedding_count,
            MIN(vector_dims(embedding)) AS min_dimensions,
            MAX(vector_dims(embedding)) AS max_dimensions
        FROM {EMBEDDINGS_TABLE_NAME}
        """
    )

    if rows:
        print(
            "Embedding rows:",
            rows[0]["embedding_count"],
        )

        print(
            "Minimum dimensions:",
            rows[0]["min_dimensions"],
        )

        print(
            "Maximum dimensions:",
            rows[0]["max_dimensions"],
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Embedding Pipeline
# MAGIC
# MAGIC This function:
# MAGIC
# MAGIC 1. Finds new or updated weather documents.
# MAGIC 2. Creates text chunks.
# MAGIC 3. Loads the embedding model.
# MAGIC 4. Generates embeddings.
# MAGIC 5. Writes vectors to Lakebase.
# MAGIC 6. Verifies the stored vectors.

# COMMAND ----------

def main():
    print(
        f"Embedding model: {EMBEDDING_MODEL_NAME}"
    )

    print(
        f"Weather table: {WEATHER_TABLE_NAME}"
    )

    print(
        f"Embedding table: {EMBEDDINGS_TABLE_NAME}"
    )

    documents = get_documents_to_embed()

    print(
        f"Documents requiring embeddings: {len(documents)}"
    )

    if not documents:
        print(
            "No new or updated weather documents require embedding."
        )
        verify_embeddings()
        return

    chunk_rows = build_chunk_rows(
        documents
    )

    print(
        f"Text chunks created: {len(chunk_rows)}"
    )

    if not chunk_rows:
        print(
            "No chunks were created."
        )
        return

    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    embedded_rows = embed_chunks(
        model,
        chunk_rows,
    )

    print(
        f"Embeddings generated: {len(embedded_rows)}"
    )

    document_ids = [
        document["id"]
        for document in documents
    ]

    written = replace_document_embeddings(
        document_ids,
        embedded_rows,
    )

    print(
        f"Embeddings written to Lakebase: {written}"
    )

    verify_embeddings()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute
# MAGIC
# MAGIC Run the complete embedding pipeline.

# COMMAND ----------

if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/ingest_weather_news_embeddings.py")
path.write_text(notebook, encoding="utf-8")
print(path)

if __name__ == "__main__":
    main()
