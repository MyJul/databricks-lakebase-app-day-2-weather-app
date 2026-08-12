<<<<<<< Updated upstream
"""
Ingest weather document embeddings into Lakebase.
=======
# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the Weather Intelligence application.
# MAGIC
# MAGIC Pipeline:
# MAGIC
# MAGIC ```text
# MAGIC National Weather Service API
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/sync
# MAGIC             |
# MAGIC             v
# MAGIC        weather_news
# MAGIC             |
# MAGIC             v
# MAGIC ingest_weather_embeddings
# MAGIC             |
# MAGIC       chunk narrative_text
# MAGIC             |
# MAGIC     all-MiniLM-L6-v2
# MAGIC             |
# MAGIC             v
# MAGIC     weather_embeddings
# MAGIC             |
# MAGIC       pgvector / HNSW
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/search
# MAGIC ```
# MAGIC
# MAGIC The notebook:
# MAGIC
# MAGIC 1. Reads weather documents from `weather_news`.
# MAGIC 2. Finds documents that do not yet have embeddings.
# MAGIC 3. Splits `narrative_text` into overlapping chunks.
# MAGIC 4. Generates 384-dimensional embeddings using
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2`.
# MAGIC 5. Writes embeddings directly to Lakebase.
# MAGIC 6. Stores embeddings in `weather_embeddings.embedding` as
# MAGIC    PostgreSQL `VECTOR(384)`.
# MAGIC 7. Validates pgvector and cosine similarity search.
# MAGIC
# MAGIC IMPORTANT:
# MAGIC
# MAGIC This notebook intentionally does NOT use:
# MAGIC
# MAGIC - psycopg2
# MAGIC - psycopg2-binary
# MAGIC - Spark JDBC for writes
# MAGIC - the legacy Lakebase Database Instance API
# MAGIC - a manually stored PostgreSQL password
# MAGIC
# MAGIC Database connections use:
# MAGIC
# MAGIC ```text
# MAGIC WorkspaceClient
# MAGIC       |
# MAGIC       v
# MAGIC postgres.generate_database_credential()
# MAGIC       |
# MAGIC       v
# MAGIC short-lived Lakebase credential
# MAGIC       |
# MAGIC       v
# MAGIC pg8000
# MAGIC ```
# MAGIC =======
# MAGIC 1. Reads weather documents harvested from the National Weather Service
# MAGIC    (NWS) API and stored in `weather_news`.
# MAGIC 2. Finds documents that do not yet have embeddings.
# MAGIC 3. Splits each document's `narrative_text` into overlapping chunks.
# MAGIC 4. Embeds each chunk using
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
# MAGIC 5. Writes embeddings directly to Lakebase using `pg8000`.
# MAGIC 6. Stores the embeddings in a PostgreSQL `VECTOR(384)` column.
# MAGIC 7. Verifies the pgvector column and performs a cosine-similarity test.
# MAGIC
# MAGIC The Flask application's `/weather/sync` endpoint is responsible for
# MAGIC harvesting NWS data and populating `weather_news`.
# MAGIC
# MAGIC The resulting pipeline is:
# MAGIC
# MAGIC ```text
# MAGIC NWS API
# MAGIC    |
# MAGIC    v
# MAGIC /weather/sync
# MAGIC    |
# MAGIC    v
# MAGIC weather_news
# MAGIC    |
# MAGIC    v
# MAGIC this notebook
# MAGIC    |
# MAGIC    +--> chunk narrative_text
# MAGIC    |
# MAGIC    +--> all-MiniLM-L6-v2
# MAGIC    |
# MAGIC    v
# MAGIC weather_embeddings
# MAGIC    |
# MAGIC    v
# MAGIC pgvector / HNSW
# MAGIC    |
# MAGIC    v
# MAGIC /weather/search
# MAGIC ```
# MAGIC
# MAGIC ## Important Serverless Compatibility Note
# MAGIC
# MAGIC This notebook intentionally does NOT use:
# MAGIC
# MAGIC - `psycopg2`
# MAGIC - `psycopg2-binary`
# MAGIC - `spark.write.jdbc`
# MAGIC
# MAGIC `psycopg2-binary` contains native C extensions that can cause
# MAGIC `SIGABRT 134` kernel crashes on Databricks Serverless compute.
# MAGIC
# MAGIC Instead, this notebook uses **pg8000**, a pure-Python PostgreSQL driver.
# MAGIC
# MAGIC Embeddings are also written directly to the PostgreSQL `VECTOR` column,
# MAGIC so no post-processing cast from arrays to vectors is required.
>>>>>>> Stashed changes

Pipeline:
    weather_documents
        -> find new or updated documents
        -> chunk narrative_text
        -> sentence-transformers/all-MiniLM-L6-v2
        -> VECTOR(384)
        -> weather_embeddings

<<<<<<< Updated upstream
This script intentionally uses pg8000 through lakebase.py.
It does not use psycopg2 or Spark JDBC.
"""
import os
from sentence_transformers import SentenceTransformer
import lakebase

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 50
=======
# DBTITLE 1,Setup Instructions
# MAGIC %md
# MAGIC ## ✅ Lakebase Instance Status
# MAGIC
# MAGIC This notebook connects to your Lakebase Postgres instance:
# MAGIC
# MAGIC - **Project**: `dataexpert-student`
# MAGIC - **Branch**: `production` (READY)
# MAGIC - **Endpoint**: `primary` (ACTIVE)  
# MAGIC - **Database**: `databricks_postgres`
# MAGIC - **PostgreSQL**: 17.10
# MAGIC - **pgvector**: 0.8.0 ✓
# MAGIC
# MAGIC **Connection verified and working!**
# MAGIC
# MAGIC ### Required Setup
# MAGIC
# MAGIC #### 1. Create Required Tables
# MAGIC
# MAGIC Run these SQL scripts in your Lakebase database (in order):
# MAGIC
# MAGIC ```bash
# MAGIC # From the project root:
# MAGIC psql "$(cat lakebase-url-from-secret)" -f sql/01_setup_weather_news_table.sql
# MAGIC psql "$(cat lakebase-url-from-secret)" -f sql/02_setup_weather_embeddings.sql
# MAGIC ```
# MAGIC
# MAGIC Or connect via pgAdmin/DBeaver and run the SQL files manually.
# MAGIC
# MAGIC ### 3. Sync Weather Data
# MAGIC
# MAGIC Before running this notebook, call the Flask API to fetch and store weather documents:
# MAGIC
# MAGIC ```bash
# MAGIC curl -X POST http://your-app-url/weather/sync \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{
# MAGIC     "locations": ["Chicago, IL", "Austin, TX"],
# MAGIC     "limit": 50
# MAGIC   }'
# MAGIC ```
# MAGIC
# MAGIC ### 4. Once Setup is Complete
# MAGIC
# MAGIC Run all cells in order. The notebook will:
# MAGIC 1. Connect to Lakebase
# MAGIC 2. Find unembedded weather documents  
# MAGIC 3. Chunk the narrative text
# MAGIC 4. Generate embeddings using sentence-transformers
# MAGIC 5. Write embeddings back to Lakebase with pgvector
# MAGIC
# MAGIC The notebook is idempotent - safe to re-run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC `pg8000` is used instead of `psycopg2-binary`.
# MAGIC
# MAGIC `pg8000` is a pure-Python PostgreSQL driver and avoids the native C
# MAGIC extension issues that can cause SIGABRT kernel crashes on Databricks
# MAGIC Serverless compute.
# MAGIC
# MAGIC `databricks-sdk` is used to generate a fresh Lakebase database credential.
# MAGIC
# MAGIC `sentence-transformers` generates the weather embeddings.
# MAGIC - **pg8000**: Pure-Python PostgreSQL/Lakebase driver
# MAGIC - **sentence-transformers**: Embedding model library
# MAGIC
# MAGIC The NWS API is called by the Flask application's `weather_client.py`.
# MAGIC This notebook only processes documents already stored in Lakebase.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install -q pg8000 sentence-transformers sqlalchemy requests
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
dbutils.widgets.text(
    "lakebase_endpoint_name",
    "projects/dataexpert-student/branches/production/endpoints/primary",
    "Lakebase endpoint"
)

dbutils.widgets.text(
    "lakebase_host",
    "ep-muddy-hat-d8d8gsj6.database.us-east-2.cloud.databricks.com",
    "Lakebase host"
)

dbutils.widgets.text(
    "lakebase_port",
    "5432",
    "Lakebase port"
)

dbutils.widgets.text(
    "lakebase_database",
    "databricks_postgres",
    "Lakebase database"
)

dbutils.widgets.text(
    "lakebase_user",
    "student",
    "Lakebase PostgreSQL user"
)

dbutils.widgets.text(
    "weather_table_name",
    "weather_news",
    "Source table (weather documents)"
)

dbutils.widgets.text(
    "embeddings_table_name",
    "weather_embeddings",
    "Destination table (chunk embeddings)"
)

dbutils.widgets.text(
    "embedding_model",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Embedding model"
)

dbutils.widgets.text(
    "chunk_size",
    "800",
    "Chunk size (characters)"
)

dbutils.widgets.text(
    "chunk_overlap",
    "100",
    "Chunk overlap (characters)"
)

dbutils.widgets.text(
    "batch_size",
    "50",
    "Embedding/write batch size"
)


LAKEBASE_ENDPOINT_NAME = dbutils.widgets.get("lakebase_endpoint_name")
DB_HOST = dbutils.widgets.get("lakebase_host")
DB_PORT = int(dbutils.widgets.get("lakebase_port"))
DB_NAME = dbutils.widgets.get("lakebase_database")
DB_USER = dbutils.widgets.get("lakebase_user")

WEATHER_TABLE_NAME = dbutils.widgets.get("weather_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")

EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")

CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))


# This weather application is intentionally locked to the assignment model.
if EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    EMBEDDING_DIM = 384
else:
    raise ValueError(
        f"Unsupported embedding model: {EMBEDDING_MODEL_NAME!r}. "
        "This weather app requires "
        "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."
    )


if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "chunk_overlap must be smaller than chunk_size"
    )


print("=" * 70)
print("Configuration")
print("=" * 70)
print(f"Lakebase endpoint:      {LAKEBASE_ENDPOINT_NAME}")
print(f"Lakebase host:          {DB_HOST}")
print(f"Lakebase port:          {DB_PORT}")
print(f"Lakebase database:      {DB_NAME}")
print(f"Lakebase user:          {DB_USER}")
print(f"Weather table:          {WEATHER_TABLE_NAME}")
print(f"Embedding table:        {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model:        {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimensions:   {EMBEDDING_DIM}")
print(f"Chunk size:             {CHUNK_SIZE}")
print(f"Chunk overlap:          {CHUNK_OVERLAP}")
print(f"Batch size:             {BATCH_SIZE}")

# COMMAND ----------

# DBTITLE 1,Lakebase Connection
# MAGIC %md
# MAGIC ## Lakebase Connection
# MAGIC
# MAGIC Lakebase Autoscaling Projects use the PostgreSQL API:
# MAGIC
# MAGIC `WorkspaceClient().postgres.generate_database_credential()`
# MAGIC
# MAGIC The returned OAuth credential is used as the PostgreSQL password.
# MAGIC
# MAGIC pg8000 is used as the PostgreSQL driver because it is pure Python and
# MAGIC avoids the native C-extension issues associated with psycopg2-binary
# MAGIC on Databricks Serverless compute.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install -q "pg8000" "sentence-transformers" "databricks-sdk>=0.89.0"

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Reinitialize configuration after Python restart
# MAGIC %md
# MAGIC ## Reinitialize configuration after Python restart
# MAGIC
# MAGIC The `%pip` restart clears Python variables, so the configuration widgets
# MAGIC are read again here.

# COMMAND ----------

# DBTITLE 1,Restore configuration variables
import os
import pg8000.native

from databricks.sdk import WorkspaceClient


LAKEBASE_ENDPOINT_NAME = dbutils.widgets.get(
    "lakebase_endpoint_name"
)

DB_HOST = dbutils.widgets.get(
    "lakebase_host"
)

DB_PORT = int(
    dbutils.widgets.get("lakebase_port")
)

DB_NAME = dbutils.widgets.get(
    "lakebase_database"
)

DB_USER = dbutils.widgets.get(
    "lakebase_user"
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

CHUNK_SIZE = int(
    dbutils.widgets.get("chunk_size")
)

CHUNK_OVERLAP = int(
    dbutils.widgets.get("chunk_overlap")
)

BATCH_SIZE = int(
    dbutils.widgets.get("batch_size")
)


if EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    EMBEDDING_DIM = 384
else:
    raise ValueError(
        f"Unsupported embedding model: {EMBEDDING_MODEL_NAME!r}"
    )


print("=" * 70)
print("Configuration restored after Python restart")
print("=" * 70)
print(f"Lakebase endpoint:      {LAKEBASE_ENDPOINT_NAME}")
print(f"Lakebase host:          {DB_HOST}")
print(f"Lakebase database:      {DB_NAME}")
print(f"Lakebase user:          {DB_USER}")
print(f"Weather table:          {WEATHER_TABLE_NAME}")
print(f"Embedding table:        {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model:        {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimensions:   {EMBEDDING_DIM}")

# COMMAND ----------

# DBTITLE 1,Create Lakebase Connection
# MAGIC %md
# MAGIC ## Create Lakebase Connection
# MAGIC
# MAGIC A fresh OAuth database credential is generated through the modern
# MAGIC Lakebase PostgreSQL API every time this function creates a connection.
# MAGIC
# MAGIC No secret scope is required.
# MAGIC No `lakebase.py` file is required.
# MAGIC No legacy Lakebase instance name is required.

# COMMAND ----------

# DBTITLE 1,Define connection helper functions
import base64
from urllib.parse import urlparse

w = WorkspaceClient()


def get_lakebase_credentials():
    """
    Read the Lakebase credentials from the stored connection URL secret.
    
    This endpoint uses password authentication, not OAuth tokens.
    The secret is double Base64-encoded.
    
    Returns:
        tuple: (username, password, host, port, database)
    """
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    
    if not secret.value:
        raise ValueError("Secret database/lakebase-url is empty.")
    
    # Double decode: the secret is Base64-encoded twice
    first_decode = base64.b64decode(secret.value).decode("utf-8")
    lakebase_url = base64.b64decode(first_decode).decode("utf-8")
    
    parsed = urlparse(lakebase_url)
    
    if not parsed.username or not parsed.password:
        raise ValueError("Lakebase URL does not contain username and password.")
    
    return (
        parsed.username,
        parsed.password,
        parsed.hostname,
        parsed.port or 5432,
        parsed.path.lstrip('/')
    )


def get_connection():
    """
    Open a pg8000 connection to the Lakebase Autoscaling endpoint.

    Uses credentials (username + password) from the stored secret URL.
    """
    username, password, host, port, database = get_lakebase_credentials()

    conn = pg8000.native.Connection(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
        ssl_context=True,
    )

    return conn

# COMMAND ----------

# DBTITLE 1,Test Lakebase Connection
# MAGIC %md
# MAGIC ## Test Lakebase Connection

# COMMAND ----------

# DBTITLE 1,Test connection
try:

    conn = get_connection()

    try:

        result = conn.run(
            """
            SELECT
                current_database(),
                current_user
            """
        )

        database, user = result[0]

        print("✅ Lakebase connection successful.")
        print(f"Database: {database}")
        print(f"User:     {user}")
        print(f"Host:     {DB_HOST}:{DB_PORT}")

    finally:
        conn.close()

except Exception as e:

    print("❌ Lakebase connection failed.")
    print(f"Error: {e}")

    print("\nConnection configuration:")
    print(f"  Endpoint: {LAKEBASE_ENDPOINT_NAME}")
    print(f"  Host:     {DB_HOST}")
    print(f"  Port:     {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User:     {DB_USER}")

    raise

# COMMAND ----------

# DBTITLE 1,Verify Required Tables
# MAGIC %md
# MAGIC ## Verify Required Tables
# MAGIC
# MAGIC Confirm that both weather_news and weather_embeddings exist
# MAGIC and inspect their column definitions.

# COMMAND ----------

# DBTITLE 1,Verify table schemas
conn = get_connection()

try:
    table_columns = conn.run(
        """
        SELECT
            table_name,
            column_name,
            data_type,
            udt_name
        FROM information_schema.columns
        WHERE table_name IN (:weather_table, :embedding_table)
        ORDER BY table_name, ordinal_position
        """,
        weather_table=WEATHER_TABLE_NAME,
        embedding_table=EMBEDDINGS_TABLE_NAME,
    )

finally:
    conn.close()


if not table_columns:
    raise RuntimeError(
        f"No columns found for {WEATHER_TABLE_NAME!r} or "
        f"{EMBEDDINGS_TABLE_NAME!r}. "
        "Run the required SQL setup files first."
    )


print("Database schema")
print("=" * 70)

current_table = None

for row in table_columns:

    table_name = row[0]
    column_name = row[1]
    data_type = row[2]
    udt_name = row[3]

    if table_name != current_table:
        print(f"\n{table_name}")
        print("-" * 70)
        current_table = table_name

    print(
        f"  {column_name}: "
        f"{data_type} ({udt_name})"
    )

# COMMAND ----------

# DBTITLE 1,Verify Embedding Column
# MAGIC %md
# MAGIC ## Verify Embedding Column
# MAGIC
# MAGIC Confirm that the embedding column exists, uses pgvector, and has the
# MAGIC expected 384-dimensional vector type.

# COMMAND ----------

# DBTITLE 1,Check embedding column type
conn = get_connection()

try:
    embedding_column = conn.run(
        """
        SELECT
            a.attname AS column_name,
            format_type(a.atttypid, a.atttypmod) AS formatted_type,
            t.typname AS udt_name
        FROM pg_attribute a
        JOIN pg_class c
            ON a.attrelid = c.oid
        JOIN pg_type t
            ON a.atttypid = t.oid
        WHERE c.relname = :table_name
          AND a.attname = 'embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """,
        table_name=EMBEDDINGS_TABLE_NAME,
    )

finally:
    conn.close()


if not embedding_column:
    raise RuntimeError(
        f"The {EMBEDDINGS_TABLE_NAME}.embedding column was not found."
    )


column_name = embedding_column[0][0]
formatted_type = embedding_column[0][1]
udt_name = embedding_column[0][2]

print(f"Column name:         {column_name}")
print(f"Embedding type:      {formatted_type}")
print(f"Embedding UDT:       {udt_name}")


if udt_name != "vector":
    raise RuntimeError(
        f"{EMBEDDINGS_TABLE_NAME}.embedding is not a pgvector column. "
        f"Expected UDT 'vector', received {udt_name!r}."
    )


if "384" not in formatted_type:
    raise RuntimeError(
        f"{EMBEDDINGS_TABLE_NAME}.embedding does not appear to be "
        f"VECTOR(384). Received: {formatted_type!r}"
    )


print("✅ pgvector VECTOR(384) column verified.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Weather Documents
# MAGIC
# MAGIC The Flask application's `/weather/sync` endpoint should be run before
# MAGIC this notebook.
# MAGIC
# MAGIC Example request:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "locations": ["Chicago, IL", "Austin, TX"],
# MAGIC   "limit": 50
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC The endpoint stores normalized NWS alerts and forecasts in
# MAGIC `weather_news`.

# COMMAND ----------

conn = get_connection()

try:

    result = conn.run(
        f"""
        SELECT COUNT(*)
        FROM {WEATHER_TABLE_NAME}
        """
    )

    total_documents = result[0][0]

    result = conn.run(
        f"""
        SELECT COUNT(*)
        FROM {WEATHER_TABLE_NAME}
        WHERE narrative_text IS NOT NULL
          AND TRIM(narrative_text) <> ''
        """
    )

    documents_with_text = result[0][0]

finally:

    conn.close()

print(
    f"Total weather documents:   {total_documents}"
)

document_counts = query_rows(
    f"""
    SELECT
        COUNT(*) AS total_documents,
        COUNT(*) FILTER (
            WHERE narrative_text IS NOT NULL
              AND TRIM(narrative_text) <> ''
        ) AS documents_with_text
    FROM {WEATHER_TABLE_NAME}
    """
)

total_documents = document_counts[0]["total_documents"]
documents_with_text = document_counts[0]["documents_with_text"]

print(
    f"Documents containing text: {documents_with_text}"
)

if total_documents == 0:
    print(
        "\n⚠️ No weather documents are currently available.\n"
        "Run POST /weather/sync from the Flask application first."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Unembedded Weather Documents
# MAGIC
# MAGIC A document is considered unembedded when no corresponding
# MAGIC `weather_embeddings` row exists.
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC This makes the notebook safe to run repeatedly.

# COMMAND ----------

conn = get_connection()

try:

    documents = conn.run(
        f"""
        SELECT
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.narrative_text,
            d.issued_at,
            d.effective_at
        FROM {WEATHER_TABLE_NAME} d
        WHERE d.narrative_text IS NOT NULL
          AND TRIM(d.narrative_text) <> ''
          AND NOT EXISTS (
              SELECT 1
              FROM {EMBEDDINGS_TABLE_NAME} e
              WHERE e.document_id = d.id
          )
        ORDER BY d.synced_at ASC
        """
    )

finally:

    conn.close()

# Convert pg8000 tuples into dictionaries.

document_columns = [
    "id",
    "location",
    "source_type",
    "headline",
    "narrative_text",
    "issued_at",
    "effective_at",
]

documents = [
    dict(zip(document_columns, row))
    for row in documents
]

print(
    f"Found {len(documents)} unembedded weather documents."
)

=======
This makes the notebook safe to run repeatedly:

- previously embedded documents are skipped
- newly synchronized NWS documents are picked up
- existing embeddings are not unnecessarily duplicated

Because one document may produce multiple chunks, the existence check
is performed at the document level.

# COMMAND ----------

documents = query_rows(
    f"""
    SELECT
        d.id,
        d.location,
        d.source_type,
        d.headline,
        d.narrative_text,
        d.issued_at,
        d.effective_at,
        d.synced_at
    FROM {WEATHER_TABLE_NAME} d
    WHERE d.narrative_text IS NOT NULL
      AND TRIM(d.narrative_text) <> ''
      AND NOT EXISTS (
          SELECT 1
          FROM {EMBEDDINGS_TABLE_NAME} e
          WHERE e.document_id = d.id
      )
    ORDER BY d.synced_at ASC
    """,
)

print(f"Found {len(documents)} unembedded weather documents.")

>>>>>>> Stashed changes
if documents:
    print("\nSample documents:")

    for document in documents[:5]:
        narrative = document["narrative_text"] or ""

        print()
        print(f"ID:       {document['id']}")
        print(f"Location: {document['location']}")
        print(f"Type:     {document['source_type']}")
        print(f"Headline: {document['headline']}")
        print(
<<<<<<< Updated upstream
            f"Text:     "
            f"{document['narrative_text'][:300]}..."
=======
            f"\nID:       {document['id']}"
            f"\nLocation: {document['location']}"
            f"\nType:     {document['source_type']}"
            f"\nHeadline: {document['headline']}"
            f"\nText:     {narrative[:300]}..."
>>>>>>> Stashed changes
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Narrative Text
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC NWS narrative text is split into overlapping character-based chunks.
# MAGIC
# MAGIC Default:
# MAGIC
# MAGIC ```text
# MAGIC Chunk size:    800
# MAGIC Chunk overlap: 100
# MAGIC ```
# MAGIC =======
# MAGIC NWS narrative text is generally short, but alerts can contain longer
# MAGIC descriptions and instructions.
# MAGIC
# MAGIC We use the assignment-recommended:
# MAGIC
# MAGIC ```text
# MAGIC CHUNK_SIZE    = 800 characters
# MAGIC CHUNK_OVERLAP = 100 characters
# MAGIC ```
# MAGIC
# MAGIC Most NWS documents will remain a single chunk. Longer alert narratives
# MAGIC are split into overlapping windows.
# MAGIC >>>>>>> Stashed changes

# COMMAND ----------
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
def get_documents_to_embed() -> list[dict]:
    """
    Return documents that have never been embedded or were updated
    after their most recent embedding was created.
    """
    return lakebase.run_query(
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
=======
<<<<<<< Updated upstream

# COMMAND ----------

=======
>>>>>>> Stashed changes

chunk_rows = []

for document in documents:

    chunks = chunk_text(
        document["narrative_text"],
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
>>>>>>> Stashed changes
    )

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
<<<<<<< Updated upstream
            })
    return chunk_rows
=======
            }
        )

<<<<<<< Updated upstream
print(
    f"Created {len(chunk_rows)} text chunks."
)

if chunk_rows:

    print()
    print("Sample chunk:")
    print(chunk_rows[0])
=======
print(f"Created {len(chunk_rows)} text chunks.")

if chunk_rows:

    print("\nSample chunk:")
    print(f"Document: {chunk_rows[0]['document_id']}")
    print(f"Chunk:    {chunk_rows[0]['chunk_index']}")
    print(f"Text:     {chunk_rows[0]['chunk_text'][:500]}")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Embedding Model
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC `all-MiniLM-L6-v2` produces 384-dimensional vectors.
# MAGIC =======
# MAGIC The same model is used by the Flask `/weather/search` endpoint.
# MAGIC
# MAGIC `sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional
# MAGIC embeddings.
# MAGIC >>>>>>> Stashed changes

# COMMAND ----------

from sentence_transformers import SentenceTransformer

print(f"Loading {EMBEDDING_MODEL_NAME}...")

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("✅ Embedding model loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Chunk Embeddings
# MAGIC
# MAGIC Embeddings are generated in batches to control memory usage.
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC Embeddings are normalized so cosine similarity can be used consistently
# MAGIC during pgvector search.
# MAGIC =======
# MAGIC `normalize_embeddings=True` normalizes the vectors. This is compatible
# MAGIC with cosine similarity using pgvector's `<=>` operator.
# MAGIC >>>>>>> Stashed changes

# COMMAND ----------

embedded_rows = []

if not chunk_rows:

    print("No chunks require embedding.")

else:

    for start in range(0, len(chunk_rows), BATCH_SIZE):

        batch = chunk_rows[start:start + BATCH_SIZE]
>>>>>>> Stashed changes

def embed_chunks(
    model: SentenceTransformer,
    chunk_rows: list[dict],
) -> list[dict]:
    """Generate 384-dimensional embeddings for all chunks."""
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
<<<<<<< Updated upstream
        for row, vector in zip(
            batch,
            vectors,
=======
>>>>>>> Stashed changes

        for row, vector in zip(batch, vectors):

            vector_dimension = len(vector)

            if vector_dimension != EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {EMBEDDING_DIM}-dimensional vector but "
                    f"received {vector_dimension} dimensions."
                )

            row_with_embedding = dict(row)

            row_with_embedding["embedding"] = [
                float(value)
                for value in vector
            ]

            embedded_rows.append(row_with_embedding)

        processed = min(
            start + BATCH_SIZE,
            len(chunk_rows),
        )

<<<<<<< Updated upstream
    print(
        f"Embedded "
        f"{min(start + BATCH_SIZE, len(chunk_rows))}"
        f"/{len(chunk_rows)} chunks"
    )

print()
print(
    f"Generated {len(embedded_rows)} embeddings."
)
=======
        print(
            f"Embedded {processed}/{len(chunk_rows)} chunks"
        )

print(f"\nGenerated {len(embedded_rows)} embeddings.")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Generated Embedding
# MAGIC
# MAGIC This confirms that the embedding has the expected 384 dimensions before
# MAGIC anything is written to Lakebase.

# COMMAND ----------

if embedded_rows:

    sample = embedded_rows[0]

<<<<<<< Updated upstream
    print(
        f"Document ID:       "
        f"{sample['document_id']}"
    )

    print(
        f"Chunk index:       "
        f"{sample['chunk_index']}"
    )

    print(
        f"Chunk text:        "
        f"{sample['chunk_text'][:500]}"
    )

    print(
        f"Vector dimensions: "
        f"{len(sample['embedding'])}"
    )

    print(
        f"Model:             "
        f"{EMBEDDING_MODEL_NAME}"
    )
=======
    print(f"Document ID:       {sample['document_id']}")
    print(f"Chunk index:       {sample['chunk_index']}")
    print(f"Chunk text:        {sample['chunk_text'][:500]}")
    print(f"Vector dimensions: {len(sample['embedding'])}")
    print(f"Model:             {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Vector Values
# MAGIC
# MAGIC `pg8000` does not require Spark JDBC or a Spark-specific
# MAGIC `stringtype=unspecified` workaround.
# MAGIC
# MAGIC Each embedding is converted to a PostgreSQL vector literal:
# MAGIC
# MAGIC ```text
# MAGIC [0.0123,-0.0345,...]
# MAGIC ```
# MAGIC
# MAGIC and explicitly cast in SQL:
# MAGIC
# MAGIC ```sql
# MAGIC :embedding::vector
# MAGIC ```
# MAGIC
# MAGIC This writes directly into the `VECTOR(384)` column.

# COMMAND ----------

insert_rows = []

for row in embedded_rows:

    embedding_id = (
        f"{row['document_id']}_{row['chunk_index']}"
    )

    vector_string = (
        "["
        + ",".join(
            str(float(value))
            for value in row["embedding"]
        )
        + "]"
    )

    insert_rows.append(
        {
            "id": embedding_id,
            "document_id": row["document_id"],
            "chunk_index": int(row["chunk_index"]),
            "chunk_text": row["chunk_text"],
            "embedding": vector_string,
            "model_name": EMBEDDING_MODEL_NAME,
        }
    )

print(f"Prepared {len(insert_rows)} rows for Lakebase.")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Lakebase
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC Embeddings are written directly using pg8000.
# MAGIC
# MAGIC No Spark JDBC write is used.
# MAGIC
# MAGIC No psycopg2 is used.
# MAGIC =======
# MAGIC Embeddings are written directly through `pg8000`.
# MAGIC
# MAGIC No Spark JDBC is used.
# MAGIC
# MAGIC No `psycopg2` is used.
# MAGIC
# MAGIC No post-processing array-to-vector conversion is required.
# MAGIC >>>>>>> Stashed changes
# MAGIC
# MAGIC Each chunk receives a stable ID:
# MAGIC
# MAGIC ```text
# MAGIC document_id + "_" + chunk_index
# MAGIC <<<<<<< Updated upstream
# MAGIC ```
# MAGIC
# MAGIC The vector is explicitly cast:
# MAGIC
# MAGIC ```sql
# MAGIC CAST(:embedding AS vector)
# MAGIC ```
# MAGIC
# MAGIC This preserves the PostgreSQL `vector` type.

# COMMAND ----------

if not embedded_rows:

    print(
        "No new embeddings to write."
    )
=======
```

The `weather_embeddings.id` primary key makes the operation safely
repeatable.

The `ON CONFLICT` clause updates the existing chunk if the source text
or embedding changes.

# COMMAND ----------

if not insert_rows:

    print("No new embeddings to write.")
>>>>>>> Stashed changes

else:

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
            :id,
            :document_id,
            :chunk_index,
            :chunk_text,
            :embedding::vector,
            :model_name,
            now()
        )
        ON CONFLICT (id) DO UPDATE
        SET
            document_id = EXCLUDED.document_id,
            chunk_index = EXCLUDED.chunk_index,
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = CURRENT_TIMESTAMP
    """

    total_written = 0

    conn = get_connection()

    try:

        for start in range(
            0,
            len(insert_rows),
            BATCH_SIZE,
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
def vector_to_string(
    vector: list[float],
) -> str:
    """Convert a Python vector to PostgreSQL vector literal syntax."""
    return (
=======
            conn.commit()

            processed = min(
                start + BATCH_SIZE,
                len(insert_rows),
            )

<<<<<<< Updated upstream
=======
            print(
                f"Wrote {processed}/{len(insert_rows)} embeddings"
            )

    except Exception:

        conn.rollback()
        raise

>>>>>>> Stashed changes
    finally:

        conn.close()

<<<<<<< Updated upstream
    print()
    print(
        f"Successfully wrote "
        f"{total_written} embeddings "
=======
    print(
        f"\n✅ Successfully wrote {total_written} embeddings "
>>>>>>> Stashed changes
        f"to {EMBEDDINGS_TABLE_NAME}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embeddings
# MAGIC
# MAGIC Confirm:
# MAGIC
# MAGIC 1. Embeddings exist.
# MAGIC <<<<<<< Updated upstream
# MAGIC 2. The expected number of documents have embeddings.
# MAGIC 3. The `embedding` column is PostgreSQL `vector`.
# MAGIC 4. The vector dimension is 384.

# COMMAND ----------

conn = get_connection()

try:

    result = conn.run(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT document_id)
        FROM {EMBEDDINGS_TABLE_NAME}
        """
    )

    total_embeddings = result[0][0]
    documents_embedded = result[0][1]

    result = conn.run(
        """
        SELECT
            column_name,
            data_type,
            udt_name
        FROM information_schema.columns
        WHERE table_name = :table_name
          AND column_name = 'embedding'
        """,
        table_name=EMBEDDINGS_TABLE_NAME,
    )

    embedding_column = result[0] if result else None

finally:

    conn.close()

print(
    f"Total embeddings:   {total_embeddings}"
=======
2. Documents have corresponding chunks.
3. The embedding column is actually a pgvector `vector` type.

# COMMAND ----------

verification = query_rows(
    f"""
    SELECT
        COUNT(*) AS total_embeddings,
        COUNT(DISTINCT document_id) AS documents_embedded
    FROM {EMBEDDINGS_TABLE_NAME}
    """
>>>>>>> Stashed changes
)

print(
    f"Total embeddings:   {verification[0]['total_embeddings']}"
)

<<<<<<< Updated upstream
if embedding_column:

    print(
        f"Embedding column:   "
        f"{embedding_column[0]}"
    )

    print(
        f"Data type:          "
        f"{embedding_column[1]}"
    )

    print(
        f"UDT:                "
        f"{embedding_column[2]}"
    )
=======
print(
    f"Documents embedded: {verification[0]['documents_embedded']}"
)

# COMMAND ----------

embedding_column = query_rows(
    """
    SELECT
        column_name,
        data_type,
        udt_name
    FROM information_schema.columns
    WHERE table_name = :table_name
      AND column_name = 'embedding'
    """,
    table_name=EMBEDDINGS_TABLE_NAME,
)

if embedding_column:

    print(
        f"Embedding data type: "
        f"{embedding_column[0]['data_type']}"
    )

    print(
        f"Embedding UDT:       "
        f"{embedding_column[0]['udt_name']}"
    )

    if embedding_column[0]["udt_name"] == "vector":
        print("✅ Embedding column is a pgvector VECTOR column.")
    else:
        print("⚠️ Embedding column is not reported as VECTOR.")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
# MAGIC <<<<<<< Updated upstream
# MAGIC ## Verify Vector Dimension
# MAGIC
# MAGIC PostgreSQL/pgvector should report a 384-dimensional vector.

# COMMAND ----------

conn = get_connection()

try:

    result = conn.run(
        f"""
        SELECT
            vector_dims(embedding)
        FROM {EMBEDDINGS_TABLE_NAME}
        WHERE embedding IS NOT NULL
        LIMIT 1
        """
    )

finally:

    conn.close()

if result:

    vector_dimension = result[0][0]

    print(
        f"Vector dimension: {vector_dimension}"
    )

    if vector_dimension != EMBEDDING_DIM:

        raise ValueError(
            f"Expected vector dimension "
            f"{EMBEDDING_DIM}, but database "
            f"contains {vector_dimension}."
        )

    print(
        "Vector dimension validation passed."
    )
=======
## Verify HNSW Index

The SQL setup should have created an HNSW index using
`vector_cosine_ops`.

This is the index used by pgvector to accelerate cosine-distance
retrieval.

# COMMAND ----------

indexes = query_rows(
    f"""
    SELECT
        indexname,
        indexdef
    FROM pg_indexes
    WHERE tablename = :table_name
    ORDER BY indexname
    """,
    table_name=EMBEDDINGS_TABLE_NAME,
)

if indexes:

    for index in indexes:

        print(f"\nIndex: {index['indexname']}")
        print(index["indexdef"])
>>>>>>> Stashed changes

else:

    print(
<<<<<<< Updated upstream
        "No vectors available for dimension validation."
=======
        f"No indexes found for {EMBEDDINGS_TABLE_NAME}. "
        "Verify that the embedding SQL setup was executed."
>>>>>>> Stashed changes
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Cosine Similarity Search
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC This validates that pgvector can perform cosine-distance search.
# MAGIC =======
# MAGIC This validation query confirms that:
# MAGIC
# MAGIC - the embedding column is queryable as a vector
# MAGIC - the `<=>` cosine-distance operator works
# MAGIC - the vector can be ranked by similarity
# MAGIC >>>>>>> Stashed changes
# MAGIC
# MAGIC The production version of this logic lives in:
# MAGIC
# MAGIC ```text
# MAGIC POST /weather/search
# MAGIC ```
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC The query vector must be generated with the same
# MAGIC `all-MiniLM-L6-v2` model.
# MAGIC =======
# MAGIC The Flask endpoint embeds a user's query using the same model and
# MAGIC searches this table using cosine distance.
# MAGIC >>>>>>> Stashed changes

# COMMAND ----------

if embedded_rows:

    test_vector = embedded_rows[0]["embedding"]

    test_vector_string = (
>>>>>>> Stashed changes
        "["
        + ",".join(
            str(float(value))
            for value in vector
        )
        + "]"
    )

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
    with lakebase.get_connection() as conn:
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

def verify_embeddings():
    """Print a simple verification of the stored vector data."""
    rows = lakebase.run_query(
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

<<<<<<< Updated upstream
if __name__ == "__main__":
    main()
=======
# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC The Weather Intelligence embedding pipeline is now:
# MAGIC =======
# MAGIC The complete Weather Intelligence pipeline is:
# MAGIC >>>>>>> Stashed changes
# MAGIC
# MAGIC ```text
# MAGIC National Weather Service API
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/sync
# MAGIC             |
# MAGIC             v
# MAGIC        weather_news
# MAGIC             |
# MAGIC             v
# MAGIC ingest_weather_embeddings
# MAGIC             |
# MAGIC <<<<<<< Updated upstream
# MAGIC      narrative_text
# MAGIC             |
# MAGIC       chunking
# MAGIC =======
# MAGIC       narrative_text
# MAGIC >>>>>>> Stashed changes
# MAGIC             |
# MAGIC      800-char chunks
# MAGIC       100-char overlap
# MAGIC             |
# MAGIC             v
# MAGIC   all-MiniLM-L6-v2
# MAGIC        384 dimensions
# MAGIC             |
# MAGIC             v
# MAGIC     weather_embeddings
# MAGIC             |
# MAGIC       pgvector VECTOR(384)
# MAGIC             |
# MAGIC       HNSW / cosine
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/search
# MAGIC ```
# MAGIC
# MAGIC Example search request:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "query": "flash flood risk this weekend",
# MAGIC   "top_k": 5
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC <<<<<<< Updated upstream
# MAGIC The Flask API should embed the user's query using the same
# MAGIC `all-MiniLM-L6-v2` model and use pgvector's cosine-distance operator
# MAGIC (`<=>`) to retrieve the most semantically relevant weather chunks.
# MAGIC =======
# MAGIC The Flask API should return the most semantically relevant weather
# MAGIC <<<<<<< Updated upstream
# MAGIC chunks ranked by cosine similarity.
# MAGIC
# MAGIC ## Serverless Compatibility
# MAGIC
# MAGIC This notebook uses:
# MAGIC
# MAGIC - `pg8000` for PostgreSQL connectivity
# MAGIC - `sentence-transformers` for embeddings
# MAGIC - direct PostgreSQL inserts
# MAGIC - PostgreSQL `VECTOR(384)` casting
# MAGIC
# MAGIC It intentionally avoids:
# MAGIC
# MAGIC - `psycopg2-binary`
# MAGIC - Spark JDBC writes
# MAGIC - Spark writes to pgvector
# MAGIC - array-to-vector post-processing
# MAGIC =======
# MAGIC chunks ranked using cosine similarity.
# MAGIC >>>>>>> Stashed changes
# MAGIC >>>>>>> Stashed changes
>>>>>>> Stashed changes
