# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///

# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the Weather Intelligence homework.
# MAGIC
# MAGIC It:
# MAGIC
# MAGIC 1. Reads weather documents harvested by the Flask application's
# MAGIC    `/weather/sync` endpoint from the `weather_news` Lakebase table.
# MAGIC 2. Finds weather documents that do not yet have embeddings.
# MAGIC 3. Splits each document's `narrative_text` into overlapping chunks.
# MAGIC 4. Embeds each chunk using
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
# MAGIC 5. Writes embeddings directly to Lakebase using `pg8000`.
# MAGIC 6. Stores embeddings in a pgvector `VECTOR(384)` column.
# MAGIC 7. Verifies cosine-similarity search against the resulting embeddings.
# MAGIC
# MAGIC The pipeline is:
# MAGIC
# MAGIC NWS API
# MAGIC   -> /weather/sync
# MAGIC   -> weather_news
# MAGIC   -> this notebook
# MAGIC   -> weather_embeddings
# MAGIC   -> /weather/search
# MAGIC
# MAGIC ## Lakebase authentication
# MAGIC
# MAGIC This notebook intentionally does NOT use psycopg2.
# MAGIC
# MAGIC `psycopg2-binary` contains native C extensions that can cause
# MAGIC SIGABRT/kernel crashes on Databricks Serverless compute.
# MAGIC
# MAGIC Instead, this notebook uses:
# MAGIC
# MAGIC - `pg8000` for PostgreSQL connectivity
# MAGIC - the Databricks SDK to generate a fresh Lakebase OAuth database
# MAGIC   credential for each connection
# MAGIC - the existing `database/lakebase-url` secret for connection metadata
# MAGIC
# MAGIC Lakebase database credentials are short-lived, so the notebook does not
# MAGIC treat the password contained in the stored URL as a long-lived credential.
# MAGIC
# MAGIC The current Databricks SDK pattern is:
# MAGIC
# MAGIC `WorkspaceClient().postgres.generate_database_credential(...)`
# MAGIC
# MAGIC and the returned token is supplied to PostgreSQL as the password.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC Required packages:
# MAGIC
# MAGIC - `pg8000`: pure-Python PostgreSQL driver
# MAGIC - `sentence-transformers`: text embedding model
# MAGIC
# MAGIC We intentionally do NOT install psycopg2 or psycopg2-binary.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install -q pg8000 sentence-transformers databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC These widgets allow the notebook to be run manually or scheduled as a
# MAGIC Databricks Job without modifying the notebook source.

# COMMAND ----------

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

dbutils.widgets.text(
    "lakebase_secret_scope",
    "database",
    "Lakebase secret scope"
)

dbutils.widgets.text(
    "lakebase_secret_key",
    "lakebase-url",
    "Lakebase URL secret key"
)

# Optional override.
#
# Leave blank to derive the endpoint from the Lakebase hostname.
#
# Example:
# projects/dataexpert-student/branches/production/endpoints/primary
#
dbutils.widgets.text(
    "lakebase_endpoint",
    "",
    "Lakebase endpoint resource name (optional)"
)

WEATHER_TABLE_NAME = dbutils.widgets.get("weather_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")

CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

LAKEBASE_SECRET_SCOPE = dbutils.widgets.get("lakebase_secret_scope")
LAKEBASE_SECRET_KEY = dbutils.widgets.get("lakebase_secret_key")

LAKEBASE_ENDPOINT_OVERRIDE = dbutils.widgets.get(
    "lakebase_endpoint"
).strip()

# The weather assignment specifies this model.
if EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    EMBEDDING_DIM = 384
else:
    raise ValueError(
        f"Unsupported embedding model: {EMBEDDING_MODEL_NAME!r}. "
        "This weather app is configured for "
        "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."
    )

if CHUNK_SIZE <= 0:
    raise ValueError("chunk_size must be greater than zero")

if CHUNK_OVERLAP < 0:
    raise ValueError("chunk_overlap cannot be negative")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("chunk_overlap must be smaller than chunk_size")

if BATCH_SIZE <= 0:
    raise ValueError("batch_size must be greater than zero")

print(f"Weather document table: {WEATHER_TABLE_NAME}")
print(f"Embedding table:        {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model:        {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimensions:   {EMBEDDING_DIM}")
print(f"Chunk size:             {CHUNK_SIZE}")
print(f"Chunk overlap:          {CHUNK_OVERLAP}")
print(f"Batch size:             {BATCH_SIZE}")
print(f"Secret scope:           {LAKEBASE_SECRET_SCOPE}")
print(f"Secret key:             {LAKEBASE_SECRET_KEY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import base64
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import pg8000.native

from databricks.sdk import WorkspaceClient

from sentence_transformers import SentenceTransformer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve Lakebase Connection Metadata
# MAGIC
# MAGIC The existing application secret is:
# MAGIC
# MAGIC `database/lakebase-url`
# MAGIC
# MAGIC The setup script stores the Lakebase connection URL in that secret:
# MAGIC
# MAGIC `postgresql://role:password@host:5432/database?...`
# MAGIC
# MAGIC
# MAGIC We use the URL for:
# MAGIC
# MAGIC - host
# MAGIC - port
# MAGIC - database
# MAGIC - database username
# MAGIC
# MAGIC However, we do NOT use the password from the URL.
# MAGIC
# MAGIC Instead, the password is replaced with a freshly generated OAuth
# MAGIC database credential from the Databricks SDK.

# COMMAND ----------

# DBTITLE 1,Read Lakebase URL from Databricks Secret

w = WorkspaceClient()


def get_lakebase_url() -> str:
    """
    Read the existing Lakebase URL from the Databricks secret.

    The setup script stores this value in:
        scope = database
        key   = lakebase-url
    """

    secret = w.secrets.get_secret(
        scope=LAKEBASE_SECRET_SCOPE,
        key=LAKEBASE_SECRET_KEY,
    )

    if not secret.value:
        raise ValueError(
            f"Secret {LAKEBASE_SECRET_SCOPE}/{LAKEBASE_SECRET_KEY} "
            "exists but contains no value."
        )

    # Databricks secrets are commonly returned base64 encoded.
    try:
        decoded = base64.b64decode(secret.value).decode("utf-8")

        if decoded.startswith("postgresql://"):
            return decoded

    except Exception:
        pass

    # Fall back to treating the returned value as the URL itself.
    if secret.value.startswith("postgresql://"):
        return secret.value

    raise ValueError(
        "The Lakebase secret does not contain a valid PostgreSQL URL."
    )


lakebase_url = get_lakebase_url()

# Do not print the password/token.
parsed = urlparse(lakebase_url)

if not parsed.hostname:
    raise ValueError(
        f"Unable to determine Lakebase hostname from secret URL."
    )

if not parsed.username:
    raise ValueError(
        "The Lakebase URL does not contain a database username."
    )

DB_HOST = parsed.hostname
DB_PORT = parsed.port or 5432
DB_NAME = parsed.path.lstrip("/")

DB_USER = parsed.username

if not DB_NAME:
    raise ValueError(
        "The Lakebase URL does not contain a database name."
    )

print("Lakebase connection metadata loaded.")
print(f"Host:     {DB_HOST}")
print(f"Port:     {DB_PORT}")
print(f"Database: {DB_NAME}")
print(f"User:     {DB_USER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve Lakebase Endpoint
# MAGIC
# MAGIC Lakebase OAuth credentials are generated for a specific endpoint.
# MAGIC
# MAGIC If `lakebase_endpoint` is supplied as a widget, it is used directly.
# MAGIC
# MAGIC Otherwise this notebook derives the endpoint resource name from the
# MAGIC Lakebase hostname.
# MAGIC
# MAGIC For this project, the expected resource is:
# MAGIC
# MAGIC `projects/dataexpert-student/branches/production/endpoints/primary`
# MAGIC
# MAGIC
# MAGIC You can override this through the widget if your endpoint differs.

# COMMAND ----------

# DBTITLE 1,Resolve Endpoint Resource Name

def derive_lakebase_endpoint(hostname: str) -> str:
    """
    Derive the Lakebase project/branch information from a Lakebase hostname.

    Example hostname:

        ep-muddy-hat-d8d8gsj6.database.us-east-2.cloud.databricks.com

    The hostname alone does not reliably expose the endpoint ID, so for this
    project we use the known production/primary endpoint convention unless an
    explicit endpoint override is supplied.
    """

    # Current project configuration.
    #
    # If this project is later moved to another Lakebase project/branch,
    # provide the full endpoint resource through the widget instead.
    return (
        "projects/dataexpert-student/"
        "branches/production/"
        "endpoints/primary"
    )


if LAKEBASE_ENDPOINT_OVERRIDE:
    LAKEBASE_ENDPOINT = LAKEBASE_ENDPOINT_OVERRIDE
else:
    LAKEBASE_ENDPOINT = derive_lakebase_endpoint(DB_HOST)

print(f"Lakebase endpoint: {LAKEBASE_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate a Fresh Lakebase OAuth Credential
# MAGIC
# MAGIC The credential generated by the Databricks SDK is short-lived.
# MAGIC
# MAGIC We generate a new credential whenever `get_connection()` creates a new
# MAGIC database connection.
# MAGIC
# MAGIC This avoids relying on an expired password embedded in the stored
# MAGIC `lakebase-url` secret.
# MAGIC
# MAGIC Databricks documents `generate_database_credential()` as the current
# MAGIC mechanism for obtaining an OAuth credential for Lakebase Postgres.

# COMMAND ----------

# DBTITLE 1,Create Lakebase Connection Helper

def get_database_credential():
    """
    Generate a fresh OAuth database credential for the Lakebase endpoint.
    """

    credential = w.postgres.generate_database_credential(
        endpoint=LAKEBASE_ENDPOINT
    )

    if not credential or not credential.token:
        raise RuntimeError(
            "Databricks did not return a Lakebase database credential."
        )

    return credential


def get_connection():
    """
    Open a pg8000 connection using a freshly generated Lakebase OAuth token.

    The OAuth token is passed to PostgreSQL as the password.
    """

    credential = get_database_credential()

    conn = pg8000.native.Connection(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=credential.token,
        ssl_context=True,
    )

    return conn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Lakebase Connection
# MAGIC
# MAGIC This is the critical connection test.
# MAGIC
# MAGIC Unlike the previous version, this connection:
# MAGIC
# MAGIC - does not use `psycopg2`
# MAGIC - does not use `lakebase.get_connection()`
# MAGIC - does not use the password stored in `lakebase-url`
# MAGIC - generates a fresh OAuth database credential
# MAGIC - uses that credential with `pg8000`

# COMMAND ----------

# DBTITLE 1,Test Fresh OAuth Connection

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
        print(f"Endpoint: {LAKEBASE_ENDPOINT}")

    finally:

        conn.close()

except Exception as e:

    print("❌ Lakebase connection failed.")
    print()
    print(f"Host:     {DB_HOST}")
    print(f"Port:     {DB_PORT}")
    print(f"Database: {DB_NAME}")
    print(f"User:     {DB_USER}")
    print(f"Endpoint: {LAKEBASE_ENDPOINT}")
    print()
    print(f"Error: {e}")

    error_str = str(e).lower()

    if "external authorization failed" in error_str:

        print()
        print("⚠️ External authorization failed.")

        print()
        print("The notebook generated a fresh OAuth credential, so this")
        print("error is no longer being treated as a stale secret password.")

        print()
        print("Check the following:")

        print(
            "1. The Lakebase endpoint exists and is available."
        )

        print(
            "2. The Databricks identity running this notebook has "
            "permission to generate a database credential for the endpoint."
        )

        print(
            "3. The PostgreSQL role/user in the lakebase-url secret "
            f"({DB_USER!r}) is a valid role for this Lakebase database."
        )

        print(
            "4. The endpoint is not blocked by IP ACL/private-link "
            "configuration."
        )

        print(
            "5. The endpoint resource name is correct:"
        )

        print(
            f"   {LAKEBASE_ENDPOINT}"
        )

        print()
        print(
            "If the project uses a different endpoint, set the "
            "'lakebase_endpoint' widget to its full resource name."
        )

    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Required Tables
# MAGIC
# MAGIC Expected source table:
# MAGIC
# MAGIC `weather_news`
# MAGIC
# MAGIC Expected destination table:
# MAGIC
# MAGIC `weather_embeddings`
# MAGIC
# MAGIC Expected `weather_news` columns:
# MAGIC
# MAGIC - `id`
# MAGIC - `location`
# MAGIC - `source_type`
# MAGIC - `headline`
# MAGIC - `narrative_text`
# MAGIC - `issued_at`
# MAGIC - `effective_at`
# MAGIC - `payload`
# MAGIC - `synced_at`
# MAGIC
# MAGIC Expected `weather_embeddings` columns:
# MAGIC
# MAGIC - `id`
# MAGIC - `document_id`
# MAGIC - `chunk_index`
# MAGIC - `chunk_text`
# MAGIC - `embedding`
# MAGIC - `model_name`
# MAGIC - `created_at`

# COMMAND ----------

# DBTITLE 1,Inspect Table Structures

conn = get_connection()

try:

    result = conn.run(
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


for row in result:

    table_name, column_name, data_type, udt_name = row

    print(
        f"{table_name}.{column_name}: "
        f"{data_type} ({udt_name})"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Weather Documents
# MAGIC
# MAGIC The Flask application's `/weather/sync` endpoint should be run before
# MAGIC this notebook.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "locations": ["Chicago, IL", "Austin, TX"],
# MAGIC   "limit": 50
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC That endpoint stores normalized NWS alerts and forecasts in
# MAGIC `weather_news`.

# COMMAND ----------

# DBTITLE 1,Count Weather Documents

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


print(f"Total weather documents:   {total_documents}")
print(f"Documents containing text: {documents_with_text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Unembedded Weather Documents
# MAGIC
# MAGIC A document is considered unembedded when there is no corresponding row
# MAGIC in `weather_embeddings`.
# MAGIC
# MAGIC This makes the notebook safe to run repeatedly:
# MAGIC
# MAGIC - previously embedded documents are skipped
# MAGIC - newly synchronized NWS documents are picked up
# MAGIC - existing embeddings are not duplicated

# COMMAND ----------

# DBTITLE 1,Load Unembedded Documents

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


# Convert tuples to dictionaries.
documents = [
    {
        "id": row[0],
        "location": row[1],
        "source_type": row[2],
        "headline": row[3],
        "narrative_text": row[4],
        "issued_at": row[5],
        "effective_at": row[6],
    }
    for row in documents
]


print(
    f"Found {len(documents)} unembedded weather documents."
)


if documents:

    for document in documents[:5]:

        print(
            f"\nID: {document['id']}"
            f"\nLocation: {document['location']}"
            f"\nType: {document['source_type']}"
            f"\nHeadline: {document['headline']}"
            f"\nText: {document['narrative_text'][:300]}..."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Narrative Text
# MAGIC
# MAGIC NWS narrative text is generally short, but alerts can contain substantial
# MAGIC descriptions and instructions.
# MAGIC
# MAGIC The assignment recommendation is:
# MAGIC
# MAGIC - `CHUNK_SIZE = 800`
# MAGIC - `CHUNK_OVERLAP = 100`
# MAGIC
# MAGIC The overlap preserves context when a sentence or concept crosses a chunk
# MAGIC boundary.

# COMMAND ----------

# DBTITLE 1,Chunk Function

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Split text into overlapping character-based chunks.
    """

    if not text:
        return []

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap

    chunks = []

    for start in range(0, len(text), step):

        chunk = text[
            start:start + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks

# COMMAND ----------

# DBTITLE 1,Create Chunk Rows

chunk_rows = []

for document in documents:

    chunks = chunk_text(
        document["narrative_text"]
    )

    for chunk_index, text in enumerate(chunks):

        chunk_rows.append(
            {
                "document_id": document["id"],
                "location": document["location"],
                "headline": document["headline"],
                "source_type": document["source_type"],
                "chunk_index": chunk_index,
                "chunk_text": text,
            }
        )


print(
    f"Created {len(chunk_rows)} text chunks."
)


if chunk_rows:

    print("\nSample chunk:")
    print(chunk_rows[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Embedding Model
# MAGIC
# MAGIC The same model is used by the Flask `/weather/search` endpoint.
# MAGIC
# MAGIC `all-MiniLM-L6-v2` produces 384-dimensional embeddings.

# COMMAND ----------

# DBTITLE 1,Load Sentence Transformer

print(
    f"Loading {EMBEDDING_MODEL_NAME}..."
)

model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    "Embedding model loaded successfully."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Chunk Embeddings
# MAGIC
# MAGIC Embeddings are generated in batches to control memory usage.
# MAGIC
# MAGIC Embeddings are normalized so cosine similarity can be calculated directly
# MAGIC using pgvector's `<=>` cosine-distance operator.

# COMMAND ----------

# DBTITLE 1,Generate Embeddings

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
                f"Expected {EMBEDDING_DIM}-dimensional vector "
                f"but received {len(vector)} dimensions."
            )

        row_with_embedding = dict(row)

        row_with_embedding["embedding"] = (
            vector.tolist()
        )

        embedded_rows.append(
            row_with_embedding
        )

    processed = min(
        start + BATCH_SIZE,
        len(chunk_rows),
    )

    print(
        f"Embedded {processed}/{len(chunk_rows)} chunks"
    )


print(
    f"\nGenerated {len(embedded_rows)} embeddings."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Generated Embeddings

# COMMAND ----------

# DBTITLE 1,Preview Embedding

if embedded_rows:

    sample = embedded_rows[0]

    print(
        f"Document ID:      {sample['document_id']}"
    )

    print(
        f"Chunk index:       {sample['chunk_index']}"
    )

    print(
        f"Chunk text:        {sample['chunk_text'][:500]}"
    )

    print(
        f"Vector dimensions: {len(sample['embedding'])}"
    )

    print(
        f"Model:             {EMBEDDING_MODEL_NAME}"
    )

else:

    print(
        "No new embeddings were generated."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Lakebase
# MAGIC
# MAGIC The embeddings are written directly through `pg8000`.
# MAGIC
# MAGIC We intentionally do not use:
# MAGIC
# MAGIC - psycopg2
# MAGIC - psycopg2-binary
# MAGIC - Spark JDBC
# MAGIC
# MAGIC
# MAGIC Each chunk receives a stable ID:
# MAGIC
# MAGIC `document_id + "_" + chunk_index`
# MAGIC
# MAGIC
# MAGIC The pgvector value is passed as a PostgreSQL vector literal and explicitly
# MAGIC cast with:
# MAGIC
# MAGIC ```sql
# MAGIC CAST(:embedding AS vector)
# MAGIC ```
# MAGIC
# MAGIC
# MAGIC This avoids attempting to make Spark understand the pgvector type.

# COMMAND ----------

# DBTITLE 1,Insert Embeddings Using pg8000

if not embedded_rows:

    print(
        "No new embeddings to write."
    )

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
            CAST(:embedding AS vector),
            :model_name,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (id) DO UPDATE
        SET
            document_id = EXCLUDED.document_id,
            chunk_index = EXCLUDED.chunk_index,
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = EXCLUDED.created_at
    """

    total_written = 0

    conn = get_connection()

    try:

        for start in range(
            0,
            len(embedded_rows),
            BATCH_SIZE,
        ):

            batch = embedded_rows[
                start:start + BATCH_SIZE
            ]

            for row in batch:

                embedding_id = (
                    f"{row['document_id']}_"
                    f"{row['chunk_index']}"
                )

                vector_string = (
                    "["
                    + ",".join(
                        str(float(value))
                        for value in row["embedding"]
                    )
                    + "]"
                )

                conn.run(
                    insert_sql,
                    id=embedding_id,
                    document_id=row["document_id"],
                    chunk_index=int(
                        row["chunk_index"]
                    ),
                    chunk_text=row["chunk_text"],
                    embedding=vector_string,
                    model_name=EMBEDDING_MODEL_NAME,
                )

                total_written += 1

            print(
                f"Wrote "
                f"{min(start + BATCH_SIZE, len(embedded_rows))}"
                f"/{len(embedded_rows)} embeddings"
            )

        # pg8000.native executes statements directly.
        # Explicit transaction control keeps the batch atomic.
        conn.run("COMMIT")

    except Exception:

        try:
            conn.run("ROLLBACK")
        except Exception:
            pass

        raise

    finally:

        conn.close()


    print(
        f"\nSuccessfully wrote {total_written} embeddings "
        f"to {EMBEDDINGS_TABLE_NAME}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embeddings
# MAGIC
# MAGIC Confirm:
# MAGIC
# MAGIC - total embedding count
# MAGIC - number of documents represented
# MAGIC - embedding column type
# MAGIC - pgvector dimensions

# COMMAND ----------

# DBTITLE 1,Verify Embedding Table

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

    total_embeddings, documents_embedded = result[0]

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
)

print(
    f"Documents embedded: {documents_embedded}"
)


if embedding_column:

    column_name, data_type, udt_name = (
        embedding_column
    )

    print(
        f"Embedding column:   {column_name}"
    )

    print(
        f"Embedding type:     {data_type}"
    )

    print(
        f"Embedding UDT:      {udt_name}"
    )

else:

    print(
        "⚠️ Could not find embedding column."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify pgvector Dimension
# MAGIC
# MAGIC The weather app expects:
# MAGIC
# MAGIC `VECTOR(384)`
# MAGIC
# MAGIC This query inspects the PostgreSQL column definition.

# COMMAND ----------

# DBTITLE 1,Check Vector Dimension

conn = get_connection()

try:

    result = conn.run(
        """
        SELECT
            a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c
          ON c.oid = a.attrelid
        WHERE c.relname = :table_name
          AND a.attname = 'embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """,
        table_name=EMBEDDINGS_TABLE_NAME,
    )

finally:

    conn.close()


if result:

    vector_typmod = result[0][0]

    print(
        f"PostgreSQL vector typmod: {vector_typmod}"
    )

    print(
        f"Expected dimensions:       {EMBEDDING_DIM}"
    )

    # pgvector stores dimension in the type modifier.
    # The exact typmod value can vary by pgvector version, so the
    # definitive validation is the vector_dims() query below.

else:

    print(
        "⚠️ Unable to inspect vector typmod."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Actual Vector Dimensions
# MAGIC
# MAGIC Use pgvector's `vector_dims()` function to verify that stored vectors
# MAGIC actually contain 384 dimensions.

# COMMAND ----------

# DBTITLE 1,Validate Vector Dimensions

conn = get_connection()

try:

    result = conn.run(
        f"""
        SELECT DISTINCT
            vector_dims(embedding)
        FROM {EMBEDDINGS_TABLE_NAME}
        WHERE embedding IS NOT NULL
        """

    )

finally:

    conn.close()


dimensions = [
    row[0]
    for row in result
]


print(
    f"Stored vector dimensions: {dimensions}"
)


if dimensions and dimensions != [EMBEDDING_DIM]:

    raise ValueError(
        f"Expected stored vectors to have "
        f"{EMBEDDING_DIM} dimensions, "
        f"but found {dimensions}."
    )


if dimensions:

    print(
        f"✅ All stored embeddings are {EMBEDDING_DIM}-dimensional."
    )

else:

    print(
        "No vectors are currently stored."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Cosine Similarity Search
# MAGIC
# MAGIC This validates that:
# MAGIC
# MAGIC - the embedding column is a pgvector column
# MAGIC - vectors are compatible with the expected dimension
# MAGIC - the cosine-distance operator `<=>` works
# MAGIC - the stored embeddings can be queried directly from Lakebase
# MAGIC
# MAGIC The production version of this logic lives in:
# MAGIC
# MAGIC `POST /weather/search`

# COMMAND ----------

# DBTITLE 1,Run Similarity Search Test

if embedded_rows:

    test_vector = embedded_rows[0]["embedding"]

    test_vector_string = (
        "["
        + ",".join(
            str(float(value))
            for value in test_vector
        )
        + "]"
    )

    conn = get_connection()

    try:

        results = conn.run(
            f"""
            SELECT
                e.id,
                e.document_id,
                e.chunk_index,
                e.chunk_text,
                1 - (
                    e.embedding <=> CAST(:query_vector AS vector)
                ) AS similarity
            FROM {EMBEDDINGS_TABLE_NAME} e
            ORDER BY
                e.embedding <=> CAST(:query_vector AS vector)
            LIMIT 5
            """,
            query_vector=test_vector_string,
        )

    finally:

        conn.close()


    print(
        "Top 5 similarity results:\n"
    )

    for result in results:

        (
            embedding_id,
            document_id,
            chunk_index,
            chunk_text,
            similarity,
        ) = result

        print(
            f"Similarity: {similarity:.4f}"
        )

        print(
            f"Document:   {document_id}"
        )

        print(
            f"Chunk:      {chunk_index}"
        )

        print(
            f"Text:       {chunk_text[:300]}"
        )

        print(
            "-" * 80
        )

else:

    print(
        "No embeddings available for similarity test."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify HNSW Index
# MAGIC
# MAGIC The weather application should have an HNSW cosine-similarity index on
# MAGIC the `weather_embeddings.embedding` column.
# MAGIC
# MAGIC This cell does not create or modify the index. It simply reports the
# MAGIC indexes currently present on the destination table.

# COMMAND ----------

# DBTITLE 1,Inspect Embedding Indexes

conn = get_connection()

try:

    indexes = conn.run(
        """
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE tablename = :table_name
        ORDER BY indexname
        """,
        table_name=EMBEDDINGS_TABLE_NAME,
    )

finally:

    conn.close()


print(
    f"Indexes on {EMBEDDINGS_TABLE_NAME}:"
)

for index_name, index_definition in indexes:

    print()
    print(
        f"{index_name}:"
    )
    print(
        index_definition
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete
# MAGIC
# MAGIC The weather embedding pipeline is now:
# MAGIC
# MAGIC ```text
# MAGIC National Weather Service API
# MAGIC             |
# MAGIC             v
# MAGIC       /weather/sync
# MAGIC             |
# MAGIC             v
# MAGIC       weather_news
# MAGIC             |
# MAGIC             v
# MAGIC   ingest_weather_embeddings
# MAGIC             |
# MAGIC      chunk narrative_text
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
# MAGIC Example search:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "query": "flash flood risk this weekend",
# MAGIC   "top_k": 5
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC The Flask API should return the most semantically relevant weather
# MAGIC chunks ranked using cosine similarity.
# MAGIC
# MAGIC ## Connection architecture
# MAGIC
# MAGIC ```text
# MAGIC database/lakebase-url
# MAGIC          |
# MAGIC          +--> host
# MAGIC          +--> port
# MAGIC          +--> database
# MAGIC          +--> database user
# MAGIC
# MAGIC WorkspaceClient()
# MAGIC          |
# MAGIC          v
# MAGIC generate_database_credential()
# MAGIC          |
# MAGIC          v
# MAGIC short-lived OAuth token
# MAGIC          |
# MAGIC          v
# MAGIC pg8000
# MAGIC          |
# MAGIC          v
# MAGIC Lakebase PostgreSQL
# MAGIC ```
# MAGIC
# MAGIC No psycopg2 or psycopg2-binary is used by this notebook.
