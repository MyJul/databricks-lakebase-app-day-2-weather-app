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

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install -q pg8000 sentence-transformers databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC The notebook can be run manually or scheduled as a Databricks Job.
# MAGIC
# MAGIC The Lakebase endpoint must be the **Lakebase PostgreSQL endpoint resource
# MAGIC name**, not the old Database Instance name.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```text
# MAGIC projects/dataexpert-student/branches/production/endpoints/primary
# MAGIC ```
# MAGIC
# MAGIC The exact endpoint for your project should be confirmed before running
# MAGIC the notebook.

# COMMAND ----------

dbutils.widgets.text(
    "lakebase_endpoint_name",
    "projects/dataexpert-student/branches/production/endpoints/primary",
    "Lakebase PostgreSQL endpoint"
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

LAKEBASE_ENDPOINT_NAME = dbutils.widgets.get(
    "lakebase_endpoint_name"
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

# Assignment requires all-MiniLM-L6-v2.
if EMBEDDING_MODEL_NAME != "sentence-transformers/all-MiniLM-L6-v2":
    raise ValueError(
        f"Unsupported embedding model: {EMBEDDING_MODEL_NAME!r}. "
        "This application uses "
        "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."
    )

EMBEDDING_DIM = 384

if CHUNK_SIZE <= 0:
    raise ValueError("chunk_size must be greater than zero")

if CHUNK_OVERLAP < 0:
    raise ValueError("chunk_overlap cannot be negative")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "chunk_overlap must be smaller than chunk_size"
    )

if BATCH_SIZE <= 0:
    raise ValueError(
        "batch_size must be greater than zero"
    )

print("Configuration")
print("=" * 70)
print(f"Lakebase endpoint:      {LAKEBASE_ENDPOINT_NAME}")
print(f"Weather table:          {WEATHER_TABLE_NAME}")
print(f"Embedding table:        {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model:        {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimensions:   {EMBEDDING_DIM}")
print(f"Chunk size:             {CHUNK_SIZE}")
print(f"Chunk overlap:          {CHUNK_OVERLAP}")
print(f"Batch size:             {BATCH_SIZE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries
# MAGIC
# MAGIC `pg8000` is the only PostgreSQL driver used by this notebook.

# COMMAND ----------

import os
import pg8000.native

from databricks.sdk import WorkspaceClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Databricks Workspace Client
# MAGIC
# MAGIC The notebook uses the Databricks identity available to the compute
# MAGIC environment.
# MAGIC
# MAGIC No static Lakebase database password is stored in the notebook.

# COMMAND ----------

w = WorkspaceClient()

print("Databricks WorkspaceClient initialized successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve Lakebase Endpoint
# MAGIC
# MAGIC Lakebase Autoscaling Projects use the PostgreSQL endpoint API.
# MAGIC
# MAGIC This is intentionally different from the legacy:
# MAGIC
# MAGIC ```python
# MAGIC w.database.generate_database_credential(
# MAGIC     instance_names=[...]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC We instead use:
# MAGIC
# MAGIC ```python
# MAGIC w.postgres.generate_database_credential(
# MAGIC     endpoint=...
# MAGIC )
# MAGIC ```

# COMMAND ----------

import re

endpoint_pattern = (
    r"^projects/[^/]+/branches/[^/]+/endpoints/[^/]+$"
)

if not re.match(
    endpoint_pattern,
    LAKEBASE_ENDPOINT_NAME
):
    raise ValueError(
        "LAKEBASE_ENDPOINT_NAME does not appear to be a "
        "Lakebase PostgreSQL endpoint resource name.\n\n"
        "Expected format:\n"
        "projects/<project>/branches/<branch>/endpoints/<endpoint>\n\n"
        f"Received:\n{LAKEBASE_ENDPOINT_NAME}"
    )

print(
    "Lakebase endpoint format validated:"
)
print(
    LAKEBASE_ENDPOINT_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Fresh Lakebase Credential
# MAGIC
# MAGIC Lakebase credentials are short-lived.
# MAGIC
# MAGIC The credential is generated when the notebook needs to connect instead
# MAGIC of relying on a potentially stale password stored in a secret.

# COMMAND ----------

def generate_lakebase_credential():
    """
    Generate a fresh short-lived Lakebase PostgreSQL credential.
    """

    credential = (
        w.postgres.generate_database_credential(
            endpoint=LAKEBASE_ENDPOINT_NAME
        )
    )

    return credential


# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse Lakebase Endpoint
# MAGIC
# MAGIC The PostgreSQL endpoint hostname is derived from the Lakebase endpoint
# MAGIC resource.
# MAGIC
# MAGIC The generated credential contains the authentication token.
# MAGIC
# MAGIC For the Weather Intelligence project the endpoint hostname follows the
# MAGIC Lakebase PostgreSQL format:
# MAGIC
# MAGIC ```text
# MAGIC ep-<endpoint>.<project>.<region>.cloud.databricks.com
# MAGIC ```
# MAGIC
# MAGIC The endpoint resource itself is authoritative for authentication.

# COMMAND ----------

def get_connection():
    """
    Create a pg8000 connection using a freshly generated Lakebase credential.

    IMPORTANT:
    This function intentionally does not use:
        - psycopg2
        - lakebase.py
        - a stored database password
        - the legacy Database Instance credential API
    """

    credential = generate_lakebase_credential()

    # The credential returned by the Databricks PostgreSQL API provides
    # the database username and password/token needed by the endpoint.
    #
    # Different SDK versions expose these fields slightly differently,
    # so handle the common representations explicitly.

    if hasattr(credential, "token"):
        password = credential.token
    elif hasattr(credential, "password"):
        password = credential.password
    elif isinstance(credential, dict):
        password = (
            credential.get("token")
            or credential.get("password")
        )
    else:
        raise RuntimeError(
            "Could not determine the Lakebase credential token "
            "from the Databricks SDK response."
        )

    if not password:
        raise RuntimeError(
            "Lakebase credential was generated but no token/password "
            "was returned."
        )

    # Lakebase PostgreSQL endpoint.
    #
    # The endpoint hostname used by the project is resolved from the
    # endpoint resource. For the current Weather Intelligence project,
    # this is the standard Lakebase endpoint hostname.
    #
    # If your endpoint uses a different hostname, set these values in
    # the configuration cell rather than changing the authentication
    # mechanism.

    db_host = os.environ.get(
        "LAKEBASE_DB_HOST"
    )

    db_port = int(
        os.environ.get(
            "LAKEBASE_DB_PORT",
            "5432"
        )
    )

    db_name = os.environ.get(
        "LAKEBASE_DB_NAME",
        "databricks_postgres"
    )

    db_user = os.environ.get(
        "LAKEBASE_DB_USER",
        "student"
    )

    if not db_host:
        raise RuntimeError(
            "LAKEBASE_DB_HOST is not configured.\n\n"
            "Set the PostgreSQL hostname for your Lakebase endpoint "
            "before running this notebook."
        )

    return pg8000.native.Connection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=password,
        ssl_context=True,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connection Configuration
# MAGIC
# MAGIC IMPORTANT:
# MAGIC
# MAGIC The Lakebase PostgreSQL endpoint resource name and the PostgreSQL
# MAGIC hostname are two different values.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```text
# MAGIC Endpoint resource:
# MAGIC projects/dataexpert-student/branches/production/endpoints/primary
# MAGIC
# MAGIC PostgreSQL host:
# MAGIC ep-muddy-hat-d8d8gsj6.database.us-east-2.cloud.databricks.com
# MAGIC ```
# MAGIC
# MAGIC The endpoint resource is used to generate the credential.
# MAGIC
# MAGIC The hostname is used by pg8000 to establish the PostgreSQL connection.

# COMMAND ----------

# If these environment variables are not already configured,
# set them here for the notebook.
#
# IMPORTANT:
# Replace the values below only if your Lakebase endpoint differs.

DB_HOST = os.environ.get(
    "LAKEBASE_DB_HOST",
    "ep-muddy-hat-d8d8gsj6.database.us-east-2.cloud.databricks.com"
)

DB_PORT = int(
    os.environ.get(
        "LAKEBASE_DB_PORT",
        "5432"
    )
)

DB_NAME = os.environ.get(
    "LAKEBASE_DB_NAME",
    "databricks_postgres"
)

DB_USER = os.environ.get(
    "LAKEBASE_DB_USER",
    "student"
)

os.environ["LAKEBASE_DB_HOST"] = DB_HOST
os.environ["LAKEBASE_DB_PORT"] = str(DB_PORT)
os.environ["LAKEBASE_DB_NAME"] = DB_NAME
os.environ["LAKEBASE_DB_USER"] = DB_USER

print("Lakebase PostgreSQL configuration")
print("=" * 70)
print(f"Host:     {DB_HOST}")
print(f"Port:     {DB_PORT}")
print(f"Database: {DB_NAME}")
print(f"User:     {DB_USER}")
print(f"Endpoint: {LAKEBASE_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Lakebase Connection
# MAGIC
# MAGIC This test generates a **fresh credential** immediately before connecting.
# MAGIC
# MAGIC If this fails with:
# MAGIC
# MAGIC ```text
# MAGIC External authorization failed
# MAGIC ```
# MAGIC
# MAGIC the problem is no longer stale credentials stored in a secret.
# MAGIC
# MAGIC The likely causes then become:
# MAGIC
# MAGIC - incorrect Lakebase endpoint resource
# MAGIC - endpoint paused/unavailable
# MAGIC - Databricks App/compute identity does not have access
# MAGIC - network/IP ACL restrictions
# MAGIC - incorrect PostgreSQL hostname
# MAGIC - incorrect database user

# COMMAND ----------

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

        print("Lakebase connection successful.")
        print(f"Database: {database}")
        print(f"User:     {user}")
        print(f"Host:     {DB_HOST}:{DB_PORT}")

    finally:

        conn.close()

except Exception as e:

    print("Lakebase connection FAILED.")
    print()
    print(f"Error: {e}")
    print()
    print("Connection details:")
    print(f"  Endpoint: {LAKEBASE_ENDPOINT_NAME}")
    print(f"  Host:     {DB_HOST}")
    print(f"  Port:     {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User:     {DB_USER}")
    print()
    print(
        "The notebook generated a fresh credential using "
        "WorkspaceClient.postgres.generate_database_credential()."
    )

    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Required Tables
# MAGIC
# MAGIC The Lakebase database should contain:
# MAGIC
# MAGIC ```text
# MAGIC weather_news
# MAGIC weather_embeddings
# MAGIC ```
# MAGIC
# MAGIC Expected `weather_news` columns:
# MAGIC
# MAGIC - id
# MAGIC - location
# MAGIC - source_type
# MAGIC - headline
# MAGIC - narrative_text
# MAGIC - issued_at
# MAGIC - effective_at
# MAGIC - payload
# MAGIC - synced_at
# MAGIC
# MAGIC Expected `weather_embeddings` columns:
# MAGIC
# MAGIC - id
# MAGIC - document_id
# MAGIC - chunk_index
# MAGIC - chunk_text
# MAGIC - embedding
# MAGIC - model_name
# MAGIC - created_at

# COMMAND ----------

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

print("Database schema")
print("=" * 70)

for row in result:

    table_name = row[0]
    column_name = row[1]
    data_type = row[2]
    udt_name = row[3]

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

print(
    f"Documents containing text: {documents_with_text}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Unembedded Weather Documents
# MAGIC
# MAGIC A document is considered unembedded when no corresponding
# MAGIC `weather_embeddings` row exists.
# MAGIC
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

if documents:

    for document in documents[:5]:

        print()
        print(f"ID:       {document['id']}")
        print(f"Location: {document['location']}")
        print(f"Type:     {document['source_type']}")
        print(f"Headline: {document['headline']}")
        print(
            f"Text:     "
            f"{document['narrative_text'][:300]}..."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Narrative Text
# MAGIC
# MAGIC NWS narrative text is split into overlapping character-based chunks.
# MAGIC
# MAGIC Default:
# MAGIC
# MAGIC ```text
# MAGIC Chunk size:    800
# MAGIC Chunk overlap: 100
# MAGIC ```

# COMMAND ----------

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

        if start + chunk_size >= len(text):

            break

    return chunks


# COMMAND ----------

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

    print()
    print("Sample chunk:")
    print(chunk_rows[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Embedding Model
# MAGIC
# MAGIC `all-MiniLM-L6-v2` produces 384-dimensional vectors.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

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
# MAGIC Embeddings are normalized so cosine similarity can be used consistently
# MAGIC during pgvector search.

# COMMAND ----------

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
                f"Expected {EMBEDDING_DIM}-dimensional "
                f"vector but received "
                f"{len(vector)} dimensions."
            )

        row_with_embedding = dict(row)

        row_with_embedding[
            "embedding"
        ] = vector.tolist()

        embedded_rows.append(
            row_with_embedding
        )

    print(
        f"Embedded "
        f"{min(start + BATCH_SIZE, len(chunk_rows))}"
        f"/{len(chunk_rows)} chunks"
    )

print()
print(
    f"Generated {len(embedded_rows)} embeddings."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Generated Embeddings

# COMMAND ----------

if embedded_rows:

    sample = embedded_rows[0]

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Lakebase
# MAGIC
# MAGIC Embeddings are written directly using pg8000.
# MAGIC
# MAGIC No Spark JDBC write is used.
# MAGIC
# MAGIC No psycopg2 is used.
# MAGIC
# MAGIC Each chunk receives a stable ID:
# MAGIC
# MAGIC ```text
# MAGIC document_id + "_" + chunk_index
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
            created_at = CURRENT_TIMESTAMP
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
                    chunk_index=row["chunk_index"],
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

    finally:

        conn.close()

    print()
    print(
        f"Successfully wrote "
        f"{total_written} embeddings "
        f"to {EMBEDDINGS_TABLE_NAME}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embeddings
# MAGIC
# MAGIC Confirm:
# MAGIC
# MAGIC 1. Embeddings exist.
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
)

print(
    f"Documents embedded: {documents_embedded}"
)

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

# COMMAND ----------

# MAGIC %md
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

else:

    print(
        "No vectors available for dimension validation."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Cosine Similarity Search
# MAGIC
# MAGIC This validates that pgvector can perform cosine-distance search.
# MAGIC
# MAGIC The production version of this logic lives in:
# MAGIC
# MAGIC ```text
# MAGIC POST /weather/search
# MAGIC ```
# MAGIC
# MAGIC The query vector must be generated with the same
# MAGIC `all-MiniLM-L6-v2` model.

# COMMAND ----------

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
                    e.embedding
                    <=>
                    CAST(:query_vector AS vector)
                ) AS similarity
            FROM {EMBEDDINGS_TABLE_NAME} e
            ORDER BY
                e.embedding
                <=>
                CAST(:query_vector AS vector)
            LIMIT 5
            """,
            query_vector=test_vector_string,
        )

    finally:

        conn.close()

    print(
        "Top 5 cosine similarity results:"
    )

    print()

    for result in results:

        embedding_id = result[0]
        document_id = result[1]
        chunk_index = result[2]
        chunk_text = result[3]
        similarity = result[4]

        print(
            f"Similarity: {similarity:.4f}"
        )

        print(
            f"Embedding:  {embedding_id}"
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
# MAGIC ## Pipeline Complete
# MAGIC
# MAGIC The Weather Intelligence embedding pipeline is now:
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
# MAGIC ingest_weather_embeddings
# MAGIC             |
# MAGIC      narrative_text
# MAGIC             |
# MAGIC       chunking
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
# MAGIC Example search request:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "query": "flash flood risk this weekend",
# MAGIC   "top_k": 5
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC The Flask API should embed the user's query using the same
# MAGIC `all-MiniLM-L6-v2` model and use pgvector's cosine-distance operator
# MAGIC (`<=>`) to retrieve the most semantically relevant weather chunks.
