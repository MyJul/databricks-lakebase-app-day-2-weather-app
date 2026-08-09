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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC Required packages:
# MAGIC
# MAGIC - **pg8000**: Pure-Python PostgreSQL/Lakebase driver
# MAGIC - **sentence-transformers**: Embedding model library
# MAGIC
# MAGIC The NWS API is called by the Flask application's `weather_client.py`.
# MAGIC This notebook only processes documents already stored in Lakebase.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install -q pg8000 sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC These widgets allow the notebook to be run manually or scheduled as a
# MAGIC Databricks Job without modifying the source code.

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
    "database_secret_scope",
    "database",
    "Lakebase secret scope"
)

dbutils.widgets.text(
    "database_secret_key",
    "lakebase-url",
    "Lakebase secret key"
)

WEATHER_TABLE_NAME = dbutils.widgets.get("weather_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")

CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

DATABASE_SECRET_SCOPE = dbutils.widgets.get("database_secret_scope")
DATABASE_SECRET_KEY = dbutils.widgets.get("database_secret_key")

# The assignment specifies all-MiniLM-L6-v2 / 384 dimensions.
if EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    EMBEDDING_DIM = 384
else:
    raise ValueError(
        f"Unsupported embedding model: {EMBEDDING_MODEL_NAME!r}. "
        "This weather app is configured for "
        "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."
    )

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("chunk_overlap must be smaller than chunk_size")

if CHUNK_SIZE <= 0:
    raise ValueError("chunk_size must be greater than zero")

if BATCH_SIZE <= 0:
    raise ValueError("batch_size must be greater than zero")

print(f"Weather document table: {WEATHER_TABLE_NAME}")
print(f"Embedding table:        {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model:        {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimensions:   {EMBEDDING_DIM}")
print(f"Chunk size:              {CHUNK_SIZE}")
print(f"Chunk overlap:           {CHUNK_OVERLAP}")
print(f"Batch size:              {BATCH_SIZE}")
print(f"Secret scope:            {DATABASE_SECRET_SCOPE}")
print(f"Secret key:              {DATABASE_SECRET_KEY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve Lakebase Connection
# MAGIC
# MAGIC The Lakebase connection URL is stored in the Databricks secret:
# MAGIC
# MAGIC ```text
# MAGIC scope = database
# MAGIC key   = lakebase-url
# MAGIC ```
# MAGIC
# MAGIC The value is the same Base64-encoded PostgreSQL URL used by the
# MAGIC application, for example:
# MAGIC
# MAGIC ```text
# MAGIC postgresql://role:password@host:5432/database?sslmode=require
# MAGIC ```
# MAGIC
# MAGIC We intentionally create the connection directly with `pg8000`.
# MAGIC
# MAGIC We do NOT import `lakebase.py` here because an implementation of
# MAGIC `lakebase.py` that imports `psycopg2-binary` can cause Serverless kernel
# MAGIC crashes.

# COMMAND ----------

import base64
import re
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

import pg8000.native

w = WorkspaceClient()


def get_lakebase_url() -> str:
    """Read and decode the Lakebase PostgreSQL URL from Databricks Secrets."""
    secret = w.secrets.get_secret(
        scope=DATABASE_SECRET_SCOPE,
        key=DATABASE_SECRET_KEY,
    )

    if not secret.value:
        raise ValueError(
            f"Secret {DATABASE_SECRET_SCOPE}/{DATABASE_SECRET_KEY} is empty."
        )

    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

if not parsed.hostname:
    raise ValueError("Lakebase URL does not contain a hostname.")

if not parsed.username:
    raise ValueError("Lakebase URL does not contain a username.")

if not parsed.password:
    raise ValueError("Lakebase URL does not contain a password.")

if not parsed.path or parsed.path == "/":
    raise ValueError("Lakebase URL does not contain a database name.")

DB_HOST = parsed.hostname
DB_PORT = parsed.port or 5432
DB_NAME = parsed.path.lstrip("/")
DB_USER = parsed.username
DB_PASSWORD = parsed.password

print(f"Lakebase host:     {DB_HOST}")
print(f"Lakebase port:     {DB_PORT}")
print(f"Lakebase database: {DB_NAME}")
print(f"Lakebase user:     {DB_USER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PostgreSQL Identifier Validation
# MAGIC
# MAGIC Table names come from Databricks widgets. They are used in SQL
# MAGIC identifiers rather than SQL parameters, so validate them before using
# MAGIC them in queries.

# COMMAND ----------

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(value: str, name: str) -> str:
    """Validate a PostgreSQL table identifier supplied through a widget."""
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(
            f"Invalid {name}: {value!r}. "
            "Only letters, numbers, and underscores are allowed, "
            "and the name must not begin with a number."
        )
    return value


WEATHER_TABLE_NAME = validate_identifier(
    WEATHER_TABLE_NAME,
    "weather_table_name",
)

EMBEDDINGS_TABLE_NAME = validate_identifier(
    EMBEDDINGS_TABLE_NAME,
    "embeddings_table_name",
)

print("Table names validated successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## pg8000 Connection Helper
# MAGIC
# MAGIC `pg8000.native.Connection` is used instead of psycopg2.
# MAGIC
# MAGIC SSL is enabled using:
# MAGIC
# MAGIC ```python
# MAGIC ssl_context=True
# MAGIC ```
# MAGIC
# MAGIC which provides the equivalent secure connection behavior needed for
# MAGIC the Lakebase PostgreSQL endpoint.

# COMMAND ----------

def get_connection():
    """Return a new pg8000 native Lakebase connection."""
    return pg8000.native.Connection(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        ssl_context=True,
    )


def query_rows(sql: str, **params):
    """
    Execute a SELECT statement with pg8000 and return dictionaries.

    pg8000.native returns rows as tuples. We retrieve the column names from
    the result metadata and convert each row into a dictionary so the rest
    of the notebook can use named fields.
    """
    conn = get_connection()

    try:
        result = conn.run(sql, **params)

        if not result:
            return []

        columns = [column["name"] for column in conn.columns]

        return [
            dict(zip(columns, row))
            for row in result
        ]

    finally:
        conn.close()


def execute_sql(sql: str, **params):
    """Execute a write/DDL statement using pg8000."""
    conn = get_connection()

    try:
        result = conn.run(sql, **params)
        conn.commit()
        return result

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Lakebase Connection

# COMMAND ----------

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

finally:
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Required Tables
# MAGIC
# MAGIC The SQL setup files should already have created:
# MAGIC
# MAGIC ### `weather_news`
# MAGIC
# MAGIC Expected columns:
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
# MAGIC ### `weather_embeddings`
# MAGIC
# MAGIC Expected columns:
# MAGIC
# MAGIC - `id`
# MAGIC - `document_id`
# MAGIC - `chunk_index`
# MAGIC - `chunk_text`
# MAGIC - `embedding`
# MAGIC - `model_name`
# MAGIC - `created_at`

# COMMAND ----------

table_columns = query_rows(
    """
    SELECT
        table_name,
        column_name,
        data_type,
        udt_name
    FROM information_schema.columns
    WHERE table_name IN (:weather_table, :embeddings_table)
    ORDER BY table_name, ordinal_position
    """,
    weather_table=WEATHER_TABLE_NAME,
    embeddings_table=EMBEDDINGS_TABLE_NAME,
)

if not table_columns:
    raise RuntimeError(
        f"No columns found for {WEATHER_TABLE_NAME!r} or "
        f"{EMBEDDINGS_TABLE_NAME!r}. "
        "Run the required SQL setup files first."
    )

for row in table_columns:
    print(
        f"{row['table_name']}.{row['column_name']}: "
        f"{row['data_type']} ({row['udt_name']})"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embedding Column
# MAGIC
# MAGIC The embedding column must be a PostgreSQL `vector` column.
# MAGIC
# MAGIC The expected configuration is:
# MAGIC
# MAGIC ```text
# MAGIC vector(384)
# MAGIC ```
# MAGIC
# MAGIC The SQL setup should also create an HNSW index using
# MAGIC `vector_cosine_ops`.

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

if not embedding_column:
    raise RuntimeError(
        f"The {EMBEDDINGS_TABLE_NAME}.embedding column was not found."
    )

print(f"Embedding data type: {embedding_column[0]['data_type']}")
print(f"Embedding UDT:       {embedding_column[0]['udt_name']}")

if embedding_column[0]["udt_name"] != "vector":
    raise RuntimeError(
        f"{EMBEDDINGS_TABLE_NAME}.embedding is not a pgvector column. "
        f"Expected udt_name='vector', received "
        f"{embedding_column[0]['udt_name']!r}."
    )

print("✅ pgvector column verified.")

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
# MAGIC That endpoint stores normalized NWS alerts and forecasts in
# MAGIC `weather_news`.

# COMMAND ----------

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

print(f"Total weather documents:   {total_documents}")
print(f"Documents containing text: {documents_with_text}")

if total_documents == 0:
    print(
        "\n⚠️ No weather documents are currently available.\n"
        "Run POST /weather/sync from the Flask application first."
    )

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
# MAGIC - existing embeddings are not unnecessarily duplicated
# MAGIC
# MAGIC Because one document may produce multiple chunks, the existence check
# MAGIC is performed at the document level.

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

if documents:
    print("\nSample documents:")

    for document in documents[:5]:
        narrative = document["narrative_text"] or ""

        print(
            f"\nID:       {document['id']}"
            f"\nLocation: {document['location']}"
            f"\nType:     {document['source_type']}"
            f"\nHeadline: {document['headline']}"
            f"\nText:     {narrative[:300]}..."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Narrative Text
# MAGIC
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

    for start in range(0, len(text), step):

        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


chunk_rows = []

for document in documents:

    chunks = chunk_text(
        document["narrative_text"],
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
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

print(f"Created {len(chunk_rows)} text chunks.")

if chunk_rows:

    print("\nSample chunk:")
    print(f"Document: {chunk_rows[0]['document_id']}")
    print(f"Chunk:    {chunk_rows[0]['chunk_index']}")
    print(f"Text:     {chunk_rows[0]['chunk_text'][:500]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Embedding Model
# MAGIC
# MAGIC The same model is used by the Flask `/weather/search` endpoint.
# MAGIC
# MAGIC `sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional
# MAGIC embeddings.

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
# MAGIC `normalize_embeddings=True` normalizes the vectors. This is compatible
# MAGIC with cosine similarity using pgvector's `<=>` operator.

# COMMAND ----------

embedded_rows = []

if not chunk_rows:

    print("No chunks require embedding.")

else:

    for start in range(0, len(chunk_rows), BATCH_SIZE):

        batch = chunk_rows[start:start + BATCH_SIZE]

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

        print(
            f"Embedded {processed}/{len(chunk_rows)} chunks"
        )

print(f"\nGenerated {len(embedded_rows)} embeddings.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Generated Embedding
# MAGIC
# MAGIC This confirms that the embedding has the expected 384 dimensions before
# MAGIC anything is written to Lakebase.

# COMMAND ----------

if embedded_rows:

    sample = embedded_rows[0]

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Lakebase
# MAGIC
# MAGIC Embeddings are written directly through `pg8000`.
# MAGIC
# MAGIC No Spark JDBC is used.
# MAGIC
# MAGIC No `psycopg2` is used.
# MAGIC
# MAGIC No post-processing array-to-vector conversion is required.
# MAGIC
# MAGIC Each chunk receives a stable ID:
# MAGIC
# MAGIC ```text
# MAGIC document_id + "_" + chunk_index
# MAGIC ```
# MAGIC
# MAGIC The `weather_embeddings.id` primary key makes the operation safely
# MAGIC repeatable.
# MAGIC
# MAGIC The `ON CONFLICT` clause updates the existing chunk if the source text
# MAGIC or embedding changes.

# COMMAND ----------

if not insert_rows:

    print("No new embeddings to write.")

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
            created_at = EXCLUDED.created_at
    """

    total_written = 0

    conn = get_connection()

    try:

        for start in range(
            0,
            len(insert_rows),
            BATCH_SIZE,
        ):

            batch = insert_rows[
                start:start + BATCH_SIZE
            ]

            for row in batch:

                conn.run(
                    insert_sql,
                    **row,
                )

                total_written += 1

            conn.commit()

            processed = min(
                start + BATCH_SIZE,
                len(insert_rows),
            )

            print(
                f"Wrote {processed}/{len(insert_rows)} embeddings"
            )

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    print(
        f"\n✅ Successfully wrote {total_written} embeddings "
        f"to {EMBEDDINGS_TABLE_NAME}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embeddings
# MAGIC
# MAGIC Confirm:
# MAGIC
# MAGIC 1. Embeddings exist.
# MAGIC 2. Documents have corresponding chunks.
# MAGIC 3. The embedding column is actually a pgvector `vector` type.

# COMMAND ----------

verification = query_rows(
    f"""
    SELECT
        COUNT(*) AS total_embeddings,
        COUNT(DISTINCT document_id) AS documents_embedded
    FROM {EMBEDDINGS_TABLE_NAME}
    """
)

print(
    f"Total embeddings:   {verification[0]['total_embeddings']}"
)

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify HNSW Index
# MAGIC
# MAGIC The SQL setup should have created an HNSW index using
# MAGIC `vector_cosine_ops`.
# MAGIC
# MAGIC This is the index used by pgvector to accelerate cosine-distance
# MAGIC retrieval.

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

else:

    print(
        f"No indexes found for {EMBEDDINGS_TABLE_NAME}. "
        "Verify that the embedding SQL setup was executed."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Cosine Similarity Search
# MAGIC
# MAGIC This validation query confirms that:
# MAGIC
# MAGIC - the embedding column is queryable as a vector
# MAGIC - the `<=>` cosine-distance operator works
# MAGIC - the vector can be ranked by similarity
# MAGIC
# MAGIC The production version of this logic lives in:
# MAGIC
# MAGIC ```text
# MAGIC POST /weather/search
# MAGIC ```
# MAGIC
# MAGIC The Flask endpoint embeds a user's query using the same model and
# MAGIC searches this table using cosine distance.

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

    results = query_rows(
        f"""
        SELECT
            e.id,
            e.document_id,
            e.chunk_index,
            e.chunk_text,
            1 - (
                e.embedding <=> :query_vector::vector
            ) AS similarity
        FROM {EMBEDDINGS_TABLE_NAME} e
        ORDER BY
            e.embedding <=> :query_vector::vector
        LIMIT 5
        """,
        query_vector=test_vector_string,
    )

    print("Top 5 cosine-similarity results:\n")

    for result in results:

        print(
            f"Similarity: {float(result['similarity']):.4f}"
        )

        print(
            f"Document:   {result['document_id']}"
        )

        print(
            f"Chunk:      {result['chunk_index']}"
        )

        print(
            f"Text:       {result['chunk_text'][:300]}"
        )

        print("-" * 80)

else:

    print(
        "No newly generated embeddings available for "
        "the similarity test."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete
# MAGIC
# MAGIC The complete Weather Intelligence pipeline is:
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
# MAGIC   ingest_weather_embeddings
# MAGIC             |
# MAGIC       narrative_text
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
# MAGIC The Flask API should return the most semantically relevant weather
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
