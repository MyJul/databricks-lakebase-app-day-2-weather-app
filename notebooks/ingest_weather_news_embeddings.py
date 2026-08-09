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
# MAGIC 1. Reads weather documents previously harvested from the National Weather
# MAGIC    Service (NWS) API and stored in the `weather_news` Lakebase table.
# MAGIC 2. Finds documents that do not yet have embeddings.
# MAGIC 3. Splits each document's `narrative_text` into overlapping chunks.
# MAGIC 4. Embeds each chunk using
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
# MAGIC 5. Writes the chunk embeddings directly to Lakebase using pg8000.
# MAGIC 6. Stores embeddings in a pgvector `VECTOR(384)` column.
# MAGIC 7. Creates an HNSW cosine-similarity index for vector search.
# MAGIC 8. Validates the resulting embeddings with a cosine similarity query.
# MAGIC
# MAGIC The Flask application's `/weather/sync` endpoint is responsible for
# MAGIC harvesting NWS data and populating `weather_news`.
# MAGIC
# MAGIC The resulting pipeline is:
# MAGIC
# MAGIC NWS API
# MAGIC   -> /weather/sync
# MAGIC   -> weather_news
# MAGIC   -> this notebook
# MAGIC   -> weather_embeddings
# MAGIC   -> /weather/search
# MAGIC
# MAGIC IMPORTANT:
# MAGIC
# MAGIC This notebook intentionally does NOT use psycopg2 or Spark JDBC.
# MAGIC
# MAGIC PostgreSQL/Lakebase connections are made with pg8000, a pure-Python
# MAGIC PostgreSQL driver that avoids native C-extension issues on Databricks
# MAGIC Serverless compute.
# MAGIC
# MAGIC Authentication uses a fresh Lakebase OAuth database credential generated
# MAGIC through the Databricks SDK.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC Required packages:
# MAGIC
# MAGIC - pg8000: Pure-Python PostgreSQL/Lakebase driver
# MAGIC - sentence-transformers: Text embedding model
# MAGIC - databricks-sdk: Generates fresh Lakebase OAuth database credentials
# MAGIC
# MAGIC The NWS API itself is called by the Flask application's
# MAGIC `weather_client.py`, not by this notebook.

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
# MAGIC Databricks Job without changing the source code.

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

# Lakebase connection settings.
#
# These defaults match the Lakebase endpoint currently being used by the
# weather application. They can be overridden when the notebook is run as
# a Databricks Job.

dbutils.widgets.text(
    "db_host",
    "ep-muddy-hat-d8d8gsj6.database.us-east-2.cloud.databricks.com",
    "Lakebase PostgreSQL host"
)

dbutils.widgets.text(
    "db_port",
    "5432",
    "Lakebase PostgreSQL port"
)

dbutils.widgets.text(
    "db_name",
    "databricks_postgres",
    "Lakebase database"
)

dbutils.widgets.text(
    "db_user",
    "student",
    "Lakebase PostgreSQL user/role"
)

dbutils.widgets.text(
    "lakebase_endpoint",
    "projects/dataexpert-student/branches/production/endpoints/primary",
    "Lakebase endpoint resource"
)

WEATHER_TABLE_NAME = dbutils.widgets.get("weather_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")

EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")

CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

DB_HOST = dbutils.widgets.get("db_host")
DB_PORT = int(dbutils.widgets.get("db_port"))
DB_NAME = dbutils.widgets.get("db_name")
DB_USER = dbutils.widgets.get("db_user")

LAKEBASE_ENDPOINT = dbutils.widgets.get("lakebase_endpoint")

# The assignment specifies all-MiniLM-L6-v2 / 384 dimensions.
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
print(f"Chunk size:              {CHUNK_SIZE}")
print(f"Chunk overlap:           {CHUNK_OVERLAP}")
print(f"Batch size:              {BATCH_SIZE}")
print()
print(f"Lakebase host:           {DB_HOST}")
print(f"Lakebase port:           {DB_PORT}")
print(f"Lakebase database:       {DB_NAME}")
print(f"Lakebase user:           {DB_USER}")
print(f"Lakebase endpoint:       {LAKEBASE_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakebase Connection
# MAGIC
# MAGIC Lakebase authentication uses a fresh OAuth database credential generated
# MAGIC through the Databricks SDK.
# MAGIC
# MAGIC The generated database credential is used as the PostgreSQL password.
# MAGIC
# MAGIC A fresh credential is generated whenever `get_connection()` opens a new
# MAGIC PostgreSQL connection. This avoids relying on an expired credential stored
# MAGIC in a Databricks secret.
# MAGIC
# MAGIC pg8000 is used instead of psycopg2 because pg8000 is pure Python and does
# MAGIC not depend on the native PostgreSQL C extensions that can cause problems
# MAGIC on Databricks Serverless compute.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import pg8000.native

# Databricks SDK client.
#
# In a Databricks notebook, WorkspaceClient() uses the notebook's available
# Databricks authentication context.
w = WorkspaceClient()


def generate_database_credential():
    """
    Generate a fresh Lakebase OAuth database credential.

    The returned token is used as the PostgreSQL password.
    """

    credential = w.postgres.generate_database_credential(
        endpoint=LAKEBASE_ENDPOINT
    )

    if not credential or not credential.token:
        raise RuntimeError(
            "Databricks SDK returned a database credential without a token."
        )

    print(
        "Generated fresh Lakebase database credential "
        f"(expires: {credential.expire_time})"
    )

    return credential


def get_connection():
    """
    Open a new pg8000 PostgreSQL connection using a fresh Lakebase OAuth token.
    """

    credential = generate_database_credential()

    conn = pg8000.native.Connection(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=credential.token,
        ssl_context=True,
    )

    return conn


print("Lakebase connection helper initialized.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Lakebase Connection
# MAGIC
# MAGIC This cell confirms that:
# MAGIC
# MAGIC 1. The Databricks SDK can generate a Lakebase database credential.
# MAGIC 2. pg8000 can authenticate to the Lakebase PostgreSQL endpoint.
# MAGIC 3. The configured database and user are valid.

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

        if not result:
            raise RuntimeError(
                "Connection succeeded but the validation query returned no rows."
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
    print()
    print(f"Error: {e}")
    print()
    print("Connection configuration:")
    print(f"  Host:     {DB_HOST}")
    print(f"  Port:     {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User:     {DB_USER}")
    print(f"  Endpoint: {LAKEBASE_ENDPOINT}")
    print()
    print(
        "If this reports 'External authorization failed', verify that:"
    )
    print(
        "  1. The Lakebase endpoint exists and is accessible."
    )
    print(
        "  2. The configured DB_USER is a valid PostgreSQL role."
    )
    print(
        "  3. That role is authorized to access this Lakebase endpoint."
    )
    print(
        "  4. The Databricks identity running this notebook has permission "
        "to generate a database credential."
    )

    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Required Source Table
# MAGIC
# MAGIC The Flask application's `/weather/sync` endpoint should already have
# MAGIC created and populated:
# MAGIC
# MAGIC `weather_news`
# MAGIC
# MAGIC Expected columns include:
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

# COMMAND ----------

conn = get_connection()

try:

    table_exists_result = conn.run(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = :table_name
        )
        """,
        table_name=WEATHER_TABLE_NAME,
    )

    source_table_exists = table_exists_result[0][0]

finally:
    conn.close()

if not source_table_exists:
    raise RuntimeError(
        f"Required source table {WEATHER_TABLE_NAME!r} does not exist."
    )

print(f"✅ Source table exists: {WEATHER_TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Weather Table Structure

# COMMAND ----------

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
        WHERE table_schema = 'public'
          AND table_name = :table_name
        ORDER BY ordinal_position
        """,
        table_name=WEATHER_TABLE_NAME,
    )

finally:
    conn.close()

print(f"Columns in {WEATHER_TABLE_NAME}:")

for row in table_columns:

    table_name, column_name, data_type, udt_name = row

    print(
        f"  {table_name}.{column_name}: "
        f"{data_type} ({udt_name})"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Weather Documents
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

conn = get_connection()

try:

    total_result = conn.run(
        f"""
        SELECT COUNT(*)
        FROM {WEATHER_TABLE_NAME}
        """
    )

    total_documents = total_result[0][0]

    text_result = conn.run(
        f"""
        SELECT COUNT(*)
        FROM {WEATHER_TABLE_NAME}
        WHERE narrative_text IS NOT NULL
          AND TRIM(narrative_text) <> ''
        """
    )

    documents_with_text = text_result[0][0]

finally:
    conn.close()

print(f"Total weather documents:       {total_documents}")
print(f"Documents containing text:     {documents_with_text}")

if total_documents == 0:

    print()
    print("⚠️ No weather documents were found.")
    print(
        "Run the Flask application's /weather/sync endpoint before "
        "running the embedding pipeline."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create / Verify Weather Embeddings Table
# MAGIC
# MAGIC The weather application uses:
# MAGIC
# MAGIC `weather_embeddings`
# MAGIC
# MAGIC The embedding column is PostgreSQL pgvector:
# MAGIC
# MAGIC `VECTOR(384)`
# MAGIC
# MAGIC An HNSW index using cosine distance is created for semantic search.
# MAGIC
# MAGIC If the table already exists, this notebook does not overwrite it.

# COMMAND ----------

conn = get_connection()

try:

    # Enable pgvector if it is not already enabled.
    conn.run(
        """
        CREATE EXTENSION IF NOT EXISTS vector
        """
    )

    # Create the destination table if it does not exist.
    #
    # The table intentionally uses VECTOR(384) because
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings.
    conn.run(
        f"""
        CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT {EMBEDDINGS_TABLE_NAME}_document_chunk_unique
                UNIQUE (document_id, chunk_index)
        )
        """
    )

    # Create an HNSW index for cosine similarity search.
    conn.run(
        f"""
        CREATE INDEX IF NOT EXISTS
            {EMBEDDINGS_TABLE_NAME}_embedding_hnsw_idx
        ON {EMBEDDINGS_TABLE_NAME}
        USING hnsw (embedding vector_cosine_ops)
        """
    )

finally:
    conn.close()

print(f"✅ Embedding table verified: {EMBEDDINGS_TABLE_NAME}")
print(f"✅ pgvector dimension:       {EMBEDDING_DIM}")
print("✅ HNSW cosine index verified.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Embeddings Table Structure

# COMMAND ----------

conn = get_connection()

try:

    embedding_columns = conn.run(
        """
        SELECT
            table_name,
            column_name,
            data_type,
            udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = :table_name
        ORDER BY ordinal_position
        """,
        table_name=EMBEDDINGS_TABLE_NAME,
    )

finally:
    conn.close()

print(f"Columns in {EMBEDDINGS_TABLE_NAME}:")

for row in embedding_columns:

    table_name, column_name, data_type, udt_name = row

    print(
        f"  {table_name}.{column_name}: "
        f"{data_type} ({udt_name})"
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
# MAGIC - Previously embedded documents are skipped.
# MAGIC - Newly synchronized NWS documents are picked up.
# MAGIC - Existing embeddings are not duplicated.
# MAGIC
# MAGIC Each document can contain multiple chunks.

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
        """
    )

finally:
    conn.close()

print(f"Found {len(documents)} unembedded weather documents.")

if documents:

    for document in documents[:5]:

        (
            document_id,
            location,
            source_type,
            headline,
            narrative_text,
            issued_at,
            effective_at,
            synced_at,
        ) = document

        print(
            f"\nID: {document_id}"
            f"\nLocation: {location}"
            f"\nType: {source_type}"
            f"\nHeadline: {headline}"
            f"\nText: {str(narrative_text)[:300]}..."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Narrative Text
# MAGIC
# MAGIC NWS narrative text is generally short, but alerts can contain substantial
# MAGIC descriptions and instructions.
# MAGIC
# MAGIC We use:
# MAGIC
# MAGIC - `CHUNK_SIZE = 800`
# MAGIC - `CHUNK_OVERLAP = 100`
# MAGIC
# MAGIC The overlap preserves some context when a sentence or concept crosses a
# MAGIC chunk boundary.

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

    text = str(text).strip()

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

    (
        document_id,
        location,
        source_type,
        headline,
        narrative_text,
        issued_at,
        effective_at,
        synced_at,
    ) = document

    chunks = chunk_text(narrative_text)

    for chunk_index, text in enumerate(chunks):

        chunk_rows.append(
            {
                "document_id": document_id,
                "location": location,
                "headline": headline,
                "source_type": source_type,
                "chunk_index": chunk_index,
                "chunk_text": text,
            }
        )

print(f"Created {len(chunk_rows)} text chunks.")

if chunk_rows:

    print("\nSample chunk:")

    print(
        f"Document ID: {chunk_rows[0]['document_id']}"
    )

    print(
        f"Chunk index: {chunk_rows[0]['chunk_index']}"
    )

    print(
        f"Text: {chunk_rows[0]['chunk_text'][:500]}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Embedding Model
# MAGIC
# MAGIC The same model is used by the Flask `/weather/search` endpoint.
# MAGIC
# MAGIC `all-MiniLM-L6-v2` produces 384-dimensional embeddings.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

print(f"Loading {EMBEDDING_MODEL_NAME}...")

model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("✅ Embedding model loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Chunk Embeddings
# MAGIC
# MAGIC Embeddings are generated in batches to avoid unnecessarily large memory
# MAGIC usage.
# MAGIC
# MAGIC Embeddings are normalized so cosine similarity can be used consistently
# MAGIC during pgvector search.

# COMMAND ----------

embedded_rows = []

for start in range(0, len(chunk_rows), BATCH_SIZE):

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

    for row, vector in zip(batch, vectors):

        vector_dimension = len(vector)

        if vector_dimension != EMBEDDING_DIM:

            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dimensional vector "
                f"but received {vector_dimension} dimensions."
            )

        row_with_embedding = dict(row)

        row_with_embedding["embedding"] = (
            vector.astype(float).tolist()
        )

        embedded_rows.append(
            row_with_embedding
        )

    print(
        f"Embedded "
        f"{min(start + BATCH_SIZE, len(chunk_rows))}"
        f"/{len(chunk_rows)} chunks"
    )

print(
    f"\nGenerated {len(embedded_rows)} embeddings."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Generated Embeddings

# COMMAND ----------

if embedded_rows:

    sample = embedded_rows[0]

    print(
        f"Document ID: {sample['document_id']}"
    )

    print(
        f"Chunk index: {sample['chunk_index']}"
    )

    print(
        f"Chunk text:  {sample['chunk_text'][:500]}"
    )

    print(
        f"Vector dimensions: {len(sample['embedding'])}"
    )

    print(
        f"Model: {EMBEDDING_MODEL_NAME}"
    )

else:

    print(
        "No new embeddings were generated."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Lakebase
# MAGIC
# MAGIC Embeddings are written directly through pg8000.
# MAGIC
# MAGIC The vector is passed as a PostgreSQL vector literal and explicitly cast
# MAGIC using:
# MAGIC
# MAGIC `CAST(:embedding AS vector)`
# MAGIC
# MAGIC
# MAGIC Each chunk receives a stable ID:
# MAGIC
# MAGIC `document_id + "_" + chunk_index`
# MAGIC
# MAGIC
# MAGIC The operation uses `ON CONFLICT` so the notebook can safely be re-run.

# COMMAND ----------

if not embedded_rows:

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

        # Explicitly commit the transaction.
        conn.commit()

    except Exception:

        try:
            conn.rollback()
        except Exception:
            pass

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
# MAGIC - total number of embeddings
# MAGIC - number of distinct documents
# MAGIC - pgvector data type
# MAGIC - embedding dimension

# COMMAND ----------

conn = get_connection()

try:

    counts_result = conn.run(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT document_id)
        FROM {EMBEDDINGS_TABLE_NAME}
        """
    )

    total_embeddings, documents_embedded = (
        counts_result[0]
    )

    column_result = conn.run(
        """
        SELECT
            column_name,
            data_type,
            udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = :table_name
          AND column_name = 'embedding'
        """,
        table_name=EMBEDDINGS_TABLE_NAME,
    )

    embedding_column = (
        column_result[0]
        if column_result
        else None
    )

    dimension_result = conn.run(
        f"""
        SELECT
            vector_dims(embedding)
        FROM {EMBEDDINGS_TABLE_NAME}
        WHERE embedding IS NOT NULL
        LIMIT 1
        """
    )

    embedding_dimension = (
        dimension_result[0][0]
        if dimension_result
        else None
    )

finally:

    conn.close()

print(
    f"Total embeddings:       {total_embeddings}"
)

print(
    f"Documents embedded:     {documents_embedded}"
)

if embedding_column:

    (
        column_name,
        data_type,
        udt_name,
    ) = embedding_column

    print(
        f"Embedding data type:    {data_type}"
    )

    print(
        f"Embedding UDT:          {udt_name}"
    )

if embedding_dimension:

    print(
        f"Embedding dimensions:   {embedding_dimension}"
    )

    if embedding_dimension != EMBEDDING_DIM:

        raise ValueError(
            f"Expected {EMBEDDING_DIM}-dimensional vectors "
            f"but database contains {embedding_dimension}-dimensional vectors."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify HNSW Index
# MAGIC
# MAGIC Confirm that the pgvector HNSW cosine-similarity index exists.

# COMMAND ----------

conn = get_connection()

try:

    index_result = conn.run(
        """
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = :table_name
        ORDER BY indexname
        """,
        table_name=EMBEDDINGS_TABLE_NAME,
    )

finally:

    conn.close()

print(
    f"Indexes on {EMBEDDINGS_TABLE_NAME}:"
)

for index_name, index_definition in index_result:

    print()
    print(index_name)
    print(index_definition)

hnsw_indexes = [
    row
    for row in index_result
    if "hnsw" in row[1].lower()
]

if hnsw_indexes:

    print(
        "\n✅ HNSW vector index detected."
    )

else:

    print(
        "\n⚠️ No HNSW index was detected."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Cosine Similarity Search
# MAGIC
# MAGIC This validation query confirms that:
# MAGIC
# MAGIC 1. The embedding column is a pgvector.
# MAGIC 2. The cosine-distance operator `<=>` works.
# MAGIC 3. Embeddings can be ordered by semantic similarity.
# MAGIC
# MAGIC The production version of this logic lives in:
# MAGIC
# MAGIC `POST /weather/search`
# MAGIC
# MAGIC The Flask endpoint embeds the user's query with the same model and uses
# MAGIC pgvector's cosine-distance operator.

# COMMAND ----------

if embedded_rows:

    test_vector = (
        embedded_rows[0]["embedding"]
    )

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
            f"Similarity: {float(similarity):.4f}"
        )

        print(
            f"Document:   {document_id}"
        )

        print(
            f"Chunk:      {chunk_index}"
        )

        print(
            f"Text:       {str(chunk_text)[:300]}"
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
# MAGIC ## Final Pipeline Validation
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
# MAGIC       VECTOR(384)
# MAGIC             |
# MAGIC        HNSW / cosine
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

# COMMAND ----------

print("=" * 70)
print("WEATHER EMBEDDING PIPELINE COMPLETE")
print("=" * 70)

print()
print(f"Source table:       {WEATHER_TABLE_NAME}")
print(f"Embedding table:    {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model:    {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimension:{EMBEDDING_DIM}")
print(f"Chunk size:         {CHUNK_SIZE}")
print(f"Chunk overlap:      {CHUNK_OVERLAP}")

print()
print(f"Weather documents:  {total_documents}")
print(f"Documents with text:{documents_with_text}")
print(f"Embeddings:         {total_embeddings}")
print(f"Documents embedded: {documents_embedded}")

print()
print("Lakebase driver:    pg8000")
print("Authentication:     Databricks SDK OAuth credential")
print("Vector search:      pgvector / HNSW / cosine similarity")

print()
print("Pipeline:")
print("  NWS API")
print("    -> /weather/sync")
print("    -> weather_news")
print("    -> ingest_weather_embeddings")
print("    -> chunk narrative_text")
print("    -> all-MiniLM-L6-v2")
print("    -> weather_embeddings")
print("    -> pgvector / HNSW")
print("    -> /weather/search")

print()
print("✅ Notebook completed successfully.")
