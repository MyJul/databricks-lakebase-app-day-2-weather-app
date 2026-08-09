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
# MAGIC 1. Reads weather documents previously harvested from the National Weather
# MAGIC    Service (NWS) API and stored in the `weather_news` Lakebase table.
# MAGIC 2. Finds documents that do not yet have embeddings.
# MAGIC 3. Splits each document's `narrative_text` into overlapping chunks.
# MAGIC 4. Embeds each chunk using
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
# MAGIC 5. Writes the chunk embeddings directly to Lakebase using psycopg2.
# MAGIC 6. Stores embeddings in a pgvector `VECTOR(384)` column with an HNSW
# MAGIC    cosine-similarity index.
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
# MAGIC The notebook intentionally does NOT use Spark JDBC for writes.
# MAGIC Lakebase writes are performed with psycopg2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC Required packages:
# MAGIC - psycopg2-binary: PostgreSQL/Lakebase connection
# MAGIC - sentence-transformers: text embedding model
# MAGIC
# MAGIC The NWS API itself is called by the Flask application's
# MAGIC `weather_client.py`, not by this notebook.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install -q psycopg2-binary sentence-transformers

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

WEATHER_TABLE_NAME = dbutils.widgets.get("weather_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

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

print(f"Weather document table: {WEATHER_TABLE_NAME}")
print(f"Embedding table:        {EMBEDDINGS_TABLE_NAME}")
print(f"Embedding model:        {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimensions:   {EMBEDDING_DIM}")
print(f"Chunk size:              {CHUNK_SIZE}")
print(f"Chunk overlap:           {CHUNK_OVERLAP}")
print(f"Batch size:              {BATCH_SIZE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakebase Connection
# MAGIC
# MAGIC This notebook uses the same `lakebase.py` connection helper used by the
# MAGIC Flask application.
# MAGIC
# MAGIC The assignment specifically requires psycopg2 for the embedding pipeline.

# COMMAND ----------

import lakebase

# Test the Lakebase connection.

with lakebase.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT current_database(), current_user")
        database, user = cur.fetchone()

print("Lakebase connection successful.")
print(f"Database: {database}")
print(f"User:     {user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Required Tables
# MAGIC
# MAGIC The SQL setup files should already have created:
# MAGIC
# MAGIC - `weather_news`
# MAGIC - `weather_embeddings`
# MAGIC
# MAGIC The expected `weather_news` structure includes:
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
# MAGIC The expected `weather_embeddings` structure includes:
# MAGIC
# MAGIC - `id`
# MAGIC - `document_id`
# MAGIC - `chunk_index`
# MAGIC - `chunk_text`
# MAGIC - `embedding`
# MAGIC - `model_name`
# MAGIC - `created_at`

# COMMAND ----------

from psycopg2.extras import RealDictCursor

with lakebase.get_connection() as conn:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:

        cur.execute(
            """
            SELECT
                table_name,
                column_name,
                data_type,
                udt_name
            FROM information_schema.columns
            WHERE table_name IN (%s, %s)
            ORDER BY table_name, ordinal_position
            """,
            (WEATHER_TABLE_NAME, EMBEDDINGS_TABLE_NAME),
        )

        table_columns = cur.fetchall()

for row in table_columns:
    print(
        f"{row['table_name']}.{row['column_name']}: "
        f"{row['data_type']} ({row['udt_name']})"
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

with lakebase.get_connection() as conn:
    with conn.cursor() as cur:

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {WEATHER_TABLE_NAME}
            """
        )

        total_documents = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {WEATHER_TABLE_NAME}
            WHERE narrative_text IS NOT NULL
              AND TRIM(narrative_text) <> ''
            """
        )

        documents_with_text = cur.fetchone()[0]

print(f"Total weather documents:       {total_documents}")
print(f"Documents containing text:     {documents_with_text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Unembedded Weather Documents
# MAGIC
# MAGIC A document is considered unembedded when there is no corresponding
# MAGIC row in `weather_embeddings`.
# MAGIC
# MAGIC This makes the notebook safe to run repeatedly:
# MAGIC
# MAGIC - previously embedded documents are skipped
# MAGIC - newly synchronized NWS documents are picked up
# MAGIC - existing embeddings are not duplicated

# COMMAND ----------

with lakebase.get_connection() as conn:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:

        cur.execute(
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

        documents = cur.fetchall()

print(f"Found {len(documents)} unembedded weather documents.")

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
# MAGIC We use:
# MAGIC
# MAGIC - `CHUNK_SIZE = 800`
# MAGIC - `CHUNK_OVERLAP = 100`
# MAGIC
# MAGIC These values match the assignment recommendation.
# MAGIC
# MAGIC The overlap preserves some context when a sentence or concept crosses
# MAGIC a chunk boundary.

# COMMAND ----------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
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

    chunks = chunk_text(document["narrative_text"])

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
    print(chunk_rows[0])

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

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Embedding model loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Chunk Embeddings
# MAGIC
# MAGIC Embeddings are generated in batches to avoid unnecessarily large memory
# MAGIC usage.

# COMMAND ----------

import numpy as np

embedded_rows = []

for start in range(0, len(chunk_rows), BATCH_SIZE):

    batch = chunk_rows[start:start + BATCH_SIZE]

    texts = [row["chunk_text"] for row in batch]

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    for row, vector in zip(batch, vectors):

        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dimensional vector but received "
                f"{len(vector)} dimensions."
            )

        row_with_embedding = dict(row)
        row_with_embedding["embedding"] = vector.tolist()

        embedded_rows.append(row_with_embedding)

    print(
        f"Embedded {min(start + BATCH_SIZE, len(chunk_rows))}"
        f"/{len(chunk_rows)} chunks"
    )

print(f"\nGenerated {len(embedded_rows)} embeddings.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Generated Embeddings

# COMMAND ----------

if embedded_rows:

    sample = embedded_rows[0]

    print(f"Document ID: {sample['document_id']}")
    print(f"Chunk index: {sample['chunk_index']}")
    print(f"Chunk text:  {sample['chunk_text'][:500]}")
    print(f"Vector dimensions: {len(sample['embedding'])}")
    print(f"Model: {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Lakebase
# MAGIC
# MAGIC The embeddings are written directly through psycopg2.
# MAGIC
# MAGIC The vector is passed as a PostgreSQL vector literal and explicitly cast
# MAGIC using:
# MAGIC
# MAGIC ```sql
# MAGIC %s::vector
# MAGIC ```
# MAGIC
# MAGIC This avoids the Spark JDBC / pgvector issues described in the assignment.
# MAGIC
# MAGIC Each chunk receives a stable ID:
# MAGIC
# MAGIC `document_id + "_" + chunk_index`
# MAGIC
# MAGIC The table also has a unique constraint/primary key on `id`, allowing the
# MAGIC notebook to be safely re-run.

# COMMAND ----------

from psycopg2.extras import execute_values
from datetime import datetime, timezone

if not embedded_rows:

    print("No new embeddings to write.")

else:

    insert_rows = []

    for row in embedded_rows:

        embedding_id = (
            f"{row['document_id']}_{row['chunk_index']}"
        )

        vector_string = (
            "["
            + ",".join(str(float(value)) for value in row["embedding"])
            + "]"
        )

        insert_rows.append(
            (
                embedding_id,
                row["document_id"],
                row["chunk_index"],
                row["chunk_text"],
                vector_string,
                EMBEDDING_MODEL_NAME,
            )
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
        VALUES %s
        ON CONFLICT (id) DO UPDATE
        SET
            document_id = EXCLUDED.document_id,
            chunk_index = EXCLUDED.chunk_index,
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = EXCLUDED.created_at
    """

    template = """
        (
            %s,
            %s,
            %s,
            %s,
            %s::vector,
            %s,
            now()
        )
    """

    total_written = 0

    with lakebase.get_connection() as conn:

        with conn.cursor() as cur:

            for start in range(0, len(insert_rows), BATCH_SIZE):

                batch = insert_rows[start:start + BATCH_SIZE]

                execute_values(
                    cur,
                    insert_sql,
                    batch,
                    template=template,
                    page_size=BATCH_SIZE,
                )

                total_written += len(batch)

                print(
                    f"Wrote {min(start + BATCH_SIZE, len(insert_rows))}"
                    f"/{len(insert_rows)} embeddings"
                )

        conn.commit()

    print(
        f"\nSuccessfully wrote {total_written} embeddings "
        f"to {EMBEDDINGS_TABLE_NAME}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embeddings
# MAGIC
# MAGIC Confirm that the embedding column is actually PostgreSQL's `vector`
# MAGIC type rather than a text or array column.

# COMMAND ----------

with lakebase.get_connection() as conn:

    with conn.cursor(cursor_factory=RealDictCursor) as cur:

        cur.execute(
            f"""
            SELECT
                COUNT(*) AS total_embeddings,
                COUNT(DISTINCT document_id) AS documents_embedded
            FROM {EMBEDDINGS_TABLE_NAME}
            """
        )

        counts = cur.fetchone()

        cur.execute(
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

        embedding_column = cur.fetchone()

print(f"Total embeddings:       {counts['total_embeddings']}")
print(f"Documents embedded:     {counts['documents_embedded']}")

if embedding_column:
    print(
        f"Embedding data type:    {embedding_column['data_type']}"
    )
    print(
        f"Embedding UDT:          {embedding_column['udt_name']}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Cosine Similarity Search
# MAGIC
# MAGIC This is a small validation query to confirm the pgvector column and HNSW
# MAGIC index are usable.
# MAGIC
# MAGIC The production version of this logic lives in:
# MAGIC
# MAGIC `POST /weather/search`
# MAGIC
# MAGIC The Flask endpoint embeds the user's query with the same model and uses
# MAGIC pgvector's cosine-distance operator (`<=>`).

# COMMAND ----------

if embedded_rows:

    test_vector = embedded_rows[0]["embedding"]

    test_vector_string = (
        "["
        + ",".join(str(float(value)) for value in test_vector)
        + "]"
    )

    with lakebase.get_connection() as conn:

        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            cur.execute(
                f"""
                SELECT
                    e.id,
                    e.document_id,
                    e.chunk_index,
                    e.chunk_text,
                    1 - (
                        e.embedding <=> %s::vector
                    ) AS similarity
                FROM {EMBEDDINGS_TABLE_NAME} e
                ORDER BY e.embedding <=> %s::vector
                LIMIT 5
                """,
                (
                    test_vector_string,
                    test_vector_string,
                ),
            )

            results = cur.fetchall()

    print("Top 5 similarity results:\n")

    for result in results:

        print(
            f"Similarity: {result['similarity']:.4f}"
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
        "No embeddings available for similarity test."
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
