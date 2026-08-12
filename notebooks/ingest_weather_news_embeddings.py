# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Ingest Weather Documents → Vector Embeddings
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
# MAGIC This script intentionally uses pg8000 through `lakebase.py`.
# MAGIC It does not use psycopg2 or Spark JDBC.

# COMMAND ----------

# MAGIC %pip install -q pg8000 sentence-transformers trafilatura requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 50

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Narrative Text
# MAGIC
# MAGIC Split weather narrative text into overlapping character-based chunks.

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
# MAGIC Return documents that have never been embedded or were updated
# MAGIC after their most recent embedding was created.

# COMMAND ----------

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
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Chunk Rows
# MAGIC
# MAGIC Create chunk records for each weather document.

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
# MAGIC Generate 384-dimensional embeddings for each text chunk.

# COMMAND ----------

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
# MAGIC Convert Python vectors into PostgreSQL vector literal syntax.

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
# MAGIC Replace embeddings for documents processed in this run.
# MAGIC
# MAGIC This allows updated NWS alerts to be re-embedded instead of leaving
# MAGIC stale chunks in the vector table.

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Stored Embeddings
# MAGIC
# MAGIC Confirm the number of stored embeddings and their vector dimensions.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Embedding Pipeline
# MAGIC
# MAGIC This section:
# MAGIC
# MAGIC 1. Finds new or updated weather documents.
# MAGIC 2. Creates text chunks.
# MAGIC 3. Loads the embedding model.
# MAGIC 4. Generates embeddings.
# MAGIC 5. Writes them to Lakebase.
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

# COMMAND ----------

if __name__ == "__main__":
    main()