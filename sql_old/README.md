# SQL Setup Files for Lakebase

These SQL files must be run manually in your Lakebase Postgres database before running the weather ingestion and embedding pipeline. The weather app uses the **National Weather Service (NWS) API** as its source of weather alerts and narrative weather information.

## Tables
The weather pipeline uses two primary tables:
1. `weather_news` — Stores normalized NWS weather alert/document data and the original API payload.
2. `weather_news_chunk_embeddings` — Stores chunked narrative text and its 384-dimensional vector embeddings.

---

## Setup Order

### 1. Run `01_setup_weather_news_table.sql`

Creates the `weather_news` table for storing normalized weather documents from the NWS API.

The table stores:

* NWS alert ID
* Location
* Source type
* Event and headline
* Narrative text
* Description
* Instructions
* Severity, certainty, and urgency
* Effective, onset, and expiration timestamps
* NWS sender information
* Geographic information
* Original NWS JSON payload
* Sync and audit timestamps

### NWS Alert ID

The NWS API provides a stable `id` for each alert.

The NWS alert ID is used directly as:

```sql
weather_news.id
```

This allows the ingestion process to safely upsert alerts using:

```sql
ON CONFLICT (id)
```

rather than generating a new UUID for every synchronization.

---

### 2. Run `02_setup_weather_news_chunk_embeddings_table.sql`

Creates the `weather_news_chunk_embeddings` table for storing embeddings generated from chunks of the weather narrative text.

The table contains:

* `id` — Deterministic chunk ID
* `document_id` — Foreign key to `weather_news.id`
* `chunk_index` — Position of the chunk within the source document
* `chunk_text` — Text represented by the embedding
* `embedding` — 384-dimensional pgvector embedding
* `model_name` — Embedding model used
* `embedded_at` — Timestamp when the chunk was embedded

The table also includes:

* A foreign key from `document_id` to `weather_news.id`
* A uniqueness constraint on `(document_id, chunk_index)`
* An HNSW index using cosine distance
* An index on `document_id`

---

## Embedding Model

The current weather embedding pipeline uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model produces **384-dimensional embeddings**.

Therefore, the embedding column is defined as:

```sql
embedding VECTOR(384) NOT NULL
```

If the embedding model changes in the future, the vector dimension in the table definition and the embedding pipeline must be updated together.

### Supported example dimensions

| Model                                     | Dimension |
| ----------------------------------------- | --------: |
| `sentence-transformers/all-MiniLM-L6-v2`  |       384 |
| `sentence-transformers/all-mpnet-base-v2` |       768 |
| `BAAI/bge-small-en-v1.5`                  |       384 |
| `BAAI/bge-base-en-v1.5`                   |       768 |
| `BAAI/bge-large-en-v1.5`                  |      1024 |

The current SQL setup uses **384** and does not require replacing a `{{EMBEDDING_DIM}}` placeholder.

---


## Why Manual SQL Setup?

The tables and indexes are created manually in Lakebase because PostgreSQL-specific features such as:

* `CREATE EXTENSION vector`
* `VECTOR(384)` columns
* HNSW vector indexes
* PostgreSQL foreign keys
* PostgreSQL `ON CONFLICT` upserts

are best handled directly through PostgreSQL.

The weather ingestion and embedding scripts use **`psycopg2`** and the same `get_connection()` helper used by the Lakebase application.
This approach provides:

* ✅ Proper pgvector `VECTOR(384)` columns
* ✅ HNSW indexes for fast cosine similarity search
* ✅ Foreign-key integrity between documents and embeddings
* ✅ Deterministic document and chunk IDs
* ✅ Idempotent `ON CONFLICT` upserts
* ✅ No Spark JDBC dependency for Lakebase writes
* ✅ No post-processing array-to-vector conversion

