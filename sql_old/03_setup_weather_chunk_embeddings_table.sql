-- Setup script for weather_news_chunk_embeddings table
-- Run this manually in your Lakebase Postgres database before running the notebook
-- Replace {{EMBEDDING_DIM}} with your model's dimension (e.g., 384 for all-MiniLM-L6-v2)

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the chunk embeddings table
CREATE TABLE IF NOT EXISTS weather_news_chunk_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_weather_news_chunk
        UNIQUE (document_id, chunk_index),
    CONSTRAINT fk_weather_news_chunk_document
        FOREIGN KEY (document_id)
        REFERENCES weather_news(id)
        ON DELETE CASCADE
);

-- Create HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_weather_news_chunk_embeddings_embedding
ON weather_news_chunk_embeddings
USING hnsw (embedding vector_cosine_ops);

-- Create an index for retrieving all chunks belonging
-- to a particular weather document
CREATE INDEX IF NOT EXISTS idx_weather_news_chunk_embeddings_document_id
ON weather_news_chunk_embeddings(document_id);

-- Verify the table was created
SELECT
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_news_chunk_embeddings'
ORDER BY ordinal_position;
