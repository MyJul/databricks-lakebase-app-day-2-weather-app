-- Setup script for weather_embeddings table
-- Run this manually in your Lakebase Postgres database before running the notebook
-- Requires pgvector extension

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the weather_embeddings table
-- Uses composite TEXT id (document_id_chunk_index) for idempotent inserts
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_weather_embeddings_document
        FOREIGN KEY (document_id)
        REFERENCES weather_news(id)
        ON DELETE CASCADE,
    CONSTRAINT uq_weather_embedding_chunk
        UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings(document_id);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);
