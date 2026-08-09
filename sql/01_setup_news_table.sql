-- Setup script for weather_news table
-- Create indices location, event, source, and effective time windows
-- Run this manually in your Lakebase Postgres database before running the notebook

-- Create the weather_news table
CREATE TABLE IF NOT EXISTS weather_news (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'alert',
    event TEXT,
    headline TEXT,
    narrative_text TEXT NOT NULL,
    description TEXT,
    instruction TEXT,
    severity TEXT,
    certainty TEXT,
    urgency TEXT,
    effective_at TIMESTAMPTZ,
    onset_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    sender_name TEXT,
    sender_id TEXT,
    area_desc TEXT,
    geocode JSONB,
    geometry JSONB,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_weather_news_location
    ON weather_news(location);

CREATE INDEX IF NOT EXISTS idx_weather_news_event
    ON weather_news(event);

CREATE INDEX IF NOT EXISTS idx_weather_news_effective
    ON weather_news(effective_at);

CREATE INDEX IF NOT EXISTS idx_weather_news_expires
    ON weather_news(expires_at);

CREATE INDEX IF NOT EXISTS idx_weather_news_source_type
    ON weather_news(source_type);
