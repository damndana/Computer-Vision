-- Optional: vector backup alongside FAISS files (see meal_pipeline.embedding_generator --write-pg)
-- meal_id should match your meals table primary key (e.g. database_nutristeppe.id).
CREATE TABLE IF NOT EXISTS meal_embeddings (
    meal_id BIGINT PRIMARY KEY,
    embedding_dim INTEGER NOT NULL,
    embedding_vector BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meal_embeddings_updated ON meal_embeddings (updated_at DESC);
