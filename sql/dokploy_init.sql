-- =============================================================================
-- PostgreSQL: database computer_vision (create DB via scripts/create_database.sh)
-- Production tables: database_nutristeppe, results
-- =============================================================================

CREATE TABLE IF NOT EXISTS database_nutristeppe (
    id                      BIGSERIAL PRIMARY KEY,
    bls_code                TEXT,
    name                    TEXT NOT NULL,
    name_en                 TEXT DEFAULT '',
    protein                 DOUBLE PRECISION DEFAULT 0,
    fat                     DOUBLE PRECISION DEFAULT 0,
    carbohydrate            DOUBLE PRECISION DEFAULT 0,
    fiber                   DOUBLE PRECISION DEFAULT 0,
    kilocalories            DOUBLE PRECISION DEFAULT 0,
    saturated_fat_mg        DOUBLE PRECISION DEFAULT 0,
    sugar_mg                DOUBLE PRECISION DEFAULT 0,
    salt_total_mg           DOUBLE PRECISION DEFAULT 0,
    health_index            DOUBLE PRECISION,
    serving_size_g          DOUBLE PRECISION DEFAULT 100,
    ingredients             TEXT,
    steps                   TEXT,
    kcal_portion            DOUBLE PRECISION DEFAULT 0,
    protein_portion         DOUBLE PRECISION DEFAULT 0,
    fat_portion             DOUBLE PRECISION DEFAULT 0,
    carbohydrate_portion    DOUBLE PRECISION DEFAULT 0,
    salt_total_mg_portion   DOUBLE PRECISION DEFAULT 0,
    sugar_mg_portion        DOUBLE PRECISION DEFAULT 0,
    fiber_portion           DOUBLE PRECISION DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_nutristeppe_name_lower
    ON database_nutristeppe (LOWER(name));
CREATE INDEX IF NOT EXISTS idx_nutristeppe_name_en_lower
    ON database_nutristeppe (LOWER(name_en));

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_trgm_name
    ON database_nutristeppe USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_trgm_name_en
    ON database_nutristeppe USING gin (name_en gin_trgm_ops);

COMMENT ON TABLE database_nutristeppe IS 'Справочник блюд Nutristeppe';

-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results (
    id                   SERIAL PRIMARY KEY,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_name            TEXT NOT NULL,
    user_dish_name       TEXT,
    user_portion         DOUBLE PRECISION,
    gemini_dish_name     TEXT,
    gemini_portion       DOUBLE PRECISION,
    matched_db_dish      TEXT,
    matched_db_name_en   TEXT,
    algorithm_results    JSONB,
    verification_status  BOOLEAN,
    verification_detail  JSONB,
    image_jpeg           BYTEA
);

CREATE INDEX IF NOT EXISTS idx_results_created_at ON results (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_results_user_name ON results (user_name);

COMMENT ON TABLE results IS 'История анализов приложения';
