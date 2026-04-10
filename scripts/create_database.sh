#!/usr/bin/env bash
# Create database computer_vision and tables database_nutristeppe + results.
#
# Usage:
#   ./scripts/create_database.sh
#
# Optional env:
#   POSTGRES_DB          (default: computer_vision)
#   POSTGRES_ADMIN_USER  (default: $USER)
#   PGHOST, PGPORT, PGPASSWORD

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SQL_FILE="${ROOT}/sql/dokploy_init.sql"
DB_NAME="${POSTGRES_DB:-computer_vision}"
DB_ADMIN_USER="${POSTGRES_ADMIN_USER:-${USER}}"

export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"

if ! command -v psql >/dev/null 2>&1; then
  echo "psql not found. Install: brew install libpq && brew link --force libpq"
  exit 1
fi

if [[ ! -f "${SQL_FILE}" ]]; then
  echo "Missing ${SQL_FILE}"
  exit 1
fi

psql_admin() {
  psql -v ON_ERROR_STOP=1 -U "${DB_ADMIN_USER}" "$@"
}

echo "Using host=${PGHOST} port=${PGPORT} admin_user=${DB_ADMIN_USER} db=${DB_NAME}"

EXISTS="$(psql_admin -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" || true)"
if [[ "${EXISTS}" != "1" ]]; then
  psql_admin -d postgres -c "CREATE DATABASE \"${DB_NAME}\";"
  echo "Created database: ${DB_NAME}"
else
  echo "Database '${DB_NAME}' exists — applying schema (IF NOT EXISTS)."
fi

psql_admin -d "${DB_NAME}" -f "${SQL_FILE}"
echo "Done. Tables: database_nutristeppe, results"
echo "export DATABASE_URL=postgresql://${DB_ADMIN_USER}@${PGHOST}:${PGPORT}/${DB_NAME}"
