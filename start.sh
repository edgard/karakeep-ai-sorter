#!/usr/bin/env bash
set -euo pipefail

# Quick-start runner for karakeep_ai_sorter.py
# Edit the exports below or set them in your shell/.env before running.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export KARAKEEP_API_BASE="${KARAKEEP_API_BASE:-https://your-karakeep-host/api/v1}"
export KARAKEEP_API_KEY="${KARAKEEP_API_KEY:-your-karakeep-api-key}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}"
export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-60}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

# List handling
export SOURCE_LISTS="${SOURCE_LISTS:-}"                         # comma-separated list names; only bookmarks in these lists are processed (if set)
export INBOX_LISTS="${INBOX_LISTS:-}"                           # comma-separated list names; used only as sources, never targets
export SKIP_LISTS="${SKIP_LISTS:-}"                             # comma-separated list names to ignore entirely
export FAILED_LIST="${FAILED_LIST:-Uncategorized}"              # name of the fallback list
export SHOPPING_LIST="${SHOPPING_LIST:-Shopping}"               # shopping list name; low-confidence picks go to Uncategorized
export CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.50}"     # global min confidence before falling back

# Run controls
export DRY_RUN="${DRY_RUN:-}"                                   # set to 1 to log only
export MAX_BOOKMARKS="${MAX_BOOKMARKS:-}"                       # set to limit batch size

python3 "$ROOT_DIR/karakeep_ai_sorter.py"
