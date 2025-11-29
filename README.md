# Karakeep AI Sorter

Reclassifies Karakeep bookmarks using an OpenAI-compatible endpoint and moves them into your lists. No bookmark content is modified—only list membership is changed.

## Quick start (local)

```bash
pip install -r requirements.txt
export KARAKEEP_API_BASE="https://your-karakeep/api/v1"
export KARAKEEP_API_KEY="your-karakeep-key"
export OPENAI_API_KEY="your-openai-or-openrouter-key"
./start.sh
```

Key envs (defaults in `start.sh` and script docstring):

- `KARAKEEP_API_BASE`, `KARAKEEP_API_KEY`
- `OPENAI_API_KEY`, `OPENAI_MODEL` (default gpt-5-mini), `OPENAI_BASE_URL`
- `SOURCE_LISTS` (comma-separated lists to process; targets), `INBOX_LISTS` (sources only, never targets)
- `SKIP_LISTS` (ignored completely)
- `FAILED_LIST` (fallback list, default `Uncategorized`)
- `SHOPPING_LIST` (default `Shopping`), `CONFIDENCE_THRESHOLD` (global, default 0.50)
- `DRY_RUN` (set to 1 for no writes), `MAX_BOOKMARKS` (limit processed items)

## Docker

```bash
docker build -t karakeep-ai-sorter .
docker run --rm \
  -e KARAKEEP_API_BASE=... \
  -e KARAKEEP_API_KEY=... \
  -e OPENAI_API_KEY=... \
  karakeep-ai-sorter
```

## GitHub Actions release

On pushes to `main`, builds and pushes `ghcr.io/<owner>/karakeep-ai-sorter:latest` and creates a GitHub Release (see `.github/workflows/release.yml`).

## Behavior

- Categories are your manual lists (excluding skips). `SOURCE_LISTS` are both sources and targets; `INBOX_LISTS` are sources only (never targets).
- Bookmarks are processed only if they’re in any source/inbox list when those are set.
- Low confidence (< `CONFIDENCE_THRESHOLD`, default 0.50) → `Uncategorized`. Shopping predictions need ≥0.75 confidence if the shopping list exists.
- Operations performed: add to target list; remove from other category lists and any inbox lists. No bookmark content/tags are changed.
