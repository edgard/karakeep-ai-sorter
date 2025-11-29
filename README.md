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

Pre-built multi-platform images (linux/amd64, linux/arm64) are available from GitHub Container Registry:

```bash
docker run --rm \
  -e KARAKEEP_API_BASE="https://your-karakeep/api/v1" \
  -e KARAKEEP_API_KEY="your-karakeep-key" \
  -e OPENAI_API_KEY="your-openai-or-openrouter-key" \
  ghcr.io/edgard/karakeep-ai-sorter:latest
```

Available tags:
- `:latest` - Latest release
- `:v1.2.3` - Specific version
- `:v1.2` - Latest patch of minor version
- `:v1` - Latest minor of major version

Or build locally:

```bash
docker build -t karakeep-ai-sorter .
docker run --rm \
  -e KARAKEEP_API_BASE=... \
  -e KARAKEEP_API_KEY=... \
  -e OPENAI_API_KEY=... \
  karakeep-ai-sorter
```

## Releases

Releases are automatically created on merge to `main` using [semantic versioning](https://semver.org/) based on [conventional commits](https://www.conventionalcommits.org/):
- `feat:` → minor version bump (v1.0.0 → v1.1.0)
- `fix:` → patch version bump (v1.0.0 → v1.0.1)
- `BREAKING CHANGE:` or `feat!:` → major version bump (v1.0.0 → v2.0.0)

Each release includes multi-platform Docker images and auto-generated release notes.

## Behavior

- Categories are your manual lists (excluding skips). `SOURCE_LISTS` are both sources and targets; `INBOX_LISTS` are sources only (never targets).
- Bookmarks are processed only if they’re in any source/inbox list when those are set.
- Low confidence (< `CONFIDENCE_THRESHOLD`, default 0.50) → `Uncategorized`. Shopping predictions need ≥0.75 confidence if the shopping list exists.
- Operations performed: add to target list; remove from other category lists and any inbox lists. No bookmark content/tags are changed.
