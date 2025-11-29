#!/usr/bin/env python3
"""
Karakeep AI Sorter: classify Karakeep bookmarks via an OpenAI-compatible endpoint and move them into lists.

Env vars:
  KARAKEEP_API_BASE   Base API URL; can be host or host/api or host/api/v1
  KARAKEEP_API_KEY    Personal API key
  OPENAI_API_KEY      OpenAI key (can also be an OpenRouter key)
  OPENAI_MODEL        Defaults to gpt-5-mini
  OPENAI_TIMEOUT      Seconds timeout for OpenAI requests (default 60)
  OPENAI_BASE_URL     Defaults to https://api.openai.com/v1 (set to https://openrouter.ai/api/v1 for OpenRouter)
  SKIP_LISTS          Comma-separated list names to ignore (never read from or move to)
  SOURCE_LISTS        Comma-separated list names; only bookmarks in these lists are processed
  SHOPPING_LIST       Name of the shopping list; low-confidence items go to Uncategorized
  FAILED_LIST         Name of the fallback/uncategorized list (default: Uncategorized)
  DRY_RUN             If set, no writes are made (logs only)
  MAX_BOOKMARKS       Optional limit for a small batch run
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from karakeep_python_api import APIError, KarakeepAPI
from openai import OpenAI

REASON_MAX = 160


def env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise SystemExit(f"Missing required env var: {name}")
    return val


def env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_csv(name: str) -> List[str]:
    raw = os.getenv(name, "")
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


LIST_ICON = "📁"

# Attribution headers used only when hitting OpenRouter
DEFAULT_ATTR_SITE = "https://www.github.com/edgard/karakeep-reclassifier"
DEFAULT_ATTR_TITLE = "karakeep-reclassifier"

UNCATEGORIZED_LIST_NAME = os.getenv("FAILED_LIST", "Uncategorized")


def normalize_base_url(base: str) -> str:
    """
    Ensure the Karakeep base URL includes the /api/v1 prefix required by the server.

    - If base already contains "/api/v1", keep as-is.
    - If it ends with "/api", append "/v1".
    - Otherwise, append "/api/v1".
    """
    parsed = urlsplit(base)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        normalized_path = path
    elif path.endswith("/api"):
        normalized_path = f"{path}/v1"
    elif path.endswith("/v1"):
        normalized_path = path
    else:
        normalized_path = f"{path}/api/v1"

    normalized = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            normalized_path,
            parsed.query,
            parsed.fragment,
        )
    )
    if normalized != base:
        logging.info("Normalized KARAKEEP_API_BASE to %s", normalized)
    return normalized


def configure_logging() -> None:
    level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    # Silence chatty third-party loggers by default
    for noisy in ["karakeep_python_api", "httpx", "urllib3", "loguru"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Disable loguru default handler if present to avoid duplicate lines
    try:  # pragma: no cover
        from loguru import logger as _loguru_logger

        _loguru_logger.remove()
        _loguru_logger.disable("karakeep_python_api")
        # prevent downstream re-adding noisy handlers
        _loguru_logger.add = lambda *_, **__: 0
    except Exception:
        pass


@dataclass
class BookmarkInfo:
    id: str
    url: str
    title: Optional[str]
    description: Optional[str]
    tags: List[str]
    list_ids: List[str]


def load_categories_and_ids(
    api: KarakeepAPI,
    uncategorized_name: str,
    dry_run: bool,
    skip_lists: Sequence[str],
    source_lists: Sequence[str],
    inbox_lists: Sequence[str],
) -> Tuple[List[str], Dict[str, str], List[str], List[str]]:
    skip_set = {s.lower() for s in skip_lists}
    source_set = {s.lower() for s in source_lists}
    inbox_set = {s.lower() for s in inbox_lists}

    if uncategorized_name.lower() in skip_set:
        logging.error("uncategorized list '%s' cannot be in SKIP_LISTS", uncategorized_name)
        raise SystemExit(1)
    for s in list(source_lists) + list(inbox_lists):
        if s.lower() in skip_set:
            logging.error("source/inbox list '%s' is also in SKIP_LISTS; remove it from skips", s)
            raise SystemExit(1)
    existing = api.get_all_lists()
    categories: List[str] = []
    ids: Dict[str, str] = {}
    source_list_ids: List[str] = []
    inbox_list_ids: List[str] = []
    found_inbox_names: set[str] = set()

    def handle_item(item: object) -> None:
        name = getattr(item, "name", None) if not isinstance(item, dict) else item.get("name")
        list_id = getattr(item, "id", None) if not isinstance(item, dict) else item.get("id")
        list_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if not name or not list_id:
            return
        name_l = str(name).lower()
        if name_l in source_set:
            if str(list_id) not in source_list_ids:
                source_list_ids.append(str(list_id))
        if name_l in inbox_set:
            if str(list_id) not in inbox_list_ids:
                inbox_list_ids.append(str(list_id))
            found_inbox_names.add(name_l)
            # inbox lists are sources only; keep id for presence checks but do not add as a target category
            ids[str(name)] = str(list_id)
            return
        if name_l in skip_set:
            return
        if list_type in (None, "manual"):
            if name not in ids:
                categories.append(str(name))
            ids[str(name)] = str(list_id)

    if isinstance(existing, list):
        for item in existing:
            handle_item(item)
    elif isinstance(existing, dict):
        for item in existing.get("lists", []) or []:
            handle_item(item)

    missing_sources = [s for s in source_lists if s.lower() not in {n.lower() for n in ids} and s.lower() not in skip_set]
    missing_inbox = [s for s in inbox_lists if s.lower() not in found_inbox_names and s.lower() not in skip_set]
    if missing_sources:
        logging.error("source lists not found: %s", ",".join(missing_sources))
        raise SystemExit(1)
    if missing_inbox:
        logging.error("inbox lists not found: %s", ",".join(missing_inbox))
        raise SystemExit(1)

    if uncategorized_name in categories:
        categories = [c for c in categories if c != uncategorized_name]

    if not categories:
        logging.error("no manual lists available for classification after skips/source filters")
        raise SystemExit(1)

    # Ensure failed list exists (or placeholder in dry-run)
    if uncategorized_name not in ids:
        if dry_run:
            ids[uncategorized_name] = f"dry-{uncategorized_name}"
            logging.info("ensure list (dry-run) name=%s", uncategorized_name)
        else:
            created = api.create_a_new_list(
                name=uncategorized_name,
                icon=LIST_ICON,
                description="auto-created by reclassify script",
                list_type="manual",
            )
            ids[uncategorized_name] = str(getattr(created, "id", None) if not isinstance(created, dict) else created.get("id"))
            logging.info("ensure list created name=%s id=%s", uncategorized_name, ids[uncategorized_name])

    return categories, ids, source_list_ids, inbox_list_ids


def iter_bookmarks(api: KarakeepAPI, page_size: int = 100, limit: Optional[int] = None) -> Iterable[object]:
    cursor: Optional[str] = None
    seen = 0
    while True:
        page = api.get_all_bookmarks(limit=page_size, cursor=cursor, include_content=True)
        if hasattr(page, "bookmarks"):
            bookmarks = page.bookmarks  # type: ignore[attr-defined]
            cursor = page.nextCursor  # type: ignore[attr-defined]
        elif isinstance(page, dict):
            bookmarks = page.get("bookmarks") or page.get("data") or []
            cursor = page.get("nextCursor") or page.get("cursor")
        else:
            bookmarks = []
            cursor = None

        if not bookmarks:
            break

        for bm in bookmarks:
            yield bm
            seen += 1
            if limit is not None and seen >= limit:
                return

        if not cursor:
            break


def get_list_ids_for_bookmark(api: KarakeepAPI, bookmark_id: str) -> List[str]:
    resp = api.get_lists_of_a_bookmark(bookmark_id)
    ids: List[str] = []
    if isinstance(resp, list):
        for item in resp:
            if hasattr(item, "id"):
                ids.append(str(item.id))
            elif isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
    elif isinstance(resp, dict):
        for item in resp.get("lists", []) or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item.get("id")))
    return ids


def extract_tags(bookmark: object) -> List[str]:
    tags: List[str] = []
    raw_tags = getattr(bookmark, "tags", None)
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if hasattr(tag, "name"):
                tags.append(str(tag.name))
            elif isinstance(tag, dict):
                name = tag.get("name") or tag.get("tag") or tag.get("tagName")
                if name:
                    tags.append(str(name))
            elif isinstance(tag, str):
                tags.append(tag)
    return tags


def extract_url_and_description(bookmark: object) -> Tuple[str, Optional[str]]:
    content = getattr(bookmark, "content", None)
    url = ""
    description: Optional[str] = None

    if isinstance(content, dict):
        ctype = content.get("type")
        if ctype == "link":
            url = str(content.get("url") or "")
            description = content.get("description")
        elif ctype == "text":
            url = str(content.get("sourceUrl") or "")
            description = content.get("description") or content.get("summary") or content.get("text")
        else:
            url = str(content.get("sourceUrl") or "")
    elif content is not None:
        ctype = getattr(content, "type", None)
        if ctype == "link":
            url = str(getattr(content, "url", "") or "")
            description = getattr(content, "description", None)
        elif ctype == "text":
            url = str(getattr(content, "sourceUrl", "") or "")
            description = getattr(content, "description", None) or getattr(content, "text", None)
        else:
            url = str(getattr(content, "sourceUrl", "") or "")

    if not description:
        description = getattr(bookmark, "summary", None)
    if not description and isinstance(bookmark, dict):
        description = bookmark.get("summary") or bookmark.get("note")
    else:
        if not description:
            description = getattr(bookmark, "note", None)

    return url, description


def to_bookmark_info(bookmark: object, list_ids: List[str]) -> BookmarkInfo:
    tags = extract_tags(bookmark)
    url, description = extract_url_and_description(bookmark)
    title = bookmark.get("title") if isinstance(bookmark, dict) else getattr(bookmark, "title", None)
    bookmark_id = bookmark.get("id") if isinstance(bookmark, dict) else getattr(bookmark, "id", "")
    return BookmarkInfo(
        id=str(bookmark_id),
        url=url,
        title=title,
        description=description,
        tags=tags,
        list_ids=list_ids,
    )


def add_to_list(api: KarakeepAPI, list_id: str, bookmark_id: str, dry_run: bool) -> bool:
    if dry_run:
        logging.debug("list_add bookmark=%s list_id=%s status=dry-run", bookmark_id, list_id)
        return True
    try:
        api.add_a_bookmark_to_a_list(list_id, bookmark_id)
        logging.debug("list_add bookmark=%s list_id=%s status=ok", bookmark_id, list_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("list_add bookmark=%s list_id=%s status=error err=%s", bookmark_id, list_id, exc)
        return False


def remove_from_list(api: KarakeepAPI, list_id: str, bookmark_id: str, dry_run: bool) -> None:
    if dry_run:
        logging.debug("list_remove bookmark=%s list_id=%s status=dry-run", bookmark_id, list_id)
        return
    try:
        api.remove_a_bookmark_from_a_list(list_id, bookmark_id)
        logging.debug("list_remove bookmark=%s list_id=%s status=ok", bookmark_id, list_id)
    except APIError as exc:
        logging.warning("list_remove bookmark=%s list_id=%s status=warn err=%s", bookmark_id, list_id, exc)


class OpenAIClassifier:
    def __init__(
        self,
        api_key: str,
        model: str,
        site: Optional[str],
        title: Optional[str],
        timeout: float,
        base_url: str,
        categories: Sequence[str],
    ) -> None:
        headers: Dict[str, str] = {}
        # Only send attribution headers when using OpenRouter
        if "openrouter.ai" in base_url:
            headers = {k: v for k, v in {"HTTP-Referer": site, "X-Title": title}.items() if v}
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers if headers else None,
            timeout=timeout,
        )
        self.model = model
        self.categories = list(categories)

    def classify(self, bookmark: BookmarkInfo) -> Tuple[str, float, str]:
        prompt = self._build_prompt(bookmark)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": ("Classify the bookmark into one of the provided categories. " "Prefer a specific category over Shopping unless it is clearly a shopping/product intent. " "If you cannot infer, respond with Uncategorized."),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        message_content = resp.choices[0].message.content
        content = message_content or ""
        return self._parse_response(content)

    def _build_prompt(self, bookmark: BookmarkInfo) -> str:
        tag_part = ", ".join(bookmark.tags) if bookmark.tags else "none"
        allowed = ", ".join(self.categories + [UNCATEGORIZED_LIST_NAME])
        desc = (bookmark.description or "").strip()
        if desc:
            desc = desc[:500]
        text = f"Allowed categories: {allowed}.\n" f"Title: {bookmark.title or 'n/a'}\n" f"URL: {bookmark.url}\n" f"Tags: {tag_part}\n" f"Description: {desc or 'n/a'}\n" 'Respond as JSON: {"category":"<one>","confidence":0-1,"reason":"..."}.\n' "Use Uncategorized if you cannot infer."
        return text

    @staticmethod
    def _parse_response(content: str) -> Tuple[str, float, str]:
        try:
            parsed = json.loads(content.strip())
        except json.JSONDecodeError:
            # attempt to extract JSON block
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(content[start : end + 1])
            else:
                raise
        category = parsed.get("category", UNCATEGORIZED_LIST_NAME)
        confidence = float(parsed.get("confidence", 0))
        reason = parsed.get("reason", "")
        return category, confidence, reason


def choose_target_list(
    category: str,
    confidence: float,
    allowed_categories: Sequence[str],
    shopping_list: Optional[str],
    confidence_threshold: float,
    shopping_min_conf: float,
) -> str:
    if category not in allowed_categories:
        return UNCATEGORIZED_LIST_NAME
    if confidence < confidence_threshold:
        return UNCATEGORIZED_LIST_NAME
    if shopping_list and category == shopping_list and confidence < shopping_min_conf:
        return UNCATEGORIZED_LIST_NAME
    return category


def main() -> None:
    configure_logging()
    base_url = env("KARAKEEP_API_BASE")
    api_key = env("KARAKEEP_API_KEY")
    oa_key = env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    oa_timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    oa_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    dry_run = env_bool("DRY_RUN", False)
    skip_lists = env_csv("SKIP_LISTS")
    source_lists = env_csv("SOURCE_LISTS")
    inbox_lists = env_csv("INBOX_LISTS")
    shopping_list = os.getenv("SHOPPING_LIST", "Shopping")
    shopping_min_conf = 0.75
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
    max_bms = os.getenv("MAX_BOOKMARKS")
    max_bms_int = int(max_bms) if max_bms else None

    kk = KarakeepAPI(api_key=api_key, api_endpoint=normalize_base_url(base_url), verbose=False)
    logging.info(
        "start base=%s model=%s dry_run=%s skip=%s source=%s inbox=%s shopping=%s uncategorized=%s conf_thresh=%.2f max=%s",
        kk.api_endpoint,
        model,
        dry_run,
        ",".join(skip_lists) or "none",
        ",".join(source_lists) or "none",
        ",".join(inbox_lists) or "none",
        shopping_list,
        UNCATEGORIZED_LIST_NAME,
        confidence_threshold,
        max_bms_int if max_bms_int is not None else "all",
    )
    categories, list_ids, source_list_ids, inbox_list_ids = load_categories_and_ids(kk, UNCATEGORIZED_LIST_NAME, dry_run, skip_lists, source_lists, inbox_lists)
    effective_source_ids = set(source_list_ids) | set(inbox_list_ids)
    classifier = OpenAIClassifier(
        api_key=oa_key,
        model=model,
        site=DEFAULT_ATTR_SITE,
        title=DEFAULT_ATTR_TITLE,
        timeout=oa_timeout,
        base_url=oa_base,
        categories=categories,
    )

    if shopping_list not in categories:
        logging.info("shopping list '%s' not in categories; shopping threshold disabled", shopping_list)
    shopping_active = shopping_list in categories
    target_list_names = categories + [UNCATEGORIZED_LIST_NAME]
    current_category_lists = {list_ids[name] for name in target_list_names if list_ids.get(name)}
    inbox_id_set = set(inbox_list_ids)

    moved = 0
    skipped = 0
    failed = 0
    processed = 0
    started = time.time()

    for bm in iter_bookmarks(kk):
        bm_id = getattr(bm, "id", "")

        try:
            list_ids_for_bm = get_list_ids_for_bookmark(kk, getattr(bm, "id", ""))
        except Exception as exc:  # noqa: BLE001
            logging.error("bookmark=%s fetch_lists error=%s", bm_id, exc)
            failed += 1
            continue

        # If source or inbox lists are specified, process only bookmarks that are in at least one
        if effective_source_ids and not effective_source_ids.intersection(set(list_ids_for_bm)):
            skipped += 1
            continue

        bm_info = to_bookmark_info(bm, list_ids_for_bm)

        try:
            category, confidence, reason = classifier.classify(bm_info)
        except Exception as exc:  # noqa: BLE001
            logging.error("bookmark=%s status=classify_error error=%s", bm_info.id, exc)
            failed += 1
            continue

        target_name = choose_target_list(
            category,
            confidence,
            categories,
            shopping_list if shopping_active else None,
            confidence_threshold,
            shopping_min_conf if shopping_active else confidence_threshold,
        )
        target_id = list_ids.get(target_name, list_ids[UNCATEGORIZED_LIST_NAME])

        already_in_target = target_id in bm_info.list_ids
        move_ok = True
        if not already_in_target:
            move_ok = add_to_list(kk, target_id, bm_info.id, dry_run)
        removed_any = False
        removal_ids = current_category_lists.union(inbox_id_set)
        for lid in bm_info.list_ids:
            if lid in removal_ids and lid != target_id:
                remove_from_list(kk, lid, bm_info.id, dry_run)
                removed_any = True

        if (not already_in_target and move_ok) or removed_any:
            moved += 1
            logging.info(
                "bookmark=%s status=moved target=%s conf=%.2f added=%s removed=%s",
                bm_info.id,
                target_name,
                confidence,
                not already_in_target,
                removed_any,
            )
        else:
            skipped += 1
            logging.info(
                "bookmark=%s status=skipped reason=already-in-target target=%s conf=%.2f",
                bm_info.id,
                target_name,
                confidence,
            )

        processed += 1
        if max_bms_int is not None and processed >= max_bms_int:
            break

    elapsed = time.time() - started
    logging.info(
        "done: moved=%s skipped=%s failed=%s elapsed=%.1fs",
        moved,
        skipped,
        failed,
        elapsed,
    )


if __name__ == "__main__":
    main()
