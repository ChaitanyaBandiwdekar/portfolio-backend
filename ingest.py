"""Ingestion CLI: turn docs/*.md into embedded, idempotent rows in Supabase.

Usage: python ingest.py
"""

import hashlib
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langsmith import traceable

import app.llm as llm
from app.supabase_client import get_supabase_client

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBED_GROUP_SIZE = 5
EMBED_SLEEP_SECONDS = 1.0
MAX_RETRIES = 5

logger = logging.getLogger(__name__)


# --- Pure functions (no network calls) -------------------------------------


def chunk_markdown(text: str) -> list[dict]:
    """Split markdown into heading-aware chunks.

    Returns a list of {"text": <heading-path-prefixed chunk>, "heading_path": str}.
    """
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    sections = header_splitter.split_text(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(sections)

    chunks = []
    for doc in docs:
        heading_path = " > ".join(
            doc.metadata[key] for key in ("h1", "h2", "h3") if key in doc.metadata
        )
        prefixed = f"{heading_path}\n\n{doc.page_content}" if heading_path else doc.page_content
        chunks.append({"text": prefixed, "heading_path": heading_path})
    return chunks


def content_hash(source: str, chunk_text: str) -> str:
    return hashlib.sha256((source + chunk_text).encode("utf-8")).hexdigest()


def build_rows(source: str, title: str, text: str) -> list[dict]:
    """Markdown text -> row dicts (no embedding yet) ready for hashing/upsert."""
    rows = []
    for chunk in chunk_markdown(text):
        rows.append(
            {
                "source": source,
                "title": title,
                "content": chunk["text"],
                "content_hash": content_hash(source, chunk["text"]),
            }
        )
    return rows


def stale_hashes(existing_hashes: set[str], current_hashes: set[str]) -> set[str]:
    """Hashes present in a previous run but not the current one -> should be deleted."""
    return existing_hashes - current_hashes


def title_from_markdown(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


# --- Network-touching functions ---------------------------------------------


def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            return llm.embed_documents(texts)
        except Exception as exc:
            is_last = attempt == MAX_RETRIES - 1
            is_rate_limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if is_last or not is_rate_limited:
                raise
            logger.warning("embed retry %d after error: %s", attempt + 1, exc)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Embed in small groups with short sleeps between groups (rate-limit friendly)."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_GROUP_SIZE):
        group = texts[i : i + EMBED_GROUP_SIZE]
        vectors.extend(_embed_with_retry(group))
        if i + EMBED_GROUP_SIZE < len(texts):
            time.sleep(EMBED_SLEEP_SECONDS)
    return vectors


def ingest_file(source: str, rows: list[dict], client) -> None:
    """Embed + upsert the given rows, then delete stale rows for this source."""
    if rows:
        vectors = embed_chunks([row["content"] for row in rows])
        for row, vector in zip(rows, vectors):
            row["embedding"] = vector
        client.table("documents").upsert(rows, on_conflict="content_hash").execute()

    current_hashes = {row["content_hash"] for row in rows}
    existing = (
        client.table("documents").select("content_hash").eq("source", source).execute()
    )
    existing_hashes = {row["content_hash"] for row in existing.data}
    to_delete = stale_hashes(existing_hashes, current_hashes)
    if to_delete:
        client.table("documents").delete().eq("source", source).in_(
            "content_hash", list(to_delete)
        ).execute()


@traceable(name="ingest")
def ingest_docs(docs_dir: Path = DOCS_DIR) -> int:
    client = get_supabase_client()
    total = 0
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = title_from_markdown(text) or path.stem
        rows = build_rows(path.name, title, text)
        ingest_file(path.name, rows, client)
        total += len(rows)
        logger.info("ingested %s: %d chunks", path.name, len(rows))
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunk_count = ingest_docs()
    print(f"Ingested {chunk_count} chunks")
