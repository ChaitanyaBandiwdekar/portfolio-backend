"""All langchain_google_genai construction lives here (provider isolation).

Nothing else in this codebase should import langchain_google_genai.
"""

import math
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

_EMBED_MODEL = "models/gemini-embedding-001"
_DIMS = 768
_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-flash-lite-latest")

_doc_embedder: GoogleGenerativeAIEmbeddings | None = None
_query_embedder: GoogleGenerativeAIEmbeddings | None = None
_chat_model: ChatGoogleGenerativeAI | None = None


def _require_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return api_key


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def _get_doc_embedder() -> GoogleGenerativeAIEmbeddings:
    global _doc_embedder
    if _doc_embedder is None:
        _require_api_key()
        _doc_embedder = GoogleGenerativeAIEmbeddings(
            model=_EMBED_MODEL,
            output_dimensionality=_DIMS,
            task_type="RETRIEVAL_DOCUMENT",
        )
    return _doc_embedder


def _get_query_embedder() -> GoogleGenerativeAIEmbeddings:
    global _query_embedder
    if _query_embedder is None:
        _require_api_key()
        _query_embedder = GoogleGenerativeAIEmbeddings(
            model=_EMBED_MODEL,
            output_dimensionality=_DIMS,
            task_type="RETRIEVAL_QUERY",
        )
    return _query_embedder


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Lazily construct the chat model used by the agent (app/agent.py)."""
    global _chat_model
    if _chat_model is None:
        _require_api_key()
        _chat_model = ChatGoogleGenerativeAI(model=_CHAT_MODEL)
    return _chat_model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for ingestion (RETRIEVAL_DOCUMENT), L2-normalized."""
    raw = _get_doc_embedder().embed_documents(texts)
    return [_l2_normalize(v) for v in raw]


def embed_query(text: str) -> list[float]:
    """Embed a search query (RETRIEVAL_QUERY), L2-normalized."""
    raw = _get_query_embedder().embed_query(text)
    return _l2_normalize(raw)
