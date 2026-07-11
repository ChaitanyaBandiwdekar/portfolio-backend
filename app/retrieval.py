"""Query -> ranked, titled chunks via a direct match_documents RPC call.

Deliberately does not use LangChain's SupabaseVectorStore (archived,
unmaintained langchain-community package) — this is a 10-line parameterized
RPC on our own schema instead.
"""

from dataclasses import dataclass

from app.llm import embed_query
from app.supabase_client import get_supabase_client

MATCH_COUNT = 5
SIMILARITY_THRESHOLD = 0.5


@dataclass
class RetrievedChunk:
    title: str
    content: str
    similarity: float


def retrieve(query: str) -> list[RetrievedChunk]:
    """Embed the query, fetch top matches, and filter to those above threshold."""
    vector = embed_query(query)
    client = get_supabase_client()
    response = client.rpc(
        "match_documents",
        {"query_embedding": vector, "match_count": MATCH_COUNT},
    ).execute()

    return [
        RetrievedChunk(title=row["title"], content=row["content"], similarity=row["similarity"])
        for row in response.data
        if row["similarity"] >= SIMILARITY_THRESHOLD
    ]
