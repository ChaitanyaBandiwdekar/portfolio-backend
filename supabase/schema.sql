-- Phase 1: Supabase schema for the portfolio chatbot vector store.

create extension if not exists vector with schema extensions;

create table documents (
  id bigint generated always as identity primary key,
  source text,
  title text,
  content text,
  content_hash text unique not null,
  embedding vector(768)
);

create index on documents using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

create or replace function match_documents(
  query_embedding vector(768),
  match_count int
)
returns table (id bigint, source text, title text, content text, similarity float)
language sql stable
as $$
  select id, source, title, content,
         1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;

alter table documents enable row level security;
-- No policies created: the service-role key bypasses RLS server-side,
-- and the anon key is granted no access.
