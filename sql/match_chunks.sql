-- ============================================================
-- CivicSync Phase 2C — pgvector Retrieval Function
-- Run this once in your Supabase Dashboard → SQL Editor
-- ============================================================

create or replace function public.match_chunks(
    query_embedding extensions.vector(768),
    match_count integer,
    filter_domain text default null
)
returns table (
    chunk_id bigint,
    content text,
    source_file text,
    page_start integer,
    page_end integer,
    domain text,
    similarity double precision
)
language sql
stable
as $$
    select
        c.id as chunk_id,
        c.content,
        d.filename as source_file,
        c.page_start,
        c.page_end,
        d.domain,
        (1 - (e.embedding <=> query_embedding))::double precision
            as similarity
    from public.embeddings e
    join public.chunks c
        on e.chunk_id = c.id
    join public.documents d
        on c.document_id = d.id
    where
        filter_domain is null
        or d.domain = filter_domain
    order by e.embedding <=> query_embedding
    limit match_count;
$$;
