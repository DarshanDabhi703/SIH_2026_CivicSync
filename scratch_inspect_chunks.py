"""Inspect actual chunk content for the 4 failing domains."""
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.retriever import retrieve

queries = [
    ('labour',        'My company has not paid my salary for two months.'),
    ('land_property', 'My landlord won\'t return my security deposit.'),
    ('insurance',     'My insurer rejected my hospital claim.'),
    ('women_safety',  'Someone is harassing me at my workplace.'),
]

for domain, q in queries:
    print('\n' + '='*64)
    print(f'DOMAIN: {domain}')
    print(f'QUERY : {q}')
    print('='*64)
    chunks = retrieve(q, top_k=5, domain=domain)
    for i, c in enumerate(chunks or [], 1):
        cid = c.get('chunk_id')
        sim = c.get('similarity', 0)
        pg  = c.get('page_start')
        content = (c.get('content') or '').strip()
        print(f'\n--- CHUNK {i} | id={cid} | sim={sim:.4f} | page={pg} ---')
        print(content[:800])
        if len(content) > 800:
            print('...[truncated]')
