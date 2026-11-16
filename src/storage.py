"""Real pgvector retrieval, BM25 and RRF over the same corpus."""
import importlib.util
import json
import re
from functools import lru_cache
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from rank_bm25 import BM25Okapi

from .common import (CORPORA, DATABASE_URL, DAY6, DAY6_CONFIG, DAY6_EXPERIMENT, DIMENSIONS,
                     EMBEDDING_MODEL, ROOT, api_client, digest, read_json, record_usage)


@lru_cache(maxsize=1)
def bge_model():
    import torch
    torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(DAY6_CONFIG['model'], revision=DAY6_CONFIG['model_revision'],
                               device='cpu', cache_folder=str(ROOT / '.cache/huggingface'),
                               local_files_only=True)


def bge_vectors(texts, query=False):
    cfg = DAY6_CONFIG
    inputs = [cfg['query_instruction'] + text for text in texts] if query else texts
    model = bge_model()
    if any(len(model.tokenizer(text)['input_ids']) > 512 for text in inputs):
        raise ValueError('BGE input exceeds 512 tokens')
    return model.encode(inputs, normalize_embeddings=True, batch_size=16,
                        show_progress_bar=len(inputs) > 10, convert_to_numpy=True)


def openai_vectors(texts):
    cache = ROOT / '.cache/embeddings' / (digest([EMBEDDING_MODEL, DIMENSIONS, texts]) + '.npy')
    if cache.exists():
        vectors = np.load(cache, allow_pickle=False)
    else:
        vectors = []
        for offset in range(0, len(texts), 64):
            response = api_client().embeddings.create(
                model=EMBEDDING_MODEL, dimensions=DIMENSIONS, input=texts[offset:offset + 64])
            record_usage('embedding', response.model, response.usage)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        vectors = np.asarray(vectors, dtype=np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f'{cache.stem}-{uuid4().hex}.npy')
        np.save(temporary, vectors)
        temporary.replace(cache)
    if vectors.shape != (len(texts), DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding dimensions or values')
    return vectors


def connect():
    conn = psycopg.connect(DATABASE_URL, connect_timeout=10)
    register_vector(conn)
    return conn


def ingest_nist(corpus='nist'):
    chunks = read_json(ROOT / f'data/processed/{corpus}_chunks.json')
    vectors = openai_vectors([item['text'] for item in chunks])
    with connect() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS lab7_chunks (
            corpus text NOT NULL, id text NOT NULL, text text NOT NULL,
            source_id text NOT NULL, source text NOT NULL, page text NOT NULL,
            metadata jsonb NOT NULL, model text NOT NULL, embedding vector(1536) NOT NULL,
            PRIMARY KEY (corpus, id))''')
        conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (f'lab7_{corpus}_ingest',))
        conn.execute('DELETE FROM lab7_chunks WHERE corpus=%s', (corpus,))
        with conn.cursor() as cursor:
            cursor.executemany('INSERT INTO lab7_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                [(corpus, item['id'], item['text'], item['source_id'], item['source'], item['page'],
                  Jsonb(item['metadata']), EMBEDDING_MODEL, vector)
                 for item, vector in zip(chunks, vectors)])
    return {'corpus': corpus, 'chunks': len(chunks), 'embedding_model': EMBEDDING_MODEL,
            'dimensions': DIMENSIONS}


def restore_day6():
    """Reuse the original Day 6 store and immutable B chunks in its original scope."""
    chunk_path = DAY6 / 'results/chunks_B.jsonl'
    chunks = [SimpleNamespace(**json.loads(line))
              for line in chunk_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    cfg = DAY6_CONFIG
    cache = ROOT / '.cache/embeddings' / (digest([cfg, [item.text for item in chunks]]) + '.npy')
    if cache.exists():
        vectors = np.load(cache, allow_pickle=False)
    else:
        vectors = bge_vectors([item.text for item in chunks])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    spec = importlib.util.spec_from_file_location('day6_store', DAY6 / 'starter/src/store.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = module.PgVectorStore(DATABASE_URL, dimensions=768, experiment=DAY6_EXPERIMENT)
    store.create_schema()
    store.upsert('B', chunks, vectors)
    return {'corpus': 'day6', 'chunks': store.count('B'), 'experiment': DAY6_EXPERIMENT,
            'embedding_model': cfg['model'], 'dimensions': 768}


def tokens(text):
    return re.findall(r'[\w]+', text.lower())


class Retriever:
    def __init__(self, corpus):
        if corpus not in {'day6', *CORPORA}:
            raise ValueError('Unknown corpus')
        self.corpus = corpus
        self.pdf_path = ROOT / CORPORA[corpus]['path'] if corpus in CORPORA else None
        with connect() as conn:
            if corpus in CORPORA:
                rows = conn.execute('''SELECT id,text,source_id,source,page,metadata,model
                    FROM lab7_chunks WHERE corpus=%s ORDER BY id''', (corpus,)).fetchall()
                if any(row[6] != EMBEDDING_MODEL for row in rows):
                    raise ValueError('Embedding model changed; reindex NIST before querying')
            else:
                rows = conn.execute('''SELECT chunk_id,content,document_id,source,page,metadata
                    FROM lab6_chunks WHERE experiment=%s AND strategy='B' ORDER BY chunk_id''',
                    (DAY6_EXPERIMENT,)).fetchall()
        self.chunks = [{'id': row[0], 'text': row[1], 'source_id': row[2],
                        'source': row[3], 'page': row[4], 'metadata': row[5]} for row in rows]
        if not self.chunks:
            raise ValueError(f'{corpus} is empty. Run ingest first.')
        self.by_id = {item['id']: item for item in self.chunks}
        self.bm25 = BM25Okapi([tokens(item['text']) for item in self.chunks])

    def dense(self, query, top_k, filters):
        vector = openai_vectors([query])[0] if self.corpus in CORPORA else bge_vectors([query], query=True)[0]
        with connect() as conn:
            if self.corpus in CORPORA:
                rows = conn.execute('''SELECT id,embedding <=> %s AS distance FROM lab7_chunks
                    WHERE corpus=%s AND metadata @> %s ORDER BY distance,id LIMIT %s''',
                    (vector, self.corpus, Jsonb(filters), top_k)).fetchall()
            else:
                rows = conn.execute('''SELECT chunk_id,embedding <=> %s AS distance FROM lab6_chunks
                    WHERE experiment=%s AND strategy='B' AND metadata @> %s
                    ORDER BY distance,chunk_id LIMIT %s''',
                    (vector, DAY6_EXPERIMENT, Jsonb(filters), top_k)).fetchall()
        return [{**self.by_id[row[0]], 'score': 1 - float(row[1]), 'rank': rank}
                for rank, row in enumerate(rows, 1)]

    def lexical(self, query, top_k, filters):
        scores = self.bm25.get_scores(tokens(query))
        candidates = [(i, float(score)) for i, score in enumerate(scores)
                      if score > 0 and all(self.chunks[i]['metadata'].get(k) == value
                                          for k, value in filters.items())]
        references = set(re.findall(r'\b(figure|table|map)\s+([a-z0-9]+(?:\.[a-z0-9]+)*)', query.lower()))

        def priority(candidate):
            index, score = candidate
            label = self.chunks[index]['metadata'].get('label', '').lower()
            label_refs = set(re.findall(r'\b(figure|table|map)\s+([a-z0-9]+(?:\.[a-z0-9]+)*)', label))
            explicit_reference = self.corpus == 'wdr' and bool(references & label_refs)
            return (not explicit_reference, -score, self.chunks[index]['id'])

        candidates.sort(key=priority)
        return [{**self.chunks[i], 'score': score, 'rank': rank}
                for rank, (i, score) in enumerate(candidates[:top_k], 1)]

    def search(self, query, method='hybrid', top_k=6, filters=None):
        if not query.strip() or not 1 <= top_k <= 30:
            raise ValueError('Invalid query or top_k')
        filters = filters or {}
        if set(filters) - {'kind', 'topic', 'pdf_page', 'printed_page'}:
            raise ValueError('Unsupported metadata filter')
        if method == 'dense':
            return self.dense(query, top_k, filters)
        if method == 'bm25':
            return self.lexical(query, top_k, filters)
        if method != 'hybrid':
            raise ValueError('Method must be dense, bm25 or hybrid')
        dense = self.dense(query, max(20, top_k), filters)
        lexical = self.lexical(query, max(20, top_k), filters)
        scores, entries = {}, {}
        for ranking in [dense, lexical]:
            for item in ranking:
                identity = item['id']
                entries[identity] = item
                scores[identity] = scores.get(identity, 0) + 1 / (60 + item['rank'])
        ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))[:top_k]
        return [{**entries[identity], 'score': scores[identity], 'rank': rank}
                for rank, identity in enumerate(ordered, 1)]

    def evidence(self, identity):
        item = self.by_id[identity]
        visual_id = item['metadata'].get('visual_id')
        if visual_id:
            return [chunk for chunk in self.chunks if chunk['metadata'].get('visual_id') == visual_id]
        return [chunk for chunk in self.chunks
                if chunk['source_id'] == item['source_id'] and chunk['page'] == item['page']]
