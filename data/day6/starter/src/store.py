"""PostgreSQL contains text, traceable metadata AND vector(768) in one row."""
import json
import numpy as np
import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector

ALLOWED_FILTERS = {'topic', 'version', 'language'}


class PgVectorStore:
    def __init__(self, database_url, dimensions=768, experiment='default'):
        if dimensions != 768:
            raise ValueError('This schema is fixed to BGE-base 768 dimensions')
        self.url, self.dimensions, self.experiment = database_url, dimensions, experiment

    def connect(self):
        connection = psycopg.connect(self.url, connect_timeout=10)
        register_vector(connection)
        return connection

    def create_schema(self):
        with psycopg.connect(self.url) as conn:
            conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            conn.execute('''CREATE TABLE IF NOT EXISTS lab6_chunks (
              experiment text NOT NULL, strategy text NOT NULL CHECK(strategy IN ('A','B')),
              chunk_id text NOT NULL, document_id text NOT NULL,
              source text NOT NULL, page text NOT NULL, content text NOT NULL,
              metadata jsonb NOT NULL, embedding vector(768) NOT NULL,
              PRIMARY KEY(experiment,strategy,chunk_id))''')
            conn.execute('CREATE INDEX IF NOT EXISTS lab6_scope_idx ON lab6_chunks(experiment,strategy)')
            conn.execute("CREATE INDEX IF NOT EXISTS lab6_topic_idx ON lab6_chunks((metadata->>'topic'))")
        # No HNSW/IVFFlat: exact cosine search, no ANN recall confound.

    def upsert(self, strategy, chunks, vectors):
        if strategy not in {'A', 'B'} or len(chunks) != len(vectors) or not chunks:
            raise ValueError('Invalid strategy or vector/chunk lengths')
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.dimensions) or not np.isfinite(vectors).all():
            raise ValueError('Invalid embedding dimensions or non-finite vectors')
        if (np.linalg.norm(vectors, axis=1) == 0).any():
            raise ValueError('Zero vectors are invalid for cosine')
        rows = [(self.experiment, strategy, c.chunk_id, c.document_id, c.source, c.page,
                 c.text, Jsonb(c.metadata), v) for c, v in zip(chunks, vectors)]
        with self.connect() as conn:
            # Atomic replacement of this experiment/strategy removes obsolete chunks.
            conn.execute('DELETE FROM lab6_chunks WHERE experiment=%s AND strategy=%s', (self.experiment, strategy))
            with conn.cursor() as cur:
                cur.executemany('INSERT INTO lab6_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)', rows)

    def count(self, strategy):
        with self.connect() as conn:
            return conn.execute('SELECT count(*) FROM lab6_chunks WHERE experiment=%s AND strategy=%s',
                                (self.experiment, strategy)).fetchone()[0]

    def search(self, strategy, query_vector, top_k=3, filters=None):
        if strategy not in {'A', 'B'} or not 1 <= top_k <= 100:
            raise ValueError('Invalid search parameters')
        filters = filters or {}
        if set(filters) - ALLOWED_FILTERS:
            raise ValueError('Filter field is not permitted')
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.shape != (self.dimensions,) or not np.isfinite(query_vector).all() or np.linalg.norm(query_vector) == 0:
            raise ValueError('Invalid query vector')
        clauses = ['experiment=%s', 'strategy=%s']
        params = [self.experiment, strategy]
        for key, value in filters.items():
            if not isinstance(value, str):
                raise ValueError('Filter values must be strings')
            clauses.append('metadata ->> %s = %s')
            params.extend([key, value])
        sql = '''SELECT chunk_id,document_id,source,page,content,metadata,
                 embedding <=> %s AS distance FROM lab6_chunks WHERE ''' + ' AND '.join(clauses) + ' ORDER BY distance,chunk_id LIMIT %s'
        with self.connect() as conn:
            rows = conn.execute(sql, [query_vector, *params, top_k]).fetchall()
        return [{'chunk_id': r[0], 'document_id': r[1], 'source': r[2], 'page': r[3],
                 'text': r[4], 'metadata': r[5], 'distance': float(r[6]), 'score': 1 - float(r[6])} for r in rows]

    def versions(self):
        with self.connect() as conn:
            return conn.execute("SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname='vector'").fetchone()
