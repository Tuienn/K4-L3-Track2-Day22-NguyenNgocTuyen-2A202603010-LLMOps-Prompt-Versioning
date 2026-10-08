"""Persist Gemini embeddings and pace batches to respect the free-tier quota."""
import hashlib
import json
import threading
import time
from pathlib import Path

from diskcache import Cache
from langchain_core.embeddings import Embeddings


class CachedEmbeddings(Embeddings):
    """Cache document/query vectors separately; retry transient quota errors."""
    def __init__(self, delegate, namespace):
        self.delegate = delegate
        self.namespace = namespace
        self.cache = Cache(str(Path(__file__).resolve().parents[2] / '.cache' / 'embeddings'))
        self.lock = threading.Lock()
        self.next_request = 0.0

    def _key(self, text, task):
        return hashlib.sha256(json.dumps([self.namespace, task, text]).encode()).hexdigest()

    def _call(self, operation, count):
        with self.lock:
            time.sleep(max(0, self.next_request - time.monotonic()))
            for attempt in range(5):
                try:
                    result = operation()
                    self.next_request = time.monotonic() + count * 0.75
                    return result
                except Exception as error:
                    if not any(token in str(error) for token in ('429', 'RESOURCE_EXHAUSTED', '503')) or attempt == 4:
                        raise
                    print(f'Embedding quota/transient error; retry {attempt + 1}/4 in 60s', flush=True)
                    time.sleep(60)

    def embed_documents(self, texts):
        missing = list(dict.fromkeys(t for t in texts if self._key(t, 'document') not in self.cache))
        for start in range(0, len(missing), 30):
            batch = missing[start:start + 30]
            vectors = self._call(lambda: self.delegate.embed_documents(batch), len(batch))
            for text, vector in zip(batch, vectors):
                self.cache[self._key(text, 'document')] = vector
        return [self.cache[self._key(text, 'document')] for text in texts]

    def embed_query(self, text):
        key = self._key(text, 'query')
        if key not in self.cache:
            self.cache[key] = self._call(lambda: self.delegate.embed_query(text), 1)
        return self.cache[key]
