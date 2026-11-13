"""Exact NumPy baseline or real FAISS Flat/HNSW with explicit ID mapping."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import HNSW_M, EF_CONSTRUCTION, EF_SEARCH
from .vectors import unit_rows


class VectorIndex:
    def __init__(self, vectors, items, *, backend='numpy', kind='flat', encoder='fixture-4d-v1',
                 hnsw_m=HNSW_M, ef_construction=EF_CONSTRUCTION, ef_search=EF_SEARCH):
        self.vectors = unit_rows(vectors)
        if not len(items) or len(items) != len(self.vectors):
            raise ValueError('vectors and nonempty metadata must align')
        if len({item['id'] for item in items}) != len(items):
            raise ValueError('document IDs must be unique')
        if backend not in ['numpy', 'faiss'] or kind not in ['flat', 'hnsw']:
            raise ValueError('unknown backend/index kind')
        if backend == 'numpy' and kind != 'flat':
            raise ValueError('HNSW requires FAISS; no silent fallback')
        if any(type(v) is not int or v < 1 for v in (hnsw_m, ef_construction, ef_search)) or hnsw_m < 2:
            raise ValueError('HNSW requires M >= 2 and positive integer ef settings')
        self.items, self.backend, self.kind, self.encoder = items, backend, kind, encoder
        self.hnsw_m, self.ef_construction, self.ef_search = hnsw_m, ef_construction, ef_search
        self.index = None
        if backend == 'faiss':
            import faiss
            dim = self.vectors.shape[1]
            if kind == 'flat':
                self.index = faiss.IndexFlatIP(dim)
            else:
                self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = ef_construction
                self.index.hnsw.efSearch = ef_search
            self.index.add(self.vectors)
        self.version = self._version()

    def _version(self):
        settings = {'encoder': self.encoder, 'backend': self.backend, 'kind': self.kind,
                    'items': self.items, 'shape': self.vectors.shape,
                    'hnsw_m': self.hnsw_m, 'ef_construction': self.ef_construction,
                    'ef_search': self.ef_search}
        digest = hashlib.sha256(json.dumps(settings, ensure_ascii=False, sort_keys=True).encode('utf-8'))
        digest.update(self.vectors.tobytes())
        return digest.hexdigest()[:24]

    def set_ef_search(self, value):
        """Offline tuning only; do not mutate a shared index during serving."""
        if type(value) is not int or value < 1 or self.kind != 'hnsw':
            raise ValueError('positive efSearch and an HNSW index required')
        self.ef_search = value
        self.index.hnsw.efSearch = value
        self.version = self._version()

    def search(self, query, k=5, filters=None):
        if k < 0:
            raise ValueError('negative k')
        q = unit_rows(np.asarray(query).reshape(1, -1), dim=self.vectors.shape[1])
        if k == 0:
            return []
        filters = filters or {}
        if set(filters) - {'doc_id', 'country', 'domain'}:
            raise ValueError('unsupported metadata filter')
        # For this small reproduction, scoped search is an exact prefilter.
        # It is explicitly not a claim that FAISS HNSW has native ACL filtering.
        if self.backend == 'numpy' or filters:
            rows = [i for i, item in enumerate(self.items)
                    if all(item.get(key) == value for key, value in filters.items())]
            rows.sort(key=lambda i: (-float(self.vectors[i] @ q[0]), i))
            chosen = rows[:k]
            return [(i, float(self.vectors[i] @ q[0])) for i in chosen]
        scores, rows = self.index.search(q, min(k, len(self.items)))
        return [(int(i), float(score)) for i, score in zip(rows[0], scores[0]) if i >= 0]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / 'vectors.npy', self.vectors, allow_pickle=False)
        (directory / 'manifest.json').write_text(json.dumps({
            'schema_version': 2, 'encoder': self.encoder, 'backend': self.backend,
            'kind': self.kind, 'items': self.items, 'dimension': self.vectors.shape[1],
            'hnsw_m': self.hnsw_m, 'ef_construction': self.ef_construction,
            'ef_search': self.ef_search, 'index_version': self.version,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        if self.backend == 'faiss':
            import faiss
            faiss.write_index(self.index, str(directory / 'index.faiss'))

    @classmethod
    def load(cls, directory, expected_encoder=None):
        directory = Path(directory)
        data = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if expected_encoder and expected_encoder != data['encoder']:
            raise ValueError('encoder/index version mismatch; rebuild index')
        stored_vectors = np.load(directory / 'vectors.npy', allow_pickle=False)
        instance = cls(stored_vectors, data['items'],
                       backend=data['backend'], kind=data['kind'], encoder=data['encoder'],
                       hnsw_m=data['hnsw_m'], ef_search=data['ef_search'],
                       ef_construction=data.get('ef_construction', 80))
        if not np.allclose(np.linalg.norm(stored_vectors, axis=1), 1, atol=1e-5):
            raise ValueError('stored index vectors must be normalized')
        # Preserve the stored bytes for a stable version on roundtrip; a second
        # normalization could otherwise change the last float32 bit.
        instance.vectors = np.ascontiguousarray(stored_vectors, dtype=np.float32)
        instance.version = instance._version()
        if data.get('index_version') and data['index_version'] != instance.version:
            raise ValueError('index vectors/metadata/version mismatch')
        if instance.vectors.shape[1] != data['dimension']:
            raise ValueError('index dimension mismatch')
        if data['backend'] == 'faiss':
            import faiss
            instance.index = faiss.read_index(str(directory / 'index.faiss'))
            if instance.index.d != data['dimension'] or instance.index.ntotal != len(data['items']):
                raise ValueError('serialized FAISS and metadata mismatch')
        return instance
