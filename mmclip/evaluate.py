"""Offline ANN parameter comparison; this does not measure domain semantic recall."""
import time
import numpy as np
from .config import HNSW_M, EF_CONSTRUCTION, EF_SEARCH
from .index import VectorIndex


def _measure(flat, hnsw, queries, k, repeats):
    if k < 1 or len(queries) < 1 or repeats < 1:
        raise ValueError('positive k, query count and repetitions required')
    recalls, latencies = [], []
    # Warm up both paths; each reported latency is one query after warmup.
    for query in queries:
        flat.search(query, k)
        hnsw.search(query, k)
    for query in queries:
        gold = {row for row, _ in flat.search(query, k)}
        for _ in range(repeats):
            started = time.perf_counter()
            found = {row for row, _ in hnsw.search(query, k)}
            latencies.append((time.perf_counter() - started) * 1000)
            recalls.append(len(gold & found) / len(gold))
    import faiss
    return {'query_count': len(queries), 'k': k, 'repeats': repeats,
            'hnsw_m': hnsw.hnsw_m, 'ef_construction': hnsw.ef_construction, 'ef_search': hnsw.ef_search,
            'ann_recall_at_k': float(np.mean(recalls)),
            'hnsw_p50_ms': float(np.percentile(latencies, 50)),
            'hnsw_p95_ms': float(np.percentile(latencies, 95)),
            'flat_serialized_bytes': int(faiss.serialize_index(flat.index).nbytes),
            'hnsw_serialized_bytes': int(faiss.serialize_index(hnsw.index).nbytes),
            'raw_vector_bytes': int(flat.vectors.nbytes),
            'note': 'ANN neighbor overlap, not semantic recall; serialized bytes exclude NumPy copy, metadata and peak RAM'}


def evaluate(vectors, items, queries, k=3, *, hnsw_m=HNSW_M, ef_construction=EF_CONSTRUCTION,
             ef_search=EF_SEARCH, repeats=5):
    flat = VectorIndex(vectors, items, backend='faiss', kind='flat')
    hnsw = VectorIndex(vectors, items, backend='faiss', kind='hnsw', hnsw_m=hnsw_m,
                      ef_construction=ef_construction, ef_search=ef_search)
    return _measure(flat, hnsw, queries, k, repeats)


def sweep(vectors, items, queries, *, m_values=(16, 32), ef_values=(64, 128, 256),
          ef_construction=EF_CONSTRUCTION, k=3, repeats=5):
    if not m_values or not ef_values:
        raise ValueError('nonempty parameter grids required')
    flat = VectorIndex(vectors, items, backend='faiss', kind='flat')
    results = []
    for m in m_values:
        hnsw = VectorIndex(vectors, items, backend='faiss', kind='hnsw', hnsw_m=m,
                          ef_construction=ef_construction)
        for ef in ef_values:
            hnsw.set_ef_search(ef)
            results.append(_measure(flat, hnsw, queries, k, repeats))
    return {'mode': 'offline_parameter_sweep', 'measurements': results,
            'note': 'No parameter is declared optimal. Use held-out queries at the intended data scale.'}
