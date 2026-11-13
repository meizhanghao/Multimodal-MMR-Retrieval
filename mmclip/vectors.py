"""Numerical contracts shared by encoding, indexing and MMR."""
import numpy as np


def unit_rows(values, *, dim=None):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or (dim is not None and x.shape[1] != dim):
        raise ValueError('expected an N x D matrix with the configured dimension')
    if not np.isfinite(x).all():
        raise ValueError('vectors must be finite')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError('zero vectors have no cosine direction')
    return np.ascontiguousarray(x / norms, dtype=np.float32)


def mmr(query, candidates, k, diversity=0.55):
    """Return row indices; diversity is lambda (larger means more relevance).

    Incremental maximum similarity needs O(N) auxiliary space, not N x N.
    First selection maximizes relevance. Later negative similarities remain
    negative; initializing them to zero would silently change the objective.
    """
    if k < 0 or not 0 <= diversity <= 1:
        raise ValueError('k >= 0 and lambda in [0, 1] are required')
    matrix = np.asarray(candidates, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError('candidates must be N x D')
    if not len(matrix) or not k:
        return []
    matrix = unit_rows(matrix)
    q = unit_rows(np.asarray(query).reshape(1, -1), dim=matrix.shape[1])[0]
    relevance = matrix @ q
    selected = [int(np.argmax(relevance))]
    remaining = np.ones(len(matrix), dtype=bool)
    remaining[selected[0]] = False
    max_similarity = matrix @ matrix[selected[0]]
    for _ in range(1, min(k, len(matrix))):
        scores = diversity * relevance - (1 - diversity) * max_similarity
        scores[~remaining] = -np.inf
        chosen = int(np.argmax(scores))
        selected.append(chosen)
        remaining[chosen] = False
        max_similarity = np.maximum(max_similarity, matrix @ matrix[chosen])
    return selected
