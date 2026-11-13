import numpy as np
from .vectors import mmr


class RetrievalPipeline:
    def __init__(self, index):
        self.index = index

    def search_vector(self, query, *, candidate_k=6, top_k=3, diversity=0.55, filters=None):
        if top_k < 0 or candidate_k < top_k:
            raise ValueError('require candidate_k >= top_k >= 0')
        hits = self.index.search(query, candidate_k, filters)
        rows = [row for row, _ in hits]
        selected = mmr(query, self.index.vectors[rows], top_k, diversity) if rows else []
        # Use the same result count for the relevance-only baseline and MMR.
        # N -> K reduction measures a budget, not semantic deduplication.
        selected_rows = [rows[i] for i in selected]
        baseline_rows = rows[:len(selected)]
        def mean_similarity(chosen):
            if len(chosen) < 2:
                return None
            vectors = self.index.vectors[chosen]
            values = (vectors @ vectors.T)[np.triu_indices(len(chosen), k=1)]
            return float(values.mean())
        baseline_similarity = mean_similarity(baseline_rows)
        selected_similarity = mean_similarity(selected_rows)
        return {
            'schema_version': 1,
            'encoder': self.index.encoder,
            'index_version': self.index.version,
            'backend': self.index.backend,
            'index_kind': self.index.kind,
            'candidate_ids': [self.index.items[row]['id'] for row in rows],
            'mmr_metrics': {
                'candidate_count': len(rows), 'selected_count': len(selected), 'lambda_mult': diversity,
                'candidate_reduction_ratio': 1 - len(selected) / len(rows) if rows else 0.0,
                'relevance_top_k_mean_pairwise_cosine': baseline_similarity,
                'selected_mean_pairwise_cosine': selected_similarity,
                'pairwise_cosine_reduction': (baseline_similarity - selected_similarity
                    if baseline_similarity is not None else None),
            },
            'results': [{**self.index.items[rows[i]], 'cosine_score': hits[i][1], 'rank': rank}
                        for rank, i in enumerate(selected, 1)],
        }

    @staticmethod
    def as_retrieval_response(result):
        """Text-only handoff to an upper RAG layer; no prompt or image token estimate."""
        chunks = [{'chunk_id': item['id'], 'text': item.get('text', ''),
                   'image_url': item.get('image_url'), 'image_reference': item.get('image'),
                   'score': item['cosine_score'], 'rank': item['rank'],
                   'source_doc': {'doc_id': item.get('doc_id'), 'paragraph_id': item.get('paragraph_id'),
                                  'source': item.get('source')}} for item in result['results']]
        estimate = sum((len(item['text'].encode('utf-8')) + 3) // 4 for item in chunks)
        return {'schema_version': 1, 'chunks': chunks, 'encoder': result['encoder'],
                'index_version': result['index_version'], 'estimated_total_tokens': estimate,
                'token_estimate_method': 'ceil(utf8_bytes/4) per text; excludes images and prompt',
                'mmr_metrics': result['mmr_metrics']}
