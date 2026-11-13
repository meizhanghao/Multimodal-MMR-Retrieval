"""Regression checks for the reviewed temperature, HNSW and serving contracts."""
import asyncio
import json
import math
from pathlib import Path
import time
import numpy as np
import pytest
from mmclip.cache import TTLCache
from mmclip.config import ServiceConfig
from mmclip.index import VectorIndex
from mmclip.pipeline import RetrievalPipeline
from mmclip.batching import BatchEncoder, QueueFullError
from mmclip.encoders import FixtureEncoder

ROOT = Path(__file__).resolve().parents[1]


def fixture_index():
    from mmclip.data import load_pairs
    data = json.loads((ROOT / 'examples/features.json').read_text(encoding='utf-8'))
    pairs, _ = load_pairs(ROOT / 'examples/pairs.json')
    return VectorIndex(data['image_features'], pairs)


@pytest.mark.integration
def test_learnable_temperature_receives_gradient_and_is_capped():
    torch = pytest.importorskip('torch')
    from mmclip.training import DualProjection, symmetric_infonce
    torch.manual_seed(11)
    head = DualProjection(4, 4)
    assert float(head.logit_scale.detach()) == pytest.approx(math.log(1/0.07))
    text, image = head(torch.eye(4), torch.flip(torch.eye(4), dims=[0]))
    loss = symmetric_infonce(text, image, logit_scale=head.scale())
    loss.backward()
    assert head.logit_scale.grad is not None and torch.isfinite(head.logit_scale.grad)
    assert abs(float(head.logit_scale.grad)) > 1e-6
    with torch.no_grad():
        head.logit_scale.fill_(1000)
    assert float(head.scale().detach()) == pytest.approx(100, rel=1e-5)
    fixed = DualProjection(4, 4, learnable_temperature=False)
    assert not fixed.logit_scale.requires_grad


@pytest.mark.integration
def test_fixed_and_learned_temperature_checkpoint_compatibility(tmp_path):
    torch = pytest.importorskip('torch')
    from mmclip.training import train_features
    from mmclip.encoders import ProjectedEncoder
    base = FixtureEncoder(ROOT / 'examples/features.json')
    for trainable in [False, True]:
        out = tmp_path / str(trainable)
        report = train_features(ROOT / 'examples/features.json', out, steps=5,
                                learnable_temperature=trainable, batch_size=4)
        assert len(report['losses']) == 5
        assert report['trainable_parameters'] == (33 if trainable else 32)
        if not trainable:
            assert report['temperatures'] == pytest.approx([0.07]*5)
        projected = ProjectedEncoder(base, out / 'projection.pt')
        assert projected.encode(['医院设备采购']).shape == (1,4)
    saved = torch.load(out / 'projection.pt', weights_only=True)
    del saved['temperature_config']
    del saved['state_dict']['logit_scale']
    saved['schema_version'] = 1
    old = tmp_path / 'old-projection.pt'
    torch.save(saved, old)
    old_encoder = ProjectedEncoder(base, old)
    assert np.allclose(old_encoder.encode(['医院设备采购']), projected.encode(['医院设备采购']))


@pytest.mark.integration
def test_hnsw_parameters_and_versions_persist_and_detect_tamper(tmp_path):
    pytest.importorskip('faiss')
    index = VectorIndex([[1,0], [0,1]], [{'id':'a'}, {'id':'b'}], backend='faiss', kind='hnsw',
                        hnsw_m=16, ef_construction=200, ef_search=128)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.version == index.version
    assert loaded.index.hnsw.efConstruction == 200
    assert loaded.index.hnsw.efSearch == 128
    loaded.set_ef_search(64)
    assert loaded.index.hnsw.efSearch == 64 and loaded.version != index.version
    manifest = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    manifest['items'][0]['id'] = 'tampered-id'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='version mismatch'):
        VectorIndex.load(tmp_path)


@pytest.mark.integration
def test_old_index_manifest_keeps_historical_construction_setting(tmp_path):
    pytest.importorskip('faiss')
    old = VectorIndex([[1,0],[0,1]], [{'id':'a'}, {'id':'b'}], backend='faiss', kind='hnsw', ef_construction=80)
    old.save(tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    manifest['schema_version'] = 1
    del manifest['ef_construction'], manifest['index_version']
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    loaded = VectorIndex.load(tmp_path)
    assert loaded.ef_construction == loaded.index.hnsw.efConstruction == 80


def test_encoding_cache_expiry_eviction_and_copy_isolation():
    now = [0.0]
    cache = TTLCache(2, 3600, clock=lambda: now[0])
    cache.put(('model-v1', 'hash'), np.array([1,2]))
    cache.get(('model-v1', 'hash'))[0] = 99
    assert cache.get(('model-v1', 'hash'))[0] == 1
    assert cache.get(('model-v2', 'hash')) is None
    now[0] = 3600
    assert cache.get(('model-v1', 'hash')) is None
    for key in ['a', 'b', 'c']:
        cache.put(key, np.array([1]))
    assert cache.get('a') is None and len(cache.entries) == 2


def test_batcher_reports_batch_size_and_invalidates_on_model_version_change():
    class Encoder:
        fingerprint = 'v1'
        calls = 0
        def encode(self, texts):
            self.calls += 1
            return np.ones((len(texts), 2), dtype=np.float32)
    async def scenario():
        encoder = Encoder()
        batcher = BatchEncoder(encoder, batch_size=4, wait_ms=10)
        results = await asyncio.gather(*(batcher.encode_with_info(str(i)) for i in range(4)))
        assert all(result.batch_size == 4 and not result.cache_hit for result in results)
        cached = await batcher.encode_with_info('0')
        assert cached.cache_hit and cached.batch_size == 0 and encoder.calls == 1
        encoder.fingerprint = 'v2'
        await batcher.encode('0')
        assert encoder.calls == 2
        await batcher.close()
    asyncio.run(scenario())


def test_mmr_metric_distinguishes_budget_reduction_from_redundancy():
    pipeline = RetrievalPipeline(fixture_index())
    result = pipeline.search_vector([1,0,0,0], candidate_k=6, top_k=3, diversity=1)
    metrics = result['mmr_metrics']
    assert metrics['candidate_reduction_ratio'] == 0.5
    assert metrics['pairwise_cosine_reduction'] == pytest.approx(0)
    assert 'mmr_dedup_ratio' not in result
    single = pipeline.search_vector([1,0,0,0], top_k=1)
    assert single['mmr_metrics']['selected_mean_pairwise_cosine'] is None


@pytest.mark.integration
def test_api_retrieval_contract_validation_and_structured_logs(caplog):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from mmclip.api import create_app
    caplog.set_level('INFO', logger='mmclip.events')
    app = create_app(fixture_index(), FixtureEncoder(ROOT / 'examples/features.json'))
    with TestClient(app) as client:
        health = client.get('/health').json()
        assert health['service_config'] == {
            'queue_capacity':128, 'max_batch':16, 'max_delay_ms':5.0,
            'cache_size':128, 'cache_ttl_seconds':3600.0, 'request_timeout_seconds':10.0}
        response = client.post('/v1/retrieve', json={'query':'医院设备采购'}, headers={'X-Request-ID':'review-001'})
        data = response.json()
        assert response.status_code == 200
        assert data['chunks'][0]['chunk_id'] == 'hospital-1'
        assert data['chunks'][0]['source_doc']['doc_id'] == 'demo-doc-1'
        assert data['index_version'] == health['index_version']
        assert data['estimated_total_tokens'] > 0 and 'total_tokens' not in data
        assert data['request_id'] == response.headers['X-Request-ID'] == 'review-001'
        assert data['chunks'][0]['image_url'] is None
        assert client.post('/encode', json={'query':'医院设备采购'}).json()['cache_hit']
        for body in [{'query':' '}, {'query':'医院设备采购','top_k':51,'candidate_k':60},
                     {'query':'医院设备采购','candidate_k':2,'top_k':3},
                     {'query':'医院设备采购','lambda_mult':1.1}, {'query':'医院设备采购','image':'x'}]:
            assert client.post('/v1/retrieve', json=body).status_code == 422
    logs = [json.loads(record.message) for record in caplog.records if record.name == 'mmclip.events']
    log = next(row for row in logs if row.get('request_id') == 'review-001')
    assert log['batch_size'] == 1 and len(log['query_hash']) == 64 and log['error_code'] is None
    assert '医院设备采购' not in json.dumps(logs, ensure_ascii=False)


@pytest.mark.integration
def test_api_queue_error_and_whole_request_timeout(monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from mmclip.api import create_app
    app = create_app(fixture_index(), FixtureEncoder(ROOT / 'examples/features.json'),
                     ServiceConfig(request_timeout_seconds=0.015, max_delay_ms=0))
    original = app.state.batcher.encode_with_info
    async def full(*args, **kwargs):
        raise QueueFullError('queue full')
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.batcher, 'encode_with_info', full)
        response = client.post('/v1/retrieve', json={'query':'医院设备采购'})
        assert response.status_code == 429 and response.headers['Retry-After'] == '1'
        monkeypatch.setattr(app.state.batcher, 'encode_with_info', original)
        def slow(*args, **kwargs):
            time.sleep(0.04)
            return {}
        monkeypatch.setattr(app.state.pipeline, 'search_vector', slow)
        # Encoding succeeds, but the retrieval stage consumes the deadline.
        assert client.post('/v1/retrieve', json={'query':'医院设备采购'}).status_code == 504


@pytest.mark.integration
def test_hnsw_sweep_reports_all_configurations():
    pytest.importorskip('faiss')
    from mmclip.evaluate import sweep
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(32,8))
    report = sweep(vectors, [{'id':str(i)} for i in range(32)], vectors[:3], repeats=1)
    assert {(row['hnsw_m'], row['ef_search']) for row in report['measurements']} == {
        (16,64), (16,128), (16,256), (32,64), (32,128), (32,256)}
    assert all(row['ef_construction'] == 200 and 0 <= row['ann_recall_at_k'] <= 1
               for row in report['measurements'])
