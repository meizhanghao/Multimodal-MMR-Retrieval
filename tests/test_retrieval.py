import asyncio
import json
from pathlib import Path
import time
import numpy as np
import pytest
from mmclip.vectors import mmr, unit_rows
from mmclip.index import VectorIndex
from mmclip.encoders import FixtureEncoder
from mmclip.data import load_pairs
from mmclip.pipeline import RetrievalPipeline
from mmclip.batching import BatchEncoder, QueueFullError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    data = json.loads((ROOT / 'examples/features.json').read_text(encoding='utf-8'))
    items, _ = load_pairs(ROOT / 'examples/pairs.json')
    return data, items


def test_mmr_matches_independent_brute_selection():
    rng = np.random.default_rng(19)
    for _ in range(100):
        x = unit_rows(rng.normal(size=(8, 5)))
        q = unit_rows(rng.normal(size=(1, 5)))[0]
        lam = float(rng.uniform())
        expected = [int(np.argmax(x @ q))]
        while len(expected) < 5:
            choices = [i for i in range(len(x)) if i not in expected]
            best = max(choices, key=lambda i: (lam*float(x[i]@q)-(1-lam)*max(float(x[i]@x[j]) for j in expected), -i))
            expected.append(best)
        assert mmr(q, x, 5, lam) == expected


def test_mmr_negative_similarity_and_extremes():
    x = [[1,0],[-1,0],[0,1]]
    assert mmr([1,0], x, 99, 0) == [0,1,2]
    assert mmr([1,0], x, 3, 1) == [0,2,1]
    assert mmr([1,0], np.empty((0,2)), 3) == []
    assert mmr([1,0], x, 0) == []


@pytest.mark.parametrize('vectors', [[[0,0]], [[float('nan'),1]], [[float('inf'),0]]])
def test_bad_vectors_rejected(vectors):
    with pytest.raises(ValueError):
        unit_rows(vectors)


def test_index_roundtrip_ids_and_model_version(fixture, tmp_path):
    data, items = fixture
    index = VectorIndex(data['image_features'], items)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, 'fixture-4d-v1')
    assert loaded.items[loaded.search([1,0,0,0],1)[0][0]]['id'] == 'hospital-1'
    with pytest.raises(ValueError):
        VectorIndex.load(tmp_path, 'different-model')
    hits = loaded.search([1,0,0,0], 99, {'domain': 'medical'})
    assert len(hits) == 3 and all(items[i]['domain'] == 'medical' for i, _ in hits)


@pytest.mark.integration
def test_real_faiss_flat_hnsw(fixture, tmp_path):
    pytest.importorskip('faiss')
    data, items = fixture
    for kind in ['flat','hnsw']:
        index = VectorIndex(data['image_features'], items, backend='faiss', kind=kind)
        index.save(tmp_path / kind)
        loaded = VectorIndex.load(tmp_path / kind)
        result = RetrievalPipeline(loaded).search_vector([1,0,0,0])
        assert result['results'][0]['id'] == 'hospital-1'
        assert len(set(result['candidate_ids'])) == len(items)
        assert loaded.search([1,0,0,0], k=0) == []


def test_fixture_encoder_refuses_unknown_and_images_are_portable(fixture):
    encoder = FixtureEncoder(ROOT / 'examples/features.json')
    assert encoder.encode(['different/machine/hospital-1.png'], 'image').shape == (1,4)
    with pytest.raises(ValueError):
        encoder.encode(['任意未知句子'])


def test_batching_preserves_request_order_and_caches():
    class Encoder:
        fingerprint = 'test-v1'
        calls = []
        def encode(self, texts):
            self.calls.append(texts)
            return np.array([[int(t), 1] for t in texts], dtype=np.float32)
    async def scenario():
        encoder = Encoder()
        batch = BatchEncoder(encoder, batch_size=4, wait_ms=15)
        values = await asyncio.gather(*(batch.encode(str(i)) for i in range(4)))
        assert [value[0] for value in values] == [0,1,2,3]
        assert len(encoder.calls) == 1
        await batch.encode('1')
        assert len(encoder.calls) == 1
        await batch.close()
    asyncio.run(scenario())


def test_queue_full_timeout_and_error_do_not_hang():
    class Encoder:
        fingerprint = 'slow'
        def encode(self, texts):
            time.sleep(0.03)
            if 'bad' in texts:
                raise ValueError('bad input')
            return np.ones((len(texts), 2), dtype=np.float32)
    async def scenario():
        batch = BatchEncoder(Encoder(), capacity=1, batch_size=1, wait_ms=0)
        tasks = [asyncio.create_task(batch.encode(str(i), timeout=0.005)) for i in range(8)]
        result = await asyncio.gather(*tasks, return_exceptions=True)
        assert any(isinstance(item, QueueFullError) for item in result)
        assert any(isinstance(item, TimeoutError) for item in result)
        await asyncio.sleep(0.04)
        with pytest.raises(ValueError):
            await batch.encode('bad')
        await batch.close()
    asyncio.run(scenario())


@pytest.mark.integration
def test_fastapi_lifespan_and_search(fixture):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from mmclip.api import create_app
    data, items = fixture
    app = create_app(VectorIndex(data['image_features'], items), FixtureEncoder(ROOT / 'examples/features.json'))
    with TestClient(app) as client:
        result = client.post('/search', json={'query': '医院设备采购'}).json()
        assert result['results'][0]['id'] == 'hospital-1'
        assert client.post('/search', json={'query': '未知输入'}).status_code == 422
        assert client.get('/documents/not-found').status_code == 404


@pytest.mark.integration
def test_projection_training_updates_heads_only(tmp_path):
    torch = pytest.importorskip('torch')
    from mmclip.training import train_features, symmetric_infonce
    report = train_features(ROOT / 'examples/features.json', tmp_path, steps=30)
    assert report['mode'] == 'synthetic_feature_training'
    assert report['losses'][-1] < report['losses'][0]
    assert report['trainable_parameters'] == 33
    features = torch.eye(3, requires_grad=True)
    loss = symmetric_infonce(features, features)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert (tmp_path / 'projection.pt').is_file()


@pytest.mark.integration
def test_paired_feature_export_keeps_row_identity_and_provenance(tmp_path):
    torch = pytest.importorskip('torch')
    from mmclip.features import export_features
    from PIL import Image

    class MockTower:
        def encode_text(self, tokens):
            return tokens.float()

        def encode_image(self, images):
            return images.mean(dim=(2, 3))

    class MockClip:
        @staticmethod
        def tokenize(texts):
            return torch.tensor([[1, 0, 0] if text == '红色' else [0, 1, 0] for text in texts])

    class MockEncoder:
        fingerprint, device = 'test-frozen-tower', 'cpu'
        model, clip = MockTower(), MockClip()

        @staticmethod
        def preprocess(image):
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()

    rows = []
    for i, (text, color) in enumerate([('红色', 'red'), ('绿色', 'lime')]):
        filename = f'{i}.png'
        Image.new('RGB', (8, 8), color).save(tmp_path / filename)
        rows.append({'id': f'pair-{i}', 'doc_id': 'd', 'paragraph_id': f'p{i}',
                     'text': text, 'image': filename})
    source = tmp_path / 'pairs.json'
    source.write_text(json.dumps({'synthetic': True, 'items': rows}), encoding='utf-8')
    output = tmp_path / 'features.json'
    export_features(source, MockEncoder(), output, batch_size=1)
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['ids'] == ['pair-0', 'pair-1']
    assert result['synthetic'] is True
    assert np.allclose(result['text_features'], result['image_features'])
    assert result['encoder'] == 'test-frozen-tower'
