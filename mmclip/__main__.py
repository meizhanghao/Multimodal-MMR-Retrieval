import argparse
import json
import logging
from pathlib import Path
import numpy as np
from .data import load_pairs
from .encoders import FixtureEncoder, ChineseCLIPEncoder, ProjectedEncoder
from .index import VectorIndex
from .pipeline import RetrievalPipeline
from .config import HNSW_M, EF_CONSTRUCTION, EF_SEARCH, INITIAL_TEMPERATURE, MAX_LOGIT_SCALE, ServiceConfig


def main():
    parser = argparse.ArgumentParser(description='Inspectable multimodal retrieval input/output')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo')
    demo.add_argument('--backend', choices=['numpy', 'faiss'], default='numpy')
    demo.add_argument('--kind', choices=['flat', 'hnsw'], default='flat')
    demo.add_argument('--out', default='runs/demo')
    build = commands.add_parser('build')
    build.add_argument('--pairs', default='examples/pairs.json')
    build.add_argument('--encoder', choices=['fixture', 'chinese-clip'], default='fixture')
    build.add_argument('--device', default='cpu')
    build.add_argument('--projection')
    build.add_argument('--backend', choices=['numpy', 'faiss'], default='faiss')
    build.add_argument('--kind', choices=['flat', 'hnsw'], default='flat')
    build.add_argument('--out', default='runs/index')
    search = commands.add_parser('search')
    search.add_argument('--index', default='runs/index')
    search.add_argument('--query', default='医院设备采购')
    search.add_argument('--vector', help='JSON vector; ideal upstream feature input')
    search.add_argument('--projection')
    search.add_argument('--encoder', choices=['fixture', 'chinese-clip'], default='fixture')
    serve = commands.add_parser('serve')
    serve.add_argument('--index', default='runs/index')
    serve.add_argument('--encoder', choices=['fixture', 'chinese-clip'], default='fixture')
    serve.add_argument('--projection')
    serve.add_argument('--port', type=int, default=8001)
    train = commands.add_parser('train')
    train.add_argument('--features', default='examples/features.json')
    train.add_argument('--out', default='runs/train')
    train.add_argument('--steps', type=int, default=60)
    train.add_argument('--learning-rate', type=float, default=0.01)
    train.add_argument('--temperature', type=float, default=INITIAL_TEMPERATURE)
    train.add_argument('--fixed-temperature', action='store_true')
    train.add_argument('--max-logit-scale', type=float, default=MAX_LOGIT_SCALE)
    train.add_argument('--batch-size', type=int, default=32)
    export = commands.add_parser('export-features')
    export.add_argument('--pairs', default='examples/pairs.json')
    export.add_argument('--device', default='cpu')
    export.add_argument('--out', default='runs/real-features.json')
    evaluate_parser = commands.add_parser('evaluate')
    evaluate_parser.add_argument('--sweep', action='store_true')
    evaluate_parser.add_argument('--m-values', type=int, nargs='+', default=[16, 32])
    evaluate_parser.add_argument('--ef-values', type=int, nargs='+', default=[64, 128, 256])
    evaluate_parser.add_argument('--repeats', type=int, default=5)
    for command in (demo, build, evaluate_parser):
        command.add_argument('--hnsw-m', type=int, default=HNSW_M)
        command.add_argument('--ef-construction', type=int, default=EF_CONSTRUCTION)
        command.add_argument('--ef-search', type=int, default=EF_SEARCH)
    service_defaults = ServiceConfig()
    serve.add_argument('--queue-capacity', type=int, default=service_defaults.queue_capacity)
    serve.add_argument('--max-batch', type=int, default=service_defaults.max_batch)
    serve.add_argument('--max-delay-ms', type=float, default=service_defaults.max_delay_ms)
    serve.add_argument('--cache-size', type=int, default=service_defaults.cache_size)
    serve.add_argument('--cache-ttl-seconds', type=float, default=service_defaults.cache_ttl_seconds)
    serve.add_argument('--request-timeout-seconds', type=float, default=service_defaults.request_timeout_seconds)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parents[1] / 'examples' / 'features.json'
    data = json.loads(fixture.read_text(encoding='utf-8'))
    if args.command == 'export-features':
        from .features import export_features
        print(json.dumps(export_features(args.pairs, ChineseCLIPEncoder(device=args.device), args.out),
                         ensure_ascii=False, indent=2))
        return
    if args.command == 'train':
        from .training import train_features
        report = train_features(args.features, args.out, args.steps, args.learning_rate,
                               initial_temperature=args.temperature, learnable_temperature=not args.fixed_temperature,
                               max_logit_scale=args.max_logit_scale, batch_size=args.batch_size)
        print(json.dumps({k: v for k, v in report.items() if k not in ['losses', 'temperatures']}, ensure_ascii=False, indent=2))
        return
    items, rejected = load_pairs(Path(__file__).resolve().parents[1] / 'examples' / 'pairs.json')
    if args.command == 'evaluate':
        from .evaluate import evaluate, sweep
        if args.sweep:
            result = sweep(data['image_features'], items, data['text_features'], m_values=args.m_values,
                           ef_values=args.ef_values, ef_construction=args.ef_construction, repeats=args.repeats)
        else:
            result = evaluate(data['image_features'], items, data['text_features'], hnsw_m=args.hnsw_m,
                              ef_construction=args.ef_construction, ef_search=args.ef_search, repeats=args.repeats)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == 'demo':
        encoder = FixtureEncoder(fixture)
        index = VectorIndex(data['image_features'], items, backend=args.backend, kind=args.kind,
                            hnsw_m=args.hnsw_m, ef_construction=args.ef_construction, ef_search=args.ef_search)
        result = RetrievalPipeline(index).search_vector(encoder.encode(['医院设备采购'])[0])
        result['input_mode'], result['rejected_rows'] = 'ideal_upstream_features', rejected
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        index.save(out / 'index')
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    elif args.command == 'build':
        encoder = (FixtureEncoder(fixture) if args.encoder == 'fixture'
                   else ChineseCLIPEncoder(device=args.device))
        if args.projection:
            encoder = ProjectedEncoder(encoder, args.projection)
        items, rejected = load_pairs(args.pairs, check_images=args.encoder == 'chinese-clip')
        if rejected:
            raise ValueError('invalid dataset rows; inspect: ' + json.dumps(rejected, ensure_ascii=False))
        if args.encoder == 'fixture' and [item['id'] for item in items] != data['ids']:
            raise ValueError('fixture feature IDs and pair IDs must align in the same order')
        vectors = (np.asarray(data['image_features']) if args.encoder == 'fixture' and not args.projection
                   else encoder.encode([x['image_path'] for x in items], 'image'))
        if args.encoder == 'fixture' and Path(args.pairs).resolve() != (fixture.parent / 'pairs.json').resolve():
            raise ValueError('fixture features only align with bundled pairs.json')
        index = VectorIndex(vectors, items, backend=args.backend, kind=args.kind, encoder=encoder.fingerprint,
                            hnsw_m=args.hnsw_m, ef_construction=args.ef_construction, ef_search=args.ef_search)
        index.save(args.out)
        result = {'items': len(items), 'dimension': vectors.shape[1], 'index': args.out,
                  'encoder': encoder.fingerprint, 'index_version': index.version, 'rejected': rejected}
    else:
        encoder = FixtureEncoder(fixture) if args.encoder == 'fixture' else ChineseCLIPEncoder()
        if args.projection:
            encoder = ProjectedEncoder(encoder, args.projection)
        index = VectorIndex.load(args.index, expected_encoder=None if getattr(args, 'vector', None) else encoder.fingerprint)
        if args.command == 'serve':
            import uvicorn
            from .api import create_app
            logging.basicConfig(level=logging.INFO, format='%(message)s')
            config = ServiceConfig(queue_capacity=args.queue_capacity, max_batch=args.max_batch,
                                   max_delay_ms=args.max_delay_ms, cache_size=args.cache_size,
                                   cache_ttl_seconds=args.cache_ttl_seconds,
                                   request_timeout_seconds=args.request_timeout_seconds)
            uvicorn.run(create_app(index, encoder, config), host='127.0.0.1', port=args.port)
            return
        query = json.loads(args.vector) if args.vector else encoder.encode([args.query])[0]
        result = RetrievalPipeline(index).search_vector(query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
