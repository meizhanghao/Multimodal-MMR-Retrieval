"""Text retrieval contract for an upper RAG layer, with bounded request waiting."""
from contextlib import asynccontextmanager
from dataclasses import asdict
import asyncio
import json
import logging
import re
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .batching import BatchEncoder, QueueFullError
from .cache import query_hash
from .config import ServiceConfig
from .encoders import FixtureEncoder
from .pipeline import RetrievalPipeline

LOGGER = logging.getLogger('mmclip.events')


class EncodeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    query: str = Field(min_length=1, max_length=2048)

    @field_validator('query')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('query must contain non-whitespace characters')
        return value


class SearchRequest(EncodeRequest):
    top_k: int = Field(default=3, ge=1, le=50)
    candidate_k: int = Field(default=6, ge=1, le=100)
    diversity: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def budget(self):
        if self.candidate_k < self.top_k:
            raise ValueError('candidate_k must be >= top_k')
        return self


class RetrieveRequest(EncodeRequest):
    top_k: int = Field(default=3, ge=1, le=50)
    candidate_k: int = Field(default=6, ge=1, le=100)
    lambda_mult: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def budget(self):
        if self.candidate_k < self.top_k:
            raise ValueError('candidate_k must be >= top_k')
        return self


def create_app(index, encoder, config=None):
    if index.encoder != encoder.fingerprint:
        raise ValueError('query encoder and index must have the same version')
    config = config or ServiceConfig()
    batcher = BatchEncoder(encoder, capacity=config.queue_capacity, batch_size=config.max_batch,
        wait_ms=config.max_delay_ms, cache_size=config.cache_size, cache_ttl_seconds=config.cache_ttl_seconds)
    pipeline = RetrievalPipeline(index)

    @asynccontextmanager
    async def lifespan(app):
        await batcher.start()
        try:
            yield
        finally:
            await batcher.close()

    app = FastAPI(title='Chinese-CLIP FAISS MMR reproduction', lifespan=lifespan)
    app.state.batcher = batcher
    app.state.pipeline = pipeline

    @app.middleware('http')
    async def request_log(request: Request, call_next):
        supplied = request.headers.get('X-Request-ID', '')
        request.state.request_id = supplied if re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', supplied) else uuid.uuid4().hex
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers['X-Request-ID'] = request.state.request_id
            return response
        finally:
            info = getattr(request.state, 'encoding', None)
            LOGGER.info(json.dumps({'event': 'http_request', 'request_id': request.state.request_id,
                'path': request.url.path, 'query_hash': getattr(request.state, 'query_hash', None),
                'batch_size': info.batch_size if info else None, 'cache_hit': info.cache_hit if info else None,
                'queue_ms': round(info.queue_ms, 3) if info else None,
                'encode_ms': round(info.encode_ms, 3) if info else None,
                'latency_ms': round((time.perf_counter()-started)*1000, 3), 'status_code': status_code,
                'error_code': getattr(request.state, 'error_code', None) or (f'HTTP_{status_code}' if status_code >= 400 else None)}))

    @app.get('/health')
    def health():
        base = getattr(encoder, 'base', encoder)
        return {'mode': 'fixture' if isinstance(base, FixtureEncoder) else 'model',
                'encoder': encoder.fingerprint, 'index_version': index.version, 'backend': index.backend,
                'service_config': asdict(config),
                'hnsw_config': {'m': index.hnsw_m, 'ef_construction': index.ef_construction, 'ef_search': index.ef_search}}

    async def execute(body, request, *, search=False, contract=False):
        request.state.query_hash = query_hash(body.query)
        try:
            async with asyncio.timeout(config.request_timeout_seconds):
                encoded = await batcher.encode_with_info(body.query, timeout=config.request_timeout_seconds)
                request.state.encoding = encoded
                if not search:
                    return {'encoder': encoder.fingerprint, 'vector': encoded.vector.tolist(),
                            'request_id': request.state.request_id, 'cache_hit': encoded.cache_hit}
                result = await asyncio.to_thread(pipeline.search_vector, encoded.vector,
                    candidate_k=body.candidate_k, top_k=body.top_k,
                    diversity=body.lambda_mult if contract else body.diversity)
                if contract:
                    result = pipeline.as_retrieval_response(result)
                return {**result, 'request_id': request.state.request_id}
        except QueueFullError as exc:
            request.state.error_code = 'QUEUE_FULL'
            raise HTTPException(429, 'encoder queue full', headers={'Retry-After': '1'}) from exc
        except TimeoutError as exc:
            request.state.error_code = 'REQUEST_TIMEOUT'
            raise HTTPException(504, 'request waiting deadline exceeded') from exc
        except ValueError as exc:
            request.state.error_code = 'INVALID_INPUT'
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            request.state.error_code = type(exc).__name__
            raise HTTPException(500, 'internal retrieval error; see request_id in server log') from exc

    @app.post('/encode')
    async def encode(body: EncodeRequest, request: Request):
        return await execute(body, request)

    @app.post('/search')
    async def search(body: SearchRequest, request: Request):
        return await execute(body, request, search=True)

    @app.post('/v1/retrieve')
    async def retrieve(body: RetrieveRequest, request: Request):
        return await execute(body, request, search=True, contract=True)

    @app.get('/documents/{item_id}')
    def document(item_id: str):
        for item in index.items:
            if item['id'] == item_id:
                return item
        raise HTTPException(404, 'unknown document ID')

    return app
