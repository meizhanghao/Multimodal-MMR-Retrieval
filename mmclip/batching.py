"""Bounded per-process queue, dynamic batches and versioned encoding TTL cache."""
import asyncio
from dataclasses import dataclass
import json
import logging
import math
import time
import numpy as np
from .cache import TTLCache, query_hash
from .config import ServiceConfig

LOGGER = logging.getLogger('mmclip.events')
DEFAULTS = ServiceConfig()


class QueueFullError(RuntimeError):
    pass


@dataclass
class EncodingResult:
    vector: np.ndarray
    batch_size: int
    cache_hit: bool
    queue_ms: float = 0.0
    encode_ms: float = 0.0


class BatchEncoder:
    def __init__(self, encoder, *, capacity=DEFAULTS.queue_capacity, batch_size=DEFAULTS.max_batch,
                 wait_ms=DEFAULTS.max_delay_ms, cache_size=DEFAULTS.cache_size,
                 cache_ttl_seconds=DEFAULTS.cache_ttl_seconds):
        if min(capacity, batch_size, cache_size) < 1 or not math.isfinite(wait_ms) or wait_ms < 0:
            raise ValueError('invalid queue/cache/batch settings')
        self.encoder, self.batch_size, self.wait_ms = encoder, batch_size, wait_ms
        self.queue = asyncio.Queue(maxsize=capacity)
        self.cache = TTLCache(cache_size, cache_ttl_seconds)
        self.worker = None
        self.closed = False

    async def start(self):
        if self.closed:
            raise RuntimeError('batcher closed')
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())

    async def encode(self, text, timeout=DEFAULTS.request_timeout_seconds):
        return (await self.encode_with_info(text, timeout)).vector

    async def encode_with_info(self, text, timeout=DEFAULTS.request_timeout_seconds):
        if self.closed:
            raise RuntimeError('batcher closed')
        if not isinstance(text, str) or not text.strip() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('nonempty text and a positive finite timeout required')
        await self.start()
        key = (self.encoder.fingerprint, query_hash(text))
        cached = self.cache.get(key)
        if cached is not None:
            return EncodingResult(cached, batch_size=0, cache_hit=True)
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((key, text, future, time.perf_counter()))
        except asyncio.QueueFull as exc:
            raise QueueFullError('encoder queue full') from exc
        return await asyncio.wait_for(future, timeout=timeout)

    async def _run(self):
        while True:
            first = await self.queue.get()
            if first is None:
                self.queue.task_done()
                return
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.wait_ms / 1000
            while len(batch) < self.batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    self.queue.task_done()
                    raise RuntimeError('unexpected shutdown sentinel')
                batch.append(item)
            live = [item for item in batch if not item[2].done()]
            started, error_code = time.perf_counter(), None
            try:
                if live:
                    vectors = np.asarray(await asyncio.to_thread(self.encoder.encode, [item[1] for item in live]),
                                         dtype=np.float32)
                    if vectors.ndim != 2 or len(vectors) != len(live) or not np.isfinite(vectors).all():
                        raise ValueError('encoder returned an invalid or misaligned batch')
                    encode_ms = (time.perf_counter() - started) * 1000
                    for (key, _, future, enqueued), vector in zip(live, vectors):
                        self.cache.put(key, vector)
                        if not future.done():
                            future.set_result(EncodingResult(vector.copy(), len(live), False,
                                queue_ms=(started-enqueued)*1000, encode_ms=encode_ms))
            except Exception as exc:
                error_code = type(exc).__name__
                for _, _, future, _ in live:
                    if not future.done():
                        future.set_exception(exc)
            finally:
                if live:
                    LOGGER.info(json.dumps({'event': 'encoding_batch', 'batch_size': len(live),
                        'latency_ms': round((time.perf_counter()-started)*1000, 3), 'error_code': error_code}))
                for _ in batch:
                    self.queue.task_done()

    async def close(self):
        self.closed = True
        if self.worker is not None:
            await self.queue.join()
            await self.queue.put(None)
            await self.worker
