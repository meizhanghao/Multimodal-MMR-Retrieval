"""Inspectable starting settings; they are not claimed benchmark optima."""
from dataclasses import dataclass
import math

HNSW_M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 64
INITIAL_TEMPERATURE = 0.07
MAX_LOGIT_SCALE = 100.0


@dataclass(frozen=True)
class ServiceConfig:
    queue_capacity: int = 128
    max_batch: int = 16
    max_delay_ms: float = 5.0
    cache_size: int = 128
    cache_ttl_seconds: float = 3600.0
    request_timeout_seconds: float = 10.0

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.queue_capacity, self.max_batch, self.cache_size)):
            raise ValueError('queue, batch and cache sizes must be positive integers')
        for value, allow_zero in [(self.max_delay_ms, True), (self.cache_ttl_seconds, False),
                                   (self.request_timeout_seconds, False)]:
            if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                raise ValueError('invalid service timing configuration')
