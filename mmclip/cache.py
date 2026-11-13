"""Per-process bounded LRU with monotonic expiry; no cached retrieval results."""
from collections import OrderedDict
import hashlib
import math
import time


def query_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class TTLCache:
    def __init__(self, capacity, ttl_seconds, clock=time.monotonic):
        if capacity < 1 or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError('positive cache capacity and TTL required')
        self.capacity, self.ttl_seconds, self.clock = capacity, ttl_seconds, clock
        self.entries = OrderedDict()

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            return None
        deadline, payload = value
        if self.clock() >= deadline:
            del self.entries[key]
            return None
        self.entries.move_to_end(key)
        return payload.copy()

    def put(self, key, value):
        self.entries[key] = (self.clock() + self.ttl_seconds, value.copy())
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
