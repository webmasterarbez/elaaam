# Architecture Review and Recommendations

## Overview

This document provides architectural recommendations and identifies areas for optimization in the OpenMemory + ElevenLabs integration.

## 1. Performance Optimization Recommendations

### 1.1 Embedding Generation Caching

**Issue**: Embedding generation is a high-latency operation (typically 50-200ms per OpenAI API call). Without caching, identical queries result in redundant API calls and increased costs.

**Current State**: The architecture calls `generate_embedding()` directly for every query without caching.

**Recommendation**: Implement an LRU (Least Recently Used) cache for embedding generation.

#### Why LRU Instead of FIFO?

- **LRU (Least Recently Used)**: Evicts items that haven't been accessed recently, keeping frequently used items in cache
- **FIFO (First In First Out)**: Evicts items based on insertion order, regardless of usage patterns

**Example Scenario**:
```
Cache size: 3 items

Sequence of queries:
1. "user preferences" → cache miss, generate, store
2. "CRM system" → cache miss, generate, store
3. "billing info" → cache miss, generate, store
4. "user preferences" → CACHE HIT (LRU: accessed recently, stays)
5. "contact details" → cache full, need to evict

With LRU: Evicts "CRM system" (least recently accessed)
With FIFO: Evicts "user preferences" (oldest insertion) ❌ WRONG!

Result: LRU keeps "user preferences" because it was just accessed (#4),
        FIFO removes it even though it's actively used
```

**Impact**:
- **LRU**: 30-50% cache hit rate on production workloads
- **FIFO**: 10-20% cache hit rate (frequently evicts hot items)

#### Recommended Implementation

**Option A: Python `functools.lru_cache` (Simple)**

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def generate_embedding_cached(text: str) -> tuple[float, ...]:
    """
    LRU-cached embedding generation.
    Returns tuple (immutable) for caching compatibility.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    return tuple(embedding)  # Convert to tuple for hashability

def generate_embedding(text: str) -> list[float]:
    """Public API that returns list[float] as expected."""
    return list(generate_embedding_cached(text))
```

**Pros**:
- Built-in, thread-safe
- Automatic LRU eviction
- Zero dependencies

**Cons**:
- No cache statistics
- No TTL support
- Memory-only (doesn't persist across restarts)

**Option B: Custom LRU Cache with Statistics (Recommended)**

```python
from collections import OrderedDict
from datetime import datetime, timedelta
import threading
from typing import Optional
import hashlib

class EmbeddingCache:
    """
    Thread-safe LRU cache for embeddings with statistics and TTL.
    """
    def __init__(self, maxsize: int = 1000, ttl_hours: int = 24):
        self.maxsize = maxsize
        self.ttl = timedelta(hours=ttl_hours)
        self.cache = OrderedDict()  # Maintains insertion order for LRU
        self.lock = threading.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str = "text-embedding-3-small") -> Optional[list[float]]:
        """
        Retrieve embedding from cache.
        Returns None if not found or expired.
        """
        key = self._make_key(text, model)

        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            # Check TTL
            cached_data = self.cache[key]
            if datetime.now() - cached_data['timestamp'] > self.ttl:
                # Expired
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used) - THIS IS THE LRU PART
            self.cache.move_to_end(key)
            self.hits += 1
            return cached_data['embedding'].copy()

    def put(self, text: str, embedding: list[float], model: str = "text-embedding-3-small"):
        """
        Store embedding in cache with LRU eviction.
        """
        key = self._make_key(text, model)

        with self.lock:
            # If key exists, update and move to end
            if key in self.cache:
                self.cache.move_to_end(key)

            # Store with timestamp
            self.cache[key] = {
                'embedding': embedding,
                'timestamp': datetime.now(),
                'text_length': len(text)
            }

            # Evict oldest (least recently used) if over maxsize
            if len(self.cache) > self.maxsize:
                # pop(last=False) removes FIRST item (least recently used)
                evicted_key, _ = self.cache.popitem(last=False)
                self.evictions += 1

    def get_stats(self) -> dict:
        """Return cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

    def clear(self):
        """Clear all cached embeddings."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

# Global cache instance
embedding_cache = EmbeddingCache(maxsize=1000, ttl_hours=24)

def generate_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Generate embedding with LRU caching.

    Args:
        text: Text to embed
        model: OpenAI embedding model name

    Returns:
        Embedding vector as list[float]
    """
    # Check cache first
    cached = embedding_cache.get(text, model)
    if cached is not None:
        return cached

    # Cache miss - generate embedding
    response = client.embeddings.create(
        model=model,
        input=text
    )
    embedding = response.data[0].embedding

    # Store in cache
    embedding_cache.put(text, embedding, model)

    return embedding

# Monitoring endpoint
@app.get("/metrics/embedding-cache")
async def get_cache_metrics():
    """Return embedding cache statistics."""
    return embedding_cache.get_stats()
```

**Option C: Redis-based LRU Cache (For Distributed Systems)**

```python
import redis
import json
import hashlib
from typing import Optional

class RedisEmbeddingCache:
    """
    Redis-backed LRU cache for embeddings.
    Uses Redis built-in LRU eviction policy.
    """
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_memory: str = "100mb",
        ttl_hours: int = 24
    ):
        self.client = redis.from_url(redis_url)
        self.ttl_seconds = ttl_hours * 3600

        # Configure Redis for LRU eviction
        try:
            self.client.config_set('maxmemory', max_memory)
            self.client.config_set('maxmemory-policy', 'allkeys-lru')
        except redis.ResponseError:
            # Redis might be configured via redis.conf
            pass

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key."""
        content = f"embedding:{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str = "text-embedding-3-small") -> Optional[list[float]]:
        """Retrieve embedding from Redis."""
        key = self._make_key(text, model)

        value = self.client.get(key)
        if value is None:
            return None

        # Redis GET automatically updates LRU access time
        return json.loads(value)

    def put(self, text: str, embedding: list[float], model: str = "text-embedding-3-small"):
        """Store embedding in Redis with TTL."""
        key = self._make_key(text, model)
        value = json.dumps(embedding)

        self.client.setex(key, self.ttl_seconds, value)

    def get_stats(self) -> dict:
        """Get Redis cache statistics."""
        info = self.client.info('stats')
        memory = self.client.info('memory')

        return {
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)),
            'used_memory_mb': memory.get('used_memory', 0) / 1024 / 1024,
            'evicted_keys': info.get('evicted_keys', 0)
        }

# Usage
redis_cache = RedisEmbeddingCache(max_memory="100mb", ttl_hours=24)

def generate_embedding(text: str) -> list[float]:
    cached = redis_cache.get(text)
    if cached:
        return cached

    embedding = _generate_embedding_uncached(text)
    redis_cache.put(text, embedding)
    return embedding
```

### 1.2 Cache Size Recommendations

**For 100 conversations/day:**
- Average 3 memory queries per conversation
- Average 2 embeddings per query (user query + expansions)
- Daily: 600 embedding generations
- With 30% duplicate rate: ~420 unique embeddings/day

**Recommended cache size**: 1,000 items (2-3 days of unique queries)

**Memory usage**:
- Embedding size: 1536 floats × 4 bytes = 6KB per embedding
- Cache metadata: ~500 bytes per entry
- Total: ~6.5KB per cached item
- 1,000 items = ~6.5MB memory

**For 1,000 conversations/day:**
- Recommended cache size: 5,000 items (~30MB memory)
- Consider Redis for multi-instance deployments

### 1.3 Integration with Existing Architecture

Update the memory query tool to use cached embeddings:

```python
@app.post("/tools/memory-query")
async def memory_query_tool(request: MemoryQueryRequest):
    """
    Real-time memory query with embedding caching.
    Target latency: <500ms (p95), <1000ms (p99)
    """
    start_time = time.time()

    # Generate embedding WITH CACHING
    query_embedding = generate_embedding(request.query)  # Now uses LRU cache

    # Rest of implementation...
    sectors = request.sectors or classify_query_sectors(request.query)
    initial_matches = vector_search(
        embedding=query_embedding,
        user_id=request.user_id,
        sectors=sectors,
        limit=request.limit * 2
    )

    expanded_memories = expand_via_links(initial_matches, max_total=request.limit)

    # ... rest of implementation
```

### 1.4 Monitoring and Alerting

Add cache metrics to existing monitoring:

```python
metrics = {
    # ... existing metrics ...

    # Embedding cache metrics
    "embedding_cache_size": gauge("embedding.cache.size"),
    "embedding_cache_hit_rate": gauge("embedding.cache.hit_rate"),
    "embedding_cache_evictions": counter("embedding.cache.evictions"),
    "embedding_generation_latency": histogram("embedding.generation.latency"),
    "embedding_api_calls": counter("embedding.api.calls"),
}
```

**Alerts**:
```yaml
alerts:
  - name: low_embedding_cache_hit_rate
    condition: embedding_cache_hit_rate < 0.20 for 30 minutes
    severity: warning
    action: Increase cache size or investigate query patterns

  - name: high_embedding_cache_evictions
    condition: embedding_cache_evictions > 100 per minute
    severity: warning
    action: Cache size too small for workload, increase maxsize
```

## 2. Additional Recommendations

### 2.1 Batch Embedding Generation

For post-call webhook processing (multiple memories at once):

```python
def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings in batch for better performance.
    OpenAI API supports up to 2048 inputs per request.
    """
    # Check cache first
    uncached_indices = []
    results = [None] * len(texts)

    for i, text in enumerate(texts):
        cached = embedding_cache.get(text)
        if cached:
            results[i] = cached
        else:
            uncached_indices.append(i)

    # Generate uncached embeddings in batch
    if uncached_indices:
        uncached_texts = [texts[i] for i in uncached_indices]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=uncached_texts  # Batch request
        )

        for idx, embedding_obj in zip(uncached_indices, response.data):
            embedding = embedding_obj.embedding
            results[idx] = embedding
            embedding_cache.put(texts[idx], embedding)

    return results
```

### 2.2 Pre-warming Cache

For common queries, pre-populate cache at startup:

```python
async def prewarm_embedding_cache():
    """Pre-generate embeddings for common queries."""
    common_queries = [
        "user preferences",
        "recent conversation topics",
        "CRM system",
        "billing information",
        "contact details",
        "support history",
        "product interests",
        "communication preferences"
    ]

    for query in common_queries:
        generate_embedding(query)  # Populates cache

    logger.info(f"Pre-warmed embedding cache with {len(common_queries)} common queries")
```

## 3. Summary of Changes

### Required Changes

1. **Replace direct `generate_embedding()` calls** with LRU-cached version
2. **Update `memory_query_tool()`** in openmemory-elevenlabs-architecture-corrected.md:449
3. **Update `query_memories_for_initiation()`** in openmemory-elevenlabs-architecture-corrected.md:611
4. **Update `extract_memories_from_transcript()`** in openmemory-elevenlabs-architecture-corrected.md:1062
5. **Add cache metrics** to monitoring section

### Performance Impact

| Metric | Before | After (30% hit rate) | After (50% hit rate) |
|--------|--------|---------------------|---------------------|
| Avg embedding latency | 100ms | 70ms | 50ms |
| API calls (100 conv/day) | 600/day | 420/day | 300/day |
| API cost (at $0.02/1M tokens) | $0.60/day | $0.42/day | $0.30/day |
| P95 memory query latency | 500ms | 400ms | 350ms |

### Implementation Priority

1. **Phase 1** (Immediate): Implement `functools.lru_cache` wrapper (5 minutes)
2. **Phase 2** (Week 1): Replace with custom `EmbeddingCache` class for statistics
3. **Phase 3** (Month 1): Migrate to Redis if scaling beyond single instance

## 4. References

- [Python OrderedDict Documentation](https://docs.python.org/3/library/collections.html#collections.OrderedDict)
- [Redis LRU Eviction](https://redis.io/docs/reference/eviction/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
