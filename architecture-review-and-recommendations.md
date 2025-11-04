# Architecture Review and Recommendations
## OpenMemory + ElevenLabs Integration

**Review Date**: 2025-11-04
**Reviewer**: Claude (Architecture Analysis)
**Target Scale**: 100-1000 conversations/day

---

## Executive Summary

The proposed architecture for integrating OpenMemory's cognitive memory system with ElevenLabs Agents is **fundamentally sound** with well-documented integration points and appropriate technology choices for the target scale. However, there are several areas that require enhancement to ensure production readiness, scalability, and operational excellence.

**Overall Assessment**: ⚠️ **Good Foundation, Needs Hardening**

**Key Findings**:
- ✅ Well-designed three-phase integration model
- ✅ Appropriate performance targets defined
- ⚠️ Database layer needs optimization for vector operations
- ⚠️ Missing critical production concerns (caching, rate limiting, observability)
- ⚠️ Security needs more detail
- ⚠️ No deployment/operational strategy

---

## Architecture Strengths

### 1. Clear Integration Model ✅
The three-phase approach (initiation → real-time → post-call) properly separates concerns and aligns with ElevenLabs' capabilities:
- Pre-conversation context loading via initiation webhooks
- Real-time memory queries via server tools
- Async memory extraction via post-call webhooks

### 2. Appropriate Technology Choices ✅
- SQLite with WAL mode for 100 conversations/day is pragmatic
- FastAPI for async HTTP handling
- Background worker pool for async processing
- Idempotent webhook handling

### 3. Comprehensive Documentation ✅
- Detailed ERD, sequence diagrams, and system architecture
- Clear performance targets (p95, p99 latencies)
- Implementation code examples provided
- Testing checklist included

### 4. Memory System Design ✅
- Five sector types (semantic, episodic, procedural, emotional, reflective)
- Graph-based memory linking
- Composite scoring for relevance (similarity + salience + recency + links)
- Memory decay model

---

## Critical Issues and Recommendations

### 1. Database Architecture 🔴 **HIGH PRIORITY**

#### Issues:
- **Vector operations in SQLite**: SQLite lacks native vector similarity operations, requiring Python-side computation
- **Scalability ceiling**: Even with WAL mode, SQLite will struggle beyond 1000 conversations/day
- **No vector indexing**: Linear scan for similarity search will become prohibitively slow
- **Concurrent write bottleneck**: WAL helps but doesn't eliminate write serialization

#### Recommendations:

**Short-term** (for 100 conversations/day):
```sql
-- 1. Add missing indexes for performance
CREATE INDEX idx_memories_user_sector_state ON memories(user_id, sector_type, state)
  WHERE state = 'active';
CREATE INDEX idx_memories_recency ON memories(recency DESC);
CREATE INDEX idx_webhook_status_created ON webhook_events(status, created_at)
  WHERE status IN ('pending', 'processing');
CREATE INDEX idx_conversation_user ON conversations(user_id, start_time DESC);

-- 2. Implement ANALYZE for query optimization
PRAGMA optimize;
ANALYZE;

-- 3. Add partitioning for old data
CREATE TABLE memories_archived AS SELECT * FROM memories WHERE state = 'archived';
DELETE FROM memories WHERE state = 'archived';
```

**Medium-term** (for 1000+ conversations/day):
```python
# Migrate to PostgreSQL with pgvector extension
from pgvector.sqlalchemy import Vector

class MemoryEmbedding(Base):
    __tablename__ = 'memory_embeddings'

    id = Column(Integer, primary_key=True)
    memory_id = Column(Integer, ForeignKey('memories.id'))
    embedding = Column(Vector(1536))  # Native vector type

    # Create IVFFlat or HNSW index for fast similarity search
    __table_args__ = (
        Index('idx_embedding_ivfflat', 'embedding',
              postgresql_using='ivfflat',
              postgresql_with={'lists': 100},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )
```

**Benefits of PostgreSQL + pgvector**:
- 10-100x faster vector similarity search
- Built-in parallel query execution
- Better connection pooling
- Native JSON operations
- Point-in-time recovery

---

### 2. Performance and Caching 🟡 **MEDIUM PRIORITY**

#### Issues:
- No caching layer for frequently accessed memories
- Repeated embedding generation for same queries
- No connection pooling mentioned
- Memory query does full vector search every time

#### Recommendations:

**Implement Redis caching layer**:
```python
from redis import asyncio as aioredis
import hashlib

class MemoryQueryCache:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.ttl = 300  # 5 minutes

    async def get_or_compute(self, user_id: str, query: str, compute_fn):
        # Cache key based on user + query hash
        cache_key = f"memory:query:{user_id}:{hashlib.sha256(query.encode()).hexdigest()[:16]}"

        # Try cache first
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Compute and cache
        result = await compute_fn()
        await self.redis.setex(cache_key, self.ttl, json.dumps(result))
        return result

# Usage in memory query tool
@app.post("/tools/memory-query")
async def memory_query_tool(request: MemoryQueryRequest):
    cache = MemoryQueryCache(redis_client)

    async def compute():
        # Existing query logic
        return perform_memory_search(request)

    return await cache.get_or_compute(request.user_id, request.query, compute)
```

**Implement embedding cache**:
```python
# Cache embeddings for common queries
embedding_cache = {}

async def generate_embedding_cached(text: str) -> List[float]:
    text_hash = hashlib.sha256(text.encode()).hexdigest()

    if text_hash in embedding_cache:
        return embedding_cache[text_hash]

    embedding = await generate_embedding(text)
    embedding_cache[text_hash] = embedding

    # Limit cache size (LRU)
    if len(embedding_cache) > 10000:
        embedding_cache.pop(next(iter(embedding_cache)))

    return embedding
```

**Add connection pooling**:
```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'sqlite:///memories.db',
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600    # Recycle connections every hour
)
```

**Expected improvements**:
- 50-80% reduction in p95 latency for repeated queries
- 3-5x reduction in embedding API costs
- Better database connection utilization

---

### 3. Security Hardening 🔴 **HIGH PRIORITY**

#### Issues:
- Bearer token authentication mentioned but not implemented
- No rate limiting on API endpoints
- No encryption at rest for sensitive conversation data
- HMAC verification code vulnerable to timing attacks
- No API key rotation strategy

#### Recommendations:

**Implement proper authentication**:
```python
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets

security = HTTPBearer()

class APIKeyManager:
    def __init__(self):
        self.valid_keys = set(os.getenv('API_KEYS', '').split(','))

    async def verify_api_key(
        self,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ) -> str:
        if not secrets.compare_digest(credentials.credentials,
                                      self.valid_keys):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return credentials.credentials

api_key_manager = APIKeyManager()

@app.post("/tools/memory-query")
async def memory_query_tool(
    request: MemoryQueryRequest,
    api_key: str = Depends(api_key_manager.verify_api_key)
):
    # Existing logic
    pass
```

**Add rate limiting**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/tools/memory-query")
@limiter.limit("100/minute")  # Per IP
async def memory_query_tool(request: Request, ...):
    pass

@app.post("/webhook/conversation-init")
@limiter.limit("200/minute")
async def conversation_initiation(request: Request, ...):
    pass
```

**Fix HMAC verification timing attack**:
```python
def verify_elevenlabs_signature(signature_header: str, body: bytes, secret: str) -> bool:
    try:
        parts = dict(item.split('=') for item in signature_header.split(','))
        timestamp = parts['t']
        received_sig = parts['v1']

        # Check timestamp freshness (5 min window)
        if abs(time.time() - int(timestamp)) > 300:
            logger.warning("Signature timestamp outside acceptable window")
            return False

        # Compute expected signature
        payload = f"{timestamp}.{body.decode('utf-8')}"
        expected_sig = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        # FIXED: Use constant-time comparison
        return secrets.compare_digest(expected_sig, received_sig)
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        return False
```

**Add encryption at rest for sensitive data**:
```python
from cryptography.fernet import Fernet

class DataEncryption:
    def __init__(self):
        self.cipher = Fernet(os.getenv('ENCRYPTION_KEY').encode())

    def encrypt_conversation(self, content: str) -> bytes:
        return self.cipher.encrypt(content.encode())

    def decrypt_conversation(self, encrypted: bytes) -> str:
        return self.cipher.decrypt(encrypted).decode()

# Store encrypted in database
db.execute("""
    INSERT INTO conversation_turns (conversation_fk, role, message, ...)
    VALUES (?, ?, ?, ...)
""", (conversation_id, turn['role'],
      encryptor.encrypt_conversation(turn['message']), ...))
```

---

### 4. Observability and Monitoring 🟡 **MEDIUM PRIORITY**

#### Issues:
- Metrics defined but no implementation
- No distributed tracing
- No structured logging
- No centralized log aggregation
- Alert definitions without tooling

#### Recommendations:

**Implement structured logging**:
```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Usage with context
@app.post("/tools/memory-query")
async def memory_query_tool(request: MemoryQueryRequest):
    log = logger.bind(
        user_id=request.user_id,
        query=request.query[:50],  # Truncate for privacy
        endpoint="memory_query"
    )

    start_time = time.time()
    try:
        result = await perform_query(request)
        latency = (time.time() - start_time) * 1000

        log.info("memory_query_success",
                 latency_ms=latency,
                 result_count=len(result['memories']))

        return result
    except Exception as e:
        log.error("memory_query_error", error=str(e), exc_info=True)
        raise
```

**Add OpenTelemetry tracing**:
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add OTLP exporter (to Jaeger, Tempo, etc.)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Manual tracing for critical paths
@app.post("/tools/memory-query")
async def memory_query_tool(request: MemoryQueryRequest):
    with tracer.start_as_current_span("memory_query") as span:
        span.set_attribute("user_id", request.user_id)
        span.set_attribute("query_length", len(request.query))

        with tracer.start_as_current_span("generate_embedding"):
            embedding = await generate_embedding(request.query)

        with tracer.start_as_current_span("vector_search"):
            results = await vector_search(embedding, request.user_id)

        with tracer.start_as_current_span("graph_expansion"):
            expanded = await expand_via_links(results)

        return expanded
```

**Implement Prometheus metrics**:
```python
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

# Define metrics
memory_query_latency = Histogram(
    'memory_query_latency_seconds',
    'Memory query latency',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

memory_query_total = Counter(
    'memory_query_total',
    'Total memory queries',
    ['status']  # success/error
)

webhook_queue_depth = Gauge(
    'webhook_queue_depth',
    'Number of pending webhook events'
)

# Use in endpoints
@app.post("/tools/memory-query")
@memory_query_latency.time()
async def memory_query_tool(request: MemoryQueryRequest):
    try:
        result = await perform_query(request)
        memory_query_total.labels(status='success').inc()
        return result
    except Exception as e:
        memory_query_total.labels(status='error').inc()
        raise

# Background task to update queue depth
async def update_queue_metrics():
    while True:
        depth = db.execute(
            "SELECT COUNT(*) FROM webhook_events WHERE status='pending'"
        ).fetchone()[0]
        webhook_queue_depth.set(depth)
        await asyncio.sleep(10)

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**Setup alerting with AlertManager**:
```yaml
# alertmanager.yml
groups:
  - name: memory_system_alerts
    rules:
      - alert: HighMemoryQueryLatency
        expr: histogram_quantile(0.95, memory_query_latency_seconds) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Memory query p95 latency is high"
          description: "p95 latency is {{ $value }}s (threshold: 0.5s)"

      - alert: WebhookQueueBacklog
        expr: webhook_queue_depth > 50
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Webhook processing queue is backing up"
          description: "Queue depth: {{ $value }} events"

      - alert: HighErrorRate
        expr: rate(memory_query_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on memory queries"
```

---

### 5. Error Handling and Resilience 🟡 **MEDIUM PRIORITY**

#### Issues:
- No circuit breaker for external services (LLM, embedding API)
- Basic retry logic without jitter
- No fallback strategies
- No graceful degradation

#### Recommendations:

**Implement circuit breaker pattern**:
```python
from circuitbreaker import circuit
import backoff

class ExternalServiceError(Exception):
    pass

# Circuit breaker for LLM API
@circuit(failure_threshold=5, recovery_timeout=60)
@backoff.on_exception(
    backoff.expo,
    (ExternalServiceError, TimeoutError),
    max_tries=3,
    jitter=backoff.full_jitter
)
async def call_llm_api(prompt: str, **kwargs):
    try:
        response = await llm_client.generate(prompt, **kwargs)
        if response.status_code != 200:
            raise ExternalServiceError(f"LLM API error: {response.status_code}")
        return response.json()
    except httpx.TimeoutException as e:
        logger.error("LLM API timeout", exc_info=True)
        raise TimeoutError("LLM API timed out") from e

# Circuit breaker for embedding API
@circuit(failure_threshold=5, recovery_timeout=60)
@backoff.on_exception(
    backoff.expo,
    (ExternalServiceError, TimeoutError),
    max_tries=3,
    jitter=backoff.full_jitter
)
async def generate_embedding(text: str) -> List[float]:
    try:
        response = await embedding_client.create(input=text)
        return response['data'][0]['embedding']
    except Exception as e:
        logger.error("Embedding API error", exc_info=True)
        raise ExternalServiceError("Embedding generation failed") from e
```

**Implement fallback strategies**:
```python
@app.post("/tools/memory-query")
async def memory_query_tool(request: MemoryQueryRequest):
    try:
        # Primary path: full vector search
        return await full_memory_query(request)
    except ExternalServiceError:
        # Fallback 1: Use cached embeddings, keyword search
        logger.warning("Falling back to keyword search due to embedding API failure")
        return await keyword_memory_search(request)
    except Exception as e:
        # Fallback 2: Return empty results with error indication
        logger.error("All memory query methods failed", exc_info=True)
        return {
            "memories": [],
            "summary": "Unable to retrieve memories at this time",
            "error": "service_degraded"
        }

async def keyword_memory_search(request: MemoryQueryRequest):
    """Fallback search using SQL LIKE and FTS"""
    # Use SQLite FTS5 for keyword search
    results = db.execute("""
        SELECT m.* FROM memories_fts
        JOIN memories m ON memories_fts.rowid = m.id
        WHERE memories_fts MATCH ?
          AND m.user_id = ?
          AND m.state = 'active'
        ORDER BY rank
        LIMIT ?
    """, (request.query, request.user_id, request.limit)).fetchall()

    return format_memory_results(results, "keyword_search")
```

**Add health checks**:
```python
from fastapi import status

class HealthCheck(BaseModel):
    status: str
    database: str
    llm_api: str
    embedding_api: str
    redis: str

@app.get("/health", response_model=HealthCheck)
async def health_check():
    health = {
        "status": "healthy",
        "database": "unknown",
        "llm_api": "unknown",
        "embedding_api": "unknown",
        "redis": "unknown"
    }

    # Check database
    try:
        db.execute("SELECT 1").fetchone()
        health["database"] = "healthy"
    except Exception as e:
        health["database"] = "unhealthy"
        health["status"] = "degraded"

    # Check LLM API
    try:
        await call_llm_api("test", max_tokens=1, timeout=2)
        health["llm_api"] = "healthy"
    except Exception:
        health["llm_api"] = "unhealthy"
        health["status"] = "degraded"

    # Check embedding API
    try:
        await generate_embedding("test")
        health["embedding_api"] = "healthy"
    except Exception:
        health["embedding_api"] = "unhealthy"
        health["status"] = "degraded"

    # Check Redis
    try:
        await redis_client.ping()
        health["redis"] = "healthy"
    except Exception:
        health["redis"] = "unhealthy"
        # Redis is optional, don't degrade status

    status_code = (
        status.HTTP_200_OK if health["status"] == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(content=health, status_code=status_code)

@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe"""
    try:
        db.execute("SELECT 1").fetchone()
        return {"status": "ready"}
    except Exception:
        return JSONResponse(
            content={"status": "not_ready"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
```

---

### 6. Memory System Enhancements 🟢 **LOW PRIORITY**

#### Issues:
- No memory pruning/archival strategy for old memories
- No conflict resolution when memories contradict
- Memory decay mentioned but not implemented
- No memory consolidation/summarization

#### Recommendations:

**Implement memory decay and pruning**:
```python
from datetime import datetime, timedelta

class MemoryDecayManager:
    def __init__(self, db):
        self.db = db

    async def apply_decay(self):
        """Apply time-based decay to memory salience"""
        # Decay formula: new_salience = salience * e^(-decay_rate * days_old)
        self.db.execute("""
            UPDATE memories
            SET salience = salience * EXP(
                -decay_rate *
                (julianday('now') - julianday(recency))
            )
            WHERE state = 'active'
              AND decay_rate > 0
        """)
        self.db.commit()

    async def archive_old_memories(self, threshold_days: int = 180):
        """Archive memories older than threshold with low salience"""
        cutoff_date = datetime.now() - timedelta(days=threshold_days)

        archived = self.db.execute("""
            UPDATE memories
            SET state = 'archived'
            WHERE state = 'active'
              AND recency < ?
              AND salience < 0.3
            RETURNING id
        """, (cutoff_date,)).fetchall()

        logger.info(f"Archived {len(archived)} old memories")
        self.db.commit()

        return len(archived)

    async def consolidate_similar_memories(self, user_id: str, similarity_threshold: float = 0.95):
        """Merge very similar memories to reduce redundancy"""
        # Find highly similar memory pairs
        similar_pairs = self.db.execute("""
            SELECT
                m1.id as id1,
                m2.id as id2,
                m1.content as content1,
                m2.content as content2,
                m1.salience + m2.salience as combined_salience
            FROM memory_links ml
            JOIN memories m1 ON ml.source_memory_id = m1.id
            JOIN memories m2 ON ml.target_memory_id = m2.id
            WHERE ml.link_weight > ?
              AND m1.user_id = ?
              AND m1.state = 'active'
              AND m2.state = 'active'
              AND m1.sector_type = m2.sector_type
        """, (similarity_threshold, user_id)).fetchall()

        for pair in similar_pairs:
            # Use LLM to merge memories
            merged_content = await self.merge_memories_llm(
                pair['content1'],
                pair['content2']
            )

            # Update first memory with merged content
            self.db.execute("""
                UPDATE memories
                SET content = ?,
                    salience = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (merged_content, pair['combined_salience'], pair['id1']))

            # Archive second memory
            self.db.execute("""
                UPDATE memories
                SET state = 'archived'
                WHERE id = ?
            """, (pair['id2'],))

        self.db.commit()

# Run as background task
async def memory_maintenance_task():
    decay_manager = MemoryDecayManager(db)

    while True:
        try:
            # Run every 6 hours
            await decay_manager.apply_decay()
            await decay_manager.archive_old_memories()

            # Run consolidation for active users
            active_users = db.execute("""
                SELECT DISTINCT user_id FROM conversations
                WHERE start_time > datetime('now', '-7 days')
            """).fetchall()

            for user in active_users:
                await decay_manager.consolidate_similar_memories(user['user_id'])

            logger.info("Memory maintenance completed")
        except Exception as e:
            logger.error("Memory maintenance failed", exc_info=True)

        await asyncio.sleep(6 * 3600)  # 6 hours
```

**Implement memory conflict resolution**:
```python
class MemoryConflictResolver:
    async def detect_conflicts(self, user_id: str, new_memory: dict):
        """Detect if new memory conflicts with existing memories"""
        # Find semantically similar memories
        similar = await find_similar_memories(
            user_id,
            new_memory['embedding'],
            threshold=0.8,
            limit=5
        )

        conflicts = []
        for existing in similar:
            # Use LLM to detect contradictions
            is_conflict = await self.check_contradiction_llm(
                new_memory['content'],
                existing['content']
            )

            if is_conflict:
                conflicts.append(existing)

        return conflicts

    async def resolve_conflict(self, new_memory: dict, conflicting_memories: list):
        """Resolve conflicts by choosing most recent or reinforcing existing"""
        if not conflicting_memories:
            return new_memory

        # Strategy 1: Recency wins (for factual updates)
        if new_memory['sector_type'] == 'semantic':
            # Mark old memories as outdated
            for conflict in conflicting_memories:
                self.db.execute("""
                    UPDATE memories
                    SET state = 'archived',
                        metadata = json_set(metadata, '$.superseded_by', ?)
                    WHERE id = ?
                """, (new_memory['id'], conflict['id']))

            return new_memory

        # Strategy 2: Boost salience (for reinforced memories)
        if new_memory['sector_type'] == 'episodic':
            # Similar episodic memory = reinforcement
            for conflict in conflicting_memories:
                self.db.execute("""
                    UPDATE memories
                    SET salience = MIN(salience + 0.1, 1.0),
                        recency = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (conflict['id'],))

            # Don't add new memory, existing is reinforced
            return None

        return new_memory
```

---

### 7. API Design Improvements 🟢 **LOW PRIORITY**

#### Issues:
- No API versioning
- No pagination for large result sets
- No request validation beyond Pydantic
- No API documentation (OpenAPI)

#### Recommendations:

**Add API versioning**:
```python
from fastapi import APIRouter

# Version 1 API
api_v1 = APIRouter(prefix="/api/v1")

@api_v1.post("/tools/memory-query")
async def memory_query_v1(request: MemoryQueryRequest):
    # V1 implementation
    pass

# Version 2 API (future)
api_v2 = APIRouter(prefix="/api/v2")

@api_v2.post("/tools/memory-query")
async def memory_query_v2(request: MemoryQueryRequestV2):
    # V2 with breaking changes
    pass

# Mount routers
app.include_router(api_v1)
app.include_router(api_v2)

# Legacy endpoints redirect to v1
@app.post("/tools/memory-query")
async def memory_query_legacy(request: MemoryQueryRequest):
    return await memory_query_v1(request)
```

**Add pagination**:
```python
from pydantic import BaseModel, Field
from typing import Optional

class PaginationParams(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    cursor: Optional[str] = None  # For cursor-based pagination

class PaginatedResponse(BaseModel):
    items: list
    total: int
    has_more: bool
    next_cursor: Optional[str] = None

@app.get("/api/v1/memories/{user_id}")
async def list_memories(
    user_id: str,
    pagination: PaginationParams = Depends()
):
    # Count total
    total = db.execute("""
        SELECT COUNT(*) FROM memories
        WHERE user_id = ? AND state = 'active'
    """, (user_id,)).fetchone()[0]

    # Fetch page
    memories = db.execute("""
        SELECT * FROM memories
        WHERE user_id = ? AND state = 'active'
        ORDER BY recency DESC
        LIMIT ? OFFSET ?
    """, (user_id, pagination.limit, pagination.offset)).fetchall()

    has_more = (pagination.offset + pagination.limit) < total

    return PaginatedResponse(
        items=memories,
        total=total,
        has_more=has_more
    )
```

**Enhance OpenAPI documentation**:
```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="OpenMemory + ElevenLabs Integration API",
        version="1.0.0",
        description="""
        API for integrating OpenMemory cognitive memory system with ElevenLabs Agents.

        ## Integration Points

        1. **Conversation Initiation**: Load user context before conversation starts
        2. **Server Tools**: Real-time memory queries during conversation
        3. **Post-Call Webhooks**: Extract memories after conversation ends

        ## Authentication

        All endpoints require Bearer token authentication except webhooks.
        """,
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key"
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

---

### 8. Deployment and Operations 🟡 **MEDIUM PRIORITY**

#### Issues:
- No CI/CD pipeline
- No environment management
- No database migration strategy
- No backup/recovery strategy
- No container orchestration

#### Recommendations:

**Create Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health/live')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Create docker-compose for development**:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:////data/memories.db
      - REDIS_URL=redis://redis:6379
      - ELEVENLABS_WEBHOOK_SECRET=${ELEVENLABS_WEBHOOK_SECRET}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/data
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  worker:
    build: .
    command: python worker.py
    environment:
      - DATABASE_URL=sqlite:////data/memories.db
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/data
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

**Implement database migrations with Alembic**:
```python
# alembic/env.py
from alembic import context
from sqlalchemy import engine_from_config, pool

# Import your models
from app.models import Base

config = context.config
target_metadata = Base.metadata

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()
```

```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Add tool_invocations table"

# Apply migration
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

**Setup backup strategy**:
```bash
#!/bin/bash
# backup.sh - Database backup script

BACKUP_DIR="/backups"
DB_PATH="/data/memories.db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup
echo "Starting backup at $TIMESTAMP"
sqlite3 $DB_PATH ".backup '$BACKUP_DIR/memories_$TIMESTAMP.db'"

# Compress backup
gzip "$BACKUP_DIR/memories_$TIMESTAMP.db"

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR/memories_$TIMESTAMP.db.gz" \
    "s3://my-backups/memories/$TIMESTAMP.db.gz"

# Clean old backups
find $BACKUP_DIR -name "memories_*.db.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: memories_$TIMESTAMP.db.gz"
```

**Create Kubernetes deployment** (for scale):
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memory-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: memory-api
  template:
    metadata:
      labels:
        app: memory-api
    spec:
      containers:
      - name: api
        image: memory-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          value: redis://redis:6379
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: memory-api
spec:
  selector:
    app: memory-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2) 🔴
**Goal**: Production-ready core system

1. **Database optimization**
   - Add missing indexes
   - Implement connection pooling
   - Setup PRAGMA optimizations
   - Add database backup script

2. **Security hardening**
   - Implement proper authentication
   - Add rate limiting
   - Fix HMAC timing attack
   - Add API key rotation

3. **Basic observability**
   - Structured logging
   - Prometheus metrics
   - Health check endpoints

### Phase 2: Reliability (Week 3-4) 🟡
**Goal**: Resilient system with monitoring

1. **Error handling**
   - Circuit breakers for external services
   - Retry logic with jitter
   - Fallback strategies
   - Graceful degradation

2. **Caching layer**
   - Redis setup
   - Query result caching
   - Embedding caching
   - Cache invalidation strategy

3. **Monitoring & alerting**
   - OpenTelemetry tracing
   - Grafana dashboards
   - Alert rules
   - Log aggregation

### Phase 3: Scaling (Week 5-6) 🟢
**Goal**: Ready for 1000+ conversations/day

1. **Database migration**
   - PostgreSQL + pgvector setup
   - Data migration script
   - Performance testing
   - Rollback plan

2. **Memory system enhancements**
   - Memory decay implementation
   - Conflict resolution
   - Memory consolidation
   - Archival strategy

3. **Deployment automation**
   - Docker containerization
   - CI/CD pipeline (GitHub Actions)
   - Database migrations (Alembic)
   - Kubernetes manifests (if needed)

### Phase 4: Optimization (Week 7-8) 🟢
**Goal**: Fine-tuned production system

1. **Performance optimization**
   - Query optimization
   - Index tuning
   - Connection pool tuning
   - Cache hit rate optimization

2. **API improvements**
   - API versioning
   - Pagination
   - Enhanced documentation
   - Client SDK (optional)

3. **Testing & validation**
   - Load testing (100-1000 conversations/day)
   - Integration tests
   - Chaos engineering tests
   - Performance benchmarking

---

## Cost-Benefit Analysis

### High Priority Items (Implement First)

| Item | Estimated Effort | Risk Mitigated | Performance Gain |
|------|-----------------|----------------|------------------|
| Database indexes | 4 hours | HIGH | 50-100x query speedup |
| Authentication & rate limiting | 8 hours | CRITICAL | Security |
| HMAC timing fix | 1 hour | MEDIUM | Security |
| Connection pooling | 2 hours | MEDIUM | 2-3x throughput |
| Structured logging | 4 hours | LOW | Debuggability |
| Health checks | 2 hours | MEDIUM | Reliability |
| **Total** | **21 hours** | | |

### Medium Priority Items (Implement Second)

| Item | Estimated Effort | Risk Mitigated | Performance Gain |
|------|-----------------|----------------|------------------|
| Redis caching | 12 hours | MEDIUM | 50-80% latency reduction |
| Circuit breakers | 8 hours | MEDIUM | Availability |
| Prometheus metrics | 6 hours | LOW | Observability |
| OpenTelemetry tracing | 8 hours | LOW | Debuggability |
| Docker deployment | 6 hours | LOW | Ops efficiency |
| **Total** | **40 hours** | | |

### Low Priority Items (Nice to Have)

| Item | Estimated Effort | Risk Mitigated | Performance Gain |
|------|-----------------|----------------|------------------|
| Memory decay system | 16 hours | LOW | Data quality |
| API versioning | 4 hours | LOW | Future flexibility |
| Pagination | 4 hours | LOW | Scalability |
| PostgreSQL migration | 24 hours | MEDIUM | 10-100x vector search |
| Kubernetes setup | 16 hours | LOW | Enterprise scale |
| **Total** | **64 hours** | | |

---

## Testing Strategy

### Unit Tests
```python
# tests/test_memory_query.py
import pytest
from app.memory import MemoryQueryEngine

@pytest.fixture
def memory_engine():
    return MemoryQueryEngine(db_url="sqlite:///:memory:")

def test_composite_scoring(memory_engine):
    """Test composite score calculation"""
    score = memory_engine.calculate_score(
        similarity=0.9,
        salience=0.8,
        recency_score=0.7,
        link_weight=0.6
    )

    expected = 0.6 * 0.9 + 0.2 * 0.8 + 0.1 * 0.7 + 0.1 * 0.6
    assert abs(score - expected) < 0.001

def test_graph_expansion(memory_engine):
    """Test single-waypoint graph expansion"""
    initial = [{"id": 1, "score": 0.9}]
    expanded = await memory_engine.expand_via_links(initial, max_total=5)

    assert len(expanded) <= 5
    assert expanded[0]["id"] == 1  # Original should be first
    assert all(m["score"] > 0 for m in expanded)
```

### Integration Tests
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_memory_query_endpoint(client):
    """Test memory query API endpoint"""
    response = client.post(
        "/api/v1/tools/memory-query",
        json={
            "query": "user's CRM system",
            "user_id": "test_user",
            "limit": 3
        },
        headers={"Authorization": "Bearer test_token"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "memories" in data
    assert "summary" in data
    assert len(data["memories"]) <= 3

def test_rate_limiting(client):
    """Test rate limiting on API endpoints"""
    # Make 101 requests (limit is 100/minute)
    for i in range(101):
        response = client.post("/api/v1/tools/memory-query", ...)

    assert response.status_code == 429  # Too Many Requests
```

### Load Tests
```python
# tests/load_test.py
from locust import HttpUser, task, between

class MemoryAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def query_memory(self):
        self.client.post(
            "/api/v1/tools/memory-query",
            json={
                "query": "test query",
                "user_id": f"user_{self.user_id}",
                "limit": 3
            },
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def webhook_post_call(self):
        self.client.post(
            "/webhook/post-call",
            json={
                "webhook_event_id": "evt_test",
                "data": {"conversation_id": "conv_test", ...}
            }
        )

# Run: locust -f tests/load_test.py --host=http://localhost:8000
```

---

## Conclusion

The proposed OpenMemory + ElevenLabs integration architecture is **solid and well-designed** for the initial target of 100 conversations/day. However, to achieve production readiness and scale to 1000+ conversations/day, the following **critical improvements** must be implemented:

### Immediate Action Items (Week 1):
1. ✅ Add database indexes (4 hours) - **10x+ query performance**
2. ✅ Implement authentication & rate limiting (8 hours) - **Critical security**
3. ✅ Fix HMAC timing vulnerability (1 hour) - **Security hardening**
4. ✅ Add structured logging (4 hours) - **Operational visibility**
5. ✅ Implement health checks (2 hours) - **Reliability**

### Near-term Priorities (Week 2-4):
6. ✅ Deploy Redis caching layer (12 hours) - **50-80% latency reduction**
7. ✅ Add circuit breakers (8 hours) - **Improved resilience**
8. ✅ Setup Prometheus metrics (6 hours) - **Production monitoring**
9. ✅ Containerize with Docker (6 hours) - **Deployment simplification**

### Growth Path (Month 2-3):
10. ✅ Migrate to PostgreSQL + pgvector (24 hours) - **10-100x vector search performance**
11. ✅ Implement memory lifecycle management (16 hours) - **Data quality**
12. ✅ Setup CI/CD pipeline (16 hours) - **Deployment automation**

**Total estimated effort for production readiness**: ~125 hours (3-4 weeks with 1 developer)

**Expected outcomes**:
- 🎯 Support 1000+ conversations/day
- 🎯 p95 latency <500ms for all endpoints
- 🎯 99.9% uptime
- 🎯 Production-grade security and monitoring
- 🎯 Clear path to 10,000+ conversations/day

The architecture is well-positioned for success with these enhancements.
