# Technical Implementation Plan: OpenMemory + ElevenLabs Integration

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
│                  (Nginx / Caddy)                         │
└────────┬────────────────────────────────┬───────────────┘
         │                                │
         ▼                                ▼
┌─────────────────────┐         ┌─────────────────────┐
│  Webhook Handler    │         │  Tool API           │
│  (FastAPI)          │         │  (FastAPI)          │
│  - Initiation       │         │  - memory_query     │
│  - Post-call        │         │  - health checks    │
└────────┬────────────┘         └──────────┬──────────┘
         │                                  │
         │ Writes to queue                  │ Reads from DB
         ▼                                  ▼
┌─────────────────────────────────────────────────────────┐
│               SQLite Database (WAL mode)                │
│  - conversations, memories, embeddings                  │
│  - webhook_events (event queue)                         │
│  - tool_invocations (logs)                              │
└────────┬────────────────────────────────────────────────┘
         │
         │ Async polling
         ▼
┌─────────────────────┐
│  Worker Pool (2-3)  │
│  - Process webhooks │
│  - Extract memories │
│  - Generate embeddings │
│  - Build links      │
└─────────────────────┘
```

### Technology Stack

#### Backend
- **Language**: Python 3.11+
- **Web Framework**: FastAPI 0.104+
- **Async Runtime**: asyncio + uvicorn
- **Database**: SQLite 3.41+ with WAL mode
- **ORM**: SQLAlchemy 2.0+ (for future PostgreSQL migration)
- **Database Migrations**: Alembic

#### External Services
- **LLM Provider**: OpenAI GPT-4o-mini (memory extraction, summarization)
- **Embedding Provider**: OpenAI text-embedding-3-small (1536 dimensions)
- **Voice Agent Platform**: ElevenLabs Agents

#### Caching & Queue (Phase 2)
- **Cache**: Redis 7+ (for memory query caching, embedding caching)
- **Queue**: Redis Streams (for webhook event processing)

#### Monitoring & Observability
- **Metrics**: Prometheus + Grafana
- **Tracing**: OpenTelemetry + Jaeger
- **Logging**: structlog with JSON output
- **Health Checks**: FastAPI built-in

#### Deployment
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Docker Compose (dev/staging), Kubernetes (production scale)
- **CI/CD**: GitHub Actions
- **Hosting**: Cloud provider (AWS/GCP/Azure) or VPS

## Database Design

### Schema (11 Tables)

#### Integration Layer

```sql
-- Webhook event queue
CREATE TABLE webhook_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,  -- ElevenLabs webhook_event_id
    event_type TEXT NOT NULL,  -- post_call_transcription|audio|failure
    payload JSON NOT NULL,  -- Full webhook body
    conversation_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|completed|failed|dead_letter
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

CREATE INDEX idx_webhook_status_created ON webhook_events(status, created_at)
  WHERE status IN ('pending', 'processing');
CREATE INDEX idx_webhook_conversation ON webhook_events(conversation_id);
```

#### Conversation Domain

```sql
-- Core conversation records
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT UNIQUE NOT NULL,  -- ElevenLabs UUID
    agent_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_secs INTEGER,
    status TEXT NOT NULL,  -- in_progress|completed|failed
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_conversation_user ON conversations(user_id, start_time DESC);
CREATE INDEX idx_conversation_status ON conversations(status);

-- Turn-by-turn transcript
CREATE TABLE conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_fk INTEGER NOT NULL,
    role TEXT NOT NULL,  -- user|agent
    message TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    turn_order INTEGER NOT NULL,
    FOREIGN KEY (conversation_fk) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX idx_turns_conversation ON conversation_turns(conversation_fk, turn_order);

-- AI-generated conversation insights
CREATE TABLE conversation_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_fk INTEGER NOT NULL UNIQUE,
    summary TEXT,
    call_successful BOOLEAN,
    evaluation_json JSON,
    data_collection JSON,  -- action_items, sentiment, etc.
    FOREIGN KEY (conversation_fk) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Server tool invocation logs
CREATE TABLE tool_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_fk INTEGER,
    tool_name TEXT NOT NULL,
    request_params JSON NOT NULL,
    response_data JSON,
    latency_ms INTEGER,
    success BOOLEAN DEFAULT 1,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_fk) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX idx_tool_conversation ON tool_invocations(conversation_fk);
CREATE INDEX idx_tool_name ON tool_invocations(tool_name, timestamp DESC);
CREATE INDEX idx_tool_latency ON tool_invocations(latency_ms) WHERE latency_ms > 1000;
```

#### Memory Domain

```sql
-- User profiles
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    summary TEXT,  -- Auto-generated from memories
    total_memories INTEGER DEFAULT 0,
    last_interaction TIMESTAMP,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Memory nodes
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    sector_type TEXT NOT NULL,  -- semantic|episodic|procedural|emotional|reflective
    salience REAL NOT NULL DEFAULT 0.5,  -- 0.0 to 1.0
    recency TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decay_rate REAL DEFAULT 0.01,
    state TEXT NOT NULL DEFAULT 'active',  -- active|paused|archived
    source_conversation TEXT,  -- Optional conversation_id
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_memories_user_sector_state ON memories(user_id, sector_type, state)
  WHERE state = 'active';
CREATE INDEX idx_memories_recency ON memories(recency DESC);
CREATE INDEX idx_memories_salience ON memories(salience DESC);

-- Vector embeddings
CREATE TABLE memory_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    sector TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- Serialized numpy array
    embedding_model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX idx_embeddings_memory ON memory_embeddings(memory_id);
CREATE INDEX idx_embeddings_sector ON memory_embeddings(sector);

-- Memory graph links
CREATE TABLE memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id INTEGER NOT NULL,
    target_memory_id INTEGER NOT NULL,
    link_weight REAL NOT NULL,  -- 0.0 to 1.0 (similarity score)
    link_type TEXT DEFAULT 'automatic',  -- automatic|manual|reinforced
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE(source_memory_id, target_memory_id)
);

CREATE INDEX idx_links_source ON memory_links(source_memory_id, link_weight DESC);
CREATE INDEX idx_links_target ON memory_links(target_memory_id);

-- Junction: conversations ↔ memories
CREATE TABLE conversation_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_fk INTEGER NOT NULL,
    memory_id INTEGER NOT NULL,
    extraction_type TEXT NOT NULL,  -- auto|manual|reinforced
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_fk) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE(conversation_fk, memory_id)
);

CREATE INDEX idx_conv_memories_conversation ON conversation_memories(conversation_fk);
CREATE INDEX idx_conv_memories_memory ON conversation_memories(memory_id);
```

### SQLite Configuration

```python
# Must be set on every connection
PRAGMA journal_mode = WAL;          # CRITICAL: enables concurrent reads/writes
PRAGMA synchronous = NORMAL;        # Balance safety/performance
PRAGMA cache_size = -10000;         # 10MB cache
PRAGMA temp_store = MEMORY;         # In-memory temp tables
PRAGMA busy_timeout = 5000;         # Wait 5s on lock contention
PRAGMA foreign_keys = ON;           # Enforce foreign key constraints
```

## API Design

### Endpoint Specification

#### 1. Conversation Initiation Webhook

```python
POST /webhook/conversation-init
Content-Type: application/json

# Request (from ElevenLabs)
{
  "caller_id": "+15551234567",
  "agent_id": "agent_xyz",
  "called_number": "+15559876543",
  "call_sid": "CA123abc..."
}

# Response (to ElevenLabs)
{
  "dynamic_variables": {
    "user_id": "user_123",
    "user_name": "John Doe",
    "memory_context": "Last discussed billing issue on Nov 1...",
    "recent_topics": "billing, support tickets",
    "last_interaction": "2025-11-01T14:30:00Z"
  },
  "conversation_config_override": {
    "agent": {
      "first_message": "Hi John! How can I help you today?",
      "language": "en"
    }
  }
}

# Performance Target: <1000ms (p95)
```

#### 2. Memory Query Server Tool

```python
POST /tools/memory-query
Content-Type: application/json
Authorization: Bearer {api_token}

# Request (from ElevenLabs agent)
{
  "query": "user's CRM system preferences",
  "user_id": "user_123",
  "limit": 3,
  "sectors": ["semantic", "procedural"]  # optional
}

# Response
{
  "memories": [
    {
      "content": "User's company uses Salesforce CRM",
      "sector": "procedural",
      "salience": 0.9,
      "timestamp": "2025-10-15T10:00:00Z",
      "confidence": 0.92
    },
    {
      "content": "Needs API integration with Salesforce",
      "sector": "semantic",
      "salience": 0.85,
      "timestamp": "2025-10-15T10:05:00Z",
      "confidence": 0.88
    }
  ],
  "summary": "User's company uses Salesforce CRM and needs API integration support",
  "query_time_ms": 342
}

# Performance Target: <500ms (p95)
```

#### 3. Post-Call Webhook

```python
POST /webhook/post-call
Content-Type: application/json
ElevenLabs-Signature: t=1730764890,v1={hmac_signature}

# Request (from ElevenLabs)
{
  "webhook_event_id": "evt_123abc",
  "event_type": "post_call_transcription",
  "timestamp": 1730764890,
  "data": {
    "conversation_id": "conv_abc123",
    "agent_id": "agent_xyz",
    "status": "done",
    "metadata": {
      "start_time": "2025-11-04T18:00:00Z",
      "end_time": "2025-11-04T18:05:30Z",
      "duration_secs": 330
    },
    "transcript": [
      {"role": "user", "message": "I need help with...", "timestamp": 1730764800},
      {"role": "agent", "message": "I can help you with that!", "timestamp": 1730764805}
    ],
    "analysis": {
      "transcript_summary": "User inquired about...",
      "call_successful": true,
      "data_collection": {
        "action_items": ["Send pricing PDF"],
        "customer_sentiment": "positive"
      }
    },
    "conversation_initiation_client_data": {
      "dynamic_variables": {"user_id": "user_123"}
    }
  }
}

# Response (immediate)
HTTP 200 OK

# Performance Target: <100ms acknowledgment
# Async processing: <5000ms total
```

#### 4. Health Check Endpoints

```python
GET /health
# Returns: {"status": "healthy|degraded|unhealthy", "database": "...", "llm_api": "...", ...}

GET /health/live
# Liveness probe (Kubernetes)
# Returns: {"status": "alive"}

GET /health/ready
# Readiness probe (Kubernetes)
# Returns: {"status": "ready"} or 503
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### Goals
- Set up database with all tables and indexes
- Implement basic FastAPI application structure
- Configure SQLite with WAL mode
- Set up development environment (Docker Compose)
- Implement database migrations with Alembic

#### Deliverables
- ✅ SQLite database with all 11 tables
- ✅ Alembic migration scripts
- ✅ Docker Compose for local development
- ✅ Basic FastAPI app with health checks
- ✅ Structured logging configured
- ✅ Basic unit tests for database models

#### Technical Tasks
1. **Database Setup**
   - Create SQLAlchemy models for all 11 tables
   - Write Alembic migration for initial schema
   - Implement database connection manager with WAL mode
   - Add indexes for performance

2. **FastAPI Application**
   - Set up FastAPI app with uvicorn
   - Configure CORS, middleware
   - Implement health check endpoints
   - Set up structlog for JSON logging

3. **Development Environment**
   - Create Dockerfile for API service
   - Create docker-compose.yml (API, worker placeholder)
   - Set up environment variable management (.env files)
   - Document local setup in README

### Phase 2: Conversation Initiation Webhook (Week 2-3)

#### Goals
- Implement webhook endpoint for conversation initiation
- Implement user lookup and creation
- Implement fast memory query for top-5 memories
- Format memories as dynamic variables
- Meet <2s response time target

#### Deliverables
- ✅ `/webhook/conversation-init` endpoint
- ✅ User management logic (create, lookup)
- ✅ Fast memory retrieval (top-5 by composite score)
- ✅ Dynamic variable formatting
- ✅ Latency monitoring
- ✅ Integration tests

#### Technical Tasks
1. **Endpoint Implementation**
   ```python
   @app.post("/webhook/conversation-init")
   async def conversation_initiation(request: InitiationRequest):
       # 1. Map caller_id to user_id
       # 2. Get or create user
       # 3. Query top 5 memories
       # 4. Format as dynamic variables
       # 5. Return within 2s
   ```

2. **User Management**
   ```python
   async def get_or_create_user(caller_id: str) -> User:
       # Check if user exists by caller_id mapping
       # If not, create guest user
       # Return user object
   ```

3. **Memory Query (Fast Path)**
   ```python
   async def query_memories_for_initiation(
       user_id: str,
       query: str = "recent conversation topics",
       limit: int = 5
   ) -> List[Memory]:
       # Simple query without full vector search (too slow)
       # Use recency + salience only
       # Target: <500ms
   ```

4. **Testing**
   - Unit tests for user creation
   - Unit tests for memory formatting
   - Integration test: full webhook request/response
   - Load test: sustain <2s at 10 concurrent requests

### Phase 3: Memory Query Server Tool (Week 3-4)

#### Goals
- Implement vector similarity search
- Implement composite scoring
- Implement graph expansion (single-waypoint)
- Generate memory summaries with LLM
- Meet <500ms response time target

#### Deliverables
- ✅ `/tools/memory-query` endpoint with authentication
- ✅ Embedding generation integration (OpenAI)
- ✅ Vector similarity search
- ✅ Composite scoring algorithm
- ✅ Graph expansion logic
- ✅ LLM-based summarization
- ✅ Tool invocation logging
- ✅ Performance optimization

#### Technical Tasks
1. **Embedding Integration**
   ```python
   async def generate_embedding(text: str) -> List[float]:
       # Call OpenAI API: text-embedding-3-small
       # Return 1536-dimension vector
       # Cache common queries
   ```

2. **Vector Similarity Search**
   ```python
   async def vector_search(
       embedding: List[float],
       user_id: str,
       sectors: List[str],
       limit: int
   ) -> List[Memory]:
       # Calculate cosine similarity for each memory
       # Apply sector filter
       # Sort by composite score:
       #   0.6 * similarity +
       #   0.2 * salience +
       #   0.1 * recency_score +
       #   0.1 * avg_link_weight
       # Return top N
   ```

3. **Graph Expansion**
   ```python
   async def expand_via_links(
       initial_matches: List[Memory],
       max_total: int = 5
   ) -> List[Memory]:
       # For each initial match:
       #   Get top 2 linked memories (by link_weight)
       #   Add to result set (up to max_total)
       # Re-sort by boosted score
       # Return expanded set
   ```

4. **LLM Summarization**
   ```python
   async def generate_memory_summary(
       memories: List[Memory],
       query: str
   ) -> str:
       # Construct prompt with memories
       # Call GPT-4o-mini (fast, cheap)
       # Return 1-2 sentence summary
       # Target: <300ms
   ```

5. **Authentication**
   ```python
   from fastapi.security import HTTPBearer

   security = HTTPBearer()

   async def verify_api_key(
       credentials: HTTPAuthorizationCredentials = Depends(security)
   ) -> str:
       # Verify bearer token against stored API keys
       # Return user/client identifier
   ```

6. **Rate Limiting**
   ```python
   from slowapi import Limiter

   limiter = Limiter(key_func=get_remote_address)

   @app.post("/tools/memory-query")
   @limiter.limit("100/minute")
   async def memory_query_tool(...):
       # Endpoint logic
   ```

### Phase 4: Post-Call Webhook & Memory Extraction (Week 4-6)

#### Goals
- Implement post-call webhook with HMAC verification
- Implement async event processing with workers
- Implement LLM-based memory extraction
- Implement sector classification
- Implement automatic link building
- Meet <100ms acknowledgment, <5s total processing

#### Deliverables
- ✅ `/webhook/post-call` endpoint with signature verification
- ✅ Async webhook processing worker
- ✅ Memory extraction pipeline (LLM-based)
- ✅ Sector classification logic
- ✅ Embedding generation and storage
- ✅ Automatic link building (similarity > 0.8)
- ✅ User summary regeneration
- ✅ Retry logic with exponential backoff
- ✅ Dead letter queue handling

#### Technical Tasks
1. **Webhook Endpoint**
   ```python
   @app.post("/webhook/post-call")
   async def post_call_webhook(request: Request):
       # 1. Verify HMAC signature (constant-time comparison)
       # 2. Parse payload
       # 3. Insert into webhook_events (status='pending')
       # 4. Return 200 OK immediately (<100ms)
   ```

2. **HMAC Verification**
   ```python
   def verify_elevenlabs_signature(
       signature_header: str,
       body: bytes,
       secret: str
   ) -> bool:
       # Parse signature header: t=timestamp,v1=signature
       # Check timestamp freshness (5 min window)
       # Compute HMAC-SHA256
       # Compare with secrets.compare_digest (constant-time)
   ```

3. **Background Worker**
   ```python
   async def webhook_processor_worker():
       while True:
           # Poll webhook_events WHERE status='pending'
           events = fetch_pending_events(limit=5)

           for event in events:
               # Mark as 'processing'
               # Process event in transaction
               # Mark as 'completed' or retry

           await asyncio.sleep(0.5)  # Poll every 500ms
   ```

4. **Memory Extraction Pipeline**
   ```python
   async def process_webhook_event(event: WebhookEvent):
       with db.transaction():
           # 1. Store conversation + turns
           # 2. Store analysis
           # 3. Extract memories from transcript
           # 4. Link memories to conversation
           # 5. Update user summary

   async def extract_memories_from_transcript(
       user_id: str,
       transcript: List[Turn],
       conversation_id: str
   ) -> List[int]:  # memory IDs
       # 1. Concatenate transcript
       # 2. Call LLM with extraction prompt
       # 3. Parse JSON response (structured output)
       # 4. For each extracted memory:
       #    - Create memory node
       #    - Generate embedding
       #    - Find similar memories
       #    - Create links (similarity > 0.8)
       # 5. Return memory IDs
   ```

5. **Sector Classification Prompt**
   ```
   Analyze this conversation and extract memories into categories:

   1. SEMANTIC: Facts, preferences, attributes (e.g., "uses Salesforce CRM")
   2. EPISODIC: Specific events, actions (e.g., "called about billing on Nov 1")
   3. PROCEDURAL: How-to knowledge (e.g., "prefers email for updates")
   4. EMOTIONAL: Sentiment (e.g., "frustrated with billing issues")
   5. REFLECTIVE: Patterns (e.g., "tends to call during lunch hours")

   For each memory, provide:
   - category: semantic|episodic|procedural|emotional|reflective
   - content: concise description (<100 chars)
   - salience: 0.0-1.0 (importance score)

   Output as JSON array.
   ```

6. **Automatic Link Building**
   ```python
   async def build_automatic_links(
       new_memory_id: int,
       user_id: str,
       embedding: List[float]
   ):
       # Find similar existing memories (cosine > 0.8)
       similar = await find_similar_memories(
           user_id,
           embedding,
           threshold=0.8,
           limit=5
       )

       # Create bidirectional links
       for existing_memory in similar:
           create_link(
               source=new_memory_id,
               target=existing_memory.id,
               weight=existing_memory.similarity
           )
   ```

7. **User Summary Regeneration**
   ```python
   async def regenerate_user_summary(user_id: str):
       # Get top 10 high-salience memories
       top_memories = get_top_memories(user_id, limit=10)

       # Call LLM to generate summary
       prompt = f"Summarize this user based on their memories:\n{memories_text}"
       summary = await llm_generate(prompt, max_tokens=200)

       # Update user record
       update_user(user_id, summary=summary)
   ```

### Phase 5: Monitoring & Observability (Week 6-7)

#### Goals
- Implement Prometheus metrics
- Set up Grafana dashboards
- Implement OpenTelemetry tracing
- Configure alerts
- Set up log aggregation

#### Deliverables
- ✅ Prometheus metrics exposed at `/metrics`
- ✅ Grafana dashboards for all key metrics
- ✅ OpenTelemetry tracing for critical paths
- ✅ Alert rules configured (high latency, errors, queue depth)
- ✅ Structured logs aggregated (Loki or similar)

#### Technical Tasks
1. **Prometheus Metrics**
   ```python
   from prometheus_client import Counter, Histogram, Gauge

   # Define metrics
   memory_query_latency = Histogram(
       'memory_query_latency_seconds',
       'Memory query latency',
       buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
   )

   memory_query_total = Counter(
       'memory_query_total',
       'Total memory queries',
       ['status']
   )

   webhook_queue_depth = Gauge(
       'webhook_queue_depth',
       'Pending webhook events'
   )

   # Use in endpoints
   @memory_query_latency.time()
   async def memory_query_tool(...):
       ...
   ```

2. **Grafana Dashboards**
   - Dashboard 1: API Performance (latency, throughput, error rate)
   - Dashboard 2: Memory System (query performance, extraction quality)
   - Dashboard 3: Webhooks (queue depth, processing time, retries)
   - Dashboard 4: External APIs (OpenAI latency, error rate)

3. **OpenTelemetry Tracing**
   ```python
   from opentelemetry import trace

   tracer = trace.get_tracer(__name__)

   @app.post("/tools/memory-query")
   async def memory_query_tool(request: MemoryQueryRequest):
       with tracer.start_as_current_span("memory_query") as span:
           span.set_attribute("user_id", request.user_id)

           with tracer.start_as_current_span("generate_embedding"):
               embedding = await generate_embedding(request.query)

           with tracer.start_as_current_span("vector_search"):
               results = await vector_search(embedding, request.user_id)

           return results
   ```

4. **Alert Configuration**
   ```yaml
   # prometheus-alerts.yml
   groups:
     - name: memory_system
       rules:
         - alert: HighMemoryQueryLatency
           expr: histogram_quantile(0.95, memory_query_latency_seconds) > 0.5
           for: 5m
           labels:
             severity: warning
           annotations:
             summary: "Memory query p95 latency is high"

         - alert: WebhookQueueBacklog
           expr: webhook_queue_depth > 50
           for: 10m
           labels:
             severity: critical
           annotations:
             summary: "Webhook queue is backing up"
   ```

### Phase 6: Security & Production Hardening (Week 7-8)

#### Goals
- Implement authentication and authorization
- Implement rate limiting
- Add encryption at rest for sensitive data
- Security audit and penetration testing
- Implement backup and recovery

#### Deliverables
- ✅ Bearer token authentication for all tool endpoints
- ✅ Rate limiting on all public endpoints
- ✅ Encryption at rest for conversation transcripts
- ✅ Security audit completed
- ✅ Automated database backups
- ✅ Disaster recovery plan documented

#### Technical Tasks
1. **Authentication**
   ```python
   # API key management
   class APIKeyManager:
       def __init__(self):
           self.valid_keys = load_api_keys()

       async def verify(self, token: str) -> bool:
           return secrets.compare_digest(token, self.valid_keys.get(hash(token)))

   # Use in endpoints
   @app.post("/tools/memory-query")
   async def memory_query_tool(
       request: MemoryQueryRequest,
       api_key: str = Depends(api_key_manager.verify)
   ):
       ...
   ```

2. **Encryption at Rest**
   ```python
   from cryptography.fernet import Fernet

   class DataEncryption:
       def __init__(self, key: bytes):
           self.cipher = Fernet(key)

       def encrypt(self, data: str) -> bytes:
           return self.cipher.encrypt(data.encode())

       def decrypt(self, encrypted: bytes) -> str:
           return self.cipher.decrypt(encrypted).decode()

   # Encrypt conversation messages before storing
   db.execute(
       "INSERT INTO conversation_turns (message) VALUES (?)",
       (encryptor.encrypt(turn.message),)
   )
   ```

3. **Automated Backups**
   ```bash
   #!/bin/bash
   # backup.sh

   BACKUP_DIR="/backups"
   TIMESTAMP=$(date +%Y%m%d_%H%M%S)

   # SQLite backup
   sqlite3 /data/memories.db ".backup '$BACKUP_DIR/memories_$TIMESTAMP.db'"
   gzip "$BACKUP_DIR/memories_$TIMESTAMP.db"

   # Upload to S3
   aws s3 cp "$BACKUP_DIR/memories_$TIMESTAMP.db.gz" \
       "s3://backups/memories/$TIMESTAMP.db.gz"

   # Cleanup old backups (>30 days)
   find $BACKUP_DIR -name "*.db.gz" -mtime +30 -delete
   ```

## Testing Strategy

### Unit Tests
- **Database Models**: Test all CRUD operations
- **Business Logic**: Memory scoring, graph expansion, sector classification
- **Utilities**: HMAC verification, encryption, embedding generation

```python
# tests/test_memory_scoring.py
def test_composite_score():
    score = calculate_composite_score(
        similarity=0.9,
        salience=0.8,
        recency_score=0.7,
        link_weight=0.6
    )
    expected = 0.6*0.9 + 0.2*0.8 + 0.1*0.7 + 0.1*0.6
    assert abs(score - expected) < 0.001
```

### Integration Tests
- **API Endpoints**: Test full request/response cycle
- **Database Transactions**: Test rollback on error
- **External APIs**: Mock OpenAI responses

```python
# tests/test_api.py
@pytest.mark.asyncio
async def test_memory_query_endpoint(client):
    response = await client.post(
        "/tools/memory-query",
        json={"query": "test", "user_id": "user_123", "limit": 3},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "memories" in response.json()
```

### Load Tests
- **Locust**: Simulate 100 conversations/day
- **Target**: Maintain <500ms p95 under load

```python
# tests/load_test.py
from locust import HttpUser, task, between

class MemoryAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def query_memory(self):
        self.client.post(
            "/tools/memory-query",
            json={"query": "test", "user_id": f"user_{self.user_id}", "limit": 3},
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def webhook_post_call(self):
        self.client.post(
            "/webhook/post-call",
            json={"webhook_event_id": "evt_test", "data": {...}}
        )
```

### User Acceptance Tests
- **Scenario 1**: New user calls, receives personalized greeting
- **Scenario 2**: Agent queries memory during conversation
- **Scenario 3**: New memories extracted after call
- **Scenario 4**: User calls back, agent recalls previous context

## Deployment Strategy

### Development Environment
```bash
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:////data/memories.db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/data

  worker:
    build: .
    command: python worker.py
    environment:
      - DATABASE_URL=sqlite:////data/memories.db
    volumes:
      - ./data:/data
    deploy:
      replicas: 2
```

### Production Deployment
1. **Build Docker image**: `docker build -t memory-api:v1.0 .`
2. **Push to registry**: `docker push registry.example.com/memory-api:v1.0`
3. **Deploy to server**: `docker-compose -f docker-compose.prod.yml up -d`
4. **Run migrations**: `docker exec memory-api alembic upgrade head`
5. **Monitor dashboards**: Verify metrics, check logs
6. **Gradual rollout**: Start with 10% traffic, increase to 100%

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build and push Docker image
        run: |
          docker build -t memory-api:${{ github.sha }} .
          docker push registry.example.com/memory-api:${{ github.sha }}

      - name: Deploy to production
        run: |
          ssh deploy@server 'docker-compose pull && docker-compose up -d'
```

## Migration Path to PostgreSQL (>1000 conversations/day)

### When to Migrate
- Write throughput >500 writes/sec
- Vector search latency >1s (p95)
- Database size >10GB
- Need for horizontal scaling

### Migration Steps
1. **Set up PostgreSQL with pgvector**
   ```sql
   CREATE EXTENSION vector;

   CREATE TABLE memory_embeddings (
       id SERIAL PRIMARY KEY,
       memory_id INTEGER NOT NULL,
       embedding vector(1536),  -- Native vector type
       ...
   );

   CREATE INDEX ON memory_embeddings USING ivfflat (embedding vector_cosine_ops);
   ```

2. **Data Migration**
   ```python
   # Export from SQLite
   sqlite_data = export_all_tables()

   # Import to PostgreSQL
   for table, rows in sqlite_data.items():
       pg_bulk_insert(table, rows)
   ```

3. **Update Application Code**
   ```python
   # Change from SQLite-specific
   embedding = pickle.loads(row['embedding'])

   # To PostgreSQL pgvector
   from pgvector.sqlalchemy import Vector
   embedding = row['embedding']  # Already a vector
   ```

4. **Performance Comparison**
   - SQLite: Linear scan, O(n) for vector search
   - PostgreSQL + pgvector: Indexed search, O(log n)
   - Expected: 10-100x speedup on vector queries

## Risk Mitigation

### Risk: OpenAI API Latency Spikes
- **Mitigation**: Cache embeddings, use circuit breaker, fallback to keyword search
- **Monitoring**: Track OpenAI API latency, set alerts

### Risk: SQLite Performance Limits
- **Mitigation**: Load test early, plan PostgreSQL migration
- **Monitoring**: Track database query latency, WAL size

### Risk: Memory Extraction Quality Issues
- **Mitigation**: Human review sample, iterative prompt tuning
- **Monitoring**: Track extraction success rate, manual evaluation

### Risk: Webhook Delivery Failures
- **Mitigation**: Implement retry logic, poll for missed events
- **Monitoring**: Track webhook retry count, dead letter queue depth

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Initiation webhook latency | <1s p95 | 🎯 |
| Memory query latency | <500ms p95 | 🎯 |
| Memory extraction latency | <3s p95 | 🎯 |
| System uptime | 99.9% | 🎯 |
| Memory recall accuracy | >90% | 🎯 |
| Test coverage | >80% | 🎯 |

---

**Version**: 1.0
**Last Updated**: 2025-11-05
**Approved By**: Engineering Team
**Next Review**: 2025-12-05
