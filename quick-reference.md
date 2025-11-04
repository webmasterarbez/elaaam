# OpenMemory + ElevenLabs Integration: Quick Reference

## Critical Corrections from Initial Design

### ❌ INCORRECT ASSUMPTION
"ElevenLabs only provides post-call webhooks, requiring pre-conversation memory injection via dynamic variables"

### ✅ ACTUAL CAPABILITIES
ElevenLabs provides **THREE integration points** for complete memory lifecycle:

1. **Conversation Initiation Webhook** - Pre-conversation context loading
2. **Server Tools** - Real-time memory queries DURING conversation
3. **Post-Call Webhooks** - Post-conversation memory extraction

**Key Insight**: Server tools enable **dynamic, on-demand memory access** during conversations, not just static pre-loaded context!

## Three Integration Points

### 1. Conversation Initiation Webhook (Pre-Conversation)

**When**: Before conversation starts (during Twilio connection/dialing period)

**Flow**: 
```
ElevenLabs → Your API: {caller_id, agent_id, ...}
Your API → ElevenLabs: {dynamic_variables, conversation_config_override}
```

**Purpose**: Load initial memory context

**Implementation**:
```python
@app.post("/webhook/conversation-init")
async def conversation_initiation(request: Request):
    # 1. Map caller_id to user_id
    # 2. Query OpenMemory for top 5 memories
    # 3. Return as dynamic_variables
    # MUST respond in <2 seconds
```

**Target**: <2s response time
**Format**: Returns `dynamic_variables` object with memory context

---

### 2. Server Tools (During Conversation)

**When**: Real-time during conversation when agent needs memory context

**Flow**:
```
Agent LLM detects need → Calls your tool endpoint
Your API: Query OpenMemory → Return memories
Agent: Uses memories in response
```

**Purpose**: On-demand memory retrieval

**Implementation**:
```python
@app.post("/tools/memory-query")
async def memory_query_tool(request: MemoryQueryRequest):
    # 1. Generate embedding for query
    # 2. Vector similarity search
    # 3. Single-waypoint graph expansion
    # 4. Return formatted memories + summary
    # Target: <500ms
```

**Target**: <500ms (p95), <1000ms (p99)
**Format**: JSON response with memories array + summary

**Agent Configuration**:
```json
{
  "tool_name": "query_user_memory",
  "description": "Search user's conversation history...",
  "url": "https://your-api.com/tools/memory-query",
  "body_parameters": [
    {"name": "query", "type": "string", "required": true},
    {"name": "user_id", "type": "string", "value": "{{user_id}}"}
  ]
}
```

---

### 3. Post-Call Webhooks (Post-Conversation)

**When**: After conversation ends

**Flow**:
```
ElevenLabs → Your webhook: {transcript, analysis, ...}
Your API: Acknowledge immediately (<100ms) → Queue for processing
Background worker: Extract memories asynchronously (1-5s)
```

**Purpose**: Extract and store new memories

**Implementation**:
```python
@app.post("/webhook/post-call")
async def post_call_webhook(request: Request):
    # 1. Verify HMAC signature
    # 2. INSERT into webhook_events (status='pending')
    # 3. Return HTTP 200 immediately
    # Background worker processes async
```

**Critical**: <100ms acknowledgment, then async processing

---

## Database Schema (11 Tables)

### Integration Layer
- `webhook_events` - Event queue for async processing

### Conversation Domain  
- `conversations` - Core conversation records
- `conversation_turns` - Turn-by-turn transcript
- `conversation_analysis` - AI-generated insights
- `tool_invocations` - Server tool call logs

### Memory Domain
- `users` - User profiles
- `memories` - Memory nodes (5 sectors)
- `memory_embeddings` - Vector embeddings per sector
- `memory_links` - Graph relationships (single-waypoint)
- `conversation_memories` - Junction: conversations ↔ memories

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] SQLite database with WAL mode (`PRAGMA journal_mode = WAL`)
- [ ] All 11 tables created with proper foreign keys
- [ ] Indexes created (especially on webhook status, memory user_id+sector)
- [ ] Embedding generation API integrated (OpenAI or local)
- [ ] LLM API for memory extraction (GPT-4o-mini recommended)

### Phase 2: Conversation Initiation Webhook
- [ ] Endpoint created: `/webhook/conversation-init`
- [ ] Caller ID to user ID mapping logic
- [ ] Fast memory query (<500ms for top-5)
- [ ] Dynamic variables formatting
- [ ] Response time monitoring (<2s target)
- [ ] Configure in ElevenLabs agent settings (Twilio integration)

### Phase 3: Server Tools
- [ ] Endpoint created: `/tools/memory-query`
- [ ] Vector similarity search implemented
- [ ] Composite scoring (0.6×sim + 0.2×sal + 0.1×rec + 0.1×link)
- [ ] Single-waypoint graph expansion
- [ ] Memory summary generation (LLM-based)
- [ ] Tool invocation logging
- [ ] Response time monitoring (<500ms target)
- [ ] Configure tool in ElevenLabs agent:
  - [ ] Tool name and description
  - [ ] URL endpoint
  - [ ] Body parameters (query, user_id, limit)
  - [ ] Dynamic variable assignment
- [ ] Update agent system prompt to use tool

### Phase 4: Post-Call Webhook
- [ ] Endpoint created: `/webhook/post-call`
- [ ] HMAC signature verification
- [ ] Idempotent event handling (via event_id uniqueness)
- [ ] Fast acknowledgment (<100ms)
- [ ] Background worker pool (2-3 workers)
- [ ] Memory extraction pipeline (LLM-based)
- [ ] Sector classification (semantic, episodic, procedural, emotional, reflective)
- [ ] Embedding generation and storage
- [ ] Automatic link building (cosine > 0.8)
- [ ] User summary regeneration
- [ ] Webhook retry logic (exponential backoff)
- [ ] Dead letter queue for failures
- [ ] Configure in ElevenLabs agent settings

### Phase 5: Monitoring & Optimization
- [ ] Latency monitoring (p95, p99 for all endpoints)
- [ ] Error rate tracking
- [ ] Queue depth monitoring
- [ ] Database size monitoring (especially WAL)
- [ ] Memory extraction success rate
- [ ] Tool call frequency and patterns
- [ ] Alerts configured (slow responses, high queue depth, extraction failures)
- [ ] Caching layer for frequent memory queries (optional, for scale)

---

## Key Performance Targets

| Metric | Target (p95) | Target (p99) | Critical |
|--------|--------------|--------------|----------|
| Initiation webhook | <1000ms | <1800ms | 2000ms |
| Server tool query | <500ms | <1000ms | 1500ms |
| Post-call ack | <50ms | <100ms | 200ms |
| Memory extraction | <3000ms | <5000ms | 10000ms |

---

## Critical SQLite Configuration

```sql
PRAGMA journal_mode = WAL;          -- REQUIRED for concurrent access
PRAGMA synchronous = NORMAL;        -- Balance safety/performance
PRAGMA cache_size = -10000;         -- 10MB cache
PRAGMA temp_store = MEMORY;         -- In-memory temp tables
PRAGMA busy_timeout = 5000;         -- Wait 5s on lock contention
```

**Without WAL mode, concurrent writes will fail!**

---

## Example Agent System Prompt

```
You are a helpful AI assistant with access to the user's conversation history.

USER CONTEXT:
- User ID: {{user_id}}
- Name: {{user_name}}
- Memory Context: {{memory_context}}
- Recent Topics: {{recent_topics}}

CAPABILITIES:
You have the query_user_memory tool to search past conversations.
Use this when:
- User references past discussions ("What did we talk about?")
- You need context about their preferences or history
- Uncertain about details from {{memory_context}}

INSTRUCTIONS:
- Use the user's name naturally
- Reference {{memory_context}} for initial context
- Call query_user_memory for specific historical details
- Always verify before making assumptions based on memory

Example:
User: "What CRM do we use?"
You: [Calls query_user_memory with query="user's CRM system"]
Then responds with the retrieved information.
```

---

## Common Pitfalls to Avoid

1. **❌ Not using WAL mode** - Will cause write lock contention
2. **❌ Synchronous webhook processing** - Must acknowledge <100ms
3. **❌ Not implementing HMAC verification** - Security risk
4. **❌ Missing idempotency** - Can process same event twice
5. **❌ Slow initiation webhook** - Delays conversation start
6. **❌ Not configuring server tools properly** - Agent won't call them
7. **❌ Overly complex tool descriptions** - Confuses agent LLM
8. **❌ Not monitoring latencies** - Can't detect performance issues
9. **❌ Forgetting to update agent system prompt** - Won't use tools effectively
10. **❌ Not setting dynamic variable values** - Tools won't work (e.g., "{{user_id}}")

---

## Scaling Thresholds

**100 conversations/day** (current design):
- ✅ SQLite with WAL mode
- ✅ 2-3 background workers
- ✅ 2 vCPU, 4GB RAM

**1,000 conversations/day**:
- ⚠️ Consider Redis for event queue
- ⚠️ 10-15 background workers
- ⚠️ Add caching layer (Redis) for frequent queries
- ⚠️ 4 vCPU, 8GB RAM

**10,000+ conversations/day**:
- 🔴 Migrate to PostgreSQL with pgvector
- 🔴 Dedicated embedding service
- 🔴 Message queue (RabbitMQ/SQS)
- 🔴 Horizontal scaling with load balancer
- 🔴 Separate read replicas

---

## Testing Checklist

### Unit Tests
- [ ] Memory extraction from transcripts
- [ ] Sector classification logic
- [ ] Composite scoring formula
- [ ] Graph expansion (single-waypoint)
- [ ] HMAC signature verification
- [ ] Embedding generation and storage

### Integration Tests
- [ ] Initiation webhook end-to-end
- [ ] Server tool query end-to-end
- [ ] Post-call webhook → memory extraction
- [ ] Database transactions (rollback on error)
- [ ] Retry logic and exponential backoff

### Load Tests
- [ ] 100 conversations/day sustained
- [ ] Burst of 10 conversations/hour
- [ ] Initiation webhook <2s under load
- [ ] Server tool <500ms under load
- [ ] Queue processing keeps up with ingestion

### User Acceptance Tests
- [ ] Agent uses initial context correctly
- [ ] Agent calls tools when appropriate
- [ ] Agent responses include retrieved memories
- [ ] New memories extracted after conversations
- [ ] Memory recall improves over time

---

## Next Steps

1. **Set up infrastructure** - SQLite, API server, worker pool
2. **Implement initiation webhook** - Get basic context loading working
3. **Add server tool** - Enable real-time memory queries
4. **Implement post-call processing** - Complete memory extraction loop
5. **Configure ElevenLabs agent** - Tools, system prompt, webhooks
6. **Test end-to-end** - Make test calls, verify memory flow
7. **Monitor and optimize** - Track latencies, tune performance
8. **Scale gradually** - Start at 10 conversations/day, increase slowly

---

## Resources

- [Full Architecture Document](./openmemory-elevenlabs-architecture-corrected.md)
- [Database ERD](./database-erd.mmd)
- [Data Flow Sequence Diagram](./data-flow-sequence.mmd)
- [System Architecture Diagram](./system-architecture.mmd)
- [ElevenLabs Docs - Personalization](https://elevenlabs.io/docs/agents-platform/customization/personalization)
- [ElevenLabs Docs - Server Tools](https://elevenlabs.io/docs/agents-platform/customization/tools/server-tools)
- [ElevenLabs Docs - Post-Call Webhooks](https://elevenlabs.io/docs/agents-platform/workflows/post-call-webhooks)
- [OpenMemory GitHub](https://github.com/CaviraOSS/OpenMemory)
