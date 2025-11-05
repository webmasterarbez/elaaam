# OpenMemory + ElevenLabs Agents: Corrected Integration Architecture

## Executive Summary

This architecture integrates OpenMemory's cognitive memory system with ElevenLabs Agents through **three critical integration points**:

1. **Conversation Initiation Webhook** - ElevenLabs calls your endpoint at conversation start, you return memory context via dynamic variables
2. **Server Tools** - Agent makes real-time API calls during conversation to query memories on-demand  
3. **Post-Call Webhooks** - ElevenLabs sends conversation data after completion for memory extraction

**Key Correction**: ElevenLabs DOES support real-time memory access during conversations via server tools, not just static pre-loaded context. This enables dynamic, context-aware memory retrieval throughout the conversation.

## ElevenLabs Integration Points

### 1. Conversation Initiation Webhook (Pre-Conversation)

**Purpose**: Load initial memory context before conversation starts

**Flow**:
```
Inbound Twilio Call
    ↓
ElevenLabs → POST https://your-api.com/webhook/initiation
Request Body: {
  "caller_id": "+15551234567",
  "agent_id": "agent_xyz", 
  "called_number": "+15559876543",
  "call_sid": "CA123abc..."
}
    ↓
Your API:
  1. Identify user from caller_id
  2. Query OpenMemory for relevant memories
  3. Format as dynamic variables
  4. Return within <2s
    ↓
Response: {
  "dynamic_variables": {
    "user_id": "user_123",
    "user_name": "John Doe",
    "memory_context": "Last discussed billing issue on Nov 1. Prefers email updates. Account type: Premium.",
    "recent_topics": "pricing, support tickets",
    "user_preferences": "technical depth: high, communication style: direct"
  },
  "conversation_config_override": {
    "agent": {
      "prompt": null,  // Optional: override system prompt
      "first_message": "Hi John! I see you called about billing last week. How can I help today?",
      "language": "en"
    }
  }
}
    ↓
Conversation starts with memory-aware context
```

**Critical Requirements**:
- Response time <2 seconds (during connection/dialing period)
- Return ALL dynamic variables defined in agent config
- Dynamic variables populated in agent's system prompt via {{variable_name}} syntax

### 2. Server Tools (During Conversation)

**Purpose**: Real-time memory queries as conversation progresses

**Agent Configuration Example**:
```json
{
  "tool_name": "query_user_memory",
  "description": "Search the user's conversation history and preferences to find relevant information. Use this when the user references past conversations or when you need context about their preferences.",
  "url": "https://your-api.com/tools/memory-query",
  "method": "POST",
  "body_parameters": [
    {
      "name": "query",
      "type": "string",
      "description": "Natural language query describing what memory to retrieve (e.g. 'what did we discuss about pricing?', 'user's preferred communication style')",
      "required": true
    },
    {
      "name": "user_id",
      "type": "string", 
      "description": "The user's unique identifier",
      "required": true,
      "value": "{{user_id}}"  // Dynamic variable from initiation webhook
    },
    {
      "name": "limit",
      "type": "integer",
      "description": "Maximum number of memories to return (default: 3)",
      "required": false,
      "value": 3
    }
  ],
  "dynamic_variable_assignment": [
    {
      "source_path": "$.memories",  // JSONPath in response
      "target_variable": "retrieved_memories"  // Variable name in agent context
    }
  ]
}
```

**Server Tool Flow**:
```
User: "What did we talk about last time?"
    ↓
Agent detects need for memory context
    ↓
Agent calls: POST https://your-api.com/tools/memory-query
{
  "query": "previous conversation topics with user",
  "user_id": "user_123",
  "limit": 3
}
    ↓
Your API:
  1. Parse query
  2. Generate embedding
  3. Vector search in OpenMemory
  4. Apply composite scoring
  5. Return formatted memories
    ↓
Response (<1s):
{
  "memories": [
    {
      "content": "Discussed upgrading to enterprise plan, concerned about price",
      "sector": "semantic",
      "salience": 0.9,
      "timestamp": "2025-11-01T14:30:00Z"
    },
    {
      "content": "User requested detailed pricing breakdown via email",
      "sector": "procedural",
      "salience": 0.8,
      "timestamp": "2025-11-01T14:35:00Z"
    }
  ],
  "summary": "Last week you asked about enterprise pricing and wanted details emailed"
}
    ↓
Agent: "Last week you asked about enterprise pricing and wanted the details emailed. I can help you with that!"
```

**Best Practices for Server Tools**:
- Tool response time target: <500ms (p95), <1s (p99)
- Return concise, formatted text the agent can use directly
- Include confidence scores if multiple results
- Use dynamic variables ({{user_id}}) to pass context automatically
- Define clear tool descriptions so agent knows when to call

### 3. Post-Call Webhooks (Post-Conversation)

**Purpose**: Extract and store new memories after conversation ends

**Flow**:
```
Conversation ends
    ↓
ElevenLabs → POST https://your-api.com/webhook/post-call
{
  "webhook_event_id": "evt_123",
  "event_type": "post_call_transcription",
  "timestamp": 1730764890,
  "data": {
    "conversation_id": "conv_abc",
    "agent_id": "agent_xyz",
    "status": "done",
    "metadata": {
      "start_time": "2025-11-04T18:00:00Z",
      "end_time": "2025-11-04T18:05:30Z",
      "duration_secs": 330
    },
    "transcript": [
      {
        "role": "user",
        "message": "I need to upgrade to the enterprise plan",
        "timestamp": 1730764800
      },
      {
        "role": "agent", 
        "message": "I can help you with that! Let me pull up the enterprise options.",
        "timestamp": 1730764805
      }
      // ... full transcript
    ],
    "analysis": {
      "transcript_summary": "User inquired about enterprise plan upgrade, discussed pricing and features, agreed to trial period",
      "call_successful": true,
      "evaluation_results": {...},
      "data_collection": {
        "action_items": ["Send enterprise pricing PDF", "Schedule demo call"],
        "customer_sentiment": "positive"
      }
    },
    "conversation_initiation_client_data": {
      "dynamic_variables": {
        "user_id": "user_123",
        "user_name": "John Doe"
      }
    }
  }
}
    ↓
Your API (<100ms):
  1. Verify HMAC signature
  2. Insert into webhook_events table (status='pending')
  3. Return HTTP 200
    ↓
Background Worker (1-5s):
  1. Extract memories from transcript using LLM
  2. Classify into sectors (semantic, episodic, procedural, emotional, reflective)
  3. Generate embeddings
  4. Insert into memories table
  5. Build automatic links to similar memories
  6. Update user summary
  7. Mark webhook as processed
```

**Webhook Types**:
- `post_call_transcription`: Full transcript + analysis (most common)
- `post_call_audio`: Base64 MP3 audio (optional, large files)
- `call_initiation_failure`: Failed call attempts (errors, no answer, etc.)

## Complete Entity Relationship Diagram

### Primary Entities (11 Tables)

```
┌─────────────────────────────────────────────┐
│         INTEGRATION LAYER                   │
└─────────────────────────────────────────────┘

┌──────────────────────────┐
│   webhook_events         │  Event queue
│   (id INTEGER PK)        │
│   - event_id TEXT UK     │  ElevenLabs webhook_event_id
│   - event_type TEXT      │  post_call_transcription|audio|failure
│   - payload JSON         │  Full webhook body
│   - conversation_id TEXT │  
│   - status TEXT          │  pending|processing|completed|failed
│   - retry_count INT      │
│   - created_at TIMESTAMP │
└────────┬─────────────────┘
         │
┌────────▼─────────────────┐
│   conversations          │  Core records
│   (id INTEGER PK)        │
│   - conversation_id UK   │  ElevenLabs UUID
│   - agent_id TEXT        │
│   - user_id TEXT FK      │  → users.user_id
│   - start_time TIMESTAMP │
│   - end_time TIMESTAMP   │
│   - duration_secs INT    │
│   - status TEXT          │
│   - metadata JSON        │
└────────┬─────────────────┘
         │ 1:N
┌────────▼─────────────────┐
│   conversation_turns     │  Transcript
│   (id INTEGER PK)        │
│   - conversation_fk FK   │  → conversations.id
│   - role TEXT            │  user|agent
│   - message TEXT         │
│   - timestamp INT        │
│   - turn_order INT       │
└──────────────────────────┘

┌──────────────────────────┐
│   conversation_analysis  │  AI insights (1:1)
│   (id INTEGER PK)        │
│   - conversation_fk FK   │  → conversations.id
│   - summary TEXT         │
│   - call_successful BOOL │
│   - evaluation_json JSON │
│   - data_collection JSON │
└──────────────────────────┘

┌─────────────────────────────────────────────┐
│         OPENMEMORY DOMAIN                   │
└─────────────────────────────────────────────┘

┌──────────────────────────┐
│   users                  │
│   (user_id TEXT PK)      │
│   - summary TEXT         │  Auto-generated
│   - total_memories INT   │
│   - created_at TIMESTAMP │
└────────┬─────────────────┘
         │ 1:N
┌────────▼─────────────────────┐     ┌─────────────────────┐
│   memories                   │     │   memory_links      │
│   (id INTEGER PK)            │◄─N:N┤   (id INTEGER PK)   │
│   - user_id TEXT FK          │     │   - source_id FK    │
│   - content TEXT             │     │   - target_id FK    │
│   - sector_type TEXT         │     │   - link_weight     │
│   - salience REAL            │     │   - link_type TEXT  │
│   - recency TIMESTAMP        │     └─────────────────────┘
│   - decay_rate REAL          │
│   - state TEXT               │  active|paused|archived
│   - source_conversation TEXT │  Optional conv_id
│   - metadata JSON            │
└────────┬─────────────────────┘
         │ 1:N
┌────────▼─────────────────────┐
│   memory_embeddings          │  Vector storage
│   (id INTEGER PK)            │
│   - memory_id INT FK         │  → memories.id
│   - sector TEXT              │  semantic|episodic|etc.
│   - embedding BLOB           │  Serialized vector
│   - embedding_model TEXT     │
│   - dimension INT            │
└──────────────────────────────┘

┌──────────────────────────┐
│   conversation_memories  │  Junction table
│   (id INTEGER PK)        │
│   - conversation_fk FK   │  → conversations.id
│   - memory_id FK         │  → memories.id
│   - extraction_type TEXT │  auto|manual|reinforced
│   - confidence REAL      │
└──────────────────────────┘

┌──────────────────────────┐
│   tool_invocations       │  Server tool call logs
│   (id INTEGER PK)        │
│   - conversation_fk FK   │  → conversations.id
│   - tool_name TEXT       │  query_user_memory, etc.
│   - request_params JSON  │
│   - response_data JSON   │
│   - latency_ms INT       │
│   - timestamp TIMESTAMP  │
└──────────────────────────┘
```

### SQL Schema Definitions

```sql
-- Core tables identical to previous version
-- Adding new table for server tool tracking:

CREATE TABLE tool_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_fk INTEGER NOT NULL,
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

## Complete Data Flow Architecture

### Flow 1: Inbound Call with Memory Context

```
┌─────────────┐
│ User Calls  │
│ Twilio #    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ ElevenLabs: Incoming call detected          │
└──────┬──────────────────────────────────────┘
       │ During connection/dialing period
       ▼
┌─────────────────────────────────────────────┐
│ ElevenLabs → Your Initiation Webhook        │
│ POST /webhook/conversation-init             │
│ {caller_id, agent_id, called_number, ...}  │
└──────┬──────────────────────────────────────┘
       │ <2s response time
       ▼
┌─────────────────────────────────────────────┐
│ Your API:                                   │
│ 1. Map caller_id → user_id                 │
│ 2. Query OpenMemory:                        │
│    SELECT * FROM memories                   │
│    WHERE user_id = ? AND state = 'active'  │
│    ORDER BY composite_score DESC LIMIT 5   │
│ 3. Format memories as string               │
│ 4. Return dynamic_variables                │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Response: {                                 │
│   "dynamic_variables": {                    │
│     "user_id": "123",                       │
│     "memory_context": "Last called about..." │
│     "recent_topics": "billing, support"     │
│   },                                        │
│   "conversation_config_override": {         │
│     "agent": {                              │
│       "first_message": "Hi! ..."           │
│     }                                       │
│   }                                         │
│ }                                           │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ ElevenLabs: Agent starts with context      │
│ System prompt populated with:               │
│ {{user_id}}, {{memory_context}}, etc.      │
└─────────────────────────────────────────────┘
```

### Flow 2: Real-Time Memory Query During Conversation

```
┌─────────────────────────────────────────────┐
│ Conversation in progress                    │
│ User: "What did I tell you about my CRM?"  │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Agent LLM: Determines tool needed           │
│ - Analyzes query: requires memory context   │
│ - Selects tool: query_user_memory          │
│ - Extracts params: query="user CRM system"  │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Agent → Server Tool Call                    │
│ POST /tools/memory-query                    │
│ {                                           │
│   "query": "user CRM system preferences",   │
│   "user_id": "{{user_id}}",  // Auto-fill  │
│   "limit": 3                                │
│ }                                           │
└──────┬──────────────────────────────────────┘
       │ Target: <500ms
       ▼
┌─────────────────────────────────────────────┐
│ Your Memory Query API:                      │
│ 1. Generate query embedding                 │
│ 2. Vector similarity search                 │
│ 3. Composite scoring:                       │
│    0.6×sim + 0.2×salience + 0.1×recency +  │
│    0.1×link_weight                          │
│ 4. Single-waypoint expansion (1-hop)       │
│ 5. Format response                          │
│ 6. Log to tool_invocations table           │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Response: {                                 │
│   "memories": [                             │
│     {                                       │
│       "content": "Uses Salesforce CRM",    │
│       "sector": "procedural",              │
│       "salience": 0.9                       │
│     },                                      │
│     {                                       │
│       "content": "Needs API integration",  │
│       "sector": "semantic",                │
│       "salience": 0.8                       │
│     }                                       │
│   ],                                        │
│   "summary": "User's company uses Salesforce │
│     CRM and needs API integration support"  │
│ }                                           │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Agent receives tool response                │
│ - Appends to conversation context           │
│ - dynamic_variable {{retrieved_memories}}   │
│   updated                                   │
│ - Agent formulates response                 │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Agent: "I see you use Salesforce CRM and   │
│ need API integration. Let me help with..."  │
└─────────────────────────────────────────────┘
```

### Flow 3: Post-Call Memory Extraction

```
┌─────────────────────────────────────────────┐
│ Conversation ends                           │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ ElevenLabs → Post-Call Webhook             │
│ POST /webhook/post-call                     │
│ {                                           │
│   webhook_event_id, event_type,            │
│   data: {conversation_id, transcript, ...}  │
│ }                                           │
└──────┬──────────────────────────────────────┘
       │ <100ms acknowledgment
       ▼
┌─────────────────────────────────────────────┐
│ Webhook Handler:                            │
│ 1. Verify HMAC signature                    │
│ 2. INSERT INTO webhook_events               │
│    SET status='pending'                     │
│ 3. Return HTTP 200                          │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Background Worker (async, 1-5s):           │
│ 1. Poll webhook_events WHERE pending       │
│ 2. BEGIN TRANSACTION                        │
│ 3. Store conversation + turns               │
│ 4. Extract memories via LLM                 │
│ 5. Generate embeddings                      │
│ 6. Build memory links                       │
│ 7. Update user summary                      │
│ 8. COMMIT                                   │
│ 9. Mark webhook complete                    │
└─────────────────────────────────────────────┘
```

## Service Implementation: API Endpoints

### 1. Conversation Initiation Webhook

```python
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import time

app = FastAPI()

class InitiationRequest(BaseModel):
    caller_id: str
    agent_id: str
    called_number: str
    call_sid: str

@app.post("/webhook/conversation-init")
async def conversation_initiation(request: Request):
    start_time = time.time()
    
    # Parse request
    data = await request.json()
    caller_id = data.get('caller_id')
    agent_id = data.get('agent_id')
    
    # Map caller to user
    user = get_user_by_phone(caller_id)
    if not user:
        user_id = create_guest_user(caller_id)
    else:
        user_id = user.user_id
    
    # Query OpenMemory for top memories
    memories = query_memories_for_initiation(
        user_id=user_id,
        query="recent conversation topics and preferences",
        limit=5
    )
    
    # Format memory context
    memory_context = format_memories_for_agent(memories)
    recent_topics = ", ".join([m.get('topic') for m in memories if m.get('topic')])
    
    # Construct response
    response = {
        "dynamic_variables": {
            "user_id": user_id,
            "user_name": user.name if user else "there",
            "memory_context": memory_context,
            "recent_topics": recent_topics,
            "last_interaction": memories[0]['timestamp'] if memories else None
        },
        "conversation_config_override": {
            "agent": {
                "first_message": f"Hi{' ' + user.name if user else ''}! How can I help you today?",
                "language": "en"
            }
        }
    }
    
    # Log latency (should be <2s)
    latency = (time.time() - start_time) * 1000
    log_initiation_latency(agent_id, latency)
    
    if latency > 2000:
        logger.warning(f"Slow initiation webhook: {latency}ms")
    
    return response

def query_memories_for_initiation(user_id: str, query: str, limit: int = 5):
    """
    Fast memory query for conversation initiation.
    Target: <500ms for top-5 memories
    """
    
    query_embedding = generate_embedding(query)  # Uses LRU cache - see architecture-review-and-recommendations.md
    
    # Vector similarity with composite scoring
    memories = db.execute("""
        SELECT 
            m.id,
            m.content,
            m.sector_type,
            m.salience,
            m.recency,
            m.metadata,
            (
                0.6 * cosine_similarity(me.embedding, ?) +
                0.2 * m.salience +
                0.1 * recency_score(m.recency) +
                0.1 * COALESCE(ml.avg_link_weight, 0)
            ) AS score
        FROM memories m
        JOIN memory_embeddings me ON m.id = me.memory_id
        LEFT JOIN (
            SELECT source_memory_id, AVG(link_weight) as avg_link_weight
            FROM memory_links GROUP BY source_memory_id
        ) ml ON m.id = ml.source_memory_id
        WHERE m.user_id = ?
          AND m.state = 'active'
          AND me.sector IN ('semantic', 'episodic', 'procedural')
        ORDER BY score DESC
        LIMIT ?
    """, (query_embedding, user_id, limit))
    
    return memories.fetchall()

def format_memories_for_agent(memories):
    """Format memories as natural language for agent context"""
    if not memories:
        return "No previous conversation history."
    
    formatted = []
    for mem in memories:
        formatted.append(f"- {mem['content']} (from {mem['recency']})")
    
    return "\n".join(formatted)
```

### 2. Server Tool: Memory Query

```python
from fastapi import FastAPI
from pydantic import BaseModel
import time

class MemoryQueryRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 3
    sectors: list[str] = None  # Optional sector filter

@app.post("/tools/memory-query")
async def memory_query_tool(request: MemoryQueryRequest):
    """
    Real-time memory query server tool for ElevenLabs agent.
    Target latency: <500ms (p95), <1000ms (p99)

    NOTE: Uses LRU-cached embedding generation for optimal performance.
    See architecture-review-and-recommendations.md for cache implementation.
    """
    start_time = time.time()

    # Get conversation_id from headers if available
    conversation_id = request.headers.get('X-Conversation-ID')

    # Generate embedding (LRU cached - 30-50% hit rate in production)
    query_embedding = generate_embedding(request.query)
    
    # Determine relevant sectors if not specified
    sectors = request.sectors or classify_query_sectors(request.query)
    
    # Vector search with single-waypoint expansion
    initial_matches = vector_search(
        embedding=query_embedding,
        user_id=request.user_id,
        sectors=sectors,
        limit=request.limit * 2  # Over-fetch for expansion
    )
    
    # Expand via memory links (1-hop)
    expanded_memories = expand_via_links(initial_matches, max_total=request.limit)
    
    # Format response
    memories = []
    for mem in expanded_memories[:request.limit]:
        memories.append({
            "content": mem['content'],
            "sector": mem['sector_type'],
            "salience": mem['salience'],
            "timestamp": mem['recency'].isoformat() if mem['recency'] else None,
            "confidence": mem['score']
        })
    
    # Generate natural language summary
    summary = generate_memory_summary(memories, request.query)
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Log tool invocation
    log_tool_invocation(
        conversation_id=conversation_id,
        tool_name="memory_query",
        request_params=request.dict(),
        response_data={"memory_count": len(memories)},
        latency_ms=latency_ms
    )
    
    # Alert if slow
    if latency_ms > 1000:
        logger.warning(f"Slow memory query: {latency_ms}ms")
    
    return {
        "memories": memories,
        "summary": summary,
        "query_time_ms": latency_ms
    }

def vector_search(embedding, user_id, sectors, limit):
    """Core vector similarity search"""
    return db.execute("""
        SELECT 
            m.*,
            (
                0.6 * cosine_similarity(me.embedding, ?) +
                0.2 * m.salience +
                0.1 * recency_score(m.recency)
            ) AS score
        FROM memories m
        JOIN memory_embeddings me ON m.id = me.memory_id
        WHERE m.user_id = ?
          AND m.state = 'active'
          AND me.sector IN ({})
        ORDER BY score DESC
        LIMIT ?
    """.format(','.join(['?']*len(sectors))), 
    (embedding, user_id, *sectors, limit)).fetchall()

def expand_via_links(initial_matches, max_total=5):
    """Single-waypoint graph expansion"""
    expanded = list(initial_matches)
    seen_ids = {m['id'] for m in initial_matches}
    
    for match in initial_matches:
        if len(expanded) >= max_total:
            break
            
        # Get top-weighted links
        links = db.execute("""
            SELECT m.*, ml.link_weight
            FROM memory_links ml
            JOIN memories m ON ml.target_memory_id = m.id
            WHERE ml.source_memory_id = ?
              AND m.state = 'active'
              AND m.id NOT IN ({})
            ORDER BY ml.link_weight DESC
            LIMIT 2
        """.format(','.join(['?']*len(seen_ids))), 
        (match['id'], *seen_ids)).fetchall()
        
        for link in links:
            if len(expanded) >= max_total:
                break
            # Boost score by link weight
            link['score'] = match['score'] * 0.8 + link['link_weight'] * 0.2
            expanded.append(link)
            seen_ids.add(link['id'])
    
    # Re-sort by score
    expanded.sort(key=lambda x: x['score'], reverse=True)
    return expanded[:max_total]

def generate_memory_summary(memories, query):
    """Generate natural language summary using LLM"""
    if not memories:
        return "No relevant memories found."
    
    # Use lightweight LLM for fast summary
    prompt = f"""Summarize these memories in 1-2 sentences to answer: "{query}"

Memories:
{chr(10).join([f"- {m['content']}" for m in memories])}

Summary:"""
    
    return llm_generate(prompt, max_tokens=100, model="gpt-3.5-turbo")
```

### 3. Post-Call Webhook Handler

```python
import hmac
import hashlib
import json

@app.post("/webhook/post-call")
async def post_call_webhook(request: Request):
    """
    Handle ElevenLabs post-call webhook.
    MUST respond <100ms to avoid retries.
    """
    # Verify HMAC signature
    signature = request.headers.get('ElevenLabs-Signature')
    body = await request.body()
    
    if not verify_elevenlabs_signature(signature, body, WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Parse payload
    payload = await request.json()
    event_id = payload['webhook_event_id']
    event_type = payload['event_type']
    
    # Idempotent insert (prevents duplicate processing)
    try:
        db.execute("""
            INSERT INTO webhook_events (event_id, event_type, payload, conversation_id, status)
            VALUES (?, ?, ?, ?, 'pending')
        """, (
            event_id,
            event_type,
            json.dumps(payload),
            payload['data'].get('conversation_id'),
            'pending'
        ))
        db.commit()
    except sqlite3.IntegrityError:
        # Already processed
        logger.info(f"Duplicate webhook {event_id}, returning 200")
    
    # Return immediately (CRITICAL)
    return Response(status_code=200)

def verify_elevenlabs_signature(signature_header: str, body: bytes, secret: str) -> bool:
    """
    Verify HMAC-SHA256 signature from ElevenLabs.
    Format: "t=1234567890,v1=abc123..."
    """
    try:
        parts = dict(item.split('=') for item in signature_header.split(','))
        timestamp = parts['t']
        received_sig = parts['v1']
        
        # Check timestamp freshness (5 min window)
        if abs(time.time() - int(timestamp)) > 300:
            return False
        
        # Compute expected signature
        payload = f"{timestamp}.{body.decode('utf-8')}"
        expected_sig = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_sig, received_sig)
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        return False

# Background worker for async processing
async def webhook_processor_worker():
    """
    Polls webhook_events table and processes pending webhooks.
    Runs 2-3 workers in parallel for 100 conversations/day.
    """
    while True:
        # Fetch pending events
        events = db.execute("""
            SELECT * FROM webhook_events 
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 5
        """).fetchall()
        
        for event in events:
            try:
                # Mark processing
                db.execute("UPDATE webhook_events SET status='processing' WHERE event_id=?", 
                          (event['event_id'],))
                db.commit()
                
                # Process with full transaction
                await process_webhook_event(event)
                
                # Mark complete
                db.execute("""
                    UPDATE webhook_events 
                    SET status='completed', processed_at=CURRENT_TIMESTAMP
                    WHERE event_id=?
                """, (event['event_id'],))
                db.commit()
                
            except Exception as e:
                logger.error(f"Error processing webhook {event['event_id']}: {e}")
                
                # Retry logic
                retry_count = event['retry_count'] + 1
                if retry_count > 5:
                    db.execute("""
                        UPDATE webhook_events 
                        SET status='dead_letter', error_message=?
                        WHERE event_id=?
                    """, (str(e), event['event_id']))
                else:
                    db.execute("""
                        UPDATE webhook_events 
                        SET status='pending', retry_count=?
                        WHERE event_id=?
                    """, (retry_count, event['event_id']))
                db.commit()
                
                # Exponential backoff
                await asyncio.sleep(2 ** retry_count)
        
        # Poll every 500ms
        await asyncio.sleep(0.5)

async def process_webhook_event(event):
    """
    Full processing of webhook: store conversation, extract memories.
    Runs in transaction for atomicity.
    """
    payload = json.loads(event['payload'])
    data = payload['data']
    
    # Start transaction
    with db.begin():
        # 1. Store conversation
        conversation_id = db.execute("""
            INSERT INTO conversations (
                conversation_id, agent_id, user_id, 
                start_time, end_time, duration_secs, status, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """, (
            data['conversation_id'],
            data['agent_id'],
            data.get('conversation_initiation_client_data', {}).get('dynamic_variables', {}).get('user_id'),
            data['metadata']['start_time'],
            data['metadata']['end_time'],
            data['metadata']['duration_secs'],
            data['status'],
            json.dumps(data['metadata'])
        )).fetchone()[0]
        
        # 2. Store transcript turns
        for i, turn in enumerate(data.get('transcript', [])):
            db.execute("""
                INSERT INTO conversation_turns (
                    conversation_fk, role, message, timestamp, turn_order
                )
                VALUES (?, ?, ?, ?, ?)
            """, (conversation_id, turn['role'], turn['message'], turn['timestamp'], i))
        
        # 3. Store analysis
        if 'analysis' in data:
            db.execute("""
                INSERT INTO conversation_analysis (
                    conversation_fk, summary, call_successful, 
                    evaluation_json, data_collection
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation_id,
                data['analysis'].get('transcript_summary'),
                data['analysis'].get('call_successful'),
                json.dumps(data['analysis'].get('evaluation_results', {})),
                json.dumps(data['analysis'].get('data_collection', {}))
            ))
        
        # 4. Extract and store memories
        user_id = data.get('conversation_initiation_client_data', {}).get('dynamic_variables', {}).get('user_id')
        if user_id:
            memories = await extract_memories_from_transcript(
                user_id=user_id,
                transcript=data.get('transcript', []),
                conversation_id=conversation_id
            )
            
            # Link memories to conversation
            for memory_id in memories:
                db.execute("""
                    INSERT INTO conversation_memories (
                        conversation_fk, memory_id, extraction_type
                    )
                    VALUES (?, ?, 'auto')
                """, (conversation_id, memory_id))
        
        # Transaction commits automatically

async def extract_memories_from_transcript(user_id, transcript, conversation_id):
    """
    Use LLM to extract structured memories from transcript.
    Returns list of created memory IDs.
    """
    # Concatenate transcript
    full_text = "\n".join([f"{t['role']}: {t['message']}" for t in transcript])
    
    # LLM extraction
    extraction_prompt = f"""Analyze this conversation and extract key information into structured memory categories.

CONVERSATION:
{full_text}

Extract memories into these categories:
1. SEMANTIC: Facts, preferences, user attributes
2. EPISODIC: Specific events, actions taken
3. PROCEDURAL: How-to knowledge, processes mentioned
4. EMOTIONAL: Sentiment, satisfaction level
5. REFLECTIVE: Patterns, behavioral insights

For each memory, provide:
- category (semantic|episodic|procedural|emotional|reflective)
- content (concise, <100 chars)
- salience (0.0-1.0)

Output as JSON array."""

    response = await llm_generate(
        prompt=extraction_prompt,
        response_format={"type": "json_object"},
        model="gpt-4o-mini"
    )
    
    extracted = json.loads(response)['memories']
    
    # Create memory nodes
    memory_ids = []
    for fact in extracted:
        # Insert memory
        memory_id = db.execute("""
            INSERT INTO memories (
                user_id, content, sector_type, salience, source_conversation
            )
            VALUES (?, ?, ?, ?, ?)
            RETURNING id
        """, (
            user_id,
            fact['content'],
            fact['category'],
            fact['salience'],
            conversation_id
        )).fetchone()[0]
        
        # Generate embedding (LRU cached)
        if fact['category'] in ['semantic', 'episodic', 'procedural']:
            embedding = await generate_embedding(fact['content'])  # LRU cached
            
            db.execute("""
                INSERT INTO memory_embeddings (memory_id, sector, embedding, dimension)
                VALUES (?, ?, ?, ?)
            """, (memory_id, fact['category'], embedding, len(embedding)))
            
            # Build automatic links to similar memories
            similar = await find_similar_memories(user_id, embedding, threshold=0.8, limit=3)
            for sim_id, similarity in similar:
                db.execute("""
                    INSERT INTO memory_links (source_memory_id, target_memory_id, link_weight)
                    VALUES (?, ?, ?)
                """, (memory_id, sim_id, similarity))
        
        memory_ids.append(memory_id)
    
    # Update user summary
    await regenerate_user_summary(user_id)
    
    return memory_ids
```

## ElevenLabs Agent Configuration

### System Prompt with Dynamic Variables

```
You are a helpful AI assistant with access to the user's conversation history and preferences.

USER CONTEXT:
- User ID: {{user_id}}
- Name: {{user_name}}
- Previous Interactions: {{memory_context}}
- Recent Topics: {{recent_topics}}

CAPABILITIES:
You have access to the following tools:
1. query_user_memory: Search the user's past conversations and preferences
   - Use this when the user references past discussions
   - Use this when you need more context about their preferences or history
   
2. [other tools as needed]

INSTRUCTIONS:
- Reference the user's name naturally in conversation
- Use the memory_context to provide personalized, contextually relevant responses
- When the user mentions something from a past conversation, use query_user_memory to retrieve specific details
- Always verify information before making assumptions based on memory
- If memory is uncertain, ask clarifying questions

EXAMPLE USAGE:
User: "What did we discuss last time?"
You: [Call query_user_memory tool with query="last conversation topics"]
Then respond based on the retrieved memories.
```

### Tool Definitions in Agent Config

```json
{
  "tools": [
    {
      "type": "webhook",
      "name": "query_user_memory",
      "description": "Search the user's conversation history and preferences. Use when the user references past conversations or when you need historical context.",
      "url": "https://your-api.com/tools/memory-query",
      "method": "POST",
      "authentication": {
        "type": "bearer_token",
        "secret_name": "api_token"
      },
      "body_parameters": [
        {
          "name": "query",
          "type": "string",
          "description": "Natural language query describing what to retrieve (e.g. 'user's CRM preferences', 'last discussion about billing')",
          "required": true
        },
        {
          "name": "user_id",
          "type": "string",
          "description": "User identifier",
          "required": true,
          "value": "{{user_id}}"
        },
        {
          "name": "limit",
          "type": "integer",
          "description": "Maximum memories to return (1-10)",
          "required": false,
          "value": 3
        }
      ],
      "dynamic_variable_assignment": [
        {
          "source_path": "$.summary",
          "target_variable": "retrieved_memory_summary"
        }
      ]
    }
  ]
}
```

## Performance Targets and Monitoring

### Latency SLAs

| Component | Target (p95) | Target (p99) | Critical Threshold |
|-----------|--------------|--------------|-------------------|
| Initiation Webhook | <1000ms | <1800ms | 2000ms |
| Server Tool Query | <500ms | <1000ms | 1500ms |
| Post-Call Webhook Ack | <50ms | <100ms | 200ms |
| Memory Extraction (async) | <3000ms | <5000ms | 10000ms |

### Key Metrics

```python
metrics = {
    # Initiation webhooks
    "initiation_webhook_latency_p95": histogram("initiation.latency"),
    "initiation_webhook_failures": counter("initiation.failures"),
    
    # Server tools
    "tool_query_latency_p95": histogram("tool.memory_query.latency"),
    "tool_query_rate": counter("tool.memory_query.calls"),
    "tool_query_errors": counter("tool.memory_query.errors"),

    # Embedding cache (LRU) - see architecture-review-and-recommendations.md
    "embedding_cache_hit_rate": gauge("embedding.cache.hit_rate"),
    "embedding_cache_size": gauge("embedding.cache.size"),
    "embedding_cache_evictions": counter("embedding.cache.evictions"),
    "embedding_generation_latency": histogram("embedding.generation.latency"),
    "embedding_api_calls": counter("embedding.api.calls"),
    
    # Post-call processing
    "webhook_queue_depth": gauge("webhook.queue.pending"),
    "webhook_processing_time_p95": histogram("webhook.processing_time"),
    "memory_extraction_success_rate": gauge("memory.extraction.success_rate"),
    
    # Memory system
    "total_memories": gauge("memory.total"),
    "memory_query_accuracy": gauge("memory.retrieval.relevance_score"),
    "average_memories_per_user": gauge("memory.per_user.avg"),
    
    # Database
    "db_query_latency_p95": histogram("db.query.latency"),
    "wal_size_mb": gauge("db.wal.size_mb"),
    "db_connections": gauge("db.connections.active")
}
```

### Alerts

```yaml
alerts:
  - name: slow_initiation_webhook
    condition: initiation_webhook_latency_p95 > 1500ms for 5 minutes
    severity: warning
    action: Scale up resources, check memory query performance
    
  - name: high_tool_latency
    condition: tool_query_latency_p95 > 800ms for 5 minutes
    severity: warning
    action: Add caching layer, optimize vector search
    
  - name: webhook_queue_backlog
    condition: webhook_queue_depth > 50 for 10 minutes
    severity: critical
    action: Scale workers, investigate processing bottleneck
    
  - name: memory_extraction_failures
    condition: memory_extraction_success_rate < 0.90 for 15 minutes
    severity: critical
    action: Check LLM API, review extraction prompts
```

## Deployment Architecture for 100 Conversations/Day

```
┌────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
│                  (Nginx / Caddy)                        │
└────────┬───────────────────────────────┬───────────────┘
         │                               │
         ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  Webhook Handler    │         │  Tool API           │
│  (FastAPI)          │         │  (FastAPI)          │
│  - Initiation       │         │  - memory_query     │
│  - Post-call        │         │  - [other tools]    │
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

### Infrastructure Requirements

**For 100 conversations/day:**
- CPU: 2 vCPU
- RAM: 4GB
- Disk: 20GB SSD
- Network: Minimal (<1GB/month)

**Scaling to 1000 conversations/day:**
- CPU: 4 vCPU
- RAM: 8GB
- Workers: 10-15 async workers
- Consider: Redis for queue, PostgreSQL for database
- Add: Memory query caching layer (Redis)

## Conclusion

This corrected architecture leverages ElevenLabs' **three-phase integration model**:

1. **Conversation Initiation Webhooks** provide initial memory context (<2s)
2. **Server Tools** enable real-time memory queries during conversations (<500ms)
3. **Post-Call Webhooks** extract new memories asynchronously (1-5s)

The key architectural insight is that **server tools enable dynamic memory access**, allowing the agent to retrieve specific memories on-demand rather than frontloading all context. This results in more relevant responses, lower latency at conversation start, and better scalability.

The complete ERD, API implementations, and deployment specs provide a production-ready blueprint for 100-1000 conversations/day with SQLite, with clear upgrade paths to Redis/PostgreSQL for higher scales.

**Performance Optimization**: All `generate_embedding()` calls use LRU (Least Recently Used) caching to minimize API latency and costs. See [architecture-review-and-recommendations.md](./architecture-review-and-recommendations.md) for implementation details and performance impact analysis.
