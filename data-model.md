# ELAAAM Data Model
## Comprehensive Entity Relationship & Layer Architecture

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture Layers](#architecture-layers)
3. [Layer 1: Database Schema (Persistence)](#layer-1-database-schema-persistence)
4. [Layer 2: Service Models (Business Logic)](#layer-2-service-models-business-logic)
5. [Layer 3: UI Data Models (Presentation)](#layer-3-ui-data-models-presentation)
6. [Data Flow](#data-flow)
7. [Entity Relationships](#entity-relationships)

---

## Overview

ELAAAM (ElevenLabs + OpenMemory AI Agent Application) uses a **three-tier data architecture**:

- **Database Layer**: SQLite schema with 11 normalized tables
- **Service Layer**: Python domain models and DTOs for business logic
- **UI Layer**: Simplified, aggregated models for frontend consumption

This document provides complete entity definitions across all three layers and their relationships.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│            UI Data Models (Layer 3)             │
│   Presentation layer - simplified, aggregated   │
└────────────────┬────────────────────────────────┘
                 │ Transform/Serialize
┌────────────────▼────────────────────────────────┐
│         Service Models (Layer 2)                │
│   Business logic layer - domain entities        │
└────────────────┬────────────────────────────────┘
                 │ ORM Mapping
┌────────────────▼────────────────────────────────┐
│         Database Schema (Layer 1)               │
│   Persistence layer - SQLite tables             │
└─────────────────────────────────────────────────┘
```

---

## Layer 1: Database Schema (Persistence)

### Schema Organization

The database is organized into three domains:

1. **Integration Domain**: Webhook event processing
2. **Conversation Domain**: Conversation lifecycle and analysis
3. **Memory Domain**: User memory graph and embeddings

### 1.1 Integration Domain

#### Table: `webhook_events`
Event queue for async webhook processing.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `event_id` | TEXT | UNIQUE, NOT NULL | ElevenLabs event UUID (idempotency key) |
| `event_type` | TEXT | NOT NULL | `post_call_transcription`, `audio`, `failure` |
| `payload` | JSON | NOT NULL | Full webhook body |
| `conversation_id` | TEXT | FOREIGN KEY → conversations | Links to conversation |
| `status` | TEXT | NOT NULL | `pending`, `processing`, `completed`, `failed` |
| `retry_count` | INTEGER | DEFAULT 0 | Number of retry attempts |
| `error_message` | TEXT | NULL | Error details if failed |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Webhook received time |
| `processed_at` | TIMESTAMP | NULL | Processing completion time |

**Indexes**:
- `idx_webhook_status_created` on `(status, created_at)`
- `idx_webhook_event_id` on `(event_id)` [UNIQUE]

**Relations**:
- `conversation_id` → `conversations.conversation_id` (CASCADE DELETE)

---

### 1.2 Conversation Domain

#### Table: `conversations`
Core conversation records.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `conversation_id` | TEXT | UNIQUE, NOT NULL | ElevenLabs conversation UUID |
| `agent_id` | TEXT | NOT NULL | ElevenLabs agent ID |
| `user_id` | TEXT | FOREIGN KEY → users | User identifier |
| `start_time` | TIMESTAMP | NOT NULL | Conversation start time |
| `end_time` | TIMESTAMP | NULL | Conversation end time |
| `duration_secs` | INTEGER | NULL | Duration in seconds |
| `status` | TEXT | NOT NULL | `active`, `done`, `failed` |
| `metadata` | JSON | NULL | Custom metadata (caller_id, etc.) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

**Indexes**:
- `idx_conv_user_id` on `(user_id)`
- `idx_conv_conversation_id` on `(conversation_id)` [UNIQUE]
- `idx_conv_start_time` on `(start_time DESC)`

**Relations**:
- `user_id` → `users.user_id` (CASCADE DELETE)

---

#### Table: `conversation_turns`
Turn-by-turn conversation transcript.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `conversation_fk` | INTEGER | FOREIGN KEY → conversations | Parent conversation |
| `role` | TEXT | NOT NULL | `user` or `agent` |
| `message` | TEXT | NOT NULL | Message content |
| `timestamp` | INTEGER | NOT NULL | Unix timestamp |
| `turn_order` | INTEGER | NOT NULL | Sequential order in conversation |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation time |

**Indexes**:
- `idx_turns_conversation` on `(conversation_fk, turn_order)`

**Relations**:
- `conversation_fk` → `conversations.id` (CASCADE DELETE)

---

#### Table: `conversation_analysis`
AI-generated conversation insights.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `conversation_fk` | INTEGER | FOREIGN KEY → conversations | Parent conversation (UNIQUE) |
| `summary` | TEXT | NULL | LLM-generated summary |
| `call_successful` | BOOLEAN | NULL | Success indicator |
| `evaluation_json` | JSON | NULL | Evaluation criteria results |
| `data_collection` | JSON | NULL | Action items, sentiment, etc. |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Analysis creation time |

**Indexes**:
- `idx_analysis_conversation` on `(conversation_fk)` [UNIQUE]

**Relations**:
- `conversation_fk` → `conversations.id` (CASCADE DELETE)

---

#### Table: `tool_invocations`
Server tool call logs during conversations.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `conversation_fk` | INTEGER | FOREIGN KEY → conversations | Parent conversation |
| `tool_name` | TEXT | NOT NULL | Name of invoked tool (e.g., `query_user_memory`) |
| `request_params` | JSON | NOT NULL | Tool input parameters |
| `response_data` | JSON | NULL | Tool response |
| `latency_ms` | INTEGER | NULL | Response time in milliseconds |
| `success` | BOOLEAN | NOT NULL | Success/failure indicator |
| `error_message` | TEXT | NULL | Error details if failed |
| `timestamp` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Invocation time |

**Indexes**:
- `idx_tools_conversation` on `(conversation_fk, timestamp)`
- `idx_tools_latency` on `(latency_ms)` where `success = 1`

**Relations**:
- `conversation_fk` → `conversations.id` (CASCADE DELETE)

---

### 1.3 Memory Domain

#### Table: `users`
User profiles with auto-generated summaries.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `user_id` | TEXT | PRIMARY KEY | Unique user identifier |
| `summary` | TEXT | NULL | Auto-generated user summary |
| `total_memories` | INTEGER | DEFAULT 0 | Count of active memories |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | User creation time |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Last update time |

**Indexes**:
- `idx_users_updated` on `(updated_at DESC)`

---

#### Table: `memories`
Memory nodes in the knowledge graph.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `user_id` | TEXT | FOREIGN KEY → users | Memory owner |
| `content` | TEXT | NOT NULL | Memory text content |
| `sector_type` | TEXT | NOT NULL | `semantic`, `episodic`, `procedural`, `emotional`, `reflective` |
| `salience` | REAL | NOT NULL | Importance score (0.0-1.0) |
| `recency` | TIMESTAMP | NOT NULL | Last accessed/reinforced time |
| `decay_rate` | REAL | DEFAULT 0.0 | Memory decay rate (0.0-1.0) |
| `state` | TEXT | DEFAULT 'active' | `active`, `paused`, `archived` |
| `source_conversation` | TEXT | NULL | Conversation ID that created memory |
| `metadata` | JSON | NULL | Custom metadata |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Creation time |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Last update time |

**Indexes**:
- `idx_memories_user_id` on `(user_id, state)`
- `idx_memories_sector` on `(sector_type)`
- `idx_memories_recency` on `(recency DESC)`
- `idx_memories_salience` on `(salience DESC)`

**Relations**:
- `user_id` → `users.user_id` (CASCADE DELETE)

---

#### Table: `memory_embeddings`
Vector embeddings for semantic search.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `memory_id` | INTEGER | FOREIGN KEY → memories | Parent memory |
| `sector` | TEXT | NOT NULL | Sector type for this embedding |
| `embedding` | BLOB | NOT NULL | Serialized vector (numpy/pickle) |
| `embedding_model` | TEXT | NOT NULL | Model name (e.g., `text-embedding-3-small`) |
| `dimension` | INTEGER | NOT NULL | Vector dimension (e.g., 1536) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Embedding creation time |

**Indexes**:
- `idx_embeddings_memory` on `(memory_id, sector)`

**Relations**:
- `memory_id` → `memories.id` (CASCADE DELETE)

---

#### Table: `memory_links`
Graph edges connecting related memories.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `source_memory_id` | INTEGER | FOREIGN KEY → memories | Source node |
| `target_memory_id` | INTEGER | FOREIGN KEY → memories | Target node |
| `link_weight` | REAL | NOT NULL | Link strength (0.0-1.0) |
| `link_type` | TEXT | NOT NULL | `associative`, `temporal`, `causal` |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Link creation time |

**Indexes**:
- `idx_links_source` on `(source_memory_id, link_weight DESC)`
- `idx_links_target` on `(target_memory_id, link_weight DESC)`

**Relations**:
- `source_memory_id` → `memories.id` (CASCADE DELETE)
- `target_memory_id` → `memories.id` (CASCADE DELETE)

---

#### Table: `conversation_memories`
Junction table: conversations ↔ memories.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `conversation_fk` | INTEGER | FOREIGN KEY → conversations | Conversation reference |
| `memory_id` | INTEGER | FOREIGN KEY → memories | Memory reference |
| `extraction_type` | TEXT | NOT NULL | `auto`, `manual`, `reinforced` |
| `confidence` | REAL | NOT NULL | Extraction confidence (0.0-1.0) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Junction creation time |

**Indexes**:
- `idx_conv_memories_conv` on `(conversation_fk)`
- `idx_conv_memories_memory` on `(memory_id)`

**Relations**:
- `conversation_fk` → `conversations.id` (CASCADE DELETE)
- `memory_id` → `memories.id` (CASCADE DELETE)

---

## Layer 2: Service Models (Business Logic)

Service models are Python classes representing domain entities with business logic. They map to database tables but include computed properties, validation, and methods.

### 2.1 Integration Domain Models

#### `WebhookEvent`
```python
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class WebhookEventType(str, Enum):
    POST_CALL_TRANSCRIPTION = "post_call_transcription"
    AUDIO = "audio"
    FAILURE = "failure"

class WebhookStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class WebhookEvent(BaseModel):
    id: Optional[int] = None
    event_id: str = Field(..., description="ElevenLabs event UUID")
    event_type: WebhookEventType
    payload: Dict[str, Any]
    conversation_id: Optional[str] = None
    status: WebhookStatus = WebhookStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True

    def mark_processing(self):
        """Transition to processing state"""
        self.status = WebhookStatus.PROCESSING

    def mark_completed(self):
        """Transition to completed state"""
        self.status = WebhookStatus.COMPLETED
        self.processed_at = datetime.utcnow()

    def mark_failed(self, error: str):
        """Transition to failed state with error"""
        self.status = WebhookStatus.FAILED
        self.error_message = error
        self.retry_count += 1

    def should_retry(self, max_retries: int = 3) -> bool:
        """Check if event should be retried"""
        return self.status == WebhookStatus.FAILED and self.retry_count < max_retries
```

---

### 2.2 Conversation Domain Models

#### `Conversation`
```python
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, computed_field

class ConversationStatus(str, Enum):
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"

class Conversation(BaseModel):
    id: Optional[int] = None
    conversation_id: str = Field(..., description="ElevenLabs conversation UUID")
    agent_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_secs: Optional[int] = None
    status: ConversationStatus
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships (loaded separately)
    turns: List['ConversationTurn'] = []
    analysis: Optional['ConversationAnalysis'] = None
    tool_invocations: List['ToolInvocation'] = []

    class Config:
        use_enum_values = True

    @computed_field
    @property
    def is_active(self) -> bool:
        """Check if conversation is still active"""
        return self.status == ConversationStatus.ACTIVE

    @computed_field
    @property
    def turn_count(self) -> int:
        """Total number of turns in conversation"""
        return len(self.turns)

    def mark_done(self):
        """Mark conversation as complete"""
        self.status = ConversationStatus.DONE
        self.end_time = datetime.utcnow()
        if self.start_time:
            self.duration_secs = int((self.end_time - self.start_time).total_seconds())
```

#### `ConversationTurn`
```python
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class TurnRole(str, Enum):
    USER = "user"
    AGENT = "agent"

class ConversationTurn(BaseModel):
    id: Optional[int] = None
    conversation_fk: int
    role: TurnRole
    message: str
    timestamp: int = Field(..., description="Unix timestamp")
    turn_order: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

    @property
    def formatted_timestamp(self) -> str:
        """Format timestamp as human-readable string"""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
```

#### `ConversationAnalysis`
```python
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class ConversationAnalysis(BaseModel):
    id: Optional[int] = None
    conversation_fk: int
    summary: Optional[str] = None
    call_successful: Optional[bool] = None
    evaluation_json: Optional[Dict[str, Any]] = None
    data_collection: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def action_items(self) -> List[str]:
        """Extract action items from data collection"""
        if self.data_collection and "action_items" in self.data_collection:
            return self.data_collection["action_items"]
        return []

    @property
    def sentiment(self) -> Optional[str]:
        """Extract customer sentiment"""
        if self.data_collection and "customer_sentiment" in self.data_collection:
            return self.data_collection["customer_sentiment"]
        return None
```

#### `ToolInvocation`
```python
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, computed_field

class ToolInvocation(BaseModel):
    id: Optional[int] = None
    conversation_fk: int
    tool_name: str
    request_params: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    latency_ms: Optional[int] = None
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def is_slow(self) -> bool:
        """Check if invocation exceeded performance threshold"""
        return self.latency_ms is not None and self.latency_ms > 1000

    @computed_field
    @property
    def latency_seconds(self) -> Optional[float]:
        """Convert latency to seconds"""
        if self.latency_ms is not None:
            return self.latency_ms / 1000.0
        return None
```

---

### 2.3 Memory Domain Models

#### `User`
```python
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, computed_field

class User(BaseModel):
    user_id: str
    summary: Optional[str] = None
    total_memories: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    memories: List['Memory'] = []
    conversations: List['Conversation'] = []

    @computed_field
    @property
    def has_memories(self) -> bool:
        """Check if user has any memories"""
        return self.total_memories > 0

    def increment_memories(self):
        """Increment memory count"""
        self.total_memories += 1
        self.updated_at = datetime.utcnow()
```

#### `Memory`
```python
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, computed_field

class MemorySector(str, Enum):
    SEMANTIC = "semantic"        # Facts, preferences, attributes
    EPISODIC = "episodic"        # Specific events and experiences
    PROCEDURAL = "procedural"    # How-to knowledge
    EMOTIONAL = "emotional"      # Sentiment, feelings
    REFLECTIVE = "reflective"    # Patterns, insights

class MemoryState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"

class Memory(BaseModel):
    id: Optional[int] = None
    user_id: str
    content: str
    sector_type: MemorySector
    salience: float = Field(..., ge=0.0, le=1.0)
    recency: datetime
    decay_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    state: MemoryState = MemoryState.ACTIVE
    source_conversation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    embeddings: List['MemoryEmbedding'] = []
    outgoing_links: List['MemoryLink'] = []
    incoming_links: List['MemoryLink'] = []

    class Config:
        use_enum_values = True

    @computed_field
    @property
    def is_active(self) -> bool:
        """Check if memory is active"""
        return self.state == MemoryState.ACTIVE

    @computed_field
    @property
    def age_days(self) -> int:
        """Days since memory was last accessed"""
        delta = datetime.utcnow() - self.recency
        return delta.days

    @computed_field
    @property
    def decayed_salience(self) -> float:
        """Calculate salience with time decay applied"""
        if self.decay_rate == 0.0:
            return self.salience
        decay_factor = (1.0 - self.decay_rate) ** self.age_days
        return self.salience * decay_factor

    def reinforce(self):
        """Reinforce memory (update recency)"""
        self.recency = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def archive(self):
        """Archive this memory"""
        self.state = MemoryState.ARCHIVED
        self.updated_at = datetime.utcnow()
```

#### `MemoryEmbedding`
```python
from datetime import datetime
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field

class MemoryEmbedding(BaseModel):
    id: Optional[int] = None
    memory_id: int
    sector: str
    embedding: bytes = Field(..., description="Serialized numpy array")
    embedding_model: str
    dimension: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def vector(self) -> np.ndarray:
        """Deserialize embedding to numpy array"""
        import pickle
        return pickle.loads(self.embedding)

    @staticmethod
    def serialize_vector(vec: np.ndarray) -> bytes:
        """Serialize numpy array to bytes"""
        import pickle
        return pickle.dumps(vec)

    def cosine_similarity(self, other_vector: np.ndarray) -> float:
        """Calculate cosine similarity with another vector"""
        vec = self.vector
        dot_product = np.dot(vec, other_vector)
        norm_product = np.linalg.norm(vec) * np.linalg.norm(other_vector)
        return dot_product / norm_product if norm_product > 0 else 0.0
```

#### `MemoryLink`
```python
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class LinkType(str, Enum):
    ASSOCIATIVE = "associative"  # Semantic association
    TEMPORAL = "temporal"        # Time-based connection
    CAUSAL = "causal"           # Cause-effect relationship

class MemoryLink(BaseModel):
    id: Optional[int] = None
    source_memory_id: int
    target_memory_id: int
    link_weight: float = Field(..., ge=0.0, le=1.0)
    link_type: LinkType
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

    @property
    def is_strong_link(self) -> bool:
        """Check if link weight is strong (>0.7)"""
        return self.link_weight > 0.7
```

#### `ConversationMemory`
```python
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class ExtractionType(str, Enum):
    AUTO = "auto"              # Automatically extracted by LLM
    MANUAL = "manual"          # Manually added by human
    REINFORCED = "reinforced"  # Existing memory reinforced

class ConversationMemory(BaseModel):
    id: Optional[int] = None
    conversation_fk: int
    memory_id: int
    extraction_type: ExtractionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

    @property
    def is_high_confidence(self) -> bool:
        """Check if extraction confidence is high (>0.8)"""
        return self.confidence > 0.8
```

---

### 2.4 Data Transfer Objects (DTOs)

DTOs are used for API request/response serialization.

#### Webhook DTOs
```python
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class InitiationWebhookRequest(BaseModel):
    """Request from ElevenLabs for conversation initiation"""
    caller_id: str
    agent_id: str
    called_number: str
    call_sid: str

class DynamicVariables(BaseModel):
    """Dynamic variables returned to ElevenLabs"""
    user_id: str
    user_name: Optional[str] = None
    memory_context: str = ""
    recent_topics: str = ""
    user_preferences: str = ""

class InitiationWebhookResponse(BaseModel):
    """Response to ElevenLabs initiation webhook"""
    dynamic_variables: DynamicVariables
    conversation_config_override: Optional[Dict[str, Any]] = None
```

#### Memory Query DTOs
```python
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class MemoryQueryRequest(BaseModel):
    """Request for server tool memory query"""
    query: str
    user_id: str
    limit: int = 3
    sector_filter: Optional[List[str]] = None

class MemoryResult(BaseModel):
    """Individual memory result"""
    content: str
    sector: str
    salience: float
    timestamp: datetime
    confidence: Optional[float] = None

class MemoryQueryResponse(BaseModel):
    """Response from memory query tool"""
    memories: List[MemoryResult]
    summary: str
    query_latency_ms: Optional[int] = None
```

#### Post-Call Webhook DTO
```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class TranscriptTurn(BaseModel):
    role: str
    message: str
    timestamp: int

class AnalysisData(BaseModel):
    transcript_summary: str
    call_successful: bool
    evaluation_results: Optional[Dict[str, Any]] = None
    data_collection: Optional[Dict[str, Any]] = None

class PostCallWebhookRequest(BaseModel):
    """Post-call webhook from ElevenLabs"""
    webhook_event_id: str
    event_type: str
    timestamp: int
    data: Dict[str, Any]  # Contains conversation_id, transcript, analysis, etc.
```

---

## Layer 3: UI Data Models (Presentation)

UI models are simplified, aggregated views designed for frontend consumption. They combine data from multiple service models and database tables.

### 3.1 Dashboard Models

#### `UserDashboard`
Aggregated view of user's conversation activity and memory stats.

```typescript
interface UserDashboard {
  userId: string;
  userName?: string;
  summary?: string;

  // Memory statistics
  memoryStats: {
    totalMemories: number;
    activeSemantic: number;
    activeEpisodic: number;
    activeProcedural: number;
    activeEmotional: number;
    activeReflective: number;
  };

  // Conversation statistics
  conversationStats: {
    totalConversations: number;
    successfulCalls: number;
    averageDuration: number;  // seconds
    lastConversationDate?: string;  // ISO 8601
  };

  // Recent activity
  recentConversations: ConversationSummary[];  // Last 5
  topMemories: MemorySummary[];  // Top 5 by salience
}
```

---

### 3.2 Conversation Models

#### `ConversationSummary`
Simplified conversation view for lists.

```typescript
interface ConversationSummary {
  conversationId: string;
  agentId: string;
  startTime: string;  // ISO 8601
  duration: number;   // seconds
  status: 'active' | 'done' | 'failed';
  turnCount: number;
  summary?: string;
  successful?: boolean;
  sentiment?: string;
}
```

#### `ConversationDetail`
Full conversation view with transcript and analysis.

```typescript
interface ConversationDetail {
  conversationId: string;
  agentId: string;
  userId: string;
  startTime: string;  // ISO 8601
  endTime?: string;   // ISO 8601
  duration?: number;  // seconds
  status: 'active' | 'done' | 'failed';

  // Transcript
  transcript: TranscriptTurn[];

  // Analysis
  analysis?: {
    summary: string;
    successful: boolean;
    sentiment?: string;
    actionItems: string[];
    evaluationScores?: Record<string, number>;
  };

  // Tool usage
  toolInvocations: ToolInvocationSummary[];

  // Extracted memories
  extractedMemories: MemorySummary[];

  // Metadata
  metadata?: Record<string, any>;
}

interface TranscriptTurn {
  role: 'user' | 'agent';
  message: string;
  timestamp: string;  // ISO 8601
  turnOrder: number;
}

interface ToolInvocationSummary {
  toolName: string;
  timestamp: string;  // ISO 8601
  latencyMs?: number;
  success: boolean;
  errorMessage?: string;
}
```

---

### 3.3 Memory Models

#### `MemorySummary`
Simplified memory view for lists.

```typescript
interface MemorySummary {
  memoryId: number;
  content: string;
  sector: 'semantic' | 'episodic' | 'procedural' | 'emotional' | 'reflective';
  salience: number;  // 0.0-1.0
  ageDays: number;
  state: 'active' | 'paused' | 'archived';
  sourceConversation?: string;
}
```

#### `MemoryDetail`
Full memory view with relationships.

```typescript
interface MemoryDetail {
  memoryId: number;
  userId: string;
  content: string;
  sector: 'semantic' | 'episodic' | 'procedural' | 'emotional' | 'reflective';
  salience: number;  // 0.0-1.0
  decayedSalience: number;  // With decay applied
  recency: string;  // ISO 8601
  ageDays: number;
  state: 'active' | 'paused' | 'archived';
  sourceConversation?: string;
  createdAt: string;  // ISO 8601
  updatedAt: string;  // ISO 8601

  // Relationships
  relatedMemories: RelatedMemory[];
  sourceConversationDetails?: ConversationSummary;

  // Metadata
  metadata?: Record<string, any>;
}

interface RelatedMemory {
  memoryId: number;
  content: string;
  linkWeight: number;  // 0.0-1.0
  linkType: 'associative' | 'temporal' | 'causal';
  direction: 'outgoing' | 'incoming';
}
```

#### `MemoryGraph`
Graph visualization of memory connections.

```typescript
interface MemoryGraph {
  userId: string;
  nodes: MemoryNode[];
  edges: MemoryEdge[];
}

interface MemoryNode {
  id: number;
  content: string;
  sector: string;
  salience: number;
  ageDays: number;
  state: string;
}

interface MemoryEdge {
  sourceId: number;
  targetId: number;
  weight: number;
  type: 'associative' | 'temporal' | 'causal';
}
```

---

### 3.4 Analytics Models

#### `PerformanceMetrics`
Real-time performance monitoring.

```typescript
interface PerformanceMetrics {
  timeRange: {
    start: string;  // ISO 8601
    end: string;    // ISO 8601
  };

  // Webhook performance
  webhooks: {
    initiationP95Ms: number;
    initiationP99Ms: number;
    postCallAckP95Ms: number;
    postCallAckP99Ms: number;
  };

  // Tool performance
  tools: {
    memoryQueryP95Ms: number;
    memoryQueryP99Ms: number;
    averageLatencyMs: number;
    successRate: number;  // 0.0-1.0
  };

  // Memory extraction
  extraction: {
    averageExtractionMs: number;
    memoriesExtractedPerCall: number;
    extractionSuccessRate: number;  // 0.0-1.0
  };

  // Conversation stats
  conversations: {
    total: number;
    successful: number;
    failed: number;
    averageDuration: number;
  };
}
```

#### `MemoryAnalytics`
Memory system analytics.

```typescript
interface MemoryAnalytics {
  userId: string;

  // Distribution by sector
  sectorDistribution: {
    semantic: number;
    episodic: number;
    procedural: number;
    emotional: number;
    reflective: number;
  };

  // Salience distribution
  salienceDistribution: {
    high: number;      // >0.7
    medium: number;    // 0.4-0.7
    low: number;       // <0.4
  };

  // Age distribution
  ageDistribution: {
    recent: number;    // <7 days
    current: number;   // 7-30 days
    old: number;       // >30 days
  };

  // Graph metrics
  graphMetrics: {
    totalNodes: number;
    totalEdges: number;
    averageConnections: number;
    isolatedNodes: number;
    stronglyConnectedClusters: number;
  };

  // Temporal trends
  memoryGrowth: TimeSeriesData[];
}

interface TimeSeriesData {
  timestamp: string;  // ISO 8601
  value: number;
}
```

---

### 3.5 Search & Query Models

#### `SearchRequest`
Unified search across conversations and memories.

```typescript
interface SearchRequest {
  query: string;
  userId?: string;
  filters?: {
    dateRange?: {
      start: string;
      end: string;
    };
    sectors?: string[];
    conversationStatus?: string[];
    minSalience?: number;
  };
  limit?: number;
  offset?: number;
}

interface SearchResponse {
  conversations: ConversationSummary[];
  memories: MemorySummary[];
  totalConversations: number;
  totalMemories: number;
}
```

---

## Data Flow

### Flow 1: Conversation Initiation

```
┌─────────────────┐
│  ElevenLabs API │
└────────┬────────┘
         │ POST /webhook/initiation
         │ {caller_id, agent_id, ...}
         ▼
┌─────────────────────────────┐
│  Layer 3: API Handler       │
│  - Parse request            │
│  - Validate HMAC            │
└────────┬────────────────────┘
         │ InitiationWebhookRequest DTO
         ▼
┌─────────────────────────────┐
│  Layer 2: Service Layer     │
│  - Identify user from       │
│    caller_id                │
│  - Query Memory domain      │
│  - Retrieve top 5 memories  │
│  - Format context           │
└────────┬────────────────────┘
         │ User, Memory models
         ▼
┌─────────────────────────────┐
│  Layer 1: Database          │
│  - SELECT FROM users        │
│  - SELECT FROM memories     │
│  - SELECT FROM embeddings   │
│  - Vector similarity search │
└────────┬────────────────────┘
         │ Query results
         ▼
┌─────────────────────────────┐
│  Layer 2: Response Builder  │
│  - Build DynamicVariables   │
│  - Create response DTO      │
└────────┬────────────────────┘
         │ InitiationWebhookResponse
         ▼
┌─────────────────────────────┐
│  Layer 3: JSON Response     │
│  {dynamic_variables: {...}} │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  ElevenLabs API │
│  Starts call    │
└─────────────────┘
```

---

### Flow 2: Real-Time Memory Query (Server Tool)

```
┌─────────────────┐
│  ElevenLabs     │
│  Agent LLM      │
└────────┬────────┘
         │ POST /tools/memory-query
         │ {query, user_id, limit}
         ▼
┌─────────────────────────────┐
│  Layer 3: Tool Handler      │
│  - Parse request            │
│  - Validate parameters      │
└────────┬────────────────────┘
         │ MemoryQueryRequest DTO
         ▼
┌─────────────────────────────┐
│  Layer 2: Query Service     │
│  - Generate query embedding │
│  - Perform vector search    │
│  - Apply composite scoring: │
│    0.6×sim + 0.2×sal +      │
│    0.1×rec + 0.1×link       │
│  - Single-waypoint expansion│
└────────┬────────────────────┘
         │ Memory models
         ▼
┌─────────────────────────────┐
│  Layer 1: Database          │
│  - Vector similarity search │
│  - JOIN memory_links        │
│  - ORDER BY composite score │
│  - LIMIT n                  │
└────────┬────────────────────┘
         │ Query results
         ▼
┌─────────────────────────────┐
│  Layer 2: Response Builder  │
│  - Format memories          │
│  - Generate summary         │
│  - Track invocation         │
└────────┬────────────────────┘
         │ MemoryQueryResponse
         ▼
┌─────────────────────────────┐
│  Layer 3: JSON Response     │
│  {memories: [...],          │
│   summary: "..."}           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  Agent continues│
│  conversation   │
└─────────────────┘
```

---

### Flow 3: Post-Call Memory Extraction

```
┌─────────────────┐
│  ElevenLabs API │
└────────┬────────┘
         │ POST /webhook/post-call
         │ {transcript, analysis, ...}
         ▼
┌─────────────────────────────┐
│  Layer 3: Webhook Handler   │
│  - Validate HMAC signature  │
│  - Quick ACK (50-100ms)     │
│  - Queue for async process  │
└────────┬────────────────────┘
         │ PostCallWebhookRequest
         ▼
┌─────────────────────────────┐
│  Layer 1: Database          │
│  - INSERT webhook_events    │
│    status='pending'         │
└────────┬────────────────────┘
         │
         ▼ (async background worker)
┌─────────────────────────────┐
│  Layer 2: Extraction Worker │
│  - Load webhook event       │
│  - UPDATE status=processing │
│  - Store conversation       │
│  - Store transcript turns   │
│  - Store analysis           │
│  - LLM memory extraction    │
│  - Generate embeddings      │
│  - Build memory links       │
│  - UPDATE status=completed  │
└────────┬────────────────────┘
         │ Multiple models
         ▼
┌─────────────────────────────────────────────┐
│  Layer 1: Database (Transaction)            │
│  - INSERT INTO conversations                │
│  - INSERT INTO conversation_turns (bulk)    │
│  - INSERT INTO conversation_analysis        │
│  - INSERT INTO memories (5 sectors)         │
│  - INSERT INTO memory_embeddings            │
│  - INSERT INTO memory_links                 │
│  - INSERT INTO conversation_memories        │
│  - UPDATE users.total_memories              │
│  - UPDATE webhook_events.status             │
└─────────────────────────────────────────────┘
```

---

## Entity Relationships

### Complete ERD (All Layers)

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 1: DATABASE                         │
└─────────────────────────────────────────────────────────────────┘

     webhook_events                    conversations
     ┌─────────────┐                  ┌──────────────┐
     │ id (PK)     │                  │ id (PK)      │
     │ event_id    │──triggers────────│ conv_id      │
     │ type        │    processing    │ agent_id     │
     │ payload     │                  │ user_id (FK) │───┐
     │ conv_id(FK) │                  │ start_time   │   │
     │ status      │                  │ end_time     │   │
     │ retry_count │                  │ duration     │   │
     └─────────────┘                  │ status       │   │
                                      └──────┬───────┘   │
                                             │           │
                    ┌────────────────────────┼───────────┼──────────┐
                    │                        │           │          │
                    ▼                        ▼           │          ▼
         conversation_turns        conversation_      │    tool_invocations
         ┌────────────────┐         analysis         │    ┌──────────────┐
         │ id (PK)        │        ┌─────────────┐   │    │ id (PK)      │
         │ conv_fk (FK)   │        │ id (PK)     │   │    │ conv_fk (FK) │
         │ role           │        │ conv_fk(FK) │   │    │ tool_name    │
         │ message        │        │ summary     │   │    │ req_params   │
         │ timestamp      │        │ successful  │   │    │ response     │
         │ turn_order     │        │ evaluation  │   │    │ latency_ms   │
         └────────────────┘        │ collection  │   │    │ success      │
                                   └─────────────┘   │    └──────────────┘
                                                      │
                                                      │
                                                      ▼
                                                    users
                                                  ┌──────────────┐
                                                  │ user_id (PK) │
                                                  │ summary      │
                                                  │ total_mems   │
                                                  │ created_at   │
                                                  │ updated_at   │
                                                  └──────┬───────┘
                                                         │
                                                         │ owns
                                                         ▼
                                                    memories
                                         ┌──────────┬──────────┬──────────┐
                                         │          │          │          │
                                         │ id (PK)  │          │          │
                                         │ user_id(FK)        │          │
                                         │ content  │          │          │
                                         │ sector   │          │          │
                                         │ salience │          │          │
                                         │ recency  │          │          │
                                         │ state    │          │          │
                                         └────┬─────┴──────┬───┴────┬─────┘
                                              │            │        │
                      ┌───────────────────────┼────────────┘        │
                      │                       │                     │
                      ▼                       ▼                     ▼
              memory_embeddings       memory_links        conversation_memories
              ┌───────────────┐      ┌──────────────┐    ┌───────────────────┐
              │ id (PK)       │      │ id (PK)      │    │ id (PK)           │
              │ memory_id(FK) │      │ source_id(FK)│    │ conv_fk (FK)      │
              │ sector        │      │ target_id(FK)│    │ memory_id (FK)    │
              │ embedding     │      │ link_weight  │    │ extraction_type   │
              │ model         │      │ link_type    │    │ confidence        │
              │ dimension     │      └──────────────┘    └───────────────────┘
              └───────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 2: SERVICE MODELS                        │
└─────────────────────────────────────────────────────────────────┘

     WebhookEvent               Conversation
     ┌─────────────┐           ┌──────────────────┐
     │ + id        │           │ + id             │
     │ + event_id  │           │ + conversation_id│
     │ + type      │◀─triggers─│ + user_id        │
     │ + status    │           │ + status         │
     │ + payload   │           │ + turns: List    │───┐
     │ + retry_cnt │           │ + analysis: Obj  │   │
     │             │           │ + tools: List    │   │
     │ Methods:    │           │                  │   │
     │ mark_proc() │           │ Methods:         │   │
     │ mark_done() │           │ mark_done()      │   │
     │ mark_fail() │           │ is_active()      │   │
     │ should_retry()          │ turn_count()     │   │
     └─────────────┘           └──────────────────┘   │
                                                       │
                                                       │
                                    ┌──────────────────┼─────────────┐
                                    │                  │             │
                                    ▼                  ▼             ▼
                         ConversationTurn   ConversationAnalysis  ToolInvocation
                         ┌──────────────┐   ┌──────────────────┐ ┌──────────────┐
                         │ + role       │   │ + summary        │ │ + tool_name  │
                         │ + message    │   │ + successful     │ │ + latency_ms │
                         │ + timestamp  │   │ + action_items() │ │ + success    │
                         │ + turn_order │   │ + sentiment()    │ │ + is_slow()  │
                         └──────────────┘   └──────────────────┘ └──────────────┘


                                      User
                                  ┌──────────────┐
                                  │ + user_id    │
                                  │ + summary    │
                                  │ + total_mems │
                                  │ + memories   │
                                  │              │
                                  │ Methods:     │
                                  │ has_mems()   │
                                  │ incr_mems()  │
                                  └──────┬───────┘
                                         │
                                         │
                                         ▼
                                     Memory
                               ┌──────────────────┐
                               │ + content        │
                               │ + sector         │
                               │ + salience       │
                               │ + recency        │
                               │ + state          │
                               │ + embeddings     │
                               │ + links          │
                               │                  │
                               │ Methods:         │
                               │ is_active()      │
                               │ age_days()       │
                               │ decayed_sal()    │
                               │ reinforce()      │
                               │ archive()        │
                               └──────┬───────────┘
                                      │
                        ┌─────────────┼─────────────┐
                        │             │             │
                        ▼             ▼             ▼
                MemoryEmbedding  MemoryLink  ConversationMemory
                ┌──────────────┐ ┌──────────┐ ┌────────────────┐
                │ + embedding  │ │ + weight │ │ + conv_fk      │
                │ + model      │ │ + type   │ │ + memory_id    │
                │ + dimension  │ │          │ │ + confidence   │
                │              │ │ Methods: │ │                │
                │ Methods:     │ │ is_strong│ │ Methods:       │
                │ vector()     │ │          │ │ is_high_conf() │
                │ cosine_sim() │ └──────────┘ └────────────────┘
                └──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 3: UI DATA MODELS                        │
└─────────────────────────────────────────────────────────────────┘

                          UserDashboard
                     ┌──────────────────────┐
                     │ + userId             │
                     │ + userName           │
                     │ + memoryStats        │────┐
                     │ + conversationStats  │    │
                     │ + recentConversations│    │
                     │ + topMemories        │    │
                     └───────────┬──────────┘    │
                                 │                │
                ┌────────────────┼────────────────┘
                │                │
                ▼                ▼
     ConversationSummary    MemorySummary
     ┌─────────────────┐   ┌──────────────┐
     │ + conversationId│   │ + memoryId   │
     │ + duration      │   │ + content    │
     │ + turnCount     │   │ + sector     │
     │ + summary       │   │ + salience   │
     │ + successful    │   │ + ageDays    │
     │ + sentiment     │   │ + state      │
     └─────────────────┘   └──────────────┘
              │
              │ detail view
              ▼
     ConversationDetail
     ┌──────────────────────┐
     │ + conversationId     │
     │ + transcript         │───► TranscriptTurn[]
     │ + analysis           │
     │ + toolInvocations    │───► ToolInvocationSummary[]
     │ + extractedMemories  │───► MemorySummary[]
     └──────────────────────┘


                    MemoryGraph
              ┌──────────────────┐
              │ + userId         │
              │ + nodes          │───► MemoryNode[]
              │ + edges          │───► MemoryEdge[]
              └──────────────────┘


                PerformanceMetrics
           ┌──────────────────────────┐
           │ + webhooks               │
           │   - initiationP95/P99    │
           │   - postCallAckP95/P99   │
           │ + tools                  │
           │   - queryP95/P99         │
           │   - successRate          │
           │ + extraction             │
           │ + conversations          │
           └──────────────────────────┘
```

---

## Key Design Patterns

### 1. Layer Separation
- **Database Layer**: Normalized, efficient storage
- **Service Layer**: Rich domain models with business logic
- **UI Layer**: Denormalized, pre-aggregated for performance

### 2. DTO Pattern
- DTOs for API boundaries (request/response)
- Service models for internal logic
- Clear transformation between layers

### 3. Computed Properties
- Service models include computed fields (e.g., `is_active`, `age_days`)
- Reduces duplication and ensures consistency
- Uses `@computed_field` decorator in Pydantic

### 4. Enum-Based Type Safety
- All categorical fields use enums
- Prevents invalid states
- Enables type checking

### 5. Relationship Loading
- Relationships loaded separately to avoid N+1 queries
- Service layer handles eager/lazy loading decisions
- UI layer receives pre-joined data

### 6. Embedding Serialization
- Embeddings stored as BLOB in database
- Service layer provides serialization/deserialization
- UI layer never sees raw embeddings

### 7. Composite Scoring
- Multi-factor relevance scoring in service layer:
  - 0.6 × cosine similarity
  - 0.2 × salience
  - 0.1 × recency
  - 0.1 × link weight
- UI layer receives pre-scored results

---

## Usage Examples

### Example 1: Create User and Memory
```python
# Service Layer
user = User(user_id="user_123")
memory = Memory(
    user_id=user.user_id,
    content="User prefers technical explanations",
    sector_type=MemorySector.SEMANTIC,
    salience=0.85,
    recency=datetime.utcnow()
)

# Generate embedding
embedding_vector = embedding_service.generate(memory.content)
embedding = MemoryEmbedding(
    memory_id=memory.id,
    sector="semantic",
    embedding=MemoryEmbedding.serialize_vector(embedding_vector),
    embedding_model="text-embedding-3-small",
    dimension=1536
)

# Database Layer
db.insert_user(user)
db.insert_memory(memory)
db.insert_embedding(embedding)
```

### Example 2: Query Memories for UI
```python
# Service Layer
memories = memory_service.query(
    user_id="user_123",
    query="What are the user's preferences?",
    limit=5
)

# Transform to UI Layer
memory_summaries = [
    MemorySummary(
        memoryId=m.id,
        content=m.content,
        sector=m.sector_type.value,
        salience=m.salience,
        ageDays=m.age_days,
        state=m.state.value
    )
    for m in memories
]

# Return to frontend
return {"memories": memory_summaries}
```

### Example 3: Build Dashboard
```python
# Service Layer - aggregate data
user = user_service.get(user_id)
conversations = conversation_service.list(user_id, limit=5)
top_memories = memory_service.top_by_salience(user_id, limit=5)
stats = analytics_service.get_user_stats(user_id)

# Transform to UI Layer
dashboard = UserDashboard(
    userId=user.user_id,
    userName=user.metadata.get("name") if user.metadata else None,
    memoryStats=stats["memory_stats"],
    conversationStats=stats["conversation_stats"],
    recentConversations=[
        ConversationSummary.from_conversation(c)
        for c in conversations
    ],
    topMemories=[
        MemorySummary.from_memory(m)
        for m in top_memories
    ]
)

return dashboard
```

---

## Conclusion

This three-layer data model provides:
- **Normalized persistence** for efficient storage
- **Rich domain models** for business logic
- **Optimized UI models** for frontend performance

Each layer has clear responsibilities and transformation boundaries, enabling independent evolution and testing.
