# Project Specification: OpenMemory + ElevenLabs Integration

## Problem Statement

Organizations using ElevenLabs AI voice agents need their agents to maintain contextual, personalized conversations across multiple interactions. Currently, agents lack persistent memory, forcing users to re-explain context in every conversation. This creates poor user experience, reduces efficiency, and limits the agents' ability to provide personalized service.

## Solution Overview

Build an integration between OpenMemory's cognitive memory system and ElevenLabs Agents that enables:

1. **Pre-conversation context loading**: Agent starts conversations with relevant user history
2. **Real-time memory access**: Agent queries memories during conversations when needed
3. **Post-conversation learning**: System extracts and stores new memories after each call

This creates a continuous memory loop where agents get smarter with each interaction.

## Target Users

### Primary Users
- **ElevenLabs Agent Developers**: Teams building AI voice agents for customer service, sales, or support
- **Enterprise Customers**: Companies deploying AI voice agents at scale

### User Personas

#### Persona 1: Sarah - Customer Support Team Lead
- **Role**: Manages AI voice agent deployment for e-commerce support
- **Pain Points**:
  - Agents can't remember previous customer issues
  - Users have to repeat information every call
  - No visibility into conversation patterns
- **Goals**:
  - Reduce average handle time
  - Improve customer satisfaction scores
  - Enable agents to provide personalized service
- **Success Criteria**:
  - 30% reduction in calls where users say "I already told you this"
  - 20% improvement in CSAT scores

#### Persona 2: Marcus - AI Engineer
- **Role**: Builds and maintains voice agent integrations
- **Pain Points**:
  - Difficult to implement custom memory solutions
  - Managing conversation state is complex
  - No good tools for memory debugging
- **Goals**:
  - Easy integration with existing ElevenLabs setup
  - Reliable, fast memory retrieval
  - Clear monitoring and debugging tools
- **Success Criteria**:
  - Integration takes <1 week
  - Memory queries respond in <500ms
  - Clear logs when things go wrong

#### Persona 3: David - Product Manager
- **Role**: Owns AI voice agent product roadmap
- **Pain Points**:
  - Need to justify ROI of AI voice agents
  - Difficult to measure conversation quality improvements
  - Scaling costs are unclear
- **Goals**:
  - Demonstrate measurable business impact
  - Predictable scaling costs
  - Data-driven product decisions
- **Success Criteria**:
  - Can show memory recall improves outcomes
  - Understands cost per conversation
  - Has dashboards showing key metrics

## Requirements

### Functional Requirements

#### FR-1: Conversation Initiation
**Description**: When a conversation starts, load relevant user context and provide it to the agent.

**Acceptance Criteria**:
- System receives caller_id from ElevenLabs webhook
- Maps caller_id to user_id in user database
- Queries top 5 most relevant memories for user
- Formats memories as dynamic variables
- Returns response to ElevenLabs within 2 seconds
- If user is new, creates guest user profile
- Handles unknown caller_ids gracefully

**Priority**: P0 (Must Have)
**Story Points**: 8

#### FR-2: Real-Time Memory Query
**Description**: During conversation, agent can query user's memory on-demand.

**Acceptance Criteria**:
- Exposes `/tools/memory-query` endpoint
- Accepts natural language query, user_id, and limit parameters
- Generates embedding for query
- Performs vector similarity search across user's memories
- Applies composite scoring (similarity + salience + recency + links)
- Expands results via memory graph (single-waypoint)
- Returns memories and natural language summary
- Responds within 500ms (p95)
- Logs all tool invocations with latency

**Priority**: P0 (Must Have)
**Story Points**: 13

#### FR-3: Post-Call Memory Extraction
**Description**: After conversation ends, extract new memories and store them.

**Acceptance Criteria**:
- Receives post-call webhook from ElevenLabs with transcript
- Verifies HMAC signature
- Acknowledges webhook within 100ms
- Queues event for async processing
- Background worker extracts memories using LLM
- Classifies memories into 5 sectors (semantic, episodic, procedural, emotional, reflective)
- Generates embeddings for each memory
- Stores memories in database
- Creates links to similar existing memories
- Updates user summary
- Marks webhook as completed
- Implements retry logic with exponential backoff
- Dead-letters events after 5 failed retries

**Priority**: P0 (Must Have)
**Story Points**: 21

#### FR-4: Memory Sector Classification
**Description**: Categorize memories into different types for better retrieval.

**Acceptance Criteria**:
- Implements 5 memory sectors:
  - **Semantic**: Facts, preferences, user attributes
  - **Episodic**: Specific events, actions taken
  - **Procedural**: How-to knowledge, processes
  - **Emotional**: Sentiment, satisfaction levels
  - **Reflective**: Patterns, behavioral insights
- Uses LLM to classify extracted memories
- Generates sector-specific embeddings
- Allows querying across specific sectors
- Tracks sector distribution per user

**Priority**: P0 (Must Have)
**Story Points**: 8

#### FR-5: Memory Graph and Links
**Description**: Create relationships between related memories for context expansion.

**Acceptance Criteria**:
- Automatically creates links between similar memories (cosine > 0.8)
- Stores link weight (similarity score)
- Implements single-waypoint graph expansion
- Supports bidirectional links
- Prevents duplicate links
- Allows manual link creation (future enhancement)
- Link weights decay over time

**Priority**: P1 (Should Have)
**Story Points**: 8

#### FR-6: User Profile Management
**Description**: Maintain user profiles with auto-generated summaries.

**Acceptance Criteria**:
- Creates user profile on first interaction
- Auto-generates user summary from memories
- Updates summary after new memories added
- Tracks total memory count per user
- Stores user metadata (created_at, last_interaction)
- Supports user deletion with cascade

**Priority**: P1 (Should Have)
**Story Points**: 5

#### FR-7: Memory Decay and Archival
**Description**: Implement time-based memory decay to keep relevant memories fresh.

**Acceptance Criteria**:
- Applies decay formula: salience * e^(-decay_rate * days_old)
- Runs decay process every 6 hours
- Archives memories older than 180 days with salience < 0.3
- Keeps archived memories queryable
- Allows manual override of decay rate per memory
- Supports memory reactivation if reinforced

**Priority**: P2 (Nice to Have)
**Story Points**: 8

#### FR-8: Memory Conflict Resolution
**Description**: Handle contradictory memories intelligently.

**Acceptance Criteria**:
- Detects conflicts when new memory contradicts existing
- For semantic memories: marks old as superseded, keeps new
- For episodic memories: boosts salience (reinforcement)
- Uses LLM to determine if memories conflict
- Logs all conflict resolutions
- Allows manual conflict resolution

**Priority**: P2 (Nice to Have)
**Story Points**: 13

### Non-Functional Requirements

#### NFR-1: Performance
- Conversation initiation webhook: <1000ms (p95), <1800ms (p99)
- Memory query server tool: <500ms (p95), <1000ms (p99)
- Post-call webhook acknowledgment: <50ms (p95), <100ms (p99)
- Memory extraction: <3000ms (p95), <5000ms (p99)
- Database query latency: <100ms (p95)

#### NFR-2: Scalability
- Support 100 conversations/day initially (SQLite)
- Clear migration path to 1000 conversations/day (PostgreSQL + Redis)
- Horizontal scaling possible for 10,000+ conversations/day
- Database size growth: Linear with conversation count
- Memory storage: ~10KB per conversation average

#### NFR-3: Reliability
- System uptime: 99.9% (43 minutes downtime per month)
- Error rate: <0.1% of all requests
- Webhook processing: 100% eventual consistency
- No data loss on system failures
- Graceful degradation when external APIs fail

#### NFR-4: Security
- All API endpoints require authentication (Bearer token)
- HMAC signature verification on all webhooks
- Rate limiting: 100 requests/minute per IP for tools, 200/minute for webhooks
- Constant-time comparison for sensitive operations
- Encryption at rest for conversation transcripts
- No logging of PII or sensitive data
- API keys rotatable without downtime

#### NFR-5: Observability
- Structured JSON logging for all operations
- Prometheus metrics for all endpoints (latency, error rate, throughput)
- OpenTelemetry traces for critical paths
- Health check endpoints (/health, /health/live, /health/ready)
- Webhook queue depth monitoring
- Alert on: high latency, high error rate, queue backlog, service degradation

#### NFR-6: Maintainability
- Type hints throughout Python codebase
- Unit tests for business logic (>80% coverage)
- Integration tests for critical paths
- API documentation (OpenAPI/Swagger)
- Database migrations with Alembic
- Docker containerization
- Clear deployment procedures

## User Stories

### Epic 1: Conversation Initiation

#### Story 1.1: Load User Context at Call Start
**As a** customer support agent (AI)
**I want** to access the user's recent conversation history when they call
**So that** I can provide personalized service without asking them to repeat information

**Acceptance Criteria**:
- Given a user has called before
- When they initiate a new call
- Then I receive their top 5 relevant memories
- And the memories are formatted in natural language
- And I can reference them in my greeting

**Priority**: P0
**Story Points**: 5

#### Story 1.2: Handle New Users Gracefully
**As a** customer support agent (AI)
**I want** to handle first-time callers without errors
**So that** new users have a smooth experience

**Acceptance Criteria**:
- Given a user has never called before
- When they initiate their first call
- Then a guest profile is created automatically
- And I receive an appropriate greeting for new users
- And no error occurs

**Priority**: P0
**Story Points**: 3

### Epic 2: Real-Time Memory Access

#### Story 2.1: Query User Memory During Conversation
**As a** customer support agent (AI)
**I want** to query the user's past conversations on-demand
**So that** I can answer questions about our history together

**Acceptance Criteria**:
- Given a user asks "What did we discuss last time?"
- When I call the query_user_memory tool
- Then I receive relevant memories from past conversations
- And the response includes a natural language summary
- And I can use this to formulate my response

**Priority**: P0
**Story Points**: 8

#### Story 2.2: Query Specific Memory Types
**As a** customer support agent (AI)
**I want** to query specific types of memories (facts vs events)
**So that** I can get more precise information

**Acceptance Criteria**:
- Given I need factual information about the user
- When I specify sector_type="semantic" in my query
- Then I only receive semantic memories (facts, preferences)
- And episodic memories are filtered out

**Priority**: P1
**Story Points**: 3

### Epic 3: Post-Call Learning

#### Story 3.1: Extract Memories After Conversation
**As a** system administrator
**I want** memories to be automatically extracted after each call
**So that** agents get smarter over time

**Acceptance Criteria**:
- Given a conversation has ended
- When the post-call webhook is received
- Then memories are extracted within 5 seconds
- And they are categorized into appropriate sectors
- And they are available for the next conversation

**Priority**: P0
**Story Points**: 13

#### Story 3.2: Link Related Memories
**As a** customer support agent (AI)
**I want** related memories to be linked together
**So that** I can access connected context

**Acceptance Criteria**:
- Given a new memory is created
- When it's similar to existing memories (similarity > 0.8)
- Then automatic links are created
- And I can traverse these links during retrieval

**Priority**: P1
**Story Points**: 5

### Epic 4: Memory Management

#### Story 4.1: Decay Old Memories
**As a** product manager
**I want** old memories to gradually lose importance
**So that** agents focus on recent, relevant information

**Acceptance Criteria**:
- Given a memory is 90 days old
- When the decay process runs
- Then its salience is reduced by decay formula
- And very old, low-salience memories are archived

**Priority**: P2
**Story Points**: 5

#### Story 4.2: Resolve Memory Conflicts
**As a** customer support agent (AI)
**I want** contradictory information to be resolved automatically
**So that** I don't give conflicting responses

**Acceptance Criteria**:
- Given a user says "I changed my email address"
- When this conflicts with an existing email memory
- Then the old memory is marked as superseded
- And the new memory becomes active
- And I only see the current information

**Priority**: P2
**Story Points**: 8

### Epic 5: Monitoring and Operations

#### Story 5.1: Monitor System Health
**As a** DevOps engineer
**I want** to monitor all system metrics in real-time
**So that** I can detect and resolve issues quickly

**Acceptance Criteria**:
- Given the system is running
- When I access the Grafana dashboard
- Then I see latency metrics for all endpoints
- And I see error rates, queue depth, and throughput
- And I receive alerts when SLAs are violated

**Priority**: P1
**Story Points**: 8

#### Story 5.2: Trace Slow Requests
**As a** DevOps engineer
**I want** to trace requests through the entire system
**So that** I can identify bottlenecks

**Acceptance Criteria**:
- Given a memory query is slow
- When I check the tracing dashboard
- Then I can see timing for each step (embedding, search, expansion)
- And I can identify which step is the bottleneck

**Priority**: P1
**Story Points**: 5

### Epic 6: Developer Experience

#### Story 6.1: Easy Local Development Setup
**As a** developer
**I want** to run the entire system locally with one command
**So that** I can develop and test efficiently

**Acceptance Criteria**:
- Given I have Docker installed
- When I run `docker-compose up`
- Then all services start (API, workers, Redis, databases)
- And I can make test API calls
- And I see logs from all components

**Priority**: P1
**Story Points**: 5

#### Story 6.2: Clear API Documentation
**As a** API consumer
**I want** comprehensive API documentation
**So that** I can integrate easily

**Acceptance Criteria**:
- Given I visit /docs endpoint
- When I browse the OpenAPI documentation
- Then I see all endpoints with descriptions
- And I see request/response examples
- And I can test endpoints directly

**Priority**: P1
**Story Points**: 3

## Out of Scope (Not Included in MVP)

### Future Enhancements
- Multi-modal memories (images, audio clips)
- Memory sharing across users (team knowledge)
- Manual memory editing UI
- Memory analytics dashboard
- Multi-tenant support
- Advanced memory consolidation (LLM-based merging)
- Federated learning across conversations
- Memory export/import

### Explicitly Not Doing
- Building a custom LLM (using OpenAI)
- Real-time conversation transcription (ElevenLabs provides this)
- Voice synthesis (ElevenLabs handles this)
- Custom embedding models (using OpenAI)
- User authentication/authorization (assumed handled upstream)
- Billing and subscription management

## Success Criteria

### Technical Success Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Initiation webhook latency | <1s p95 | Prometheus histogram |
| Memory query latency | <500ms p95 | Prometheus histogram |
| Memory extraction latency | <3s p95 | Prometheus histogram |
| System uptime | 99.9% | Uptime monitoring |
| Error rate | <0.1% | Error logs + metrics |
| Memory recall accuracy | >90% | Manual evaluation |
| Test coverage | >80% | pytest-cov |

### Business Success Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Conversation continuity | >80% conversations reference past context | Log analysis |
| User satisfaction | >4.0/5.0 rating | Post-call surveys |
| Repeat call time savings | 30% reduction | Time comparison |
| Memory extraction quality | >90% of important facts captured | Manual review |
| Developer onboarding time | <1 week to integration | Developer surveys |

### Launch Criteria (MVP)
- ✅ All P0 requirements implemented and tested
- ✅ Load testing passes at 100 conversations/day
- ✅ All success metrics meet targets in staging
- ✅ Security audit completed (authentication, HMAC, rate limiting)
- ✅ Monitoring dashboards configured and tested
- ✅ Documentation complete (API, deployment, operations)
- ✅ Incident response playbook written
- ✅ Backup and recovery tested

## Assumptions and Dependencies

### Assumptions
- ElevenLabs account and API access available
- OpenAI API access for embeddings and LLM
- Users consent to conversation storage
- Caller ID is reliable identifier (or alternative provided)
- Average conversation length is 5-10 minutes
- Average of 3-5 memories extracted per conversation
- Embedding dimension is 1536 (OpenAI text-embedding-3-small)

### Dependencies
- **ElevenLabs API**: Webhook delivery, agent configuration
- **OpenAI API**: Embeddings, memory extraction LLM
- **Infrastructure**: Server with 2 vCPU, 4GB RAM minimum
- **Network**: Stable connection for webhook delivery
- **External services uptime**: ElevenLabs 99.9%, OpenAI 99.9%

### Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| OpenAI API latency spikes | High | Medium | Cache embeddings, use circuit breaker, fallback to keyword search |
| ElevenLabs webhook delivery failures | High | Low | Implement retry logic, poll for missed events |
| SQLite performance limits | Medium | High | Plan PostgreSQL migration, load test early |
| Memory extraction quality issues | High | Medium | Human-in-loop review, iterative prompt tuning |
| Webhook signature verification bugs | Critical | Low | Thorough testing, constant-time comparison |
| Cost overruns on API calls | Medium | Medium | Set budget alerts, cache aggressively |

## Privacy and Compliance

### Data Handling
- **Personal Data Stored**: Caller ID, conversation transcripts, extracted memories
- **Retention Policy**: Memories retained indefinitely unless user requests deletion
- **Right to be Forgotten**: Implement user deletion endpoint
- **Data Encryption**: Encrypt transcripts at rest, TLS in transit
- **Access Controls**: API authentication required for all endpoints

### Compliance Considerations
- **GDPR**: User consent required, data deletion on request
- **CCPA**: User data access and deletion rights
- **HIPAA**: Not in scope for MVP (healthcare conversations excluded)
- **SOC 2**: Prepare for audit (logging, access controls, encryption)

## Appendix

### Glossary
- **Memory**: A discrete piece of information extracted from a conversation
- **Sector**: Category of memory (semantic, episodic, procedural, emotional, reflective)
- **Salience**: Importance score of a memory (0.0 to 1.0)
- **Embedding**: Vector representation of memory content for similarity search
- **Graph expansion**: Following memory links to find related context
- **Waypoint**: Intermediate memory in graph traversal
- **Decay rate**: Rate at which memory salience decreases over time
- **Composite score**: Weighted combination of similarity, salience, recency, and link weight

### References
- [OpenMemory GitHub](https://github.com/CaviraOSS/OpenMemory)
- [ElevenLabs Agents Documentation](https://elevenlabs.io/docs/agents-platform)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLite WAL Mode](https://www.sqlite.org/wal.html)

---

**Version**: 1.0
**Last Updated**: 2025-11-05
**Approved By**: Product Team
**Next Review**: 2025-12-05
