# Implementation Tasks: OpenMemory + ElevenLabs Integration

## Overview

This document provides a comprehensive, actionable task list for implementing the OpenMemory + ElevenLabs integration. Tasks are organized by phase and include time estimates, dependencies, and acceptance criteria.

**Total Estimated Time**: 8 weeks (1 developer full-time)
**Phases**: 6
**Total Tasks**: 127

---

## Phase 1: Core Infrastructure (Week 1-2)

**Goal**: Set up foundational infrastructure and database
**Estimated Time**: 10 days
**Priority**: P0

### 1.1 Project Setup (2 days)

#### Task 1.1.1: Initialize Project Repository
- [ ] Create Git repository with .gitignore
- [ ] Set up branch protection rules
- [ ] Create README.md with project overview
- [ ] Add LICENSE file
- [ ] Create .env.example template
- **Estimate**: 2 hours
- **Dependencies**: None
- **Acceptance**: Repository created, basic docs in place

#### Task 1.1.2: Set Up Python Environment
- [ ] Create requirements.txt with all dependencies
- [ ] Set up virtual environment
- [ ] Configure Python 3.11+
- [ ] Install development dependencies (pytest, black, mypy)
- [ ] Configure pre-commit hooks
- **Estimate**: 2 hours
- **Dependencies**: 1.1.1
- **Acceptance**: `pip install -r requirements.txt` works

#### Task 1.1.3: Configure Environment Variables
- [ ] Define all required environment variables
- [ ] Create .env.example with placeholders
- [ ] Document each variable in README
- [ ] Set up config.py for environment management
- [ ] Add validation for required variables
- **Estimate**: 2 hours
- **Dependencies**: 1.1.2
- **Acceptance**: Config validates and loads correctly

#### Task 1.1.4: Set Up Development Docker Environment
- [ ] Create Dockerfile for API service
- [ ] Create Dockerfile for worker service
- [ ] Create docker-compose.yml
- [ ] Configure volume mounts for development
- [ ] Add health checks to services
- **Estimate**: 4 hours
- **Dependencies**: 1.1.2
- **Acceptance**: `docker-compose up` runs successfully

### 1.2 Database Setup (3 days)

#### Task 1.2.1: Create SQLAlchemy Models
- [ ] Create base model with common fields
- [ ] Create User model
- [ ] Create Memory model
- [ ] Create MemoryEmbedding model
- [ ] Create MemoryLink model
- [ ] Create Conversation model
- [ ] Create ConversationTurn model
- [ ] Create ConversationAnalysis model
- [ ] Create ConversationMemory model
- [ ] Create ToolInvocation model
- [ ] Create WebhookEvent model
- [ ] Add relationships between models
- [ ] Add indexes to models
- **Estimate**: 1 day
- **Dependencies**: 1.1.2
- **Acceptance**: All 11 models created with proper relationships

#### Task 1.2.2: Set Up Alembic for Migrations
- [ ] Initialize Alembic
- [ ] Configure alembic.ini
- [ ] Set up env.py with async support
- [ ] Create initial migration script
- [ ] Test migration up and down
- [ ] Document migration workflow in README
- **Estimate**: 3 hours
- **Dependencies**: 1.2.1
- **Acceptance**: `alembic upgrade head` creates all tables

#### Task 1.2.3: Implement Database Connection Manager
- [ ] Create database.py module
- [ ] Implement connection pooling
- [ ] Configure SQLite with WAL mode
- [ ] Set PRAGMA settings (synchronous, cache_size, etc.)
- [ ] Add connection health check
- [ ] Implement transaction context manager
- **Estimate**: 4 hours
- **Dependencies**: 1.2.1
- **Acceptance**: Database connections work with WAL mode

#### Task 1.2.4: Create Database Indexes
- [ ] Add index on webhook_events (status, created_at)
- [ ] Add index on conversations (user_id, start_time)
- [ ] Add index on memories (user_id, sector_type, state)
- [ ] Add index on memories (recency, salience)
- [ ] Add index on memory_links (source_id, link_weight)
- [ ] Add index on conversation_turns (conversation_fk, turn_order)
- [ ] Add index on tool_invocations (conversation_fk, timestamp)
- [ ] Test index effectiveness with EXPLAIN QUERY PLAN
- **Estimate**: 3 hours
- **Dependencies**: 1.2.2
- **Acceptance**: All indexes created, queries use them

#### Task 1.2.5: Write Database Utility Functions
- [ ] Create CRUD operations for each model
- [ ] Implement bulk insert for efficiency
- [ ] Create query helpers for common patterns
- [ ] Add database seeding script for testing
- [ ] Implement database reset script
- **Estimate**: 4 hours
- **Dependencies**: 1.2.3
- **Acceptance**: All CRUD operations tested

### 1.3 FastAPI Application Setup (3 days)

#### Task 1.3.1: Create Basic FastAPI Application
- [ ] Create main.py with FastAPI app
- [ ] Configure CORS middleware
- [ ] Set up exception handlers
- [ ] Configure request logging middleware
- [ ] Add startup and shutdown events
- **Estimate**: 3 hours
- **Dependencies**: 1.1.2
- **Acceptance**: FastAPI app starts and responds to /

#### Task 1.3.2: Implement Structured Logging
- [ ] Install and configure structlog
- [ ] Create logging configuration
- [ ] Add request ID middleware
- [ ] Implement log context manager
- [ ] Add logging to all routes (decorator or middleware)
- [ ] Configure log output format (JSON for production)
- **Estimate**: 4 hours
- **Dependencies**: 1.3.1
- **Acceptance**: All requests logged with structured format

#### Task 1.3.3: Implement Health Check Endpoints
- [ ] Create /health endpoint (detailed health)
- [ ] Create /health/live endpoint (liveness probe)
- [ ] Create /health/ready endpoint (readiness probe)
- [ ] Add database health check
- [ ] Add external API health checks (OpenAI)
- [ ] Return appropriate HTTP status codes
- **Estimate**: 3 hours
- **Dependencies**: 1.3.1, 1.2.3
- **Acceptance**: All health endpoints return correct status

#### Task 1.3.4: Set Up API Versioning
- [ ] Create /api/v1 router
- [ ] Set up route organization (webhooks, tools, admin)
- [ ] Add API version to response headers
- [ ] Document versioning strategy
- **Estimate**: 2 hours
- **Dependencies**: 1.3.1
- **Acceptance**: Routes organized under /api/v1

#### Task 1.3.5: Configure OpenAPI Documentation
- [ ] Customize OpenAPI schema
- [ ] Add API description and metadata
- [ ] Add security scheme documentation
- [ ] Add examples to all endpoints
- [ ] Configure Swagger UI at /docs
- [ ] Configure ReDoc at /redoc
- **Estimate**: 3 hours
- **Dependencies**: 1.3.4
- **Acceptance**: /docs shows comprehensive API documentation

### 1.4 Testing Infrastructure (2 days)

#### Task 1.4.1: Set Up pytest Configuration
- [ ] Create pytest.ini with settings
- [ ] Configure test discovery
- [ ] Set up fixtures for database
- [ ] Create conftest.py with shared fixtures
- [ ] Configure test database (separate from dev)
- **Estimate**: 2 hours
- **Dependencies**: 1.2.3
- **Acceptance**: `pytest` discovers and runs tests

#### Task 1.4.2: Create Test Utilities
- [ ] Create factory functions for models
- [ ] Create mock data generators
- [ ] Implement database fixture (setup/teardown)
- [ ] Create API client fixture
- [ ] Add assertion helpers
- **Estimate**: 4 hours
- **Dependencies**: 1.4.1
- **Acceptance**: Test utilities available and documented

#### Task 1.4.3: Write Initial Unit Tests
- [ ] Test database models (CRUD operations)
- [ ] Test health check endpoints
- [ ] Test logging configuration
- [ ] Test environment configuration
- [ ] Achieve >80% coverage on existing code
- **Estimate**: 4 hours
- **Dependencies**: 1.4.2
- **Acceptance**: All tests pass, coverage >80%

#### Task 1.4.4: Set Up Code Quality Tools
- [ ] Configure Black (code formatting)
- [ ] Configure Flake8 (linting)
- [ ] Configure mypy (type checking)
- [ ] Configure isort (import sorting)
- [ ] Add pre-commit hooks for all tools
- [ ] Document code quality standards
- **Estimate**: 3 hours
- **Dependencies**: 1.1.2
- **Acceptance**: All code quality checks pass

---

## Phase 2: Conversation Initiation Webhook (Week 2-3)

**Goal**: Implement webhook endpoint for conversation initiation
**Estimated Time**: 5 days
**Priority**: P0

### 2.1 User Management (2 days)

#### Task 2.1.1: Implement User CRUD Operations
- [ ] Create user creation endpoint (internal)
- [ ] Implement get_user_by_id
- [ ] Implement get_user_by_caller_id
- [ ] Implement update_user
- [ ] Implement delete_user (with cascade)
- **Estimate**: 3 hours
- **Dependencies**: 1.2.5
- **Acceptance**: User CRUD operations work correctly

#### Task 2.1.2: Implement Caller ID to User ID Mapping
- [ ] Create phone number normalization function
- [ ] Add caller_id field to User model (or separate table)
- [ ] Implement lookup by caller_id
- [ ] Handle multiple caller IDs per user
- [ ] Add tests for phone number formats
- **Estimate**: 3 hours
- **Dependencies**: 2.1.1
- **Acceptance**: Caller ID lookup works for various formats

#### Task 2.1.3: Implement Guest User Creation
- [ ] Create guest user creation logic
- [ ] Generate unique guest user IDs
- [ ] Set default guest user properties
- [ ] Add guest user flag to User model
- [ ] Test guest user creation flow
- **Estimate**: 2 hours
- **Dependencies**: 2.1.1
- **Acceptance**: Guest users created automatically

#### Task 2.1.4: Implement User Summary Generation
- [ ] Create prompt template for user summarization
- [ ] Implement LLM call for summary generation
- [ ] Update user summary after memory changes
- [ ] Add summary to User model
- [ ] Test summary quality with sample data
- **Estimate**: 4 hours
- **Dependencies**: 2.1.1
- **Acceptance**: User summaries generated accurately

### 2.2 Memory Query for Initiation (1 day)

#### Task 2.2.1: Implement Fast Memory Query
- [ ] Create query_memories_for_initiation function
- [ ] Use recency + salience only (no vector search)
- [ ] Return top 5 memories
- [ ] Optimize query with proper indexes
- [ ] Achieve <500ms query time
- **Estimate**: 3 hours
- **Dependencies**: 1.2.4
- **Acceptance**: Query returns in <500ms

#### Task 2.2.2: Implement Memory Formatting for Dynamic Variables
- [ ] Create format_memories_for_agent function
- [ ] Convert memories to natural language
- [ ] Extract recent topics from memories
- [ ] Format timestamps in friendly format
- [ ] Test formatting with various memory types
- **Estimate**: 2 hours
- **Dependencies**: 2.2.1
- **Acceptance**: Memories formatted as readable text

#### Task 2.2.3: Optimize Memory Query Performance
- [ ] Benchmark current query performance
- [ ] Add caching for repeated queries (if needed)
- [ ] Optimize database query (use EXPLAIN)
- [ ] Add performance logging
- [ ] Test under load (10 concurrent requests)
- **Estimate**: 3 hours
- **Dependencies**: 2.2.1
- **Acceptance**: Meets <500ms target under load

### 2.3 Initiation Webhook Endpoint (2 days)

#### Task 2.3.1: Create Webhook Request/Response Models
- [ ] Create InitiationRequest model (Pydantic)
- [ ] Create InitiationResponse model
- [ ] Create DynamicVariables model
- [ ] Create ConversationConfigOverride model
- [ ] Add validation rules
- **Estimate**: 2 hours
- **Dependencies**: None
- **Acceptance**: Models validate correctly

#### Task 2.3.2: Implement Conversation Initiation Endpoint
- [ ] Create POST /webhook/conversation-init
- [ ] Parse incoming request
- [ ] Map caller_id to user_id
- [ ] Get or create user
- [ ] Query top 5 memories
- [ ] Format dynamic variables
- [ ] Return response within 2s
- **Estimate**: 4 hours
- **Dependencies**: 2.1.2, 2.2.1, 2.3.1
- **Acceptance**: Endpoint returns valid response

#### Task 2.3.3: Add Error Handling
- [ ] Handle invalid caller_id gracefully
- [ ] Handle database errors
- [ ] Handle timeout scenarios
- [ ] Return appropriate HTTP status codes
- [ ] Log all errors with context
- **Estimate**: 2 hours
- **Dependencies**: 2.3.2
- **Acceptance**: All error scenarios handled

#### Task 2.3.4: Implement Latency Monitoring
- [ ] Add latency tracking to endpoint
- [ ] Log slow requests (>1s)
- [ ] Add Prometheus histogram for latency
- [ ] Set up alert for high latency
- [ ] Test latency under load
- **Estimate**: 2 hours
- **Dependencies**: 2.3.2
- **Acceptance**: Latency tracked and logged

#### Task 2.3.5: Write Integration Tests
- [ ] Test successful user lookup
- [ ] Test guest user creation
- [ ] Test memory retrieval and formatting
- [ ] Test error scenarios
- [ ] Test latency requirements
- **Estimate**: 4 hours
- **Dependencies**: 2.3.2
- **Acceptance**: All tests pass

---

## Phase 3: Memory Query Server Tool (Week 3-4)

**Goal**: Implement real-time memory query endpoint
**Estimated Time**: 10 days
**Priority**: P0

### 3.1 Embedding Integration (2 days)

#### Task 3.1.1: Set Up OpenAI API Client
- [ ] Install OpenAI Python SDK
- [ ] Configure API key management
- [ ] Create OpenAI client wrapper
- [ ] Add retry logic for API calls
- [ ] Implement circuit breaker
- **Estimate**: 3 hours
- **Dependencies**: 1.1.3
- **Acceptance**: OpenAI client configured and tested

#### Task 3.1.2: Implement Embedding Generation
- [ ] Create generate_embedding function
- [ ] Use text-embedding-3-small model
- [ ] Handle API errors gracefully
- [ ] Log embedding generation time
- [ ] Add unit tests
- **Estimate**: 2 hours
- **Dependencies**: 3.1.1
- **Acceptance**: Embeddings generated successfully

#### Task 3.1.3: Implement Embedding Caching
- [ ] Create in-memory LRU cache
- [ ] Hash text for cache key
- [ ] Set cache size limit (10,000 entries)
- [ ] Add cache hit rate metrics
- [ ] Test cache effectiveness
- **Estimate**: 3 hours
- **Dependencies**: 3.1.2
- **Acceptance**: Cache reduces API calls by >50%

#### Task 3.1.4: Implement Embedding Storage
- [ ] Create store_embedding function
- [ ] Serialize numpy array to BLOB
- [ ] Handle different embedding dimensions
- [ ] Add embedding retrieval function
- [ ] Test with sample embeddings
- **Estimate**: 2 hours
- **Dependencies**: 3.1.2, 1.2.3
- **Acceptance**: Embeddings stored and retrieved correctly

### 3.2 Vector Similarity Search (3 days)

#### Task 3.2.1: Implement Cosine Similarity Function
- [ ] Create cosine_similarity function
- [ ] Use numpy for efficiency
- [ ] Handle edge cases (zero vectors)
- [ ] Add unit tests with known values
- [ ] Benchmark performance
- **Estimate**: 2 hours
- **Dependencies**: None
- **Acceptance**: Cosine similarity calculated correctly

#### Task 3.2.2: Implement Vector Search Function
- [ ] Create vector_search function
- [ ] Load all user embeddings
- [ ] Calculate similarity for each
- [ ] Apply sector filters
- [ ] Return top N matches
- **Estimate**: 4 hours
- **Dependencies**: 3.2.1, 3.1.4
- **Acceptance**: Vector search returns relevant results

#### Task 3.2.3: Implement Composite Scoring
- [ ] Create composite_score function
- [ ] Weight: 60% similarity, 20% salience, 10% recency, 10% links
- [ ] Implement recency_score (time decay)
- [ ] Calculate average link weight
- [ ] Sort results by composite score
- **Estimate**: 3 hours
- **Dependencies**: 3.2.2
- **Acceptance**: Composite scores calculated correctly

#### Task 3.2.4: Optimize Vector Search Performance
- [ ] Benchmark current performance
- [ ] Use numpy vectorization
- [ ] Pre-compute recency scores
- [ ] Cache link weights
- [ ] Achieve <300ms search time
- **Estimate**: 4 hours
- **Dependencies**: 3.2.3
- **Acceptance**: Search completes in <300ms for 1000 memories

#### Task 3.2.5: Add Sector-Specific Search
- [ ] Implement sector filtering
- [ ] Allow multiple sector selection
- [ ] Optimize queries with sector index
- [ ] Test with different sector combinations
- [ ] Document sector types
- **Estimate**: 2 hours
- **Dependencies**: 3.2.2
- **Acceptance**: Sector filtering works correctly

### 3.3 Graph Expansion (2 days)

#### Task 3.3.1: Implement Graph Expansion Function
- [ ] Create expand_via_links function
- [ ] Get top 2 links for each initial match
- [ ] Add linked memories to result set
- [ ] Prevent duplicates
- [ ] Respect max_total limit
- **Estimate**: 3 hours
- **Dependencies**: 3.2.2
- **Acceptance**: Graph expansion works correctly

#### Task 3.3.2: Implement Score Boosting for Linked Memories
- [ ] Boost linked memory scores by link weight
- [ ] Formula: original_score * 0.8 + link_weight * 0.2
- [ ] Re-sort after expansion
- [ ] Add tests with known link structures
- **Estimate**: 2 hours
- **Dependencies**: 3.3.1
- **Acceptance**: Scores boosted correctly

#### Task 3.3.3: Optimize Graph Traversal
- [ ] Use efficient graph data structure
- [ ] Batch link queries
- [ ] Cache frequently accessed links
- [ ] Achieve <100ms expansion time
- **Estimate**: 3 hours
- **Dependencies**: 3.3.1
- **Acceptance**: Expansion completes in <100ms

### 3.4 Memory Query Endpoint (2 days)

#### Task 3.4.1: Create Memory Query Request/Response Models
- [ ] Create MemoryQueryRequest model
- [ ] Create MemoryQueryResponse model
- [ ] Create Memory model (API response)
- [ ] Add validation rules
- [ ] Add documentation
- **Estimate**: 2 hours
- **Dependencies**: None
- **Acceptance**: Models validate correctly

#### Task 3.4.2: Implement Memory Query Endpoint
- [ ] Create POST /tools/memory-query
- [ ] Parse request
- [ ] Generate query embedding
- [ ] Perform vector search
- [ ] Expand via graph
- [ ] Generate summary
- [ ] Return response
- **Estimate**: 4 hours
- **Dependencies**: 3.1.2, 3.2.2, 3.3.1, 3.4.1
- **Acceptance**: Endpoint returns relevant memories

#### Task 3.4.3: Implement Memory Summary Generation
- [ ] Create generate_memory_summary function
- [ ] Use GPT-4o-mini for fast generation
- [ ] Create prompt template
- [ ] Limit to 1-2 sentences
- [ ] Achieve <300ms generation time
- **Estimate**: 3 hours
- **Dependencies**: 3.1.1
- **Acceptance**: Summaries generated accurately and quickly

#### Task 3.4.4: Add Tool Invocation Logging
- [ ] Log all tool invocations to database
- [ ] Include request params, response data
- [ ] Track latency for each invocation
- [ ] Log success/failure status
- [ ] Add tests for logging
- **Estimate**: 2 hours
- **Dependencies**: 3.4.2
- **Acceptance**: All invocations logged

### 3.5 Authentication & Rate Limiting (1 day)

#### Task 3.5.1: Implement Bearer Token Authentication
- [ ] Create APIKeyManager class
- [ ] Load API keys from environment
- [ ] Implement verify_api_key dependency
- [ ] Use constant-time comparison
- [ ] Add to memory query endpoint
- **Estimate**: 3 hours
- **Dependencies**: 1.1.3
- **Acceptance**: Only valid API keys accepted

#### Task 3.5.2: Implement Rate Limiting
- [ ] Install slowapi library
- [ ] Configure rate limiter
- [ ] Add 100/minute limit to tools
- [ ] Add 200/minute limit to webhooks
- [ ] Return 429 when exceeded
- **Estimate**: 2 hours
- **Dependencies**: None
- **Acceptance**: Rate limits enforced

#### Task 3.5.3: Write Security Tests
- [ ] Test invalid API keys rejected
- [ ] Test rate limiting works
- [ ] Test constant-time comparison
- [ ] Test various attack scenarios
- **Estimate**: 3 hours
- **Dependencies**: 3.5.1, 3.5.2
- **Acceptance**: All security tests pass

---

## Phase 4: Post-Call Webhook & Memory Extraction (Week 4-6)

**Goal**: Implement async memory extraction pipeline
**Estimated Time**: 15 days
**Priority**: P0

### 4.1 Post-Call Webhook Endpoint (2 days)

#### Task 4.1.1: Implement HMAC Signature Verification
- [ ] Create verify_elevenlabs_signature function
- [ ] Parse signature header (t=timestamp,v1=signature)
- [ ] Check timestamp freshness (5 min window)
- [ ] Compute HMAC-SHA256
- [ ] Use secrets.compare_digest (constant-time)
- [ ] Add unit tests with known signatures
- **Estimate**: 3 hours
- **Dependencies**: None
- **Acceptance**: Signature verification works correctly

#### Task 4.1.2: Create Webhook Request Models
- [ ] Create PostCallWebhookRequest model
- [ ] Create WebhookData model
- [ ] Create Transcript model
- [ ] Create Analysis model
- [ ] Add validation
- **Estimate**: 2 hours
- **Dependencies**: None
- **Acceptance**: Models parse ElevenLabs webhook payload

#### Task 4.1.3: Implement Post-Call Webhook Endpoint
- [ ] Create POST /webhook/post-call
- [ ] Verify HMAC signature
- [ ] Parse payload
- [ ] Insert into webhook_events (status=pending)
- [ ] Return 200 OK immediately
- [ ] Achieve <100ms response time
- **Estimate**: 4 hours
- **Dependencies**: 4.1.1, 4.1.2
- **Acceptance**: Webhook acknowledged in <100ms

#### Task 4.1.4: Implement Idempotency
- [ ] Use event_id as unique key
- [ ] Handle duplicate webhooks gracefully
- [ ] Return 200 for duplicates (already processed)
- [ ] Log duplicate attempts
- [ ] Test with repeated requests
- **Estimate**: 2 hours
- **Dependencies**: 4.1.3
- **Acceptance**: Duplicate webhooks handled correctly

### 4.2 Background Worker (3 days)

#### Task 4.2.1: Create Worker Main Loop
- [ ] Create worker.py module
- [ ] Implement main async loop
- [ ] Poll webhook_events table (status=pending)
- [ ] Process events one by one
- [ ] Sleep 500ms between polls
- **Estimate**: 3 hours
- **Dependencies**: 1.2.3
- **Acceptance**: Worker processes events continuously

#### Task 4.2.2: Implement Event Processing Pipeline
- [ ] Create process_webhook_event function
- [ ] Mark event as processing
- [ ] Call processing functions in order
- [ ] Mark as completed on success
- [ ] Handle errors with retry logic
- **Estimate**: 4 hours
- **Dependencies**: 4.2.1
- **Acceptance**: Events processed successfully

#### Task 4.2.3: Implement Retry Logic
- [ ] Increment retry_count on failure
- [ ] Apply exponential backoff (2^retry seconds)
- [ ] Dead-letter after 5 retries
- [ ] Log all retries with reason
- [ ] Test with simulated failures
- **Estimate**: 3 hours
- **Dependencies**: 4.2.2
- **Acceptance**: Failed events retried correctly

#### Task 4.2.4: Add Worker Health Monitoring
- [ ] Track worker heartbeat
- [ ] Log processing stats (events/sec, errors)
- [ ] Expose metrics endpoint for worker
- [ ] Add queue depth metric
- [ ] Set up alerts for worker failures
- **Estimate**: 3 hours
- **Dependencies**: 4.2.1
- **Acceptance**: Worker health visible in metrics

#### Task 4.2.5: Implement Graceful Shutdown
- [ ] Handle SIGTERM signal
- [ ] Finish processing current event
- [ ] Close database connections
- [ ] Log shutdown
- [ ] Test with Docker stop
- **Estimate**: 2 hours
- **Dependencies**: 4.2.1
- **Acceptance**: Worker shuts down gracefully

### 4.3 Conversation Storage (2 days)

#### Task 4.3.1: Implement Store Conversation Function
- [ ] Create store_conversation function
- [ ] Parse conversation data from webhook
- [ ] Insert into conversations table
- [ ] Handle missing fields gracefully
- [ ] Return conversation ID
- **Estimate**: 2 hours
- **Dependencies**: 1.2.5
- **Acceptance**: Conversations stored correctly

#### Task 4.3.2: Implement Store Transcript Function
- [ ] Create store_transcript function
- [ ] Insert all turns into conversation_turns
- [ ] Use bulk insert for efficiency
- [ ] Maintain turn order
- [ ] Handle large transcripts
- **Estimate**: 2 hours
- **Dependencies**: 4.3.1
- **Acceptance**: Transcripts stored correctly

#### Task 4.3.3: Implement Store Analysis Function
- [ ] Create store_analysis function
- [ ] Parse analysis data from webhook
- [ ] Insert into conversation_analysis
- [ ] Handle optional fields
- [ ] Test with various analysis formats
- **Estimate**: 2 hours
- **Dependencies**: 4.3.1
- **Acceptance**: Analysis stored correctly

#### Task 4.3.4: Implement Transaction Handling
- [ ] Wrap all storage in database transaction
- [ ] Rollback on any error
- [ ] Log transaction failures
- [ ] Test rollback scenarios
- **Estimate**: 2 hours
- **Dependencies**: 4.3.1, 4.3.2, 4.3.3
- **Acceptance**: Transactions ensure data consistency

### 4.4 Memory Extraction (4 days)

#### Task 4.4.1: Design Memory Extraction Prompt
- [ ] Create prompt template for memory extraction
- [ ] Define 5 sector types clearly
- [ ] Request structured JSON output
- [ ] Include examples in prompt
- [ ] Test prompt with sample transcripts
- **Estimate**: 4 hours
- **Dependencies**: None
- **Acceptance**: Prompt extracts quality memories

#### Task 4.4.2: Implement Memory Extraction Function
- [ ] Create extract_memories_from_transcript function
- [ ] Concatenate transcript into text
- [ ] Call LLM with extraction prompt
- [ ] Parse JSON response
- [ ] Validate extracted memories
- **Estimate**: 4 hours
- **Dependencies**: 3.1.1, 4.4.1
- **Acceptance**: Memories extracted from transcripts

#### Task 4.4.3: Implement Sector Classification
- [ ] Classify each memory into sector
- [ ] Validate sector types
- [ ] Handle ambiguous cases
- [ ] Log classification decisions
- [ ] Test with diverse memory types
- **Estimate**: 2 hours
- **Dependencies**: 4.4.2
- **Acceptance**: Memories classified correctly

#### Task 4.4.4: Implement Salience Scoring
- [ ] Parse salience from LLM output
- [ ] Validate range (0.0 to 1.0)
- [ ] Apply default if missing
- [ ] Test salience distribution
- **Estimate**: 2 hours
- **Dependencies**: 4.4.2
- **Acceptance**: Salience scores reasonable

#### Task 4.4.5: Optimize Extraction Performance
- [ ] Benchmark current extraction time
- [ ] Use GPT-4o-mini for speed
- [ ] Batch multiple extractions if needed
- [ ] Achieve <3s extraction time
- [ ] Test under load
- **Estimate**: 3 hours
- **Dependencies**: 4.4.2
- **Acceptance**: Extraction completes in <3s

#### Task 4.4.6: Implement Quality Monitoring
- [ ] Log extraction success/failure rate
- [ ] Sample and review extracted memories
- [ ] Track average memories per conversation
- [ ] Set up alerts for quality issues
- [ ] Create manual review process
- **Estimate**: 3 hours
- **Dependencies**: 4.4.2
- **Acceptance**: Extraction quality monitored

### 4.5 Memory Storage & Linking (3 days)

#### Task 4.5.1: Implement Store Memory Function
- [ ] Create store_memory function
- [ ] Insert memory into memories table
- [ ] Generate and store embedding
- [ ] Link to source conversation
- [ ] Return memory ID
- **Estimate**: 3 hours
- **Dependencies**: 3.1.2, 4.4.2
- **Acceptance**: Memories stored with embeddings

#### Task 4.5.2: Implement Find Similar Memories Function
- [ ] Create find_similar_memories function
- [ ] Use vector search for similarity
- [ ] Set threshold (e.g., 0.8)
- [ ] Limit results (e.g., top 5)
- [ ] Exclude source memory itself
- **Estimate**: 2 hours
- **Dependencies**: 3.2.2
- **Acceptance**: Similar memories found correctly

#### Task 4.5.3: Implement Automatic Link Building
- [ ] Create build_automatic_links function
- [ ] Find similar memories (cosine > 0.8)
- [ ] Create bidirectional links
- [ ] Store link weight (similarity score)
- [ ] Prevent duplicate links
- **Estimate**: 3 hours
- **Dependencies**: 4.5.2
- **Acceptance**: Links created automatically

#### Task 4.5.4: Implement Batch Memory Storage
- [ ] Store all extracted memories in transaction
- [ ] Generate embeddings for all at once
- [ ] Build links for all new memories
- [ ] Handle partial failures
- [ ] Optimize for performance
- **Estimate**: 4 hours
- **Dependencies**: 4.5.1, 4.5.3
- **Acceptance**: Batch storage efficient and reliable

#### Task 4.5.5: Link Memories to Conversation
- [ ] Create conversation_memories records
- [ ] Set extraction_type='auto'
- [ ] Store confidence scores
- [ ] Test linkage
- **Estimate**: 2 hours
- **Dependencies**: 4.5.1
- **Acceptance**: Memories linked to conversations

### 4.6 Integration & Testing (1 day)

#### Task 4.6.1: Write End-to-End Integration Test
- [ ] Simulate full webhook flow
- [ ] Send post-call webhook
- [ ] Wait for worker to process
- [ ] Verify conversation stored
- [ ] Verify memories extracted
- [ ] Verify links created
- **Estimate**: 4 hours
- **Dependencies**: All Phase 4 tasks
- **Acceptance**: Full flow works end-to-end

#### Task 4.6.2: Load Test Memory Extraction
- [ ] Simulate 100 conversations/day
- [ ] Monitor queue depth
- [ ] Monitor processing latency
- [ ] Verify no data loss
- [ ] Tune worker count if needed
- **Estimate**: 3 hours
- **Dependencies**: 4.6.1
- **Acceptance**: System handles 100 conversations/day

---

## Phase 5: Monitoring & Observability (Week 6-7)

**Goal**: Implement comprehensive monitoring
**Estimated Time**: 5 days
**Priority**: P1

### 5.1 Prometheus Metrics (2 days)

#### Task 5.1.1: Add Prometheus Client
- [ ] Install prometheus_client library
- [ ] Create metrics module
- [ ] Define all metrics (Counter, Histogram, Gauge)
- [ ] Expose /metrics endpoint
- [ ] Test metrics collection
- **Estimate**: 2 hours
- **Dependencies**: 1.3.1
- **Acceptance**: Metrics exposed at /metrics

#### Task 5.1.2: Add Latency Metrics
- [ ] Add histogram for initiation webhook
- [ ] Add histogram for memory query
- [ ] Add histogram for memory extraction
- [ ] Add histogram for database queries
- [ ] Configure appropriate buckets
- **Estimate**: 3 hours
- **Dependencies**: 5.1.1
- **Acceptance**: All latencies tracked

#### Task 5.1.3: Add Throughput Metrics
- [ ] Add counter for total requests
- [ ] Add counter for successful requests
- [ ] Add counter for failed requests
- [ ] Add counter per endpoint
- [ ] Track requests per second
- **Estimate**: 2 hours
- **Dependencies**: 5.1.1
- **Acceptance**: Throughput metrics available

#### Task 5.1.4: Add Business Metrics
- [ ] Add gauge for total users
- [ ] Add gauge for total memories
- [ ] Add gauge for webhook queue depth
- [ ] Add gauge for average memories per user
- [ ] Add gauge for memory extraction success rate
- **Estimate**: 3 hours
- **Dependencies**: 5.1.1
- **Acceptance**: Business metrics tracked

### 5.2 Grafana Dashboards (1 day)

#### Task 5.2.1: Set Up Prometheus Data Source
- [ ] Install Prometheus
- [ ] Configure scraping of API metrics
- [ ] Configure scraping of worker metrics
- [ ] Test data collection
- **Estimate**: 2 hours
- **Dependencies**: 5.1.1
- **Acceptance**: Prometheus scraping metrics

#### Task 5.2.2: Create API Performance Dashboard
- [ ] Panel: Request rate (per endpoint)
- [ ] Panel: Latency percentiles (p50, p95, p99)
- [ ] Panel: Error rate
- [ ] Panel: Active requests
- [ ] Panel: Response status codes
- **Estimate**: 3 hours
- **Dependencies**: 5.2.1
- **Acceptance**: Dashboard shows API metrics

#### Task 5.2.3: Create Memory System Dashboard
- [ ] Panel: Total memories
- [ ] Panel: Memories per user
- [ ] Panel: Memory query latency
- [ ] Panel: Extraction success rate
- [ ] Panel: Embedding API latency
- **Estimate**: 2 hours
- **Dependencies**: 5.2.1
- **Acceptance**: Dashboard shows memory metrics

#### Task 5.2.4: Create Webhook Processing Dashboard
- [ ] Panel: Webhook queue depth
- [ ] Panel: Processing latency
- [ ] Panel: Retry count
- [ ] Panel: Dead letter count
- [ ] Panel: Worker health
- **Estimate**: 2 hours
- **Dependencies**: 5.2.1
- **Acceptance**: Dashboard shows webhook metrics

### 5.3 OpenTelemetry Tracing (1 day)

#### Task 5.3.1: Set Up OpenTelemetry
- [ ] Install opentelemetry libraries
- [ ] Configure OTLP exporter
- [ ] Set up Jaeger backend
- [ ] Auto-instrument FastAPI
- [ ] Test trace collection
- **Estimate**: 3 hours
- **Dependencies**: 1.3.1
- **Acceptance**: Traces collected in Jaeger

#### Task 5.3.2: Add Manual Tracing for Critical Paths
- [ ] Add span for memory query flow
- [ ] Add span for embedding generation
- [ ] Add span for vector search
- [ ] Add span for graph expansion
- [ ] Add span for memory extraction
- [ ] Add attributes to spans (user_id, etc.)
- **Estimate**: 4 hours
- **Dependencies**: 5.3.1
- **Acceptance**: Critical paths traced

### 5.4 Alerting (1 day)

#### Task 5.4.1: Configure AlertManager
- [ ] Install AlertManager
- [ ] Configure receivers (email, Slack)
- [ ] Test alert delivery
- **Estimate**: 2 hours
- **Dependencies**: 5.2.1
- **Acceptance**: AlertManager configured

#### Task 5.4.2: Create Alert Rules
- [ ] Alert: High memory query latency (p95 > 500ms for 5m)
- [ ] Alert: High error rate (>1% for 5m)
- [ ] Alert: Webhook queue backlog (>50 for 10m)
- [ ] Alert: Memory extraction failures (success rate <90% for 15m)
- [ ] Alert: External API errors (OpenAI down)
- **Estimate**: 3 hours
- **Dependencies**: 5.4.1
- **Acceptance**: All alert rules configured

#### Task 5.4.3: Test Alert Scenarios
- [ ] Trigger each alert manually
- [ ] Verify alert delivery
- [ ] Document alert response procedures
- [ ] Create runbook for each alert
- **Estimate**: 3 hours
- **Dependencies**: 5.4.2
- **Acceptance**: Alerts tested and documented

---

## Phase 6: Security & Production Hardening (Week 7-8)

**Goal**: Prepare for production deployment
**Estimated Time**: 10 days
**Priority**: P0

### 6.1 Security Enhancements (3 days)

#### Task 6.1.1: Implement Encryption at Rest
- [ ] Install cryptography library
- [ ] Create DataEncryption class
- [ ] Generate encryption key (store securely)
- [ ] Encrypt conversation messages before storage
- [ ] Decrypt on retrieval
- [ ] Test encryption/decryption
- **Estimate**: 4 hours
- **Dependencies**: 1.2.3
- **Acceptance**: Sensitive data encrypted

#### Task 6.1.2: Implement API Key Rotation
- [ ] Support multiple active API keys
- [ ] Add API key creation endpoint (admin)
- [ ] Add API key revocation endpoint (admin)
- [ ] Add API key expiration
- [ ] Document rotation process
- **Estimate**: 4 hours
- **Dependencies**: 3.5.1
- **Acceptance**: API keys rotatable without downtime

#### Task 6.1.3: Add Request Validation
- [ ] Validate all input types
- [ ] Sanitize user inputs
- [ ] Add max length limits
- [ ] Prevent SQL injection (use parameterized queries)
- [ ] Prevent XSS (not applicable for API)
- **Estimate**: 3 hours
- **Dependencies**: All endpoints
- **Acceptance**: All inputs validated

#### Task 6.1.4: Implement CORS Properly
- [ ] Configure allowed origins
- [ ] Set appropriate CORS headers
- [ ] Test with browser clients
- [ ] Document CORS policy
- **Estimate**: 2 hours
- **Dependencies**: 1.3.1
- **Acceptance**: CORS configured securely

#### Task 6.1.5: Security Audit
- [ ] Review all authentication code
- [ ] Review all HMAC verification
- [ ] Check for timing attacks
- [ ] Review error messages (no info leakage)
- [ ] Run security scanner (Bandit)
- [ ] Fix all identified issues
- **Estimate**: 4 hours
- **Dependencies**: All security tasks
- **Acceptance**: No critical security issues

### 6.2 Backup & Recovery (2 days)

#### Task 6.2.1: Implement Database Backup Script
- [ ] Create backup.sh script
- [ ] Use SQLite .backup command
- [ ] Compress backups with gzip
- [ ] Add timestamp to backup filename
- [ ] Test backup creation
- **Estimate**: 2 hours
- **Dependencies**: 1.2.3
- **Acceptance**: Backups created successfully

#### Task 6.2.2: Set Up Backup Automation
- [ ] Schedule backups with cron (daily)
- [ ] Upload backups to S3/cloud storage
- [ ] Implement backup retention (30 days)
- [ ] Delete old backups automatically
- [ ] Monitor backup success/failure
- **Estimate**: 3 hours
- **Dependencies**: 6.2.1
- **Acceptance**: Backups automated and monitored

#### Task 6.2.3: Implement Database Restore
- [ ] Create restore.sh script
- [ ] Download backup from S3
- [ ] Verify backup integrity
- [ ] Restore database from backup
- [ ] Test restore procedure
- **Estimate**: 2 hours
- **Dependencies**: 6.2.1
- **Acceptance**: Database restored successfully

#### Task 6.2.4: Document Disaster Recovery Plan
- [ ] Document backup procedure
- [ ] Document restore procedure
- [ ] Define RTO and RPO targets
- [ ] Create incident response checklist
- [ ] Test full disaster recovery
- **Estimate**: 3 hours
- **Dependencies**: 6.2.3
- **Acceptance**: DR plan documented and tested

### 6.3 Deployment Automation (3 days)

#### Task 6.3.1: Create Production Dockerfile
- [ ] Optimize Dockerfile for production
- [ ] Use multi-stage build
- [ ] Run as non-root user
- [ ] Add health check
- [ ] Minimize image size
- **Estimate**: 3 hours
- **Dependencies**: 1.1.4
- **Acceptance**: Production image built successfully

#### Task 6.3.2: Create Production Docker Compose
- [ ] Create docker-compose.prod.yml
- [ ] Configure resource limits
- [ ] Add restart policies
- [ ] Configure logging drivers
- [ ] Add volumes for data persistence
- **Estimate**: 2 hours
- **Dependencies**: 6.3.1
- **Acceptance**: Production compose file works

#### Task 6.3.3: Set Up CI/CD Pipeline
- [ ] Create .github/workflows/test.yml (run tests)
- [ ] Create .github/workflows/build.yml (build Docker image)
- [ ] Create .github/workflows/deploy.yml (deploy to server)
- [ ] Configure secrets in GitHub
- [ ] Test full CI/CD flow
- **Estimate**: 4 hours
- **Dependencies**: 6.3.1
- **Acceptance**: CI/CD pipeline works end-to-end

#### Task 6.3.4: Create Deployment Scripts
- [ ] Create deploy.sh script
- [ ] Pull latest Docker images
- [ ] Run database migrations
- [ ] Restart services with zero downtime
- [ ] Verify deployment health
- **Estimate**: 3 hours
- **Dependencies**: 6.3.2
- **Acceptance**: Deployments automated

#### Task 6.3.5: Document Deployment Procedures
- [ ] Document local setup
- [ ] Document staging deployment
- [ ] Document production deployment
- [ ] Document rollback procedure
- [ ] Create deployment checklist
- **Estimate**: 3 hours
- **Dependencies**: 6.3.4
- **Acceptance**: Deployment fully documented

### 6.4 Production Readiness (2 days)

#### Task 6.4.1: Configure Production Environment Variables
- [ ] Set all production API keys
- [ ] Configure database path
- [ ] Set log levels
- [ ] Configure monitoring endpoints
- [ ] Document all environment variables
- **Estimate**: 2 hours
- **Dependencies**: 1.1.3
- **Acceptance**: Production environment configured

#### Task 6.4.2: Implement Graceful Degradation
- [ ] Add circuit breaker for OpenAI API
- [ ] Fallback to keyword search if embedding fails
- [ ] Return partial results if some components fail
- [ ] Log all degradation events
- [ ] Test degraded mode
- **Estimate**: 4 hours
- **Dependencies**: 3.1.1, 3.2.2
- **Acceptance**: System degrades gracefully

#### Task 6.4.3: Load Testing
- [ ] Set up Locust for load testing
- [ ] Test 100 conversations/day (4.2/hour)
- [ ] Test burst scenarios (10 conversations in 10 minutes)
- [ ] Monitor all metrics during load test
- [ ] Verify no errors or data loss
- [ ] Document load test results
- **Estimate**: 4 hours
- **Dependencies**: All implementation tasks
- **Acceptance**: System handles target load

#### Task 6.4.4: Chaos Testing
- [ ] Test with database connection failures
- [ ] Test with OpenAI API failures
- [ ] Test with worker crashes
- [ ] Test with disk full scenarios
- [ ] Verify recovery in all cases
- **Estimate**: 4 hours
- **Dependencies**: 6.4.2
- **Acceptance**: System recovers from failures

#### Task 6.4.5: Create Incident Response Playbook
- [ ] Document common issues and resolutions
- [ ] Create alert response procedures
- [ ] Document escalation paths
- [ ] Create on-call schedule
- [ ] Test incident response
- **Estimate**: 3 hours
- **Dependencies**: All monitoring tasks
- **Acceptance**: Incident response documented

---

## Optional Phase: Memory System Enhancements (Future)

**Goal**: Advanced memory management features
**Estimated Time**: 10 days
**Priority**: P2

### 7.1 Memory Decay & Archival

#### Task 7.1.1: Implement Memory Decay Function
- [ ] Create apply_decay function
- [ ] Apply decay formula: salience * e^(-decay_rate * days_old)
- [ ] Run decay process every 6 hours
- [ ] Log decay statistics
- [ ] Test decay over time
- **Estimate**: 3 hours

#### Task 7.1.2: Implement Memory Archival
- [ ] Create archive_old_memories function
- [ ] Archive memories >180 days old with salience <0.3
- [ ] Keep archived memories queryable
- [ ] Allow reactivation
- [ ] Test archival process
- **Estimate**: 3 hours

### 7.2 Memory Conflict Resolution

#### Task 7.2.1: Implement Conflict Detection
- [ ] Create detect_conflicts function
- [ ] Use LLM to identify contradictions
- [ ] Find similar memories that conflict
- [ ] Log all detected conflicts
- [ ] Test with known conflicts
- **Estimate**: 4 hours

#### Task 7.2.2: Implement Conflict Resolution
- [ ] Create resolve_conflict function
- [ ] For semantic: mark old as superseded, keep new
- [ ] For episodic: boost salience (reinforcement)
- [ ] Log resolution decisions
- [ ] Test resolution strategies
- **Estimate**: 4 hours

### 7.3 Memory Consolidation

#### Task 7.3.1: Implement Memory Consolidation
- [ ] Find highly similar memories (similarity >0.95)
- [ ] Use LLM to merge memories
- [ ] Update first memory with merged content
- [ ] Archive second memory
- [ ] Test consolidation quality
- **Estimate**: 4 hours

---

## Summary

### Total Task Count: 127 tasks
### Total Estimated Time: 8 weeks (1 developer full-time)

### Phase Breakdown:
- **Phase 1**: Core Infrastructure - 19 tasks, 10 days
- **Phase 2**: Conversation Initiation - 13 tasks, 5 days
- **Phase 3**: Memory Query Tool - 21 tasks, 10 days
- **Phase 4**: Memory Extraction - 26 tasks, 15 days
- **Phase 5**: Monitoring - 16 tasks, 5 days
- **Phase 6**: Production Hardening - 21 tasks, 10 days
- **Phase 7** (Optional): Enhancements - 11 tasks, 10 days

### Critical Path:
1. Core Infrastructure (Week 1-2)
2. Conversation Initiation (Week 2-3)
3. Memory Query Tool (Week 3-4)
4. Memory Extraction (Week 4-6)
5. Monitoring (Week 6-7)
6. Production Hardening (Week 7-8)

### Parallel Work Opportunities:
- Phase 2 and Phase 3 can partially overlap (Week 3)
- Phase 5 can start during Phase 4 (Week 5-6)
- Testing can happen continuously throughout

### Risk Buffer:
- Add 20% buffer for unexpected issues (1.6 weeks)
- Total project timeline: 9-10 weeks

---

**Version**: 1.0
**Last Updated**: 2025-11-05
**Status**: Ready for Implementation
