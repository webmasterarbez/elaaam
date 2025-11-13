# Project Constitution: OpenMemory + ElevenLabs Integration

## Project Vision

Build a production-ready integration between OpenMemory's cognitive memory system and ElevenLabs Agents, enabling AI voice agents to maintain contextual, personalized conversations through intelligent memory management.

## Core Principles

### 1. Performance First
- **Sub-second response times**: All user-facing operations must complete within 1 second (p95)
- **Async processing**: Long-running operations must be asynchronous to maintain responsiveness
- **Scalability mindset**: Design for 100 conversations/day initially, with clear path to 1000+
- **Measure everything**: Instrument all critical paths with metrics (latency, error rates, queue depth)

### 2. Reliability and Resilience
- **Graceful degradation**: System must continue operating even when external services fail
- **Idempotent operations**: All webhook processing must be idempotent to handle retries
- **Circuit breakers**: Protect against cascading failures from external APIs
- **Data integrity**: Use database transactions to ensure consistency

### 3. Security by Design
- **Authentication required**: All API endpoints must enforce authentication (except public webhooks)
- **Signature verification**: All ElevenLabs webhooks must verify HMAC signatures
- **Rate limiting**: Protect against abuse and ensure fair usage
- **Constant-time comparisons**: Prevent timing attacks in security-critical operations
- **Encryption at rest**: Sensitive conversation data must be encrypted in database

### 4. Developer Experience
- **Code clarity over cleverness**: Readable, maintainable code is more valuable than clever optimizations
- **Comprehensive logging**: Structured logging with context for debugging and monitoring
- **Type safety**: Use type hints throughout Python codebase
- **Documentation**: All public APIs and complex logic must be documented
- **Testing**: Unit tests for business logic, integration tests for critical paths

### 5. Data Quality and Privacy
- **User consent**: Only store memories with user awareness and consent
- **Memory decay**: Implement time-based salience decay to keep memories fresh
- **Conflict resolution**: Handle contradictory memories intelligently
- **Archival strategy**: Archive old, low-salience memories to keep system performant
- **Privacy by default**: No logging of sensitive user information

### 6. Operational Excellence
- **Observable systems**: Metrics, traces, and logs must be easily accessible
- **Automated deployments**: CI/CD pipeline for consistent, safe deployments
- **Database migrations**: All schema changes must use migration tools (Alembic)
- **Backup and recovery**: Automated backups with tested recovery procedures
- **Health checks**: Liveness and readiness probes for orchestration

## Technology Stack Constraints

### Core Technologies
- **Language**: Python 3.11+
- **Web Framework**: FastAPI (for async HTTP handling)
- **Database (Initial)**: SQLite with WAL mode (100-1000 conversations/day)
- **Database (Scale)**: PostgreSQL with pgvector extension (1000+ conversations/day)
- **Caching**: Redis (when scaling beyond 100 conversations/day)
- **LLM Provider**: OpenAI (GPT-4o-mini for memory extraction)
- **Embedding Provider**: OpenAI (text-embedding-3-small)

### Infrastructure
- **Containerization**: Docker for all services
- **Orchestration**: Docker Compose (dev), Kubernetes (production scale)
- **Monitoring**: Prometheus + Grafana
- **Tracing**: OpenTelemetry
- **Logging**: Structured JSON logs with structlog

## Development Workflow

### Git Branching Strategy
- **Main branch**: Production-ready code only
- **Feature branches**: `claude/feature-name-{session_id}` for new features
- **Pull requests**: All changes must go through PR review
- **Commit messages**: Clear, descriptive messages explaining the "why"

### Code Review Standards
- **Security review**: All authentication and webhook handling code
- **Performance review**: All database queries and external API calls
- **Test coverage**: All new features must include tests
- **Documentation review**: All API changes must update documentation

### Deployment Process
1. **Local testing**: All changes tested locally with docker-compose
2. **Staging environment**: Integration testing in staging
3. **Performance testing**: Load tests before production deployment
4. **Gradual rollout**: Canary deployments for risky changes
5. **Monitoring**: Watch dashboards for 24 hours post-deployment

## Quality Gates

### Before Merging
- ✅ All tests pass (unit + integration)
- ✅ No security vulnerabilities (dependency scan)
- ✅ Code coverage maintained or improved
- ✅ Performance benchmarks within targets
- ✅ Documentation updated

### Before Production Deployment
- ✅ Database migration tested with rollback
- ✅ Load tests pass at target scale
- ✅ Monitoring dashboards configured
- ✅ Alerts configured for new features
- ✅ Rollback plan documented

## Performance SLAs

| Metric | Target (p95) | Target (p99) | Critical Threshold |
|--------|--------------|--------------|-------------------|
| Conversation Initiation Webhook | <1000ms | <1800ms | 2000ms |
| Memory Query Server Tool | <500ms | <1000ms | 1500ms |
| Post-Call Webhook Acknowledgment | <50ms | <100ms | 200ms |
| Memory Extraction (async) | <3000ms | <5000ms | 10000ms |
| System Uptime | 99.9% | - | 99.0% |

## Error Budget

- **Monthly error budget**: 99.9% uptime = 43 minutes downtime per month
- **Error rate budget**: <0.1% of all requests can return 5xx errors
- **Latency budget**: 95% of requests must meet SLA targets

If error budget is exhausted, all feature work stops until reliability is restored.

## Non-Negotiables

### Must Have
1. **HMAC signature verification** on all webhooks
2. **WAL mode enabled** on SQLite
3. **Async webhook processing** with <100ms acknowledgment
4. **Structured logging** for all operations
5. **Health check endpoints** for monitoring
6. **Database backups** with tested recovery

### Must Not Do
1. **Synchronous processing** of post-call webhooks
2. **Logging sensitive user data** (PII, full conversations)
3. **Deploying without tests** for critical paths
4. **Breaking changes** without versioning
5. **Committing secrets** to version control

## Decision-Making Framework

### When to use SQLite
- ✅ Development and testing environments
- ✅ Initial production (<1000 conversations/day)
- ✅ Single-server deployments
- ❌ Multi-server deployments
- ❌ High-scale production (>1000 conversations/day)

### When to add caching (Redis)
- ✅ p95 latency exceeds 500ms for memory queries
- ✅ Embedding API costs become significant
- ✅ Traffic exceeds 100 conversations/day
- ❌ Premature optimization before measuring

### When to migrate to PostgreSQL
- ✅ Write throughput exceeds SQLite capacity (~1000 writes/sec)
- ✅ Vector search becomes bottleneck (linear scan too slow)
- ✅ Need for horizontal scaling
- ✅ Advanced features (full-text search, partitioning, replication)

### When to add a feature
Ask:
1. Does it align with core mission (better memory-enabled conversations)?
2. Can we measure its impact?
3. Does it maintain or improve performance?
4. Can we support it operationally?

If all answers are "yes", proceed. Otherwise, defer or decline.

## Success Metrics

### Product Metrics
- **Memory recall accuracy**: >90% of relevant memories retrieved
- **Conversation continuity**: Users feel "remembered" across calls
- **Memory extraction quality**: >90% of important facts captured

### Technical Metrics
- **Latency**: All endpoints meet SLA targets
- **Uptime**: 99.9% system availability
- **Error rate**: <0.1% of requests fail
- **Processing lag**: Webhook queue depth <10 events

### Operational Metrics
- **Mean Time to Recovery (MTTR)**: <30 minutes
- **Deployment frequency**: Multiple times per week
- **Change failure rate**: <5%
- **Lead time for changes**: <24 hours

## Governance

### Roles and Responsibilities
- **Product Owner**: Prioritizes features, defines success metrics
- **Tech Lead**: Ensures architectural consistency, reviews critical code
- **DevOps Engineer**: Maintains infrastructure, monitoring, deployments
- **Developers**: Implement features, write tests, maintain documentation

### Weekly Cadence
- **Monday**: Planning and prioritization
- **Wednesday**: Architecture review for in-progress features
- **Friday**: Demo completed work, retrospective

### Escalation Path
1. Developer discussion
2. Tech Lead review
3. Architecture committee (for significant decisions)
4. Product Owner (for scope/priority conflicts)

## Review and Evolution

This constitution is a living document. It should be reviewed:
- **Quarterly**: Are principles still relevant? Are metrics being met?
- **After incidents**: Did our principles prevent or contribute to the issue?
- **Before major changes**: Does the change align with our principles?

Changes to this document require:
- Proposal with rationale
- Team discussion and consensus
- Documented decision and effective date

---

**Version**: 1.0
**Last Updated**: 2025-11-05
**Next Review**: 2026-02-05
