# ElevenLabs API Authentication Guide

## Overview

This document identifies the authentication methods for each API used in the OpenMemory + ElevenLabs integration. The authentication flows are categorized into three groups:

1. **ElevenLabs → Your API** (Inbound): ElevenLabs calling your webhooks and tools
2. **Your API → External Services** (Outbound): Your system calling external APIs
3. **Configuration**: How authentication is configured in ElevenLabs agent settings

---

## 1. ElevenLabs → Your API (Inbound)

These are endpoints that **you host** and ElevenLabs calls.

### 1.1 Conversation Initiation Webhook

**Endpoint**: `POST /webhook/conversation-init`

**Authentication Method**: None explicitly required, but recommended options:

#### Option A: IP Allowlisting (Recommended)
- Allow only ElevenLabs IP addresses to access this endpoint
- Configure firewall/load balancer rules
- Most secure for webhook endpoints

#### Option B: Static Bearer Token
- Add `Authorization: Bearer <token>` header validation
- Configure the bearer token in ElevenLabs agent settings
- Example implementation:

```python
@app.post("/webhook/conversation-init")
async def conversation_initiation(request: Request):
    auth_header = request.headers.get('Authorization')
    expected_token = os.getenv('INITIATION_WEBHOOK_SECRET')

    if not auth_header or auth_header != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Process webhook...
```

#### Option C: No Authentication
- Not recommended for production
- Only for development/testing

**Security Note**: This endpoint is called **before** conversation starts, so it's critical to protect it from unauthorized access.

---

### 1.2 Post-Call Webhook

**Endpoint**: `POST /webhook/post-call`

**Authentication Method**: **HMAC-SHA256 Signature Verification** (Required)

#### How It Works:
1. ElevenLabs signs each webhook request with HMAC-SHA256
2. Signature is sent in the `ElevenLabs-Signature` header
3. Your server verifies the signature using a shared secret

#### Header Format:
```
ElevenLabs-Signature: t=1234567890,v1=abc123def456...
```

Where:
- `t` = Unix timestamp when the request was sent
- `v1` = HMAC-SHA256 signature

#### Implementation:

```python
import hmac
import hashlib
import time

def verify_elevenlabs_signature(signature_header: str, body: bytes, secret: str) -> bool:
    """
    Verify HMAC-SHA256 signature from ElevenLabs.
    """
    try:
        # Parse signature header
        parts = dict(item.split('=') for item in signature_header.split(','))
        timestamp = parts['t']
        received_sig = parts['v1']

        # Check timestamp freshness (5 min window to prevent replay attacks)
        if abs(time.time() - int(timestamp)) > 300:
            return False

        # Compute expected signature
        payload = f"{timestamp}.{body.decode('utf-8')}"
        expected_sig = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_sig, received_sig)
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        return False

@app.post("/webhook/post-call")
async def post_call_webhook(request: Request):
    signature = request.headers.get('ElevenLabs-Signature')
    body = await request.body()

    # Verify signature using webhook secret from environment
    webhook_secret = os.getenv('ELEVENLABS_WEBHOOK_SECRET')

    if not verify_elevenlabs_signature(signature, body, webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Process webhook...
```

#### Configuration:
- **Webhook Secret**: Obtained from ElevenLabs dashboard when configuring the webhook
- **Storage**: Store in environment variable `ELEVENLABS_WEBHOOK_SECRET`
- **Security**: Never commit secrets to version control

#### Security Benefits:
- ✅ Verifies request came from ElevenLabs
- ✅ Prevents replay attacks (timestamp validation)
- ✅ Protects webhook data integrity
- ✅ Uses constant-time comparison to prevent timing attacks

---

### 1.3 Server Tools (Memory Query Tool)

**Endpoint**: `POST /tools/memory-query`

**Authentication Method**: **Bearer Token** (Configured in ElevenLabs Agent)

#### How It Works:
1. You configure a bearer token secret in ElevenLabs agent settings
2. ElevenLabs includes `Authorization: Bearer <token>` header in each tool call
3. Your API validates the token

#### Agent Configuration (ElevenLabs Dashboard):

```json
{
  "tool_name": "query_user_memory",
  "description": "Search user's conversation history...",
  "url": "https://your-api.com/tools/memory-query",
  "method": "POST",
  "authentication": {
    "type": "bearer_token",
    "secret_name": "api_token"
  },
  "body_parameters": [
    {"name": "query", "type": "string", "required": true},
    {"name": "user_id", "type": "string", "value": "{{user_id}}"}
  ]
}
```

#### Implementation:

```python
@app.post("/tools/memory-query")
async def memory_query_tool(request: MemoryQueryRequest, authorization: str = Header(None)):
    """
    Memory query tool with bearer token authentication.
    """
    expected_token = os.getenv('TOOL_API_TOKEN')

    # Validate authorization header
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.replace('Bearer ', '')

    if not hmac.compare_digest(token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid token")

    # Process tool request...
```

#### Configuration Steps:
1. **Generate token**: Use a secure random string (e.g., `openssl rand -hex 32`)
2. **Store in environment**: `TOOL_API_TOKEN=your_secure_token_here`
3. **Configure in ElevenLabs**:
   - Go to agent settings → Tools → Add authentication
   - Select "Bearer Token"
   - Name: `api_token` (matches `secret_name` in config)
   - Value: Paste your generated token

#### Security Notes:
- Use a strong, randomly generated token (minimum 32 bytes)
- Rotate tokens periodically
- Use HTTPS for all tool endpoints (required by ElevenLabs)
- Log failed authentication attempts

---

## 2. Your API → External Services (Outbound)

These are external APIs that **your system calls**.

### 2.1 OpenAI API (Embeddings & LLM)

**Used For**:
- Generating embeddings for memory queries
- LLM-based memory extraction from transcripts
- Memory summary generation

**Authentication Method**: **API Key**

#### Implementation:

```python
import openai
from openai import OpenAI

# Initialize client with API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Generate embeddings
def generate_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# LLM generation
def llm_generate(prompt: str, max_tokens: int = 100, model: str = "gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
```

#### Configuration:
- **API Key Source**: OpenAI Dashboard → API Keys → Create new secret key
- **Environment Variable**: `OPENAI_API_KEY=sk-proj-...`
- **Security**:
  - Never commit to version control
  - Use different keys for dev/prod
  - Monitor usage and set limits in OpenAI dashboard
  - Rotate keys periodically

#### Models Used in Architecture:
- **Embeddings**: `text-embedding-3-small` (1536 dimensions, fast and cost-effective)
- **Memory Extraction**: `gpt-4o-mini` (fast, cheap, good for structured extraction)
- **Optional**: `gpt-3.5-turbo` for memory summaries

---

### 2.2 Alternative Embedding Services

If not using OpenAI, you may use:

#### Option A: Local Embedding Model (No Authentication)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()
```

**Pros**: No API costs, no rate limits, offline capable
**Cons**: Requires GPU for good performance, lower quality than OpenAI

#### Option B: Hugging Face Inference API (API Key)
```python
import requests

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"},
        json={"inputs": text}
    )
    return response.json()
```

**Configuration**: `HF_API_TOKEN` from Hugging Face account settings

---

### 2.3 Alternative LLM Services

If not using OpenAI:

#### Option A: Anthropic Claude (API Key)
```python
import anthropic

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def llm_generate(prompt: str, max_tokens: int = 100):
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

**Configuration**: `ANTHROPIC_API_KEY` from Anthropic Console

#### Option B: Local LLM via Ollama (No Authentication)
```python
import requests

def llm_generate(prompt: str, max_tokens: int = 100, model: str = "llama2"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()['response']
```

**Setup**: Run Ollama locally, no API key needed

---

## 3. ElevenLabs Agent Configuration

### 3.1 Accessing ElevenLabs Dashboard

**URL**: https://elevenlabs.io/app/conversational-ai

**Authentication**: Username/password or SSO (managed by ElevenLabs account)

### 3.2 Configuring Webhooks

#### Conversation Initiation Webhook:
1. Navigate to: Agent Settings → Personalization → Conversation Initiation
2. Enable "Call external API before conversation"
3. Configure:
   - **URL**: `https://your-api.com/webhook/conversation-init`
   - **Method**: POST
   - **Timeout**: 2000ms
   - **Authentication** (optional): Select "Bearer Token" and add token

#### Post-Call Webhook:
1. Navigate to: Agent Settings → Webhooks → Post-Call
2. Add webhook URL: `https://your-api.com/webhook/post-call`
3. Select events: `post_call_transcription`, `post_call_audio` (optional)
4. **Signing Secret**: Copy this secret (shown only once)
5. Store secret as `ELEVENLABS_WEBHOOK_SECRET` in your environment

### 3.3 Configuring Server Tools

1. Navigate to: Agent Settings → Tools → Add Tool
2. Configure:
   - **Type**: Webhook
   - **Name**: `query_user_memory`
   - **Description**: "Search user's conversation history and preferences..."
   - **URL**: `https://your-api.com/tools/memory-query`
   - **Method**: POST
   - **Authentication**: Bearer Token
     - Click "Add Secret"
     - **Name**: `api_token`
     - **Value**: Your generated token (from `TOOL_API_TOKEN`)
3. Add body parameters:
   - `query` (string, required, agent-provided)
   - `user_id` (string, required, value: `{{user_id}}`)
   - `limit` (integer, optional, default: 3)

### 3.4 Updating System Prompt

Update your agent's system prompt to reference dynamic variables and tools:

```
You are a helpful AI assistant with access to user conversation history.

USER CONTEXT:
- User ID: {{user_id}}
- Name: {{user_name}}
- Previous Interactions: {{memory_context}}
- Recent Topics: {{recent_topics}}

CAPABILITIES:
You have the query_user_memory tool to search past conversations.
Use when the user references past discussions or you need historical context.

[Rest of system prompt...]
```

---

## 4. Security Best Practices Summary

### 4.1 Secrets Management

| Secret | Purpose | Storage | Rotation |
|--------|---------|---------|----------|
| `ELEVENLABS_WEBHOOK_SECRET` | Post-call webhook HMAC verification | Environment variable | When compromised |
| `TOOL_API_TOKEN` | Server tool authentication | Environment variable | Every 90 days |
| `INITIATION_WEBHOOK_SECRET` | Optional initiation webhook auth | Environment variable | Every 90 days |
| `OPENAI_API_KEY` | Embeddings & LLM | Environment variable | Every 6 months |
| `ANTHROPIC_API_KEY` | Alternative LLM | Environment variable | Every 6 months |

### 4.2 Security Checklist

- [ ] All webhooks use HTTPS (required by ElevenLabs)
- [ ] Post-call webhook implements HMAC verification
- [ ] HMAC timestamp validation prevents replay attacks (5-minute window)
- [ ] Server tools use bearer token authentication
- [ ] Bearer tokens are strong (minimum 32 bytes random)
- [ ] All secrets stored in environment variables (not code)
- [ ] Secrets not committed to version control (.env in .gitignore)
- [ ] Different secrets for dev/staging/prod environments
- [ ] Failed authentication attempts are logged
- [ ] Rate limiting implemented on all public endpoints
- [ ] IP allowlisting configured (optional but recommended)
- [ ] API key usage monitored in provider dashboards
- [ ] Secrets rotation schedule established

### 4.3 Common Security Mistakes to Avoid

❌ **Don't**: Hardcode secrets in code
✅ **Do**: Use environment variables

❌ **Don't**: Skip HMAC verification on post-call webhooks
✅ **Do**: Always verify signatures

❌ **Don't**: Use weak bearer tokens (e.g., "token123")
✅ **Do**: Use cryptographically random tokens

❌ **Don't**: Commit `.env` files to git
✅ **Do**: Add `.env` to `.gitignore`

❌ **Don't**: Use HTTP endpoints
✅ **Do**: Use HTTPS for all external endpoints

❌ **Don't**: Share API keys across environments
✅ **Do**: Use separate keys for dev/prod

❌ **Don't**: Ignore failed authentication attempts
✅ **Do**: Log and monitor authentication failures

---

## 5. Environment Variables Template

Create a `.env` file (never commit this):

```bash
# ElevenLabs Integration
ELEVENLABS_WEBHOOK_SECRET=your_webhook_signing_secret_from_dashboard
TOOL_API_TOKEN=your_generated_bearer_token_32_bytes_min
INITIATION_WEBHOOK_SECRET=optional_bearer_token_for_init_webhook

# OpenAI (or alternative LLM provider)
OPENAI_API_KEY=sk-proj-your_openai_api_key
# OR for Anthropic:
# ANTHROPIC_API_KEY=sk-ant-your_anthropic_key
# OR for Hugging Face:
# HF_API_TOKEN=hf_your_huggingface_token

# Database
DATABASE_PATH=/path/to/database.db

# API Server
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production  # or development

# Optional: IP Allowlisting
ALLOWED_IPS=1.2.3.4,5.6.7.8  # ElevenLabs IPs
```

### Generating Secure Tokens

```bash
# Generate bearer token for tools
openssl rand -hex 32

# Generate webhook secret (if not provided by ElevenLabs)
openssl rand -hex 32

# Or in Python:
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 6. Testing Authentication

### 6.1 Test Post-Call Webhook Signature

```python
# test_webhook_signature.py
import hmac
import hashlib
import time
import requests

def test_post_call_webhook():
    webhook_url = "https://your-api.com/webhook/post-call"
    secret = "your_webhook_secret"

    # Sample payload
    payload = {
        "webhook_event_id": "evt_test123",
        "event_type": "post_call_transcription",
        "data": {"conversation_id": "test_conv"}
    }

    # Generate signature
    timestamp = str(int(time.time()))
    payload_str = f"{timestamp}.{json.dumps(payload)}"
    signature = hmac.new(
        secret.encode(),
        payload_str.encode(),
        hashlib.sha256
    ).hexdigest()

    # Send request
    headers = {
        "ElevenLabs-Signature": f"t={timestamp},v1={signature}",
        "Content-Type": "application/json"
    }

    response = requests.post(webhook_url, json=payload, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_post_call_webhook()
```

### 6.2 Test Server Tool Authentication

```bash
# Test with valid token
curl -X POST https://your-api.com/tools/memory-query \
  -H "Authorization: Bearer your_tool_api_token" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "user_id": "test_user", "limit": 3}'

# Test with invalid token (should return 401)
curl -X POST https://your-api.com/tools/memory-query \
  -H "Authorization: Bearer invalid_token" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "user_id": "test_user", "limit": 3}'
```

---

## 7. Monitoring & Alerting

### Key Authentication Metrics to Monitor:

```python
# Example metrics
metrics = {
    "webhook_signature_failures": counter("webhook.signature.failed"),
    "tool_auth_failures": counter("tool.auth.failed"),
    "initiation_auth_failures": counter("initiation.auth.failed"),
    "invalid_tokens_rate": gauge("auth.invalid_tokens.rate"),
    "openai_api_errors": counter("openai.api.errors"),
}

# Alert conditions
alerts = {
    "high_auth_failures": "auth failures > 10/minute for 5 minutes",
    "webhook_signature_failures": "signature failures > 5 in 10 minutes",
    "api_key_invalid": "OpenAI returns 401 Unauthorized",
}
```

### Logging Authentication Events:

```python
import logging

logger = logging.getLogger(__name__)

# Log successful authentications (info level)
logger.info(f"Webhook authenticated: event_id={event_id}, source_ip={ip}")

# Log failed authentications (warning level)
logger.warning(f"Auth failed: endpoint={endpoint}, ip={ip}, reason=invalid_token")

# Log security events (error level)
logger.error(f"Potential attack: {failed_attempts} failed auth from {ip} in 1 minute")
```

---

## 8. Troubleshooting Guide

### Problem: Post-call webhook returns 401

**Causes**:
- Invalid webhook secret
- Incorrect signature verification logic
- Timestamp too old (>5 minutes)

**Solutions**:
1. Verify secret matches ElevenLabs dashboard
2. Check signature parsing logic
3. Ensure server time is synchronized (use NTP)
4. Add debug logging to signature verification

### Problem: Server tool returns 401

**Causes**:
- Bearer token mismatch
- Token not configured in ElevenLabs
- Missing Authorization header

**Solutions**:
1. Verify `TOOL_API_TOKEN` environment variable
2. Check ElevenLabs tool configuration → Authentication → Secrets
3. Ensure secret_name matches in tool config
4. Test with curl to isolate issue

### Problem: OpenAI API returns 401

**Causes**:
- Invalid API key
- Expired API key
- API key from wrong organization

**Solutions**:
1. Verify API key in OpenAI dashboard
2. Check organization ID if using multiple orgs
3. Regenerate API key if expired
4. Test key with curl:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Problem: Webhook receiving unauthorized requests

**Causes**:
- Endpoint exposed without protection
- No IP allowlisting
- Secrets leaked

**Solutions**:
1. Implement HMAC verification (post-call)
2. Add bearer token auth (initiation)
3. Configure IP allowlisting
4. Rotate all secrets
5. Check for secrets in git history

---

## Summary

### Authentication Flow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ ElevenLabs Agent                                            │
└───┬─────────────────────────────┬──────────────────────┬────┘
    │                             │                      │
    │ No auth or                  │ Bearer Token         │ HMAC-SHA256
    │ Bearer Token                │ (configured in       │ (webhook secret)
    │ (optional)                  │ tool settings)       │
    │                             │                      │
    ▼                             ▼                      ▼
┌─────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
│ Initiation      │  │ Server Tool         │  │ Post-Call        │
│ Webhook         │  │ (memory-query)      │  │ Webhook          │
│ /webhook/init   │  │ /tools/memory-query │  │ /webhook/post    │
└─────────────────┘  └─────────────────────┘  └──────────────────┘
         │                      │                       │
         └──────────────────────┴───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Your Integration API  │
                    │ (FastAPI/Flask)       │
                    └───────────────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼ API Key                   ▼ API Key
         ┌─────────────────┐        ┌──────────────────┐
         │ OpenAI API      │        │ Alternative LLM  │
         │ (Embeddings)    │        │ (Anthropic, etc) │
         └─────────────────┘        └──────────────────┘
```

### Quick Reference Table

| API | Direction | Auth Method | Secret Location |
|-----|-----------|-------------|-----------------|
| Conversation Initiation Webhook | ElevenLabs → You | Optional Bearer Token | `INITIATION_WEBHOOK_SECRET` |
| Post-Call Webhook | ElevenLabs → You | **HMAC-SHA256** (Required) | `ELEVENLABS_WEBHOOK_SECRET` |
| Server Tools (memory-query) | ElevenLabs → You | **Bearer Token** (Required) | `TOOL_API_TOKEN` |
| OpenAI Embeddings API | You → OpenAI | **API Key** | `OPENAI_API_KEY` |
| OpenAI LLM API | You → OpenAI | **API Key** | `OPENAI_API_KEY` |
| Alternative LLM (Anthropic) | You → Anthropic | **API Key** | `ANTHROPIC_API_KEY` |
| Local Embedding Model | Local | None | N/A |

---

## Next Steps

1. **Set up environment variables** - Create `.env` file with all required secrets
2. **Implement authentication** - Add auth validation to all endpoints
3. **Configure ElevenLabs** - Set up webhooks and tools with proper secrets
4. **Test authentication** - Use provided test scripts
5. **Monitor authentication** - Set up logging and alerts
6. **Document secrets** - Maintain secure secret rotation schedule

For full implementation details, see:
- [Main Architecture Document](./openmemory-elevenlabs-architecture-corrected.md)
- [Quick Reference](./quick-reference.md)
- [ElevenLabs Documentation](https://elevenlabs.io/docs/agents-platform)
