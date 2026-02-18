# MemoryDB OpenClaw Plugin

OpenClaw plugin that connects to a self-hosted [MemoryDB](../README.md) REST API for long-term AI agent memory.

## What it does

**Tools** (available to the agent during conversations):
- `memory_search` -- hybrid search (semantic + BM25) with optional container inheritance
- `memory_store` -- save information to long-term memory (auto-chunked, embedded, indexed)
- `memory_forget` -- delete memories by ID or search query (GDPR-compliant)
- `memory_profile` -- get a synthesized profile/summary for a container

**Automatic behaviors:**
- **Conversation logging** -- logs all inbound/outbound messages across every channel
- **Session profile preload** -- injects key facts + recent context on session start
- **Auto-recall** -- injects relevant memories before each agent turn
- **Auto-capture** -- detects and stores important info from conversations (no LLM calls)
- **Compaction archiving** -- logs compaction events when context is compressed
- **Session lifecycle** -- logs session end events with message count and duration

**Commands:**
- `/memory <query>` -- quick search without invoking the AI agent
- `openclaw memorydb status|search|profile` -- CLI commands

## Prerequisites

A running MemoryDB instance:

```bash
cd /path/to/memorydb
docker compose up -d
```

## Installation

**1. Install dependencies:**

```bash
cd openclaw-plugin
npm install
```

**2. Add to OpenClaw config:**

```json
{
  "plugins": {
    "load": {
      "paths": ["/path/to/memorydb/openclaw-plugin"]
    },
    "slots": {
      "memory": "memory-memorydb"
    },
    "entries": {
      "memory-memorydb": {
        "enabled": true,
        "config": {
          "baseUrl": "http://localhost:8080",
          "container": "openclaw.my-agent",
          "autoRecall": true,
          "autoCapture": true,
          "conversationLog": true,
          "sessionProfile": true,
          "inherit": true
        }
      }
    }
  }
}
```

For Docker Compose deployments where OpenClaw and MemoryDB share a network, use the service name: `"baseUrl": "http://memorydb:8080"`.

**3. Restart the gateway:**

```bash
openclaw gateway restart
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `baseUrl` | string | `http://localhost:8080` | MemoryDB API URL |
| `container` | string | `default` | Default container/namespace for memories |
| `autoRecall` | boolean | `false` | Inject relevant memories before each agent turn |
| `autoCapture` | boolean | `false` | Auto-detect and store important info from conversations |
| `autoRecallLimit` | number | `5` | Max memories to inject per auto-recall |
| `conversationLog` | boolean | `true` | Log all inbound/outbound messages to MemoryDB |
| `sessionProfile` | boolean | `true` | Preload container profile at session start |
| `inherit` | boolean | `true` | Include parent container results when searching |
| `apiKey` | string | -- | Bearer token for API auth (if configured) |

## Updating

Since the plugin loads from the local filesystem, updating is:

```bash
cd /path/to/memorydb
git pull
cd openclaw-plugin
npm install
openclaw gateway restart
```

No reinstallation or config changes needed.

## How auto-capture works

Auto-capture uses rule-based heuristics (no LLM calls) to detect messages worth remembering:

- Explicit requests ("remember that...")
- Preferences ("I prefer...", "I like...", "I hate...")
- Decisions ("we decided...", "going with...", "switching to...")
- Personal facts ("my X is...", "always...", "never...")
- Entities (emails, phone numbers, IP addresses)

Captured content is tagged automatically and stored with `source: auto-capture` metadata.

**Limits:** Max 3 captures per conversation turn. Messages under 10 or over 1000 chars are skipped.

## How auto-recall works

Before each agent turn, the user's prompt is searched against stored memories. The top N results are injected as context:

```xml
<relevant-memories>
The following memories from long-term storage may be relevant:
- [87%] Seth prefers Elastic SIEM over Splunk (preference)
- [72%] Production VLAN is 20, Management is 50 (fact)
</relevant-memories>
```

## Architecture

```
+---------------+     HTTP/REST     +--------------+
|   OpenClaw    | -----------------> |   MemoryDB   |
|   Gateway     |                   |   (FastAPI)  |
|               |  POST /v1/search  |              |
|  memory-      |  POST /v1/docs    | PostgreSQL   |
|  memorydb     |  GET /v1/profile  | + pgvector   |
|  plugin       |  DELETE /v1/mem   |              |
+---------------+                   +--------------+
```

No MCP, no extra protocol layers. Just HTTP calls to your API.
