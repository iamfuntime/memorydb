#!/usr/bin/env python3
"""
MemoryDB MCP Server

Thin MCP wrapper around the MemoryDB HTTP API. Exposes memory_search,
memory_store, memory_forget, memory_profile, and memory_stats as native
MCP tools for Claude Code / Claude Desktop.

Usage:
    python server.py --url http://localhost:8080 --token YOUR_TOKEN --container openclaw.vanguard

Register with Claude Code:
    claude mcp add memorydb -- python ~/git/memorydb/mcp-server/server.py \
        --url http://localhost:8080 --token YOUR_TOKEN --container openclaw.vanguard
"""

import argparse
import json
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# CLI args (parsed before MCP server starts)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="MemoryDB MCP Server")
parser.add_argument("--url", default="http://localhost:8080", help="MemoryDB API base URL")
parser.add_argument("--token", default="", help="MemoryDB API bearer token")
parser.add_argument("--container", default="openclaw.vanguard", help="Default memory container")
args = parser.parse_args()

BASE_URL = args.url.rstrip("/")
API_TOKEN = args.token
DEFAULT_CONTAINER = args.container

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, headers=_headers(), timeout=30.0)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "MemoryDB",
    instructions="Long-term memory for AI agents via MemoryDB (semantic + keyword hybrid search)",
)


# ---------------------------------------------------------------------------
# Tool: memory_search
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_search(
    query: str,
    container: str | None = None,
    limit: int = 10,
    include_related: bool = False,
    inherit: bool = True,
) -> str:
    """Search long-term memories using hybrid search (semantic + keyword).

    Use when you need context about user preferences, past decisions,
    previously discussed topics, or stored facts. Always search before
    answering technical questions.

    Args:
        query: Natural language search query
        container: Container to search (default: configured container)
        limit: Max results to return (default: 10)
        include_related: Include graph-related memories (default: false)
        inherit: Search parent containers too (default: true)
    """
    with _client() as client:
        resp = client.post("/v1/search", json={
            "query": query,
            "container": container or DEFAULT_CONTAINER,
            "limit": limit,
            "include_related": include_related,
            "inherit": inherit,
        })
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results:
        return "No relevant memories found."

    lines = []
    for i, r in enumerate(results, 1):
        score = f"{r['similarity'] * 100:.0f}%"
        tags = f" ({', '.join(r['tags'])})" if r.get("tags") else ""
        lines.append(f"{i}. [{score}] {r['content']}{tags}")

    return f"Found {len(results)} memories:\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: memory_store
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_store(
    content: str,
    container: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Save important information to long-term memory.

    Use for preferences, facts, decisions, entities, or anything worth
    remembering across sessions. Content is automatically chunked,
    embedded, and indexed for hybrid search.

    Args:
        content: Information to remember
        container: Container/namespace (default: configured container)
        tags: Tags for categorization (e.g. preference, decision, fact, entity)
    """
    with _client() as client:
        resp = client.post("/v1/documents", json={
            "container": container or DEFAULT_CONTAINER,
            "content": content,
            "content_type": "text",
            "tags": tags or [],
            "metadata": {"source": "mcp-server"},
        })
        resp.raise_for_status()
        data = resp.json()

    preview = content[:100] + ("..." if len(content) > 100 else "")
    return f'Stored memory: "{preview}" (id: {data["id"]}, status: {data["status"]})'


# ---------------------------------------------------------------------------
# Tool: memory_forget
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_forget(
    memory_id: str | None = None,
    query: str | None = None,
) -> str:
    """Delete a specific memory by ID, or search for memories to delete.

    If a query is provided and there's a single high-confidence match
    (>90% similarity), it's deleted automatically. Otherwise, candidates
    are listed for you to pick from.

    Args:
        memory_id: Specific memory UUID to delete
        query: Search query to find memories to delete
    """
    if not memory_id and not query:
        return "Provide either memory_id or query."

    with _client() as client:
        if memory_id:
            resp = client.delete(f"/v1/memories/{memory_id}")
            resp.raise_for_status()
            return f"Memory {memory_id} deleted."

        # Search first, then decide
        resp = client.post("/v1/search", json={
            "query": query,
            "container": DEFAULT_CONTAINER,
            "limit": 5,
        })
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if not results:
            return "No matching memories found to delete."

        # Single high-confidence match: auto-delete
        if len(results) == 1 and results[0]["similarity"] > 0.9:
            target = results[0]
            del_resp = client.delete(f"/v1/memories/{target['id']}")
            del_resp.raise_for_status()
            preview = target["content"][:80]
            return f'Deleted: "{preview}..." (id: {target["id"]})'

        # Multiple candidates: list them
        lines = []
        for r in results:
            score = f"{r['similarity'] * 100:.0f}%"
            preview = r["content"][:60]
            lines.append(f"- `{r['id']}` — {preview}... ({score})")

        return (
            f"Found {len(results)} candidates. Use memory_id to delete a specific one:\n"
            + "\n".join(lines)
        )


# ---------------------------------------------------------------------------
# Tool: memory_profile
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_profile(
    container: str | None = None,
    fact_limit: int = 20,
    recent_limit: int = 10,
) -> str:
    """Get a synthesized profile/summary for a memory container.

    Returns key facts, preferences, and recent memories. Useful for
    getting a quick overview of what's known about a user or topic.

    Args:
        container: Container to profile (default: configured container)
        fact_limit: Max facts to include (default: 20)
        recent_limit: Max recent memories (default: 10)
    """
    ctr = container or DEFAULT_CONTAINER
    with _client() as client:
        resp = client.get(
            f"/v1/profile/{ctr}",
            params={"fact_limit": fact_limit, "recent_limit": recent_limit},
        )
        resp.raise_for_status()
        data = resp.json()

    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool: memory_stats
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_stats() -> str:
    """Get system-wide statistics for the MemoryDB instance.

    Returns counts for documents, memories (by type), relationships,
    and containers. Useful for health checks and understanding the
    current state of the memory system.
    """
    with _client() as client:
        resp = client.get("/v1/stats")
        resp.raise_for_status()
        data = resp.json()

    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
