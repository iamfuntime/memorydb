#!/usr/bin/env python3
"""Ingest OpenClaw agent files into MemoryDB.

Discovers agent identities, memories, logs, configs, and system-level docs
from ~/agents/, ~/.openclaw/, and ~/CLAUDE.md, then POSTs each as a document
to the MemoryDB API.

Usage:
    python scripts/ingest_openclaw.py                  # full ingest
    python scripts/ingest_openclaw.py --dry-run         # preview without posting
    python scripts/ingest_openclaw.py --base-url URL    # custom endpoint
    python scripts/ingest_openclaw.py --delay 0.5       # seconds between requests
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ── Constants ────────────────────────────────────────────────────────────────

AGENTS_DIR = Path.home() / "agents"
OPENCLAW_DIR = Path.home() / ".openclaw"
CLAUDE_MD = Path.home() / "CLAUDE.md"

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_DELAY = 0.3
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

SKIP_AGENTS = {"beacon", "echo", "nexus"}

SKIP_SUBDIRS = {
    "epstein-files",
    "soul-guardian",
    "canvas",
    ".git",
}

SKIP_EXTENSIONS = {".pre-clawsec", ".sample", ".patch", ".zip", ".log", ".quarantine"}

SKIP_FILENAMES = {"cron-health.json"}

CONTENT_TYPE_MAP = {
    ".md": "text",
    ".py": "code",
    ".json": "json",
    ".yaml": "text",
    ".yml": "text",
    ".sh": "code",
}

OBSIDIAN_SUMMARY = (
    "# Obsidian Vault — Brain\n\n"
    "The user maintains an Obsidian vault at ~/Obsidian/Brain/ containing "
    "personal knowledge management notes, daily journals, project documentation, "
    "and reference material. This vault is the user's primary knowledge base "
    "outside of the agent ecosystem. It is NOT ingested into MemoryDB — only "
    "this summary exists here for cross-reference."
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class FileEntry:
    path: Path
    container: str
    content_type: str
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    content: str | None = None  # for synthetic entries


# ── Tag mapping ──────────────────────────────────────────────────────────────

# Patterns checked in order; first match wins for the "primary" tags.
# Additional tags are always appended (e.g. agent name, code).
TAG_RULES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"SOUL\.md$"), ["soul", "identity", "agent-config"]),
    (re.compile(r"MEMORY\.md$"), ["memory", "persistent", "agent-config"]),
    (re.compile(r"HEARTBEAT\.md$"), ["heartbeat", "monitoring", "agent-config"]),
    (re.compile(r"BOOTSTRAP\.md$"), ["bootstrap", "initialization", "agent-config"]),
    (re.compile(r"IDENTITY\.md$"), ["identity", "capabilities", "agent-config"]),
    (re.compile(r"AGENTS\.md$"), ["agents", "relationships", "agent-config"]),
    (re.compile(r"TOOLS\.md$"), ["tools", "capabilities", "agent-config"]),
    (re.compile(r"USER\.md$"), ["user", "preferences", "agent-config"]),
    (re.compile(r"ROUTING\.md$"), ["routing", "agent-config"]),
    (re.compile(r"SLACK-ROUTING\.md$"), ["routing", "agent-config"]),
    (re.compile(r"SESSION-.*\.md$"), ["session", "agent-config"]),
    (re.compile(r"memory/\d{4}-\d{2}-\d{2}"), ["memory", "daily-log"]),
    (re.compile(r"memory/.*\.md$"), ["memory", "reference"]),
    (re.compile(r"plans/.*\.md$"), ["plan", "strategy"]),
    (re.compile(r"research/.*\.md$"), ["research", "analysis"]),
    (re.compile(r"docs/.*\.md$"), ["documentation"]),
    (re.compile(r"siem-rules/.*\.md$"), ["siem", "detection-rules"]),
    (re.compile(r"(?i)(REPORT|TASK).*\.md$"), ["report"]),
    (re.compile(r"\.(py|sh)$"), ["code", "script"]),
]


def tag_file(path: Path, agent_name: str | None) -> list[str]:
    """Determine tags for a file based on its name and path."""
    name = path.name
    rel = str(path)

    tags: list[str] = []

    for pattern, pattern_tags in TAG_RULES:
        if pattern.search(rel):
            tags = list(pattern_tags)
            break

    if not tags:
        if path.suffix == ".md":
            tags = ["document"]
        elif path.suffix in (".json", ".yaml", ".yml"):
            tags = ["config"]
        else:
            tags = ["document"]

    # Extract date from daily log filenames like 2026-02-07.md
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if date_match and "daily-log" in tags:
        tags.append(date_match.group(1))

    return tags


# ── File discovery ───────────────────────────────────────────────────────────


def _should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped based on skip rules."""
    parts = set(path.parts)
    if parts & SKIP_SUBDIRS:
        return True
    if path.suffix in SKIP_EXTENSIONS:
        return True
    if path.name in SKIP_FILENAMES:
        return True
    if not path.is_file():
        return True
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return True
        if path.stat().st_size == 0:
            return True
    except OSError:
        return True
    return False


def discover_system_files() -> list[FileEntry]:
    """Discover system-level files for the 'openclaw' container."""
    entries: list[FileEntry] = []

    # ~/CLAUDE.md
    if CLAUDE_MD.exists() and CLAUDE_MD.stat().st_size > 0:
        entries.append(FileEntry(
            path=CLAUDE_MD,
            container="openclaw",
            content_type="text",
            tags=["system", "instructions", "claude-md"],
            metadata={"source_path": str(CLAUDE_MD)},
        ))

    return entries


def generate_obsidian_summary() -> FileEntry:
    """Create a synthetic entry summarizing the Obsidian vault."""
    return FileEntry(
        path=Path("~/Obsidian/Brain/"),
        container="openclaw",
        content_type="text",
        tags=["obsidian", "vault", "summary"],
        metadata={"source_path": "~/Obsidian/Brain/", "synthetic": True},
        content=OBSIDIAN_SUMMARY,
    )


def _discover_agent_dir(agent_dir: Path, agent_name: str) -> list[FileEntry]:
    """Walk an agent directory and collect FileEntry objects."""
    entries: list[FileEntry] = []
    container = f"openclaw.{agent_name}"

    if not agent_dir.is_dir():
        return entries

    for path in sorted(agent_dir.rglob("*")):
        if _should_skip_path(path):
            continue

        content_type = CONTENT_TYPE_MAP.get(path.suffix, "text")

        entries.append(FileEntry(
            path=path,
            container=container,
            content_type=content_type,
            tags=tag_file(path, agent_name),
            metadata={"agent": agent_name, "source_path": str(path)},
        ))

    return entries


def _discover_openclaw_workspace(agent_name: str) -> list[FileEntry]:
    """Discover workspace config files for an agent from ~/.openclaw/workspace-{name}/."""
    entries: list[FileEntry] = []
    workspace_dir = OPENCLAW_DIR / f"workspace-{agent_name}"
    container = f"openclaw.{agent_name}"

    if not workspace_dir.is_dir():
        return entries

    for path in sorted(workspace_dir.rglob("*")):
        if _should_skip_path(path):
            continue

        content_type = CONTENT_TYPE_MAP.get(path.suffix, "text")

        entries.append(FileEntry(
            path=path,
            container=container,
            content_type=content_type,
            tags=tag_file(path, agent_name) + ["workspace-config"],
            metadata={"agent": agent_name, "source_path": str(path)},
        ))

    return entries


def _discover_openclaw_agent_config(agent_name: str) -> list[FileEntry]:
    """Discover agent-level config files from ~/.openclaw/agents/{name}/agent/.

    Skips .json files (may contain auth credentials).
    """
    entries: list[FileEntry] = []
    agent_config_dir = OPENCLAW_DIR / "agents" / agent_name / "agent"
    container = f"openclaw.{agent_name}"

    if not agent_config_dir.is_dir():
        return entries

    for path in sorted(agent_config_dir.rglob("*")):
        if _should_skip_path(path):
            continue
        # Skip JSON and auth-related files in .openclaw (auth safety)
        if path.suffix == ".json" or "auth-" in path.name:
            continue

        content_type = CONTENT_TYPE_MAP.get(path.suffix, "text")

        entries.append(FileEntry(
            path=path,
            container=container,
            content_type=content_type,
            tags=tag_file(path, agent_name) + ["agent-config"],
            metadata={"agent": agent_name, "source_path": str(path)},
        ))

    return entries


def discover_agent_files() -> list[FileEntry]:
    """Discover all per-agent files across all three source locations."""
    entries: list[FileEntry] = []

    if not AGENTS_DIR.is_dir():
        print(f"WARNING: Agents directory not found: {AGENTS_DIR}")
        return entries

    agent_names = sorted(
        d.name for d in AGENTS_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP_AGENTS
    )

    for name in agent_names:
        agent_dir = AGENTS_DIR / name

        # Check if agent dir has any files (skip truly empty ones)
        files = list(agent_dir.rglob("*"))
        real_files = [f for f in files if f.is_file()]
        if not real_files:
            print(f"  Skipping empty agent: {name}")
            continue

        # Source 1: ~/agents/{name}/
        entries.extend(_discover_agent_dir(agent_dir, name))

        # Source 2: ~/.openclaw/workspace-{name}/
        entries.extend(_discover_openclaw_workspace(name))

        # Source 3: ~/.openclaw/agents/{name}/agent/
        entries.extend(_discover_openclaw_agent_config(name))

    return entries


# ── API interaction ──────────────────────────────────────────────────────────


def health_check(client: httpx.Client, base_url: str) -> bool:
    """Verify the MemoryDB API is reachable and the database is connected."""
    try:
        resp = client.get(f"{base_url}/health")
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.ConnectError) as e:
        print(f"ERROR: Cannot reach MemoryDB at {base_url}: {e}")
        return False

    # Check database connectivity via stats endpoint
    try:
        resp = client.get(f"{base_url}/v1/stats")
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        print(f"ERROR: MemoryDB database not connected. Start PostgreSQL first.")
        return False

    return True


def post_document(
    client: httpx.Client,
    base_url: str,
    entry: FileEntry,
    dry_run: bool,
) -> bool:
    """POST a single document to the MemoryDB API. Returns True on success."""
    # Read content
    if entry.content is not None:
        content = entry.content
    else:
        try:
            content = entry.path.read_text(errors="replace")
        except OSError as e:
            print(f"  ERROR reading {entry.path}: {e}")
            return False

    if not content.strip():
        return False

    payload = {
        "content": content,
        "container": entry.container,
        "content_type": entry.content_type,
        "metadata": entry.metadata,
        "tags": entry.tags,
    }

    if dry_run:
        return True

    try:
        resp = client.post(f"{base_url}/v1/documents", json=payload, timeout=30.0)
        if resp.status_code == 202:
            doc = resp.json()
            return True
        else:
            print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
            return False
    except httpx.HTTPError as e:
        print(f"  ERROR posting {entry.path}: {e}")
        return False


# ── Summary ──────────────────────────────────────────────────────────────────


def print_discovery_summary(entries: list[FileEntry]) -> None:
    """Print a summary of discovered files grouped by container."""
    containers: dict[str, int] = {}
    for e in entries:
        containers[e.container] = containers.get(e.container, 0) + 1

    print("\n── Discovery Summary ──────────────────────────────────────")
    print(f"  Total files: {len(entries)}")
    print()
    for container in sorted(containers):
        print(f"  {container}: {containers[container]} files")
    print()


def print_final_summary(
    total: int, success: int, failed: int, failures: list[str]
) -> None:
    """Print final ingestion results."""
    print("\n── Ingestion Summary ─────────────────────────────────────")
    print(f"  Total:    {total}")
    print(f"  Success:  {success}")
    print(f"  Failed:   {failed}")
    if failures:
        print("\n  Failed files:")
        for f in failures:
            print(f"    - {f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest OpenClaw agent files into MemoryDB"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview file discovery without posting to the API",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"MemoryDB API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between API requests (default: {DEFAULT_DELAY})",
    )
    args = parser.parse_args()

    print("OpenClaw -> MemoryDB Ingestion")
    print("=" * 50)

    if not args.dry_run:
        print(
            "\nWARNING: Re-running ingestion creates duplicate documents.\n"
            "If re-ingesting, clear containers first:\n"
            f'  curl -X DELETE "{args.base_url}/v1/containers/openclaw?confirm=true"\n'
        )

    # ── Discover files ──
    print("Discovering files...")

    entries: list[FileEntry] = []
    entries.extend(discover_system_files())
    entries.append(generate_obsidian_summary())
    entries.extend(discover_agent_files())

    print_discovery_summary(entries)

    if not entries:
        print("No files discovered. Nothing to do.")
        sys.exit(0)

    if args.dry_run:
        print("DRY RUN — no documents posted.")
        print("\nFile details:")
        for e in entries:
            src = "(synthetic)" if e.content is not None else str(e.path)
            print(f"  [{e.container}] {e.content_type:5s} {src}")
            print(f"           tags: {e.tags}")
        sys.exit(0)

    # ── Health check ──
    client = httpx.Client()
    if not health_check(client, args.base_url):
        sys.exit(1)

    print(f"API healthy. Posting {len(entries)} documents...\n")

    # ── Ingest ──
    success = 0
    failed = 0
    failures: list[str] = []

    for i, entry in enumerate(entries, 1):
        label = "(synthetic)" if entry.content is not None else str(entry.path)
        short = label if len(label) < 70 else "..." + label[-67:]
        print(f"  [{i}/{len(entries)}] {entry.container} <- {short}", end="", flush=True)

        ok = post_document(client, args.base_url, entry, dry_run=False)
        if ok:
            success += 1
            print(" OK")
        else:
            failed += 1
            failures.append(label)
            print(" FAIL")

        if i < len(entries):
            time.sleep(args.delay)

    client.close()

    print_final_summary(len(entries), success, failed, failures)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
