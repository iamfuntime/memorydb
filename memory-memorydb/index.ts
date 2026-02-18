/**
 * OpenClaw Plugin: Memory (MemoryDB)
 *
 * Connects OpenClaw to a self-hosted MemoryDB REST API for long-term
 * agent memory with hybrid search (vector + BM25), auto-recall, and
 * auto-capture.
 *
 * MemoryDB API docs: POST /v1/documents, POST /v1/search, GET /v1/profile/:container
 * DELETE /v1/memories/:id, GET /v1/memories/:id
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";

// ============================================================================
// Config
// ============================================================================

type MemoryDBConfig = {
  baseUrl: string;
  container: string;
  autoCapture: boolean;
  autoRecall: boolean;
  autoRecallLimit: number;
  conversationLog: boolean;
  sessionProfile: boolean;
  inherit: boolean;
  apiKey?: string;
};

function parseConfig(raw: Record<string, unknown> | undefined): MemoryDBConfig {
  return {
    baseUrl: (raw?.baseUrl as string) ?? "http://localhost:8080",
    container: (raw?.container as string) ?? "default",
    autoCapture: (raw?.autoCapture as boolean) ?? false,
    autoRecall: (raw?.autoRecall as boolean) ?? false,
    autoRecallLimit: (raw?.autoRecallLimit as number) ?? 5,
    conversationLog: (raw?.conversationLog as boolean) ?? true,
    sessionProfile: (raw?.sessionProfile as boolean) ?? true,
    inherit: (raw?.inherit as boolean) ?? true,
    apiKey: raw?.apiKey as string | undefined,
  };
}

// ============================================================================
// HTTP Client
// ============================================================================

class MemoryDBClient {
  constructor(
    private baseUrl: string,
    private apiKey?: string,
  ) {
    // Strip trailing slash
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  async ingest(container: string, content: string, tags: string[] = [], metadata: Record<string, unknown> = {}): Promise<{ id: string; status: string }> {
    const res = await fetch(`${this.baseUrl}/v1/documents`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        container,
        content,
        content_type: "text",
        tags,
        metadata,
      }),
    });
    if (!res.ok) throw new Error(`MemoryDB ingest failed: ${res.status} ${await res.text()}`);
    return res.json();
  }

  async search(query: string, container?: string, limit = 10, includeRelated = false, inherit = false): Promise<{
    query: string;
    results: Array<{
      id: string;
      container: string;
      content: string;
      memory_type: string;
      similarity: number;
      tags: string[];
      created_at: string;
      related: unknown[];
    }>;
    total: number;
  }> {
    const res = await fetch(`${this.baseUrl}/v1/search`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        query,
        container,
        limit,
        include_related: includeRelated,
        inherit,
      }),
    });
    if (!res.ok) throw new Error(`MemoryDB search failed: ${res.status} ${await res.text()}`);
    return res.json();
  }

  async getMemory(memoryId: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}/v1/memories/${memoryId}`, {
      headers: this.headers(),
    });
    if (!res.ok) throw new Error(`MemoryDB get failed: ${res.status} ${await res.text()}`);
    return res.json();
  }

  async deleteMemory(memoryId: string): Promise<{ id: string; deleted: boolean }> {
    const res = await fetch(`${this.baseUrl}/v1/memories/${memoryId}`, {
      method: "DELETE",
      headers: this.headers(),
    });
    if (!res.ok) throw new Error(`MemoryDB delete failed: ${res.status} ${await res.text()}`);
    return res.json();
  }

  async getProfile(container: string, factLimit = 20, recentLimit = 10): Promise<Record<string, unknown>> {
    const res = await fetch(
      `${this.baseUrl}/v1/profile/${encodeURIComponent(container)}?fact_limit=${factLimit}&recent_limit=${recentLimit}`,
      { headers: this.headers() },
    );
    if (!res.ok) throw new Error(`MemoryDB profile failed: ${res.status} ${await res.text()}`);
    return res.json();
  }

  async getStats(): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}/v1/stats`, { headers: this.headers() });
    if (!res.ok) throw new Error(`MemoryDB stats failed: ${res.status} ${await res.text()}`);
    return res.json();
  }

  async health(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, { headers: this.headers() });
      return res.ok;
    } catch {
      return false;
    }
  }
}

// ============================================================================
// Capture heuristics (rule-based, no LLM needed)
// ============================================================================

const CAPTURE_TRIGGERS = [
  /remember\b/i,
  /\bprefer\b|\brather\b|\bdon't like\b|\bi like\b|\bi love\b|\bi hate\b/i,
  /\bdecided\b|\bwe'll use\b|\bgoing with\b|\bswitching to\b/i,
  /\bmy .{2,20} is\b/i,
  /\balways\b|\bnever\b|\bimportant\b/i,
  /[\w.-]+@[\w.-]+\.\w+/,        // emails
  /\+\d{10,}/,                     // phone numbers
  /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/, // IPs
];

function shouldCapture(text: string): boolean {
  if (text.length < 10 || text.length > 1000) return false;
  // Skip system/agent boilerplate
  if (text.includes("<relevant-memories>")) return false;
  if (text.startsWith("<") && text.includes("</")) return false;
  if (text.includes("HEARTBEAT_OK")) return false;
  if (text.includes("NO_REPLY")) return false;
  return CAPTURE_TRIGGERS.some((r) => r.test(text));
}

function detectTags(text: string): string[] {
  const tags: string[] = [];
  const lower = text.toLowerCase();
  if (/prefer|rather|like|love|hate|want/i.test(lower)) tags.push("preference");
  if (/decided|will use|going with|switching/i.test(lower)) tags.push("decision");
  if (/[\w.-]+@[\w.-]+\.\w+|\+\d{10,}|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/i.test(lower)) tags.push("entity");
  if (/my .{2,20} is|always|never/i.test(lower)) tags.push("fact");
  if (/remember/i.test(lower)) tags.push("explicit");
  if (tags.length === 0) tags.push("general");
  return tags;
}

// ============================================================================
// Plugin
// ============================================================================

const memoryDBPlugin = {
  id: "memory-memorydb",
  name: "Memory (MemoryDB)",
  description: "Self-hosted AI agent memory via MemoryDB REST API",
  kind: "memory" as const,

  register(api: OpenClawPluginApi) {
    const cfg = parseConfig(api.pluginConfig);
    const client = new MemoryDBClient(cfg.baseUrl, cfg.apiKey);

    api.logger.info(`memory-memorydb: registered (url: ${cfg.baseUrl}, container: ${cfg.container})`);

    // ========================================================================
    // Tools
    // ========================================================================

    api.registerTool(
      {
        name: "memory_search",
        label: "Memory Search",
        description:
          "Search long-term memories using hybrid search (semantic + keyword). " +
          "Use when you need context about user preferences, past decisions, " +
          "previously discussed topics, or stored facts.",
        parameters: Type.Object({
          query: Type.String({ description: "Natural language search query" }),
          container: Type.Optional(
            Type.String({ description: "Container to search (default: plugin config)" }),
          ),
          limit: Type.Optional(
            Type.Number({ description: "Max results (default: 10)" }),
          ),
          include_related: Type.Optional(
            Type.Boolean({ description: "Include graph-related memories (default: false)" }),
          ),
          inherit: Type.Optional(
            Type.Boolean({ description: "Search parent containers too (default: config inherit setting)" }),
          ),
        }),
        async execute(_id, params) {
          const {
            query,
            container = cfg.container,
            limit = 10,
            include_related = false,
            inherit = cfg.inherit,
          } = params as {
            query: string;
            container?: string;
            limit?: number;
            include_related?: boolean;
            inherit?: boolean;
          };

          try {
            const result = await client.search(query, container, limit, include_related, inherit);

            if (result.results.length === 0) {
              return {
                content: [{ type: "text", text: "No relevant memories found." }],
                details: { count: 0 },
              };
            }

            const text = result.results
              .map(
                (r, i) =>
                  `${i + 1}. [${(r.similarity * 100).toFixed(0)}%] ${r.content}` +
                  (r.tags.length ? ` (${r.tags.join(", ")})` : ""),
              )
              .join("\n");

            return {
              content: [
                {
                  type: "text",
                  text: `Found ${result.results.length} memories:\n\n${text}`,
                },
              ],
              details: {
                count: result.results.length,
                memories: result.results.map((r) => ({
                  id: r.id,
                  content: r.content,
                  similarity: r.similarity,
                  tags: r.tags,
                  memory_type: r.memory_type,
                })),
              },
            };
          } catch (err) {
            return {
              content: [
                { type: "text", text: `Memory search failed: ${String(err)}` },
              ],
            };
          }
        },
      },
      { name: "memory_search" },
    );

    api.registerTool(
      {
        name: "memory_store",
        label: "Memory Store",
        description:
          "Save important information to long-term memory. Use for preferences, " +
          "facts, decisions, entities, or anything worth remembering across sessions. " +
          "Content is automatically chunked, embedded, and indexed for hybrid search.",
        parameters: Type.Object({
          content: Type.String({ description: "Information to remember" }),
          container: Type.Optional(
            Type.String({ description: "Container/namespace (default: plugin config)" }),
          ),
          tags: Type.Optional(
            Type.Array(Type.String(), {
              description: "Tags for categorization (e.g. preference, decision, fact)",
            }),
          ),
        }),
        async execute(_id, params) {
          const {
            content,
            container = cfg.container,
            tags = [],
          } = params as {
            content: string;
            container?: string;
            tags?: string[];
          };

          try {
            const result = await client.ingest(container, content, tags);
            return {
              content: [
                {
                  type: "text",
                  text: `Stored memory: "${content.slice(0, 100)}${content.length > 100 ? "..." : ""}"`,
                },
              ],
              details: { id: result.id, status: result.status },
            };
          } catch (err) {
            return {
              content: [
                { type: "text", text: `Memory store failed: ${String(err)}` },
              ],
            };
          }
        },
      },
      { name: "memory_store" },
    );

    api.registerTool(
      {
        name: "memory_forget",
        label: "Memory Forget",
        description:
          "Delete a specific memory by ID or search for memories to delete. GDPR-compliant.",
        parameters: Type.Object({
          memory_id: Type.Optional(
            Type.String({ description: "Specific memory UUID to delete" }),
          ),
          query: Type.Optional(
            Type.String({
              description: "Search query to find memories to delete (shows candidates if ambiguous)",
            }),
          ),
        }),
        async execute(_id, params) {
          const { memory_id, query } = params as {
            memory_id?: string;
            query?: string;
          };

          try {
            if (memory_id) {
              const result = await client.deleteMemory(memory_id);
              return {
                content: [
                  { type: "text", text: `Memory ${memory_id} deleted.` },
                ],
                details: result,
              };
            }

            if (query) {
              const results = await client.search(query, cfg.container, 5);

              if (results.results.length === 0) {
                return {
                  content: [
                    { type: "text", text: "No matching memories found to delete." },
                  ],
                };
              }

              // If one high-confidence match, delete it
              if (results.results.length === 1 && results.results[0].similarity > 0.9) {
                const target = results.results[0];
                await client.deleteMemory(target.id);
                return {
                  content: [
                    {
                      type: "text",
                      text: `Deleted: "${target.content.slice(0, 80)}..."`,
                    },
                  ],
                  details: { id: target.id, deleted: true },
                };
              }

              // Multiple candidates — show list
              const list = results.results
                .map(
                  (r) =>
                    `- \`${r.id}\` — ${r.content.slice(0, 60)}... (${(r.similarity * 100).toFixed(0)}%)`,
                )
                .join("\n");

              return {
                content: [
                  {
                    type: "text",
                    text: `Found ${results.results.length} candidates. Use memory_id to delete a specific one:\n${list}`,
                  },
                ],
                details: {
                  candidates: results.results.map((r) => ({
                    id: r.id,
                    content: r.content.slice(0, 100),
                    similarity: r.similarity,
                  })),
                },
              };
            }

            return {
              content: [
                { type: "text", text: "Provide either memory_id or query." },
              ],
            };
          } catch (err) {
            return {
              content: [
                { type: "text", text: `Memory forget failed: ${String(err)}` },
              ],
            };
          }
        },
      },
      { name: "memory_forget" },
    );

    api.registerTool(
      {
        name: "memory_profile",
        label: "Memory Profile",
        description:
          "Get a synthesized profile/summary for a container — key facts, " +
          "preferences, and recent memories. Useful for getting a quick overview " +
          "of what's known about a user or topic.",
        parameters: Type.Object({
          container: Type.Optional(
            Type.String({ description: "Container to profile (default: plugin config)" }),
          ),
          fact_limit: Type.Optional(
            Type.Number({ description: "Max facts to include (default: 20)" }),
          ),
          recent_limit: Type.Optional(
            Type.Number({ description: "Max recent memories (default: 10)" }),
          ),
        }),
        async execute(_id, params) {
          const {
            container = cfg.container,
            fact_limit = 20,
            recent_limit = 10,
          } = params as {
            container?: string;
            fact_limit?: number;
            recent_limit?: number;
          };

          try {
            const profile = await client.getProfile(container, fact_limit, recent_limit);
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify(profile, null, 2),
                },
              ],
              details: profile,
            };
          } catch (err) {
            return {
              content: [
                { type: "text", text: `Memory profile failed: ${String(err)}` },
              ],
            };
          }
        },
      },
      { name: "memory_profile" },
    );

    // ========================================================================
    // CLI Commands
    // ========================================================================

    api.registerCli(
      ({ program }) => {
        const mem = program.command("memorydb").description("MemoryDB memory plugin commands");

        mem
          .command("status")
          .description("Check MemoryDB connection and stats")
          .action(async () => {
            const healthy = await client.health();
            console.log(`MemoryDB: ${healthy ? "✅ connected" : "❌ unreachable"} (${cfg.baseUrl})`);
            if (healthy) {
              try {
                const stats = await client.getStats();
                console.log(JSON.stringify(stats, null, 2));
              } catch (e) {
                console.log(`Stats unavailable: ${e}`);
              }
            }
          });

        mem
          .command("search")
          .description("Search memories")
          .argument("<query>", "Search query")
          .option("--container <name>", "Container", cfg.container)
          .option("--limit <n>", "Max results", "10")
          .action(async (query, opts) => {
            const result = await client.search(query, opts.container, parseInt(opts.limit));
            console.log(JSON.stringify(result, null, 2));
          });

        mem
          .command("profile")
          .description("Show container profile")
          .argument("[container]", "Container name", cfg.container)
          .action(async (container) => {
            const profile = await client.getProfile(container);
            console.log(JSON.stringify(profile, null, 2));
          });
      },
      { commands: ["memorydb"] },
    );

    // ========================================================================
    // Auto-reply command: /memory
    // ========================================================================

    api.registerCommand({
      name: "memory",
      description: "Quick memory search (e.g. /memory what does Seth prefer)",
      acceptsArgs: true,
      handler: async (ctx) => {
        if (!ctx.args?.trim()) {
          const healthy = await client.health();
          return {
            text: `🧠 MemoryDB: ${healthy ? "✅ connected" : "❌ unreachable"}\nURL: ${cfg.baseUrl}\nContainer: ${cfg.container}\nAuto-recall: ${cfg.autoRecall}\nAuto-capture: ${cfg.autoCapture}`,
          };
        }
        try {
          const result = await client.search(ctx.args.trim(), cfg.container, 5);
          if (result.results.length === 0) return { text: "No memories found." };
          const lines = result.results.map(
            (r, i) => `${i + 1}. [${(r.similarity * 100).toFixed(0)}%] ${r.content.slice(0, 120)}`,
          );
          return { text: `🧠 ${result.results.length} memories:\n${lines.join("\n")}` };
        } catch (err) {
          return { text: `Memory search failed: ${err}` };
        }
      },
    });

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    // Session profile cache: profile text keyed by sessionId (one-shot inject)
    const sessionProfiles = new Map<string, string>();

    // -- message_received: log inbound messages --
    if (cfg.conversationLog) {
      api.on("message_received", async (event, ctx) => {
        try {
          await client.ingest(cfg.container, event.content,
            ["conversation", "inbound", ctx.channelId],
            {
              source: "conversation-log",
              channel: ctx.channelId,
              from: event.from,
              conversationId: ctx.conversationId,
              timestamp: event.timestamp ?? Date.now(),
            },
          );
        } catch (err) {
          api.logger.warn(`memory-memorydb: conversation log (inbound) failed: ${String(err)}`);
        }
      });
    }

    // -- message_sent: log outbound replies --
    if (cfg.conversationLog) {
      api.on("message_sent", async (event, ctx) => {
        if (!event.success) return;
        try {
          await client.ingest(cfg.container, event.content,
            ["conversation", "outbound", ctx.channelId],
            {
              source: "conversation-log",
              channel: ctx.channelId,
              to: event.to,
              conversationId: ctx.conversationId,
            },
          );
        } catch (err) {
          api.logger.warn(`memory-memorydb: conversation log (outbound) failed: ${String(err)}`);
        }
      });
    }

    // -- session_start: preload profile --
    if (cfg.sessionProfile) {
      api.on("session_start", async (event) => {
        try {
          const profile = await client.getProfile(cfg.container, 10, 5) as {
            static_facts?: Array<{ content: string }>;
            recent_context?: Array<{ content: string }>;
          };

          const parts: string[] = [];

          if (profile.static_facts?.length) {
            parts.push("Key facts:");
            for (const f of profile.static_facts) {
              parts.push(`- ${f.content}`);
            }
          }
          if (profile.recent_context?.length) {
            parts.push("Recent context:");
            for (const r of profile.recent_context) {
              parts.push(`- ${r.content}`);
            }
          }

          if (parts.length > 0) {
            sessionProfiles.set(event.sessionId, parts.join("\n"));
            api.logger.info(`memory-memorydb: cached profile for session ${event.sessionId}`);
          }
        } catch (err) {
          api.logger.warn(`memory-memorydb: session profile preload failed: ${String(err)}`);
        }
      });
    }

    // -- before_agent_start: always register, combine profile + auto-recall --
    api.on("before_agent_start", async (event) => {
      const blocks: string[] = [];

      // Profile injection (one-shot per session)
      if (cfg.sessionProfile && event.sessionId) {
        const profileText = sessionProfiles.get(event.sessionId);
        if (profileText) {
          blocks.push(
            `<agent-profile>\n${profileText}\n</agent-profile>`,
          );
          sessionProfiles.delete(event.sessionId);
          api.logger.info("memory-memorydb: injecting profile into context");
        }
      }

      // Auto-recall
      if (cfg.autoRecall && event.prompt && event.prompt.length >= 5) {
        try {
          const result = await client.search(
            event.prompt,
            cfg.container,
            cfg.autoRecallLimit,
            false,
            cfg.inherit,
          );

          if (result.results.length > 0) {
            const memoryContext = result.results
              .map(
                (r) =>
                  `- [${(r.similarity * 100).toFixed(0)}%] ${r.content}` +
                  (r.tags.length ? ` (${r.tags.join(", ")})` : ""),
              )
              .join("\n");

            blocks.push(
              `<relevant-memories>\n` +
              `The following memories from long-term storage may be relevant:\n` +
              `${memoryContext}\n` +
              `</relevant-memories>`,
            );

            api.logger.info(
              `memory-memorydb: injecting ${result.results.length} memories into context`,
            );
          }
        } catch (err) {
          api.logger.warn(`memory-memorydb: auto-recall failed: ${String(err)}`);
        }
      }

      if (blocks.length > 0) {
        return { prependContext: blocks.join("\n\n") };
      }
    });

    // -- before_compaction: archive context before compression --
    if (cfg.conversationLog) {
      api.on("before_compaction", async (event, ctx) => {
        try {
          await client.ingest(cfg.container,
            `Session compacted. ${event.messageCount} messages, ${event.compactingCount ?? "?"} compacted.`,
            ["session", "compaction"],
            {
              source: "session-lifecycle",
              sessionKey: ctx.sessionKey,
              messageCount: event.messageCount,
            },
          );
        } catch (err) {
          api.logger.warn(`memory-memorydb: compaction log failed: ${String(err)}`);
        }
      });
    }

    // -- session_end: cleanup cached profile + log session end --
    api.on("session_end", async (event, ctx) => {
      sessionProfiles.delete(event.sessionId);

      if (!cfg.conversationLog) return;

      try {
        const mins = event.durationMs ? Math.round(event.durationMs / 60000) : null;
        await client.ingest(cfg.container,
          `Session ended. ${event.messageCount} messages` + (mins ? `, ${mins} min` : ""),
          ["session", "lifecycle"],
          {
            source: "session-lifecycle",
            sessionId: event.sessionId,
            agentId: ctx.agentId,
            messageCount: event.messageCount,
            durationMs: event.durationMs,
          },
        );
      } catch (err) {
        api.logger.warn(`memory-memorydb: session end log failed: ${String(err)}`);
      }
    });

    // Auto-capture: analyze user messages and store important info
    if (cfg.autoCapture) {
      api.on("agent_end", async (event) => {
        if (!event.success || !event.messages || event.messages.length === 0) return;

        try {
          const texts: string[] = [];
          for (const msg of event.messages) {
            if (!msg || typeof msg !== "object") continue;
            const msgObj = msg as Record<string, unknown>;

            // Only capture from user messages (not agent output)
            if (msgObj.role !== "user") continue;

            const content = msgObj.content;
            if (typeof content === "string") {
              texts.push(content);
            } else if (Array.isArray(content)) {
              for (const block of content) {
                if (
                  block &&
                  typeof block === "object" &&
                  "type" in block &&
                  (block as Record<string, unknown>).type === "text" &&
                  "text" in block &&
                  typeof (block as Record<string, unknown>).text === "string"
                ) {
                  texts.push((block as Record<string, unknown>).text as string);
                }
              }
            }
          }

          const toCapture = texts.filter(shouldCapture);
          if (toCapture.length === 0) return;

          let stored = 0;
          for (const text of toCapture.slice(0, 3)) {
            const tags = detectTags(text);
            await client.ingest(cfg.container, text, tags, {
              source: "auto-capture",
            });
            stored++;
          }

          if (stored > 0) {
            api.logger.info(`memory-memorydb: auto-captured ${stored} memories`);
          }
        } catch (err) {
          api.logger.warn(`memory-memorydb: auto-capture failed: ${String(err)}`);
        }
      });
    }

    // ========================================================================
    // Service (health monitoring)
    // ========================================================================

    api.registerService({
      id: "memory-memorydb",
      start: async () => {
        const healthy = await client.health();
        if (healthy) {
          api.logger.info(
            `memory-memorydb: connected to ${cfg.baseUrl} (container: ${cfg.container})`,
          );
        } else {
          api.logger.warn(
            `memory-memorydb: MemoryDB unreachable at ${cfg.baseUrl} — memories will fail until API is available`,
          );
        }
      },
      stop: () => {
        api.logger.info("memory-memorydb: stopped");
      },
    });
  },
};

export default memoryDBPlugin;
