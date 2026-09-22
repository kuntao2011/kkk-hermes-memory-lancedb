"""LanceDB Vector Memory Provider — HNSW-indexed semantic search via Ollama bge-m3.

Stores conversation turns / facts as vector embeddings using Ollama's /api/embed endpoint
(bge-m3:567m), then retrieves using HNSW ANN index via LanceDB. All embedding runs
locally against a running Ollama instance — no external API needed.

Requires: lancedb (pip install lancedb --python ~/.hermes/venv/bin/python3)
Config (memory.provider = lancedb-embed, in $HERMES_HOME/config.yaml):
  plugins.lancedb-embed:
    base_url: http://localhost:11434        # Ollama server base URL
    embedding_model: bge-m3:567m           # embedding model to use
    vector_dim: 1024                        # bge-m3 outputs 1024-dim vectors
    lance_dir: $HERMES_HOME/lance_memory    # LanceDB data directory
    batch_size: 32                           # max texts per embedding batch
    search_top_k: 5                         # top-k results to return per query
    min_content_len: 50                      # skip content shorter than this
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_EMBEDDING_TIMEOUT = 60  # seconds per batch
VECTOR_DIM = 1024  # bge-m3:567m


# ---------------------------------------------------------------------------
# Ollama /api/embed helper (no Python ollama package needed)
# ---------------------------------------------------------------------------

def _ollama_embed(texts: List[str], base_url: str, model: str) -> List[np.ndarray]:
    """Call Ollama /api/embed for batch embeddings. Returns list of np.float32 arrays."""
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/embed",
        json={"model": model, "input": texts},
        timeout=_EMBEDDING_TIMEOUT * 2,
    )
    resp.raise_for_status()
    return [np.array(e, dtype=np.float32) for e in resp.json()["embeddings"]]


def _ollama_embed_single(text: str, base_url: str, model: str) -> np.ndarray:
    """Embed a single text."""
    return _ollama_embed([text], base_url, model)[0]


# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------

def _get_lance_db(lance_dir: str):
    """Lazily import and connect to LanceDB."""
    import lancedb
    return lancedb.connect(lance_dir)


def _build_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("id",         pa.string()),
        pa.field("content",    pa.string()),
        pa.field("role",       pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("vector",     pa.list_(pa.float32(), VECTOR_DIM)),
        pa.field("created_at", pa.float64()),
        pa.field("metadata",   pa.string()),
    ])


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

MEMORY_STORE_SCHEMA = {
    "name": "vec_memory_add",
    "description": (
        "Store a piece of information in vector memory for semantic retrieval. "
        "Use to remember facts, decisions, preferences, commands, or any content "
        "you want to recall later via natural-language queries.\n\n"
        "The content is embedded with bge-m3:567m into a 1024-dim vector and "
        "stored in LanceDB with HNSW index. Retrieval uses HNSW ANN search.\n\n"
        "Use vec_memory_search to retrieve, vec_memory_list to view stored items."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "role": {"type": "string", "description": "Role context: 'user', 'assistant', 'system' (default: '')."},
            "session_id": {"type": "string", "description": "Session this content belongs to (default: current session)."},
            "metadata": {"type": "string", "description": "Optional JSON metadata string."},
        },
        "required": ["content"],
    },
}

MEMORY_SEARCH_SCHEMA = {
    "name": "vec_memory_search",
    "description": (
        "Semantic search over stored vector memory using natural-language query. "
        "Converts the query to a vector with bge-m3:567m and returns the most "
        "similar stored items via hybrid search (vector + full-text with RRF fusion). "
        "Use for recalling facts, "
        "decisions, preferences, and any stored context from previous sessions.\n\n"
        "Supports time-range filtering via after_timestamp / before_timestamp (Unix timestamps)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "top_k": {"type": "integer", "description": "Max results to return (default: 5, max: 20)."},
            "min_score": {"type": "number", "description": "Minimum cosine similarity threshold (0-1, default: 0.0)."},
            "session_id": {"type": "string", "description": "Limit search to a specific session (optional)."},
            "after_timestamp": {"type": "number", "description": "Unix timestamp — only return results with created_at >= this (optional)."},
            "before_timestamp": {"type": "number", "description": "Unix timestamp — only return results with created_at <= this (optional)."},
            "mode": {
                "type": "string",
                "enum": ["hybrid", "vector", "keyword"],
                "description": "Search mode: 'hybrid' (default) combines vector + keyword, 'vector' for semantic-only, 'keyword' for full-text-only."
            },
        },
        "required": ["query"],
    },
}

MEMORY_LIST_SCHEMA = {
    "name": "vec_memory_list",
    "description": (
        "List all stored memory items, optionally filtered by session. "
        "Returns id, content preview, role, session_id, timestamp, and similarity score (if searched).\n\n"
        "Supports time-range filtering via after_timestamp / before_timestamp (Unix timestamps)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max items to return (default: 20, max: 100)."},
            "session_id": {"type": "string", "description": "Filter by session ID (optional)."},
            "after_timestamp": {"type": "number", "description": "Unix timestamp — only return results with created_at >= this (optional)."},
            "before_timestamp": {"type": "number", "description": "Unix timestamp — only return results with created_at <= this (optional)."},
        },
    },
}

MEMORY_DELETE_SCHEMA = {
    "name": "vec_memory_delete",
    "description": "Delete one or more memory items by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of memory IDs to delete.",
            },
        },
        "required": ["memory_ids"],
    },
}

MEMORY_STATS_SCHEMA = {
    "name": "vec_memory_stats",
    "description": "Show memory store statistics: total items, sessions, storage size.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_cli.config import cfg_get, load_config
    try:
        config = load_config()
    except Exception:
        return {}
    # Try memory-lancedb first (this plugin's own key), fallback to lancedb-embed (backward compat)
    cfg = cfg_get(config, "plugins", "memory-lancedb", default={}) or {}
    if not cfg:
        cfg = cfg_get(config, "plugins", "lancedb-embed", default={}) or {}
    return cfg


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class LanceDBMemoryProvider(MemoryProvider):
    """Vector semantic memory via Ollama bge-m3:567m + LanceDB HNSW index."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._db = None        # lancedb.LanceDBConnection
        self._table = None     # lancedb.LanceTable
        self._session_id = ""
        self._embedding_lock = threading.Lock()
        # Cache: query -> (results, timestamp) for prefetch
        self._prefetch_cache: Dict[str, tuple] = {}
        self._prefetch_ttl = 30

    @property
    def name(self) -> str:
        return "memory-lancedb"

    def is_available(self) -> bool:
        base_url = self._config.get("base_url", "http://localhost:11434")
        model = self._config.get("embedding_model", "bge-m3:567m")
        try:
            _ollama_embed_single("health check", base_url, model)
            return True
        except Exception as e:
            logger.debug("LanceDB memory provider unavailable: %s", e)
            return False

    def get_config_schema(self):
        return [
            {"key": "base_url",         "description": "Ollama server URL",                          "default": "http://localhost:11434"},
            {"key": "embedding_model",   "description": "Embedding model available in Ollama",         "default": "bge-m3:567m"},
            {"key": "vector_dim",        "description": "Embedding vector dimension (bge-m3 = 1024)", "default": "1024"},
            {"key": "lance_dir",        "description": "LanceDB data directory",                     "default": "$HERMES_HOME/lance_memory"},
            {"key": "batch_size",       "description": "Max texts per embedding batch",              "default": "32"},
            {"key": "search_top_k",     "description": "Default top-k results per search",           "default": "5"},
            {"key": "min_content_len",  "description": "Skip content shorter than this (chars)",      "default": "50"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["memory-lancedb"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.warning("Failed to save lancedb-embed config: %s", e)

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        hermes_home = kwargs.get("hermes_home", str(get_hermes_home()))
        base_url     = self._config.get("base_url", "http://localhost:11434")
        model        = self._config.get("embedding_model", "bge-m3:567m")
        lance_dir    = self._config.get("lance_dir", f"{hermes_home}/lance_memory")

        # Expand $HERMES_HOME
        lance_dir = lance_dir.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        Path(lance_dir).parent.mkdir(parents=True, exist_ok=True)

        self._db     = _get_lance_db(lance_dir)
        self._table  = self._db.open_table("memories")
        self._session_id = session_id

        # Ensure FTS index exists for hybrid search
        # Note: LanceDB FTS doesn't support Chinese language natively.
        # Omit language param to use tantivy's default tokenizer, which
        # handles CJK characters as bigrams — effective for Chinese content.
        try:
            self._table.create_fts_index("content", replace=True)
            logger.info("LanceDB FTS index created/verified on 'content' field")
        except Exception as e:
            logger.warning("LanceDB FTS index creation failed (non-critical): %s", e)

        # Warm up Ollama
        try:
            _ollama_embed_single("warmup", base_url, model)
            logger.info("LanceDBMemoryProvider initialized — model=%s dir=%s", model, lance_dir)
        except Exception as e:
            logger.warning("LanceDBMemoryProvider warmup failed: %s", e)

    def system_prompt_block(self) -> str:
        if not self._table:
            return ""
        try:
            total = self._table.count_rows()
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Ollama Vector Memory\n"
                "Active. Empty vector store — use vec_memory_add to store facts, "
                "preferences, decisions, commands, or any content you want to recall later.\n"
                "Use vec_memory_search for semantic retrieval, vec_memory_list to view stored items."
            )
        return (
            f"# Ollama Vector Memory\n"
            f"Active. {total} items stored with bge-m3:567m embeddings (1024-dim) via LanceDB HNSW.\n"
            f"Use vec_memory_add to store, vec_memory_search to retrieve, "
            f"vec_memory_list to browse, vec_memory_stats for overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not self._table:
            return ""
        cache_key = query[:200]
        if cache_key in self._prefetch_cache:
            results, cached_at = self._prefetch_cache[cache_key]
            if datetime.now().timestamp() - cached_at < self._prefetch_ttl:
                return self._format_results(results[:3])
            del self._prefetch_cache[cache_key]
        try:
            results = self._do_search(query, top_k=3, session_id=session_id)
            self._prefetch_cache[cache_key] = (results, datetime.now().timestamp())
            return self._format_results(results)
        except Exception as e:
            logger.debug("prefetch search failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query or not self._table:
            return
        def _bg():
            try:
                results = self._do_search(query, top_k=3, session_id=session_id)
                self._prefetch_cache[query[:200]] = (results, datetime.now().timestamp())
            except Exception:
                pass
        t = threading.Thread(target=_bg, daemon=True, name="lancedb-prefetch")
        t.start()

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "",
                  timestamp: float = None) -> None:
        if not user_content.strip() and not assistant_content.strip():
            return
        if not self._table:
            return
        sid = session_id or self._session_id
        ts = timestamp if isinstance(timestamp, (int, float)) and timestamp > 0 \
            else datetime.now().timestamp()
        ts_iso = datetime.fromtimestamp(ts).isoformat()
        meta = json.dumps({
            "user_preview": user_content[:100],
            "asst_preview": assistant_content[:100],
            # message_timestamps[]: 统一格式，与 optimize_lance_memory.py 一致
            # 用于：created_at 取 [0]，结束时间取 [-1]，数组长度判断 Twig 跨了多少条消息
            "message_timestamps": [ts_iso],
        }, ensure_ascii=False)

        def _store():
            try:
                combined = f"[user]\n{user_content}\n[assistant]\n{assistant_content}"
                if len(combined) < int(self._config.get("min_content_len", 50)):
                    return
                with self._embedding_lock:
                    vec = _ollama_embed_single(combined, self.base_url, self.model)
                self._table.add([{
                    "id":         str(uuid.uuid4()),
                    "content":    combined,
                    "role":       "turn",
                    "session_id": sid,
                    "vector":     vec.tolist(),
                    "created_at": ts,
                    "metadata":   meta,
                }])
            except Exception as e:
                logger.debug("sync_turn store failed: %s", e)
        t = threading.Thread(target=_store, daemon=True, name="lancedb-sync")
        t.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._table or not messages:
            return
        pairs = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                content = msg["content"].strip()
                if len(content) < 10:
                    continue
                assistant_content = ""
                assistant_ts = None
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    assistant_content = messages[i + 1].get("content", "") or ""
                    assistant_ts = messages[i + 1].get("timestamp")
                user_ts = msg.get("timestamp")
                pairs.append({
                    "user_content": content,
                    "asst_content": assistant_content,
                    "user_ts": user_ts,
                    "asst_ts": assistant_ts,
                })
        if not pairs:
            return

        def _batch_store():
            try:
                min_len = int(self._config.get("min_content_len", 50))
                now_ts = datetime.now().timestamp()
                rows = []
                for pair in pairs:
                    u = pair["user_content"]
                    a = pair["asst_content"]
                    u_ts = pair["user_ts"]
                    a_ts = pair["asst_ts"]
                    combined = f"[user]\n{u}\n[assistant]\n{a}"

                    # Build message_timestamps[]: 统一格式，与 optimize_lance_memory.py 一致
                    ts_list = []
                    if u_ts:
                        try:
                            ts_list.append(datetime.fromtimestamp(float(u_ts)).isoformat())
                        except Exception:
                            pass
                    if a_ts:
                        try:
                            ts_list.append(datetime.fromtimestamp(float(a_ts)).isoformat())
                        except Exception:
                            pass

                    # Build metadata: 统一 message_timestamps[] 格式
                    meta = {
                        "user_preview": u[:100],
                        "asst_preview": a[:100] if a else "",
                        "message_timestamps": ts_list,
                    }

                    # Use the earlier timestamp (user message time) as row created_at
                    row_ts = u_ts if isinstance(u_ts, (int, float)) and u_ts > 0 else now_ts

                    if len(combined) < min_len:
                        continue

                    try:
                        vec = _ollama_embed_single(combined, self.base_url, self.model)
                    except Exception:
                        continue

                    rows.append({
                        "id":         str(uuid.uuid4()),
                        "content":    combined,
                        "role":       "session_end",
                        "session_id": self._session_id,
                        "vector":     vec.tolist(),
                        "created_at": row_ts,
                        "metadata":   json.dumps(meta, ensure_ascii=False),
                    })
                if rows:
                    self._table.add(rows)
                    logger.info("LanceDB: stored %d session-end memories", len(rows))
            except Exception as e:
                logger.warning("session_end batch store failed: %s", e)

        t = threading.Thread(target=_batch_store, daemon=True, name="lancedb-session-end")
        t.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            MEMORY_STORE_SCHEMA,
            MEMORY_SEARCH_SCHEMA,
            MEMORY_LIST_SCHEMA,
            MEMORY_DELETE_SCHEMA,
            MEMORY_STATS_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "vec_memory_add":
            return self._tool_add(args)
        elif tool_name == "vec_memory_search":
            return self._tool_search(args)
        elif tool_name == "vec_memory_list":
            return self._tool_list(args)
        elif tool_name == "vec_memory_delete":
            return self._tool_delete(args)
        elif tool_name == "vec_memory_stats":
            return self._tool_stats(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        self._db = None
        self._table = None

    # ---------------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._config.get("base_url", "http://localhost:11434")

    @property
    def model(self) -> str:
        return self._config.get("embedding_model", "bge-m3:567m")

    @property
    def batch_size(self) -> int:
        return int(self._config.get("batch_size", 32))

    @property
    def search_top_k(self) -> int:
        return int(self._config.get("search_top_k", 5))

    def _do_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        session_id: str = "",
        after_timestamp: float = None,
        before_timestamp: float = None,
    ) -> List[Dict[str, Any]]:
        """Embed query and perform HNSW ANN search via LanceDB.

        Args:
            after_timestamp: Unix timestamp — only return results with created_at >= this.
            before_timestamp: Unix timestamp — only return results with created_at <= this.
        """
        with self._embedding_lock:
            query_vec = _ollama_embed_single(query, self.base_url, self.model)

        q = query_vec.tolist()

        # Build where clause
        conditions = []
        if session_id:
            conditions.append(f"session_id = '{session_id}'")
        if after_timestamp is not None and after_timestamp > 0:
            conditions.append(f"created_at >= {after_timestamp}")
        if before_timestamp is not None and before_timestamp > 0:
            conditions.append(f"created_at <= {before_timestamp}")

        where_clause = " AND ".join(conditions) if conditions else None

        if where_clause:
            search = self._table.search(q).where(where_clause).limit(top_k * 2)
        else:
            search = self._table.search(q).limit(top_k * 2)

        # Convert _distance (LanceDB) to cosine similarity score
        # LanceDB uses L2 distance by default; convert to similarity
        results = search.to_list()
        scored = []
        for r in results:
            dist = r.get("_distance", 0.0)
            # Approximate cosine similarity from L2 distance
            # For normalized vectors: sim ≈ 1 - dist / sqrt(2) when dist is L2
            # Better: use cosine similarity directly
            # Since we use /api/embed (not normalized), compute proper cosine
            vec = np.array(r["vector"], dtype=np.float32)
            sim = float(np.dot(vec / (np.linalg.norm(vec) + 1e-9), query_vec / (np.linalg.norm(query_vec) + 1e-9)))
            if sim >= min_score:
                # Attach metadata so callers can see message_timestamps, topics, etc.
                meta = {}
                if r.get("metadata"):
                    try:
                        meta = json.loads(r["metadata"])
                    except Exception:
                        pass
                scored.append({
                    "id":         r["id"],
                    "content":    r["content"],
                    "role":       r["role"],
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "score":      round(sim, 4),
                    "metadata":   meta,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _do_fts_search(
        self,
        query: str,
        top_k: int = 5,
        session_id: str = "",
        after_timestamp: float = None,
        before_timestamp: float = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search via LanceDB FTS index."""
        from lancedb.query import MatchQuery

        # Build where clause (same as _do_search)
        conditions = []
        if session_id:
            conditions.append(f"session_id = '{session_id}'")
        if after_timestamp is not None and after_timestamp > 0:
            conditions.append(f"created_at >= {after_timestamp}")
        if before_timestamp is not None and before_timestamp > 0:
            conditions.append(f"created_at <= {before_timestamp}")
        where_clause = " AND ".join(conditions) if conditions else None

        fts_query = MatchQuery(query=query, column="content")

        if where_clause:
            results = self._table.search(fts_query).where(where_clause).limit(top_k * 2).to_list()
        else:
            results = self._table.search(fts_query).limit(top_k * 2).to_list()

        scored = []
        for r in results:
            # FTS relevance score — LanceDB returns "_score" (BM25-like, higher=better)
            score = r.get("_score", r.get("_relevance_score", r.get("_distance", 0.0)))
            if isinstance(score, (int, float)) and score < 0:
                score = max(0, 1.0 + score)  # normalize negative BM25 scores
            meta = {}
            if r.get("metadata"):
                try:
                    meta = json.loads(r["metadata"])
                except Exception:
                    pass
            scored.append({
                "id": r["id"],
                "content": r["content"],
                "role": r["role"],
                "session_id": r["session_id"],
                "created_at": r["created_at"],
                "score": round(float(score), 4),
                "metadata": meta,
                "_source": "fts",
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _do_hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        session_id: str = "",
        after_timestamp: float = None,
        before_timestamp: float = None,
        vector_weight: float = 0.7,
        fts_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector + FTS via Reciprocal Rank Fusion."""

        # Run both searches
        vector_results = self._do_search(query, top_k=top_k * 2, min_score=0.0,
                                          session_id=session_id,
                                          after_timestamp=after_timestamp,
                                          before_timestamp=before_timestamp)
        fts_results = self._do_fts_search(query, top_k=top_k * 2,
                                           session_id=session_id,
                                           after_timestamp=after_timestamp,
                                           before_timestamp=before_timestamp)

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        scores = {}  # id -> accumulated score
        items = {}   # id -> full item dict

        for rank, r in enumerate(vector_results):
            item_id = r["id"]
            scores[item_id] = scores.get(item_id, 0) + vector_weight / (k + rank + 1)
            items[item_id] = r
            items[item_id]["_source"] = "hybrid"

        for rank, r in enumerate(fts_results):
            item_id = r["id"]
            scores[item_id] = scores.get(item_id, 0) + fts_weight / (k + rank + 1)
            if item_id not in items:
                items[item_id] = r
            items[item_id]["_source"] = "hybrid"

        # Sort by RRF score, apply min_score threshold
        merged = []
        for item_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            item = items[item_id].copy()
            item["score"] = round(rrf_score, 4)
            if item["score"] >= min_score:
                merged.append(item)
            if len(merged) >= top_k:
                break

        return merged

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = []
        for r in results:
            score = r.get("score", 0)
            content = r.get("content", "")[:300]
            lines.append(f"- [{score:.3f}] {content}")
        return "## Ollama Vector Memory\n" + "\n".join(lines)

    # ---------------------------------------------------------------------------
    # Tool implementations
    # ---------------------------------------------------------------------------

    def _tool_add(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        min_len = int(self._config.get("min_content_len", 50))
        if len(content) < min_len:
            return json.dumps({
                "status": "skipped",
                "reason": f"content too short ({len(content)} < {min_len} chars)"
            })

        role     = args.get("role", "")
        sid      = args.get("session_id", "") or self._session_id
        metadata = args.get("metadata", "{}")

        try:
            with self._embedding_lock:
                vec = _ollama_embed_single(content, self.base_url, self.model)
            mem_id = str(uuid.uuid4())
            self._table.add([{
                "id":         mem_id,
                "content":    content,
                "role":       role or "turn",
                "session_id": sid,
                "vector":     vec.tolist(),
                "created_at": datetime.now().timestamp(),
                "metadata":   metadata,
            }])
            return json.dumps({"status": "added", "id": mem_id, "dimension": VECTOR_DIM})
        except Exception as e:
            return tool_error(f"Failed to add memory: {e}")

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        mode = args.get("mode", "hybrid")
        if mode not in ("hybrid", "vector", "keyword"):
            mode = "hybrid"

        top_k     = min(int(args.get("top_k", self.search_top_k)), 20)
        min_score = float(args.get("min_score", 0.0))
        session_id = args.get("session_id", "")

        # Time range filters (Unix timestamps)
        after_ts = None
        before_ts = None
        if args.get("after_timestamp"):
            try:
                after_ts = float(args["after_timestamp"])
            except (ValueError, TypeError):
                pass
        if args.get("before_timestamp"):
            try:
                before_ts = float(args["before_timestamp"])
            except (ValueError, TypeError):
                pass

        try:
            if mode == "keyword":
                results = self._do_fts_search(
                    query, top_k=top_k,
                    session_id=session_id,
                    after_timestamp=after_ts, before_timestamp=before_ts,
                )
            elif mode == "vector":
                results = self._do_search(
                    query, top_k=top_k, min_score=min_score,
                    session_id=session_id,
                    after_timestamp=after_ts, before_timestamp=before_ts,
                )
            else:  # hybrid (default)
                try:
                    results = self._do_hybrid_search(
                        query, top_k=top_k, min_score=min_score,
                        session_id=session_id,
                        after_timestamp=after_ts, before_timestamp=before_ts,
                    )
                except Exception as hybrid_err:
                    logger.warning("Hybrid search failed, falling back to vector: %s", hybrid_err)
                    results = self._do_search(
                        query, top_k=top_k, min_score=min_score,
                        session_id=session_id,
                        after_timestamp=after_ts, before_timestamp=before_ts,
                    )
                    mode = "vector (fallback)"

            if not results:
                return json.dumps({
                    "query": query, "results": [], "count": 0,
                    "message": "No matching memories found."
                })

            formatted = []
            for r in results:
                dt = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
                formatted.append({
                    "id":         r["id"],
                    "content":    r["content"],
                    "role":       r["role"],
                    "session_id": r["session_id"],
                    "timestamp":  dt,
                    "score":      r["score"],
                    "metadata":   r.get("metadata") or "{}",
                })

            return json.dumps({
                "query":   query,
                "results": formatted,
                "count":   len(formatted),
                "message": f"Found {len(formatted)} matching memories (mode={mode}).",
                "search_mode": mode,
            })
        except Exception as e:
            return tool_error(f"Search failed: {e}")

    def _tool_list(self, args: dict) -> str:
        limit      = min(int(args.get("limit", 20)), 100)
        session_id = args.get("session_id", "")

        # Time range filters (Unix timestamps)
        after_ts = None
        before_ts = None
        if args.get("after_timestamp"):
            try:
                after_ts = float(args["after_timestamp"])
            except (ValueError, TypeError):
                pass
        if args.get("before_timestamp"):
            try:
                before_ts = float(args["before_timestamp"])
            except (ValueError, TypeError):
                pass

        try:
            # Build where clause for LanceDB
            conditions = []
            if session_id:
                conditions.append(f"session_id = '{session_id}'")
            if after_ts is not None and after_ts > 0:
                conditions.append(f"created_at >= {after_ts}")
            if before_ts is not None and before_ts > 0:
                conditions.append(f"created_at <= {before_ts}")

            where_clause = " AND ".join(conditions) if conditions else None

            if where_clause:
                all_rows = self._table.search([0.0] * VECTOR_DIM).where(where_clause).limit(limit).to_list()
                rows = all_rows
            else:
                rows = self._table.search([0.0] * VECTOR_DIM).limit(limit).to_list()

            results = []
            for r in rows:
                dt = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
                results.append({
                    "id":         r["id"],
                    "content":    r["content"][:200] + ("..." if len(r["content"]) > 200 else ""),
                    "role":       r["role"],
                    "session_id": r["session_id"],
                    "timestamp":  dt,
                    "metadata":   r.get("metadata") or "{}",
                })

            return json.dumps({"results": results, "count": len(results)})
        except Exception as e:
            return tool_error(f"List failed: {e}")

    def _tool_delete(self, args: dict) -> str:
        memory_ids = args.get("memory_ids", [])
        if not memory_ids:
            return tool_error("memory_ids is required")

        try:
            placeholders = " OR ".join([f"id = '{mid}'" for mid in memory_ids])
            # LanceDB delete — DeleteResult is not JSON serializable, extract field
            result = self._table.delete(f"{placeholders}")
            deleted_count = getattr(result, 'num_deleted_rows', len(memory_ids))
            return json.dumps({"deleted": deleted_count, "requested": len(memory_ids)})
        except Exception as e:
            return tool_error(f"Delete failed: {e}")

    def _tool_stats(self, args: dict) -> str:
        try:
            total   = self._table.count_rows()
            # Distinct sessions by scanning (LanceDB doesn't have a native sessions table)
            all_rows = self._table.search([0.0] * VECTOR_DIM).limit(10000).to_list()
            sessions = len({r["session_id"] for r in all_rows if r["session_id"]})

            lance_dir = self._config.get("lance_dir", "")
            if lance_dir:
                lance_dir = lance_dir.replace("$HERMES_HOME", str(Path.home())).replace("${HERMES_HOME}", str(Path.home()))
                try:
                    size = sum(f.stat().st_size for f in Path(lance_dir).rglob("*") if f.is_file())
                    size_str = f"{size / 1024:.1f} KB"
                except Exception:
                    size_str = "unknown"
            else:
                size_str = "unknown"

            return json.dumps({
                "total_memories":     total,
                "total_sessions":     sessions,
                "embedding_model":     self.model,
                "vector_dimension":   VECTOR_DIM,
                "db_size":            size_str,
                "index":              "HNSW (IVF_HNSW_SQ) + FTS",
                "search_modes":        ["hybrid", "vector", "keyword"],
                "fts_index":           True,
            })
        except Exception as e:
            return tool_error(f"Stats failed: {e}")


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the LanceDB embed memory provider."""
    config   = _load_plugin_config()
    provider = LanceDBMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
