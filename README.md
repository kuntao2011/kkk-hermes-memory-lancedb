# Hermes Memory LanceDB Plugin

> ⚠️ **已归档 / ARCHIVED** — 此插件已停止维护，继任者为 [kkk-hermes-memory-zvec](https://github.com/kuntao2011/kkk-hermes-memory-zvec)（zvec 后端）。

独立安装的 Hermes memory provider 插件，提供 **混合检索**（HNSW 向量 + FTS 全文）能力。

## 特性

- **Hybrid Search** — 向量搜索 + FTS 全文搜索，RRF (Reciprocal Rank Fusion) 合并
- **三种搜索模式** — `hybrid`（默认）、`vector`（纯语义）、`keyword`（纯关键词）
- **Metadata 丰富** — message_timestamps[]、user/asst preview 等结构化元数据
- **时间范围过滤** — after_timestamp / before_timestamp 精确筛选
- **FTS 自动索引** — LanceDB tantivy tokenizer，支持 CJK bigram 分词
- **完全本地** — Ollama bge-m3:567m 嵌入，无需外部 API

## 安装

```bash
hermes plugins install kuntao2011/kkk-hermes-memory-lancedb
```

## 配置

在 profile 的 `config.yaml` 中：

```yaml
memory:
  provider: memory-lancedb

plugins:
  memory-lancedb:
    base_url: http://localhost:11434
    embedding_model: bge-m3:567m
    lance_dir: $HERMES_HOME/lance_memory
```

## 与内置 lancedb-embed 的区别

此插件是 `lancedb-embed` 的增强版，包含以下自定义修改：

1. **Hybrid Search** — FTS + HNSW 向量通过 RRF 融合
2. **FTS 索引自动创建** — 在 `initialize()` 时幂等创建
3. **Metadata 返回** — 搜索/列表结果包含 metadata 字段
4. **时间戳链路修复** — sync_turn/on_session_end 的 metadata 包含 message_timestamps[]
5. **FTS _score 字段修复** — 正确处理 LanceDB 0.30.2 的 BM25 分数字段名
6. **中文 FTS 兼容** — 去除 language="English"，使用 tantivy 默认 tokenizer

## 升级安全

此插件安装在 `~/.hermes/plugins/memory-lancedb/`（用户插件目录），
**不会被** `hermes update` 覆盖（内置 `plugins/memory/lancedb-embed/` 是 bundled 目录，与此插件隔离）。

## 相关技能

- `optimize-lance-memory` — 迁移后的优化维护
- `lancedb-memory-migration` — 从 FTS5 迁移到 LanceDB
