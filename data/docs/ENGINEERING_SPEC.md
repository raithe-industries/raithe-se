# raithe-se Engineering Specification

© RAiTHE INDUSTRIES INCORPORATED 2026  
Status: Confirmed. 
Date: 25 Apr 2026

## 1. Purpose

`raithe-se` is a Rust search engine for modern workstation-class PCs. It must combine a reliable search core with adaptive CUDA acceleration, LLM-assisted instant answers, semantic retrieval, and learned ranking.

The engine should scale to the host it boots on. The practical development baseline is not a tiny machine: RAiTHE development/workstation PCs are expected to have the following minimums:

- 32 GB+ RAM;
- NVIDIA CUDA GPU with 8 GB+ VRAM;
- high-clock modern CPU, approximately 4.0 GHz+ boost class;
- fast SSD or NVMe storage.

The system should still keep a debuggable Phase 1 core that can run without neural startup, but the product direction is CUDA-forward and GPU-aware.

## 2. Engineering Priorities

Priority order:

1. Correct crawl → parse → index → commit → search behaviour.
2. Durable URL identity and restart-safe recrawl.
3. Observability and reproducible local tests.
4. CUDA-aware resource planning.
5. LLM-assisted instant answers.
6. Semantic retrieval and embeddings.
7. Neural reranking and learned ranking.
8. Query rewriting and deeper query understanding.
9. Freshness, link analysis, and production hardening.

Instant answers are explicitly higher-value than query rewriting. Query rewriting is useful, but direct answers create more product value.

## 3. Runtime Phases

### 3.1 Phase 1: Reliable Search Core

Required properties:

- deterministic crawl/index/search pipeline;
- no neural model required to debug indexing;
- no CUDA/model preflight required to prove BM25 search;
- batch and periodic index commits;
- `/debug/stats`;
- HTML and JSON search results.

Data flow:

```text
seeds
  → frontier
  → robots check
  → HTTP fetch
  → HTML parse
  → document ID assignment
  → Tantivy add/upsert
  → commit
  → search
```

### 3.2 Phase 2: Durable URL Identity

Required:

- durable URL registry;
- stable URL → document ID mapping;
- restart-safe recrawl;
- index upsert/delete-by-URL or delete-by-doc-id;
- no duplicate results for the same canonical URL;
- end-to-end local crawl/search/restart test.

### 3.3 Phase 3: CUDA Neural Assistance

Required:

- host resource detection;
- CUDA provider detection;
- model availability checks;
- GPU memory budgeting;
- batching;
- fallback policy;
- neural subsystem status in debug output.

Initial CUDA neural features:

1. LLM-assisted instant answers.
2. Embedding generation.
3. Cross-encoder reranking.
4. Optional query rewriting.

### 3.4 Phase 4: Instant Answer Engine

Instant answers become a central feature.

Answer sources:

- deterministic computation;
- unit/date/time rules;
- index-grounded snippets;
- document summaries;
- LLM synthesis;
- future structured data.

Target flow:

```text
query
  → classify answerability
  → deterministic instant-answer checks
  → retrieve grounding documents
  → optional LLM synthesis
  → instant answer + citations/supporting results
```

LLM answers must be grounded when they present factual claims. If no reliable grounding exists, return search results and avoid pretending to know.

### 3.5 Phase 5: Semantic Retrieval and Ranking

Adds:

- embeddings;
- ANN index;
- BM25 + vector fusion;
- GBDT/LambdaMART features;
- PageRank/link features;
- freshness features;
- neural rerank top-k;
- final score sort.

### 3.6 Phase 6: Query Understanding

Adds:

- spell correction;
- synonym expansion;
- intent classification;
- query rewriting;
- conversational/session-aware refinements.

LLM rewriting must never be a hard dependency for normal search. If rewriting fails, use the original query.

## 4. Hardware and Resource Model

### 4.1 Practical Baseline

The practical baseline for RAiTHE machines:

| Component | Baseline |
|---|---|
| RAM | 32 GB+ |
| VRAM | 8 GB+ NVIDIA CUDA GPU |
| CPU | modern high-clock 4.0 GHz+ class |
| Storage | fast SSD/NVMe |
| OS | Linux-first |

This baseline allows `raithe-se` to assume CUDA is important and worth optimising for, while still keeping the core pipeline separable for debugging.

### 4.2 Host Resource Detection

At startup, detect:

- CPU core count;
- CPU model/frequency, where available;
- RAM total and available;
- SSD/index path free space;
- NVIDIA GPU presence;
- GPU count;
- per-GPU VRAM;
- CUDA driver/runtime availability;
- ONNX Runtime provider availability;
- model paths and model sizes.

### 4.3 CPU/GPU Division of Labour

CPU owns:

- crawler scheduling;
- DNS and HTTP orchestration;
- robots.txt;
- HTML parsing;
- URL normalisation;
- URL registry;
- crawl state;
- Tantivy indexing;
- BM25 search;
- serving/API/UI;
- metrics and logging.

GPU/CUDA owns:

- embedding inference;
- cross-encoder reranking;
- LLM-assisted instant answers;
- optional query rewriting;
- summary generation;
- future neural scoring models;
- vector-heavy batch workloads.

CPU and GPU must be coordinated through bounded queues and batchers. The crawler must not flood neural inference. Neural inference must not starve indexing. Indexing must not block serving.

### 4.4 CUDA Policy

CUDA is a first-class acceleration target.

Rules:

1. Prefer CUDA for neural workloads when available.
2. Probe CUDA with a real small inference, not only device detection.
3. Track VRAM budgets explicitly.
4. Batch embeddings/reranking to keep GPU utilisation high.
5. Fall back to CPU only if configured or if the feature is optional.
6. Never let optional neural failure corrupt or crash the core index.
7. Expose selected provider, model, batch size, and fallback status in `/debug/stats`.

### 4.5 Model Selection

Model choice should be based on VRAM.

Example tiers:

| VRAM | Behaviour |
|---|---|
| 8 GB | compact generator, moderate embedder/reranker batches |
| 12 GB | larger batches, stronger reranker |
| 16 GB | stronger generator/reranker path |
| 24 GB+ | larger generator and heavier semantic workloads |
| multi-GPU | assign workloads by queue pressure and free VRAM |

Model selection must not be hardcoded to one GPU model.

## 5. Instant Answers

### 5.1 Purpose

Instant answers are a major product feature. They should answer directly when possible instead of merely returning links.

### 5.2 Answer Types

Supported/future categories to implement:

- arithmetic;
- unit conversions;
- dates and durations;
- time zones;
- definitions from indexed documents;
- summaries of indexed documents;
- factual answers grounded in retrieved pages;
- entity cards;
- small code/config explanations;
- local/private corpus answers.

### 5.3 LLM Role

The LLM should assist users with instant answers before offering any query rewriting.

High-value LLM tasks:

- synthesise a concise answer from top retrieved documents;
- explain a technical answer with context;
- summarise multiple indexed pages;
- extract answer candidates from snippets;
- generate a direct answer with supporting sources;
- say when the corpus does not contain enough evidence.

Lower-priority LLM tasks:

- rewrite query;
- expand synonyms;
- infer user intent.

### 5.4 Grounding Rules

LLM-generated instant answers must be grounded where possible.

Rules:

- retrieve before generating;
- pass snippets/sources into the answer prompt;
- include source URLs or document IDs in structured response;
- distinguish “answer found in corpus” from “general model answer”;
- fall back to search results if confidence is low.

## 6. Durable URL Identity

A search engine must not create a new document every time it sees the same URL.

### 6.1 Required Registry

SQLite-backed registry example:

```sql
CREATE TABLE url_registry (
    url TEXT PRIMARY KEY,
    doc_id INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_http_status INTEGER,
    last_error TEXT,
    first_seen_unix_ms INTEGER NOT NULL,
    updated_unix_ms INTEGER NOT NULL
);

CREATE INDEX url_registry_status_idx ON url_registry(status);
CREATE INDEX url_registry_doc_id_idx ON url_registry(doc_id);
```

Statuses example:

```text
queued
fetching
fetched
parsed
indexed
failed_retryable
failed_permanent
robots_blocked
```

### 6.2 Required Operation

```rust
lookup_or_allocate(url) -> DocumentId
```

Rules:

- existing URL returns existing ID;
- new URL receives next monotonic ID;
- allocation is transactional;
- process restart does not reset IDs;
- no URL receives multiple IDs;
- no two URLs share one ID.

## 7. Indexing and Upsert

### 7.1 Required Upsert

Production indexing should use:

```rust
pub fn upsert(&self, doc: &ParsedDocument) -> Result<()>
```

Behaviour:

1. Delete existing document by `doc_id` or exact URL.
2. Add replacement document.
3. Increment pending count.
4. Commit by configured batch/interval policy.

### 7.2 Schema Versioning

Rules:

- new empty index directory may receive current `VERSION`;
- mismatched `VERSION` must be rejected;
- non-empty index directory with missing `VERSION` must be rejected;
- migration must be explicit;
- never silently enable an old index.

## 8. Crawler

### 8.1 Required Behaviour

- respect robots.txt;
- per-host rate limiting;
- bounded fetch concurrency;
- URL canonicalisation;
- duplicate queue suppression;
- retry state;
- debug visibility;
- no permanent death on temporary empty frontier.

### 8.2 Retry Policy

Transient fetch failures must retry.

Minimum:

- max attempts;
- requeue retryable failures;
- mark permanent after budget exhaustion;
- record last error;
- expose counts.

Better:

- exponential backoff;
- jitter;
- durable retry queue;
- host-aware retry scheduling.

## 9. Query and Serving

### 9.1 Query Processor

Phase 1 query processing is pass-through.

Advanced query features:

- tokenisation;
- spelling;
- synonyms;
- intent;
- rewrite.

Failures must fall back to original query.

### 9.2 Search Response

JSON response example:

```json
{
  "query": "rust",
  "results": [
    {
      "title": "Example",
      "url": "https://example.com/",
      "snippet": "Example snippet...",
      "score": 1.0
    }
  ],
  "instant": null,
  "total_results": 1
}
```

Future instant answer response should support evidence, example:

```json
{
  "kind": "llm_grounded",
  "display": "Concise answer...",
  "confidence": 0.82,
  "sources": [
    {
      "url": "https://example.com/a",
      "title": "Example",
      "doc_id": 42
    }
  ]
}
```

## 10. Ranking

### 10.1 Phase 1

BM25 score is authoritative.

### 10.2 GBDT

GBDT must be pass-through if no trained model exists.

### 10.3 Neural Reranking

Neural reranking is optional but CUDA-preferred.

Rules:

- only top-k;
- use useful snippets/body;
- batch candidates;
- record latency;
- final sort after score assignment;
- fail soft unless strict mode is enabled.

## 11. Observability

Required endpoints:

| Endpoint | Purpose |
|---|---|
| `/health` | basic process health |
| `/metrics` | Prometheus metrics |
| `/debug/stats` | pipeline/resource counters |
| `/search` | search API/UI |

Recommended `/debug/stats` fields:

- fetched;
- parsed;
- indexed pending;
- committed;
- frontier queue;
- URL registry status counts;
- retry queue length;
- last errors;
- selected neural provider;
- CUDA available;
- GPU name and VRAM;
- active model names;
- neural queue depths;
- average inference latency.

## 12. Testing

### 12.1 Critical Integration Test Example

```text
1. Start local HTTP server with:
   /      links to /a and /b
   /a     contains unique token alpha_raithe_phase1
   /b     contains unique token beta_raithe_phase1
2. Start raithe-se with seed /
3. Wait until committed >= 3
4. Query alpha_raithe_phase1
5. Assert /a is returned
6. Restart engine using same data dir
7. Query alpha_raithe_phase1 again
8. Assert /a is returned
9. Re-crawl root
10. Assert no duplicate /a result
```

### 12.2 CUDA/Neural Tests

When neural mode is enabled:

- provider detection test;
- small embedding inference test;
- rerank test with known candidates;
- instant-answer grounding test;
- CPU fallback test, if allowed;
- model missing/error reporting test.

## 13. Launcher

`raithe.sh` is the operational entrypoint.

Rules:

- support Phase 1 launch without model install;
- support explicit `--with-neural`;
- perform CUDA/ORT/model checks only when neural mode is requested or configured;
- detect host resources;
- print selected runtime plan;
- fail clearly when required neural resources are missing;
- never surprise-install massive dependencies without explicit neural path.

## 14. Security

Required:

- respect robots.txt;
- per-host rate limiting;
- request timeout;
- response body size limit;
- no path traversal in storage/backup;
- HTML escaping;
- CSP and security headers;
- no secrets in logs;
- no raw panic/unwrap in production path;
- clear distinction between grounded and ungrounded LLM answers.

## 15. Roadmap

### Milestone A: Reliable Phase 1

- durable URL registry;
- index upsert;
- retry requeue/backoff;
- schema version hardening;
- local integration test.

### Milestone B: Workstation Runtime Planner

- host resource detector;
- CUDA probe;
- runtime plan struct;
- batch size planner;
- VRAM budget planner;
- debug exposure.

### Milestone C: Instant Answers

- deterministic answers;
- index-grounded answers;
- LLM synthesis;
- source/evidence response model;
- confidence/fallback behaviour.

### Milestone D: Semantic Retrieval

- CUDA embeddings;
- ANN index;
- BM25/vector fusion;
- semantic debug metrics.

### Milestone E: Learned Ranking

- PageRank/link features;
- GBDT training data;
- neural rerank;
- click/session signals.

### Milestone F: Production Hardening

- CI gates;
- benchmarks;
- crash recovery;
- backups;
- index compaction policy;
- telemetry dashboards.

## 16. Definition of Done for Core Engine

The core engine is real when:

- it crawls a local multi-page site from one seed;
- it commits documents;
- it returns expected search results;
- it survives restart;
- it does not duplicate canonical URLs;
- it exposes health/debug stats;
- it can run the core path without neural startup;
- it can use CUDA for neural features when enabled.

## 17. Definition of Done for CUDA Neural Layer

The CUDA neural layer is real and working properly when:

- host resources are detected;
- CUDA provider is selected and verified with inference;
- embeddings run in batches;
- reranking runs over top-k;
- instant answers can use LLM synthesis;
- failures are visible and non-corrupting;
- CPU/GPU queues are bounded;
- VRAM use is planned, not accidental.

## 18. Amendment Log

| Version | Date | Summary |
|---|---|---|
| 2.1 draft | 25 Apr 2026 | Reframed hardware policy around CUDA workstation baseline and elevated LLM-assisted instant answers above query rewriting. |
| 2.1 final | 25 Apr 2026 | RRSP sign-off accordingly.                                                                                                 |

