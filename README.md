# © RAiTHE INDUSTRIES INCORPORATED 2026

```
                         ██████╗  █████╗ ██╗████████╗██╗  ██╗███████╗
                         ██╔══██╗██╔══██╗   ╚══██╔══╝██║  ██║██╔════╝
                         ██████╔╝███████║██║   ██║   ███████║█████╗
                         ██╔══██╗██╔══██║██║   ██║   ██╔══██║██╔══╝
                         ██║  ██║██║  ██║██║   ██║   ██║  ██║███████╗
                         ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
                                         SEARCH ENGINE
```

# raithe-se

**A world-class search engine built entirely in Rust.**

Single-node architecture. Hybrid neural ranking. LLM-assisted query
understanding. Structured instant answers. Built from the ground up per the
raithe-se Engineering Specification v1.7.

**Spec:** `raithe-se Engineering Specification v1.7 — 19 Apr 2026`  
**Rust:** `rustc 1.95.0`  
**Status:** `cargo build --release` confirmed. 147 unit tests passing.  
**Hardware:** GTX 1080 · CUDA 12.2 · Ubuntu 24.04 · i7-6700K · 32 GB

---

## Architecture

18-crate Cargo workspace. All crates `raithe-<name>`. All shared dependency
versions pinned in `[workspace.dependencies]`.

| Crate        | Responsibility                                                   |
|--------------|------------------------------------------------------------------|
| `common`     | Shared primitive types — no logic, no I/O, no async             |
| `config`     | Load, validate, and hot-reload `engine.toml`                     |
| `metrics`    | Prometheus registry + tracing init — all handles registered here |
| `storage`    | Crawl log, doc store, mmap, backups                              |
| `scraper`    | Single-URL HTTP fetch → `FetchResult`                            |
| `crawler`    | URL frontier, politeness scheduling, robots.txt, dispatch        |
| `parser`     | Raw HTML bytes → `ParsedDocument`                                |
| `indexer`    | Tantivy inverted index — schema, tokeniser, ingestion, search    |
| `neural`     | ONNX Runtime inference — hardware auto-EP, 3 model handles       |
| `semantic`   | HNSW ANN index for dense embedding retrieval (from scratch)      |
| `linkgraph`  | CSR link graph + iterative PageRank                              |
| `ranker`     | Three-phase ranking pipeline (BM25F → GBDT → cross-encoder)     |
| `query`      | Tokenise, spell-correct, intent classify, synonym expand, rewrite|
| `instant`    | Instant answers — arithmetic, unit conversions, time zones       |
| `freshness`  | Incremental re-crawl, stale detection, tombstones                |
| `session`    | In-memory session context + reformulation detection              |
| `serving`    | Axum HTTP server + search UI                                     |
| `app`        | Single-binary entry point — CLI, banner, init sequence           |

---

## Ranking Pipeline

Every query passes all three phases in order. Phase 3 is never skipped.

| Phase | Mechanism                                    | Implementation          |
|-------|----------------------------------------------|-------------------------|
| 1     | BM25F field-weighted scoring                 | Tantivy                 |
| 2     | LambdaMART re-rank                           | `gbdt` crate, 300 trees |
| 3     | Neural cross-encoder re-rank (top 32)        | BGE-reranker-large ONNX |

`raithe_rank_phase3_calls_total` must be > 0 after any search. A value of zero
is a CRITICAL defect.

---

## Neural Models

`raithe.sh` is the sole operational entrypoint. It installs and validates all
three models, then launches the binary.

**Do not use `install_models.sh` — it is superseded and removed.**

| Directory    | Model / HF ID                    | Task                                   |
|--------------|----------------------------------|----------------------------------------|
| `embedder/`  | `BAAI/bge-large-en-v1.5`         | feature-extraction → `embed()`         |
| `reranker/`  | `BAAI/bge-reranker-large`        | text-classification → `rerank()`       |
| `generator/` | `Qwen/Qwen2.5-7B-Instruct`       | text-generation → `generate()`         |

Both `model.onnx` and `tokenizer.json` must be present in each subdirectory.
A missing file is a startup `Error::ModelNotFound`. There is no silent fallback.

Generator export uses `ORTModelForCausalLM` Python API (not `optimum-cli`) to
bypass the 2 GiB protobuf limit on 7B+ models.

**Hardware auto-detection order:** CUDA → DirectML → CoreML → CPU  
Generator is hardware-adaptive: 14B model selected when RAM ≥ 64 GB and VRAM ≥ 16 GB.

**Python export stack (model install only — not runtime):**
`optimum==1.23.3` · `transformers==4.46.3` · `torch==2.5.1+cpu` · `onnxscript` · `onnx` · `onnxruntime`

**ORT shared library:** `libonnxruntime.so.1.20.1` — downloaded to
`~/.cache/ort.pyke.io/` by `cargo build --release` via the `ort` crate
(`load-dynamic` feature). `~/.local/bin` must be on `PATH` on Ubuntu.

---

## Workspace Layout

```
raithe-se/
├── Cargo.toml              # workspace root — all dep versions pinned here
├── Cargo.lock
├── raithe.sh               # SOLE operational entrypoint: install + launch
├── data/
│   ├── config/
│   │   └── engine.toml     # default runtime configuration
│   └── models/             # NOT committed — populated by raithe.sh
│       ├── embedder/
│       │   ├── model.onnx
│       │   └── tokenizer.json
│       ├── reranker/
│       │   ├── model.onnx
│       │   └── tokenizer.json
│       └── generator/
│           ├── model.onnx
│           └── tokenizer.json
└── crates/
    ├── common/   ├── config/    ├── metrics/   ├── storage/
    ├── scraper/  ├── crawler/   ├── parser/    ├── indexer/
    ├── neural/   ├── semantic/  ├── linkgraph/ ├── ranker/
    ├── query/    ├── instant/   ├── freshness/ ├── session/
    ├── serving/  └── app/
```

---

## Quick Start

```bash
# First run: install models and launch
./raithe.sh

# Subsequent runs
./raithe.sh

# Install/verify models only, do not launch
./raithe.sh --install-only

# Force re-export all models
./raithe.sh --force-install

# Skip model check, run immediately
./raithe.sh --run-only

# Pass options through to the binary
./raithe.sh --json-logs
./raithe.sh --seeds /path/to/seeds.txt
./raithe.sh --config /path/to/engine.toml
```

Build the binary manually:

```bash
cargo build --release
# ORT shared library must be on LD_LIBRARY_PATH — raithe.sh handles this.
```

Search UI: `http://localhost:8080`  
JSON search: `http://localhost:8080/search?q=<query>`  
Metrics: `http://localhost:9090/metrics`  
Health: `http://localhost:8080/health` — returns `{"status":"ok"}` only

---

## Configuration

Load priority (highest wins):

1. Compiled-in `Default` impls
2. `data/config/engine.toml`
3. Environment variables (`RAITHE__SECTION__KEY`)
4. CLI `--config` argument

Hot-reload: `config::watcher()` watches `engine.toml` via `notify`. Invalid
configs are logged and discarded — the running config is never replaced with
an invalid one.

**Key defaults:**

| Key                       | Default                            |
|---------------------------|------------------------------------|
| `crawler.max_depth`       | `6`                                |
| `crawler.max_pages`       | `1_000_000`                        |
| `crawler.requests_per_sec`| `2.0` (per host, via `governor`)   |
| `crawler.min_seeds`       | `100` (startup fails below this)   |
| `indexer.writer_heap_mb`  | `1024`                             |
| `ranker.gbdt_trees`       | `300`                              |
| `ranker.gbdt_model_path`  | `data/models/ranker/gbdt.model`    |
| `ranker.reranker_top_k`   | `32`                               |
| `serving.bind`            | `0.0.0.0:8080`                     |
| `serving.metrics_bind`    | `0.0.0.0:9090`                     |

---

## Startup Sequence

`app::main()` initialises all subsystems in this order. Any failure halts
startup with a descriptive message — no silent failures.

1. Parse CLI arguments
2. Print RAiTHE ASCII banner
3. Load `Config`
4. Initialise tracing subscriber
5. Initialise `Metrics` registry
6. Validate seed list — fail if < 100 valid seeds
7. Initialise `Storage`
8. Initialise `NeuralEngine` — probe EPs, load all three `.onnx` models
9. Initialise `SemanticIndex`
10. Initialise `Indexer`
11. Initialise `LinkGraph`
12. Initialise `QueryProcessor`
13. Initialise `Ranker`
14. Initialise `InstantEngine`
15. Initialise `SessionStore`
16. Initialise `Crawler`
17. Initialise `FreshnessManager`
18. Spawn config hot-reload watcher task
19. Spawn crawler task
20. Spawn freshness task
21. Bind HTTP server — fail clearly on port conflict
22. Log `"raithe-se ready"` with bound address, active EP, document count

---

## Observability

| Metric                            | Type      | Labels          | Description                                  |
|-----------------------------------|-----------|-----------------|----------------------------------------------|
| `raithe_pages_crawled_total`      | Counter   | `{status}`      | Pages fetched by HTTP status class           |
| `raithe_index_documents_total`    | Gauge     | —               | Total docs in active index                   |
| `raithe_query_latency_seconds`    | Histogram | `{phase}`       | Latency per pipeline phase                   |
| `raithe_neural_inference_seconds` | Histogram | `{model}`       | ONNX inference latency per model             |
| `raithe_neural_execution_provider`| Gauge     | `{provider}`    | Active EP (1.0 selected, 0.0 others)         |
| `raithe_rank_phase3_calls_total`  | Counter   | —               | Cross-encoder invocations — must be > 0      |
| `raithe_errors_total`             | Counter   | `{crate, kind}` | All errors from public crate functions       |
| `raithe_crawl_queue_depth`        | Gauge     | —               | Current frontier queue depth                 |
| `raithe_rate_limited_total`       | Counter   | `{host}`        | Requests delayed by per-host rate limiter    |

---

## Security

All 37 security issues from the prototype are addressed by design.

- Path traversal (backup): all destination paths canonicalised; any path not
  rooted under the configured backup directory is rejected.
- Integer overflow (doc ID): `checked_add` throughout; returns
  `Error::DocIdExhausted` on exhaustion.
- Per-host rate limiting: `governor::RateLimiter` keyed by registered domain,
  mandatory in `Crawler::new()`.
- Per-request rate limiting: `tower-http` middleware on all serving endpoints.
- Slowloris mitigation: Axum timeout layer on all connections.
- Input validation: all query strings sanitised before reaching indexer or
  neural inference.
- CORS: restricted to configured allow-list. No wildcard in production.
- Security headers: CSP, HSTS, `X-Content-Type-Options`, `X-Frame-Options`.
- Health endpoint: returns `{"status":"ok"}` only — no version or build info
  exposed.

---

## Coding Standard

All Rust code is written to the **Rust Best Practices (RBP)** book authored by
Robert Perreault. RBP is the sole normative Rust authority for this project.
The Engineering Specification (v1.7) is the sole authority for architecture,
data contracts, and deployment constraints.

Key rules enforced without exception:

- `thiserror` concrete `Error` enums in all library crates. `anyhow`
  permitted in `crates/app` only.
- `.unwrap()` forbidden in all non-test production code.
- No glob imports (`use crate::*` forbidden). Exception: `use super::*`
  inside `#[cfg(test)]`.
- Import groups: `std/core/alloc` → third-party → `self/super/crate`.
- `mod.rs` for all multi-file module roots. `foo.rs + foo/` layout forbidden.
- `mod.rs` contains only `mod` declarations and `pub use` re-exports.
- UK spelling throughout (`behaviour`, `colour`, `initialise`, etc.).
- `Self` used wherever possible inside `impl` blocks.
- `let mut` scoped to the minimum necessary block.
- No explicit `drop()` calls — use scoped blocks.

---

## Dependency Policy

- All versions pinned in `[workspace.dependencies]`. All crates use
  `workspace = true`.
- `cargo audit` in CI — CRITICAL or HIGH advisory blocks the build.
- `tokio::main` used only in `crates/app`.
- `ort` uses `load-dynamic` feature — ORT shared library updated independently
  of the binary.
- `trust-dns-resolver` replaces the system resolver for all crawler DNS
  lookups.
- `llama-cpp-rs` is removed. Query rewriting routes through the `neural` crate
  via Qwen2.5 exported to ONNX.

---

## Prototype Defects Fixed in v2

The original prototype (`digitalrecompense/raithe-se`) was structurally sound
but functionally hollow. Every item below was broken by design or omission.

| Defect                                    | Fix                                              |
|-------------------------------------------|--------------------------------------------------|
| GBDT degraded to 10 toy trees             | 300 LambdaMART trees, mandatory                  |
| Cross-encoder never invoked               | Phase 3 always runs; metric enforces it          |
| Neural tokeniser was hash-based           | Real `tokenizers` crate with `tokenizer.json`    |
| `anyhow` used in library crates           | Concrete `thiserror` enums everywhere            |
| `.unwrap()` scattered in production paths | Forbidden; `.expect()` only where statically safe|
| No per-host rate limiting                 | `governor::RateLimiter` per domain, mandatory    |
| Path traversal in backup                  | Canonicalise + root check on all dest paths      |
| Integer overflow in doc-ID counter        | `checked_add` + `Error::DocIdExhausted`          |
| CORS wildcard                             | Configured allow-list only                       |
| Version info leaked in health endpoint    | `{"status":"ok"}` only                           |
| EP probe hung on Linux with CUDA drivers  | `with_execution_providers()` probe replaces `is_available()` |
| `optimum-cli` failed on 7B+ models        | `ORTModelForCausalLM` Python API in `raithe.sh`  |

---

## Amendment Log

| Ver | Date        | Summary                                                                                     |
|-----|-------------|---------------------------------------------------------------------------------------------|
| 1.0 | 14 Apr 2026 | Initial specification. Supersedes all prototype-era documents.                              |
| 1.1 | 14 Apr 2026 | Neural model layout amended. Per-model subdirectories. `install_models.sh` added.           |
| 1.2 | 14 Apr 2026 | Model stack finalised. BGE embedder/reranker + Qwen generator. `cross_encoder` → `reranker`.|
| 1.3 | 15 Apr 2026 | Implementation commenced. `raithe-common` fully implemented.                                |
| 1.4 | 15 Apr 2026 | `scraper`, `parser`, `indexer`, `neural` completed. `Embedding`, `ParsedQuery`, `QueryIntent`, `RawHit` added to `common`. `ScraperConfig` added. |
| 1.5 | 15 Apr 2026 | Spec alignment pass. `crawler::run()` marked `async`. Neural key fns corrected to `&mut self`. |
| 1.6 | 16 Apr 2026 | `install_models.sh` corrected. EP probe replaced (`is_available` hang fix). `RankerConfig` gains `gbdt_model_path`. DEV-001 CLOSED. |
| 1.7 | 19 Apr 2026 | `install_models.sh` superseded by `raithe.sh`. Generator ONNX export fixed via `ORTModelForCausalLM`. `torch` pinned to `2.5.1+cpu`. 147 unit tests passing. `cargo build --release` confirmed on `raithe-server`. |

---

## Deferred to v2

- Distributed crawling (multiple crawler instances sharing frontier)
- User accounts and personalised ranking beyond session
- Real-time LLM rewriting with a larger generative model
- Image and video indexing
- Full cluster mode (sharded indexer, replicated serving)

---

## License

Proprietary — © RAiTHE INDUSTRIES INCORPORATED 2026. All rights reserved.

*Engineered in Rust — Smart paths, fast results.*
