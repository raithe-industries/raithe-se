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
understanding. Structured instant answers. Built to the raithe-se
Engineering Specification v1.0.

**Spec:** `raithe-se Engineering Specification v1.0 — 20 Apr 2026`
**Rust:** `rustc 1.95.0` (workspace minimum), built on `1.94.1+`
**Hardware:** GTX 1080 · CUDA 12.2 · Ubuntu 24.04 · i7-6700K · 32 GB

---

## Architecture

18-crate Cargo workspace. All crates `raithe-<n>`. All shared dependency
versions pinned in `[workspace.dependencies]`. ORT-related environment
variables (`ORT_DYLIB_PATH`, `LD_LIBRARY_PATH`) live in `.cargo/config.toml`
under `[env]` or are set by `raithe.sh` — never inside
`[workspace.dependencies]`.

| Crate        | Responsibility                                                   |
|--------------|------------------------------------------------------------------|
| `common`     | Shared primitive types — no logic, no I/O, no async              |
| `config`     | Load, validate, and hot-reload `engine.toml`                     |
| `metrics`    | Prometheus registry + tracing init — all handles registered here |
| `storage`    | Crawl log, doc store, mmap, backups                              |
| `scraper`    | Single-URL HTTP fetch → `FetchResult`                            |
| `crawler`    | URL frontier, politeness scheduling, robots.txt, dispatch        |
| `parser`     | Raw HTML bytes → `ParsedDocument`                                |
| `indexer`    | Tantivy inverted index — BM25F, cached reader                    |
| `neural`     | ONNX Runtime inference — hardware auto-EP, 3 model handles       |
| `semantic`   | HNSW ANN index for dense embedding retrieval (from scratch)      |
| `linkgraph`  | CSR link graph + iterative PageRank                              |
| `ranker`     | Three-phase ranking pipeline (BM25F → GBDT → cross-encoder)      |
| `query`      | Tokenise, spell-correct, intent classify, synonym expand, rewrite|
| `instant`    | Instant answers — arithmetic, unit conversions, time zones       |
| `freshness`  | Incremental re-crawl, stale detection, tombstones                |
| `session`    | In-memory session context + reformulation detection              |
| `serving`    | Axum HTTP server + search UI (JSON/HTML content negotiation)     |
| `app`        | Single-binary entry point — CLI, banner, init sequence           |

---

## Ranking Pipeline

Every query passes all three phases in order. Phase 3 is never skipped.

| Phase | Mechanism                                    | Implementation          |
|-------|----------------------------------------------|-------------------------|
| 1     | BM25F field-weighted scoring (3/2/2/1)       | Tantivy                 |
| 2     | LambdaMART re-rank                           | `gbdt` crate, 300 trees |
| 3     | Neural cross-encoder re-rank (top 32)        | BGE-reranker-large ONNX |

`raithe_rank_phase3_calls_total` must be > 0 after any search.

---

## Neural Models

`raithe.sh` is the sole operational entrypoint. It installs and validates all
three models, then launches the binary.

| Directory    | Model / HF ID                    | Task                                    |
|--------------|----------------------------------|-----------------------------------------|
| `embedder/`  | `BAAI/bge-large-en-v1.5`         | feature-extraction → `embed()`          |
| `reranker/`  | `BAAI/bge-reranker-large`        | text-classification → `rerank()`        |
| `generator/` | `Qwen/Qwen2.5-7B-Instruct`       | text-generation → `generate()` (autoregressive) |

Both `model.onnx` and `tokenizer.json` must be present in each subdirectory.
A missing file is a startup `Error::ModelNotFound`. No silent fallback.

Generator export uses the `ORTModelForCausalLM` Python API (not `optimum-cli`)
to bypass the 2 GiB protobuf limit on 7B+ models.

**Hardware auto-detection order (unconditional probe):** CUDA → DirectML → CoreML → CPU.
Each candidate is exercised against the embedder's `model.onnx` in a
bounded-duration thread (default 3 s). Hosts lacking the required toolkit
for a provider fail the probe and proceed to the next candidate.

Generator is hardware-adaptive: 14B model selected when RAM ≥ 64 GB and VRAM ≥ 16 GB.

**Single `NeuralEngine` per process**, shared via `Arc<Mutex<NeuralEngine>>`.

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
├── .cargo/config.toml      # env vars (ORT_DYLIB_PATH etc.), linker overrides
├── raithe.sh               # sole operational entrypoint: install + launch
├── data/
│   ├── config/
│   │   └── engine.toml     # default runtime configuration
│   └── models/             # NOT committed — populated by raithe.sh
│       ├── embedder/       # model.onnx + tokenizer.json
│       ├── reranker/       # model.onnx + tokenizer.json
│       └── generator/      # model.onnx + model.onnx_data + tokenizer.json
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
# ORT shared library must be on ORT_DYLIB_PATH — raithe.sh handles this.
```

Search UI: `http://localhost:8080`
JSON search: `http://localhost:8080/search?q=<query>` (Accept: application/json or `?format=json`)
Metrics: `http://localhost:9090/metrics`
Health: `http://localhost:8080/health` — returns `{"status":"ok"}` only

---

## Configuration

Load priority (highest wins):

1. Compiled-in `Default` impls
2. `data/config/engine.toml`
3. Environment variables (`RAITHE__SECTION__KEY`)
4. CLI `--config` argument

Hot-reload: `config::watch(path)` watches `engine.toml` via `notify`. Invalid
configs are logged and discarded — the running config is never replaced with
an invalid one.

**Key defaults:**

| Key                               | Default                            |
|-----------------------------------|------------------------------------|
| `crawler.max_depth`               | `6`                                |
| `crawler.max_pages`               | `1_000_000`                        |
| `crawler.requests_per_sec`        | `2.0` (per host, via `governor`)   |
| `crawler.min_seeds`               | `100` (startup fails below this)   |
| `indexer.writer_heap_mb`          | `1024`                             |
| `indexer.top_k`                   | `100`                              |
| `ranker.gbdt_trees`               | `300`                              |
| `ranker.gbdt_model_path`          | `data/models/ranker/gbdt.model`    |
| `ranker.reranker_top_k`           | `32`                               |
| `neural.generator_max_tokens`     | `256`                              |
| `serving.bind`                    | `0.0.0.0:8080`                     |
| `serving.metrics_bind`            | `0.0.0.0:9090`                     |
| `serving.max_request_bytes`       | `8192`                             |
| `serving.request_timeout_secs`    | `30`                               |
| `serving.session_max`             | `100_000`                          |
| `serving.session_ttl_secs`        | `3_600`                            |
| `freshness.stale_after_ms`        | `86_400_000` (24 hours)            |
| `freshness.scan_period_secs`      | `3_600`                            |

---

## Startup Sequence

`app::main()` initialises all subsystems in this order. Any failure halts
startup with a descriptive message.

1. Parse CLI arguments
2. Print RAiTHE ASCII banner (via `raithe.sh`; the app skips when launched via `raithe.sh`)
3. Load `Config`
4. Initialise tracing subscriber
5. Initialise `Metrics` registry
6. Validate seed list — fail if < 100 valid seeds
7. Initialise `Storage` (CrawlLog + DocStore, both wired)
8. Initialise `NeuralEngine` (one instance, `Arc<Mutex<_>>`) — probe EPs, load all three `.onnx` models
9. Initialise `SemanticIndex`
10. Initialise `Indexer` (placed into `AppState`)
11. Initialise `LinkGraph` (reachable from indexing pipeline)
12. Initialise `QueryProcessor` (shares the `Arc<Mutex<NeuralEngine>>`)
13. Initialise `Ranker` (shares the `Arc<Mutex<NeuralEngine>>`)
14. Initialise `InstantEngine`
15. Initialise `SessionStore` (params from `ServingConfig`)
16. Initialise `Crawler`
17. Initialise `FreshnessManager` (params from `FreshnessConfig`)
18. Spawn config hot-reload watcher task
19. Spawn crawler task and indexing-pipeline task
    (`FetchResult → Parser → Indexer::add → NeuralEngine::embed → SemanticIndex::insert → LinkGraph::add_edge → Crawler::enqueue_outlinks`)
20. Spawn freshness task
21. Bind HTTP server — fail clearly on port conflict (AppState includes `indexer`)
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

- Path traversal (backup): all destination paths canonicalised; any path not
  rooted under the configured backup directory is rejected.
- Integer overflow (doc ID): `checked_add` throughout; returns
  `Error::DocIdExhausted` on exhaustion.
- Per-host rate limiting: `governor::RateLimiter` keyed by registered domain,
  mandatory in `Crawler::new()`.
- Per-request rate limiting: `tower-http` middleware on all serving endpoints.
- Slowloris mitigation: Axum timeout layer on all connections.
- Body-size gate in scraper: streaming via `bytes_stream()` with a rolling
  guard.
- Input validation: all query strings sanitised before reaching indexer or
  neural inference.
- CORS: `tower-http`'s `CorsLayer` restricted to configured allow-list. No
  wildcard in production.
- Security headers: CSP, HSTS, `X-Content-Type-Options`, `X-Frame-Options`.
- Health endpoint: returns `{"status":"ok"}` only — no version or build info
  exposed.
- Session propagation: cookie (`raithe_sid`) or header (`X-Raithe-Session`).
  Missing or malformed values cause a fresh id to be minted and returned via
  `Set-Cookie`.

---

## Coding Standard

All Rust code is written to the **Rust Best Practices (RBP)** book authored by
Robert Perreault. RBP is the sole normative Rust authority for this project.
The Engineering Specification is the sole authority for architecture, data
contracts, and deployment constraints.

Key rules enforced without exception:

- `thiserror` concrete `Error` enums in all library crates. `anyhow`
  permitted in `crates/app` only.
- `.unwrap()` forbidden in all non-test production code. `.expect()` only
  where the precondition is statically verifiable.
- No glob imports. Exception: `use super::*` inside `#[cfg(test)]`.
- Import groups: `std/core/alloc` → third-party → `self/super/crate`.
- File-level order: `mod` / `pub mod` declarations first, then `pub use`
  re-exports, then `use` imports.
- `mod.rs` for all multi-file module roots. `foo.rs + foo/` layout forbidden.
- `mod.rs` contains only `mod` declarations and `pub use` re-exports.
- UK spelling throughout (`behaviour`, `colour`, `initialise`).
- `Self` used wherever possible inside `impl` blocks.
- `let mut` scoped to the minimum necessary block.
- No explicit `drop()` calls — use scoped blocks.
- Derive order: `Copy` first, then std items in lexicographic order, then
  third-party.
- Struct field visibility order: `pub` first, `pub(crate)` next, private
  last.
- Fields prefixed with `_` must be removed or wired — an unused field is a
  defect.

---

## Dependency Policy

- All versions pinned in `[workspace.dependencies]`. All crates use
  `workspace = true`.
- Environment variables live in `.cargo/config.toml` under `[env]` or are
  set by `raithe.sh` — not inside `[workspace.dependencies]`.
- `cargo audit` in CI — CRITICAL or HIGH advisory blocks the build.
- `tokio::main` used only in `crates/app`.
- `ort` uses `load-dynamic` feature — ORT shared library updated independently
  of the binary.
- `trust-dns-resolver` replaces the system resolver for all crawler DNS
  lookups.

---

## Amendment Log

| Ver | Date        | Summary                                                                       |
|-----|-------------|-------------------------------------------------------------------------------|
| 1.0 | 20 Apr 2026 | Authoritative engineering specification for raithe-se.                        |

---

## Deferred to Future Versions

- Distributed crawling (multiple crawler instances sharing frontier)
- User accounts and personalised ranking beyond session
- Real-time LLM rewriting with a larger generative model
- Image and video indexing
- Full cluster mode (sharded indexer, replicated serving)
- KV-cache optimisation for autoregressive decode in `NeuralEngine::generate`

---

## License

Proprietary — © RAiTHE INDUSTRIES INCORPORATED 2026. All rights reserved.

*Engineered in Rust — Smart paths, fast results.*
