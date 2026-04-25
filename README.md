# RAiTHE Search Engine (`raithe-se`)

© RAiTHE INDUSTRIES INCORPORATED 2026

```text
                         ██████╗  █████╗ ██╗████████╗██╗  ██╗███████╗
                         ██╔══██╗██╔══██╗   ╚══██╔══╝██║  ██║██╔════╝
                         ██████╔╝███████║██║   ██║   ███████║█████╗
                         ██╔══██╗██╔══██║██║   ██║   ██╔══██║██╔══╝
                         ██║  ██║██║  ██║██║   ██║   ██║  ██║███████╗
                         ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
                                         SEARCH ENGINE
```

`raithe-se` is a world-class search engine built in Rust for serious workstation-class PCs: fast CPUs, fast SSDs, 32 GB+ RAM, and CUDA-capable NVIDIA GPUs with 8GB VRAM minimum. It is designed to scale with the machine it boots on, using all resources; CPU, RAM, and GPU’s VRAM together rather than treating neural acceleration as an afterthought.

The project is being built in staged layers:

1. a reliable crawl → parse → index → search core;
2. durable URL identity and restart-safe indexing;
3. CUDA-accelerated neural assistance;
4. semantic retrieval, learned ranking, and high-value instant answers.

The current default runtime is **Phase 1**: a reliable BM25 search core with CUDA/neural systems kept outside the critical path until the core is stable. The long-term target is a **CUDA-first adaptive search engine** that uses available GPU, VRAM, CPU, RAM, and SSD throughput intelligently.

For the detailed architecture and roadmap, see [`docs/ENGINEERING_SPEC.md`](https://github.com/raithe-industries/raithe-se/blob/main/data/docs/ENGINEERING_SPEC.md).

---

## Current Status

### Working/default path

- Phase 1 single-node search engine.
- Polite crawler with robots.txt checks and per-host rate limiting.
- HTML fetch, parse, and link extraction pipeline.
- Tantivy inverted index with stored snippets.
- Batch and periodic index commits.
- JSON and HTML search responses.
- `/debug/stats`, `/health`, and `/metrics` endpoints.
- Launcher can skip CUDA/model preflight while the Phase 1 core is being proven.

### In progress / next required hardening

- Durable URL → document ID registry.
- True URL upsert/delete-by-URL in the indexer.
- Retry queue with backoff for transient fetch errors.
- End-to-end local crawl/search/restart integration test.
- CUDA-aware runtime planner.
- Reintroduction of neural systems with GPU scheduling, batching, and fallback rules.
- LLM-assisted instant answers as a first-class feature.
- Query rewriting as a secondary LLM feature.

---

## Design Philosophy

`raithe-se` is not meant to be constrained to one exact build machine. It should scale to the PC it is booted on.

The expected working environment is a modern creator/developer workstation:

| Component | Practical baseline |
|---|---|
| CPU | modern high-clock CPU, approximately 4.0 GHz+ boost class |
| RAM | 32 GB+ |
| GPU | NVIDIA CUDA GPU, 8 GB+ VRAM minimum |
| Storage | fast NVMe/SSD |
| OS | Linux-first development target |

The engine should still keep the crawl/index/search core independent from neural startup so debugging remains possible. But CUDA is a major design pillar: embeddings, reranking, LLM-assisted answers, and future semantic workloads should use the GPU effectively when available.

The intended policy is:

1. **Core reliability first:** crawl, parse, index, commit, and search must be accurate, boring, and dependable.
2. **CUDA matters:** GPU acceleration is a first-class path, not a bolt-on.
3. **CPU and GPU cooperate:** CPU owns orchestration, crawling, parsing, indexing, and serving; GPU owns high-throughput neural inference.
4. **Instant answers matter most:** LLM assistance should primarily improve direct answers, summaries, calculations, and structured responses.
5. **Query rewriting is useful but secondary:** rewriting should help intuitively recall, but it must not become a fragile dependency.
6. **Adaptive resource use:** model choice, batch size, GPU memory limits, writer heap, and worker counts should follow detected resources.
7. **No silent degradation:** optional neural features must report their status clearly.

---

















## Workspace Layout

```text
raithe-se/
├── Cargo.lock
├── Cargo.toml
├── crates
│   ├── app
│   │   ├── Cargo.toml
│   │   ├── src
│   │   │   ├── lib.rs
│   │   │   └── main.rs
│   │   └── tests
│   │       └── end_to_end.rs
│   ├── common
│   │   ├── Cargo.toml
│   │   └── src
│   │       ├── document_id.rs
│   │       ├── embedding.rs
│   │       ├── lib.rs
│   │       ├── query_types.rs
│   │       ├── simhash.rs
│   │       ├── timestamp.rs
│   │       └── url_type.rs
│   ├── config
│   │   ├── Cargo.toml
│   │   └── src
│   │       ├── crawler.rs
│   │       ├── engine.rs
│   │       ├── indexer.rs
│   │       ├── lib.rs
│   │       ├── neural.rs
│   │       ├── ranker.rs
│   │       ├── scraper.rs
│   │       ├── serving.rs
│   │       └── watcher.rs
│   ├── crawler
│   │   ├── Cargo.toml
│   │   └── src
│   │       ├── frontier.rs
│   │       ├── lib.rs
│   │       ├── policy.rs
│   │       └── robots.rs
│   ├── freshness
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── indexer
│   │   ├── benches
│   │   │   └── indexer_throughput.rs
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── instant
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── linkgraph
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── metrics
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── neural
│   │   ├── benches
│   │   │   └── neural_embed_latency.rs
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── parser
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── query
│   │   ├── benches
│   │   │   └── query_latency.rs
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── ranker
│   │   ├── benches
│   │   │   └── ranker_latency.rs
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── scraper
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── semantic
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   ├── serving
│   │   ├── assets
│   │   │   └── Alacarte.otf
│   │   ├── Cargo.toml
│   │   └── src
│   │       ├── lib.rs
│   │       └── templates.rs
│   ├── session
│   │   ├── Cargo.toml
│   │   └── src
│   │       └── lib.rs
│   └── storage
│       ├── Cargo.toml
│       └── src
│           ├── backup.rs
│           ├── crawl_log.rs
│           ├── doc_store.rs
│           ├── lib.rs
│           └── mmap_file.rs
├── data
│   ├── config
│   │   └── engine.toml
│   ├── crawl.log
│   ├── docs
│   │   └── ENGINEERING_SPEC.md
│   ├── index
│   │   ├── 0788215bd87e4cc892eb92ef228a0df1.fast
│   │   ├── 0788215bd87e4cc892eb92ef228a0df1.fieldnorm
│   │   ├── 0788215bd87e4cc892eb92ef228a0df1.idx
│   │   ├── 0788215bd87e4cc892eb92ef228a0df1.pos
│   │   ├── 0788215bd87e4cc892eb92ef228a0df1.store
│   │   ├── 0788215bd87e4cc892eb92ef228a0df1.term
│   │   ├── 176fdc8a9119446abeee6b3331fdebfd.fast
│   │   ├── 176fdc8a9119446abeee6b3331fdebfd.fieldnorm
│   │   ├── 176fdc8a9119446abeee6b3331fdebfd.idx
│   │   ├── 176fdc8a9119446abeee6b3331fdebfd.pos
│   │   ├── 176fdc8a9119446abeee6b3331fdebfd.store
│   │   ├── 176fdc8a9119446abeee6b3331fdebfd.term
│   │   ├── 5b9a818e1f7a4bc7814a3005f7f46a5a.fast
│   │   ├── 5b9a818e1f7a4bc7814a3005f7f46a5a.fieldnorm
│   │   ├── 5b9a818e1f7a4bc7814a3005f7f46a5a.idx
│   │   ├── 5b9a818e1f7a4bc7814a3005f7f46a5a.pos
│   │   ├── 5b9a818e1f7a4bc7814a3005f7f46a5a.store
│   │   ├── 5b9a818e1f7a4bc7814a3005f7f46a5a.term
│   │   ├── 9f7566de570f441eb7408a7ebcf35c63.fast
│   │   ├── 9f7566de570f441eb7408a7ebcf35c63.fieldnorm
│   │   ├── 9f7566de570f441eb7408a7ebcf35c63.idx
│   │   ├── 9f7566de570f441eb7408a7ebcf35c63.pos
│   │   ├── 9f7566de570f441eb7408a7ebcf35c63.store
│   │   ├── 9f7566de570f441eb7408a7ebcf35c63.term
│   │   ├── ecef9704e0eb409dae9b53a13c12dda3.fast
│   │   ├── ecef9704e0eb409dae9b53a13c12dda3.fieldnorm
│   │   ├── ecef9704e0eb409dae9b53a13c12dda3.idx
│   │   ├── ecef9704e0eb409dae9b53a13c12dda3.pos
│   │   ├── ecef9704e0eb409dae9b53a13c12dda3.store
│   │   ├── ecef9704e0eb409dae9b53a13c12dda3.term
│   │   ├── meta.json
│   │   └── VERSION
│   ├── models
│   │   ├── embedder
│   │   │   ├── config.json
│   │   │   ├── model.onnx
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── tokenizer.json
│   │   │   └── vocab.txt
│   │   ├── generator
│   │   │   ├── model.onnx
│   │   │   ├── model.onnx.data
│   │   │   └── tokenizer.json
│   │   ├── README.md
│   │   └── reranker
│   │       ├── config.json
│   │       ├── model.onnx
│   │       ├── model.onnx_data
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       └── tokenizer.json
│   └── seeds.txt
├── LICENSE.md
├── raithe.sh
├── README.md
└── target
    
```

---

## Runtime Modes

### Phase 1: BM25 core

Phase 1 proves the foundation:

```text
seed URLs
  → crawler
  → scraper
  → parser
  → Tantivy indexer
  → committed BM25 index
  → Axum serving API/UI
```

Neural features are temporarily outside the critical path so pipeline bugs are easy to find. This does not mean `raithe-se` is intended to be CPU-only, it is not. It means the search engine core must be correct before GPU features become mandatory for quality control purposes.

### Phase 2: Durable identity

Adds:

- durable URL registry;
- stable URL → document ID mapping;
- index upsert;
- restart-safe recrawl;
- no duplicate results for the same canonical URL.

### Phase 3: CUDA neural assistance

Adds CUDA-aware neural workloads:

- embedding generation;
- cross-encoder reranking;
- LLM-assisted instant answers;
- LLM-assisted query rewriting;
- model/provider health reporting;
- batch scheduling and VRAM budgets.

### Phase 4: Semantic and learned ranking

Adds:

- dense retrieval;
- BM25 + vector result fusion;
- PageRank/link features;
- freshness features;
- learned ranker inputs;
- final reranking.

---

## Quick Start

### Build

```bash
cargo build --release
```

### Run Phase 1

```bash
./raithe.sh --seeds data/seeds.txt
```

### Run binary directly

```bash
./target/release/raithe-se --seeds data/seeds.txt
```

### Enable neural/CUDA preflight explicitly

```bash
./raithe.sh --with-neural --seeds data/seeds.txt
```

Use this when intentionally testing ONNX Runtime, CUDA, neural models, embeddings, reranking, or LLM-assisted answers.

---

## Useful Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | Search UI |
| `/search?q=...` | Search, HTML or JSON depending on `Accept` header |
| `/search?q=...&format=json` | Force JSON search response |
| `/search?q=...&format=html` | Force HTML search response |
| `/debug/stats` | Crawl/index health counters |
| `/health` | Minimal health check |
| `/metrics` | Prometheus metrics |

Example:

```bash
curl 'http://127.0.0.1:8080/search?q=rust&format=json'
```

---

## Instant Answers

Instant answers are a first-class product feature.

The instant-answer path should eventually combine:

- deterministic answers for arithmetic, units, dates, and structured facts;
- index-grounded answers from crawled documents;
- LLM-assisted synthesis for concise direct answers;
- source-aware responses where possible;
- graceful fallback to normal search results.

LLM assistance is more valuable here than in query rewriting. Query rewriting can improve recall, but instant answers directly improve user value.

Target flow:

```text
query
  → deterministic instant-answer checks
  → index lookup / grounding
  → optional LLM synthesis
  → instant answer + supporting search results
```

The LLM must not hallucinate unsupported facts silently. When the answer is not grounded, the UI/API should make that clear or fall back to search results.

---

## Hardware Policy

`raithe-se` should scale to the host instead of forcing the host to match one hardcoded profile.

| Host capability | Expected behaviour |
|---|---|
| 32 GB RAM, 8 GB VRAM | Standard CUDA workstation tier |
| 64 GB+ RAM, 12–16 GB VRAM | Larger batches, stronger reranking/embedding workloads |
| 24 GB+ VRAM | Larger generator/reranker models and heavier semantic workloads |
| Multiple GPUs | Assign neural workloads by memory budget and queue pressure |
| Fast NVMe | Larger index/write buffers and faster commit cadence |
| CPU-only fallback | Core search should still be debuggable, but not the quality target |

CPU and GPU responsibilities:

| CPU | GPU/CUDA |
|---|---|
| crawling | embeddings |
| networking | cross-encoder reranking |
| robots.txt | LLM-assisted instant answers |
| parsing | query rewriting, where enabled |
| URL registry | semantic/vector-heavy batches |
| Tantivy indexing | future neural ranking models |
| BM25 search | batched inference |
| serving/API/UI | acceleration telemetry |

---

## Phase 1 Acceptance Test

The core engine is considered healthy when this passes repeatedly:

```text
1. Start a local three-page HTTP site.
2. Seed only the root page.
3. raithe-se crawls all three pages.
4. /debug/stats reports committed >= 3.
5. /search?q=<unique-term-from-page-a>&format=json returns page A.
6. Restart raithe-se.
7. The same query still returns page A.
8. Re-crawling does not create duplicate results for the same URL.
```

Until this test or an equivalent or greater test passes, advanced features should remain outside the critical path.

---

## Development Commands

```bash
cargo fmt
cargo test
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release
```

Run with debug logging:

```bash
RUST_LOG=info ./raithe.sh --seeds data/seeds.txt
```

---

## Documentation

- [`docs/ENGINEERING_SPEC.md`](docs/ENGINEERING_SPEC.md) — engineering specification and staged roadmap.
- `data/config/engine.toml` — runtime defaults.
- `raithe.sh` — operational launcher.

---

## License

Engineered in Rust; April, 2026, by AI-assisted coder, and founder, Robert Rolland "Stone" Perreault.

Proprietary — © RAiTHE INDUSTRIES INCORPORATED 2026. All rights reserved.


