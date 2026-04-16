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

**The World's Best Search Engine in Rust.**

Single-node architecture with hybrid neural ranking, LLM-assisted query
understanding, and structured instant answers. Built entirely from the ground
up per the raithe-se Engineering Specification v1.2.

## Architecture

18-crate Cargo workspace:

| Crate        | Description                                              |
|--------------|----------------------------------------------------------|
| `common`     | Shared primitive types — no logic, no I/O, no async     |
| `config`     | Config loading, validation, hot-reload                   |
| `metrics`    | Prometheus registry + tracing initialisation             |
| `storage`    | Crawl log, doc store, mmap, backups                      |
| `scraper`    | Single-URL HTTP fetch → raw bytes                        |
| `crawler`    | Frontier, politeness, robots.txt, dispatch               |
| `parser`     | Raw HTML → ParsedDocument                                |
| `indexer`    | Tantivy inverted index                                   |
| `neural`     | ONNX Runtime, hardware auto-EP, 3 model handles          |
| `semantic`   | HNSW ANN index for dense embeddings                      |
| `linkgraph`  | CSR link graph + PageRank                                |
| `ranker`     | BM25F → GBDT (300 trees) → BGE-reranker-large            |
| `query`      | Tokenise, spell-correct, intent, synonyms, rewrite       |
| `instant`    | Instant answers — calculator, conversions, time zones    |
| `freshness`  | Incremental re-crawl + tombstones                        |
| `session`    | In-memory session context + reformulation detection      |
| `serving`    | Axum HTTP server + search UI                             |
| `app`        | Single-binary entry point                                |

## Neural Models

Run `install_models.sh` to download all three models via optimum-cli.

| Directory        | Model                  | Role                              |
|------------------|------------------------|-----------------------------------|
| `embedder/`      | BAAI/bge-large-en-v1.5 | Dense bi-encoder embedding        |
| `reranker/`      | BAAI/bge-reranker-large| Phase 3 cross-encoder ranking     |
| `generator/`     | Qwen/Qwen2.5-7B-Instruct| Query understanding + rewriting  |

Hardware auto-detected at startup: CUDA → DirectML → CoreML → CPU.
Generator: hardware-adaptive — 14B selected when RAM ≥ 64 GB and VRAM ≥ 16 GB.

## Quick Start

```bash
# Add your .onnx files to data/models/ first, then:
cargo build --release
./target/release/raithe-se
```

Search UI: `http://localhost:8080`
Metrics:   `http://localhost:9090/metrics`

## Coding Standard

All code follows the Rust Best Practices book (RBP) authored by Robert Perreault.
RBP is the sole normative Rust authority for this project.

## License

Proprietary — © RAiTHE INDUSTRIES INCORPORATED 2026. All rights reserved.

---

* Engineered in Rust — Smart paths, fast results.*
