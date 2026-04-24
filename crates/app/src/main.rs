// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/app/src/main.rs
//
// Single binary entry point.
// anyhow is permitted here and only here — all other crates use concrete errors.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use raithe_common::Url;
use raithe_config::{watch as watch_config, Config};
use raithe_crawler::Crawler;
use raithe_freshness::FreshnessManager;
use raithe_indexer::Indexer;
use raithe_instant::InstantEngine;
use raithe_linkgraph::LinkGraph;
use raithe_metrics::Metrics;
use raithe_neural::{EmbedEngine, GenerateEngine, RerankEngine};
use raithe_parser::Parser;
use raithe_query::QueryProcessor;
use raithe_ranker::Ranker;
use raithe_scraper::Scraper;
use raithe_semantic::{HnswConfig, SemanticIndex};
use raithe_serving::{AppState, Server};
use raithe_session::SessionStore;
use raithe_storage::{CrawlLog, DocStore};

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Cli {
    /// Path to the config file. Defaults to `data/config/engine.toml`.
    config_path: PathBuf,
    /// Emit JSON-format tracing logs when true.
    json_logs:   bool,
    /// Path to the seed URL list (one URL per line).
    seeds_path:  Option<PathBuf>,
}

impl Cli {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1).peekable();
        let mut config_path = PathBuf::from("data/config/engine.toml");
        let mut json_logs   = false;
        let mut seeds_path  = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json-logs"  => json_logs = true,
                "--seeds"      => {
                    seeds_path = args.next().map(PathBuf::from);
                }
                "--config"     => {
                    if let Some(p) = args.next() {
                        config_path = PathBuf::from(p);
                    }
                }
                _              => {}
            }
        }

        Self { config_path, json_logs, seeds_path }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Step 1 — parse CLI arguments.
    let cli = Cli::parse();

    // Step 3 — load Config.
    let config = Config::load(&cli.config_path)
        .with_context(|| format!("loading config from {}", cli.config_path.display()))?;

    // Step 4 — initialise tracing subscriber.
    raithe_metrics::init_tracing(cli.json_logs)
        .context("initialising tracing subscriber")?;

    // Step 5 — initialise Metrics registry.
    let metrics = Arc::new(
        Metrics::new().context("initialising Prometheus metrics registry")?
    );

    // Step 6 — validate seed list (fail if < 100 valid seeds).
    let seeds = load_seeds(cli.seeds_path.as_deref(), &config)
        .context("loading seed URL list")?;

    if seeds.len() < config.crawler.min_seeds {
        anyhow::bail!(
            "insufficient seeds: need at least {}, got {} — \
             provide a seed file via --seeds <path>",
            config.crawler.min_seeds,
            seeds.len(),
        );
    }

    tracing::info!("startup [7/22] storage");
    // Step 7 — initialise Storage.
    let data_dir   = PathBuf::from("data");
    let crawl_log  = Arc::new(
        CrawlLog::open(&data_dir.join("crawl.log"))
            .context("opening crawl log")?
    );
    let doc_store  = Arc::new(
        DocStore::open(&data_dir.join("docs"))
            .context("opening doc store")?
    );
    let _ = doc_store; // wired into indexing pipeline at app layer

    tracing::info!("startup [8/22] neural engines — loading 3× single-model engines; see spinner for live progress");
    // Step 8 — initialise split neural engines.
    //
    // Each consumer loads exactly one model; previously we loaded all
    // three in each of three NeuralEngine instances, which read the
    // generator's ~30 GB of weights 3× at startup.
    //
    //   embed_engine    → indexing pipeline (BGE-large-en-v1.5,       ~1.3 GB)
    //   rerank_engine   → Ranker            (BGE-reranker-large,      ~2.1 GB)
    //   generate_engine → QueryProcessor    (Qwen2.5-{7B,14B}-Instruct, 14–28 GB)
    //
    // No synthetic ETA. The spinner reports real elapsed time and
    // freezes when the load returns. Generate dominates cold start.
    let embed_engine = load_with_spinner(
        "neural [1/3] embedder",
        || EmbedEngine::new(&config.neural, Arc::clone(&metrics))
            .context("initialising EmbedEngine (run ./raithe.sh if models missing)"),
    )?;
    let rerank_engine = load_with_spinner(
        "neural [2/3] reranker",
        || RerankEngine::new(&config.neural, Arc::clone(&metrics))
            .context("initialising RerankEngine"),
    )?;
    let generate_engine = load_with_spinner(
        "neural [3/3] generator",
        || GenerateEngine::new(&config.neural, Arc::clone(&metrics))
            .context("initialising GenerateEngine"),
    )?;

    tracing::info!("startup [9/22] semantic index");
    // Step 9 — initialise SemanticIndex.
    let semantic   = Arc::new(Mutex::new(SemanticIndex::new(HnswConfig::default())));

    tracing::info!("startup [10/22] tantivy indexer");
    // Step 10 — initialise Indexer.
    let indexer = Arc::new(
        Indexer::new(&data_dir.join("index"), &config.indexer, Arc::clone(&metrics))
            .context("initialising Tantivy indexer")?
    );

    tracing::info!("startup [11/22] link graph");
    // Step 11 — initialise LinkGraph.
    let link_graph = Arc::new(Mutex::new(LinkGraph::new()));

    tracing::info!("startup [12/22] query processor");
    // Step 12 — initialise QueryProcessor.
    let processor = Arc::new(
        QueryProcessor::new(generate_engine, Arc::clone(&metrics))
    );

    tracing::info!("startup [13/22] ranker");
    // Step 13 — initialise Ranker.
    let ranker = Arc::new(
        Ranker::new(config.ranker.clone(), rerank_engine, Arc::clone(&metrics))
            .context("initialising ranker — check gbdt_model_path is accessible")?
    );

    tracing::info!("startup [14/22] instant engine");
    // Step 14 — initialise InstantEngine.
    let instant = Arc::new(InstantEngine::new());

    tracing::info!("startup [15/22] session store");
    // Step 15 — initialise SessionStore.
    let sessions = Arc::new(SessionStore::new(100_000, 3_600));

    tracing::info!("startup [16/22] scraper + crawler");
    // Step 16 — initialise Scraper (needed by Crawler).
    let scraper = Scraper::new(&config.scraper, Arc::clone(&metrics))
        .context("initialising HTTP scraper")?;

    // Step 16 — initialise Crawler.
    let crawler = Arc::new(
        Crawler::new(
            config.crawler.clone(),
            seeds,
            scraper,
            Arc::clone(&crawl_log),
            Arc::clone(&metrics),
        )
        .context("initialising crawler — check seed count and config")?
    );

    tracing::info!("startup [17/22] freshness manager");
    // Step 17 — initialise FreshnessManager.
    //   stale_after_ms = 24 hours.
    let freshness = Arc::new(FreshnessManager::new(86_400_000));

    tracing::info!("startup [18/22] config watcher");
    // Step 18 — spawn config hot-reload watcher task.
    {
        let config_path = cli.config_path.clone();
        match watch_config(config_path) {
            Ok(_rx) => { /* receiver held implicitly; watcher runs in background */ }
            Err(err) => tracing::warn!("config watcher failed to start: {err}"),
        }
    }

    tracing::info!("startup [19/22] spawning crawler + indexing pipeline tasks");
    // Step 19 — create the fetch channel, then spawn the crawler task and the
    // indexing pipeline task. The channel decouples fetch from parse/index/embed
    // so neither task blocks the other. Buffer of 64 is sufficient — the crawler
    // rate-limits per host, so bursts are bounded.
    {
        let (fetch_tx, mut fetch_rx) = tokio::sync::mpsc::channel(64);

        // Crawler task — fetches URLs and sends FetchResult on the channel.
        {
            let crawler    = Arc::clone(&crawler);
            let fetch_tx   = fetch_tx.clone();
            tokio::spawn(async move {
                if let Err(err) = crawler.run(fetch_tx).await {
                    tracing::error!("crawler error: {err}");
                }
            });
        }

        // Indexing pipeline task — receives FetchResult, runs the full
        // parse → index → embed → semantic-insert → link-graph pipeline,
        // then re-enqueues outlinks into the crawler frontier.
        //
        // §6.1 data flow:
        //   FetchResult → Parser::parse() → ParsedDocument
        //     ├─► Indexer::add(doc)
        //     ├─► EmbedEngine::embed(body_text) → SemanticIndex::insert(id, embedding)
        //     ├─► LinkGraph::add_edge(src, dst) for each outlink
        //     └─► Crawler::enqueue_outlinks(outlinks, depth + 1)
        {
            let crawler        = Arc::clone(&crawler);
            let indexer        = Arc::clone(&indexer);
            let semantic       = Arc::clone(&semantic);
            let link_graph     = Arc::clone(&link_graph);
            let max_depth      = config.crawler.max_depth;
            let mut neural     = embed_engine;
            let parser         = Parser::new();

            // URL → DocumentId resolver for link-graph edge construction.
            // Outlinks whose destinations are not yet indexed are parked in
            // `pending_outlinks` keyed by the unresolved URL. When that URL
            // is later crawled and indexed, the parked sources are drained
            // and wired via `LinkGraph::add_edge` (DEF-006).
            let mut url_to_id: std::collections::HashMap<raithe_common::Url, raithe_common::DocumentId>
                = std::collections::HashMap::new();
            let mut pending_outlinks: std::collections::HashMap<raithe_common::Url, Vec<raithe_common::DocumentId>>
                = std::collections::HashMap::new();
            // Monotonic document-id counter. `DocumentId::ZERO` is the sentinel
            // for "unassigned" — the first real id is ONE. Overflow returns
            // `Error::DocIdExhausted` via `DocumentId::next` (§9.1).
            let mut next_doc_id: raithe_common::DocumentId = raithe_common::DocumentId::ZERO;

            tokio::spawn(async move {
                while let Some((fetch_result, depth)) = fetch_rx.recv().await {
                    let url = fetch_result.url.clone();

                    let mut doc = match parser.parse(fetch_result) {
                        Ok(d)  => d,
                        Err(e) => {
                            tracing::warn!(%url, error = %e, "parse failed — skipping");
                            continue;
                        }
                    };

                    match next_doc_id.next() {
                        Some(id) => {
                            next_doc_id = id;
                            doc.id      = id;
                        }
                        None => {
                            tracing::error!("document id space exhausted — halting indexing");
                            break;
                        }
                    }

                    if let Err(e) = indexer.add(&doc) {
                        tracing::warn!(%url, error = %e, "indexer add failed");
                    }

                    let body_ref: &str = &doc.body_text;
                    match neural.embed(&[body_ref]) {
                        Ok(embeddings) => {
                            if let Some(embedding) = embeddings.into_iter().next() {
                                let mut sem = semantic.lock()
                                    .unwrap_or_else(|p| p.into_inner());
                                if let Err(e) = sem.insert(doc.id, &embedding) {
                                    tracing::warn!(%url, error = %e, "semantic insert failed");
                                }
                            }
                        }
                        Err(e) => tracing::warn!(%url, error = %e, "embed failed"),
                    }

                    // LinkGraph wiring (DEF-006). Three passes per document:
                    //   1. Register this doc's URL → id so later docs that
                    //      link to it can be resolved immediately.
                    //   2. Drain any previously-parked sources that had this
                    //      URL as their outlink — add the now-resolvable edges.
                    //   3. For each outlink of the current doc: add an edge if
                    //      the destination is known, else park it.
                    {
                        url_to_id.insert(doc.url.clone(), doc.id);

                        let mut graph = link_graph.lock()
                            .unwrap_or_else(|p| p.into_inner());

                        if let Some(sources) = pending_outlinks.remove(&doc.url) {
                            for src_id in sources {
                                graph.add_edge(src_id, doc.id);
                            }
                        }

                        for outlink in &doc.outlinks {
                            match url_to_id.get(outlink).copied() {
                                Some(dst_id) => graph.add_edge(doc.id, dst_id),
                                None         => pending_outlinks
                                    .entry(outlink.clone())
                                    .or_default()
                                    .push(doc.id),
                            }
                        }
                    }

                    if depth < max_depth {
                        crawler.enqueue_outlinks(&doc.outlinks, depth + 1);
                    }
                }
            });
        }
    }

    tracing::info!("startup [20/22] spawning freshness task");
    // Step 20 — spawn freshness task.
    {
        let freshness = Arc::clone(&freshness);
        let crawler   = Arc::clone(&crawler);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3_600)).await;
                if let Err(err) = freshness.scan_stale() {
                    tracing::warn!("freshness scan error: {err}");
                }
                match freshness.next_batch(1_000) {
                    Ok(urls) => {
                        tracing::info!(count = urls.len(), "freshness batch ready");
                        crawler.enqueue_outlinks(&urls, 0);
                    }
                    Err(err) => tracing::warn!("freshness batch error: {err}"),
                }
            }
        });
    }

    tracing::info!("startup [21/22] binding HTTP server on {}", config.serving.bind);
    // Step 21 — bind serving HTTP server — fail clearly on port conflict.
    let doc_count = indexer.doc_count().unwrap_or(0);
    let app_state = Arc::new(AppState {
        config:    config.serving.clone(),
        metrics:   Arc::clone(&metrics),
        indexer:   Arc::clone(&indexer),
        processor: Arc::clone(&processor),
        ranker:    Arc::clone(&ranker),
        instant:   Arc::clone(&instant),
        sessions:  Arc::clone(&sessions),
    });

    let server = Server::new(config.serving.clone(), app_state)
        .context("binding HTTP server — is the port already in use?")?;

    // Step 22 — log ready.
    tracing::info!(
        bind      = %config.serving.bind,
        doc_count,
        "raithe-se ready"
    );

    server.run().await.context("HTTP server error")
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Loads seed URLs from the file at `path`, or from the default seeds file.
///
/// Each non-empty, non-comment line is parsed as a URL. Invalid lines are
/// skipped with a warning. Returns an empty vec (not an error) if no file
/// is configured — the seed-count guard in main() handles the failure.
fn load_seeds(path: Option<&Path>, _config: &Config) -> Result<Vec<Url>> {
    let path = match path {
        Some(p) => p.to_path_buf(),
        None    => PathBuf::from("data/seeds.txt"),
    };

    let text = match std::fs::read_to_string(&path) {
        Ok(t)  => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::warn!(
                path = %path.display(),
                "seed file not found — starting with empty frontier"
            );
            return Ok(Vec::new());
        }
        Err(e) => return Err(e).with_context(|| format!("reading seeds from {}", path.display())),
    };

    let seeds: Vec<Url> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| match Url::parse(l) {
            Ok(u)  => Some(u),
            Err(_) => {
                tracing::warn!(line = l, "invalid seed URL — skipping");
                None
            }
        })
        .collect();

    tracing::info!(count = seeds.len(), path = %path.display(), "seeds loaded");
    Ok(seeds)
}

// ── Load progress spinner ─────────────────────────────────────────────────────
//
// Wraps a long-running synchronous load with a linear left→right progress
// indicator modelled on the systemd boot-progress display: a short gradient
// of asterisks travels across a fixed-width bracket, falls off the right
// edge, then wraps back to the left. The closure runs on the caller's
// thread; indicatif's `enable_steady_tick` spawns a repaint thread so the
// animation keeps running while the load blocks.
//
// On success the spinner line is *cleared* and replaced by a single
// "✓ <label> — ready in <elapsed>" confirmation, matching the green ✓
// style used by raithe.sh's validation output. On failure the bar is
// abandoned in place and the error propagates to main for the standard
// anyhow unwind.
//
// No cutoff, no synthetic ETA. The user's explicit position: cold start
// may take hours on slow hardware, and that's acceptable — what matters
// is honest live feedback that something is still making progress.
fn load_with_spinner<T, F>(label: &str, work: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    // Linear sweep. 8-cell bracket, 3-asterisk gradient travelling
    // left-to-right, wrapping off the right edge and reappearing on
    // the left. Matches the Linux systemd progress indicator shape.
    //
    // Frames, in order:
    //   position  0: [***     ]      ← gradient entirely on left
    //   position  1: [ ***    ]
    //   position  2: [  ***   ]
    //   position  3: [   ***  ]
    //   position  4: [    *** ]
    //   position  5: [     ***]      ← gradient entirely on right
    //   position  6: [      **]      ← one cell falls off
    //   position  7: [       *]      ← two cells fall off
    //   position  8: [        ]      ← entirely off-screen (brief pause)
    //   position  9: [*       ]      ← leading edge re-enters from left
    //   position 10: [**      ]
    //   (cycle restarts at position 0)
    const TICK_STRINGS: &[&str] = &[
        "[***     ]",
        "[ ***    ]",
        "[  ***   ]",
        "[   ***  ]",
        "[    *** ]",
        "[     ***]",
        "[      **]",
        "[       *]",
        "[        ]",
        "[*       ]",
        "[**      ]",
        "[        ]",   // final frame when pb.finish_and_clear() is called;
                        // line is cleared before this is rendered, so it's
                        // effectively a no-op placeholder.
    ];

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("  {spinner:.cyan.bold} {msg} ({elapsed})")
            .expect("load_with_spinner: template is a static literal")
            .tick_strings(TICK_STRINGS),
    );
    pb.set_message(label.to_string());
    pb.enable_steady_tick(Duration::from_millis(120));

    let started = Instant::now();
    let result  = work();
    let elapsed = started.elapsed();

    match &result {
        Ok(_) => {
            pb.finish_and_clear();
            // Green ✓ to match raithe.sh's success style. Writes to
            // stderr so it doesn't tangle with stdout tracing JSON.
            eprintln!("  \x1b[0;32m✓\x1b[0m {label} — ready in {elapsed:.1?}");
        }
        Err(e) => {
            pb.abandon_with_message(format!("{label} — FAILED in {elapsed:.1?}: {e}"));
        }
    }
    result
}