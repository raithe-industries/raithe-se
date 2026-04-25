// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/app/src/lib.rs
//
// Library facet of the app — exposes pipeline construction so integration
// tests can build the engine without going through `fn main`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use raithe_common::Url;
use raithe_config::Config;
use raithe_crawler::Crawler;
use raithe_indexer::Indexer;
use raithe_instant::InstantEngine;
use raithe_metrics::Metrics;
use raithe_parser::Parser;
use raithe_query::QueryProcessor;
use raithe_ranker::Ranker;
use raithe_scraper::Scraper;
use raithe_serving::{AppState, DebugStats, Server};
use raithe_session::SessionStore;
use raithe_storage::{CrawlLog, DocStore};

/// Phase 1 engine handles. Held by both `main` and integration tests.
pub struct Phase1Engine {
    pub config:   Config,
    pub metrics:  Arc<Metrics>,
    pub crawler:  Arc<Crawler>,
    pub indexer:  Arc<Indexer>,
    pub debug:    Arc<DebugStats>,
    pub state:    Arc<AppState>,
}

impl Phase1Engine {
    /// Builds a Phase-1 engine in `data_dir`. Caller is responsible for
    /// spawning the crawler, indexing pipeline, and commit task — see
    /// `spawn_pipeline` for the standard wiring.
    pub fn build(config: Config, seeds: Vec<Url>, data_dir: &Path) -> Result<Self> {
        if !config.engine.phase1_only {
            anyhow::bail!("Phase1Engine::build requires engine.phase1_only = true");
        }

        let metrics = Arc::new(Metrics::new().context("metrics init")?);

        let crawl_log = Arc::new(CrawlLog::open(&data_dir.join("crawl.log")).context("crawl log")?);
        let _doc_store = Arc::new(DocStore::open(&data_dir.join("docs")).context("doc store")?);

        let indexer = Arc::new(
            Indexer::new(&data_dir.join("index"), &config.indexer, Arc::clone(&metrics))
                .context("indexer init")?
        );

        let processor = Arc::new(QueryProcessor::pass_through(Arc::clone(&metrics)));
        let ranker    = Arc::new(
            Ranker::bm25_only(config.ranker.clone(), Arc::clone(&metrics)).context("ranker init")?
        );
        let instant  = Arc::new(InstantEngine::new());
        let sessions = Arc::new(SessionStore::new(100_000, config.serving.session_ttl_secs));

        let scraper = Scraper::new(&config.scraper, Arc::clone(&metrics)).context("scraper init")?;
        let crawler = Arc::new(
            Crawler::new(
                config.crawler.clone(),
                seeds,
                scraper,
                Arc::clone(&crawl_log),
                Arc::clone(&metrics),
            ).context("crawler init")?
        );

        let debug = Arc::new(DebugStats::new());

        let state = Arc::new(AppState {
            config:    config.serving.clone(),
            metrics:   Arc::clone(&metrics),
            indexer:   Arc::clone(&indexer),
            processor,
            ranker,
            instant,
            sessions,
            crawler:   Arc::clone(&crawler),
            debug:     Arc::clone(&debug),
        });

        Ok(Self { config, metrics, crawler, indexer, debug, state })
    }

    /// Spawns the crawler, indexing pipeline, and periodic-commit task.
    /// Returns immediately. Tasks run until the runtime is shut down.
    pub fn spawn_pipeline(&self) {
        let (fetch_tx, mut fetch_rx) = tokio::sync::mpsc::channel(64);

        // Crawler.
        {
            let crawler  = Arc::clone(&self.crawler);
            let fetch_tx = fetch_tx.clone();
            tokio::spawn(async move {
                if let Err(err) = crawler.run(fetch_tx).await {
                    tracing::error!("crawler error: {err}");
                }
            });
        }

        // Indexing pipeline (parse -> add).
        {
            let crawler   = Arc::clone(&self.crawler);
            let indexer   = Arc::clone(&self.indexer);
            let debug     = Arc::clone(&self.debug);
            let max_depth = self.config.crawler.max_depth;
            let threshold = self.config.engine.commit_every_docs;
            let parser    = Parser::new();
            let mut next_doc_id: raithe_common::DocumentId = raithe_common::DocumentId::ZERO;

            tokio::spawn(async move {
                while let Some((fetch_result, depth)) = fetch_rx.recv().await {
                    let url = fetch_result.url.clone();

                    let mut doc = match parser.parse(fetch_result) {
                        Ok(d)  => d,
                        Err(e) => {
                            let msg = format!("parse {url}: {e}");
                            tracing::warn!("{msg}");
                            debug.record_parser_error(msg);
                            continue;
                        }
                    };

                    match next_doc_id.next() {
                        Some(id) => { next_doc_id = id; doc.id = id; }
                        None     => {
                            tracing::error!("document id space exhausted — halting indexing");
                            break;
                        }
                    }

                    if let Err(e) = indexer.add(&doc) {
                        let msg = format!("index {}: {e}", doc.url);
                        tracing::warn!("{msg}");
                        debug.record_parser_error(msg);
                        continue;
                    }
                    debug.record_parsed();

                    if let Err(e) = indexer.maybe_commit(threshold) {
                        tracing::warn!("maybe_commit error: {e}");
                    }

                    if depth < max_depth {
                        crawler.enqueue_outlinks(&doc.outlinks, depth + 1);
                    }
                }
            });
        }

        // Time-based commit.
        {
            let indexer = Arc::clone(&self.indexer);
            let period  = Duration::from_secs(self.config.engine.commit_every_secs.max(1));
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(period).await;
                    if indexer.pending_count() > 0 {
                        if let Err(e) = indexer.commit() {
                            tracing::warn!("periodic commit error: {e}");
                        }
                    }
                }
            });
        }
    }

    pub fn build_server(&self) -> Result<Server> {
        Server::new(self.config.serving.clone(), Arc::clone(&self.state))
            .context("binding HTTP server")
    }
}