// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/app/tests/persistence.rs
//
// End-to-end persistence test — your point 4.
//
// Mini 3-page HTTP site → crawl → search → restart → search again.
// Asserts:
//   1. crawl + search work end-to-end
//   2. doc count is stable across restart (no duplicates from re-fetch)
//   3. unique-word query returns the same single hit before and after restart
//   4. URL identity is durable (same URL → same DocumentId)

use std::sync::Arc;
use std::time::Duration;

use axum::{routing::get, Router};
use raithe_common::Url;
use raithe_config::Config;
use tokio::net::TcpListener;

mod common;
use common::*;

mod common {
    use super::*;
    use std::path::Path;

    /// Spawns a 3-page mini-site on an ephemeral port.
    /// Returns the bound base URL (e.g. "http://127.0.0.1:43521").
    pub async fn spawn_site() -> String {
        let app = Router::new()
            .route("/",  get(|| async {
                axum::response::Html(r#"<!DOCTYPE html><html><body>
                    <h1>Root</h1>
                    <p>Welcome to the persistence test. Unique anchor: zorgblat.</p>
                    <a href="/a">A</a> <a href="/b">B</a>
                </body></html>"#)
            }))
            .route("/a", get(|| async {
                axum::response::Html(r#"<!DOCTYPE html><html><body>
                    <h1>Page A</h1>
                    <p>Page A content. Unique anchor: quibbleflux.</p>
                </body></html>"#)
            }))
            .route("/b", get(|| async {
                axum::response::Html(r#"<!DOCTYPE html><html><body>
                    <h1>Page B</h1>
                    <p>Page B content. Unique anchor: pernicketybog.</p>
                </body></html>"#)
            }));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr     = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        format!("http://{addr}")
    }

    /// Builds a Config tuned for a tiny local crawl.
    pub fn dev_config(base: &str) -> Config {
        let mut c = Config::default();
        c.engine.phase1_only       = true;
        c.engine.commit_every_docs = 1;
        c.engine.commit_every_secs = 1;
        c.crawler.min_seeds        = 1;
        c.crawler.max_pages        = 10;
        c.crawler.max_depth        = 2;
        c.crawler.requests_per_sec = 50.0;
        c.serving.bind             = "127.0.0.1:0".to_owned();
        c.scraper.user_agent       = format!("raithe-test (+{base})");
        c
    }

    pub fn seeds(base: &str) -> Vec<Url> {
        vec![Url::parse(&format!("{base}/")).unwrap()]
    }

    /// Polls until `predicate` is true or `timeout` elapses. Panics on timeout.
    pub async fn wait_for<F>(timeout: Duration, label: &str, mut predicate: F)
    where F: FnMut() -> bool,
    {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if predicate() { return; }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("timeout waiting for: {label}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn persistence_no_duplicates_across_restart() {
    let base     = spawn_site().await;
    let data_dir = tempfile::tempdir().unwrap();

    // ── First run — cold start, full crawl. ────────────────────────────────
    let registry_id_root_first = {
        let engine = raithe_app::Phase1Engine::build(
            dev_config(&base),
            seeds(&base),
            data_dir.path(),
        ).unwrap();
        engine.spawn_pipeline();

        // Wait for all 3 pages to be committed.
        let indexer = Arc::clone(&engine.indexer);
        wait_for(Duration::from_secs(15), "committed >= 3", || {
            indexer.committed_count() >= 3
        }).await;

        let committed_first = indexer.committed_count();
        assert!(committed_first >= 3, "expected at least 3 committed, got {committed_first}");

        // Unique-word query returns exactly one hit per page.
        let parsed = raithe_common::ParsedQuery::raw("zorgblat");
        let hits   = indexer.search(&parsed).unwrap();
        assert_eq!(hits.len(), 1, "zorgblat must hit exactly the root page");
        assert!(hits[0].url.as_str().ends_with("/"), "got url: {}", hits[0].url);

        // Capture the registry id of root for the post-restart identity check.
        let root_url = Url::parse(&format!("{base}/")).unwrap();
        engine.registry.lookup(&root_url).unwrap()
            .expect("root must have a registry id after first crawl")
    };

    // First engine drops here — tasks abort, indexer flushes via Drop, registry
    // closes its sqlite connection. tokio runtime continues for the second run.

    // ── Second run — warm start, same data_dir. ────────────────────────────
    let engine = raithe_app::Phase1Engine::build(
        dev_config(&base),
        seeds(&base),
        data_dir.path(),
    ).unwrap();
    engine.spawn_pipeline();

    // Wait for the second crawl to finish re-fetching the same 3 pages.
    let indexer = Arc::clone(&engine.indexer);
    wait_for(Duration::from_secs(15), "second crawl quiesces", || {
        // After re-fetch, committed should *still be 3* — upsert replaces in
        // place. If duplicates were leaking, we'd see 6 here.
        indexer.committed_count() >= 3
    }).await;

    // Give the second crawl a moment to fully settle. Crawler may still be
    // re-fetching pages in the background; commit them before we assert.
    tokio::time::sleep(Duration::from_secs(3)).await;
    let _ = engine.indexer.commit();

    let committed_second = engine.indexer.committed_count();
    assert_eq!(
        committed_second, 3,
        "URL registry must prevent duplicates on restart — got {committed_second}, expected 3",
    );

    // The unique-word query must still return exactly one hit for the same URL.
    let hits = engine.indexer.search(&raithe_common::ParsedQuery::raw("zorgblat")).unwrap();
    assert_eq!(hits.len(), 1, "duplicates leaked: {hits:?}");

    // URL identity is durable: same URL → same DocumentId across runs.
    let root_url             = Url::parse(&format!("{base}/")).unwrap();
    let registry_id_root_now = engine.registry.lookup(&root_url).unwrap()
        .expect("root must still have a registry id after restart");
    assert_eq!(
        registry_id_root_now, registry_id_root_first,
        "registry must return the same DocumentId across restarts",
    );
}