// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/app/tests/end_to_end.rs
//
// Single end-to-end test: local site → crawler → parser → indexer (commit) →
// search. Catches: crawler-exit-on-empty, no-commit, parser failures, indexer
// snippet/body wiring, and serving's reliance on the indexer state.

use std::time::{Duration, Instant};

use axum::{response::Html, routing::get, Router};
use raithe_app::Phase1Engine;
use raithe_common::{ParsedQuery, Url};
use raithe_config::{
    Config, CrawlerConfig, EngineConfig, IndexerConfig, RankerConfig, ScraperConfig, ServingConfig,
};

async fn root_page()      -> Html<&'static str> { Html(r#"<html><body>links to <a href="/a">a</a> and <a href="/b">b</a></body></html>"#) }
async fn page_a()         -> Html<&'static str> { Html(r#"<html><body><h1>page a</h1><p>rust search engine</p></body></html>"#) }
async fn page_b()         -> Html<&'static str> { Html(r#"<html><body><h1>page b</h1><p>diabetes music story</p></body></html>"#) }
async fn robots()         -> &'static str       { "" } // empty = allow-all

async fn spawn_test_site() -> std::net::SocketAddr {
    let app = Router::new()
        .route("/", get(root_page))
        .route("/a", get(page_a))
        .route("/b", get(page_b))
        .route("/robots.txt", get(robots));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr     = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

fn test_config(serving_bind: String) -> Config {
    Config {
        crawler: CrawlerConfig {
            max_depth:        2,
            max_pages:        100,
            requests_per_sec: 50.0,
            min_seeds:        1,
            user_agent:       String::from("raithe-se-test/1.0"),
        },
        engine: EngineConfig {
            phase1_only:       true,
            commit_every_docs: 1, // commit on every add for fast assertion
            commit_every_secs: 1,
        },
        indexer: IndexerConfig::default(),
        neural:  Default::default(),
        ranker:  RankerConfig::default(),
        scraper: ScraperConfig::default(),
        serving: ServingConfig { bind: serving_bind, ..ServingConfig::default() },
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn end_to_end_crawl_index_search() {
    let site_addr = spawn_test_site().await;
    let seed      = Url::parse(&format!("http://{}/", site_addr)).unwrap();

    let tmp      = tempfile::tempdir().unwrap();
    let config   = test_config(String::from("127.0.0.1:0"));
    let engine   = Phase1Engine::build(config, vec![seed], tmp.path()).unwrap();
    engine.spawn_pipeline();

    // Wait for committed >= 3 (root + /a + /b).
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let committed = engine.indexer.committed_count();
        if committed >= 3 { break; }
        if Instant::now() > deadline {
            panic!(
                "timeout: committed={committed}, fetched={}, parsed={}, pending={}, frontier={}, errors={:?}",
                engine.crawler.fetched_count(),
                engine.debug.parsed.load(std::sync::atomic::Ordering::Relaxed),
                engine.indexer.pending_count(),
                engine.crawler.frontier_size(),
                engine.crawler.last_errors(),
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let hits = engine.indexer.search(&ParsedQuery::raw("rust search engine")).unwrap();
    assert!(!hits.is_empty(), "expected hits for 'rust search engine'");

    let urls: Vec<String> = hits.iter().map(|h| h.url.as_str().to_owned()).collect();
    assert!(
        urls.iter().any(|u| u.ends_with("/a")),
        "expected /a in results, got: {urls:?}"
    );

    // Snippet must be non-empty (validates body STORED fix).
    let a_hit = hits.iter().find(|h| h.url.as_str().ends_with("/a")).unwrap();
    assert!(!a_hit.snippet.is_empty(), "snippet for /a must be non-empty");
}