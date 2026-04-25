// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/serving/src/lib.rs

mod templates;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, HeaderValue, Request, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::net::TcpListener;
use tower_http::{limit::RequestBodyLimitLayer, timeout::TimeoutLayer, trace::TraceLayer};

use raithe_common::RawHit;
use raithe_config::ServingConfig;
use raithe_crawler::Crawler;
use raithe_indexer::Indexer;
use raithe_instant::{InstantAnswer, InstantEngine};
use raithe_metrics::Metrics;
use raithe_query::QueryProcessor;
use raithe_ranker::Ranker;
use raithe_session::{SessionId, SessionStore};

const PARSER_ERRORS_CAP: usize = 20;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to bind '{addr}': {source}")]
    Bind { addr: String, #[source] source: std::io::Error },
    #[error("server error: {source}")]
    Serve { #[source] source: std::io::Error },
}

pub type Result<T> = std::result::Result<T, Error>;

// ── Debug stats ──────────────────────────────────────────────────────────────

/// Counters owned by the indexing-pipeline closure. Crawler-side stats live on
/// the `Crawler` itself; this struct only tracks what the pipeline sees.
pub struct DebugStats {
    pub parsed:        AtomicU64,
    pub parser_errors: Mutex<VecDeque<String>>,
}

impl DebugStats {
    pub fn new() -> Self {
        Self {
            parsed:        AtomicU64::new(0),
            parser_errors: Mutex::new(VecDeque::with_capacity(PARSER_ERRORS_CAP)),
        }
    }

    pub fn record_parsed(&self) { self.parsed.fetch_add(1, Ordering::Relaxed); }

    pub fn record_parser_error(&self, msg: String) {
        let mut q = self.parser_errors.lock().unwrap_or_else(|p| p.into_inner());
        if q.len() == PARSER_ERRORS_CAP { q.pop_front(); }
        q.push_back(msg);
    }
}

impl Default for DebugStats {
    fn default() -> Self { Self::new() }
}

// ── AppState ────────────────────────────────────────────────────────────────

pub struct AppState {
    pub config:    ServingConfig,
    pub metrics:   Arc<Metrics>,
    pub indexer:   Arc<Indexer>,
    pub processor: Arc<QueryProcessor>,
    pub ranker:    Arc<Ranker>,
    pub instant:   Arc<InstantEngine>,
    pub sessions:  Arc<SessionStore>,
    pub crawler:   Arc<Crawler>,
    pub debug:     Arc<DebugStats>,
}

#[derive(Debug, Deserialize)]
struct SearchParams {
    q: Option<String>,
    #[allow(dead_code)]
    session_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub title:   String,
    pub url:     String,
    pub snippet: String,
    pub score:   f32,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub query:         String,
    pub results:       Vec<SearchResult>,
    pub instant:       Option<InstantAnswerResponse>,
    pub total_results: usize,
}

#[derive(Debug, Serialize)]
pub struct InstantAnswerResponse {
    pub kind:    String,
    pub display: String,
}

impl From<InstantAnswer> for InstantAnswerResponse {
    fn from(a: InstantAnswer) -> Self {
        Self { kind: format!("{:?}", a.kind), display: a.display }
    }
}

#[derive(Debug, Serialize)]
struct DebugStatsResponse {
    fetched:        u64,
    parsed:         u64,
    indexed_pending:  u64,
    committed:        u64,
    frontier_queue: u64,
    seen_urls:      usize,
    last_errors:    Vec<String>,
}

// ── Server ──────────────────────────────────────────────────────────────────

pub struct Server {
    listener: TcpListener,
    router:   Router,
}

impl Server {
    pub fn new(config: ServingConfig, state: Arc<AppState>) -> Result<Self> {
        let addr     = config.bind.clone();
        let listener = std::net::TcpListener::bind(&addr)
            .and_then(|l| { l.set_nonblocking(true)?; Ok(l) })
            .map_err(|source| Error::Bind { addr: addr.clone(), source })?;
        let listener = TcpListener::from_std(listener)
            .map_err(|source| Error::Bind { addr: addr.clone(), source })?;

        let max_body = state.config.max_request_bytes;
        let origins  = state.config.allowed_origins.clone();

        let router = Router::new()
            .route("/", get(handle_index))
            .route("/search", get(handle_search))
            .route("/health", get(handle_health))
            .route("/metrics", get(handle_metrics))
            .route("/debug/stats", get(handle_debug_stats))
            .with_state(state)
            .layer(middleware::from_fn(move |req, next| {
                let origins = origins.clone();
                security_headers(req, next, origins)
            }))
            .layer(TimeoutLayer::new(Duration::from_secs(30)))
            .layer(TraceLayer::new_for_http())
            .layer(RequestBodyLimitLayer::new(max_body));

        Ok(Self { listener, router })
    }

    pub async fn run(self) -> Result<()> {
        axum::serve(self.listener, self.router)
            .await
            .map_err(|source| Error::Serve { source })
    }

    /// Returns the actual bound address. Useful for tests using port 0.
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }
}

// ── Handlers ────────────────────────────────────────────────────────────────

async fn handle_health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn handle_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.metrics.render() {
        Ok(text) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            text,
        ).into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn handle_index() -> impl IntoResponse {
    Html(templates::render_index())
}

async fn handle_debug_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut last_errors = state.crawler.last_errors();
    if let Ok(parser_errs) = state.debug.parser_errors.lock() {
        last_errors.extend(parser_errs.iter().cloned());
    }
    // Keep at most 20 (most-recent at the back).
    let trimmed = last_errors.split_off(last_errors.len().saturating_sub(20));

    let body = DebugStatsResponse {
        fetched:         state.crawler.fetched_count(),
        parsed:          state.debug.parsed.load(Ordering::Relaxed),
        indexed_pending: state.indexer.pending_count(),
        committed:       state.indexer.committed_count(),
        frontier_queue:  state.crawler.frontier_size(),
        seen_urls:       state.crawler.seen_count(),
        last_errors:     trimmed,
    };
    Json(body)
}

async fn handle_search(
    State(state):  State<Arc<AppState>>,
    headers:       axum::http::HeaderMap,
    Query(params): Query<SearchParams>,
) -> Response {
    let raw = match params.q.as_deref().filter(|s| !s.is_empty()) {
        Some(q) => q.to_owned(),
        None    => return Html(templates::render_index()).into_response(),
    };

    let (session_id, mint_cookie) = resolve_session_id(&headers);
    if let Err(err) = state.sessions.record_query(&session_id, &raw) {
        tracing::warn!(%err, "session record_query failed");
    }

    let parsed = match state.processor.process(&raw) {
        Ok(p)  => p,
        Err(e) => {
            tracing::warn!(%e, "query processor failed");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let instant = state.instant.resolve(&parsed);

    let hits = match state.indexer.search(&parsed) {
        Ok(h)  => h,
        Err(e) => { tracing::warn!(%e, "indexer search failed"); Vec::<RawHit>::new() }
    };

    let pageranks = raithe_linkgraph::PageRankScores::new();
    let ranked = match state.ranker.rank(hits, &parsed, &pageranks) {
        Ok(r)  => r,
        Err(e) => {
            tracing::warn!(%e, "ranker failed");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let total_results = ranked.len();
    let results: Vec<SearchResult> = ranked.into_iter().map(|r| SearchResult {
        title:   r.title,
        url:     r.url.as_str().to_owned(),
        snippet: r.snippet,
        score:   r.score,
    }).collect();

    let body = SearchResponse {
        query:   parsed.original,
        results,
        instant: instant.map(InstantAnswerResponse::from),
        total_results,
    };

    let mut response = Json(body).into_response();
    if mint_cookie {
        if let Ok(v) = HeaderValue::from_str(&format!(
            "raithe_sid={session_id}; HttpOnly; SameSite=Lax; Path=/; Max-Age={}",
            state.config.session_ttl_secs,
        )) {
            response.headers_mut().insert(header::SET_COOKIE, v);
        }
    }
    response
}

// ── Session id resolution ───────────────────────────────────────────────────

fn resolve_session_id(headers: &axum::http::HeaderMap) -> (SessionId, bool) {
    if let Some(v) = headers.get("x-raithe-session").and_then(|h| h.to_str().ok()) {
        if let Ok(id) = SessionId::parse_str(v.trim()) { return (id, false); }
    }
    if let Some(cookie) = headers.get(header::COOKIE).and_then(|h| h.to_str().ok()) {
        for kv in cookie.split(';') {
            let kv = kv.trim();
            if let Some(rest) = kv.strip_prefix("raithe_sid=") {
                if let Ok(id) = SessionId::parse_str(rest.trim()) { return (id, false); }
            }
        }
    }
    (SessionId::new(), true)
}

// ── Security middleware ─────────────────────────────────────────────────────

async fn security_headers(req: Request<Body>, next: Next, origins: Vec<String>) -> Response {
    let mut response = next.run(req).await;
    let headers      = response.headers_mut();

    headers.insert(header::STRICT_TRANSPORT_SECURITY,
        HeaderValue::from_static("max-age=63072000; includeSubDomains; preload"));
    headers.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:"));
    headers.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));

    if !origins.is_empty() {
        let value = origins.join(", ");
        if let Ok(v) = HeaderValue::from_str(&value) {
            headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_response_serialises() {
        let resp = SearchResponse { query: String::from("test"), results: Vec::new(), instant: None, total_results: 0 };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("test"));
    }

    #[test]
    fn debug_stats_serialises() {
        let body = DebugStatsResponse {
            fetched: 10, parsed: 9, indexed_pending: 2, committed: 7,
            frontier_queue: 3, seen_urls: 12, last_errors: vec!["x".into()],
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"committed\":7"));
        assert!(json.contains("\"last_errors\""));
    }
}