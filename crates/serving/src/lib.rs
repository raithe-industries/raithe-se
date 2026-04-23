// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/serving/src/lib.rs
//
// Axum HTTP server — search UI, JSON API, Prometheus metrics, health.
// Security: CORS allow-list, CSP/HSTS/X-* headers, Slowloris timeout,
// request body size limit.

mod templates;

use std::sync::Arc;
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
use tower_http::{
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};

use raithe_common::RawHit;
use raithe_config::ServingConfig;
use raithe_indexer::Indexer;
use raithe_instant::{InstantAnswer, InstantEngine};
use raithe_metrics::Metrics;
use raithe_query::QueryProcessor;
use raithe_ranker::Ranker;
use raithe_session::{SessionId, SessionStore};

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to bind '{addr}': {source}")]
    Bind {
        addr:   String,
        #[source]
        source: std::io::Error,
    },

    #[error("server error: {source}")]
    Serve {
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

// ── AppState ──────────────────────────────────────────────────────────────────

/// Shared state injected into every request handler.
pub struct AppState {
    pub config:    ServingConfig,
    pub metrics:   Arc<Metrics>,
    pub indexer:   Arc<Indexer>,
    pub processor: Arc<QueryProcessor>,
    pub ranker:    Arc<Ranker>,
    pub instant:   Arc<InstantEngine>,
    pub sessions:  Arc<SessionStore>,
}

// ── Wire types ────────────────────────────────────────────────────────────────

/// Query parameters for `GET /search`.
#[derive(Debug, Deserialize)]
struct SearchParams {
    q:          Option<String>,
    /// Optional session cookie value.
    #[allow(dead_code)]
    session_id: Option<String>,
}

/// A single search result returned in the JSON response.
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub title:   String,
    pub url:     String,
    pub snippet: String,
    pub score:   f32,
}

/// Full JSON response for `GET /search`.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub query:         String,
    pub results:       Vec<SearchResult>,
    pub instant:       Option<InstantAnswerResponse>,
    pub total_results: usize,
}

/// Serialisable form of an instant answer.
#[derive(Debug, Serialize)]
pub struct InstantAnswerResponse {
    pub kind:    String,
    pub display: String,
}

impl From<InstantAnswer> for InstantAnswerResponse {
    fn from(a: InstantAnswer) -> Self {
        let kind = format!("{:?}", a.kind);
        Self {
            kind,
            display: a.display,
        }
    }
}

// ── Server ────────────────────────────────────────────────────────────────────

/// Axum HTTP server.
pub struct Server {
    listener: TcpListener,
    router:   Router,
}

impl Server {
    /// Constructs a new `Server`, binding the TCP listener immediately so
    /// port conflicts are reported at init time, not at `run` time.
    pub fn new(config: ServingConfig, state: Arc<AppState>) -> Result<Self> {
        let addr     = config.bind.clone();
        let listener = std::net::TcpListener::bind(&addr)
            .and_then(|l| {
                l.set_nonblocking(true)?;
                Ok(l)
            })
            .map_err(|source| Error::Bind { addr: addr.clone(), source })?;
        let listener = TcpListener::from_std(listener)
            .map_err(|source| Error::Bind { addr: addr.clone(), source })?;

        let max_body = state.config.max_request_bytes;
        let origins  = state.config.allowed_origins.clone();

        let router = Router::new()
            .route("/",       get(handle_index))
            .route("/search", get(handle_search))
            .route("/health", get(handle_health))
            .route("/metrics",get(handle_metrics))
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

    /// Runs the server until the process is shut down.
    pub async fn run(self) -> Result<()> {
        axum::serve(self.listener, self.router)
            .await
            .map_err(|source| Error::Serve { source })
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn handle_health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn handle_metrics(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    match state.metrics.render() {
        Ok(text) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            text,
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn handle_index() -> impl IntoResponse {
    Html(templates::render_index())
}

async fn handle_search(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Query(params): Query<SearchParams>,
) -> Response {
    let raw = match params.q.as_deref().filter(|s| !s.is_empty()) {
        Some(q) => q.to_owned(),
        None    => {
            return Html(templates::render_index()).into_response();
        }
    };

    // Session identity (DEF-005). The client's existing identifier is read
    // from `X-Raithe-Session` (API clients) or the `raithe_sid` cookie
    // (browsers). When absent or malformed, a new id is minted and returned
    // via `Set-Cookie` so the next request can re-use it.
    let (session_id, mint_cookie) = resolve_session_id(&headers);

    if let Err(err) = state.sessions.record_query(&session_id, &raw) {
        tracing::warn!(%err, "session record_query failed");
    }

    // Query processing.
    let parsed = match state.processor.process(&raw) {
        Ok(p)  => p,
        Err(err) => {
            tracing::warn!(%err, "query processor failed");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    // Instant answer (non-fatal if absent).
    let instant = state.instant.resolve(&parsed);

    // Phase 1 — Tantivy BM25F search via the indexer (DEF-004).
    let hits = match state.indexer.search(&parsed) {
        Ok(h)  => h,
        Err(err) => {
            tracing::warn!(%err, "indexer search failed");
            Vec::<RawHit>::new()
        }
    };

    // Phase 2 + 3 — three-phase ranking pipeline.
    let pageranks = raithe_linkgraph::PageRankScores::new();
    let ranked = match state.ranker.rank(hits, &parsed, &pageranks) {
        Ok(r)  => r,
        Err(err) => {
            tracing::warn!(%err, "ranker failed");
            Vec::new()
        }
    };

    let results: Vec<SearchResult> = ranked
        .into_iter()
        .map(|h| SearchResult {
            title:   h.title,
            url:     h.url.to_string(),
            snippet: h.snippet,
            score:   h.score,
        })
        .collect();

    let instant_resp = instant.map(InstantAnswerResponse::from);
    let html         = templates::render_results(&raw, &results, instant_resp.as_ref());
    let mut response = Html(html).into_response();

    if mint_cookie {
        let cookie = format!(
            "raithe_sid={session_id}; Path=/; HttpOnly; SameSite=Lax; Max-Age={}",
            state.config.session_ttl_secs,
        );
        if let Ok(value) = HeaderValue::from_str(&cookie) {
            response.headers_mut().insert(header::SET_COOKIE, value);
        }
    }

    response
}

/// Returns the session id for the current request along with a flag
/// indicating whether the caller should emit a `Set-Cookie` for it.
///
/// Resolution order:
///   1. `X-Raithe-Session` header — API clients and tests.
///   2. `Cookie: raithe_sid=<uuid>` — browser sessions.
///   3. Freshly minted `SessionId` — caller must set cookie on response.
fn resolve_session_id(headers: &axum::http::HeaderMap) -> (SessionId, bool) {
    if let Some(raw) = headers.get("x-raithe-session").and_then(|v| v.to_str().ok()) {
        if let Ok(sid) = SessionId::parse_str(raw.trim()) {
            return (sid, false);
        }
    }

    if let Some(raw) = headers.get(header::COOKIE).and_then(|v| v.to_str().ok()) {
        for part in raw.split(';') {
            let part = part.trim();
            if let Some(value) = part.strip_prefix("raithe_sid=") {
                if let Ok(sid) = SessionId::parse_str(value.trim()) {
                    return (sid, false);
                }
            }
        }
    }

    (SessionId::new(), true)
}

// ── Security header middleware ────────────────────────────────────────────────

async fn security_headers(
    req:     Request<Body>,
    next:    Next,
    origins: Vec<String>,
) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();

    // Strict-Transport-Security.
    headers.insert(
        header::STRICT_TRANSPORT_SECURITY,
        HeaderValue::from_static("max-age=63072000; includeSubDomains; preload"),
    );

    // Content-Security-Policy.
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:",
        ),
    );

    // X-Content-Type-Options.
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );

    // X-Frame-Options.
    headers.insert(
        header::X_FRAME_OPTIONS,
        HeaderValue::from_static("DENY"),
    );

    // CORS — only insert when at least one origin is configured.
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
        let resp = SearchResponse {
            query:         String::from("test"),
            results:       Vec::new(),
            instant:       None,
            total_results: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("results"));
    }

    #[test]
    fn instant_answer_response_from_instant_answer() {
        use raithe_instant::AnswerKind;
        let ia = InstantAnswer {
            kind:    AnswerKind::Calculation,
            display: String::from("42"),
            input:   String::from("6 * 7"),
        };
        let resp = InstantAnswerResponse::from(ia);
        assert_eq!(resp.display, "42");
        assert!(resp.kind.contains("Calculation"));
    }

    #[test]
    fn session_from_x_raithe_session_header() {
        let existing = SessionId::new();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "x-raithe-session",
            HeaderValue::from_str(&existing.to_string()).unwrap(),
        );
        let (sid, mint) = resolve_session_id(&headers);
        assert_eq!(sid, existing);
        assert!(!mint, "existing header must not trigger Set-Cookie");
    }

    #[test]
    fn session_from_cookie() {
        let existing = SessionId::new();
        let cookie = format!("raithe_sid={existing}; other=value");
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::COOKIE, HeaderValue::from_str(&cookie).unwrap());
        let (sid, mint) = resolve_session_id(&headers);
        assert_eq!(sid, existing);
        assert!(!mint);
    }

    #[test]
    fn session_minted_when_headers_absent() {
        let headers = axum::http::HeaderMap::new();
        let (_sid, mint) = resolve_session_id(&headers);
        assert!(mint, "absent identifier must trigger Set-Cookie");
    }

    #[test]
    fn session_minted_when_cookie_malformed() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_static("raithe_sid=not-a-uuid"),
        );
        let (_sid, mint) = resolve_session_id(&headers);
        assert!(mint, "malformed cookie must be rejected and re-minted");
    }

    #[test]
    fn header_takes_precedence_over_cookie() {
        let header_sid = SessionId::new();
        let cookie_sid = SessionId::new();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "x-raithe-session",
            HeaderValue::from_str(&header_sid.to_string()).unwrap(),
        );
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!("raithe_sid={cookie_sid}")).unwrap(),
        );
        let (sid, mint) = resolve_session_id(&headers);
        assert_eq!(sid, header_sid);
        assert!(!mint);
    }
}
