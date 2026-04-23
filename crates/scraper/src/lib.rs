// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/scraper/src/lib.rs
//
// Single-URL HTTP fetch — connection, redirect following, content-type
// gating, body extraction, and response metadata capture.
// Produces a FetchResult consumed by the parser crate.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use raithe_common::{Timestamp, Url};
use raithe_config::ScraperConfig;
use raithe_metrics::Metrics;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("HTTP client error fetching '{url}': {source}")]
    Http {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("content-type '{content_type}' is not accepted for '{url}'")]
    ContentTypeRejected { url: String, content_type: String },
    #[error("response body for '{url}' exceeded the size limit of {limit} bytes")]
    BodyTooLarge { url: String, limit: usize },
    #[error("failed to build HTTP client: {source}")]
    ClientBuild {
        #[source]
        source: reqwest::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// The outcome of a single successful HTTP fetch.
#[derive(Debug)]
pub struct FetchResult {
    /// The final URL after any redirects.
    pub url:        Url,
    /// HTTP response status code.
    pub status:     u16,
    /// Response headers as lowercase name → value pairs.
    pub headers:    HashMap<String, String>,
    /// Raw response body bytes (decompressed by reqwest).
    pub body_bytes: Vec<u8>,
    /// Wall-clock time at which the fetch completed.
    pub fetched_at: Timestamp,
}

/// Single-URL HTTP fetcher.
///
/// Built once and shared across crawler tasks. The underlying `reqwest::Client`
/// is cheaply cloneable and manages a connection pool internally.
pub struct Scraper {
    client:         reqwest::Client,
    max_body_bytes: usize,
    metrics:        Arc<Metrics>,
}

impl Scraper {
    /// Constructs a new `Scraper` from the given config and metric handles.
    ///
    /// Returns `Error::ClientBuild` if the HTTP client cannot be initialised.
    pub fn new(config: &ScraperConfig, metrics: Arc<Metrics>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(&config.user_agent)
            .timeout(Duration::from_secs(config.timeout_secs))
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects))
            .build()
            .map_err(|source| Error::ClientBuild { source })?;

        let max_body_bytes = config.max_body_bytes;

        Ok(Self {
            client,
            max_body_bytes,
            metrics,
        })
    }

    /// Fetches the given `url` and returns a `FetchResult`.
    ///
    /// Rejects responses whose `Content-Type` does not indicate HTML or plain
    /// text. Rejects bodies that exceed `max_body_bytes`. Records the HTTP
    /// status class in the `raithe_pages_crawled_total` counter.
    pub async fn fetch(&self, url: &Url) -> Result<FetchResult> {
        let response = self
            .client
            .get(url.as_str())
            .send()
            .await
            .map_err(|source| Error::Http {
                url: url.to_string(),
                source,
            })?;

        let status = response.status().as_u16();
        let status_class = status_class(status);

        self.metrics
            .pages_crawled_total
            .with_label_values(&[status_class])
            .inc();

        // Capture the final URL after redirects before consuming the response.
        let final_url = Url::parse(response.url().as_str()).unwrap_or_else(|_| url.clone());

        let headers = collect_headers(response.headers());

        // Gate on content-type before buffering the body.
        if !is_accepted_content_type(&headers) {
            let content_type = headers
                .get("content-type")
                .cloned()
                .unwrap_or_default();
            return Err(Error::ContentTypeRejected {
                url: url.to_string(),
                content_type,
            });
        }

        let raw_bytes = response.bytes().await.map_err(|source| Error::Http {
            url: url.to_string(),
            source,
        })?;

        if raw_bytes.len() > self.max_body_bytes {
            return Err(Error::BodyTooLarge {
                url:   url.to_string(),
                limit: self.max_body_bytes,
            });
        }

        let body_bytes = raw_bytes.to_vec();
        let fetched_at = Timestamp::from_millis(unix_ms_now());

        Ok(FetchResult {
            url: final_url,
            status,
            headers,
            body_bytes,
            fetched_at,
        })
    }
}

/// Returns the current Unix time in milliseconds.
fn unix_ms_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Returns the Prometheus status-class label for the given HTTP status code.
fn status_class(status: u16) -> &'static str {
    match status {
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _         => "err",
    }
}

/// Collects response headers into a lowercase name → value map.
fn collect_headers(headers: &reqwest::header::HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let value = value.to_str().ok()?;
            Some((name.as_str().to_ascii_lowercase(), value.to_owned()))
        })
        .collect()
}

/// Returns `true` when the `Content-Type` header indicates a type the parser
/// can handle (HTML variants and plain text).
///
/// Absent `Content-Type` is accepted optimistically — the parser will reject
/// unparseable bytes if needed.
fn is_accepted_content_type(headers: &HashMap<String, String>) -> bool {
    let content_type = match headers.get("content-type") {
        Some(ct) => ct.as_str(),
        None => return true,
    };
    content_type.contains("text/html")
        || content_type.contains("application/xhtml+xml")
        || content_type.contains("text/plain")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_class_labels() {
        assert_eq!(status_class(200), "2xx");
        assert_eq!(status_class(204), "2xx");
        assert_eq!(status_class(301), "3xx");
        assert_eq!(status_class(404), "4xx");
        assert_eq!(status_class(503), "5xx");
        assert_eq!(status_class(100), "err");
    }

    #[test]
    fn accepted_content_types() {
        let mut headers = HashMap::new();
        headers.insert(
            "content-type".to_owned(),
            "text/html; charset=utf-8".to_owned(),
        );
        assert!(is_accepted_content_type(&headers));

        headers.insert("content-type".to_owned(), "application/json".to_owned());
        assert!(!is_accepted_content_type(&headers));

        headers.insert(
            "content-type".to_owned(),
            "application/xhtml+xml".to_owned(),
        );
        assert!(is_accepted_content_type(&headers));

        let empty: HashMap<String, String> = HashMap::new();
        assert!(is_accepted_content_type(&empty));
    }

    #[test]
    fn collect_headers_lowercases_names() {
        let mut map = reqwest::header::HeaderMap::new();
        map.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("text/html"),
        );
        let collected = collect_headers(&map);
        assert_eq!(
            collected.get("content-type").map(String::as_str),
            Some("text/html"),
        );
    }
}
