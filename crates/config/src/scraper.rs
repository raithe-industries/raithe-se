// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/scraper.rs

/// Per-request HTTP fetch configuration used by the scraper crate.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ScraperConfig {
    /// User-Agent header sent on every request.
    pub user_agent: String,
    /// Per-request timeout in seconds (covers the entire response including body).
    pub timeout_secs: u64,
    /// TCP connection timeout in seconds.
    pub connect_timeout_secs: u64,
    /// Maximum number of redirects to follow before returning an error.
    pub max_redirects: usize,
    /// Maximum accepted response body size in bytes. Responses exceeding
    /// this limit are rejected with `Error::BodyTooLarge`.
    pub max_body_bytes: usize,
}

impl Default for ScraperConfig {
    fn default() -> Self {
        Self {
            user_agent: String::from("raithe-se/1.0"),
            timeout_secs: 30,
            connect_timeout_secs: 10,
            max_redirects: 10,
            max_body_bytes: 10 * 1024 * 1024, // 10 MiB
        }
    }
}
