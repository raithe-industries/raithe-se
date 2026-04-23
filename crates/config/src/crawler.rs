// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/crawler.rs

/// Crawler politeness and frontier configuration.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct CrawlerConfig {
    /// Maximum link depth from a seed URL.
    pub max_depth: u32,
    /// Hard cap on pages fetched across the entire run.
    pub max_pages: u64,
    /// Per-host request rate (requests/second). Enforced via `governor`.
    pub requests_per_sec: f64,
    /// Minimum number of valid seed URLs required at startup.
    /// Startup fails with a clear error if the frontier contains fewer.
    pub min_seeds: usize,
    /// User-Agent header sent on every request.
    pub user_agent: String,
}

impl Default for CrawlerConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_pages: 1_000_000,
            requests_per_sec: 2.0,
            min_seeds: 100,
            user_agent: String::from("raithe-se/1.0"),
        }
    }
}
