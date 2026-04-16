// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/serving.rs

/// Axum HTTP server configuration.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ServingConfig {
    /// Socket address for the search API and UI (e.g. `"0.0.0.0:8080"`).
    pub bind: String,
    /// Socket address for the Prometheus metrics endpoint.
    pub metrics_bind: String,
    /// Maximum accepted request body size in bytes.
    /// Requests exceeding this limit are rejected with 413.
    pub max_request_bytes: usize,
    /// CORS allowed origins. Must be explicitly set in production — no
    /// wildcard is ever permitted. Empty list disables CORS headers entirely.
    pub allowed_origins: Vec<String>,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            bind:              String::from("0.0.0.0:8080"),
            metrics_bind:      String::from("0.0.0.0:9090"),
            max_request_bytes: 8192,
            allowed_origins:   Vec::new(),
        }
    }
}
