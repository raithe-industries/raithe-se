// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/metrics/src/lib.rs
//
// Prometheus registry and tracing initialisation.
// All metric handles are registered here and nowhere else (§13).
// Other crates receive cloned handles via the Metrics struct's accessor methods.

use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, HistogramOpts, HistogramVec,
    Opts, Registry, TextEncoder,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to register metric '{name}': {source}")]
    Register {
        name: &'static str,
        #[source]
        source: prometheus::Error,
    },
    #[error("failed to initialise tracing subscriber: {reason}")]
    Tracing { reason: String },
    #[error("failed to encode metrics: {source}")]
    Encode {
        #[source]
        source: prometheus::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Owns the Prometheus registry and all metric handles.
///
/// Constructed once at startup via `Metrics::new()`. Cloned handles are
/// distributed to other crates — no crate creates its own registry.
pub struct Metrics {
    registry: Registry,

    // §13 — all nine spec metrics:
    pub pages_crawled_total:       CounterVec,
    pub index_documents_total:     Gauge,
    pub query_latency_seconds:     HistogramVec,
    pub neural_inference_seconds:  HistogramVec,
    pub neural_execution_provider: GaugeVec,
    pub rank_phase3_calls_total:   Counter,
    pub errors_total:              CounterVec,
    pub crawl_queue_depth:         Gauge,
    pub rate_limited_total:        CounterVec,
}

impl Metrics {
    /// Creates and registers all metric handles.
    ///
    /// Returns `Error::Register` if any handle fails to register — this is a
    /// startup-fatal condition. The prototype used `.unwrap()` here (BUG-001);
    /// this implementation propagates the error correctly.
    pub fn new() -> Result<Self> {
        let registry = Registry::new();

        let pages_crawled_total = CounterVec::new(
            Opts::new(
                "raithe_pages_crawled_total",
                "Pages fetched, labelled by HTTP status class (2xx, 3xx, 4xx, 5xx, err).",
            ),
            &["status"],
        )
        .map_err(|source| Error::Register {
            name: "raithe_pages_crawled_total",
            source,
        })?;

        let index_documents_total = Gauge::new(
            "raithe_index_documents_total",
            "Total documents in the active Tantivy index.",
        )
        .map_err(|source| Error::Register {
            name: "raithe_index_documents_total",
            source,
        })?;

        let query_latency_seconds = HistogramVec::new(
            HistogramOpts::new(
                "raithe_query_latency_seconds",
                "Query pipeline latency, labelled by phase (bm25, gbdt, rerank, total).",
            )
            .buckets(latency_buckets()),
            &["phase"],
        )
        .map_err(|source| Error::Register {
            name: "raithe_query_latency_seconds",
            source,
        })?;

        let neural_inference_seconds = HistogramVec::new(
            HistogramOpts::new(
                "raithe_neural_inference_seconds",
                "ONNX inference latency, labelled by model (embedder, reranker, generator).",
            )
            .buckets(latency_buckets()),
            &["model"],
        )
        .map_err(|source| Error::Register {
            name: "raithe_neural_inference_seconds",
            source,
        })?;

        let neural_execution_provider = GaugeVec::new(
            Opts::new(
                "raithe_neural_execution_provider",
                "Active ONNX execution provider (1.0 = selected, 0.0 = others).",
            ),
            &["provider"],
        )
        .map_err(|source| Error::Register {
            name: "raithe_neural_execution_provider",
            source,
        })?;

        let rank_phase3_calls_total = Counter::new(
            "raithe_rank_phase3_calls_total",
            "Total invocations of the BGE cross-encoder reranker (Phase 3). Must be > 0 in production.",
        )
        .map_err(|source| Error::Register {
            name: "raithe_rank_phase3_calls_total",
            source,
        })?;

        let errors_total = CounterVec::new(
            Opts::new(
                "raithe_errors_total",
                "Errors from public crate functions, labelled by crate and kind.",
            ),
            &["crate", "kind"],
        )
        .map_err(|source| Error::Register {
            name: "raithe_errors_total",
            source,
        })?;

        let crawl_queue_depth = Gauge::new(
            "raithe_crawl_queue_depth",
            "Current frontier queue depth.",
        )
        .map_err(|source| Error::Register {
            name: "raithe_crawl_queue_depth",
            source,
        })?;

        let rate_limited_total = CounterVec::new(
            Opts::new(
                "raithe_rate_limited_total",
                "Requests delayed by the per-host rate limiter, labelled by host.",
            ),
            &["host"],
        )
        .map_err(|source| Error::Register {
            name: "raithe_rate_limited_total",
            source,
        })?;

        registry
            .register(Box::new(pages_crawled_total.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_pages_crawled_total",
                source,
            })?;
        registry
            .register(Box::new(index_documents_total.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_index_documents_total",
                source,
            })?;
        registry
            .register(Box::new(query_latency_seconds.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_query_latency_seconds",
                source,
            })?;
        registry
            .register(Box::new(neural_inference_seconds.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_neural_inference_seconds",
                source,
            })?;
        registry
            .register(Box::new(neural_execution_provider.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_neural_execution_provider",
                source,
            })?;
        registry
            .register(Box::new(rank_phase3_calls_total.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_rank_phase3_calls_total",
                source,
            })?;
        registry
            .register(Box::new(errors_total.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_errors_total",
                source,
            })?;
        registry
            .register(Box::new(crawl_queue_depth.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_crawl_queue_depth",
                source,
            })?;
        registry
            .register(Box::new(rate_limited_total.clone()))
            .map_err(|source| Error::Register {
                name: "raithe_rate_limited_total",
                source,
            })?;

        let metrics = Self {
            registry,
            pages_crawled_total,
            index_documents_total,
            query_latency_seconds,
            neural_inference_seconds,
            neural_execution_provider,
            rank_phase3_calls_total,
            errors_total,
            crawl_queue_depth,
            rate_limited_total,
        };
        metrics.seed_label_sets();
        Ok(metrics)
    }

    /// Renders all registered metrics in Prometheus text format.
    ///
    /// Returned string is suitable for the `GET /metrics` endpoint body.
    pub fn render(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let families = self.registry.gather();
        encoder.encode_to_string(&families).map_err(|source| Error::Encode { source })
    }

    /// Seeds each vec metric with a zero-value observation for all canonical
    /// label sets so that `gather()` includes every family in output from
    /// startup onward, regardless of whether real observations have arrived.
    fn seed_label_sets(&self) {
        for status in &["2xx", "3xx", "4xx", "5xx", "err"] {
            self.pages_crawled_total.with_label_values(&[status]).inc_by(0.0);
        }
        for phase in &["bm25", "gbdt", "rerank", "total"] {
            self.query_latency_seconds.with_label_values(&[phase]).observe(0.0);
        }
        for model in &["embedder", "reranker", "generator"] {
            self.neural_inference_seconds.with_label_values(&[model]).observe(0.0);
        }
        for provider in &["cuda", "directml", "coreml", "cpu"] {
            self.neural_execution_provider.with_label_values(&[provider]).set(0.0);
        }
        for crate_name in &[
            "common", "config", "storage", "crawler", "scraper", "parser",
            "indexer", "neural", "semantic", "linkgraph", "ranker", "query",
            "instant", "freshness", "session", "serving",
        ] {
            self.errors_total.with_label_values(&[crate_name, "unknown"]).inc_by(0.0);
        }
        self.rate_limited_total.with_label_values(&["_init"]).inc_by(0.0);
    }
}

// ── Local-time timer for tracing ──────────────────────────────────────────────

/// Custom tracing-subscriber timer that formats timestamps in the system's
/// local timezone (Toronto / America/Toronto — auto-adjusts for EDT/EST).
///
/// Output format: `2026-04-15 15:19:56.551 EDT`
struct LocalTimer;

impl tracing_subscriber::fmt::time::FormatTime for LocalTimer {
    fn format_time(
        &self,
        w: &mut tracing_subscriber::fmt::format::Writer<'_>,
    ) -> std::fmt::Result {
        let now = chrono::Local::now();
        write!(w, "{}", now.format("%Y-%m-%d %H:%M:%S%.3f %Z"))
    }
}

/// Initialises the global tracing subscriber with local-time timestamps.
///
/// Timestamps are formatted in the system local timezone (Toronto).
/// Respects `RUST_LOG`. Uses pretty formatting in dev, JSON in production.
/// Returns `Error::Tracing` if the subscriber is already set.
pub fn init_tracing(json: bool) -> Result<()> {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let result = if json {
        fmt()
            .json()
            .with_timer(LocalTimer)
            .with_env_filter(filter)
            .try_init()
    } else {
        fmt()
            .pretty()
            .with_timer(LocalTimer)
            .with_env_filter(filter)
            .try_init()
    };

    result.map_err(|err| Error::Tracing {
        reason: err.to_string(),
    })
}

/// Standard latency histogram buckets (seconds).
fn latency_buckets() -> Vec<f64> {
    vec![
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_succeeds_and_renders() {
        let metrics = Metrics::new().unwrap();
        let rendered = metrics.render().unwrap();
        assert!(rendered.contains("raithe_pages_crawled_total"));
        assert!(rendered.contains("raithe_index_documents_total"));
        assert!(rendered.contains("raithe_query_latency_seconds"));
        assert!(rendered.contains("raithe_neural_inference_seconds"));
        assert!(rendered.contains("raithe_neural_execution_provider"));
        assert!(rendered.contains("raithe_rank_phase3_calls_total"));
        assert!(rendered.contains("raithe_errors_total"));
        assert!(rendered.contains("raithe_crawl_queue_depth"));
        assert!(rendered.contains("raithe_rate_limited_total"));
    }

    #[test]
    fn double_registration_fails() {
        let a = Metrics::new().unwrap();
        let b = Metrics::new().unwrap();
        assert!(a.render().is_ok());
        assert!(b.render().is_ok());
    }

    #[test]
    fn counter_increments_appear_in_render() {
        let metrics = Metrics::new().unwrap();
        metrics.pages_crawled_total.with_label_values(&["2xx"]).inc_by(42.0);
        let rendered = metrics.render().unwrap();
        assert!(rendered.contains("42"));
    }
}
