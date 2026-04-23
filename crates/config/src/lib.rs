// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/lib.rs
//
// Config loading, validation, and hot-reload.
// All sub-structs are public and derive serde::Deserialize + Default.

use std::path::Path;

use thiserror::Error;

pub mod crawler;
pub mod indexer;
pub mod neural;
pub mod ranker;
pub mod scraper;
pub mod serving;
pub mod watcher;

pub use self::crawler::CrawlerConfig;
pub use self::indexer::IndexerConfig;
pub use self::neural::NeuralConfig;
pub use self::ranker::RankerConfig;
pub use self::scraper::ScraperConfig;
pub use self::serving::ServingConfig;
pub use self::watcher::watch;

#[derive(Debug, Error)]
pub enum Error {
    #[error("cannot load config from {path}: {source}")]
    Load {
        path: String,
        #[source]
        source: Box<figment::Error>,
    },
    #[error("cannot watch config file: {source}")]
    Watch {
        #[source]
        source: notify::Error,
    },
    #[error("invalid config: {reason}")]
    Invalid { reason: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Full engine configuration. Loaded once at startup; hot-reloaded via watcher.
///
/// Load priority (highest wins):
///   1. Compiled-in defaults (`Default` impls)
///   2. `data/config/engine.toml`
///   3. Environment variables (`RAITHE__SECTION__KEY`)
///   4. Path passed to `Config::load`
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub crawler: CrawlerConfig,
    pub indexer: IndexerConfig,
    pub neural: NeuralConfig,
    pub ranker: RankerConfig,
    pub scraper: ScraperConfig,
    pub serving: ServingConfig,
}

impl Config {
    /// Loads and validates configuration from the file at `path`.
    ///
    /// Applies environment-variable overrides (`RAITHE__SECTION__KEY`) on top
    /// of the file values. Missing keys fall back to compiled-in defaults.
    /// Returns `Error::Load` on any parse or I/O failure.
    pub fn load(path: &Path) -> Result<Self> {
        use figment::providers::{Env, Format, Serialized, Toml};
        use figment::Figment;

        let path_str = path.display().to_string();
        let config = Figment::from(Serialized::defaults(Self::default()))
            .merge(Toml::file(path))
            .merge(Env::prefixed("RAITHE__").split("__"))
            .extract()
            .map_err(|source| Error::Load {
                path: path_str,
                source: Box::new(source),
            })?;
        Ok(config)
    }

    /// Validates invariants that cannot be expressed in types alone.
    ///
    /// Called by `Config::load` and by the hot-reload watcher before
    /// replacing the running config with a newly loaded one.
    pub fn validate(&self) -> Result<()> {
        if self.crawler.min_seeds == 0 {
            return Err(Error::Invalid {
                reason: String::from("crawler.min_seeds must be > 0"),
            });
        }
        if self.crawler.requests_per_sec <= 0.0 {
            return Err(Error::Invalid {
                reason: String::from("crawler.requests_per_sec must be > 0"),
            });
        }
        if self.indexer.writer_heap_mb == 0 {
            return Err(Error::Invalid {
                reason: String::from("indexer.writer_heap_mb must be > 0"),
            });
        }
        if self.ranker.gbdt_trees == 0 {
            return Err(Error::Invalid {
                reason: String::from("ranker.gbdt_trees must be > 0"),
            });
        }
        if self.serving.bind.is_empty() {
            return Err(Error::Invalid {
                reason: String::from("serving.bind must not be empty"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn zero_min_seeds_invalid() {
        let mut cfg = Config::default();
        cfg.crawler.min_seeds = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_gbdt_trees_invalid() {
        let mut cfg = Config::default();
        cfg.ranker.gbdt_trees = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn defaults_match_spec() {
        let cfg = Config::default();
        assert_eq!(cfg.crawler.max_depth, 6);
        assert_eq!(cfg.crawler.max_pages, 1_000_000);
        assert_eq!(cfg.crawler.min_seeds, 100);
        assert_eq!(cfg.indexer.writer_heap_mb, 1024);
        assert_eq!(cfg.ranker.gbdt_trees, 300);
        assert_eq!(cfg.ranker.reranker_top_k, 32);
        assert_eq!(cfg.serving.bind, "0.0.0.0:8080");
        assert_eq!(cfg.serving.metrics_bind, "0.0.0.0:9090");
        assert_eq!(cfg.serving.max_request_bytes, 8192);
    }
}
