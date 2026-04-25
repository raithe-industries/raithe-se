// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/crawler/src/lib.rs
//
// URL frontier, politeness scheduling, robots.txt, dispatch.

mod frontier;
mod policy;
mod robots;

pub use frontier::Frontier;
pub use policy::CrawlPolicy;
pub use robots::RobotsCache;

use std::collections::{HashSet, VecDeque};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use governor::{
    clock::DefaultClock,
    middleware::NoOpMiddleware,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use moka::sync::Cache;
use raithe_common::Url;
use raithe_config::CrawlerConfig;
use raithe_metrics::Metrics;
use raithe_scraper::{FetchResult, Scraper};
use raithe_storage::CrawlLog;
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Debug, Error)]
pub enum Error {
    #[error("insufficient seeds: need at least {min}, got {got}")]
    InsufficientSeeds { min: usize, got: usize },
    #[error("frontier error: {reason}")]
    Frontier { reason: String },
    #[error("scraper error: {source}")]
    Scraper { #[source] source: raithe_scraper::Error },
    #[error("storage error: {source}")]
    Storage { #[source] source: raithe_storage::Error },
    #[error("internal error: {reason}")]
    Internal { reason: String },
}

pub type Result<T> = std::result::Result<T, Error>;

type HostLimiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>;

const MAX_FETCH_ATTEMPTS: u8 = 3;

#[derive(Clone, Copy, Debug)]
enum UrlState {
    /// Currently failing; n attempts so far. Eligible for retry until n hits MAX_FETCH_ATTEMPTS.
    Pending(u8),
    /// Permanently skip — fetched ok or gave up after MAX_FETCH_ATTEMPTS.
    Done,
}

pub struct Crawler {
    config:       CrawlerConfig,
    frontier:     Frontier,
    robots:       RobotsCache,
    limiters:     Cache<String, Arc<HostLimiter>>,
    scraper:      Scraper,
    crawl_log:    Arc<CrawlLog>,
    metrics:      Arc<Metrics>,
    url_states:   Arc::new(Mutex::new(HashMap::new())),    
    fetched:      Arc<AtomicU64>,
    last_errors:  Arc<Mutex<VecDeque<String>>>,
}

impl Crawler {
    pub fn new(
        config:    CrawlerConfig,
        seeds:     Vec<Url>,
        scraper:   Scraper,
        crawl_log: Arc<CrawlLog>,
        metrics:   Arc<Metrics>,
    ) -> Result<Self> {
        if seeds.len() < config.min_seeds {
            return Err(Error::InsufficientSeeds { min: config.min_seeds, got: seeds.len() });
        }

        let frontier = Frontier::new(&seeds).map_err(|reason| Error::Frontier { reason })?;
        let robots   = RobotsCache::new(&config.user_agent);
        let limiters = Cache::builder()
            .max_capacity(65_536)
            .time_to_idle(Duration::from_secs(3_600))
            .build();

        Ok(Self {
            config,
            frontier,
            robots,
            limiters,
            scraper,
            crawl_log,
            metrics,
            seen_urls:   Arc::new(Mutex::new(HashSet::new())),
            fetched:     Arc::new(AtomicU64::new(0)),
            last_errors: Arc::new(Mutex::new(VecDeque::with_capacity(LAST_ERRORS_CAP))),
        })
    }

    /// Runs the crawl loop. Terminates only when `max_pages` is reached.
    /// An empty frontier is treated as transient — the loop sleeps and retries
    /// rather than exiting (live engines never run their seed list dry).
    pub async fn run(&self, tx: mpsc::Sender<(FetchResult, u32)>) -> Result<()> {
        let mut pages_fetched: u64 = 0;

        loop {
            if pages_fetched >= self.config.max_pages {
                tracing::info!(pages_fetched, "max_pages reached — crawl complete");
                break;
            }

            let entry = match self.frontier.pop() {
                Some(e) => e,
                None    => {
                    tokio::time::sleep(EMPTY_FRONTIER_NAP).await; 
                    continue; 
                }
            };

            let depth = entry.depth;
            let url   = entry.url;

            // Decide whether to attempt this URL now. Permanently-Done URLs skip;
            // Pending URLs are attempted again until MAX_FETCH_ATTEMPTS.
            {
                let mut states = self.url_states.lock().unwrap_or_else(|p| p.into_inner());
                match states.get(&url).copied() {
                    Some(UrlState::Done) => continue,
                    Some(UrlState::Pending(n)) if n >= MAX_FETCH_ATTEMPTS => {
                        states.insert(url.clone(), UrlState::Done);
                        continue;
                    }
                    Some(UrlState::Pending(n)) => { states.insert(url.clone(), UrlState::Pending(n + 1)); }
                    None                       => { states.insert(url.clone(), UrlState::Pending(1)); }
                }
            }

            if !self.robots.is_allowed(&url, &self.scraper).await {
                tracing::debug!(%url, "robots.txt disallowed — skipping");
                continue;
            }

            let limiter = self.limiter_for(&url);
            let host    = url.registered_domain().unwrap_or_default().to_owned();
            {
                let limiter = limiter.clone();
                let host_l  = host.clone();
                let metrics = self.metrics.clone();
                tokio::task::spawn_blocking(move || {
                    if let Err(not_until) = limiter.check() {
                        metrics.rate_limited_total.with_label_values(&[&host_l]).inc();
                        let clock = governor::clock::DefaultClock::default();
                        let wait  = not_until.wait_time_from(governor::clock::Clock::now(&clock));
                        std::thread::sleep(wait);
                    }
                })
                .await
                .map_err(|err| Error::Internal { reason: err.to_string() })?;
            }

            let fetch_result = match self.scraper.fetch(&url).await {
                Ok(r)  => r,
                Err(e) => {
                    let msg = format!("fetch {url}: {e}");
                    tracing::warn!("{msg}");
                    self.record_error(msg);
                    self.metrics.pages_crawled_total.with_label_values(&["error"]).inc();
                    self.metrics.errors_total.with_label_values(&["crawler", "fetch"]).inc();
                    continue; // state stays Pending(n) — will retry if rediscovered
                }
            };

            // ... after status_class computed ...
            {
                let mut states = self.url_states.lock().unwrap_or_else(|p| p.into_inner());
                states.insert(url.clone(), UrlState::Done);
            }

            let status       = fetch_result.status;
            let fetched_at   = fetch_result.fetched_at;
            let body_len     = fetch_result.body_bytes.len();
            let status_class = status_class(status);

            {
                let log_entry = raithe_storage::CrawlEntry {
                    id:         raithe_common::DocumentId::ZERO,
                    url:        url.clone(),
                    status,
                    fetched_at,
                    body_bytes: body_len,
                };
                self.crawl_log.append(&log_entry).map_err(|source| Error::Storage { source })?;
            }

            self.metrics.pages_crawled_total.with_label_values(&[status_class]).inc();
            pages_fetched = pages_fetched.saturating_add(1);
            self.fetched.fetch_add(1, Ordering::Relaxed);

            if status == 200 {
                let _ = tx.send((fetch_result, depth)).await;
            }

            self.metrics.crawl_queue_depth.set(self.frontier.depth() as f64);
        }

        Ok(())
    }

    pub fn enqueue_outlinks(&self, outlinks: &[Url], depth: u32) {
        let to_push: Vec<Url> = {
            let states = self.url_states.lock().unwrap_or_else(|p| p.into_inner());
            outlinks.iter()
                .filter(|u| !matches!(states.get(*u), Some(UrlState::Done)))
                .cloned()
                .collect()
        };
        for url in to_push {
            self.frontier.push(&url, depth);
        }
}

    // ── /debug/stats accessors ──────────────────────────────────────────────

    pub fn fetched_count(&self) -> u64 { self.fetched.load(Ordering::Relaxed) }

    pub fn frontier_size(&self) -> u64 { self.frontier.depth() }

    pub fn seen_count(&self) -> usize {
        self.url_states.lock().map(|s| s.len()).unwrap_or(0)
    }

    pub fn last_errors(&self) -> Vec<String> {
        self.last_errors.lock().map(|q| q.iter().cloned().collect()).unwrap_or_default()
    }

    fn record_error(&self, msg: String) {
        let mut q = self.last_errors.lock().unwrap_or_else(|p| p.into_inner());
        if q.len() == LAST_ERRORS_CAP { q.pop_front(); }
        q.push_back(msg);
    }

    fn limiter_for(&self, url: &Url) -> Arc<HostLimiter> {
        let domain = url.registered_domain().unwrap_or("__unknown__").to_owned();
        self.limiters.get_with(domain, || {
            let rps   = self.config.requests_per_sec.max(1.0) as u32;
            let rps   = NonZeroU32::new(rps).unwrap_or(NonZeroU32::MIN);
            let quota = Quota::per_second(rps);
            Arc::new(RateLimiter::direct(quota))
        })
    }
}

fn status_class(status: u16) -> &'static str {
    match status {
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _         => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_class_buckets() {
        assert_eq!(status_class(200), "2xx");
        assert_eq!(status_class(404), "4xx");
        assert_eq!(status_class(503), "5xx");
        assert_eq!(status_class(100), "other");
    }
}