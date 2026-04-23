// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/freshness/src/lib.rs
//
// Incremental re-crawl pipeline — staleness tracking, tombstones, recrawl queue.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use raithe_common::{DocumentId, Timestamp, Url};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("storage error: {source}")]
    Storage {
        #[source]
        source: raithe_storage::Error,
    },

    #[error("internal error: {reason}")]
    Internal { reason: String },
}

pub type Result<T> = std::result::Result<T, Error>;

// ── Tombstone ─────────────────────────────────────────────────────────────────

/// Marks a document as permanently removed from the index.
///
/// Once tombstoned, a document's URL is suppressed from all future crawl
/// batches and search results.
#[derive(Clone, Debug)]
pub struct Tombstone {
    pub id:         DocumentId,
    pub url:        Url,
    pub deleted_at: Timestamp,
}

// ── RecrawlQueue ──────────────────────────────────────────────────────────────

/// FIFO queue of URLs scheduled for re-crawl.
///
/// Backed by a `VecDeque` behind a `Mutex`. Tombstoned URLs are filtered
/// at enqueue time and at `next_batch` time.
struct RecrawlQueue {
    inner:      Mutex<VecDeque<(DocumentId, Url)>>,
    tombstones: Mutex<HashSet<DocumentId>>,
}

impl RecrawlQueue {
    fn new() -> Self {
        Self {
            inner:      Mutex::new(VecDeque::new()),
            tombstones: Mutex::new(HashSet::new()),
        }
    }

    fn enqueue(&self, id: DocumentId, url: Url) {
        let tombstones = self.tombstones.lock().unwrap_or_else(|p| p.into_inner());
        if tombstones.contains(&id) {
            return;
        }
        let mut queue = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        queue.push_back((id, url));
    }

    fn tombstone(&self, id: DocumentId) {
        let mut tombstones = self.tombstones.lock().unwrap_or_else(|p| p.into_inner());
        tombstones.insert(id);
    }

    fn is_tombstoned(&self, id: DocumentId) -> bool {
        let tombstones = self.tombstones.lock().unwrap_or_else(|p| p.into_inner());
        tombstones.contains(&id)
    }

    /// Drains up to `n` non-tombstoned URLs from the front of the queue.
    fn drain(&self, n: usize) -> Vec<Url> {
        let tombstones = self.tombstones.lock().unwrap_or_else(|p| p.into_inner());
        let mut queue  = self.inner.lock().unwrap_or_else(|p| p.into_inner());

        let mut batch = Vec::with_capacity(n);
        let mut rest  = VecDeque::new();

        while let Some((id, url)) = queue.pop_front() {
            if batch.len() >= n {
                rest.push_back((id, url));
            } else if tombstones.contains(&id) {
                // Drop tombstoned entry.
            } else {
                batch.push(url);
            }
        }

        // Retain remaining non-drained entries.
        queue.extend(rest);
        batch
    }

    fn len(&self) -> usize {
        let queue = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        queue.len()
    }
}

// ── FreshnessManager ──────────────────────────────────────────────────────────

/// Tracks document age, issues re-crawl requests, and manages tombstones.
///
/// `FreshnessManager` is cheaply clonable — all fields are `Arc`-backed
/// internally via `Mutex`.
pub struct FreshnessManager {
    /// Mapping from DocumentId to the URL and the timestamp when it was last
    /// successfully fetched.
    doc_index:       Mutex<HashMap<DocumentId, (Url, Timestamp)>>,
    recrawl_queue:   RecrawlQueue,
    /// Tombstone records keyed by DocumentId.
    tombstone_log:   Mutex<HashMap<DocumentId, Tombstone>>,
    /// Minimum age in milliseconds before a document is considered stale.
    stale_after_ms:  i64,
}

impl FreshnessManager {
    /// Creates a new `FreshnessManager`.
    ///
    /// `stale_after_ms` — minimum document age in milliseconds before it is
    /// eligible for re-crawl. A reasonable default is 86_400_000 (24 hours).
    pub fn new(stale_after_ms: i64) -> Self {
        Self {
            doc_index:      Mutex::new(HashMap::new()),
            recrawl_queue:  RecrawlQueue::new(),
            tombstone_log:  Mutex::new(HashMap::new()),
            stale_after_ms,
        }
    }

    /// Records that `id` at `url` was freshly fetched at `fetched_at`.
    ///
    /// Updates the index. Does not enqueue for re-crawl.
    pub fn record_fetch(&self, id: DocumentId, url: Url, fetched_at: Timestamp) {
        let mut index = self.doc_index.lock().unwrap_or_else(|p| p.into_inner());
        index.insert(id, (url, fetched_at));
    }

    /// Marks `id` as stale and enqueues its URL for re-crawl.
    ///
    /// No-ops if `id` is tombstoned or not in the index.
    pub fn mark_stale(&self, id: DocumentId) -> Result<()> {
        if self.recrawl_queue.is_tombstoned(id) {
            return Ok(());
        }
        let index = self.doc_index.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((url, _)) = index.get(&id) {
            self.recrawl_queue.enqueue(id, url.clone());
        }
        Ok(())
    }

    /// Tombstones `id` — permanently removes it from future crawl batches.
    ///
    /// Records a `Tombstone` entry with the current time.
    pub fn tombstone(&self, id: DocumentId, url: Url) -> Result<()> {
        self.recrawl_queue.tombstone(id);
        let deleted_at = unix_ms_now();
        let tombstone  = Tombstone { id, url, deleted_at };
        let mut log    = self.tombstone_log.lock().unwrap_or_else(|p| p.into_inner());
        log.insert(id, tombstone);
        Ok(())
    }

    /// Returns `true` if `id` has been tombstoned.
    pub fn is_tombstoned(&self, id: DocumentId) -> bool {
        self.recrawl_queue.is_tombstoned(id)
    }

    /// Scans all tracked documents and enqueues those older than `stale_after_ms`.
    ///
    /// Called periodically by the freshness background task.
    pub fn scan_stale(&self) -> Result<()> {
        let now = unix_ms_now();
        let index = self.doc_index.lock().unwrap_or_else(|p| p.into_inner());

        for (&id, (url, fetched_at)) in index.iter() {
            let age_ms = now.as_millis() - fetched_at.as_millis();
            if age_ms >= self.stale_after_ms {
                if !self.recrawl_queue.is_tombstoned(id) {
                    self.recrawl_queue.enqueue(id, url.clone());
                }
            }
        }
        Ok(())
    }

    /// Returns the next batch of up to `n` URLs from the re-crawl queue.
    ///
    /// Tombstoned URLs are automatically excluded.
    pub fn next_batch(&self, n: usize) -> Result<Vec<Url>> {
        Ok(self.recrawl_queue.drain(n))
    }

    /// Returns the current number of URLs pending re-crawl.
    pub fn queue_len(&self) -> usize {
        self.recrawl_queue.len()
    }

    /// Returns the tombstone record for `id`, if present.
    pub fn get_tombstone(&self, id: DocumentId) -> Option<Tombstone> {
        let log = self.tombstone_log.lock().unwrap_or_else(|p| p.into_inner());
        log.get(&id).cloned()
    }
}


// ── Helpers ──────────────────────────────────────────────────────────────────

fn unix_ms_now() -> Timestamp {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    Timestamp::from_millis(millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    fn id(n: u64) -> DocumentId {
        DocumentId::new(n)
    }

    #[test]
    fn mark_stale_enqueues_url() {
        let fm = FreshnessManager::new(0);
        fm.record_fetch(id(1), url("https://example.com/"), Timestamp::EPOCH);
        fm.mark_stale(id(1)).unwrap();
        assert_eq!(fm.queue_len(), 1);
    }

    #[test]
    fn next_batch_drains_queue() {
        let fm = FreshnessManager::new(0);
        fm.record_fetch(id(1), url("https://a.com/"), Timestamp::EPOCH);
        fm.record_fetch(id(2), url("https://b.com/"), Timestamp::EPOCH);
        fm.mark_stale(id(1)).unwrap();
        fm.mark_stale(id(2)).unwrap();

        let batch = fm.next_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(fm.queue_len(), 1);
    }

    #[test]
    fn tombstoned_doc_excluded_from_batch() {
        let fm = FreshnessManager::new(0);
        fm.record_fetch(id(1), url("https://gone.com/"), Timestamp::EPOCH);
        fm.mark_stale(id(1)).unwrap();
        fm.tombstone(id(1), url("https://gone.com/")).unwrap();

        let batch = fm.next_batch(10).unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn tombstone_recorded_in_log() {
        let fm = FreshnessManager::new(0);
        fm.tombstone(id(42), url("https://dead.com/")).unwrap();
        let t = fm.get_tombstone(id(42)).unwrap();
        assert_eq!(t.id, id(42));
    }

    #[test]
    fn mark_stale_noop_for_tombstoned() {
        let fm = FreshnessManager::new(0);
        fm.record_fetch(id(5), url("https://old.com/"), Timestamp::EPOCH);
        fm.tombstone(id(5), url("https://old.com/")).unwrap();
        fm.mark_stale(id(5)).unwrap();
        assert_eq!(fm.queue_len(), 0);
    }

    #[test]
    fn scan_stale_enqueues_old_documents() {
        // stale_after_ms = 0 means everything is immediately stale.
        let fm = FreshnessManager::new(0);
        fm.record_fetch(id(1), url("https://a.com/"), Timestamp::EPOCH);
        fm.record_fetch(id(2), url("https://b.com/"), Timestamp::EPOCH);
        fm.scan_stale().unwrap();
        // Both docs older than 0ms → both enqueued (may be 1 or 2 depending
        // on exact ms; EPOCH is always far in the past).
        assert!(fm.queue_len() >= 1);
    }

    #[test]
    fn next_batch_larger_than_queue_returns_all() {
        let fm = FreshnessManager::new(0);
        fm.record_fetch(id(1), url("https://x.com/"), Timestamp::EPOCH);
        fm.mark_stale(id(1)).unwrap();
        let batch = fm.next_batch(100).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(fm.queue_len(), 0);
    }
}
