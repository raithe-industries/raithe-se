// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/session/src/lib.rs
//
// In-memory session context — query history, reformulation detection,
// and per-session personalisation signals.

use std::sync::Arc;

use moka::sync::Cache;
use raithe_common::Timestamp;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum Error {
    #[error("session {id} not found")]
    NotFound { id: SessionId },
    #[error("invalid session id: {reason}")]
    InvalidId { reason: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// An opaque session identifier, backed by a UUID v4.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SessionId(Uuid);

impl SessionId {
    /// Generates a new random session identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Parses a UUID string into a `SessionId`.
    ///
    /// Accepts the standard 36-character hyphenated form (e.g. emitted by
    /// `Display`). Returns `Error::InvalidId` for any other input.
    pub fn parse_str(raw: &str) -> Result<Self> {
        Uuid::parse_str(raw)
            .map(Self)
            .map_err(|source| Error::InvalidId {
                reason: source.to_string(),
            })
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Per-session state held in memory.
#[derive(Clone, Debug)]
pub struct Session {
    pub id: SessionId,
    /// Query strings submitted in this session, in chronological order.
    pub queries: Vec<String>,
    /// Wall-clock time of the most recent query.
    pub last_seen: Timestamp,
    /// Whether the most recent query was detected as a reformulation of the
    /// previous one. Passed as a signal to the ranker.
    pub is_reformulation: bool,
}

impl Session {
    fn new(id: SessionId) -> Self {
        Self {
            id,
            queries: Vec::new(),
            last_seen: Timestamp::EPOCH,
            is_reformulation: false,
        }
    }
}

/// Detects when a follow-up query is a refinement of the previous one.
///
/// Uses a simple token-overlap heuristic: if the new query shares at least
/// half its tokens with the previous query it is classified as a reformulation.
pub struct ReformulationDetector;

impl ReformulationDetector {
    /// Returns `true` when `next` is a reformulation of `prev`.
    pub fn is_reformulation(prev: &str, next: &str) -> bool {
        let prev_tokens: std::collections::HashSet<&str> = prev.split_whitespace().collect();
        let next_tokens: Vec<&str> = next.split_whitespace().collect();

        if next_tokens.is_empty() || prev_tokens.is_empty() {
            return false;
        }

        let overlap = next_tokens
            .iter()
            .filter(|t| prev_tokens.contains(*t))
            .count();

        overlap * 2 >= next_tokens.len()
    }
}

/// In-memory session store backed by a Moka LRU cache with TTL eviction.
///
/// `SessionStore` is cheaply cloneable — all clones share the same underlying
/// cache via `Arc`.
#[derive(Clone)]
pub struct SessionStore {
    cache: Arc<Cache<SessionId, Session>>,
}

impl SessionStore {
    /// Creates a new store with the given capacity and TTL.
    ///
    /// Sessions inactive for longer than `ttl_secs` are evicted automatically.
    pub fn new(max_sessions: u64, ttl_secs: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_sessions)
            .time_to_idle(std::time::Duration::from_secs(ttl_secs))
            .build();
        let cache = Arc::new(cache);
        Self { cache }
    }

    /// Returns the session for the given `id`, creating one if it does not exist.
    pub fn get_or_create(&self, id: &SessionId) -> Session {
        self.cache.get(id).unwrap_or_else(|| Session::new(*id))
    }

    /// Records `query` in the session identified by `id`.
    ///
    /// Creates the session if it does not yet exist. Detects reformulation
    /// against the previous query and updates `session.is_reformulation`.
    pub fn record_query(&self, id: &SessionId, query: &str) -> Result<()> {
        let mut session = self.get_or_create(id);

        let is_reformulation = session
            .queries
            .last()
            .map(|prev| ReformulationDetector::is_reformulation(prev, query))
            .unwrap_or(false);

        session.queries.push(query.to_owned());
        session.last_seen = now();
        session.is_reformulation = is_reformulation;

        self.cache.insert(*id, session);

        Ok(())
    }

    /// Returns the number of sessions currently in the store.
    pub fn len(&self) -> u64 {
        self.cache.entry_count()
    }

    /// Returns `true` when no sessions are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn now() -> Timestamp {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    Timestamp::from_millis(millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_or_create_returns_new_session() {
        let store = SessionStore::new(100, 3600);
        let id = SessionId::new();
        let s = store.get_or_create(&id);
        assert_eq!(s.id, id);
        assert!(s.queries.is_empty());
    }

    #[test]
    fn record_query_appends() {
        let store = SessionStore::new(100, 3600);
        let id = SessionId::new();
        store.record_query(&id, "rust search engine").unwrap();
        store.record_query(&id, "fast rust search").unwrap();
        let s = store.get_or_create(&id);
        assert_eq!(s.queries.len(), 2);
    }

    #[test]
    fn reformulation_detected() {
        let store = SessionStore::new(100, 3600);
        let id = SessionId::new();
        store.record_query(&id, "rust search engine").unwrap();
        store.record_query(&id, "rust search engine fast").unwrap();
        let s = store.get_or_create(&id);
        assert!(s.is_reformulation);
    }

    #[test]
    fn no_reformulation_on_first_query() {
        let store = SessionStore::new(100, 3600);
        let id = SessionId::new();
        store.record_query(&id, "hello world").unwrap();
        let s = store.get_or_create(&id);
        assert!(!s.is_reformulation);
    }

    #[test]
    fn unrelated_query_not_reformulation() {
        assert!(!ReformulationDetector::is_reformulation(
            "rust programming language",
            "python web framework"
        ));
    }

    #[test]
    fn overlapping_query_is_reformulation() {
        assert!(ReformulationDetector::is_reformulation(
            "fast search engine",
            "fast search"
        ));
    }

    #[test]
    fn session_id_display_is_uuid_format() {
        let id = SessionId::new();
        assert_eq!(id.to_string().len(), 36);
    }

    #[test]
    fn session_id_round_trip_through_parse_str() {
        let id = SessionId::new();
        let parsed = SessionId::parse_str(&id.to_string()).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn session_id_parse_rejects_malformed() {
        assert!(SessionId::parse_str("not-a-uuid").is_err());
        assert!(SessionId::parse_str("").is_err());
        assert!(matches!(
            SessionId::parse_str("zzz"),
            Err(Error::InvalidId { .. })
        ));
    }
}
