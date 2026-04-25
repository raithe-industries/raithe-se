// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/storage/src/url_registry.rs
//
// Durable URL → DocumentId mapping. SQLite-backed.
//
// Solves the "duplicate doc on re-crawl" problem: the indexing pipeline calls
// `assign(&url)` to obtain a stable document id. The same URL returns the
// same id forever — even across process restarts, even if the URL is
// re-fetched. Combined with `Indexer::upsert`, this guarantees each URL has
// at most one live document in the index.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use raithe_common::{DocumentId, Url};
use rusqlite::{params, Connection, OptionalExtension};
use thiserror::Error;

/// Schema version of the `url_registry` table. Bump when columns change
/// incompatibly. Mismatch is detected at `open()` time and refused — operator
/// must `rm` the file (URL identity is rebuildable from a re-crawl).
const REGISTRY_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Error)]
pub enum Error {
    #[error("sqlite error: {source}")]
    Sqlite { #[source] source: rusqlite::Error },
    #[error("registry schema version mismatch — expected v{expected}, found v{found}; remove the registry file and re-crawl")]
    SchemaMismatch { expected: i64, found: i64 },
    #[error("registry lock poisoned")]
    LockPoisoned,
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<rusqlite::Error> for Error {
    fn from(source: rusqlite::Error) -> Self { Self::Sqlite { source } }
}

/// Status of a URL in the registry. Mirrors the crawler's view of the URL
/// lifecycle. Stored as a TEXT column so values are human-readable in
/// sqlite3 dumps; cheap to extend without an ALTER TABLE migration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UrlStatus {
    /// First seen, not yet fetched.
    Queued,
    /// Successfully fetched and indexed.
    Fetched,
    /// Fetch attempted and failed permanently.
    Failed,
    /// URL removed from index (404, robots-disallow on recrawl, manual purge).
    Tombstoned,
}

impl UrlStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Queued     => "queued",
            Self::Fetched    => "fetched",
            Self::Failed     => "failed",
            Self::Tombstoned => "tombstoned",
        }
    }
}

/// Durable URL → `DocumentId` mapping with status tracking.
///
/// Thread-safe via internal `Mutex<Connection>`. SQLite handles concurrent
/// readers natively (WAL mode), but a single connection serialises writes —
/// adequate for the indexing pipeline's single writer task. If we ever need
/// concurrent assignments, swap the `Mutex<Connection>` for a connection
/// pool; the public API doesn't change.
pub struct UrlRegistry {
    conn: Mutex<Connection>,
    /// Retained for diagnostics — printed in error messages.
    #[allow(dead_code)]
    path: PathBuf,
}

impl UrlRegistry {
    /// Opens or creates the registry at `path`. Creates parent directories
    /// if needed. Initialises the schema on first open and verifies the
    /// schema version on subsequent opens.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::Sqlite {
                source: rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            })?;
        }

        let conn = Connection::open(path)?;
        // WAL: allows read-during-write. Critical for restart resilience —
        // if the process is killed mid-write, the WAL replays cleanly.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // NORMAL durability: trades a worst-case ~last-second-of-writes loss
        // on hard power loss for ~10x write throughput. Acceptable because
        // URL identity is recoverable from re-crawl in the worst case.
        conn.pragma_update(None, "synchronous", "NORMAL")?;

        Self::init_schema(&conn)?;
        Self::check_version(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
            path: path.to_path_buf(),
        })
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        // `IF NOT EXISTS` — idempotent. Column types match the registry API:
        //   url        TEXT PRIMARY KEY    — natural identity, indexed by sqlite
        //   doc_id     INTEGER NOT NULL    — assigned monotonically; UNIQUE
        //   status     TEXT NOT NULL       — one of UrlStatus::as_str()
        //   updated_at INTEGER NOT NULL    — unix millis, last status change
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS url_registry (
                url        TEXT    PRIMARY KEY NOT NULL,
                doc_id     INTEGER UNIQUE      NOT NULL,
                status     TEXT                NOT NULL,
                updated_at INTEGER             NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_url_registry_doc_id
                ON url_registry(doc_id);
            CREATE TABLE IF NOT EXISTS url_registry_meta (
                key   TEXT PRIMARY KEY NOT NULL,
                value TEXT             NOT NULL
            );
            "#,
        )?;
        // Stamp the schema version on first init only (INSERT OR IGNORE).
        conn.execute(
            "INSERT OR IGNORE INTO url_registry_meta(key, value) VALUES ('schema_version', ?1)",
            params![REGISTRY_SCHEMA_VERSION.to_string()],
        )?;
        Ok(())
    }

    fn check_version(conn: &Connection) -> Result<()> {
        let stored: Option<String> = conn.query_row(
            "SELECT value FROM url_registry_meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        ).optional()?;

        let stored: i64 = stored
            .as_deref()
            .and_then(|s| s.parse().ok())
            .unwrap_or(REGISTRY_SCHEMA_VERSION);

        if stored != REGISTRY_SCHEMA_VERSION {
            return Err(Error::SchemaMismatch {
                expected: REGISTRY_SCHEMA_VERSION,
                found:    stored,
            });
        }
        Ok(())
    }

    /// Returns the `DocumentId` for `url`, allocating one if absent.
    ///
    /// Idempotent: calling twice with the same URL returns the same id.
    /// New rows start in status `Queued`. The allocation strategy is
    /// `MAX(doc_id) + 1` inside a transaction — race-safe under the
    /// internal `Mutex<Connection>`.
    pub fn assign(&self, url: &Url) -> Result<DocumentId> {
        let mut conn = self.conn.lock().map_err(|_| Error::LockPoisoned)?;
        let url_str  = url.as_str();

        // Fast path: already assigned. Avoid a transaction for the common case.
        if let Some(id) = lookup_inner(&conn, url_str)? {
            return Ok(id);
        }

        // Slow path: allocate. Wrap in a transaction so the MAX(doc_id) read
        // and INSERT are atomic — two concurrent calls (if a second writer
        // ever exists) cannot allocate the same id.
        let tx = conn.transaction()?;

        // Re-check inside the tx — another caller may have assigned the URL
        // between the fast-path check and tx start.
        if let Some(id) = lookup_inner(&tx, url_str)? {
            tx.commit()?;
            return Ok(id);
        }

        let next_id: i64 = tx.query_row(
            "SELECT COALESCE(MAX(doc_id), 0) + 1 FROM url_registry",
            [],
            |row| row.get(0),
        )?;

        let now = unix_ms_now();
        tx.execute(
            "INSERT INTO url_registry(url, doc_id, status, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![url_str, next_id, UrlStatus::Queued.as_str(), now],
        )?;
        tx.commit()?;

        // doc_id 0 is reserved (DocumentId::ZERO is the sentinel for "unassigned").
        // Our MAX+1 starts at 1, so this is safe — but assert in debug builds.
        debug_assert!(next_id > 0, "registry allocated reserved doc_id 0");
        Ok(DocumentId::new(next_id as u64))
    }

    /// Returns the `DocumentId` for `url` if assigned, otherwise `None`.
    /// Read-only — does not allocate.
    pub fn lookup(&self, url: &Url) -> Result<Option<DocumentId>> {
        let conn = self.conn.lock().map_err(|_| Error::LockPoisoned)?;
        lookup_inner(&conn, url.as_str())
    }

    /// Updates the status of a previously-assigned URL. No-op if the URL is
    /// not in the registry — callers who need atomic assign+mark should
    /// call `assign` first.
    pub fn mark_status(&self, url: &Url, status: UrlStatus) -> Result<()> {
        let conn = self.conn.lock().map_err(|_| Error::LockPoisoned)?;
        let now  = unix_ms_now();
        conn.execute(
            "UPDATE url_registry SET status = ?1, updated_at = ?2 WHERE url = ?3",
            params![status.as_str(), now, url.as_str()],
        )?;
        Ok(())
    }

    /// Returns the total number of registered URLs. Useful for /debug/stats
    /// and end-to-end test assertions.
    pub fn len(&self) -> Result<u64> {
        let conn = self.conn.lock().map_err(|_| Error::LockPoisoned)?;
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM url_registry", [], |row| row.get(0))?;
        Ok(n.max(0) as u64)
    }

    pub fn is_empty(&self) -> Result<bool> { Ok(self.len()? == 0) }
}

fn lookup_inner(conn: &Connection, url_str: &str) -> Result<Option<DocumentId>> {
    let row: Option<i64> = conn.query_row(
        "SELECT doc_id FROM url_registry WHERE url = ?1",
        params![url_str],
        |row| row.get(0),
    ).optional()?;
    Ok(row.map(|id| DocumentId::new(id as u64)))
}

fn unix_ms_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_temp() -> (UrlRegistry, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let r   = UrlRegistry::open(&dir.path().join("registry.db")).unwrap();
        (r, dir)
    }

    #[test]
    fn assign_twice_returns_same_id() {
        let (r, _d) = open_temp();
        let u       = Url::parse("https://example.com/a").unwrap();
        let id1     = r.assign(&u).unwrap();
        let id2     = r.assign(&u).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn separate_urls_get_separate_ids() {
        let (r, _d) = open_temp();
        let a       = r.assign(&Url::parse("https://example.com/a").unwrap()).unwrap();
        let b       = r.assign(&Url::parse("https://example.com/b").unwrap()).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn doc_ids_are_monotonic_starting_at_one() {
        let (r, _d) = open_temp();
        let a = r.assign(&Url::parse("https://example.com/a").unwrap()).unwrap();
        let b = r.assign(&Url::parse("https://example.com/b").unwrap()).unwrap();
        let c = r.assign(&Url::parse("https://example.com/c").unwrap()).unwrap();
        assert_eq!(a.get(), 1);
        assert_eq!(b.get(), 2);
        assert_eq!(c.get(), 3);
    }

    #[test]
    fn persists_across_reopen() {
        let dir  = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.db");
        let url  = Url::parse("https://example.com/persistent").unwrap();

        let id_first = {
            let r = UrlRegistry::open(&path).unwrap();
            r.assign(&url).unwrap()
        };

        // Reopen — same id must come back.
        let r        = UrlRegistry::open(&path).unwrap();
        let id_again = r.assign(&url).unwrap();
        assert_eq!(id_first, id_again);
        assert_eq!(r.lookup(&url).unwrap(), Some(id_first));
    }

    #[test]
    fn lookup_missing_returns_none() {
        let (r, _d) = open_temp();
        let u       = Url::parse("https://example.com/nope").unwrap();
        assert_eq!(r.lookup(&u).unwrap(), None);
    }

    #[test]
    fn mark_status_updates_row() {
        let (r, _d) = open_temp();
        let u       = Url::parse("https://example.com/x").unwrap();
        let _       = r.assign(&u).unwrap();
        r.mark_status(&u, UrlStatus::Fetched).unwrap();
        // No public getter for status (yet) — round-trip check via reopen
        // would require exposing it; for now, assert the call doesn't error
        // and that subsequent assign still returns the same id.
        let id_after = r.assign(&u).unwrap();
        assert_eq!(id_after.get(), 1);
    }

    #[test]
    fn schema_init_is_idempotent() {
        let dir  = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.db");
        let _    = UrlRegistry::open(&path).unwrap();
        let _    = UrlRegistry::open(&path).unwrap();
        let r    = UrlRegistry::open(&path).unwrap();
        assert!(r.is_empty().unwrap());
    }

    #[test]
    fn len_reflects_assignments() {
        let (r, _d) = open_temp();
        assert_eq!(r.len().unwrap(), 0);
        let _ = r.assign(&Url::parse("https://example.com/a").unwrap()).unwrap();
        let _ = r.assign(&Url::parse("https://example.com/b").unwrap()).unwrap();
        let _ = r.assign(&Url::parse("https://example.com/a").unwrap()).unwrap(); // dup
        assert_eq!(r.len().unwrap(), 2);
    }
}