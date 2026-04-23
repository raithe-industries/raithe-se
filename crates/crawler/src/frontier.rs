// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/crawler/src/frontier.rs
//
// SQLite-backed priority queue of URLs awaiting crawling.
// Lower depth = higher priority (breadth-first by default).

use std::sync::Mutex;

use raithe_common::Url;
use rusqlite::{params, Connection};

/// An entry popped from the frontier.
pub struct FrontierEntry {
    pub url:   Url,
    pub depth: u32,
}

/// SQLite-backed URL priority queue.
///
/// Thread-safe: all operations acquire a `Mutex` over the connection.
/// The database is created in-memory so the frontier does not survive restart;
/// seeds are re-loaded on each startup (acceptable for v1).
pub struct Frontier {
    conn: Mutex<Connection>,
}

impl Frontier {
    /// Creates a new in-memory frontier and populates it with `seeds` at depth 0.
    ///
    /// Returns an error string on SQLite failure.
    pub fn new(seeds: &[Url]) -> Result<Self, String> {
        let conn = Connection::open_in_memory()
            .map_err(|e| e.to_string())?;

        conn.execute_batch(
            "CREATE TABLE frontier (
                 id    INTEGER PRIMARY KEY AUTOINCREMENT,
                 depth INTEGER NOT NULL,
                 url   TEXT    NOT NULL UNIQUE
             );
             CREATE INDEX frontier_depth_idx ON frontier (depth ASC, id ASC);",
        )
        .map_err(|e| e.to_string())?;

        {
            let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
            for url in seeds {
                tx.execute(
                    "INSERT OR IGNORE INTO frontier (depth, url) VALUES (?1, ?2)",
                    params![0i64, url.as_str()],
                )
                .map_err(|e| e.to_string())?;
            }
            tx.commit().map_err(|e| e.to_string())?;
        }

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Pops the shallowest URL from the frontier, returning `None` when empty.
    pub fn pop(&self) -> Option<FrontierEntry> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());

        // Select the row with minimum depth (ties broken by insertion order).
        let row: Option<(i64, String)> = conn
            .query_row(
                "SELECT id, url FROM frontier ORDER BY depth ASC, id ASC LIMIT 1",
                [],
                |row| {
                    let id:  i64    = row.get(0)?;
                    let url: String = row.get(1)?;
                    Ok((id, url))
                },
            )
            .ok();

        let (id, url_str) = row?;

        // Fetch depth before deleting.
        let depth: i64 = conn
            .query_row(
                "SELECT depth FROM frontier WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .ok()?;

        conn.execute("DELETE FROM frontier WHERE id = ?1", params![id])
            .ok()?;

        let url = Url::parse(&url_str).ok()?;
        Some(FrontierEntry {
            url,
            depth: depth as u32,
        })
    }

    /// Enqueues `url` at the given `depth` if not already present.
    pub fn push(&self, url: &Url, depth: u32) {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        let _ = conn.execute(
            "INSERT OR IGNORE INTO frontier (depth, url) VALUES (?1, ?2)",
            params![depth as i64, url.as_str()],
        );
    }

    /// Returns the current number of URLs in the frontier.
    pub fn depth(&self) -> u64 {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        conn.query_row("SELECT COUNT(*) FROM frontier", [], |row| row.get::<_, i64>(0))
            .unwrap_or(0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn pop_empty_returns_none() {
        let frontier = Frontier::new(&[]).unwrap();
        assert!(frontier.pop().is_none());
    }

    #[test]
    fn seed_popped_at_depth_zero() {
        let seeds = vec![url("https://example.com/")];
        let frontier = Frontier::new(&seeds).unwrap();
        let entry = frontier.pop().unwrap();
        assert_eq!(entry.depth, 0);
        assert_eq!(entry.url.as_str(), "https://example.com/");
    }

    #[test]
    fn breadth_first_ordering() {
        let seeds = vec![url("https://a.com/"), url("https://b.com/")];
        let frontier = Frontier::new(&seeds).unwrap();
        frontier.push(&url("https://a.com/deep"), 1);
        // Both depth-0 seeds must come before the depth-1 entry.
        let first  = frontier.pop().unwrap();
        let second = frontier.pop().unwrap();
        let third  = frontier.pop().unwrap();
        assert_eq!(first.depth,  0);
        assert_eq!(second.depth, 0);
        assert_eq!(third.depth,  1);
    }

    #[test]
    fn duplicate_url_not_enqueued_twice() {
        let seeds = vec![url("https://example.com/")];
        let frontier = Frontier::new(&seeds).unwrap();
        frontier.push(&url("https://example.com/"), 0);
        frontier.pop().unwrap();
        assert!(frontier.pop().is_none());
    }

    #[test]
    fn depth_reflects_queue_length() {
        let seeds = vec![
            url("https://a.com/"),
            url("https://b.com/"),
            url("https://c.com/"),
        ];
        let frontier = Frontier::new(&seeds).unwrap();
        assert_eq!(frontier.depth(), 3);
        frontier.pop();
        assert_eq!(frontier.depth(), 2);
    }
}
