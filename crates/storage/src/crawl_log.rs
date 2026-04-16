// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/storage/src/crawl_log.rs
//
// Append-only crawl log. Each entry records the outcome of one fetch attempt.
// Written as newline-delimited JSON, compressed with zstd on rotation.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use raithe_common::{DocumentId, Timestamp, Url};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// A single entry in the crawl log.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CrawlEntry {
    pub id:         DocumentId,
    pub url:        Url,
    pub status:     u16,
    pub fetched_at: Timestamp,
    /// Byte length of the response body before compression.
    pub body_bytes: usize,
}

/// Append-only, mutex-guarded crawl log backed by a file on disk.
///
/// Thread-safe: all appends acquire a `Mutex` over the file handle.
/// The log file is created if it does not exist.
pub struct CrawlLog {
    path:   PathBuf,
    file:   Mutex<File>,
}

impl CrawlLog {
    /// Opens (or creates) the crawl log at the given `path`.
    pub fn open(path: &Path) -> Result<Self> {
        let path = path.to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| Error::Io {
                path: path.display().to_string(),
                source,
            })?;
        let file = Mutex::new(file);
        Ok(Self { path, file })
    }

    /// Appends `entry` to the log as a newline-delimited JSON record.
    ///
    /// Returns `Error::Io` on write failure. The mutex is held only for the
    /// duration of the write — lock poisoning is handled explicitly.
    pub fn append(&self, entry: &CrawlEntry) -> Result<()> {
        let mut line = serde_json::to_string(entry).map_err(|err| {
            Error::Serialise {
                reason: err.to_string(),
            }
        })?;
        line.push('\n');

        let mut guard = self.file.lock().unwrap_or_else(|poisoned| {
            poisoned.into_inner()
        });
        guard.write_all(line.as_bytes()).map_err(|source| Error::Io {
            path: self.path.display().to_string(),
            source,
        })?;
        guard.flush().map_err(|source| Error::Io {
            path: self.path.display().to_string(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crawl.log");
        let log = CrawlLog::open(&path).unwrap();

        let entry = CrawlEntry {
            id:         DocumentId::new(1),
            url:        Url::parse("https://example.com/").unwrap(),
            status:     200,
            fetched_at: Timestamp::from_millis(0),
            body_bytes: 1024,
        };
        log.append(&entry).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("example.com"));
        assert!(contents.contains("200"));
    }

    #[test]
    fn multiple_appends_produce_multiple_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crawl.log");
        let log = CrawlLog::open(&path).unwrap();

        for i in 0..3u64 {
            let entry = CrawlEntry {
                id:         DocumentId::new(i),
                url:        Url::parse("https://example.com/").unwrap(),
                status:     200,
                fetched_at: Timestamp::from_millis(i as i64),
                body_bytes: 100,
            };
            log.append(&entry).unwrap();
        }

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.lines().count(), 3);
    }
}
