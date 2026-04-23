// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/storage/src/doc_store.rs
//
// Key→compressed-bytes document store.
// Each document is stored as a zstd-compressed blob keyed by DocumentId.
// The counter uses checked_add — overflow returns Error::DocIdExhausted (§9.1).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use raithe_common::DocumentId;

use crate::{Error, Result};

/// In-memory document store backed by an optional flush-to-disk path.
///
/// Documents are stored as zstd-compressed byte blobs keyed by `DocumentId`.
/// The internal counter is advanced via `DocumentId::next()` — integer
/// overflow returns `Error::DocIdExhausted` rather than wrapping.
pub struct DocStore {
    _path: PathBuf,
    counter: Mutex<DocumentId>,
    map: Mutex<HashMap<DocumentId, Vec<u8>>>,
}

impl DocStore {
    /// Opens (or creates) the doc store rooted at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Self {
            _path: path.to_path_buf(),
            counter: Mutex::new(DocumentId::ZERO),
            map: Mutex::new(HashMap::new()),
        })
    }

    /// Stores `bytes` compressed with zstd and returns the assigned `DocumentId`.
    ///
    /// Returns `Error::DocIdExhausted` when the ID counter would overflow.
    pub fn put(&self, bytes: &[u8]) -> Result<DocumentId> {
        let compressed = zstd::encode_all(bytes, 3).map_err(|source| Error::Compress { source })?;

        let id = {
            let mut counter = self.counter.lock().unwrap_or_else(|p| p.into_inner());
            let next = counter.next().ok_or(Error::DocIdExhausted)?;
            *counter = next;
            next
        };

        {
            let mut map = self.map.lock().unwrap_or_else(|p| p.into_inner());
            map.insert(id, compressed);
        }

        Ok(id)
    }

    /// Retrieves and decompresses the document stored under `id`.
    ///
    /// Returns `None` if the ID is not present.
    pub fn get(&self, id: DocumentId) -> Result<Option<Vec<u8>>> {
        let map = self.map.lock().unwrap_or_else(|p| p.into_inner());
        let compressed = match map.get(&id) {
            Some(blob) => blob.clone(),
            None => return Ok(None),
        };

        let decompressed =
            zstd::decode_all(compressed.as_slice()).map_err(|source| Error::Compress { source })?;
        Ok(Some(decompressed))
    }

    /// Returns the number of documents currently stored.
    pub fn len(&self) -> usize {
        let map = self.map.lock().unwrap_or_else(|p| p.into_inner());
        map.len()
    }

    /// Returns `true` when no documents are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = DocStore::open(dir.path()).unwrap();

        let data = b"hello, raithe-se";
        let id = store.put(data).unwrap();
        let retrieved = store.get(id).unwrap().unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn missing_id_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = DocStore::open(dir.path()).unwrap();
        let absent = store.get(DocumentId::new(999)).unwrap();
        assert!(absent.is_none());
    }

    #[test]
    fn ids_increment_sequentially() {
        let dir = tempfile::tempdir().unwrap();
        let store = DocStore::open(dir.path()).unwrap();

        let a = store.put(b"a").unwrap();
        let b = store.put(b"b").unwrap();
        // Both IDs must differ and b must follow a.
        assert_ne!(a, b);
        assert!(b.get() > a.get());
    }
}
