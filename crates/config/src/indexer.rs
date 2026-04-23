// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/indexer.rs

/// Tantivy inverted-index configuration.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct IndexerConfig {
    /// Heap budget for the Tantivy index writer (megabytes).
    /// Prototype used 50 MB (DEV-007); spec mandates 1024 MB.
    pub writer_heap_mb: u64,
    /// Hard ceiling on document IDs. Checked via `checked_add` — overflow
    /// returns `Error::DocIdExhausted` rather than wrapping.
    pub max_doc_id: u64,
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            writer_heap_mb: 1024,
            max_doc_id: u64::MAX - 1,
        }
    }
}
