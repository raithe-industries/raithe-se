// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/storage/src/lib.rs
//
// All durable I/O — crawl log, doc store, mmap read access, and backups.

pub mod backup;
pub mod crawl_log;
pub mod doc_store;
pub mod mmap_file;

pub use self::backup::Backup;
pub use self::crawl_log::{CrawlEntry, CrawlLog};
pub use self::doc_store::DocStore;
pub use self::mmap_file::MmapFile;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error on '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("path traversal rejected: '{path}' is not rooted under the backup directory")]
    PathTraversal { path: String },
    #[error("document ID space exhausted — max_doc_id reached")]
    DocIdExhausted,
    #[error("compression error: {source}")]
    Compress {
        #[source]
        source: std::io::Error,
    },
    #[error("serialisation error: {reason}")]
    Serialise { reason: String },
}

pub type Result<T> = std::result::Result<T, Error>;
