// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/common/src/lib.rs
//
// Shared primitive types. No logic, no I/O, no async.

pub mod document_id;
pub mod embedding;
pub mod query_types;
pub mod simhash;
pub mod timestamp;
pub mod url_type;

pub use self::document_id::DocumentId;
pub use self::embedding::Embedding;
pub use self::query_types::{ParsedQuery, QueryIntent, RawHit};
pub use self::simhash::SimHash;
pub use self::timestamp::Timestamp;
pub use self::url_type::Url;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid URL: {reason}")]
    InvalidUrl { reason: String },
}

pub type Result<T> = std::result::Result<T, Error>;
