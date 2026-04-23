// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/common/src/document_id.rs
//
// Newtype wrapper over u64 for document identifiers.

use serde::{Deserialize, Serialize};

/// A unique, monotonically-assigned identifier for an indexed document.
///
/// Wraps a `u64`. Use `DocumentId::next` to advance the counter safely;
/// overflow returns `None` rather than wrapping.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub struct DocumentId(u64);

impl DocumentId {
    /// The smallest valid document identifier.
    pub const ZERO: Self = Self(0);

    /// Wraps the given raw `value`.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw `u64` value.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Returns the identifier that follows `self`, or `None` on overflow.
    ///
    /// Callers must propagate `None` as `Error::DocIdExhausted` — never
    /// silently wrap.
    pub fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for DocumentId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<DocumentId> for u64 {
    fn from(id: DocumentId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_increments() {
        let id = DocumentId::new(41);
        assert_eq!(id.next(), Some(DocumentId::new(42)));
    }

    #[test]
    fn next_saturates_at_max() {
        let id = DocumentId::new(u64::MAX);
        assert_eq!(id.next(), None);
    }

    #[test]
    fn round_trip_u64() {
        let id = DocumentId::new(7);
        assert_eq!(u64::from(id), 7);
    }
}
