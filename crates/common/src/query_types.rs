// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/common/src/query_types.rs
//
// Shared query pipeline data types used across indexer, ranker, instant,
// session, and serving. Defined in common to avoid circular dependencies.

use serde::{Deserialize, Serialize};

use crate::{DocumentId, Url};

/// The intent classification of a parsed query.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum QueryIntent {
    #[default]
    Informational,
    Navigational,
    Transactional,
    Local,
}

/// A fully processed query, ready for the indexer and ranker.
///
/// Produced by `QueryProcessor::process`. All pipeline stages from
/// indexer onward consume this type.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParsedQuery {
    /// The raw query string as submitted by the user.
    pub original:  String,
    /// Normalised tokens after stemming and stop-word removal.
    pub tokens:    Vec<String>,
    /// Classified intent of the query.
    pub intent:    QueryIntent,
    /// Synonym expansions for each token.
    pub synonyms:  Vec<Vec<String>>,
    /// LLM-rewritten query for improved recall; equals `original` when
    /// rewriting is disabled or unavailable.
    pub rewritten: String,
}

impl ParsedQuery {
    /// Constructs a minimal `ParsedQuery` directly from a raw string.
    ///
    /// Used in tests and early pipeline stages where full processing is not
    /// yet available. Sets `rewritten` equal to `original` and leaves
    /// `tokens`, `intent`, and `synonyms` at their defaults.
    pub fn raw(original: impl Into<String>) -> Self {
        let original = original.into();
        let rewritten = original.clone();
        Self {
            original,
            tokens:   Vec::new(),
            intent:   QueryIntent::default(),
            synonyms: Vec::new(),
            rewritten,
        }
    }
}

/// A single hit from the Phase 1 BM25F search.
///
/// Passed from `Indexer::search` into `Ranker::rank` for Phase 2 and 3
/// re-ranking.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawHit {
    /// The matched document's identifier.
    pub id:      DocumentId,
    /// The document's URL (for snippet and result display).
    pub url:     Url,
    /// BM25F score from Tantivy.
    pub score:   f32,
    /// Extracted snippet of matching body text.
    pub snippet: String,
    /// Document title.
    pub title:   String,
}
