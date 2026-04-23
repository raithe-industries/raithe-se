// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/query/src/lib.rs
//
// Query understanding pipeline — tokenisation, intent classification,
// synonym expansion, and LLM rewriting via Qwen2.5.

use std::sync::{Arc, Mutex};

use raithe_common::{ParsedQuery, QueryIntent};
use raithe_metrics::Metrics;
use raithe_neural::NeuralEngine;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("query processor lock poisoned")]
    LockPoisoned,
    #[error("LLM rewrite error: {reason}")]
    Rewrite { reason: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Query understanding pipeline.
///
/// Tokenises the raw query, classifies intent, expands synonyms, and rewrites
/// using the Qwen2.5 LLM. `NeuralEngine` is held behind a `Mutex` because
/// `generate` requires `&mut self` on the ONNX session.
pub struct QueryProcessor {
    neural: Mutex<NeuralEngine>,
    _metrics: Arc<Metrics>,
}

impl QueryProcessor {
    /// Constructs a `QueryProcessor` wrapping the given neural engine.
    pub fn new(neural: NeuralEngine, metrics: Arc<Metrics>) -> Self {
        let neural = Mutex::new(neural);
        let _metrics = metrics;
        Self { neural, _metrics }
    }

    /// Processes `raw` into a fully understood `ParsedQuery`.
    ///
    /// Steps:
    ///   1. Normalise and tokenise.
    ///   2. Classify intent.
    ///   3. Expand synonyms. TODO(impl) — synonym dictionary not yet loaded.
    ///   4. Rewrite via Qwen2.5 LLM.
    pub fn process(&self, raw: &str) -> Result<ParsedQuery> {
        let original = raw.trim().to_owned();
        let tokens = tokenise(&original);
        let intent = classify_intent(&tokens);
        let synonyms = expand_synonyms(&tokens);
        let rewritten = self.rewrite(&original)?;

        Ok(ParsedQuery {
            original,
            tokens,
            intent,
            synonyms,
            rewritten,
        })
    }

    /// Rewrites the query using the Qwen2.5 generator.
    ///
    /// Falls back to returning the original query if the neural engine is
    /// unavailable or produces an empty result, so search always proceeds.
    fn rewrite(&self, query: &str) -> Result<String> {
        let prompt = format!(
            "Rewrite the following search query to improve recall. \
             Return only the rewritten query, nothing else.\nQuery: {query}"
        );

        let mut engine = self.neural.lock().map_err(|_| Error::LockPoisoned)?;
        let rewritten = engine.generate(&prompt).map_err(|err| Error::Rewrite {
            reason: err.to_string(),
        })?;

        let rewritten = rewritten.trim().to_owned();
        if rewritten.is_empty() {
            Ok(query.to_owned())
        } else {
            Ok(rewritten)
        }
    }
}

// ── Pipeline stages ──────────────────────────────────────────────────────────

/// Lowercases, strips punctuation, and splits on whitespace.
fn tokenise(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Classifies query intent from token signals.
///
/// Heuristic rules covering the most common cases.
/// TODO(impl) — replace with a trained classifier once labelled data is
/// available (MEDIUM priority).
fn classify_intent(tokens: &[String]) -> QueryIntent {
    let navigational_signals = ["site", "www", "http", "https", ".com", ".org"];
    let local_signals = ["near", "nearby", "in", "around", "location"];
    let transactional_signals = ["buy", "shop", "price", "order", "cheap", "deal", "discount"];

    for token in tokens {
        if navigational_signals.iter().any(|s| token.contains(s)) {
            return QueryIntent::Navigational;
        }
        if transactional_signals.contains(&token.as_str()) {
            return QueryIntent::Transactional;
        }
        if local_signals.contains(&token.as_str()) {
            return QueryIntent::Local;
        }
    }

    QueryIntent::Informational
}

/// Returns per-token synonym lists.
///
/// TODO(impl) — load a real synonym dictionary (WordNet or similar).
/// Currently returns empty lists so downstream can handle the absent data.
fn expand_synonyms(tokens: &[String]) -> Vec<Vec<String>> {
    tokens.iter().map(|_| Vec::new()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenise_lowercases_and_splits() {
        let tokens = tokenise("Hello, World! Rust");
        assert_eq!(tokens, vec!["hello", "world", "rust"]);
    }

    #[test]
    fn tokenise_strips_empty_tokens() {
        let tokens = tokenise("  multiple   spaces  ");
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|t| !t.is_empty()));
    }

    #[test]
    fn intent_transactional() {
        let tokens = tokenise("buy cheap shoes");
        assert_eq!(classify_intent(&tokens), QueryIntent::Transactional);
    }

    #[test]
    fn intent_navigational() {
        let tokens = tokenise("www.example.com");
        assert_eq!(classify_intent(&tokens), QueryIntent::Navigational);
    }

    #[test]
    fn intent_local() {
        let tokens = tokenise("coffee near me");
        assert_eq!(classify_intent(&tokens), QueryIntent::Local);
    }

    #[test]
    fn intent_informational_default() {
        let tokens = tokenise("how does rust borrow checker work");
        assert_eq!(classify_intent(&tokens), QueryIntent::Informational);
    }

    #[test]
    fn synonyms_same_length_as_tokens() {
        let tokens = tokenise("fast search engine");
        let synonyms = expand_synonyms(&tokens);
        assert_eq!(synonyms.len(), tokens.len());
    }
}
