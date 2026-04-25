// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/query/src/lib.rs

use std::sync::{Arc, Mutex};

use raithe_common::{ParsedQuery, QueryIntent};
use raithe_metrics::Metrics;
use raithe_neural::GenerateEngine;
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
/// `neural` is optional — when `None` (Phase 1), `process` returns the
/// `original` string as `rewritten` and skips LLM rewriting entirely.
pub struct QueryProcessor {
    neural:   Option<Mutex<GenerateEngine>>,
    _metrics: Arc<Metrics>,
}

impl QueryProcessor {
    /// Full pipeline with LLM rewriting.
    pub fn new(neural: GenerateEngine, metrics: Arc<Metrics>) -> Self {
        Self { neural: Some(Mutex::new(neural)), _metrics: metrics }
    }

    /// Phase 1 — pass-through processor (no LLM).
    pub fn pass_through(metrics: Arc<Metrics>) -> Self {
        Self { neural: None, _metrics: metrics }
    }

    pub fn process(&self, raw: &str) -> Result<ParsedQuery> {
        let original  = raw.trim().to_owned();
        let tokens    = tokenise(&original);
        let intent    = classify_intent(&tokens);
        let synonyms  = expand_synonyms(&tokens);
        let rewritten = match &self.neural {
            Some(engine) => self.rewrite(&original, engine)?,
            None         => original.clone(),
        };

        Ok(ParsedQuery { original, tokens, intent, synonyms, rewritten })
    }

    fn rewrite(&self, query: &str, engine: &Mutex<GenerateEngine>) -> Result<String> {
        let prompt = format!(
            "Rewrite the following search query to improve recall. \
             Return only the rewritten query, nothing else.\nQuery: {query}"
        );
        let mut engine = engine.lock().map_err(|_| Error::LockPoisoned)?;
        let rewritten  = engine.generate(&prompt)
            .map_err(|err| Error::Rewrite { reason: err.to_string() })?;
        let rewritten = rewritten.trim().to_owned();
        if rewritten.is_empty() { Ok(query.to_owned()) } else { Ok(rewritten) }
    }
}

fn tokenise(query: &str) -> Vec<String> {
    query.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

fn classify_intent(tokens: &[String]) -> QueryIntent {
    let nav   = ["site", "www", "http", "https", ".com", ".org"];
    let local = ["near", "nearby", "in", "around", "location"];
    let trans = ["buy", "shop", "price", "order", "cheap", "deal", "discount"];
    for token in tokens {
        if nav.iter().any(|s| token.contains(s)) { return QueryIntent::Navigational; }
        if trans.contains(&token.as_str())       { return QueryIntent::Transactional; }
        if local.contains(&token.as_str())       { return QueryIntent::Local; }
    }
    QueryIntent::Informational
}

fn expand_synonyms(tokens: &[String]) -> Vec<Vec<String>> {
    tokens.iter().map(|_| Vec::new()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use raithe_metrics::Metrics;

    fn metrics() -> Arc<Metrics> { Arc::new(Metrics::new().unwrap()) }

    #[test]
    fn pass_through_returns_original_as_rewritten() {
        let proc   = QueryProcessor::pass_through(metrics());
        let parsed = proc.process("rust search engine").unwrap();
        assert_eq!(parsed.original, parsed.rewritten);
        assert_eq!(parsed.tokens, vec!["rust", "search", "engine"]);
    }

    #[test]
    fn tokenise_lowercases_and_splits() {
        assert_eq!(tokenise("Hello, World! Rust"), vec!["hello", "world", "rust"]);
    }

    #[test]
    fn intent_navigational() {
        assert_eq!(classify_intent(&tokenise("www.example.com")), QueryIntent::Navigational);
    }
}