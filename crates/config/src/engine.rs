// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/engine.rs
//
// Top-level engine mode flags. Phase 1 of the staged bring-up uses BM25-only
// retrieval; the heavy neural / semantic / link-graph / freshness paths stay
// dark until each subsystem has its own integration test.

/// Engine-mode configuration.
///
/// `phase1_only` (default `true`) gates every advanced subsystem:
///
/// - no `EmbedEngine`, `RerankEngine`, `GenerateEngine` load
/// - no `SemanticIndex`, `LinkGraph`, `FreshnessManager` init or task
/// - `QueryProcessor` returns `original` as `rewritten`
/// - `Ranker` skips Phase 3 cross-encoder
/// - indexing pipeline is `parse -> Indexer::add` only
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct EngineConfig {
    /// Disable all neural / semantic / link-graph / freshness paths.
    pub phase1_only: bool,
    /// Auto-commit threshold — commit when pending docs >= this.
    pub commit_every_docs: u64,
    /// Auto-commit interval (seconds) — commit at least this often when pending > 0.
    pub commit_every_secs: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            phase1_only:       true,
            commit_every_docs: 64,
            commit_every_secs: 5,
        }
    }
}