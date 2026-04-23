// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/ranker.rs

use std::path::PathBuf;

/// Three-phase ranking pipeline configuration.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct RankerConfig {
    /// Number of GBDT trees in Phase 2. Prototype used 10 (DEV-001);
    /// spec mandates 300.
    pub gbdt_trees: usize,
    /// Number of candidates passed to the BGE cross-encoder in Phase 3.
    /// Prototype never invoked the reranker (DEV-005); spec mandates 32.
    pub reranker_top_k: usize,
    /// Path to the persisted LambdaMART GBDT model file.
    /// Trained via `Ranker::train` and saved here. If absent at startup,
    /// Phase 2 falls back to BM25F ordering until the first training run.
    pub gbdt_model_path: PathBuf,
}

impl Default for RankerConfig {
    fn default() -> Self {
        Self {
            gbdt_trees: 300,
            reranker_top_k: 32,
            gbdt_model_path: PathBuf::from("data/models/ranker/gbdt.model"),
        }
    }
}
