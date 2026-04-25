// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/ranker/src/lib.rs
//
// Three-phase ranking pipeline.
//   Phase 1: BM25F scores from Tantivy (carried in RawHit).
//   Phase 2: GBDT re-rank — 300 LambdaMART trees via the gbdt crate.
//   Phase 3: BGE-reranker-large cross-encoder over top-32 candidates.

use std::path::Path;
use std::sync::{Arc, Mutex};

use gbdt::config::Config as GbdtConfig;
use gbdt::decision_tree::{Data, DataVec};
use gbdt::errors::GbdtError;
use gbdt::gradient_boost::GBDT;
use raithe_common::{DocumentId, ParsedQuery, RawHit, Url};
use raithe_config::RankerConfig;
use raithe_linkgraph::PageRankScores;
use raithe_metrics::Metrics;
use raithe_neural::RerankEngine;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("cannot load GBDT model from '{path}': {reason}")]
    ModelLoad { path: String, reason: String },
    #[error("cannot save GBDT model to '{path}': {reason}")]
    ModelSave { path: String, reason: String },
    #[error("cannot train GBDT model: {reason}")]
    Train { reason: String },
    #[error("neural reranker error: {reason}")]
    Neural { reason: String },
    #[error("ranker lock poisoned")]
    LockPoisoned,
}

pub type Result<T> = std::result::Result<T, Error>;

// ── Feature vector ────────────────────────────────────────────────────────────

/// Five hand-crafted features per candidate document.
///
/// Layout must be stable — the order here is the order the GBDT model
/// was trained on. Adding features requires retraining.
///
///   0  bm25_score    — Tantivy BM25F score (Phase 1)
///   1  pagerank      — normalised PageRank score (0.0 if not yet computed)
///   2  title_len     — character length of the document title
///   3  snippet_len   — character length of the extracted snippet
///   4  query_tokens  — number of tokens in the parsed query
#[derive(Clone, Debug)]
pub struct FeatureVector {
    pub bm25_score:   f32,
    pub pagerank:     f32,
    pub title_len:    f32,
    pub snippet_len:  f32,
    pub query_tokens: f32,
}

impl FeatureVector {
    pub(crate) fn from_hit(hit: &RawHit, pagerank: f32, query_token_count: usize) -> Self {
        let bm25_score   = hit.score;
        let title_len    = hit.title.len() as f32;
        let snippet_len  = hit.snippet.len() as f32;
        let query_tokens = query_token_count as f32;
        Self {
            bm25_score,
            pagerank,
            title_len,
            snippet_len,
            query_tokens,
        }
    }

    /// Returns the feature values in canonical order for GBDT inference.
    pub fn as_vec(&self) -> Vec<f32> {
        vec![
            self.bm25_score,
            self.pagerank,
            self.title_len,
            self.snippet_len,
            self.query_tokens,
        ]
    }
}

// ── Training sample ───────────────────────────────────────────────────────────

/// A labelled training sample for the LambdaMART model.
///
/// Collect these from user interaction signals (clicks, dwell time, explicit
/// ratings) and pass a `Vec<TrainingSample>` to `Ranker::train` to retrain
/// or fine-tune the model.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    /// Feature vector for the candidate document.
    pub features: FeatureVector,
    /// Relevance label. Use 1.0 for relevant, 0.0 for non-relevant.
    pub label: f32,
}

// ── GBDT model ────────────────────────────────────────────────────────────────

/// Wraps a trained `gbdt::gradient_boost::GBDT` model for Phase 2 re-ranking.
///
/// On startup, `GbdtModel::load` attempts to read a persisted model from
/// `RankerConfig::gbdt_model_path`. When no model exists yet, `GbdtModel`
/// operates in untrained mode — it returns the raw BM25F score unchanged so
/// Phase 2 is a transparent pass-through rather than a failure.
struct GbdtModel {
    inner: Option<GBDT>,
}

impl GbdtModel {
    fn untrained() -> Self {
        Self { inner: None }
    }

    /// Loads a persisted model from `path`.
    ///
    /// Returns `Error::ModelLoad` if the file exists but cannot be parsed.
    /// Returns an untrained model (not an error) if the file is absent.
    fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::untrained());
        }

        let path_str = path.display().to_string();
        let gbdt = GBDT::load_model(path_str.as_str())
            .map_err(|err: GbdtError| Error::ModelLoad {
                path:   path_str,
                reason: err.to_string(),
            })?;

        Ok(Self { inner: Some(gbdt) })
    }

    /// Trains a 300-tree LambdaMART model on `samples` and saves it to `path`.
    ///
    /// Creates parent directories of `path` if needed. Replaces `self.inner`
    /// with the newly trained model on success.
    fn train_and_save(&mut self, samples: &[TrainingSample], path: &Path) -> Result<()> {
        if samples.is_empty() {
            return Err(Error::Train {
                reason: String::from("samples vec is empty"),
            });
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| Error::ModelSave {
                path:   path.display().to_string(),
                reason: err.to_string(),
            })?;
        }

        let mut training_data: DataVec = samples
            .iter()
            .map(|s| Data::new_training_data(s.features.as_vec(), 1.0, s.label, None))
            .collect();

        let cfg      = gbdt_config(300);
        let mut gbdt = GBDT::new(&cfg);
        gbdt.fit(&mut training_data);

        let path_str = path.display().to_string();
        gbdt.save_model(path_str.as_str())
            .map_err(|err: GbdtError| Error::ModelSave {
                path:   path_str,
                reason: err.to_string(),
            })?;

        self.inner = Some(gbdt);
        Ok(())
    }

    /// Predicts a relevance score for the given feature vector.
    ///
    /// Returns the BM25F score unchanged when no trained model is loaded.
    fn predict(&self, features: &FeatureVector) -> f32 {
        let inner = match &self.inner {
            Some(g) => g,
            None    => return features.bm25_score,
        };

        let test_data: DataVec = vec![Data::new_test_data(features.as_vec(), None)];
        let predictions = inner.predict(&test_data);
        predictions.into_iter().next().unwrap_or(features.bm25_score)
    }

    fn is_trained(&self) -> bool {
        self.inner.is_some()
    }
}

/// Standard hyperparameters for the 300-tree LambdaMART ensemble.
///
/// `LogLikelyhood` is the pairwise logistic loss — equivalent to the
/// RankNet/LambdaMART pairwise objective. Max depth 6 and shrinkage 0.1
/// are standard LambdaMART settings.
fn gbdt_config(tree_count: usize) -> GbdtConfig {
    let mut cfg = GbdtConfig::new();
    cfg.set_feature_size(5);
    cfg.set_max_depth(6);
    cfg.set_iterations(tree_count);
    cfg.set_shrinkage(0.1);
    cfg.set_loss("LogLikelyhood");
    cfg.set_debug(false);
    cfg.set_data_sample_ratio(1.0);
    cfg.set_feature_sample_ratio(1.0);
    cfg.set_training_optimization_level(2);
    cfg
}

// ── RankedResult ──────────────────────────────────────────────────────────────

/// A fully ranked search result, produced after all three pipeline phases.
#[derive(Clone, Debug)]
pub struct RankedResult {
    pub id:           DocumentId,
    pub url:          Url,
    /// Final composite score after all active phases.
    pub score:        f32,
    /// BM25F score from Phase 1.
    pub bm25_score:   f32,
    /// GBDT LambdaMART score from Phase 2.
    /// Equals `bm25_score` when no trained model is loaded.
    pub gbdt_score:   f32,
    /// BGE cross-encoder score from Phase 3.
    /// 0.0 for candidates outside the top-k window.
    pub rerank_score: f32,
    pub snippet:      String,
    pub title:        String,
}

// ── Ranker ────────────────────────────────────────────────────────────────────

pub struct Ranker {
    config:   RankerConfig,
    gbdt:     Mutex<GbdtModel>,
    /// `None` in Phase 1 — Phase 3 is skipped.
    neural:   Option<Mutex<RerankEngine>>,
    _metrics: Arc<Metrics>,
}

impl Ranker {
    pub fn new(config: RankerConfig, neural: RerankEngine, metrics: Arc<Metrics>) -> Result<Self> {
        let gbdt = GbdtModel::load(&config.gbdt_model_path)?;
        Ok(Self {
            config,
            gbdt:     Mutex::new(gbdt),
            neural:   Some(Mutex::new(neural)),
            _metrics: metrics,
        })
    }

    /// Phase 1 — BM25F + GBDT only. No cross-encoder Phase 3.
    pub fn bm25_only(config: RankerConfig, metrics: Arc<Metrics>) -> Result<Self> {
        let gbdt = GbdtModel::load(&config.gbdt_model_path)?;
        Ok(Self {
            config,
            gbdt:     Mutex::new(gbdt),
            neural:   None,
            _metrics: metrics,
        })
    }

    pub fn is_gbdt_trained(&self) -> bool {
        self.gbdt.lock().map(|g| g.is_trained()).unwrap_or(false)
    }

    pub fn train(&self, samples: Vec<TrainingSample>) -> Result<()> {
        let mut gbdt = self.gbdt.lock().map_err(|_| Error::LockPoisoned)?;
        gbdt.train_and_save(&samples, &self.config.gbdt_model_path)
    }

    pub fn rank(
        &self,
        hits:      Vec<RawHit>,
        query:     &ParsedQuery,
        pageranks: &PageRankScores,
    ) -> Result<Vec<RankedResult>> {
        if hits.is_empty() { return Ok(Vec::new()); }

        let query_token_count = query.tokens.len();

        let mut scored: Vec<(RawHit, f32)> = {
            let gbdt = self.gbdt.lock().map_err(|_| Error::LockPoisoned)?;
            hits.into_iter().map(|hit| {
                let pagerank   = pageranks.get(&hit.id).copied().unwrap_or(0.0);
                let features   = FeatureVector::from_hit(&hit, pagerank, query_token_count);
                let gbdt_score = gbdt.predict(&features);
                (hit, gbdt_score)
            }).collect()
        };

        scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Phase 3 — only when reranker is wired.
        let rerank_scores: Vec<f32> = match &self.neural {
            Some(neural) => {
                let top_k     = self.config.reranker_top_k.min(scored.len());
                let query_str = query.rewritten.as_str();
                let cands: Vec<&str> = scored[..top_k].iter().map(|(h, _)| h.snippet.as_str()).collect();
                let mut engine = neural.lock().map_err(|_| Error::LockPoisoned)?;
                let scores = engine.rerank(query_str, &cands)
                    .map_err(|err| Error::Neural { reason: err.to_string() })?;
                let mut padded = scores;
                padded.resize(scored.len(), 0.0);
                padded
            }
            None => vec![0.0; scored.len()],
        };

        let results = scored.into_iter().enumerate().map(|(i, (hit, gbdt_score))| {
            let rerank_score = rerank_scores.get(i).copied().unwrap_or(0.0);
            // In BM25-only mode, fall back to gbdt_score (which falls back to BM25
            // when the GBDT model is untrained — see GbdtModel::predict).
            let final_score  = if rerank_score > 0.0 { rerank_score } else { gbdt_score };
            RankedResult {
                id:           hit.id,
                url:          hit.url,
                score:        final_score,
                bm25_score:   hit.score,
                gbdt_score,
                rerank_score,
                snippet:      hit.snippet,
                title:        hit.title,
            }
        }).collect();

        Ok(results)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use raithe_common::{DocumentId, Url};

    fn make_hit(id: u64, score: f32) -> RawHit {
        RawHit {
            id:      DocumentId::new(id),
            url:     Url::parse("https://example.com/").unwrap(),
            score,
            snippet: String::from("example snippet for ranking tests"),
            title:   String::from("Example Title"),
        }
    }

    fn make_samples(n: u64) -> Vec<TrainingSample> {
        (1..=n)
            .map(|i| {
                let hit   = make_hit(i, i as f32 * 0.1);
                let fv    = FeatureVector::from_hit(&hit, i as f32 * 0.01, 3);
                let label = if i > n / 2 { 1.0 } else { 0.0 };
                TrainingSample { features: fv, label }
            })
            .collect()
    }

    #[test]
    fn feature_vector_fields() {
        let hit = make_hit(1, 3.14);
        let fv  = FeatureVector::from_hit(&hit, 0.5, 3);
        assert!((fv.bm25_score - 3.14).abs() < 1e-5);
        assert!((fv.pagerank   - 0.5 ).abs() < 1e-5);
        assert_eq!(fv.query_tokens as usize, 3);
    }

    #[test]
    fn feature_vec_has_five_elements() {
        let hit = make_hit(1, 1.0);
        let fv  = FeatureVector::from_hit(&hit, 0.3, 4);
        assert_eq!(fv.as_vec().len(), 5);
    }

    #[test]
    fn untrained_model_returns_bm25() {
        let model = GbdtModel::untrained();
        let hit   = make_hit(1, 2.71);
        let fv    = FeatureVector::from_hit(&hit, 0.0, 1);
        assert!((model.predict(&fv) - 2.71).abs() < 1e-5);
        assert!(!model.is_trained());
    }

    #[test]
    fn train_and_predict() {
        let dir   = tempfile::tempdir().unwrap();
        let path  = dir.path().join("gbdt.model");
        let mut model = GbdtModel::untrained();
        model.train_and_save(&make_samples(20), &path).unwrap();
        assert!(model.is_trained());
        assert!(path.exists());
        let hit = make_hit(5, 0.5);
        let fv  = FeatureVector::from_hit(&hit, 0.05, 3);
        let _   = model.predict(&fv);
    }

    #[test]
    fn load_saved_model() {
        let dir   = tempfile::tempdir().unwrap();
        let path  = dir.path().join("gbdt.model");
        let mut model = GbdtModel::untrained();
        model.train_and_save(&make_samples(20), &path).unwrap();
        let loaded = GbdtModel::load(&path).unwrap();
        assert!(loaded.is_trained());
    }

    #[test]
    fn missing_model_file_gives_untrained() {
        let dir  = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.model");
        let model = GbdtModel::load(&path).unwrap();
        assert!(!model.is_trained());
    }

    #[test]
    fn empty_samples_returns_error() {
        let dir   = tempfile::tempdir().unwrap();
        let path  = dir.path().join("gbdt.model");
        let mut model = GbdtModel::untrained();
        assert!(model.train_and_save(&[], &path).is_err());
    }
}