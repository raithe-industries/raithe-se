// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/neural/src/lib.rs
//
// Hardware-agnostic ONNX Runtime inference — split into three engines.
//
// The previous monolithic `NeuralEngine` loaded all three models per
// instance. With each consumer (query / ranker / indexer) constructing its
// own, the generator's 30 GB fp32 weights were read 3× at startup.
//
// Now each consumer loads exactly one model: `GenerateEngine` for the
// Query processor, `RerankEngine` for the Ranker, `EmbedEngine` for the
// indexing pipeline. No shared state, no mutex contention between models,
// and cold start is bounded by the single heaviest model (generator).
//
// ORT init: ORT_DYLIB_PATH is set by raithe.sh before exec — no explicit
// init_from call is made (it deadlocks on Linux with NVIDIA drivers via
// libcuda.so constructor). CPU EP is the unconditional backstop.
//
// Graph-optimization level is pinned to `Level1` on load. ORT's Level3
// runs expensive layout / fusion passes that dominate cold start on
// large decoder models and produce hardware-specific artifacts that are
// not safely cacheable across machines. Level1 is portable and avoids
// the worst of the pass cost.

use std::path::Path;
use std::sync::Arc;

use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProviderDispatch,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use raithe_common::Embedding;
use raithe_config::NeuralConfig;
use raithe_metrics::Metrics;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("model not found — expected both model.onnx and tokenizer.json in '{path}'")]
    ModelNotFound { path: String },
    #[error("ONNX Runtime error loading '{path}': {reason}")]
    OrtLoad { path: String, reason: String },
    #[error("tokeniser error for '{path}': {reason}")]
    Tokeniser { path: String, reason: String },
    #[error("inference error ({model}): {reason}")]
    Inference { model: String, reason: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// The active ONNX Runtime execution provider.
///
/// Selected once at startup in priority order: CUDA → DirectML → CoreML → CPU.
/// Exposed as the `raithe_neural_execution_provider` Prometheus gauge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionProvider {
    Cuda,
    DirectMl,
    CoreMl,
    Cpu,
}

impl ExecutionProvider {
    /// Returns the Prometheus label string for the given provider.
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda     => "cuda",
            Self::DirectMl => "directml",
            Self::CoreMl   => "coreml",
            Self::Cpu      => "cpu",
        }
    }
}

/// A loaded ONNX model handle with its associated vocabulary tokeniser.
struct ModelHandle {
    session:   Session,
    tokeniser: tokenizers::Tokenizer,
}

impl std::fmt::Debug for ModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHandle")
            .finish_non_exhaustive()
    }
}

// ── EmbedEngine ───────────────────────────────────────────────────────────────

/// Dense-embedding engine (BGE-large-en-v1.5).
///
/// Loads exactly one model: the embedder. Constructed once and held by
/// the indexing pipeline for the lifetime of the process.
pub struct EmbedEngine {
    handle:   ModelHandle,
    provider: ExecutionProvider,
    _metrics: Arc<Metrics>,
}

impl EmbedEngine {
    /// Probes execution providers against the embedder model and loads it.
    pub fn new(config: &NeuralConfig, metrics: Arc<Metrics>) -> Result<Self> {
        validate_ort_dylib(config)?;

        let model_path = config.embedder_dir.join("model.onnx");
        let provider   = probe_best_provider(&model_path);
        record_provider_gauge(&metrics, provider);

        let handle = load_model(&config.embedder_dir, provider)?;
        Ok(Self { handle, provider, _metrics: metrics })
    }

    /// Returns the active execution provider selected at startup.
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    /// Encodes `texts` into 1024-dimensional dense embeddings.
    ///
    /// Uses the BGE-large-en-v1.5 bi-encoder. Returns `Error::Inference`
    /// if the model output dimensionality does not equal `Embedding::DIM`.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let outputs = run_session(
            &mut self.handle.session,
            &self.handle.tokeniser,
            texts,
            "embedder",
        )?;

        outputs
            .into_iter()
            .map(|vec| {
                if vec.len() != Embedding::DIM {
                    return Err(Error::Inference {
                        model:  String::from("embedder"),
                        reason: format!(
                            "expected {} dims, got {}",
                            Embedding::DIM,
                            vec.len()
                        ),
                    });
                }
                Ok(Embedding::new(vec))
            })
            .collect()
    }
}

// ── RerankEngine ──────────────────────────────────────────────────────────────

/// Cross-encoder re-ranking engine (BGE-reranker-large).
///
/// Loads exactly one model: the reranker. Constructed once and held by
/// the Ranker for the lifetime of the process.
pub struct RerankEngine {
    handle:   ModelHandle,
    provider: ExecutionProvider,
    metrics:  Arc<Metrics>,
}

impl RerankEngine {
    /// Probes execution providers against the reranker model and loads it.
    pub fn new(config: &NeuralConfig, metrics: Arc<Metrics>) -> Result<Self> {
        validate_ort_dylib(config)?;

        let model_path = config.reranker_dir.join("model.onnx");
        let provider   = probe_best_provider(&model_path);
        record_provider_gauge(&metrics, provider);

        let handle = load_model(&config.reranker_dir, provider)?;
        Ok(Self { handle, provider, metrics })
    }

    /// Returns the active execution provider selected at startup.
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    /// Scores `candidates` against `query` using the BGE-reranker-large
    /// cross-encoder.
    ///
    /// Returns one f32 relevance score per candidate, in input order.
    /// Increments `raithe_rank_phase3_calls_total`.
    pub fn rerank(&mut self, query: &str, candidates: &[&str]) -> Result<Vec<f32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let pairs: Vec<String> = candidates
            .iter()
            .map(|c| format!("{query} [SEP] {c}"))
            .collect();
        let pair_strs: Vec<&str> = pairs.iter().map(String::as_str).collect();

        let outputs = run_session(
            &mut self.handle.session,
            &self.handle.tokeniser,
            &pair_strs,
            "reranker",
        )?;

        let scores = outputs
            .into_iter()
            .map(|vec| vec.into_iter().next().unwrap_or(0.0))
            .collect();

        self.metrics.rank_phase3_calls_total.inc();

        Ok(scores)
    }
}

// ── GenerateEngine ────────────────────────────────────────────────────────────

/// Causal-LM generation engine (Qwen2.5-{7B,14B}-Instruct).
///
/// Loads exactly one model: the generator. Constructed once and held by
/// the QueryProcessor for the lifetime of the process. Dominates cold-
/// start cost; holds ~14–28 GB of weights resident per instance.
pub struct GenerateEngine {
    handle:   ModelHandle,
    provider: ExecutionProvider,
    max_new:  usize,
    _metrics: Arc<Metrics>,
}

impl GenerateEngine {
    /// Probes execution providers against the generator model and loads it.
    pub fn new(config: &NeuralConfig, metrics: Arc<Metrics>) -> Result<Self> {
        validate_ort_dylib(config)?;

        let model_path = config.generator_dir.join("model.onnx");
        let provider   = probe_best_provider(&model_path);
        record_provider_gauge(&metrics, provider);

        let handle = load_model(&config.generator_dir, provider)?;
        Ok(Self {
            handle,
            provider,
            max_new:  config.generator_max_tokens,
            _metrics: metrics,
        })
    }

    /// Returns the active execution provider selected at startup.
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    /// Generates a rewritten query string using the Qwen2.5 generator.
    ///
    /// Greedy autoregressive decode (DEF-003): feed the prompt, take the
    /// argmax over the final-step logits, append the token, and repeat
    /// until EOS or `NeuralConfig::generator_max_tokens` new tokens have
    /// been produced. No KV cache — the whole sequence is re-fed on
    /// every step. Correct for any Qwen2.5 ONNX export; a KV-cache
    /// optimisation is a later, purely internal improvement.
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        let encoding = self.handle.tokeniser
            .encode(prompt, true)
            .map_err(|err| Error::Tokeniser {
                path:   String::from("generator"),
                reason: err.to_string(),
            })?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();
        let eos_id     = self.handle.tokeniser
            .token_to_id("<|endoftext|>")
            .or_else(|| self.handle.tokeniser.token_to_id("<|im_end|>"));

        for _ in 0..self.max_new {
            let next = generator_step(&mut self.handle.session, &tokens)?;
            tokens.push(next);
            if Some(next) == eos_id {
                break;
            }
        }

        // Decode only the newly generated tail so the prompt is not echoed.
        let generated: &[u32] = tokens.get(prompt_len..).unwrap_or(&[]);
        self.handle.tokeniser
            .decode(generated, true)
            .map_err(|err| Error::Inference {
                model:  String::from("generator"),
                reason: err.to_string(),
            })
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Validates ORT_DYLIB_PATH is resolvable.
///
/// raithe.sh sets this before exec. Explicit init_from is not called here
/// because it spawns ORT's global inter-op thread pool in a constructor
/// that deadlocks on Linux systems with NVIDIA drivers present via
/// libcuda.so. With `load-dynamic`, ORT resolves the library path from
/// this env var on first session creation.
fn validate_ort_dylib(config: &NeuralConfig) -> Result<()> {
    if config.ort_dylib_path.as_os_str().is_empty()
        && std::env::var("ORT_DYLIB_PATH").unwrap_or_default().is_empty()
    {
        return Err(Error::OrtLoad {
            path:   String::from("(none)"),
            reason: String::from(
                "ORT_DYLIB_PATH not set — launch via raithe.sh which \
                 locates and exports the ORT shared library automatically",
            ),
        });
    }
    Ok(())
}

/// Records the selected EP in the Prometheus gauge — selected = 1.0, others = 0.0.
fn record_provider_gauge(metrics: &Metrics, provider: ExecutionProvider) {
    for label in &["cuda", "directml", "coreml", "cpu"] {
        let value = if *label == provider.label() { 1.0 } else { 0.0 };
        metrics
            .neural_execution_provider
            .with_label_values(&[label])
            .set(value);
    }
}

/// Loads a model from `dir`, resolving `model.onnx` and `tokenizer.json`.
///
/// The selected `provider` must already have survived `probe_best_provider`.
/// Both `model.onnx` and `tokenizer.json` must be present in `dir`; missing
/// either returns `Error::ModelNotFound`.
///
/// Graph optimization is pinned to `Level1` (basic, portable passes only).
/// Level3 runs expensive layout-transform passes that dominate cold start
/// on 7B+ decoder models; `Level1` preserves correctness and avoids the
/// pass cost. Upgrade individually per-engine later if benchmarks justify.
fn load_model(dir: &Path, provider: ExecutionProvider) -> Result<ModelHandle> {
    let model_path     = dir.join("model.onnx");
    let tokeniser_path = dir.join("tokenizer.json");

    if !model_path.exists() || !tokeniser_path.exists() {
        return Err(Error::ModelNotFound {
            path: dir.display().to_string(),
        });
    }

    let eps: Vec<ExecutionProviderDispatch> = execution_providers_for(provider);

    let session = Session::builder()
        .map_err(|err| Error::OrtLoad {
            path:   dir.display().to_string(),
            reason: err.to_string(),
        })?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|err| Error::OrtLoad {
            path:   dir.display().to_string(),
            reason: err.to_string(),
        })?
        .with_execution_providers(eps)
        .map_err(|err| Error::OrtLoad {
            path:   dir.display().to_string(),
            reason: err.to_string(),
        })?
        .commit_from_file(&model_path)
        .map_err(|err| Error::OrtLoad {
            path:   model_path.display().to_string(),
            reason: err.to_string(),
        })?;

    let tokeniser = tokenizers::Tokenizer::from_file(&tokeniser_path)
        .map_err(|err| Error::Tokeniser {
            path:   tokeniser_path.display().to_string(),
            reason: err.to_string(),
        })?;

    Ok(ModelHandle { session, tokeniser })
}

/// Returns the list of execution provider dispatches for the selected
/// provider, always terminated with CPU as the unconditional backstop.
fn execution_providers_for(provider: ExecutionProvider) -> Vec<ExecutionProviderDispatch> {
    match provider {
        ExecutionProvider::Cuda     => vec![
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ],
        ExecutionProvider::DirectMl => vec![
            DirectMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ],
        ExecutionProvider::CoreMl   => vec![
            CoreMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ],
        ExecutionProvider::Cpu      => vec![
            CPUExecutionProvider::default().build(),
        ],
    }
}

/// Execution-provider probe (§7 / v1.8 §5.11 / DEF-002).
///
/// Each candidate is tried in priority order against `model_path` inside a
/// bounded-duration thread. The first candidate that commits a session
/// within the budget wins; everything else falls through to CPU.
///
/// The per-candidate budget is intentionally small — spec §5.11 says
/// unavailable providers fail in microseconds. A budget of a few seconds
/// gives real failure paths time to unwind without letting pathological
/// registration blocks (e.g. libcuda.so with a missing toolkit) hang the
/// whole process.
fn probe_best_provider(model_path: &Path) -> ExecutionProvider {
    const PROBE_BUDGET: std::time::Duration = std::time::Duration::from_secs(3);

    for candidate in [
        ExecutionProvider::Cuda,
        ExecutionProvider::DirectMl,
        ExecutionProvider::CoreMl,
    ] {
        if probe_provider(candidate, model_path, PROBE_BUDGET) {
            return candidate;
        }
    }

    ExecutionProvider::Cpu
}

/// Exercises `candidate` against `model_path` with a hard timeout.
///
/// Returns `true` if a session builds with the candidate as the primary EP
/// within `budget`; `false` otherwise. Any panic, error, or timeout is
/// treated as "unavailable."
fn probe_provider(
    candidate:  ExecutionProvider,
    model_path: &Path,
    budget:     std::time::Duration,
) -> bool {
    if !model_path.exists() {
        return false;
    }

    let path = model_path.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel::<bool>();

    std::thread::spawn(move || {
        let eps = execution_providers_for(candidate);
        let ok  = probe_once(eps, &path).is_ok();
        let _ = tx.send(ok);
    });

    rx.recv_timeout(budget).unwrap_or(false)
}

/// Attempts one end-to-end session build against `path` with the given EPs.
///
/// Returns `Error::OrtLoad` on any builder-stage failure. Called from
/// `probe_provider` through a short-lived thread; only `.is_ok()` is tested
/// at the call site.
///
/// # Why not `Box<dyn Error + Send + Sync>`?
///
/// ort 2.0.0-rc.12's `Error<SessionBuilder>` is `!Send + !Sync` because
/// `SessionBuilder` internally holds an `Rc`. Returning a boxed-trait
/// error would need to coerce that into `Box<dyn Error + Send + Sync>`,
/// which the compiler rejects. Using the crate-local `Error` type (which
/// carries stringified reasons, not source errors) sidesteps the bound
/// entirely.
fn probe_once(eps: Vec<ExecutionProviderDispatch>, path: &Path) -> Result<Session> {
    let path_str = path.display().to_string();
    Session::builder()
        .map_err(|e| Error::OrtLoad {
            path:   path_str.clone(),
            reason: e.to_string(),
        })?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|e| Error::OrtLoad {
            path:   path_str.clone(),
            reason: e.to_string(),
        })?
        .with_execution_providers(eps)
        .map_err(|e| Error::OrtLoad {
            path:   path_str.clone(),
            reason: e.to_string(),
        })?
        .commit_from_file(path)
        .map_err(|e| Error::OrtLoad {
            path:   path_str,
            reason: e.to_string(),
        })
}

/// Runs one forward pass over `texts` through `session`, returning raw
/// output vectors per input. Used by both `EmbedEngine` and `RerankEngine`.
fn run_session(
    session: &mut Session,
    tokeniser: &tokenizers::Tokenizer,
    texts: &[&str],
    model: &str,
) -> Result<Vec<Vec<f32>>> {
    let encodings: Vec<tokenizers::Encoding> = texts
        .iter()
        .map(|text| {
            tokeniser
                .encode(*text, true)
                .map_err(|err| Error::Tokeniser {
                    path:   model.to_owned(),
                    reason: err.to_string(),
                })
        })
        .collect::<Result<_>>()?;

    // Right-pad every sequence to the longest in the batch so the
    // [batch, seq] input tensor is rectangular.
    let max_seq = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
    let batch   = encodings.len();

    let ids_flat: Vec<i64> = encodings
        .iter()
        .flat_map(|e| {
            let row = e.get_ids();
            let pad = max_seq.saturating_sub(row.len());
            row.iter().map(|&id| id as i64).chain(std::iter::repeat_n(0i64, pad))
        })
        .collect();

    let mask_flat: Vec<i64> = encodings
        .iter()
        .flat_map(|e| {
            let row = e.get_attention_mask();
            let pad = max_seq.saturating_sub(row.len());
            row.iter().map(|&m| m as i64).chain(std::iter::repeat_n(0i64, pad))
        })
        .collect();

    let ids_tensor = Tensor::<i64>::from_array(([batch, max_seq], ids_flat))
        .map_err(|err| Error::Inference {
            model:  model.to_owned(),
            reason: err.to_string(),
        })?;
    let mask_tensor = Tensor::<i64>::from_array(([batch, max_seq], mask_flat))
        .map_err(|err| Error::Inference {
            model:  model.to_owned(),
            reason: err.to_string(),
        })?;

    let outputs = session
        .run(ort::inputs![
            "input_ids"      => ids_tensor,
            "attention_mask" => mask_tensor
        ])
        .map_err(|err| Error::Inference {
            model:  model.to_owned(),
            reason: err.to_string(),
        })?;

    let first = outputs
        .values()
        .next()
        .ok_or_else(|| Error::Inference {
            model:  model.to_owned(),
            reason: String::from("no output tensors returned"),
        })?;

    let (shape, slice) = first
        .try_extract_tensor::<f32>()
        .map_err(|err| Error::Inference {
            model:  model.to_owned(),
            reason: err.to_string(),
        })?;

    // Output shape is [batch, features] (reranker logit, or pooled
    // embedder) or [batch, seq, features] (token-level embedder that
    // requires mean-pooling here).
    match shape.len() {
        2 => {
            let feature_dim = shape[1] as usize;
            let mut rows: Vec<Vec<f32>> = Vec::with_capacity(batch);
            for row_idx in 0..batch {
                let start = row_idx * feature_dim;
                let end   = start + feature_dim;
                let row   = slice.get(start..end).ok_or_else(|| Error::Inference {
                    model:  model.to_owned(),
                    reason: String::from("output slice shorter than shape implies"),
                })?;
                rows.push(row.to_vec());
            }
            Ok(rows)
        }
        3 => {
            let seq         = shape[1] as usize;
            let feature_dim = shape[2] as usize;
            let mut rows: Vec<Vec<f32>> = Vec::with_capacity(batch);
            for (row_idx, enc) in encodings.iter().enumerate().take(batch) {
                let mut pooled = vec![0.0f32; feature_dim];
                let mut n      = 0.0f32;
                for t in 0..seq {
                    let mask_v = enc.get_attention_mask().get(t).copied().unwrap_or(0);
                    if mask_v == 0 {
                        continue;
                    }
                    let base = row_idx * seq * feature_dim + t * feature_dim;
                    let seg  = slice.get(base..base + feature_dim).ok_or_else(|| Error::Inference {
                        model:  model.to_owned(),
                        reason: String::from("output slice shorter than shape implies"),
                    })?;
                    for (p, v) in pooled.iter_mut().zip(seg) {
                        *p += *v;
                    }
                    n += 1.0;
                }
                if n > 0.0 {
                    for p in pooled.iter_mut() {
                        *p /= n;
                    }
                }
                rows.push(pooled);
            }
            Ok(rows)
        }
        _ => Err(Error::Inference {
            model:  model.to_owned(),
            reason: format!("expected 2-D or 3-D output, got shape {shape:?}"),
        }),
    }
}

/// One decode step — feed `tokens`, return argmax over the final-position logits.
///
/// Called from `GenerateEngine::generate` (DEF-003). No KV cache — full
/// sequence is re-fed each step; correct for any Qwen2.5 ONNX export.
fn generator_step(session: &mut Session, tokens: &[u32]) -> Result<u32> {
    let seq = tokens.len();
    if seq == 0 {
        return Err(Error::Inference {
            model:  String::from("generator"),
            reason: String::from("cannot step on an empty token sequence"),
        });
    }

    let ids:   Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    let masks: Vec<i64> = vec![1i64; seq];

    let ids_tensor = Tensor::<i64>::from_array(([1usize, seq], ids))
        .map_err(|err| Error::Inference {
            model:  String::from("generator"),
            reason: err.to_string(),
        })?;

    let mask_tensor = Tensor::<i64>::from_array(([1usize, seq], masks))
        .map_err(|err| Error::Inference {
            model:  String::from("generator"),
            reason: err.to_string(),
        })?;

    let outputs = session
        .run(ort::inputs![
            "input_ids"      => ids_tensor,
            "attention_mask" => mask_tensor
        ])
        .map_err(|err| Error::Inference {
            model:  String::from("generator"),
            reason: err.to_string(),
        })?;

    let logits_value = outputs
        .values()
        .next()
        .ok_or_else(|| Error::Inference {
            model:  String::from("generator"),
            reason: String::from("no output tensors returned"),
        })?;

    let (shape, slice) = logits_value
        .try_extract_tensor::<f32>()
        .map_err(|err| Error::Inference {
            model:  String::from("generator"),
            reason: err.to_string(),
        })?;

    // Qwen2.5 causal-LM head emits logits of shape [batch=1, seq, vocab].
    // Argmax the final-step slice: logits[0, seq-1, :].
    if shape.len() != 3 {
        return Err(Error::Inference {
            model:  String::from("generator"),
            reason: format!("expected 3-D logits, got shape {shape:?}"),
        });
    }
    let step_seq = shape[1] as usize;
    let vocab    = shape[2] as usize;
    if step_seq == 0 || vocab == 0 {
        return Err(Error::Inference {
            model:  String::from("generator"),
            reason: format!("degenerate logits shape {shape:?}"),
        });
    }

    let last_start = (step_seq - 1) * vocab;
    let last_end   = last_start + vocab;
    let last_logits = slice.get(last_start..last_end).ok_or_else(|| Error::Inference {
        model:  String::from("generator"),
        reason: String::from("logits slice shorter than shape implies"),
    })?;

    let next = argmax_u32(last_logits).ok_or_else(|| Error::Inference {
        model:  String::from("generator"),
        reason: String::from("empty final-step logits"),
    })?;

    Ok(next)
}

/// Returns the index of the maximum value in `values`, or `None` when empty.
fn argmax_u32(values: &[f32]) -> Option<u32> {
    let mut best_idx: usize = 0;
    let mut best_val: f32   = f32::NEG_INFINITY;
    let mut seen            = false;
    for (i, &v) in values.iter().enumerate() {
        if !seen || v > best_val {
            best_idx = i;
            best_val = v;
            seen     = true;
        }
    }
    seen.then_some(best_idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_provider_labels_are_correct() {
        assert_eq!(ExecutionProvider::Cuda.label(),     "cuda");
        assert_eq!(ExecutionProvider::DirectMl.label(), "directml");
        assert_eq!(ExecutionProvider::CoreMl.label(),   "coreml");
        assert_eq!(ExecutionProvider::Cpu.label(),      "cpu");
    }

    #[test]
    #[ignore = "requires ORT shared library — run manually: cargo test -p raithe-neural -- --ignored"]
    fn provider_is_cpu_without_cuda_toolkit() {
        // On systems without CUDA toolkit ≥ 12.8, the active provider is CPU.
        assert_eq!(ExecutionProvider::Cpu.label(), "cpu");
    }

    #[test]
    fn model_not_found_when_dir_empty() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_model(dir.path(), ExecutionProvider::Cpu).unwrap_err();
        assert!(matches!(err, Error::ModelNotFound { .. }));
    }

    #[test]
    fn model_not_found_returns_err_variant() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_model(dir.path(), ExecutionProvider::Cpu);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ModelNotFound { .. }));
    }

    #[test]
    fn argmax_u32_returns_index_of_max() {
        assert_eq!(argmax_u32(&[0.1, 0.9, 0.3]), Some(1));
        assert_eq!(argmax_u32(&[-5.0, -1.0, -3.0]), Some(1));
    }

    #[test]
    fn argmax_u32_handles_first_position() {
        assert_eq!(argmax_u32(&[5.0, 1.0, 2.0]), Some(0));
    }

    #[test]
    fn argmax_u32_handles_negatives() {
        // Every value is negative; must still return a valid index, not
        // skip to None because the default f32::NEG_INFINITY would match.
        assert_eq!(argmax_u32(&[-10.0, -20.0, -5.0]), Some(2));
    }

    #[test]
    fn argmax_u32_none_when_empty() {
        assert_eq!(argmax_u32(&[]), None);
    }
}