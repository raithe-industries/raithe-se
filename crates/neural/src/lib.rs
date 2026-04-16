// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/neural/src/lib.rs
//
// Hardware-agnostic ONNX Runtime inference manager.
// Auto-detects the best execution provider at startup (CUDA → DirectML →
// CoreML → CPU). Manages three model handles: embedder, reranker, generator.

use std::path::Path;
use std::sync::Arc;

use ort::execution_providers::{
    CUDAExecutionProvider, CPUExecutionProvider, ExecutionProviderDispatch,
};
use ort::session::Session;
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

/// Hardware-agnostic ONNX Runtime inference manager.
///
/// Constructed once at startup. All three models are loaded and held for the
/// lifetime of the process. The active `ExecutionProvider` is selected by
/// probing in priority order — first successful init wins.
pub struct NeuralEngine {
    embedder:  ModelHandle,
    reranker:  ModelHandle,
    generator: ModelHandle,
    provider:  ExecutionProvider,
    metrics:   Arc<Metrics>,
}

impl NeuralEngine {
    /// Probes execution providers and loads all three ONNX models.
    ///
    /// Initialises the ONNX Runtime shared library from
    /// `config.ort_dylib_path` if set, otherwise falls back to the
    /// `ORT_DYLIB_PATH` environment variable. With `load-dynamic`, this
    /// must complete before any session is created.
    ///
    /// Returns `Error::ModelNotFound` if `model.onnx` or `tokenizer.json` are
    /// absent from any model directory. No silent fallback — run
    /// `install_models.sh` to populate the directories.
    pub fn new(config: &NeuralConfig, metrics: Arc<Metrics>) -> Result<Self> {
        // ORT's global environment is a process-wide singleton.
        // init_from must be called exactly once — subsequent calls on an
        // already-initialised environment deadlock on ORT's internal mutex.
        // Use Once to guarantee single initialisation regardless of how many
        // NeuralEngine instances are constructed.
        static ORT_INIT: std::sync::Once = std::sync::Once::new();
        static ORT_INIT_RESULT: std::sync::OnceLock<Option<String>> =
            std::sync::OnceLock::new();

        ORT_INIT.call_once(|| {
            let dylib_path = if !config.ort_dylib_path.as_os_str().is_empty() {
                config.ort_dylib_path.to_string_lossy().into_owned()
            } else {
                std::env::var("ORT_DYLIB_PATH").unwrap_or_default()
            };

            let err = if dylib_path.is_empty() {
                Some(String::from(
                    "cannot locate libonnxruntime.so — set ort_dylib_path in \
                     engine.toml or run via run.sh which sets ORT_DYLIB_PATH automatically",
                ))
            } else {
                match ort::init_from(dylib_path.as_str()) {
                    Ok(builder) => { builder.commit(); None }
                    Err(err)    => Some(err.to_string()),
                }
            };

            ORT_INIT_RESULT.set(err).ok();
        });

        if let Some(Some(reason)) = ORT_INIT_RESULT.get() {
            return Err(Error::OrtLoad {
                path:   config.ort_dylib_path.display().to_string(),
                reason: reason.clone(),
            });
        }

        let provider = probe_execution_provider();

        record_provider_gauge(&metrics, provider);

        let embedder  = load_model(&config.embedder_dir,  provider)?;
        let reranker  = load_model(&config.reranker_dir,  provider)?;
        let generator = load_model(&config.generator_dir, provider)?;

        Ok(Self {
            embedder,
            reranker,
            generator,
            provider,
            metrics,
        })
    }

    /// Returns the active execution provider selected at startup.
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    /// Encodes `texts` into 1024-dimensional dense embeddings.
    ///
    /// Uses the BGE-large-en-v1.5 bi-encoder. Returns `Error::Inference` if
    /// the model output dimensionality does not equal `Embedding::DIM`.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let outputs = run_session(
            &mut self.embedder.session,
            &self.embedder.tokeniser,
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
            &mut self.reranker.session,
            &self.reranker.tokeniser,
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

    /// Generates a rewritten query string using the Qwen2.5 generator.
    ///
    /// Used by `QueryProcessor` for LLM-assisted query understanding.
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        let outputs = run_session(
            &mut self.generator.session,
            &self.generator.tokeniser,
            &[prompt],
            "generator",
        )?;

        let tokens: Vec<u32> = outputs
            .into_iter()
            .flat_map(|v| v.into_iter().map(|f| f as u32))
            .collect();

        self.generator
            .tokeniser
            .decode(&tokens, true)
            .map_err(|err| Error::Inference {
                model:  String::from("generator"),
                reason: err.to_string(),
            })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Probes execution providers in priority order, returning the first available.
///
/// Attempts to configure a `Session::builder` with each EP candidate in turn.
/// Builder-level validation is fast (sub-millisecond) — it does not load a
/// model or block on hardware driver initialisation. CPU never fails.
fn probe_execution_provider() -> ExecutionProvider {
    // CUDA — Linux and Windows only.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        let ok = (|| -> ort::Result<()> {
            Session::builder()?.with_execution_providers([
                CUDAExecutionProvider::default().into(),
            ])?;
            Ok(())
        })()
        .is_ok();
        if ok {
            return ExecutionProvider::Cuda;
        }
    }

    // DirectML — Windows only.
    #[cfg(target_os = "windows")]
    {
        let ok = (|| -> ort::Result<()> {
            Session::builder()?.with_execution_providers([
                ort::execution_providers::DirectMLExecutionProvider::default().into(),
            ])?;
            Ok(())
        })()
        .is_ok();
        if ok {
            return ExecutionProvider::DirectMl;
        }
    }

    // CoreML — macOS and iOS only.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let ok = (|| -> ort::Result<()> {
            Session::builder()?.with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().into(),
            ])?;
            Ok(())
        })()
        .is_ok();
        if ok {
            return ExecutionProvider::CoreMl;
        }
    }

    ExecutionProvider::Cpu
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
/// Both files must be present — missing either returns `Error::ModelNotFound`.
fn load_model(dir: &Path, provider: ExecutionProvider) -> Result<ModelHandle> {
    let model_path     = dir.join("model.onnx");
    let tokeniser_path = dir.join("tokenizer.json");

    if !model_path.exists() || !tokeniser_path.exists() {
        return Err(Error::ModelNotFound {
            path: dir.display().to_string(),
        });
    }

    let ep: ExecutionProviderDispatch = match provider {
        ExecutionProvider::Cuda     => CUDAExecutionProvider::default().into(),
        ExecutionProvider::DirectMl => {
            #[cfg(target_os = "windows")]
            { ort::execution_providers::DirectMLExecutionProvider::default().into() }
            #[cfg(not(target_os = "windows"))]
            { CPUExecutionProvider::default().into() }
        }
        ExecutionProvider::CoreMl => {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            { ort::execution_providers::CoreMLExecutionProvider::default().into() }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            { CPUExecutionProvider::default().into() }
        }
        ExecutionProvider::Cpu => CPUExecutionProvider::default().into(),
    };

    let session = Session::builder()
        .map_err(|err| Error::OrtLoad {
            path:   dir.display().to_string(),
            reason: err.to_string(),
        })?
        .with_execution_providers([ep])
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

/// Tokenises `texts`, runs ONNX inference, and returns raw f32 output vectors.
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

    let batch   = encodings.len();
    let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

    let padded_ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| {
            let row = e.get_ids();
            let pad = max_len - row.len();
            row.iter().map(|&id| id as i64).chain(std::iter::repeat(0i64).take(pad))
        })
        .collect();

    let padded_masks: Vec<i64> = encodings
        .iter()
        .flat_map(|e| {
            let row = e.get_attention_mask();
            let pad = max_len - row.len();
            row.iter().map(|&m| m as i64).chain(std::iter::repeat(0i64).take(pad))
        })
        .collect();

    let ids_tensor = Tensor::<i64>::from_array(([batch, max_len], padded_ids))
        .map_err(|err| Error::Inference {
            model:  model.to_owned(),
            reason: err.to_string(),
        })?;

    let mask_tensor = Tensor::<i64>::from_array(([batch, max_len], padded_masks))
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

    let output_value = outputs
        .values()
        .next()
        .ok_or_else(|| Error::Inference {
            model:  model.to_owned(),
            reason: String::from("no output tensors returned"),
        })?;

    let (shape, slice) = output_value
        .try_extract_tensor::<f32>()
        .map_err(|err| Error::Inference {
            model:  model.to_owned(),
            reason: err.to_string(),
        })?;

    let cols = if shape.len() >= 2 { shape[1] as usize } else { slice.len() / batch };

    let result = (0..batch)
        .map(|i| slice.iter().skip(i * cols).take(cols).copied().collect())
        .collect();

    Ok(result)
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
    #[ignore = "probes ORT execution providers via FFI — run manually: cargo test -p raithe-neural -- --ignored"]
    fn probe_returns_a_valid_provider() {
        let provider = probe_execution_provider();
        assert!(["cuda", "directml", "coreml", "cpu"].contains(&provider.label()));
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
}
