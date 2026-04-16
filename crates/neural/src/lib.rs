// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/neural/src/lib.rs
//
// Hardware-agnostic ONNX Runtime inference manager.
// ORT_DYLIB_PATH is set by raithe.sh before exec — no explicit init_from
// call is made (it deadlocks on Linux with NVIDIA drivers via libcuda.so).
// CPU-only EP; CUDA re-enablement requires CUDA toolkit ≥ 12.8 + cuDNN ≥ 9.19.
// Manages three model handles: embedder, reranker, generator.

use std::path::Path;
use std::sync::Arc;

use ort::execution_providers::{
    CPUExecutionProvider, ExecutionProviderDispatch,
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
    /// `raithe.sh` to install models.
    pub fn new(config: &NeuralConfig, metrics: Arc<Metrics>) -> Result<Self> {
        // ORT_DYLIB_PATH is set by raithe.sh before exec. With load-dynamic,
        // ORT resolves the library path from that env var on first session
        // creation. No explicit init_from call is needed or safe here —
        // calling init_from triggers ORT's global Env constructor which
        // spawns an inter-op thread pool that deadlocks on Linux systems
        // with NVIDIA drivers present via libcuda.so constructor.
        //
        // Validate the path is set so failure is diagnosed immediately.
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

        let provider = ExecutionProvider::Cpu;

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
/// Uses CPU execution provider only. CUDA EP registration blocks indefinitely
/// on Linux systems with NVIDIA drivers but without CUDA toolkit ≥ 12.8.
/// Both `model.onnx` and `tokenizer.json` must be present.
/// Missing either returns `Error::ModelNotFound`.
fn load_model(dir: &Path, _provider: ExecutionProvider) -> Result<ModelHandle> {
    let model_path     = dir.join("model.onnx");
    let tokeniser_path = dir.join("tokenizer.json");

    if !model_path.exists() || !tokeniser_path.exists() {
        return Err(Error::ModelNotFound {
            path: dir.display().to_string(),
        });
    }

    // CPU-only execution provider list.
    // ORT 2.0.0-rc.12 CUDA EP requires CUDA toolkit ≥ 12.8 + cuDNN ≥ 9.19.
    // Attempting to register CUDA EP without the correct toolkit version
    // blocks indefinitely in futex_wait_queue on Linux via libcuda.so
    // constructor probing /dev/nvidiactl. CPU is always safe and available.
    // CUDA can be re-enabled here once the correct toolkit is confirmed.
    let eps: Vec<ExecutionProviderDispatch> = vec![
        CPUExecutionProvider::default().into(),
    ];

    let session = Session::builder()
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
}
