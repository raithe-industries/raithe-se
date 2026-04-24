// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/neural.rs

use std::path::PathBuf;

/// Weight-quantization format for an ONNX model on disk.
///
/// Drives both the raithe.sh export path and the runtime's expectation of
/// ops the model will use (e.g. `MatMulNBits` for int4). Written to the
/// per-model `quant.txt` sidecar so a mismatched re-config triggers
/// re-export instead of a silent wrong-format load.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// 32-bit float, no quantization. Highest quality, largest file.
    #[default]
    Fp32,
    /// 16-bit float. Halves weight footprint vs fp32 with negligible
    /// quality loss. CUDA-friendly on every consumer GPU since Pascal.
    Fp16,
    /// ORT-native block-quantized int4 via the `MatMulNBits` op.
    /// Exported with `optimum-cli --weight-format int4`. Quality loss on
    /// Qwen2.5 is ~1–2% on query-rewriting tasks; file shrinks ~8×.
    Int4,
}

impl Quantization {
    /// Returns the lowercase label written to disk and used for matching.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Int4 => "int4",
        }
    }
}

/// Execution provider requested for a specific engine.
///
/// Unlike the runtime probe in `raithe-neural::ExecutionProvider`, this is
/// the *configured intent*. `Auto` triggers the probe chain; `Cuda` / `Cpu`
/// pin the engine to the named provider and fail loudly if unavailable.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// Probe CUDA → DirectML → CoreML → CPU and use the first that works.
    #[default]
    Auto,
    /// Pin to CUDA. Fail loudly if the toolkit/driver aren't usable.
    Cuda,
    /// Pin to CPU. Used for embedder/reranker so the generator owns VRAM.
    Cpu,
}

/// Neural model directory and runtime configuration.
///
/// Each model directory must contain:
///   - `model.onnx`       — exported ONNX weights
///   - `tokenizer.json`   — HuggingFace vocabulary-based tokeniser
///   - `quant.txt`        — one-line label: fp32 / fp16 / int4
///
/// Quantized or large decoder-only LLMs additionally produce one of:
///   - `model.onnx_data` (ONNX default name)
///   - `model.onnx.data` (torch ONNXProgram.save default name)
///
/// alongside `model.onnx` (ONNX external data format). ORT loads both
/// automatically when they reside in the same directory.
///
/// Run `raithe.sh` from the workspace root to populate all three
/// directories before first startup.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct NeuralConfig {
    /// Root directory for all model subdirectories.
    pub model_dir: PathBuf,

    /// `embedder/model.onnx` + `tokenizer.json`
    /// HuggingFace: BAAI/bge-large-en-v1.5 (feature-extraction, 1024-dim)
    pub embedder_dir: PathBuf,

    /// `reranker/model.onnx` + `tokenizer.json`
    /// HuggingFace: BAAI/bge-reranker-large (text-classification)
    pub reranker_dir: PathBuf,

    /// `generator/model.onnx` + external-data + `tokenizer.json`
    /// HuggingFace: Qwen/Qwen2.5-{1.5B,3B,7B,14B}-Instruct. Exact model,
    /// quantization, and execution provider are picked by raithe.sh from
    /// the VRAM/RAM tier table and exported to this process via env
    /// overrides (`RAITHE__NEURAL__GENERATOR_QUANTIZATION=int4`, etc.).
    pub generator_dir: PathBuf,

    /// Absolute path to the ONNX Runtime shared library. Required when
    /// the `ort` crate is built with `load-dynamic`. Empty → fall back to
    /// `ORT_DYLIB_PATH` env var. raithe.sh sets this automatically.
    pub ort_dylib_path: PathBuf,

    /// Maximum number of tokens the generator may produce in a single
    /// `generate()` call. Bounds autoregressive decode cost.
    pub generator_max_tokens: usize,

    /// Quantization format the generator model on disk was exported in.
    /// Must match the `quant.txt` sidecar in `generator_dir`, else the
    /// engine refuses to load and prompts for `raithe.sh` re-export.
    pub generator_quantization: Quantization,

    /// Execution provider for the generator. Default `Auto`; raithe.sh
    /// sets this to `Cuda` when a supported GPU is detected.
    pub generator_provider: Provider,

    /// Execution provider for the embedder. Default `Cpu` so the
    /// generator owns VRAM exclusively on 8 GB-class GPUs (Option A).
    pub embedder_provider: Provider,

    /// Execution provider for the reranker. Default `Cpu` for the same
    /// reason as `embedder_provider`.
    pub reranker_provider: Provider,

    /// Soft cap on how much VRAM (bytes) the generator session may
    /// allocate. Forwarded to CUDA EP via `gpu_mem_limit`. Leaves the
    /// remainder for OS, display compositor, and CUDA runtime overhead.
    /// Default 6 GB — conservative for an 8 GB card with an active
    /// desktop. raithe.sh overrides per-tier.
    pub generator_gpu_mem_limit_bytes: u64,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            model_dir:                    PathBuf::from("data/models"),
            embedder_dir:                 PathBuf::from("data/models/embedder"),
            reranker_dir:                 PathBuf::from("data/models/reranker"),
            generator_dir:                PathBuf::from("data/models/generator"),
            ort_dylib_path:               PathBuf::new(),
            generator_max_tokens:         256,
            generator_quantization:       Quantization::Fp32,
            generator_provider:           Provider::Auto,
            embedder_provider:            Provider::Cpu,
            reranker_provider:            Provider::Cpu,
            generator_gpu_mem_limit_bytes: 6 * 1024 * 1024 * 1024,
        }
    }
}