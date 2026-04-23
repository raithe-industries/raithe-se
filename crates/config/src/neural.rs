// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/neural.rs

use std::path::PathBuf;

/// Neural model directory configuration.
///
/// Each model directory must contain:
///   - `model.onnx`       — exported ONNX weights
///   - `tokenizer.json`   — HuggingFace vocabulary-based tokeniser
///
/// Large decoder-only LLMs (generator) additionally produce `model.onnx_data`
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

    /// `generator/model.onnx` + `model.onnx_data` + `tokenizer.json`
    /// HuggingFace: Qwen/Qwen2.5-7B-Instruct (text-generation)
    /// Hardware-adaptive: `raithe.sh` selects 14B when
    /// RAM ≥ 64 GB and VRAM ≥ 16 GB; 7B on all other hardware.
    pub generator_dir: PathBuf,

    /// Absolute path to the ONNX Runtime shared library (`libonnxruntime.so`
    /// on Linux, `onnxruntime.dll` on Windows, `libonnxruntime.dylib` on
    /// macOS). Required when the `ort` crate is built with `load-dynamic`.
    ///
    /// If empty, the `ORT_DYLIB_PATH` environment variable is used instead.
    /// `raithe.sh` sets this automatically from the ort build cache.
    pub ort_dylib_path: PathBuf,

    /// Maximum number of tokens the generator may produce in a single
    /// `generate()` call. Bounds autoregressive decode cost.
    pub generator_max_tokens: usize,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("data/models"),
            embedder_dir: PathBuf::from("data/models/embedder"),
            reranker_dir: PathBuf::from("data/models/reranker"),
            generator_dir: PathBuf::from("data/models/generator"),
            ort_dylib_path: PathBuf::new(),
            generator_max_tokens: 256,
        }
    }
}
