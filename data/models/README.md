# raithe-se model files

Run `install_models.sh` from the workspace root to download and install
all three models automatically via optimum-cli.

## Expected layout (spec §7.1)

```
data/models/
├── embedder/
│   ├── model.onnx           BGE-large-en-v1.5 — dense bi-encoder embedding
│   └── tokenizer.json       HF: BAAI/bge-large-en-v1.5
├── reranker/
│   ├── model.onnx           BGE-reranker-large — Phase 3 cross-encoder ranking
│   └── tokenizer.json       HF: BAAI/bge-reranker-large
└── generator/
    ├── model.onnx           Qwen2.5-7B/14B — query understanding + rewriting
    └── tokenizer.json       HF: Qwen/Qwen2.5-7B-Instruct
```

**This directory is git-ignored. Never commit model files.**

Missing any model.onnx or tokenizer.json = `Error::ModelNotFound` at startup.
