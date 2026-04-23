# raithe-se model files

Run `./raithe.sh` from the workspace root. Models are downloaded, exported
to ONNX, validated, and installed automatically on first run.

## Expected layout (spec §7.1)

```
data/models/
├── embedder/
│   ├── model.onnx           BGE-large-en-v1.5 — dense bi-encoder (1274 MB)
│   └── tokenizer.json       HF: BAAI/bge-large-en-v1.5
├── reranker/
│   ├── model.onnx           BGE-reranker-large — cross-encoder graph (478 KB)
│   ├── model.onnx_data      BGE-reranker-large — weights (2135 MB)
│   └── tokenizer.json       HF: BAAI/bge-reranker-large
└── generator/
    ├── model.onnx           Qwen2.5-7B/14B — graph
    ├── model.onnx_data      Qwen2.5-7B/14B — weights (29051 MB for 7B)
    └── tokenizer.json       HF: Qwen/Qwen2.5-7B-Instruct
```

**This directory is git-ignored. Never commit model files.**

`model.onnx_data` is the ONNX external data file — required alongside
`model.onnx` for reranker and generator. ORT loads both automatically.

Missing any required file = `Error::ModelNotFound` at startup.
Re-run `./raithe.sh --force-install` to reinstall.
