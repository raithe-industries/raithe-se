// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/query/benches/query_latency.rs
//
// §12.3 required benchmark: QueryProcessor::process latency.
//
// Models are expected to be present under data/models/. Run
// install_models.sh from the workspace root before benchmarking.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use raithe_config::NeuralConfig;
use raithe_metrics::Metrics;
use raithe_neural::GenerateEngine;
use raithe_query::QueryProcessor;

fn make_metrics() -> Arc<Metrics> {
    Arc::new(Metrics::new().expect("internal error: metrics init failed in bench"))
}

/// Benchmarks `QueryProcessor::process` across representative query lengths.
///
/// The processor is constructed once. Each iteration submits a fresh raw
/// string through the full pipeline: tokenise → intent classify → synonym
/// expand → LLM rewrite (Qwen2.5).
fn bench_process_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_process");
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.sample_size(50);

    let metrics   = make_metrics();
    let neural    = GenerateEngine::new(&NeuralConfig::default(), Arc::clone(&metrics))
        .expect("internal error: GenerateEngine init failed — models must be present");
    let processor = QueryProcessor::new(neural, Arc::clone(&metrics));

    let queries = [
        ("short",  "rust"),
        ("medium", "fast safe systems programming language"),
        ("long",   "what is the best way to learn rust for backend web development in 2026"),
    ];

    for (label, raw) in &queries {
        group.bench_with_input(
            BenchmarkId::new("process", label),
            raw,
            |b, raw| {
                b.iter(|| {
                    processor
                        .process(raw)
                        .expect("internal error: query process failed in bench")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_process_latency);
criterion_main!(benches);