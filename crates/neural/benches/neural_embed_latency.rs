// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/neural/benches/neural_embed_latency.rs
//
// §12.3 required benchmark: NeuralEngine::embed latency per batch size.
//
// Models are expected to be present under data/models/. Run
// install_models.sh from the workspace root before benchmarking.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use raithe_config::NeuralConfig;
use raithe_metrics::Metrics;
use raithe_neural::NeuralEngine;

fn make_metrics() -> Arc<Metrics> {
    Arc::new(Metrics::new().expect("internal error: metrics init failed in bench"))
}

// Representative passage text for embedding; length approximates a typical
// body_text snippet passed from the indexing pipeline.
const PASSAGE: &str =
    "Rust is a systems programming language that runs blazingly fast, prevents \
     segfaults, and guarantees thread safety. Its ownership model eliminates \
     entire classes of bugs at compile time without a garbage collector.";

/// Benchmarks `NeuralEngine::embed` across increasing batch sizes.
///
/// Measures wall-clock latency per call and throughput in texts/sec.
/// The engine is constructed once; a new `&[&str]` slice is assembled per
/// iteration from static passage text so no allocation occurs on the hot path.
fn bench_embed_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_embed");
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.sample_size(50);

    let metrics = make_metrics();
    let mut engine = NeuralEngine::new(&NeuralConfig::default(), Arc::clone(&metrics))
        .expect("internal error: NeuralEngine init failed — models must be present");

    for batch in [1usize, 4, 8, 16, 32] {
        group.throughput(Throughput::Elements(batch as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            &batch,
            |b, &batch| {
                let texts: Vec<&str> = (0..batch).map(|_| PASSAGE).collect();
                b.iter(|| {
                    engine
                        .embed(&texts)
                        .expect("internal error: embed failed in bench")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_embed_latency);
criterion_main!(benches);
