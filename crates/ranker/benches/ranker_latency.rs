// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/ranker/benches/ranker_latency.rs
//
// §12.3 required benchmark: Ranker::rank latency (p50, p95, p99).
//
// Models are expected to be present under data/models/. Run
// install_models.sh from the workspace root before benchmarking.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use raithe_common::{DocumentId, ParsedQuery, RawHit, Url};
use raithe_config::{NeuralConfig, RankerConfig};
use raithe_linkgraph::PageRankScores;
use raithe_metrics::Metrics;
use raithe_neural::RerankEngine;
use raithe_ranker::Ranker;

fn make_metrics() -> Arc<Metrics> {
    Arc::new(Metrics::new().expect("internal error: metrics init failed in bench"))
}

fn make_hit(id: u64, score: f32) -> RawHit {
    RawHit {
        id:      DocumentId::new(id),
        url:     Url::parse("https://example.com/").expect("internal error: bench URL"),
        score,
        snippet: String::from(
            "Rust is a systems programming language focused on safety, speed, and concurrency.",
        ),
        title: format!("Result {id}"),
    }
}

fn make_hits(n: usize) -> Vec<RawHit> {
    (1..=n as u64)
        .map(|i| make_hit(i, 1.0 / i as f32))
        .collect()
}

/// Benchmarks `Ranker::rank` across representative result-set sizes.
///
/// Measures the combined GBDT Phase 2 + BGE cross-encoder Phase 3 latency.
/// The ranker is constructed once per benchmark group; hits are freshly
/// constructed per iteration to avoid measuring clone overhead.
fn bench_rank_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ranker_rank");
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.sample_size(50);

    let metrics  = make_metrics();
    let neural   = RerankEngine::new(&NeuralConfig::default(), Arc::clone(&metrics))
        .expect("internal error: RerankEngine init failed — models must be present");
    let ranker   = Ranker::new(RankerConfig::default(), neural, Arc::clone(&metrics))
        .expect("internal error: Ranker init failed");
    let query    = ParsedQuery::raw("rust systems programming");
    let pageranks = PageRankScores::new();

    for n in [10usize, 32, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let hits = make_hits(n);
                    ranker
                        .rank(hits, &query, &pageranks)
                        .expect("internal error: ranker rank failed in bench")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_rank_latency);
criterion_main!(benches);