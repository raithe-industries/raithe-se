// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/indexer/benches/indexer_throughput.rs
//
// §12.3 required benchmark: Indexer::add throughput (docs/sec).
//
// Design: `Indexer::new()` involves mmap directory creation and writer-heap
// allocation (1 GB by default). These costs must not be charged to the `add`
// measurement. The indexer is therefore constructed once per benchmark group
// and reused across all iterations. A global atomic counter ensures document
// IDs remain unique across iterations without per-iteration setup cost.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use raithe_common::{DocumentId, SimHash, Url};
use raithe_config::IndexerConfig;
use raithe_indexer::Indexer;
use raithe_metrics::Metrics;
use raithe_parser::ParsedDocument;

fn make_metrics() -> Arc<Metrics> {
    Arc::new(Metrics::new().expect("internal error: metrics init failed in bench"))
}

fn make_doc(id: u64) -> ParsedDocument {
    ParsedDocument {
        id:           DocumentId::new(id),
        url:          Url::parse("https://example.com/bench").expect("internal error: bench URL"),
        title:        format!("Benchmark document {id}"),
        description:  String::from("A document used for indexer throughput benchmarking."),
        headings:     vec![
            format!("Section {id}"),
            String::from("Introduction"),
        ],
        body_text:    format!(
            "This is benchmark document number {id}. It contains representative body text \
             with enough tokens to exercise the Tantivy tokenisation pipeline. Rust search \
             engines must be fast, correct, and safe."
        ),
        outlinks:     Vec::new(),
        content_hash: SimHash::new(id),
    }
}

/// Benchmarks `Indexer::add` throughput across batch sizes 1, 10, 100, 1 000.
///
/// The indexer is constructed once outside the timed loop so that mmap
/// directory creation and writer-heap allocation are not charged to the
/// measurement. Each call to `b.iter` adds `batch` documents with unique IDs
/// supplied by a global atomic counter.
fn bench_add_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexer_add");

    static COUNTER: AtomicU64 = AtomicU64::new(1);

    {
        let dir     = tempfile::tempdir().expect("internal error: tempdir");
        let config  = IndexerConfig::default();
        let metrics = make_metrics();
        let indexer = Indexer::new(dir.path(), &config, metrics)
            .expect("internal error: indexer init");

        for batch in [1usize, 10, 100, 1_000] {
            group.throughput(Throughput::Elements(batch as u64));

            group.bench_with_input(
                BenchmarkId::from_parameter(batch),
                &batch,
                |b, &batch| {
                    b.iter(|| {
                        let base = COUNTER.fetch_add(batch as u64, Ordering::Relaxed);
                        let docs: Vec<ParsedDocument> = (0..batch as u64)
                            .map(|i| make_doc(base + i))
                            .collect();
                        for doc in &docs {
                            indexer.add(doc).expect("internal error: indexer add");
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmarks `Indexer::add` × 100 followed by `Indexer::commit`.
///
/// Uses `BatchSize::LargeInput` so Criterion excludes the setup closure
/// (which constructs a fresh `Indexer`) from the timed measurement. Each
/// iteration therefore measures only the cost of adding 100 documents and
/// committing a single segment — preventing segment accumulation across
/// iterations from inflating later samples.
fn bench_add_and_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexer_add_and_commit");
    group.throughput(Throughput::Elements(100));
    group.sample_size(20);

    static COMMIT_COUNTER: AtomicU64 = AtomicU64::new(1_000_000);

    group.bench_function("100_docs_then_commit", |b| {
        b.iter_batched(
            || {
                let dir     = tempfile::tempdir().expect("internal error: tempdir");
                let config  = IndexerConfig::default();
                let metrics = make_metrics();
                let indexer = Indexer::new(dir.path(), &config, metrics)
                    .expect("internal error: indexer init");
                let base = COMMIT_COUNTER.fetch_add(100, Ordering::Relaxed);
                let docs: Vec<ParsedDocument> = (0..100u64)
                    .map(|i| make_doc(base + i))
                    .collect();
                (dir, indexer, docs)
            },
            |(_dir, indexer, docs)| {
                for doc in &docs {
                    indexer.add(doc).expect("internal error: indexer add");
                }
                indexer.commit().expect("internal error: indexer commit");
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_add_throughput, bench_add_and_commit);
criterion_main!(benches);
