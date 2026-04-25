// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/indexer/src/lib.rs
//
// Tantivy inverted index — schema, tokeniser, ingestion, BM25F search.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use raithe_common::{DocumentId, ParsedQuery, RawHit, Url};
use raithe_config::IndexerConfig;
use raithe_metrics::Metrics;
use raithe_parser::ParsedDocument;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, FAST, INDEXED, STORED, STRING, TEXT};
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument};
use thiserror::Error;

/// On-disk schema version. Bump when the field set changes incompatibly.
/// `Indexer::new` writes this to `<index_path>/VERSION` on first open and
/// refuses to open mismatched indexes — no silent corruption.
const INDEX_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Tantivy error: {source}")]
    Tantivy { #[source] source: tantivy::TantivyError },
    #[error("index writer lock poisoned")]
    LockPoisoned,
    #[error("invalid query '{query}': {reason}")]
    InvalidQuery { query: String, reason: String },
    #[error("document ID space exhausted")]
    DocIdExhausted,
    #[error("index schema version mismatch — expected v{expected}, found v{found}; delete the index dir and re-crawl")]
    SchemaMismatch { expected: u32, found: String },
    #[error("index version file io error: {source}")]
    VersionIo { #[source] source: std::io::Error },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<tantivy::TantivyError> for Error {
    fn from(source: tantivy::TantivyError) -> Self { Self::Tantivy { source } }
}

#[derive(Clone, Copy, Debug)]
pub struct IndexSchema {
    pub doc_id:      Field,
    pub url:         Field,
    pub title:       Field,
    pub description: Field,
    pub headings:    Field,
    pub body:        Field,
}

impl IndexSchema {
    fn build() -> (Schema, Self) {
        let mut builder = Schema::builder();

        let doc_id      = builder.add_u64_field("doc_id", INDEXED | STORED | FAST);
        let url         = builder.add_text_field("url", STRING | STORED);
        let title       = builder.add_text_field("title", TEXT | STORED);
        let description = builder.add_text_field("description", TEXT | STORED);
        let headings    = builder.add_text_field("headings", TEXT);
        // body must be STORED so search() can produce non-empty snippets.
        let body        = builder.add_text_field("body", TEXT | STORED);

        let schema = builder.build();
        let fields = Self { doc_id, url, title, description, headings, body };
        (schema, fields)
    }
}

pub struct Indexer {
    index:     Index,
    schema:    IndexSchema,
    writer:    RwLock<IndexWriter>,
    metrics:   Arc<Metrics>,
    /// Documents added since the last successful commit.
    pending:   AtomicU64,
    /// Documents in the searchable (post-commit) segment.
    committed: AtomicU64,
}

impl Indexer {
    pub fn new(path: &Path, config: &IndexerConfig, metrics: Arc<Metrics>) -> Result<Self> {
        let (schema, fields) = IndexSchema::build();

        std::fs::create_dir_all(path)
            .map_err(|err| tantivy::TantivyError::IoError(Arc::new(err)))?;
        check_or_write_version(path)?;

        let dir   = tantivy::directory::MmapDirectory::open(path).map_err(tantivy::TantivyError::from)?;
        let index = Index::open_or_create(dir, schema)?;

        let heap_bytes = config.writer_heap_mb * 1024 * 1024;
        let writer     = index.writer(heap_bytes as usize)?;
        let writer     = RwLock::new(writer);

        // Recover committed count from the on-disk index (post-restart).
        let initial_committed = {
            let reader = index.reader_builder()
                .reload_policy(ReloadPolicy::OnCommitWithDelay)
                .try_into()?;
            reader.searcher().num_docs()
        };

        Ok(Self {
            index,
            schema: fields,
            writer,
            metrics,
            pending:   AtomicU64::new(0),
            committed: AtomicU64::new(initial_committed),
        })
    }

    pub fn add(&self, doc: &ParsedDocument) -> Result<()> {
        if doc.id == DocumentId::ZERO {
            return Err(Error::InvalidQuery {
                query:  String::new(),
                reason: "document id is ZERO — caller must assign before indexing".to_owned(),
            });
        }

        let headings = doc.headings.join(" ");

        let mut tantivy_doc = TantivyDocument::default();
        tantivy_doc.add_u64(self.schema.doc_id, doc.id.get());
        tantivy_doc.add_text(self.schema.url, doc.url.as_str());
        tantivy_doc.add_text(self.schema.title, &doc.title);
        tantivy_doc.add_text(self.schema.description, &doc.description);
        tantivy_doc.add_text(self.schema.headings, &headings);
        tantivy_doc.add_text(self.schema.body, &doc.body_text);

        let writer = self.writer.write().map_err(|_| Error::LockPoisoned)?;
        writer.add_document(tantivy_doc)?;

        self.pending.fetch_add(1, Ordering::Relaxed);
        self.metrics.index_documents_total.inc();
        Ok(())
    }

    /// Commits all pending documents.
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.write().map_err(|_| Error::LockPoisoned)?;
        writer.commit()?;
        self.pending.store(0, Ordering::Relaxed);

        let count = {
            let reader = self.index.reader_builder()
                .reload_policy(ReloadPolicy::OnCommitWithDelay)
                .try_into()?;
            reader.searcher().num_docs()
        };
        self.committed.store(count, Ordering::Relaxed);
        Ok(())
    }

    /// Commits only when `pending >= threshold`. Returns `Ok(true)` if a commit ran.
    pub fn maybe_commit(&self, threshold: u64) -> Result<bool> {
        if self.pending.load(Ordering::Relaxed) >= threshold {
            self.commit()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn pending_count(&self)   -> u64 { self.pending.load(Ordering::Relaxed) }
    pub fn committed_count(&self) -> u64 { self.committed.load(Ordering::Relaxed) }

    /// Returns the highest `doc_id` observed across all segments, or `None`
    /// for an empty index. Called at startup so the in-memory monotonic
    /// counter can resume past the last persisted ID instead of restarting
    /// at zero (which would silently overwrite older docs on subsequent adds).
    pub fn max_doc_id(&self) -> Result<Option<DocumentId>> {
        let reader = self.index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        let searcher = reader.searcher();

        let mut max_id: Option<u64> = None;
        for segment_reader in searcher.segment_readers() {
            let column = segment_reader.fast_fields().u64("doc_id")?;
            for doc in 0..segment_reader.max_doc() {
                if let Some(val) = column.first(doc) {
                    max_id = Some(max_id.map_or(val, |m| m.max(val)));
                }
            }
        }
        Ok(max_id.map(DocumentId::new))
    }

    pub fn search(&self, query: &ParsedQuery) -> Result<Vec<RawHit>> {
        let reader = self.index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        let searcher = reader.searcher();

        let query_str = if query.rewritten.is_empty() { &query.original } else { &query.rewritten };

        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.schema.title, self.schema.description, self.schema.headings, self.schema.body],
        );

        let tantivy_query = query_parser
            .parse_query(query_str)
            .map_err(|err| Error::InvalidQuery {
                query:  query_str.to_owned(),
                reason: err.to_string(),
            })?;

        let top_docs = searcher.search(&tantivy_query, &TopDocs::with_limit(100))?;

        let hits = top_docs.into_iter().filter_map(|(score, addr)| {
            let retrieved: TantivyDocument = searcher.doc(addr).ok()?;
            let id = retrieved.get_first(self.schema.doc_id)
                .and_then(|v: &tantivy::schema::OwnedValue| v.as_u64())
                .map(DocumentId::new)?;
            let url_str = retrieved.get_first(self.schema.url)
                .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                .unwrap_or("");
            let url = Url::parse(url_str).ok()?;
            let title = retrieved.get_first(self.schema.title)
                .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                .unwrap_or("")
                .to_owned();
            let snippet = retrieved.get_first(self.schema.body)
                .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                .map(|b: &str| b.chars().take(200).collect::<String>())
                .unwrap_or_default();

            Some(RawHit { id, url, score, snippet, title })
        }).collect();

        Ok(hits)
    }

    pub fn doc_count(&self) -> Result<u64> { Ok(self.committed_count()) }
}

/// Reads `<path>/VERSION` and verifies it matches `INDEX_SCHEMA_VERSION`.
/// Writes the current version on first open. Returns `SchemaMismatch` when
/// an older version file is found — operator must `rm -rf` the index dir.
fn check_or_write_version(path: &Path) -> Result<()> {
    let v_path = path.join("VERSION");
    match std::fs::read_to_string(&v_path) {
        Ok(s) => {
            let trimmed = s.trim();
            let parsed: u32 = trimmed.parse().map_err(|_| Error::SchemaMismatch {
                expected: INDEX_SCHEMA_VERSION,
                found:    trimmed.to_owned(),
            })?;
            if parsed != INDEX_SCHEMA_VERSION {
                return Err(Error::SchemaMismatch {
                    expected: INDEX_SCHEMA_VERSION,
                    found:    parsed.to_string(),
                });
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::write(&v_path, INDEX_SCHEMA_VERSION.to_string())
                .map_err(|source| Error::VersionIo { source })
        }
        Err(source) => Err(Error::VersionIo { source }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raithe_common::SimHash;
    use raithe_config::IndexerConfig;

    fn make_metrics() -> Arc<Metrics> { Arc::new(Metrics::new().unwrap()) }

    fn make_doc(id: u64, title: &str, body: &str) -> ParsedDocument {
        ParsedDocument {
            id: DocumentId::new(id),
            url: Url::parse("https://example.com/").unwrap(),
            title: title.to_owned(),
            description: String::new(),
            headings: Vec::new(),
            body_text: body.to_owned(),
            outlinks: Vec::new(),
            content_hash: SimHash::new(0),
        }
    }

    #[test]
    fn add_and_search_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let indexer = Indexer::new(dir.path(), &IndexerConfig::default(), make_metrics()).unwrap();
        indexer.add(&make_doc(1, "Rust search engine", "fast reliable indexing")).unwrap();
        indexer.commit().unwrap();
        let hits = indexer.search(&ParsedQuery::raw("search engine")).unwrap();
        assert_eq!(hits[0].id, DocumentId::new(1));
        assert!(!hits[0].snippet.is_empty(), "snippet must be non-empty (body must be STORED)");
    }

    #[test]
    fn pending_resets_on_commit() {
        let dir = tempfile::tempdir().unwrap();
        let indexer = Indexer::new(dir.path(), &IndexerConfig::default(), make_metrics()).unwrap();
        indexer.add(&make_doc(1, "a", "one")).unwrap();
        indexer.add(&make_doc(2, "b", "two")).unwrap();
        assert_eq!(indexer.pending_count(), 2);
        indexer.commit().unwrap();
        assert_eq!(indexer.pending_count(), 0);
        assert_eq!(indexer.committed_count(), 2);
    }

    #[test]
    fn maybe_commit_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let indexer = Indexer::new(dir.path(), &IndexerConfig::default(), make_metrics()).unwrap();
        indexer.add(&make_doc(1, "a", "one")).unwrap();
        assert!(!indexer.maybe_commit(2).unwrap());
        indexer.add(&make_doc(2, "b", "two")).unwrap();
        assert!(indexer.maybe_commit(2).unwrap());
        assert_eq!(indexer.committed_count(), 2);
    }

    #[test]
    fn zero_id_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let indexer = Indexer::new(dir.path(), &IndexerConfig::default(), make_metrics()).unwrap();
        let mut doc = make_doc(0, "title", "body");
        doc.id = DocumentId::ZERO;
        assert!(indexer.add(&doc).is_err());
    }

    #[test]
    fn max_doc_id_empty() {
        let dir = tempfile::tempdir().unwrap();
        let indexer = Indexer::new(dir.path(), &IndexerConfig::default(), make_metrics()).unwrap();
        assert_eq!(indexer.max_doc_id().unwrap(), None);
    }

    #[test]
    fn max_doc_id_after_commit() {
        let dir = tempfile::tempdir().unwrap();
        let indexer = Indexer::new(dir.path(), &IndexerConfig::default(), make_metrics()).unwrap();
        indexer.add(&make_doc(7, "a", "one")).unwrap();
        indexer.add(&make_doc(3, "b", "two")).unwrap();
        indexer.add(&make_doc(42, "c", "three")).unwrap();
        indexer.commit().unwrap();
        assert_eq!(indexer.max_doc_id().unwrap(), Some(DocumentId::new(42)));
    }

    #[test]
    fn schema_version_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(dir.path().join("VERSION"), "1").unwrap();
        let result = Indexer::new(dir.path(), &IndexerConfig::default(), make_metrics());
        assert!(matches!(result, Err(Error::SchemaMismatch { .. })));
    }
}