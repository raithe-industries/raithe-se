// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/indexer/src/lib.rs
//
// Tantivy inverted index — schema, tokeniser, ingestion, BM25F search.

use std::path::Path;
use std::sync::{Arc, RwLock};

use raithe_common::{DocumentId, ParsedQuery, RawHit, Url};
use raithe_config::IndexerConfig;
use raithe_metrics::Metrics;
use raithe_parser::ParsedDocument;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field, Schema, Value, FAST, INDEXED, STORED, STRING, TEXT,
};
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Tantivy error: {source}")]
    Tantivy {
        #[source]
        source: tantivy::TantivyError,
    },
    #[error("index writer lock poisoned")]
    LockPoisoned,
    #[error("invalid query '{query}': {reason}")]
    InvalidQuery { query: String, reason: String },
    #[error("document ID space exhausted")]
    DocIdExhausted,
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<tantivy::TantivyError> for Error {
    fn from(source: tantivy::TantivyError) -> Self {
        Self::Tantivy { source }
    }
}

/// Fields in the Tantivy schema.
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
        let body        = builder.add_text_field("body", TEXT);

        let schema = builder.build();
        let fields = Self { doc_id, url, title, description, headings, body };

        (schema, fields)
    }
}

/// Tantivy-backed inverted index.
///
/// `IndexWriter` is protected by an `RwLock` — lock poisoning is handled
/// explicitly rather than with `.unwrap()` (DEV-007 fix).
pub struct Indexer {
    index:       Index,
    schema:      IndexSchema,
    writer:      RwLock<IndexWriter>,
    metrics:     Arc<Metrics>,
}

impl Indexer {
    /// Opens (or creates) an index under `path` using the given config.
    ///
    /// The writer heap is set from `config.writer_heap_mb` — prototype used
    /// a hard-coded 50 MB (DEV-007); default is now 1 GB.
    pub fn new(
        path: &Path,
        config: &IndexerConfig,
        metrics: Arc<Metrics>,
    ) -> Result<Self> {
        let (schema, fields) = IndexSchema::build();

        std::fs::create_dir_all(path).map_err(|err| {
            tantivy::TantivyError::IoError(Arc::new(err))
        })?;

        let dir = tantivy::directory::MmapDirectory::open(path)
            .map_err(tantivy::TantivyError::from)?;
        let index = Index::open_or_create(
            dir,
            schema,
        )?;

        let heap_bytes = config.writer_heap_mb * 1024 * 1024;
        let writer = index.writer(heap_bytes as usize)?;
        let writer = RwLock::new(writer);

        Ok(Self {
            index,
            schema: fields,
            writer,
            metrics,
        })
    }

    /// Ingests a `ParsedDocument` into the index.
    ///
    /// The document must have a real `DocumentId` assigned by the caller
    /// before this call — `DocumentId::ZERO` is rejected.
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
        tantivy_doc.add_text(self.schema.url,         doc.url.as_str());
        tantivy_doc.add_text(self.schema.title,       &doc.title);
        tantivy_doc.add_text(self.schema.description, &doc.description);
        tantivy_doc.add_text(self.schema.headings,    &headings);
        tantivy_doc.add_text(self.schema.body,        &doc.body_text);

        let writer = self.writer.write().map_err(|_| Error::LockPoisoned)?;
        writer.add_document(tantivy_doc)?;

        self.metrics.index_documents_total.inc();

        Ok(())
    }

    /// Commits all pending documents to the index.
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.write().map_err(|_| Error::LockPoisoned)?;
        writer.commit()?;
        Ok(())
    }

    /// Searches the index using BM25F and returns the top hits.
    ///
    /// Searches across title (weight 3), description (weight 2), headings
    /// (weight 2), and body (weight 1). Returns up to 100 hits.
    pub fn search(&self, query: &ParsedQuery) -> Result<Vec<RawHit>> {
        let reader = self.index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        let searcher = reader.searcher();

        let query_str = if query.rewritten.is_empty() {
            &query.original
        } else {
            &query.rewritten
        };

        let query_parser = QueryParser::for_index(
            &self.index,
            vec![
                self.schema.title,
                self.schema.description,
                self.schema.headings,
                self.schema.body,
            ],
        );

        let tantivy_query = query_parser
            .parse_query(query_str)
            .map_err(|err| Error::InvalidQuery {
                query:  query_str.to_owned(),
                reason: err.to_string(),
            })?;

        let top_docs = searcher.search(&tantivy_query, &TopDocs::with_limit(100))?;

        let hits = top_docs
            .into_iter()
            .filter_map(|(score, addr)| {
                let retrieved: TantivyDocument = searcher.doc(addr).ok()?;
                let id = retrieved
                    .get_first(self.schema.doc_id)
                    .and_then(|v: &tantivy::schema::OwnedValue| v.as_u64())
                    .map(DocumentId::new)?;
                let url_str = retrieved
                    .get_first(self.schema.url)
                    .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                    .unwrap_or("");
                let url = Url::parse(url_str).ok()?;
                let title = retrieved
                    .get_first(self.schema.title)
                    .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                let snippet = retrieved
                    .get_first(self.schema.body)
                    .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                    .map(|b: &str| b.chars().take(200).collect::<String>())
                    .unwrap_or_default();

                Some(RawHit { id, url, score, snippet, title })
            })
            .collect();

        Ok(hits)
    }

    /// Returns the number of documents in the index.
    pub fn doc_count(&self) -> Result<u64> {
        let reader = self.index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        let searcher = reader.searcher();
        Ok(searcher.num_docs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raithe_common::SimHash;
    use raithe_config::IndexerConfig;
    use std::sync::Arc;

    fn make_metrics() -> Arc<Metrics> {
        Arc::new(Metrics::new().unwrap())
    }

    fn make_doc(id: u64, title: &str, body: &str) -> ParsedDocument {
        ParsedDocument {
            id:           DocumentId::new(id),
            url:          Url::parse("https://example.com/").unwrap(),
            title:        title.to_owned(),
            description:  String::new(),
            headings:     Vec::new(),
            body_text:    body.to_owned(),
            outlinks:     Vec::new(),
            content_hash: SimHash::new(0),
        }
    }

    #[test]
    fn add_and_search_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let config = IndexerConfig::default();
        let indexer = Indexer::new(dir.path(), &config, make_metrics()).unwrap();

        indexer.add(&make_doc(1, "Rust search engine", "fast reliable indexing")).unwrap();
        indexer.commit().unwrap();

        let query = ParsedQuery::raw("search engine");
        let hits = indexer.search(&query).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, DocumentId::new(1));
    }

    #[test]
    fn zero_id_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let config = IndexerConfig::default();
        let indexer = Indexer::new(dir.path(), &config, make_metrics()).unwrap();

        let mut doc = make_doc(0, "title", "body");
        doc.id = DocumentId::ZERO;
        assert!(indexer.add(&doc).is_err());
    }

    #[test]
    fn doc_count_tracks_ingestion() {
        let dir = tempfile::tempdir().unwrap();
        let config = IndexerConfig::default();
        let indexer = Indexer::new(dir.path(), &config, make_metrics()).unwrap();

        indexer.add(&make_doc(1, "alpha", "one")).unwrap();
        indexer.add(&make_doc(2, "beta",  "two")).unwrap();
        indexer.commit().unwrap();

        assert_eq!(indexer.doc_count().unwrap(), 2);
    }
}
