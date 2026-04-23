// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/parser/src/lib.rs
//
// Raw HTML bytes → ParsedDocument.
// lol_html streams body text and outlinks. The scraper crate handles metadata
// extraction (title, description, headings) which requires DOM traversal.
// This replaces the prototype's scraper-only approach (DEV-006 fix).

use raithe_common::{DocumentId, SimHash, Url};
use raithe_scraper::FetchResult;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("HTML parse error for '{url}': {reason}")]
    Parse { url: String, reason: String },
    #[error("document body is empty for '{url}'")]
    EmptyBody { url: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// A fully extracted document ready for indexing.
#[derive(Clone, Debug)]
pub struct ParsedDocument {
    /// Assigned document identifier — callers must set this before indexing.
    /// Initialised to `DocumentId::ZERO` by `Parser::parse`.
    pub id: DocumentId,
    /// Final URL of the document.
    pub url: Url,
    /// Content of the `<title>` element.
    pub title: String,
    /// Content of the `<meta name="description">` element.
    pub description: String,
    /// Text content of all `<h1>`–`<h3>` elements, in document order.
    pub headings: Vec<String>,
    /// Boilerplate-stripped body text.
    pub body_text: String,
    /// Outbound links extracted from `<a href>` attributes.
    pub outlinks: Vec<Url>,
    /// SimHash fingerprint of the body text for near-duplicate detection.
    pub content_hash: SimHash,
}

/// Parses raw HTML bytes from a `FetchResult` into a `ParsedDocument`.
pub struct Parser;

impl Parser {
    /// Creates a new `Parser`.
    pub fn new() -> Self {
        Self
    }

    /// Parses the HTML in `result` and returns a `ParsedDocument`.
    ///
    /// Uses `lol_html` for streaming body-text and outlink extraction, and
    /// the `scraper` crate for DOM-based metadata (title, description,
    /// headings). Falls back to a full `scraper` parse if `lol_html` fails,
    /// to handle malformed documents.
    pub fn parse(&self, result: FetchResult) -> Result<ParsedDocument> {
        if result.body_bytes.is_empty() {
            return Err(Error::EmptyBody {
                url: result.url.to_string(),
            });
        }

        let meta = extract_meta(&result.body_bytes);

        let (body_text, outlinks) = match extract_body_lol_html(&result.body_bytes, &result.url) {
            Ok(pair) => pair,
            Err(_) => extract_body_scraper(&result.body_bytes, &result.url),
        };

        let content_hash = SimHash::from_tokens(body_text.split_whitespace());

        Ok(ParsedDocument {
            id: DocumentId::ZERO,
            url: result.url,
            title: meta.title,
            description: meta.description,
            headings: meta.headings,
            body_text,
            outlinks,
            content_hash,
        })
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Metadata extraction (scraper crate — DOM traversal) ──────────────────────

struct Meta {
    title: String,
    description: String,
    headings: Vec<String>,
}

fn extract_meta(bytes: &[u8]) -> Meta {
    use scraper::{Html, Selector};

    let html = std::str::from_utf8(bytes).unwrap_or("");
    let document = Html::parse_document(html);

    let title = select_text(&document, "title");
    let description = {
        let sel = Selector::parse("meta[name='description']").unwrap();
        document
            .select(&sel)
            .next()
            .and_then(|el| el.value().attr("content"))
            .unwrap_or("")
            .to_owned()
    };
    let headings = {
        let sel = Selector::parse("h1, h2, h3").unwrap();
        document
            .select(&sel)
            .map(|el| el.text().collect::<String>().trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect()
    };

    Meta {
        title,
        description,
        headings,
    }
}

fn select_text(document: &scraper::Html, selector: &str) -> String {
    let sel = scraper::Selector::parse(selector).unwrap();
    document
        .select(&sel)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_owned())
        .unwrap_or_default()
}

// ── Body + outlinks — lol_html primary ───────────────────────────────────────

fn extract_body_lol_html(
    bytes: &[u8],
    base_url: &Url,
) -> std::result::Result<(String, Vec<Url>), ()> {
    use lol_html::{element, DocumentContentHandlers, HtmlRewriter, Settings};

    let mut body_parts = Vec::<String>::new();
    let mut outlink_strs = Vec::<String>::new();

    {
        let mut rewriter = HtmlRewriter::new(
            Settings {
                element_content_handlers: vec![
                    element!("script, style, noscript, nav, footer, header", |el| {
                        el.remove();
                        Ok(())
                    }),
                    element!("a[href]", |el| {
                        if let Some(href) = el.get_attribute("href") {
                            if let Ok(url) = resolve_url(base_url, &href) {
                                outlink_strs.push(url.to_string());
                            }
                        }
                        Ok(())
                    }),
                ],
                document_content_handlers: vec![DocumentContentHandlers::default().text(|t| {
                    let text = t.as_str().trim().to_owned();
                    if !text.is_empty() {
                        body_parts.push(text);
                    }
                    Ok(())
                })],
                ..Settings::default()
            },
            |_: &[u8]| {},
        );

        rewriter.write(bytes).map_err(|_| ())?;
        rewriter.end().map_err(|_| ())?;
    }

    let body_text = body_parts.join(" ");
    let outlinks = outlink_strs
        .into_iter()
        .filter_map(|s| Url::parse(&s).ok())
        .collect();

    Ok((body_text, outlinks))
}

// ── Body + outlinks — scraper fallback ───────────────────────────────────────

fn extract_body_scraper(bytes: &[u8], base_url: &Url) -> (String, Vec<Url>) {
    use scraper::{Html, Selector};

    let html = std::str::from_utf8(bytes).unwrap_or("");
    let document = Html::parse_document(html);

    let body_text = {
        let sel = Selector::parse("body").unwrap();
        document
            .select(&sel)
            .next()
            .map(|el| el.text().collect::<Vec<_>>().join(" "))
            .unwrap_or_default()
    };

    let outlinks = {
        let sel = Selector::parse("a[href]").unwrap();
        document
            .select(&sel)
            .filter_map(|el| el.value().attr("href"))
            .filter_map(|href| resolve_url(base_url, href).ok())
            .collect()
    };

    (body_text, outlinks)
}

// ── URL resolution ────────────────────────────────────────────────────────────

/// Resolves `href` against `base`, returning an absolute `Url`.
///
/// Rejects fragment-only, javascript:, data:, and mailto: hrefs.
fn resolve_url(base: &Url, href: &str) -> std::result::Result<Url, ()> {
    let href = href.trim();
    if href.is_empty()
        || href.starts_with('#')
        || href.starts_with("javascript:")
        || href.starts_with("data:")
        || href.starts_with("mailto:")
    {
        return Err(());
    }

    if href.starts_with("http://") || href.starts_with("https://") {
        return Url::parse(href).map_err(|_| ());
    }

    base.inner()
        .join(href)
        .map_err(|_| ())
        .and_then(|u| Url::parse(u.as_str()).map_err(|_| ()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use raithe_common::Timestamp;
    use std::collections::HashMap;

    fn make_result(html: &str, url: &str) -> FetchResult {
        FetchResult {
            url: Url::parse(url).unwrap(),
            status: 200,
            headers: HashMap::new(),
            body_bytes: html.as_bytes().to_vec(),
            fetched_at: Timestamp::from_millis(0),
        }
    }

    #[test]
    fn extracts_title_and_description() {
        let html = r#"<html><head>
            <title>Hello RAiTHE</title>
            <meta name="description" content="A great search engine.">
            </head><body><p>Some body text.</p></body></html>"#;

        let doc = Parser::new()
            .parse(make_result(html, "https://example.com/"))
            .unwrap();

        assert_eq!(doc.title, "Hello RAiTHE");
        assert_eq!(doc.description, "A great search engine.");
        assert!(doc.body_text.contains("Some body text"));
    }

    #[test]
    fn extracts_headings() {
        let html = r#"<html><body>
            <h1>Main</h1><h2>Sub</h2><h3>Minor</h3><h4>Ignored</h4>
            </body></html>"#;

        let doc = Parser::new()
            .parse(make_result(html, "https://example.com/"))
            .unwrap();

        assert!(doc.headings.contains(&"Main".to_owned()));
        assert!(doc.headings.contains(&"Sub".to_owned()));
        assert!(doc.headings.contains(&"Minor".to_owned()));
        assert!(!doc.headings.contains(&"Ignored".to_owned()));
    }

    #[test]
    fn extracts_outlinks() {
        let html = r#"<html><body>
            <a href="https://other.com/page">link</a>
            <a href="/relative">rel</a>
            <a href="javascript:void(0)">bad</a>
            <a href="mailto:x@y.com">mail</a>
            </body></html>"#;

        let doc = Parser::new()
            .parse(make_result(html, "https://example.com/"))
            .unwrap();
        let urls: Vec<_> = doc.outlinks.iter().map(|u| u.as_str()).collect();

        assert!(urls.iter().any(|u| u.contains("other.com/page")));
        assert!(urls.iter().any(|u| u.contains("example.com/relative")));
        assert!(!urls.iter().any(|u| u.contains("javascript")));
        assert!(!urls.iter().any(|u| u.contains("mailto")));
    }

    #[test]
    fn empty_body_returns_error() {
        let result = FetchResult {
            url: Url::parse("https://example.com/").unwrap(),
            status: 200,
            headers: HashMap::new(),
            body_bytes: vec![],
            fetched_at: Timestamp::from_millis(0),
        };
        assert!(Parser::new().parse(result).is_err());
    }

    #[test]
    fn content_hash_differs_for_different_bodies() {
        let a = Parser::new()
            .parse(make_result(
                "<html><body>alpha beta gamma</body></html>",
                "https://a.com/",
            ))
            .unwrap();
        let b = Parser::new()
            .parse(make_result(
                "<html><body>delta epsilon zeta</body></html>",
                "https://b.com/",
            ))
            .unwrap();
        assert_ne!(a.content_hash, b.content_hash);
    }

    #[test]
    fn id_is_zero_sentinel() {
        let doc = Parser::new()
            .parse(make_result(
                "<html><body>text</body></html>",
                "https://example.com/",
            ))
            .unwrap();
        assert_eq!(doc.id, DocumentId::ZERO);
    }
}
