// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/parser/src/lib.rs
//
// Raw HTML bytes → ParsedDocument.
// lol_html streams body text and outlinks. The scraper crate handles metadata
// extraction (title, description, headings) which requires DOM traversal.

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

/// CSS selector list for non-content elements that must be stripped before
/// any text extraction. These leak code/markup into snippets if not removed:
///
///   script, noscript  — JS bootstrap (GTM, analytics, etc.)
///   style             — CSS source
///   template          — inert by spec, but `.text()` walks it anyway
///   iframe            — embedded apps with their own runtime
///   svg               — inline SVG path commands look like text to scraper
///
/// Layout/chrome elements (`nav`, `footer`, `header`) are NOT in this list:
/// while they're often boilerplate, they sometimes contain genuinely
/// query-relevant text (site-name in <header>, copyright dates in <footer>).
/// Strip them in the body-only pass below if needed, but keep them visible
/// to metadata extraction.
const NON_CONTENT_SELECTORS: &str = "script, style, noscript, template, iframe, svg";

/// Additional selectors to strip *only* from the body text (not metadata).
/// `nav`/`footer`/`header` are common boilerplate; removing them sharpens
/// snippets without harming title/description extraction.
const BODY_BOILERPLATE_SELECTORS: &str = "nav, footer, header, aside";

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
    pub fn new() -> Self { Self }

    /// Parses the HTML in `result` and returns a `ParsedDocument`.
    ///
    /// Pipeline:
    ///   1. Strip non-content nodes (script/style/etc.) via lol_html.
    ///   2. Run scraper over the cleaned bytes for metadata.
    ///   3. Run a second lol_html pass that also strips body boilerplate
    ///      (nav/footer/header) for body text + outlinks.
    pub fn parse(&self, result: FetchResult) -> Result<ParsedDocument> {
        if result.body_bytes.is_empty() {
            return Err(Error::EmptyBody { url: result.url.to_string() });
        }

        // Single sanitising pass — output reused for both metadata and body
        // extraction. Cheaper than running lol_html twice over the raw bytes,
        // and guarantees both extractors see the same cleaned tree.
        let sanitised = strip_non_content(&result.body_bytes);

        let meta                  = extract_meta(&sanitised);
        let (body_text, outlinks) = match extract_body_lol_html(&sanitised, &result.url) {
            Ok(pair) => pair,
            Err(_)   => extract_body_scraper(&sanitised, &result.url),
        };

        let body_text    = collapse_whitespace(&body_text);
        let content_hash = SimHash::from_tokens(body_text.split_whitespace());

        Ok(ParsedDocument {
            id:          DocumentId::ZERO,
            url:         result.url,
            title:       meta.title,
            description: meta.description,
            headings:    meta.headings,
            body_text,
            outlinks,
            content_hash,
        })
    }
}

impl Default for Parser {
    fn default() -> Self { Self::new() }
}

// ── HTML sanitisation ────────────────────────────────────────────────────────

/// Strips all `NON_CONTENT_SELECTORS` from the input HTML. Output is well-formed
/// HTML bytes suitable for downstream extractors. Falls back to the input
/// unchanged if lol_html fails (malformed encoding, etc.).
fn strip_non_content(bytes: &[u8]) -> Vec<u8> {
    use lol_html::{element, HtmlRewriter, Settings};

    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    {
        let mut rewriter = HtmlRewriter::new(
            Settings {
                element_content_handlers: vec![
                    element!(NON_CONTENT_SELECTORS, |el| {
                        el.remove();
                        Ok(())
                    }),
                ],
                ..Settings::default()
            },
            |chunk: &[u8]| out.extend_from_slice(chunk),
        );

        if rewriter.write(bytes).is_err() || rewriter.end().is_err() {
            return bytes.to_vec();
        }
    }
    out
}

// ── Metadata extraction (scraper crate — DOM traversal) ──────────────────────

struct Meta {
    title:       String,
    description: String,
    headings:    Vec<String>,
}

fn extract_meta(bytes: &[u8]) -> Meta {
    use scraper::{Html, Selector};

    let html     = std::str::from_utf8(bytes).unwrap_or("");
    let document = Html::parse_document(html);

    let title       = select_text(&document, "title");
    let description = {
        let sel = Selector::parse("meta[name='description']").unwrap();
        document
            .select(&sel)
            .next()
            .and_then(|el| el.value().attr("content"))
            .unwrap_or("")
            .trim()
            .to_owned()
    };
    let headings = {
        let sel = Selector::parse("h1, h2, h3").unwrap();
        document
            .select(&sel)
            .map(|el| collapse_whitespace(&el.text().collect::<String>()))
            .filter(|s| !s.is_empty())
            .collect()
    };

    Meta { title, description, headings }
}

fn select_text(document: &scraper::Html, selector: &str) -> String {
    let sel = scraper::Selector::parse(selector).unwrap();
    document
        .select(&sel)
        .next()
        .map(|el| collapse_whitespace(&el.text().collect::<String>()))
        .unwrap_or_default()
}

// ── Body + outlinks — lol_html primary ───────────────────────────────────────

fn extract_body_lol_html(
    bytes:    &[u8],
    base_url: &Url,
) -> std::result::Result<(String, Vec<Url>), ()> {
    use lol_html::{element, DocumentContentHandlers, HtmlRewriter, Settings};

    let mut body_parts   = Vec::<String>::new();
    let mut outlink_strs = Vec::<String>::new();

    {
        let mut rewriter = HtmlRewriter::new(
            Settings {
                element_content_handlers: vec![
                    // Re-strip non-content selectors as a belt-and-braces guard
                    // in case strip_non_content was a no-op (it's defensive —
                    // scraper sees a cleaned tree but malformed HTML can leak).
                    element!(NON_CONTENT_SELECTORS, |el| { el.remove(); Ok(()) }),
                    element!(BODY_BOILERPLATE_SELECTORS, |el| { el.remove(); Ok(()) }),
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
    let outlinks  = outlink_strs.into_iter().filter_map(|s| Url::parse(&s).ok()).collect();

    Ok((body_text, outlinks))
}

// ── Body + outlinks — scraper fallback ───────────────────────────────────────

fn extract_body_scraper(bytes: &[u8], base_url: &Url) -> (String, Vec<Url>) {
    use scraper::{Html, Selector};

    let html     = std::str::from_utf8(bytes).unwrap_or("");
    let document = Html::parse_document(html);

    // The fallback gets the already-sanitised bytes from strip_non_content,
    // so script/style are gone. We still drop nav/footer/header here for
    // body-only boilerplate parity with the lol_html primary path.
    let body_text = {
        let sel = Selector::parse("body").unwrap();
        document
            .select(&sel)
            .next()
            .map(|el| {
                let drop_sel = Selector::parse(BODY_BOILERPLATE_SELECTORS).unwrap();
                let drop_ids: std::collections::HashSet<_> =
                    el.select(&drop_sel).map(|e| e.id()).collect();
                el.descendants()
                    .filter_map(|n| n.value().as_text().map(|t| (n.id(), t)))
                    .filter(|(id, _)| {
                        // Skip text nodes whose ancestor is a dropped element.
                        !std::iter::successors(Some(*id), |i| {
                            document.tree.get(*i).and_then(|n| n.parent().map(|p| p.id()))
                        })
                        .any(|aid| drop_ids.contains(&aid))
                    })
                    .map(|(_, t)| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
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

// ── Whitespace normalisation ─────────────────────────────────────────────────

/// Collapses runs of any whitespace (incl. \n, \t, &nbsp;) to single spaces.
/// Output is trimmed. Critical for snippet quality — without this, broken
/// JS-formatted code that survived stripping renders as a multi-line wall.
fn collapse_whitespace(s: &str) -> String {
    let mut out      = String::with_capacity(s.len());
    let mut prev_ws  = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    if out.ends_with(' ') { out.pop(); }
    out
}

// ── URL resolution ───────────────────────────────────────────────────────────

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
            url:        Url::parse(url).unwrap(),
            status:     200,
            headers:    HashMap::new(),
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
        let doc = Parser::new().parse(make_result(html, "https://example.com/")).unwrap();
        assert_eq!(doc.title, "Hello RAiTHE");
        assert_eq!(doc.description, "A great search engine.");
        assert!(doc.body_text.contains("Some body text"));
    }

    #[test]
    fn extracts_headings() {
        let html = r#"<html><body>
            <h1>Main</h1><h2>Sub</h2><h3>Minor</h3><h4>Ignored</h4>
            </body></html>"#;
        let doc = Parser::new().parse(make_result(html, "https://example.com/")).unwrap();
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
        let doc  = Parser::new().parse(make_result(html, "https://example.com/")).unwrap();
        let urls: Vec<_> = doc.outlinks.iter().map(|u| u.as_str()).collect();
        assert!( urls.iter().any(|u| u.contains("other.com/page")));
        assert!( urls.iter().any(|u| u.contains("example.com/relative")));
        assert!(!urls.iter().any(|u| u.contains("javascript")));
        assert!(!urls.iter().any(|u| u.contains("mailto")));
    }

    #[test]
    fn empty_body_returns_error() {
        let result = FetchResult {
            url:        Url::parse("https://example.com/").unwrap(),
            status:     200,
            headers:    HashMap::new(),
            body_bytes: vec![],
            fetched_at: Timestamp::from_millis(0),
        };
        assert!(Parser::new().parse(result).is_err());
    }

    #[test]
    fn content_hash_differs_for_different_bodies() {
        let a = Parser::new().parse(make_result("<html><body>alpha beta gamma</body></html>", "https://a.com/")).unwrap();
        let b = Parser::new().parse(make_result("<html><body>delta epsilon zeta</body></html>", "https://b.com/")).unwrap();
        assert_ne!(a.content_hash, b.content_hash);
    }

    #[test]
    fn id_is_zero_sentinel() {
        let doc = Parser::new().parse(make_result("<html><body>text</body></html>", "https://example.com/")).unwrap();
        assert_eq!(doc.id, DocumentId::ZERO);
    }

    // ── new — sanitisation tests ────────────────────────────────────────────

    #[test]
    fn strips_script_bodies() {
        let html = r#"<html><head>
            <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime()});})(window,document,'script','dataLayer','GTM-X');</script>
            </head><body><p>Real content here.</p></body></html>"#;
        let doc = Parser::new().parse(make_result(html, "https://example.com/")).unwrap();
        assert!(doc.body_text.contains("Real content"));
        assert!(!doc.body_text.contains("gtm.start"),    "GTM bootstrap leaked: {:?}", doc.body_text);
        assert!(!doc.body_text.contains("dataLayer"),    "JS variable leaked: {:?}",   doc.body_text);
        assert!(!doc.body_text.contains("getElementsByTagName"));
    }

    #[test]
    fn strips_style_bodies() {
        let html = r#"<html><head><style>.foo { color: red; --x: var(--bg); }</style></head>
            <body><p>Hello world.</p></body></html>"#;
        let doc = Parser::new().parse(make_result(html, "https://example.com/")).unwrap();
        assert!(doc.body_text.contains("Hello world"));
        assert!(!doc.body_text.contains("color: red"));
        assert!(!doc.body_text.contains("--bg"));
    }

    #[test]
    fn strips_noscript_and_iframe() {
        let html = r#"<html><body>
            <noscript>JS-disabled-fallback please enable javascript</noscript>
            <iframe>iframe-fallback-text</iframe>
            <p>Visible content.</p>
            </body></html>"#;
        let doc = Parser::new().parse(make_result(html, "https://example.com/")).unwrap();
        assert!(doc.body_text.contains("Visible content"));
        assert!(!doc.body_text.contains("JS-disabled-fallback"));
        assert!(!doc.body_text.contains("iframe-fallback-text"));
    }

    #[test]
    fn whitespace_is_collapsed() {
        let html = "<html><body><p>foo   bar\n\n\tbaz</p></body></html>";
        let doc  = Parser::new().parse(make_result(html, "https://example.com/")).unwrap();
        assert_eq!(doc.body_text, "foo bar baz");
    }

    #[test]
    fn nav_and_footer_stripped_from_body_but_title_intact() {
        let html = r#"<html><head><title>Site Title</title></head>
            <body>
                <nav>HOME ABOUT CONTACT</nav>
                <main><p>Article content.</p></main>
                <footer>© 2026 boilerplate</footer>
            </body></html>"#;
        let doc = Parser::new().parse(make_result(html, "https://example.com/")).unwrap();
        assert_eq!(doc.title, "Site Title");
        assert!( doc.body_text.contains("Article content"));
        assert!(!doc.body_text.contains("HOME ABOUT CONTACT"));
        assert!(!doc.body_text.contains("boilerplate"));
    }
}