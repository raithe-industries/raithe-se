// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/serving/src/templates.rs
//
// Inline HTML templates for the search UI.
// Dark mode. Script/cursive "Search" wordmark. Minimal input + magnifier.
// Instant-answer panel. RAiTHE footer branding.

use crate::{InstantAnswerResponse, SearchResult};

// ── Common shell ──────────────────────────────────────────────────────────────

const STYLE: &str = r#"
<style>
  :root {
    --bg:      #0d0d0d;
    --surface: #161616;
    --border:  #2a2a2a;
    --accent:  #e8e8e8;
    --muted:   #888;
    --green:   #4ade80;
    --font:    'Segoe UI', system-ui, sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--accent);
    font-family: var(--font);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  a { color: var(--green); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .wordmark {
    font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', serif;
    font-style: italic;
    font-size: 3rem;
    letter-spacing: -0.02em;
    color: #fff;
    user-select: none;
  }
  .search-form {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2rem;
    padding: 0.5rem 1.25rem;
    width: 100%;
    max-width: 600px;
  }
  .search-form input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--accent);
    font-size: 1rem;
    font-family: var(--font);
  }
  .search-form button {
    background: transparent;
    border: none;
    cursor: pointer;
    color: var(--muted);
    display: flex;
    align-items: center;
    padding: 0;
  }
  .search-form button:hover { color: var(--accent); }
  .magnifier { width: 20px; height: 20px; }
  footer {
    margin-top: auto;
    padding: 1.5rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
  }
  .instant-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    max-width: 640px;
  }
  .instant-panel .label {
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .instant-panel .value {
    font-size: 1.5rem;
    font-weight: 600;
  }
  .result { margin-bottom: 1.5rem; max-width: 640px; }
  .result-title { font-size: 1.1rem; }
  .result-url   { font-size: 0.8rem; color: var(--muted); margin: 0.2rem 0; }
  .result-snippet { font-size: 0.95rem; color: #ccc; line-height: 1.5; }
</style>
"#;

const MAGNIFIER_SVG: &str = r#"<svg class="magnifier" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>"#;

const FOOTER: &str = r#"<footer>© RAiTHE INDUSTRIES INCORPORATED 2026</footer>"#;

fn search_form(query: &str) -> String {
    let escaped = html_escape(query);
    format!(
        r#"<form class="search-form" action="/search" method="get">
  <input type="text" name="q" value="{escaped}" placeholder="Search…" autofocus autocomplete="off">
  <button type="submit" aria-label="Search">{MAGNIFIER_SVG}</button>
</form>"#
    )
}

// ── Index page ────────────────────────────────────────────────────────────────

pub fn render_index() -> String {
    let form = search_form("");
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAiTHE Search</title>
  {STYLE}
  <style>
    .hero {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 2rem;
      padding: 2rem;
    }}
  </style>
</head>
<body>
  <main class="hero">
    <span class="wordmark">Search</span>
    {form}
  </main>
  {FOOTER}
</body>
</html>"#
    )
}

// ── Results page ──────────────────────────────────────────────────────────────

pub fn render_results(
    query: &str,
    results: &[SearchResult],
    instant: Option<&InstantAnswerResponse>,
) -> String {
    let form = search_form(query);

    let instant_html = instant
        .map(|ia| {
            let display = html_escape(&ia.display);
            format!(
                r#"<div class="instant-panel">
  <div class="label">Instant Answer</div>
  <div class="value">{display}</div>
</div>"#
            )
        })
        .unwrap_or_default();

    let results_html: String = if results.is_empty() {
        let q = html_escape(query);
        format!(r#"<p style="color:var(--muted)">No results for <em>{q}</em>.</p>"#)
    } else {
        results
            .iter()
            .map(|r| {
                let title = html_escape(&r.title);
                let url = html_escape(&r.url);
                let snippet = html_escape(&r.snippet);
                format!(
                    r#"<div class="result">
  <div class="result-title"><a href="{url}">{title}</a></div>
  <div class="result-url">{url}</div>
  <div class="result-snippet">{snippet}</div>
</div>"#
                )
            })
            .collect()
    };

    let count = results.len();
    let q = html_escape(query);

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{q} — RAiTHE Search</title>
  {STYLE}
  <style>
    header {{
      display: flex;
      align-items: center;
      gap: 1.5rem;
      padding: 1rem 2rem;
      border-bottom: 1px solid var(--border);
    }}
    .wordmark {{ font-size: 1.5rem; }}
    main {{ padding: 2rem; }}
    .meta {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <header>
    <a href="/" style="text-decoration:none"><span class="wordmark">Search</span></a>
    {form}
  </header>
  <main>
    <p class="meta">{count} result(s) for <em>{q}</em></p>
    {instant_html}
    {results_html}
  </main>
  {FOOTER}
</body>
</html>"#
    )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Minimal HTML escaping for user-controlled strings inserted into templates.
fn html_escape(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '&' => "&amp;".chars().collect::<Vec<_>>(),
            '<' => "&lt;".chars().collect(),
            '>' => "&gt;".chars().collect(),
            '"' => "&quot;".chars().collect(),
            '\'' => "&#39;".chars().collect(),
            _ => vec![c],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_escape_ampersand() {
        assert_eq!(html_escape("a&b"), "a&amp;b");
    }

    #[test]
    fn html_escape_angle_brackets() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
    }

    #[test]
    fn render_index_contains_wordmark() {
        let html = render_index();
        assert!(html.contains("Search"));
        assert!(html.contains("RAiTHE"));
    }

    #[test]
    fn render_results_empty_shows_no_results() {
        let html = render_results("rust", &[], None);
        assert!(html.contains("No results"));
    }

    #[test]
    fn render_results_with_instant_answer() {
        let ia = InstantAnswerResponse {
            kind: String::from("Calculation"),
            display: String::from("42"),
        };
        let html = render_results("6 * 7", &[], Some(&ia));
        assert!(html.contains("42"));
        assert!(html.contains("Instant Answer"));
    }

    #[test]
    fn render_results_escapes_xss_in_query() {
        let html = render_results("<script>alert(1)</script>", &[], None);
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
    }
}
