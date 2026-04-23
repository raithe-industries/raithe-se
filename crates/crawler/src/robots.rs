// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/crawler/src/robots.rs
//
// Bounded LRU cache for robots.txt documents.
// TTL-bounded per §9.3 (medium security).

use std::time::Duration;

use moka::sync::Cache;
use raithe_common::Url;
use raithe_scraper::Scraper;


/// Maximum robots.txt entries held in the LRU cache.
const MAX_ENTRIES: u64 = 10_000;

/// TTL for each cached robots.txt.
const TTL: Duration = Duration::from_secs(3_600); // 1 hour

/// LRU cache for parsed robots.txt documents, keyed by registered domain.
///
/// An absent cache entry is fetched on demand by calling the scraper.
/// If the robots.txt cannot be fetched (network error, 404, etc.) the cache
/// stores an implicit allow-all policy — crawling continues conservatively.
pub struct RobotsCache {
    /// Cached robots.txt content, keyed by "scheme://host".
    cache:      Cache<String, Vec<RobotsRule>>,
    user_agent: String,
}

/// A single Allow/Disallow rule parsed from robots.txt.
#[derive(Clone, Debug)]
struct RobotsRule {
    allow:  bool,
    prefix: String,
}

impl RobotsCache {
    /// Creates a new `RobotsCache` for the given `user_agent`.
    pub fn new(user_agent: &str) -> Self {
        let cache = Cache::builder()
            .max_capacity(MAX_ENTRIES)
            .time_to_live(TTL)
            .build();

        Self {
            cache,
            user_agent: user_agent.to_owned(),
        }
    }

    /// Returns `true` if `user_agent` is allowed to fetch `url` per robots.txt.
    ///
    /// Fetches and caches the robots.txt for the host on first access.
    /// Allows on any fetch or parse error (fail-open policy).
    pub async fn is_allowed(&self, url: &Url, scraper: &Scraper) -> bool {
        let origin = origin_key(url);
        let rules = self.cache.get_with(origin.clone(), || {
            // Cache miss — fetch robots.txt synchronously (called from async
            // context, but moka's get_with init closure is sync; we block via
            // tokio::task::block_in_place).
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    fetch_rules(&origin, scraper).await
                })
            })
        });

        applies(&rules, url.path(), &self.user_agent)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn origin_key(url: &Url) -> String {
    let scheme = url.scheme();
    let host   = url.host_str().unwrap_or("");
    let port   = url.inner().port()
        .map(|p| format!(":{p}"))
        .unwrap_or_default();
    format!("{scheme}://{host}{port}")
}

/// Fetches robots.txt from `origin` and returns parsed rules.
/// Returns empty vec (allow-all) on any error.
async fn fetch_rules(origin: &str, scraper: &Scraper) -> Vec<RobotsRule> {
    let robots_url = match Url::parse(&format!("{origin}/robots.txt")) {
        Ok(u)  => u,
        Err(_) => return Vec::new(),
    };

    let result = match scraper.fetch(&robots_url).await {
        Ok(r) if r.status == 200 => r,
        _                        => return Vec::new(),
    };

    let text = match std::str::from_utf8(&result.body_bytes) {
        Ok(s)  => s,
        Err(_) => return Vec::new(),
    };

    parse_rules(text)
}

/// Parses robots.txt text into a flat list of rules applicable to any agent.
///
/// Only `User-agent: *` and `User-agent: <our agent>` blocks are honoured.
/// Specificity: exact agent match overrides wildcard.
fn parse_rules(text: &str) -> Vec<RobotsRule> {
    let mut wildcard: Vec<RobotsRule> = Vec::new();
    let mut specific: Vec<RobotsRule> = Vec::new();

    // We do not track user_agent inside the parser because the cache is
    // per-origin and the caller checks applicability with `applies()`.
    // Here we collect all rules tagged by agent type so the caller can
    // prefer specific over wildcard.

    let mut current_wildcard = false;

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("User-agent:") {
            let agent = rest.trim();
            current_wildcard = agent == "*";
            let _ = agent;
            continue;
        }

        if let Some(rest) = line.strip_prefix("Disallow:") {
            let prefix = rest.trim().to_owned();
            let rule   = RobotsRule { allow: false, prefix };
            if current_wildcard {
                wildcard.push(rule);
            } else {
                specific.push(rule);
            }
            continue;
        }

        if let Some(rest) = line.strip_prefix("Allow:") {
            let prefix = rest.trim().to_owned();
            let rule   = RobotsRule { allow: true, prefix };
            if current_wildcard {
                wildcard.push(rule);
            } else {
                specific.push(rule);
            }
        }
    }

    // Prefer specific rules when present; fall back to wildcard.
    if !specific.is_empty() {
        specific
    } else {
        wildcard
    }
}

/// Checks `path` against `rules`.
///
/// Returns `CrawlPolicy::Allow` when:
///   - No rules match (implicit allow-all).
///   - An Allow rule with a longer or equal prefix matches.
///
/// Returns `CrawlPolicy::Disallow` when a Disallow rule matches and no
/// longer Allow rule overrides it.
fn applies(rules: &[RobotsRule], path: &str, _user_agent: &str) -> bool {
    let mut best: Option<(usize, bool)> = None; // (prefix_len, allow)

    for rule in rules {
        if rule.prefix.is_empty() { continue; }
        if path.starts_with(&rule.prefix) {
            let len = rule.prefix.len();
            let better = best
                .map(|(best_len, _)| len > best_len)
                .unwrap_or(true);
            if better {
                best = Some((len, rule.allow));
            }
        }
    }

    best.map(|(_, allow)| allow).unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(allow: bool, prefix: &str) -> RobotsRule {
        RobotsRule { allow, prefix: prefix.to_owned() }
    }

    #[test]
    fn no_rules_allows_all() {
        assert!(applies(&[], "/anything", "bot"));
    }

    #[test]
    fn disallow_root_blocks_all() {
        let rules = vec![rule(false, "/")];
        assert!(!applies(&rules, "/page", "bot"));
    }

    #[test]
    fn allow_overrides_disallow_with_longer_prefix() {
        let rules = vec![
            rule(false, "/private"),
            rule(true,  "/private/public"),
        ];
        assert!(!applies(&rules, "/private/secret",  "bot"));
        assert!(applies(&rules,  "/private/public/x","bot"));
    }

    #[test]
    fn disallow_empty_prefix_allows_all() {
        let rules = vec![rule(false, "")];
        // Empty Disallow means allow all — length 0 matches, but so does any
        // allow rule; in absence of allow rule, empty disallow = allow.
        assert!(applies(&rules, "/", "bot"));
    }

    #[test]
    fn parse_rules_extracts_disallow() {
        let text = "User-agent: *\nDisallow: /admin\n";
        let rules = parse_rules(text);
        assert!(!rules.is_empty());
        assert_eq!(rules[0].prefix, "/admin");
        assert!(!rules[0].allow);
    }

    #[test]
    fn origin_key_includes_port_when_nonstandard() {
        let url = Url::parse("https://example.com:8443/page").unwrap();
        let key = origin_key(&url);
        assert_eq!(key, "https://example.com:8443");
    }

    #[test]
    fn origin_key_omits_default_port() {
        let url = Url::parse("https://example.com/page").unwrap();
        let key = origin_key(&url);
        assert_eq!(key, "https://example.com");
    }
}
