// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/crawler/src/policy.rs
//
// Crawl policy decision type.

/// Decision returned by `RobotsCache::is_allowed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrawlPolicy {
    Allow,
    Disallow,
}

impl CrawlPolicy {
    /// Returns `true` when the policy permits crawling.
    pub fn is_allowed(self) -> bool {
        self == Self::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_is_allowed() {
        assert!(CrawlPolicy::Allow.is_allowed());
    }

    #[test]
    fn disallow_is_not_allowed() {
        assert!(!CrawlPolicy::Disallow.is_allowed());
    }
}
