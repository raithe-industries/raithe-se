// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/common/src/url_type.rs
//
// Newtype wrapper over `url::Url` for validated, serialisable URLs.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Error, Result};

/// A validated URL.
///
/// Wraps `url::Url` so that every `Url` value in the codebase is guaranteed
/// to be well-formed. Construction always goes through `Url::parse`, which
/// returns an error rather than storing an invalid string.
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Url(url::Url);

impl Url {
    /// Parses `raw` and returns a validated URL.
    ///
    /// Returns `Error::InvalidUrl` when the input is not a valid URL.
    pub fn parse(raw: &str) -> Result<Self> {
        let inner = url::Url::parse(raw).map_err(|source| Error::InvalidUrl {
            reason: source.to_string(),
        })?;
        Ok(Self(inner))
    }

    /// Returns the URL scheme (e.g. `"https"`).
    pub fn scheme(&self) -> &str {
        self.0.scheme()
    }

    /// Returns the host string, if present.
    pub fn host_str(&self) -> Option<&str> {
        self.0.host_str()
    }

    /// Returns the registered domain (eTLD+1), if present.
    ///
    /// Used for per-domain rate-limit keying in the crawler.
    pub fn registered_domain(&self) -> Option<&str> {
        // url::Url does not expose eTLD+1 directly; we derive it from host_str
        // by taking the last two dot-separated labels. This is a best-effort
        // approximation — a PSL-aware library is not a dependency of common.
        let host = self.0.host_str()?;
        let labels: Vec<&str> = host.split('.').collect();
        if labels.len() >= 2 {
            let start = labels.len() - 2;
            Some(&host[host.len() - labels[start..].join(".").len()..])
        } else {
            Some(host)
        }
    }

    /// Returns the path component.
    pub fn path(&self) -> &str {
        self.0.path()
    }

    /// Returns the full URL as a string slice.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Returns a reference to the inner `url::Url`.
    pub fn inner(&self) -> &url::Url {
        &self.0
    }
}

impl std::fmt::Display for Url {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<&str> for Url {
    type Error = Error;

    fn try_from(raw: &str) -> Result<Self> {
        Self::parse(raw)
    }
}

impl TryFrom<String> for Url {
    type Error = Error;

    fn try_from(raw: String) -> Result<Self> {
        Self::parse(&raw)
    }
}

impl Serialize for Url {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(self.0.as_str())
    }
}

impl<'de> Deserialize<'de> for Url {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_url_parses() {
        let url = Url::parse("https://example.com/path?q=1").unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str(), Some("example.com"));
        assert_eq!(url.path(), "/path");
    }

    #[test]
    fn invalid_url_returns_error() {
        assert!(Url::parse("not a url at all").is_err());
    }

    #[test]
    fn display_round_trips() {
        let raw = "https://example.com/";
        let url = Url::parse(raw).unwrap();
        assert_eq!(url.to_string(), raw);
    }

    #[test]
    fn serde_as_str_stable() {
        // Serde round-trip requires serde_json (not a dep of common).
        // Verify as_str is stable — the Serialize impl writes this value.
        let url = Url::parse("https://raithe.ca/").unwrap();
        assert_eq!(url.as_str(), "https://raithe.ca/");
    }

    #[test]
    fn registered_domain_two_labels() {
        let url = Url::parse("https://sub.example.com/").unwrap();
        assert_eq!(url.registered_domain(), Some("example.com"));
    }
}
